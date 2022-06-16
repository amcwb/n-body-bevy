use std::os::raw::c_void;

use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use bevy_prototype_lyon::prelude::*;
use glw::gl;
use gpu::Particle;
use rand::{distributions::Uniform, prelude::Distribution, Rng};
// use directx_math::XMScalarSinCos;
use bevy_spatial::{EfficientInsertParams, RTreeAccess3D, RTreePlugin3D, SpatialAccess};
mod gpu;
use fragile::Fragile;

const USE_3D: bool = true;

type ImplIteratorMut<'a, Item> =
    ::std::iter::Chain<::std::slice::IterMut<'a, Item>, ::std::slice::IterMut<'a, Item>>;
trait SplitOneMut {
    type Item;

    fn split_one_mut(
        &'_ mut self,
        i: usize,
    ) -> (&'_ mut Self::Item, ImplIteratorMut<'_, Self::Item>);
}

impl<T> SplitOneMut for [T] {
    type Item = T;

    fn split_one_mut(
        &'_ mut self,
        i: usize,
    ) -> (&'_ mut Self::Item, ImplIteratorMut<'_, Self::Item>) {
        let (prev, current_and_end) = self.split_at_mut(i);
        let (current, end) = current_and_end.split_at_mut(1);
        (&mut current[0], prev.iter_mut().chain(end))
    }
}

/// Tags an entity as capable of panning and orbiting.
#[derive(Component)]
struct PanOrbitCamera {
    /// The "focus point" to orbit around. It is automatically updated when panning the camera
    pub focus: Vec3,
    pub radius: f32,
    pub upside_down: bool,
}

impl Default for PanOrbitCamera {
    fn default() -> Self {
        PanOrbitCamera {
            focus: Vec3::ZERO,
            radius: 5.0,
            upside_down: false,
        }
    }
}

/// Pan the camera with middle mouse click, zoom with scroll wheel, orbit with right mouse click.
fn pan_orbit_camera(
    windows: Res<Windows>,
    mut ev_motion: EventReader<MouseMotion>,
    mut ev_scroll: EventReader<MouseWheel>,
    input_mouse: Res<Input<MouseButton>>,
    mut query: Query<(&mut PanOrbitCamera, &mut Transform, &PerspectiveProjection)>,
) {
    if !USE_3D {
        return;
    };
    // change input mapping for orbit and panning here
    let orbit_button = MouseButton::Right;
    let pan_button = MouseButton::Middle;

    let mut pan = Vec2::ZERO;
    let mut rotation_move = Vec2::ZERO;
    let mut scroll = 0.0;
    let mut orbit_button_changed = false;

    if input_mouse.pressed(orbit_button) {
        for ev in ev_motion.iter() {
            rotation_move += ev.delta;
        }
    } else if input_mouse.pressed(pan_button) {
        // Pan only if we're not rotating at the moment
        for ev in ev_motion.iter() {
            pan += ev.delta;
        }
    }
    for ev in ev_scroll.iter() {
        scroll += ev.y;
    }
    if input_mouse.just_released(orbit_button) || input_mouse.just_pressed(orbit_button) {
        orbit_button_changed = true;
    }

    for (mut pan_orbit, mut transform, projection) in query.iter_mut() {
        if orbit_button_changed {
            // only check for upside down when orbiting started or ended this frame
            // if the camera is "upside" down, panning horizontally would be inverted, so invert the input to make it correct
            let up = transform.rotation * Vec3::Y;
            pan_orbit.upside_down = up.y <= 0.0;
        }

        let mut any = false;
        if rotation_move.length_squared() > 0.0 {
            any = true;
            let window = get_primary_window_size(&windows);
            let delta_x = {
                let delta = rotation_move.x / window.x * std::f32::consts::PI * 2.0;
                if pan_orbit.upside_down {
                    -delta
                } else {
                    delta
                }
            };
            let delta_y = rotation_move.y / window.y * std::f32::consts::PI;
            let yaw = Quat::from_rotation_y(-delta_x);
            let pitch = Quat::from_rotation_x(-delta_y);
            transform.rotation = yaw * transform.rotation; // rotate around global y axis
            transform.rotation = transform.rotation * pitch; // rotate around local x axis
        } else if pan.length_squared() > 0.0 {
            any = true;
            // make panning distance independent of resolution and FOV,
            let window = get_primary_window_size(&windows);
            pan *= Vec2::new(projection.fov * projection.aspect_ratio, projection.fov) / window;
            // translate by local axes
            let right = transform.rotation * Vec3::X * -pan.x;
            let up = transform.rotation * Vec3::Y * pan.y;
            // make panning proportional to distance away from focus point
            let translation = (right + up) * pan_orbit.radius;
            pan_orbit.focus += translation;
        } else if scroll.abs() > 0.0 {
            any = true;
            pan_orbit.radius -= scroll * pan_orbit.radius * 0.2;
            // dont allow zoom to reach zero or you get stuck
            pan_orbit.radius = f32::max(pan_orbit.radius, 0.05);
        }

        if any {
            // emulating parent/child to make the yaw/y-axis rotation behave like a turntable
            // parent = x and y rotation
            // child = z-offset
            let rot_matrix = Mat3::from_quat(transform.rotation);
            transform.translation =
                pan_orbit.focus + rot_matrix.mul_vec3(Vec3::new(0.0, 0.0, pan_orbit.radius));
        }
    }
}

fn get_primary_window_size(windows: &Res<Windows>) -> Vec2 {
    let window = windows.get_primary().unwrap();
    let window = Vec2::new(window.width() as f32, window.height() as f32);
    window
}

#[derive(Component)]
struct TimeScale(f64);

#[derive(Component)]
struct MassiveObjects;

// type alias for easier usage later
type NNTree = RTreeAccess3D<Particle, EfficientInsertParams>;

const GRAVITY: f32 = 0.000000000066742 * 1000.0;
impl Particle {
    fn force_from(&self, other: &Particle, distance2: Option<f32>) -> (f32, f32, f32) {
        let _span = info_span!("calculate_force_from").entered();
        let distance2 = match distance2 {
            Some(e) => e,
            None => self.distance2_to(other),
        };

        let d_sqrt = distance2.sqrt();
        let _span2 = info_span!("calculate_force").entered();
        let f = (GRAVITY * self.mass * other.mass) / distance2;
        _span2.exit();
        let _span2 = info_span!("calculate_atan").entered();
        _span2.exit();

        let dx = other.x - self.x;
        let dy = other.y - self.y;
        let dz = other.z - self.z;
        let fx = dx / d_sqrt * f;
        let fy = dy / d_sqrt * f;
        let fz = dz / d_sqrt * f;

        let res = (fx, fy, fz);
        _span.exit();
        res
    }

    fn distance_to(&self, other: &Particle) -> f32 {
        let _span = info_span!("calculate_distance").entered();
        let res = self.distance2_to(other).sqrt();
        _span.exit();
        res
    }

    fn distance2_to(&self, other: &Particle) -> f32 {
        let _span = info_span!("calculate_distance2").entered();
        let res = (other.x - self.x).powi(2) + (other.y - self.y).powi(2) + (other.z - self.z).powi(2);
        _span.exit();
        res
    }

    fn radius(&self) -> f32 {
        self.mass.log10()
    }
}

impl PartialEq for Particle {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.vx == other.vx
            && self.vy == other.vy
            && self.mass == other.mass
        // && self.color == other.color
    }
}

fn setup_camera(mut commands: Commands) {
    if USE_3D {
        let translation = Vec3::new(-1000.0, -1000.0, -1000.0);
        let radius = translation.length();
        commands
            .spawn_bundle(PerspectiveCameraBundle {
                transform: Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
                ..default()
            })
            .insert(PanOrbitCamera {
                radius,
                ..Default::default()
            });
        commands.spawn_bundle(PointLightBundle {
            point_light: PointLight {
                intensity: 10000.0,
                shadows_enabled: false,
                range: 10000.0,
                ..default()
            },
            transform: Transform::from_xyz(-0.0, -0.0, -0.0),
            ..default()
        });
        commands.spawn_bundle(PointLightBundle {
            point_light: PointLight {
                intensity: 10000.0,
                shadows_enabled: false,
                range: 10000.0,
                ..default()
            },
            transform: Transform::from_xyz(-10.0, -10.0, -100.0),
            ..default()
        });
    } else {
        commands.spawn_bundle(OrthographicCameraBundle::new_2d());
    };
    commands.insert_resource(TimeScale(1.0));
}

fn camera_control(
    mut scroll_evr: EventReader<MouseWheel>,
    keys: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut timescale: ResMut<TimeScale>,
    mut query: Query<(&mut Transform, &mut OrthographicProjection)>,
) {
    let dist = 1.0 * time.delta().as_secs_f32();
    if !USE_3D {
        // FOR 3D RENDER
        // We have to control this ourselves for 2d render
        let mut scroll_y = 0.0;
        for ev in scroll_evr.iter() {
            match ev.unit {
                MouseScrollUnit::Line => {
                    // println!("Scroll (line units): vertical: {}, horizontal: {}", ev.y, ev.x);
                    scroll_y -= ev.y
                }
                MouseScrollUnit::Pixel => {
                    // println!("Scroll (pixel units): vertical: {}, horizontal: {}", ev.y, ev.x);
                    scroll_y -= ev.y
                }
            }
        }

        for (transform, mut projection) in query.iter_mut() {
            let mut transform: Mut<Transform> = transform;
            let mut log_scale = projection.scale.ln();

            log_scale += scroll_y * dist;

            projection.scale = log_scale.exp();
            if keys.pressed(KeyCode::A) {
                transform.translation.x -= 100.0 * projection.scale * time.delta().as_secs_f32();
            }
            if keys.pressed(KeyCode::D) {
                transform.translation.x += 100.0 * projection.scale * time.delta().as_secs_f32();
            }
            if keys.pressed(KeyCode::S) {
                transform.translation.y -= 100.0 * projection.scale * time.delta().as_secs_f32();
            }
            if keys.pressed(KeyCode::W) {
                transform.translation.y += 100.0 * projection.scale * time.delta().as_secs_f32();
            }
        }
    }

    if keys.pressed(KeyCode::Z) {
        timescale.0 = timescale.0 / 1.5;
    }
    if keys.pressed(KeyCode::X) {
        timescale.0 = timescale.0 * 1.5;
    }
}

fn main() {
    App::new()
        .insert_resource(WindowDescriptor {
            title: "N-body simulator".to_string(),
            width: 640.0,
            height: 400.0,
            ..default()
        })
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(gpu::Application::new().unwrap())
        .add_plugins(DefaultPlugins)
        .add_plugin(ShapePlugin)
        .add_plugin(RTreePlugin3D::<Particle, EfficientInsertParams> { ..default() })
        .add_startup_system(setup_camera)
        .add_startup_system(add_particles)
        .add_system(apply_forces)
        .add_system(transform_objects)
        .add_system(camera_control)
        .add_system(massive_objects_manager.before(collision_manager))
        .add_system(collision_manager)
        .add_system(pan_orbit_camera)
        .run();
}

fn add_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // let direction = 1.0; // clockwise
    // let star_mass = 2000000000.0;
    // let max_distance = 20000;
    // let max_mass = 300000;
    // let amount = 10000;

    // let center_x = 0.0; // -250.0;
    // let center_y = 0.0;
    // let center_z = 0.0;
    // let base_vx = 0.0;
    // let base_vy = 0.0;
    // let base_vz = 0.0;

    // create_system(
    //     &mut commands,
    //     &mut meshes,
    //     &mut materials,
    //     max_distance,
    //     max_mass,
    //     center_x,
    //     center_y,
    //     center_z,
    //     base_vx,
    //     base_vy,
    //     base_vz,
    //     star_mass,
    //     amount,
    //     direction,
    // );

    // // Make earth
    // create_star(
    //     &mut commands,
    //     &mut meshes,
    //     &mut materials,
    //     0.0,
    //     0.0,
    //     0.0,
    //     0.0,
    //     0.0,
    //     0.0,
    //     5_973_600_000_000_000_000_000_000.0,
    //     0.0,
    //     1.0,
    //     0.0,
    // );

    // // Make moon
    // create_star(
    //     &mut commands,
    //     &mut meshes,
    //     &mut materials,
    //     0.0,
    //     406_700_000.0,
    //     0.0,
    //     1082.0,
    //     0.0,
    //     0.0,
    //     7.342e+22,
    //     0.7,
    //     0.7,
    //     0.7,
    // );


    // Here the gravitational constant G has been set to 1, and the initial
    // conditions are r1(0) = −r3(0) = (−0.97000436, 0.24308753); 
    // r2(0) = (0,0); v1(0) = v3(0) = (0.4662036850, 0.4323657300);
    // v2(0) = (−0.93240737, −0.86473146).
    // 
    // The values are obtained from Chenciner & Montgomery (2000).
}

#[allow(dead_code)]
fn create_system(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    max_distance: i32,
    max_mass: i32,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    base_vx: f32,
    base_vy: f32,
    base_vz: f32,
    star_mass: f32,
    amount: i32,
    direction: f32,
) {
    let mut rng = rand::thread_rng();
    let distance = Uniform::from(10..=max_distance);
    let mass = Uniform::from(1..=max_mass);

    create_star(
        &mut commands,
        &mut meshes,
        &mut materials,
        center_x,
        center_y,
        center_z,
        base_vx,
        base_vy,
        base_vz,
        star_mass,
        rng.gen::<f32>(),
        rng.gen::<f32>(),
        rng.gen::<f32>(),
    );
    for _ in 0..amount {
        let r = distance.sample(&mut rng) as f32;
        let m = mass.sample(&mut rng) as f32;

        let v = ((GRAVITY * star_mass) / r).sqrt();

        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;

        let v_x = base_vx + theta.sin() * v * direction;
        let v_y = base_vy + theta.cos() * v * direction;
        let v_z = base_vz + 0.0;

        let r_x = center_x + theta.cos() * r;
        let r_y = center_y + -theta.sin() * r;
        let r_z = center_z;

        create_star(
            &mut commands,
            &mut meshes,
            &mut materials,
            r_x,
            r_y,
            r_z,
            v_x,
            v_y,
            v_z,
            m,
            rng.gen::<f32>(),
            rng.gen::<f32>(),
            rng.gen::<f32>(),
        );
    }
}

fn create_star<'a>(
    commands: &'a mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    mass: f32,
    r: f32,
    g: f32,
    b: f32,
) {
    let mut ec = if USE_3D {
        // FOR 3D RENDER
        let material = materials.add(Color::rgb(r, g, b).into());
        let mesh = meshes.add(Mesh::from(shape::Icosphere {
            subdivisions: 4,
            radius: (mass.log10() * 1000000.0) as f32,
        }));
        commands.spawn_bundle(PbrBundle {
            mesh,
            material,
            ..default()
        })
    } else {
        // FOR 2D RENDER
        let shape = shapes::Circle {
            radius: (mass.log10()) as f32,
            ..shapes::Circle::default()
        };
        commands.spawn_bundle(GeometryBuilder::build_as(
            &shape,
            DrawMode::Outlined {
                fill_mode: FillMode::color(Color::rgb(r, g, b)),
                outline_mode: StrokeMode::new(Color::WHITE, 0.5),
            },
            Transform::default(),
        ))
    };

    ec.insert(Transform { ..default() }).insert(Particle {
        mass,
        x,
        y,
        z,
        vx,
        vy,
        vz,
    });

    // if mass > 10_000.0 {
    //     commands.entity(id).insert(MassiveObjects);
    // }
}

fn apply_forces(
    timescale: Res<TimeScale>,
    time: Res<Time>,
    pool: Res<AsyncComputeTaskPool>,
    mut application: ResMut<gpu::Application>,
    mut query: Query<&mut Particle>,
) {
    let dt = time.delta_seconds() as f32 * timescale.0 as f32;
    let mut particles = Vec::<Particle>::new();
    query.for_each(|particle| {
        particles.push(*particle as Particle);
    });

    let new_particles = match application.run(dt as f32, particles) {
        Some(e) => e,
        None => return,
    };

    let mut i = 0;
    for mut particle in query.iter_mut() {
        *particle = new_particles[i];
        i += 1;
    }
}

fn massive_objects_manager(mut commands: Commands, query: Query<(Entity, &mut Particle)>) {
    query.for_each(|(entity, p)| {
        if p.mass > 10000.0 {
            commands.entity(entity).insert(MassiveObjects);
        } else {
            commands.entity(entity).remove::<MassiveObjects>();
        }
    })
}

fn collision_manager(
    mut commands: Commands,
    query: Query<(Entity, &Transform), With<MassiveObjects>>,
    mut query2: Query<&mut Particle>,
    treeaccess: Res<NNTree>,
) {
    // for (entity, transform) in query.iter() {
    //     let radius = {
    //         let p1: Particle = match query2.get(entity) {
    //             Ok(e) => *e,
    //             Err(_) => continue
    //         };

    //         // let span = info_span!("mass_check").entered();
    //         // if p1.mass > 10000.0 {
    //         //     commands.entity(entity).insert(MassiveObjects);
    //         // } else {
    //         //     commands.entity(entity).remove::<MassiveObjects>();
    //         // }
    //         // span.exit();

    //         p1.radius() * 1.20
    //     };

    //     let span = info_span!("check_tree").entered();
    //     for (_, entity2) in treeaccess.within_distance(transform.translation, radius as f32) {
    //         let [p1, p2] = match query2.get_many_mut([entity, entity2]) {
    //             Ok(e) => e,
    //             Err(_) => continue
    //         };
    //         let mut p1: Mut<Particle> = p1;
    //         let p2: Mut<Particle> = p2;

    //         // if p1.mass > 50.0 || p2.mass > 50.0 {
    //         if true {
    //             if p2.mass > p1.mass {
    //                 continue
    //             }

    //             // Combine
    //             commands.entity(entity2).despawn();

    //             let momentum_x = p1.mass * p1.vx + p2.mass * p2.vx * 0.75; // assume some energy lost;
    //             let momentum_y = p1.mass * p1.vy + p2.mass * p2.vy * 0.75; // assume some energy lost;
    //             let momentum_z = p1.mass * p1.vz + p2.mass * p2.vz * 0.75; // assume some energy lost;

    //             // if p2.color == YELLOW || p1.color == YELLOW {
    //             //     p1.color = YELLOW;
    //             // }

    //             p1.mass += p2.mass;
    //             p1.vx = momentum_x / p1.mass;
    //             p1.vy = momentum_y / p1.mass;
    //             p1.vz = momentum_z / p1.mass;
    //         }
    //     }
    //     span.exit();
    // }
}

fn transform_objects(mut query: Query<(&mut Particle, &mut Transform)>) {
    for (particle, transform) in query.iter_mut() {
        // Force typing
        let particle: Mut<Particle> = particle;
        let mut transform: Mut<Transform> = transform;

        *transform = Transform::from_xyz(particle.x as f32, particle.y as f32, particle.z as f32)
    }
}
