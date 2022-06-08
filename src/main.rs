use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use bevy_prototype_lyon::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution, Rng};
// use directx_math::XMScalarSinCos;
use bevy_spatial::{EfficientInsertParams, RTreeAccess3D, RTreePlugin3D, SpatialAccess};

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
#[derive(Component, Copy, Clone)]
struct Particle {
    mass: f64,
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64
}

#[derive(Component)]
struct TimeScale(f64);

#[derive(Component)]
struct MassiveObjects;

// type alias for easier usage later
type NNTree = RTreeAccess3D<Particle, EfficientInsertParams>;

const GRAVITY: f64 = 0.000000000066742 * 1000.0;
impl Particle {
    fn force_from(&self, other: &Particle, distance2: Option<f64>) -> (f64, f64, f64) {
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

    fn distance_to(&self, other: &Particle) -> f64 {
        let _span = info_span!("calculate_distance").entered();
        let res = self.distance2_to(other).sqrt();
        _span.exit();
        res
    }

    fn distance2_to(&self, other: &Particle) -> f64 {
        let _span = info_span!("calculate_distance2").entered();
        let res = (other.x - self.x).powi(2) + (other.y - self.y).powi(2) + (other.z - self.z).powi(2);
        _span.exit();
        res
    }

    fn radius(&self) -> f64 {
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
    // commands.spawn_bundle(OrthographicCameraBundle::new_3d());
    commands.insert_resource(TimeScale(1.0));
}

fn camera_control(
    mut scroll_evr: EventReader<MouseWheel>,
    keys: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut timescale: ResMut<TimeScale>,
    mut query: Query<&mut Transform, With<PerspectiveProjection>>,
) {
    let dist = 1.0 * time.delta().as_secs_f32();
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

    for mut transform in query.iter_mut() {
        let mut transform: Mut<Transform> = transform;
        // if keys.pressed(KeyCode::A) {
        //     transform.translation.x -= 100.0 * projection.scale * time.delta().as_secs_f32();
        // }
        // if keys.pressed(KeyCode::D) {
        //     transform.translation.x += 100.0 * projection.scale * time.delta().as_secs_f32();
        // }
        // if keys.pressed(KeyCode::S) {
        //     transform.translation.y -= 100.0 * projection.scale * time.delta().as_secs_f32();
        // }
        // if keys.pressed(KeyCode::W) {
        //     transform.translation.y += 100.0 * projection.scale * time.delta().as_secs_f32();
        // }
    }

    if keys.pressed(KeyCode::Z) {
        timescale.0 = timescale.0 / 1.02;
    }
    if keys.pressed(KeyCode::X) {
        timescale.0 = timescale.0 * 1.02;
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
        .add_plugins(DefaultPlugins)
        .add_plugin(ShapePlugin)
        .add_plugin(
            RTreePlugin3D::<Particle, EfficientInsertParams> { ..default() },
        )
        .add_startup_system(setup_camera)
        .add_startup_system(add_particles)
        .add_system(apply_forces)
        .add_system(transform_objects)
        .add_system(camera_control)
        .add_system(massive_objects_manager.before(collision_manager))
        .add_system(collision_manager)
        .run();
}

fn add_particles(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let direction = 1.0; // clockwise
    let star_mass = 2000000000.0;
    let max_distance = 5000;
    let max_mass = 3000;
    let amount = 100000;

    let center_x = 0.0; // -250.0;
    let center_y = 0.0;
    let center_z = 0.0;
    let base_vx = 0.0;
    let base_vy = 0.0;
    let base_vz = 0.0;

    create_system(
        &mut commands,
        &mut meshes,
        &mut materials,
        max_distance,
        max_mass,
        center_x,
        center_y,
        center_z,
        base_vx,
        base_vy,
        base_vz,
        star_mass,
        amount,
        direction,
    );

    // let direction = 1.0; // clockwise
    // let star_mass = 2000000000.0;
    // let max_distance = 500;
    // let max_mass = 3000;
    // let amount = 1000;

    // let center_x = -250.0; // -250.0;
    // let center_y = -100.0;
    // let center_z = -200.0;
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

    // base camera
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_xyz(-2000.0, -2000.0, -2000.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });
    commands.spawn_bundle(PointLightBundle {
        point_light: PointLight {
            intensity: 10000.0,
            shadows_enabled: false,
            range: 10000.0,
            ..default()
        },
        transform: Transform::from_xyz(-10.0, -10.0, -10.0),
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
}

fn create_system(
    mut commands: &mut Commands,
    mut meshes: &mut ResMut<Assets<Mesh>>,
    mut materials: &mut ResMut<Assets<StandardMaterial>>,
    max_distance: i32,
    max_mass: i32,
    center_x: f64,
    center_y: f64,
    center_z: f64,
    base_vx: f64,
    base_vy: f64,
    base_vz: f64,
    star_mass: f64,
    amount: i32,
    direction: f64,
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
        let r = distance.sample(&mut rng) as f64;
        let m = mass.sample(&mut rng) as f64;

        let v = ((GRAVITY * star_mass) / r).sqrt();

        let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;

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
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    mass: f64,
    r: f32,
    g: f32,
    b: f32,
) {
    // let shape = shapes::Circle {
    //     radius: (mass.log10()) as f32,
    //     ..shapes::Circle::default()
    // };
    let material = materials.add(Color::rgb(r, g, b).into());
    let mesh = meshes.add(Mesh::from(
        shape::Icosphere { subdivisions: 4, radius: (mass.log10()) as f32 },
    ));
    commands
        .spawn_bundle(PbrBundle {
            mesh, material, ..default()
        })
        .insert(Transform { ..default() })
        .insert(Particle { mass, x, y, z, vx, vy, vz });

    // if mass > 10_000.0 {
    //     commands.entity(id).insert(MassiveObjects);
    // }
}

fn apply_forces(
    timescale: Res<TimeScale>,
    time: Res<Time>,
    pool: Res<AsyncComputeTaskPool>,
    mut query: Query<&mut Particle, Without<MassiveObjects>>,
    mut massive_query: Query<&mut Particle, With<MassiveObjects>>
) {
    let dt = time.delta_seconds() as f64 * timescale.0;
    let _span_all = info_span!("apply_forces").entered();
    let mut massive_objects: Vec<Mut<Particle>> = vec![];

    // trusts that not many massive objects will be present
    let _span2 = info_span!("collect massive objs").entered();
    for particle in massive_query.iter_mut() {
        massive_objects.push(particle);
    }
    _span2.exit();

    query.par_for_each_mut(&pool, 12, |mut p2| {
        let mut fx_p2: f64 = 0.0;
        let mut fy_p2: f64 = 0.0;
        let mut fz_p2: f64 = 0.0;
        for p1 in &massive_objects {
            if p1.as_ref() == p2.as_ref() {
                return;
            }

            // if p1.mass > 50.0 || p2.mass > 50.0 {
            //     let target: &mut Particle;
            //     let origin: &mut Particle;
            //     let origin_entity: &Entity;

            //     if p1.mass > p2.mass {
            //         target = p1;
            //         origin = p2.as_mut();
            //         origin_entity = &entity2;
            //     } else {
            //         target = p2.as_mut();
            //         origin = p1;
            //         origin_entity = entity;
            //     }

            //     let distance = target.distance_to(origin);
            //     if distance < (target.radius() + origin.radius()) / 2.0 {
            //         // Combine
            //         commands.entity(*origin_entity).despawn();

            //         let momentum_x = target.mass * target.vx + origin.mass * origin.vx * 0.75; // assume some energy lost;
            //         let momentum_y = target.mass * target.vy + origin.mass * origin.vy * 0.75; // assume some energy lost;

            //         // if origin.color == YELLOW || target.color == YELLOW {
            //         //     target.color = YELLOW;
            //         // }

            //         target.mass += origin.mass;
            //         target.vx = momentum_x / target.mass;
            //         target.vy = momentum_y / target.mass;
            //     }
            // }

            let distance = p1.distance2_to(&p2);
            let _span_parent = info_span!("particle_1").entered();
            // p2
            let (t_fx_p2, t_fy_p2, t_fz_p2) = p2.force_from(&p1, Some(distance));
            {
                let _span = info_span!("sum_forces").entered();
                fx_p2 += t_fx_p2;
                fy_p2 += t_fy_p2;
                fz_p2 += t_fz_p2;
                _span.exit();

            }
            _span_parent.exit();
        }

        let _span2 = info_span!("apply_velocities").entered();
        p2.vx += (fx_p2 / p2.mass) * dt;
        p2.vy += (fy_p2 / p2.mass) * dt;
        p2.vz += (fz_p2 / p2.mass) * dt;
        _span2.exit();

        let _span2 = info_span!("apply_position").entered();
        p2.x += p2.vx * dt;
        p2.y += p2.vy * dt;
        p2.z += p2.vz * dt;
        _span2.exit();
    });

    for p1idx in 0..massive_objects.len() {
        let (p1, particles) = massive_objects.split_one_mut(p1idx);
        let mut fx_p1: f64 = 0.0;
        let mut fy_p1: f64 = 0.0;
        let mut fz_p1: f64 = 0.0;
        for p2 in particles {
            
            let distance = p1.distance2_to(&p2);
            let _span_parent = info_span!("particle_1").entered();
            // p2
            let (t_fx_p1, t_fy_p1, t_fz_p1) = p1.force_from(&p2, Some(distance));

            {
                let _span = info_span!("sum_forces").entered();
                fx_p1 += t_fx_p1;
                fy_p1 += t_fy_p1;
                fz_p1 += t_fz_p1;
                _span.exit();
            }

            let _span1 = info_span!("apply_velocities").entered();
            p1.vx += (fx_p1 / p1.mass) * dt;
            p1.vy += (fy_p1 / p1.mass) * dt;
            p1.vz += (fz_p1 / p1.mass) * dt;
            _span1.exit();

            let _span1 = info_span!("apply_position").entered();
            p1.x += p1.vx * dt;
            p1.y += p1.vy * dt;
            p1.z += p1.vz * dt;
            _span1.exit();
            _span_parent.exit();
        }
    }

    _span_all.exit();
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

fn collision_manager(mut commands: Commands, query: Query<(Entity, &Transform), With<MassiveObjects>>, mut query2: Query<&mut Particle>, treeaccess: Res<NNTree>) {
    for (entity, transform) in query.iter() {
        let radius = {
            let p1: Particle = match query2.get(entity) {
                Ok(e) => *e,
                Err(_) => continue
            };
            
            // let span = info_span!("mass_check").entered();
            // if p1.mass > 10000.0 {
            //     commands.entity(entity).insert(MassiveObjects);
            // } else {
            //     commands.entity(entity).remove::<MassiveObjects>();
            // }
            // span.exit();
            
            p1.radius() * 1.20
        };

        
        let span = info_span!("check_tree").entered();
        for (_, entity2) in treeaccess.within_distance(transform.translation, radius as f32) {
            let [p1, p2] = match query2.get_many_mut([entity, entity2]) {
                Ok(e) => e,
                Err(_) => continue
            };
            let mut p1: Mut<Particle> = p1;
            let p2: Mut<Particle> = p2;
            
            // if p1.mass > 50.0 || p2.mass > 50.0 {
            if true {
                if p2.mass > p1.mass {
                    continue
                }

                // Combine
                commands.entity(entity2).despawn();

                let momentum_x = p1.mass * p1.vx + p2.mass * p2.vx * 0.75; // assume some energy lost;
                let momentum_y = p1.mass * p1.vy + p2.mass * p2.vy * 0.75; // assume some energy lost;

                // if p2.color == YELLOW || p1.color == YELLOW {
                //     p1.color = YELLOW;
                // }

                p1.mass += p2.mass;
                p1.vx = momentum_x / p1.mass;
                p1.vy = momentum_y / p1.mass;
            }
        }
        span.exit();
    }
}

fn transform_objects(mut query: Query<(&mut Particle, &mut Transform)>) {
    for (particle, transform) in query.iter_mut() {
        // Force typing
        let particle: Mut<Particle> = particle;
        let mut transform: Mut<Transform> = transform;

        *transform = Transform::from_xyz(particle.x as f32, particle.y as f32, particle.z as f32)
    }
}