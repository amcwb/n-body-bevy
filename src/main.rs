use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use bevy_prototype_lyon::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution, Rng};
// use directx_math::XMScalarSinCos;
use bevy_spatial::{KDTreeAccess2D, KDTreePlugin2D, SpatialAccess};

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
    vx: f64,
    vy: f64,
}

#[derive(Component)]
struct TimeScale(f64);

#[derive(Component)]
struct MassiveObjects;


// type alias for easier usage later
type NNTree = KDTreeAccess2D<Particle>;

const GRAVITY: f64 = 0.000000000066742 * 1000.0;
impl Particle {
    fn force_from(&self, other: &Particle, distance2: Option<f64>) -> (f64, f64) {
        let _span = info_span!("calculate_force_from").entered();
        let distance2 = match distance2 {
            Some(e) => e,
            None => self.distance2_to(other),
        };
        let _span2 = info_span!("calculate_force").entered();
        let f = (GRAVITY * self.mass * other.mass) / distance2;
        _span2.exit();
        let _span2 = info_span!("calculate_atan").entered();
        let theta = (other.y - self.y).atan2(other.x - self.x);
        _span2.exit();

        // let mut cos: f32 = 0.0;
        // let mut sin: f32 = 0.0;
        // XMScalarSinCos(&mut sin, &mut cos, theta as f32);
        let fx = theta.cos() * f;
        let fy = theta.sin() * f;

        let res = (fx, fy);
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
        let res = (other.x - self.x).powi(2) + (other.y - self.y).powi(2);
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
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
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

    for (mut transform, mut projection) in query.iter_mut() {
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
        .add_plugins(DefaultPlugins)
        .add_plugin(ShapePlugin)
        .add_plugin(KDTreePlugin2D::<Particle> { ..default() })
        .add_startup_system(setup_camera)
        .add_startup_system(add_particles)
        .add_system(apply_forces)
        .add_system(transform_objects)
        .add_system(camera_control)
        .add_system(massive_objects_manager)
        .add_system(collision_manager)
        .run();
}

fn add_particles(mut commands: Commands) {
    let direction = 1.0; // clockwise
    let star_mass = 2000000000.0;
    let max_distance = 500;
    let max_mass = 1000;
    let amount = 1000;

    let center_x = -125.0; // -250.0;
    let center_y = 0.0;
    let base_vx = 0.0;
    let base_vy = 0.0;

    create_system(
        &mut commands,
        max_distance,
        max_mass,
        center_x,
        center_y,
        base_vx,
        base_vy,
        star_mass,
        amount,
        direction,
    );

    // let direction = -1.0; // anti-clockwise
    // let star_mass = 1000000000.0;
    // let max_distance = 500;
    // let max_mass = 500;
    // let amount = 10000;

    // let center_x = 1000.0;
    // let center_y = 0.0;
    // let base_vx = 0.0;
    // let base_vy = 0.0;

    // create_system(
    //     &mut commands,
    //     max_distance,
    //     max_mass,
    //     center_x,
    //     center_y,
    //     base_vx,
    //     base_vy,
    //     star_mass,
    //     amount,
    //     direction,
    // );
}

fn create_system(
    mut commands: &mut Commands,
    max_distance: i32,
    max_mass: i32,
    center_x: f64,
    center_y: f64,
    base_vx: f64,
    base_vy: f64,
    star_mass: f64,
    amount: i32,
    direction: f64,
) {
    let mut rng = rand::thread_rng();
    let distance = Uniform::from(10..=max_distance);
    let mass = Uniform::from(1..=max_mass);

    create_star(
        &mut commands,
        center_x,
        center_y,
        base_vx,
        base_vy,
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

        let r_x = center_x + theta.cos() * r;
        let r_y = center_y + -theta.sin() * r;

        create_star(
            &mut commands,
            r_x,
            r_y,
            v_x,
            v_y,
            m,
            rng.gen::<f32>(),
            rng.gen::<f32>(),
            rng.gen::<f32>(),
        );
    }
}

fn create_star<'a>(
    commands: &'a mut Commands,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    mass: f64,
    r: f32,
    g: f32,
    b: f32,
) {
    let shape = shapes::Circle {
        radius: (mass.log10()) as f32,
        ..shapes::Circle::default()
    };
    
    commands
        .spawn_bundle(GeometryBuilder::build_as(
            &shape,
            DrawMode::Outlined {
                fill_mode: FillMode::color(Color::rgb(r, g, b)),
                outline_mode: StrokeMode::new(Color::WHITE, 0.5),
            },
            Transform::default(),
        ))
        .insert(Particle { mass, x, y, vx, vy });

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

    query.par_for_each_mut(&pool, 64, |mut p2| {
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
            let mut fx_p2: f64 = 0.0;
            let mut fy_p2: f64 = 0.0;
            let (t_fx_p2, t_fy_p2) = p2.force_from(&p1, Some(distance));

            {
                let _span = info_span!("sum_forces").entered();
                fx_p2 += t_fx_p2;
                fy_p2 += t_fy_p2;

                let _span2 = info_span!("apply_velocities").entered();
                p2.vx += (fx_p2 / p2.mass) * dt;
                p2.vy += (fy_p2 / p2.mass) * dt;
                _span2.exit();

                let _span2 = info_span!("apply_position").entered();
                p2.x += p2.vx * dt;
                p2.y += p2.vy * dt;
                _span2.exit();
                _span.exit();
            }
            _span_parent.exit();
        }
    });

    for p1idx in 0..massive_objects.len() {
        let (p1, particles) = massive_objects.split_one_mut(p1idx);
        for p2 in particles {
            
            let distance = p1.distance2_to(&p2);
            let _span_parent = info_span!("particle_1").entered();
            // p2
            let mut fx_p1: f64 = 0.0;
            let mut fy_p1: f64 = 0.0;
            let (t_fx_p1, t_fy_p1) = p1.force_from(&p2, Some(distance));

            {
                let _span = info_span!("sum_forces").entered();
                fx_p1 += t_fx_p1;
                fy_p1 += t_fy_p1;

                let _span1 = info_span!("apply_velocities").entered();
                p1.vx += (fx_p1 / p1.mass) * dt;
                p1.vy += (fy_p1 / p1.mass) * dt;
                _span1.exit();

                let _span1 = info_span!("apply_position").entered();
                p1.x += p1.vx * dt;
                p1.y += p1.vy * dt;
                _span1.exit();
                _span.exit();
            }
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

fn collision_manager(mut commands: Commands, query: Query<(Entity, &Transform)>, mut query2: Query<&mut Particle>, treeaccess: Res<NNTree>) {
    for (entity2, transform) in query.iter() {
        let p2: Particle = match query2.get(entity2) {
            Ok(e) => *e,
            Err(_) => continue
        };
        let p2 = p2.clone();
        for (_, entity) in treeaccess.within_distance(transform.translation, p2.radius() as f32) {
            let mut p1: Mut<Particle> = query2.get_mut(entity2).unwrap();

            
            // if p1.mass > 50.0 || p2.mass > 50.0 {
            if true {
                if p1.mass > p2.mass {
                    continue
                }

                let distance = p1.distance_to(&p2);
                if distance < (p1.radius() + p2.radius()) / 2.0 {
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
        }
    }
}

fn transform_objects(mut query: Query<(&mut Particle, &mut Transform)>) {
    for (particle, transform) in query.iter_mut() {
        // Force typing
        let particle: Mut<Particle> = particle;
        let mut transform: Mut<Transform> = transform;

        *transform = Transform::from_xyz(particle.x as f32, particle.y as f32, 0.0)
    }
}