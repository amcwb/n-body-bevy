use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
};
use bevy_prototype_lyon::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution, Rng};

#[derive(Component)]
struct Particle {
    mass: f64,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

const GRAVITY: f64 = 0.000000000066742 * 1000.0;
impl Particle {
    fn force_from(&mut self, other: &Particle, distance2: Option<f64>) -> (f64, f64) {
        let _span = info_span!("calculate_force").entered();
        let distance2 = match distance2 {
            Some(e) => e,
            None => self.distance2_to(other)
        };
        let f = (GRAVITY * self.mass * other.mass) / distance2;
        let theta = (other.y - self.y).atan2(other.x - self.x);

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
}

fn camera_control(
    mut scroll_evr: EventReader<MouseWheel>,
    time: Res<Time>,
    mut query: Query<&mut OrthographicProjection>,
) {
    let dist = 1.0 * time.delta().as_secs_f32();
    let mut scroll_y = 0.0;
    for ev in scroll_evr.iter() {
        match ev.unit {
            MouseScrollUnit::Line => {
                // println!("Scroll (line units): vertical: {}, horizontal: {}", ev.y, ev.x);
                scroll_y += ev.y
            }
            MouseScrollUnit::Pixel => {
                // println!("Scroll (pixel units): vertical: {}, horizontal: {}", ev.y, ev.x);
                scroll_y += ev.y
            }
        }
    }

    for mut projection in query.iter_mut() {
        let mut log_scale = projection.scale.ln();

        log_scale += scroll_y * dist;

        projection.scale = log_scale.exp();
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
        .add_startup_system(setup_camera)
        .add_startup_system(add_particles)
        .add_system(apply_forces)
        .add_system(transform_objects)
        .add_system(camera_control)
        .run();
}

fn add_particles(mut commands: Commands) {
    let direction = 1.0; // clockwise
    let star_mass = 2000000000.0;
    let max_distance = 5000;
    let max_mass = 1000;
    let amount = 10000;

    let center_x = 0.0; // -250.0;
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
    // let amount = 100;

    // let center_x = 250.0;
    // let center_y = 0.0;
    // let base_vx = 0.0;
    // let base_vy = 0.0;

    // create_system(&mut commands, max_distance, max_mass, center_x, center_y, base_vx, base_vy, star_mass, amount, direction);
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

fn create_star(
    commands: &mut Commands,
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
}

fn apply_forces(
    time: Res<Time>,
    mut query: Query<(Entity, &mut Particle)>,
) {
    let dt = time.delta_seconds() as f64;
    let mut combinations = query.iter_combinations_mut();
    while let Some([a, b]) = combinations.fetch_next() {
        let mut p1: Mut<Particle> = a.1;
        let mut p2: Mut<Particle> = b.1;
        
        // if p1.mass > 50.0 || p2.mass > 50.0 {
        //     let target: &mut Mut<Particle>;
        //     let origin: &mut Mut<Particle>;
        //     let origin_entity: &Entity;

        //     if p1.mass > p2.mass {
        //         target = &mut p1;
        //         origin = &mut p2;
        //         origin_entity = &b.0;
        //     } else {
        //         target = &mut p2;
        //         origin = &mut p1;
        //         origin_entity = &a.0;
        //     }

        //     let distance = target.distance_to(origin.as_ref());
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

        
        if p1.mass < 1000000.0 && p2.mass < 1000000.0 {
            continue;
        }

        let distance = p1.distance2_to(&p2);

        // TODO: Handle collisions
        // if p1.distance_to(&p2) < 1.0 {
        //     continue;
        // }

        if p2.mass > 1000000.0 {
            // p1
            let mut fx_p1: f64 = 0.0;
            let mut fy_p1: f64 = 0.0;
            let (t_fx_p1, t_fy_p1) = p1.force_from(&p2, Some(distance));
            {
                let _span = info_span!("apply_velocities").entered();
                fx_p1 += t_fx_p1;
                fy_p1 += t_fy_p1;

                p1.vx += (fx_p1 / p1.mass) * dt;
                p1.vy += (fy_p1 / p1.mass) * dt;

                p1.x += p1.vx * dt;
                p1.y += p1.vy * dt;
                _span.exit();
            }
            // println!("{} -> {}: {} {}", p2.mass, p1.mass, fx_p1, fy_p1);
        }

        if p1.mass > 1000000.0 {
            // p2
            let mut fx_p2: f64 = 0.0;
            let mut fy_p2: f64 = 0.0;
            let (t_fx_p2, t_fy_p2) = p2.force_from(&p1, Some(distance));

            {
                let _span = info_span!("apply_velocities").entered();
                fx_p2 += t_fx_p2;
                fy_p2 += t_fy_p2;

                p2.vx += (fx_p2 / p2.mass) * dt;
                p2.vy += (fy_p2 / p2.mass) * dt;

                p2.x += p2.vx * dt;
                p2.y += p2.vy * dt;
                _span.exit();
            }
            // println!("{} -> {}: {} {}", p1.mass, p2.mass, fx_p2, fy_p2);
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
