struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    mass: f32
};

@group(0)
// Binding 0 matches wgpu::BindGroupEntry
@binding(0)
var<storage, read_write> v_particles: array<Particle>; // this is used as both input and output for convenience

@group(0)
// Binding 0 matches wgpu::BindGroupEntry
@binding(1)
var<uniform> dt: f32; // this is used as both input and output for convenience

let GRAVITY: f32 = 0.000000066742;

fn calculate_gravity(particle1: Particle, index: u32) -> Particle {
    var pos1: vec3<f32> = vec3<f32>(particle1.x, particle1.y, particle1.z);
    var directed_force: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    
    for ( var i: i32 = 0; i < i32(arrayLength(&v_particles)); i++ ) {
        if (i == i32(index)) {
            continue;
        }

        var particle2: Particle = v_particles[i];
        var pos2: vec3<f32> = vec3<f32>(particle2.x, particle2.y, particle2.z);

        var d_sqrt: f32 = distance(pos1, pos2);
        if (d_sqrt < 0.1) {
            continue;
        }

        var raw_force: f32 = (GRAVITY * particle1.mass * particle2.mass) / (d_sqrt * d_sqrt);
        var d_axis = pos2 - pos1;
        directed_force = directed_force + (d_axis / d_sqrt) * raw_force;
    }
    
    var velocity: vec3<f32> = vec3(particle1.vx, particle1.vy, particle1.vz) + directed_force / particle1.mass * dt;
    var position: vec3<f32> = velocity * dt;

    return Particle(
        particle1.x + position.x,
        particle1.y + position.y,
        particle1.z + position.z,
        velocity.x,
        velocity.y,
        velocity.z,
        particle1.mass
    );
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_particles[global_id.x] = calculate_gravity(v_particles[global_id.x], global_id.x);
}