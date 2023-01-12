use std::{borrow::Cow, str::FromStr};

use futures::executor;
use bevy::{prelude::{Component, Resource, info_span, Plugin, App, SystemStage, Query, Mut, Transform, World, ResMut, Res, AssetServer, Handle, Shader, FromWorld, Commands}, render::{settings, renderer::{RenderDevice, RenderContext}, RenderStage, RenderApp, render_resource::{PipelineCache, ComputePipelineDescriptor, BindGroupLayout, CachedComputePipelineId, BindGroup, CachedPipelineState, BufferId, AsBindGroup}, render_graph::{self, RenderGraph}, MainWorld}, time::Time};
use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Instance, Adapter, Device, Queue, Buffer, ShaderModule, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages, BindingType, StorageTextureAccess, TextureFormat, TextureViewDimension, BindGroupDescriptor, ComputePassDescriptor, BufferBindingType};

#[derive(Default, Copy, Clone, Debug, Component, AsBindGroup)]
pub struct Particle {
    #[uniform(0)]
    pub x: f32,
    #[uniform(0)]
    pub y: f32,
    #[uniform(0)]
    pub z: f32,
    #[uniform(0)]
    pub vx: f32,
    #[uniform(0)]
    pub vy: f32,
    #[uniform(0)]
    pub vz: f32,
    #[uniform(0)]
    pub mass: f32,
}

#[derive(Default, Copy, Clone, Debug, Component)]
pub struct ParticleWithDt {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    // pub fx: f32,
    // pub fy: f32,
    // pub fz: f32,
    pub mass: f32,
    pub dt: f32
}

impl Particle {
    pub fn expand(&self, dt: f32) -> ParticleWithDt {
        ParticleWithDt {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
            mass: self.mass,
            dt
        }
    }
}

impl ParticleWithDt {
    pub fn truncate(&self) -> Particle {
        Particle {
            x: self.x,
            y: self.y,
            z: self.z,
            vx: self.vx,
            vy: self.vy,
            vz: self.vz,
            mass: self.mass
        }
    }
}

unsafe impl Zeroable for ParticleWithDt {}
unsafe impl Pod for ParticleWithDt {}

#[derive(Resource)]
struct StagingBuffer(Buffer);

#[derive(Resource)]
struct StorageBuffer(Buffer);


#[derive(Resource)]
struct ComputeBindGroup(BindGroup);

#[derive(Resource)]
pub struct ComputePipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

enum ComputeNodeState {
    Loading,
    Update
}
#[derive(Component)]
pub struct ComputeNode {
    state: ComputeNodeState,
}

impl Default for ComputeNode {
    fn default() -> Self {
        Self {
            state: ComputeNodeState::Loading,
        }
    }
}
pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<ComputePipeline>()
            .add_system_to_stage(RenderStage::Extract, extract_particles)
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("compute", ComputeNode::default());
    }
}

#[derive(Resource)]
struct ParticleStorage(Vec<Particle>);

fn extract_particles(
    mut commands: Commands,
    mut world: ResMut<MainWorld>
) {
    let mut particles = world.query::<&Particle>();
    println!("Extract size {:?}", particles.iter(&world).len());
    commands.insert_resource(ParticleStorage(particles.iter(&world).cloned().collect::<Vec<_>>()));
}

fn queue_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    particles: Res<ParticleStorage>,
    time: Res<Time>,
) {
    let _expand_span = info_span!("expand_particles").entered();
    let expanded_particles = particles.0.iter().map(|p| p.expand(time.elapsed_seconds())).collect::<Vec<ParticleWithDt>>();
    _expand_span.exit();
    let slice_size = expanded_particles.len() * std::mem::size_of::<ParticleWithDt>();
    let size = slice_size as wgpu::BufferAddress;

    println!("Size: {:?} / {:?}", size, particles.0.len());
    if size == 0 {
        return;
    }


    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let _storage_span = info_span!("create_storage_buffer").entered();
    let storage_buffer = render_device.wgpu_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&expanded_particles),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    _storage_span.exit();

    let _staging_span = info_span!("create_staging_buffer").entered();
    let staging_buffer = render_device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    _staging_span.exit();

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });
    
    commands.insert_resource(ComputeBindGroup(bind_group));
    commands.insert_resource(StagingBuffer(staging_buffer));
    commands.insert_resource(StorageBuffer(storage_buffer));
}

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> ComputePipeline {
        let bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        // let bind_group =
        //     world
        //         .resource::<RenderDevice>()
        //         .create_bind_group(&wgpu::BindGroupDescriptor {
        //             label: None,
        //             layout: &bind_group_layout,
        //             entries: &[wgpu::BindGroupEntry {
        //                 binding: 0,
        //                 resource: storage_buffer.as_entire_binding(),
        //             }],
        //         });

        // Create bind groups from our buffers and then queue our function
        // let render_device = world.get_resource::<RenderDevice>()
        //     .cloned()
        //     .expect("Failed to get render device");
            
        // Configure shader resource
        let _shader_span = info_span!("load_shader").entered();
        let shader: Handle<Shader> = world
            .resource::<AssetServer>()
            .load("shader.wgsl");
        
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: None,
            shader,
            entry_point: Cow::from("main"),
            shader_defs: vec![],
        });
        _shader_span.exit();

        ComputePipeline {
            pipeline,
            bind_group_layout
        }
    }
}

impl render_graph::Node for ComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ComputePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            ComputeNodeState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.pipeline)
                {
                    self.state = ComputeNodeState::Update
                }
            }
            ComputeNodeState::Update => {}
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let command_bind_group = world.get_resource::<ComputeBindGroup>();
        if let Some(command_bind_group) = command_bind_group {
            println!("Requesting gpu");
            let storage_buffer = &world.resource::<StorageBuffer>().0;
            let staging_buffer = &world.resource::<StagingBuffer>().0;
            
            let pipeline_cache = world.resource::<PipelineCache>();
            let pipeline = world.resource::<ComputePipeline>();
    
            // Read only access requires us to use iter_entities
            let particles = world.iter_entities()
                .map(|e| world.get_entity(e).unwrap().get::<Particle>())
                .filter(|e| e.is_some())
                .map(|e| e.unwrap())
                .collect::<Vec<_>>();
    
            let particles_len = particles.len();
    
            let encoder = &mut render_context.command_encoder;
            encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, (particles_len * std::mem::size_of::<ParticleWithDt>()) as u64);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
    
            pass.set_bind_group(0, &command_bind_group.0, &[]);
    
            // select the pipeline based on the current state
            match self.state {
                ComputeNodeState::Loading => {}
                ComputeNodeState::Update => {
                    let update_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.pipeline)
                        .unwrap();
                        
                    pass.set_pipeline(update_pipeline);
                    pass.dispatch_workgroups(particles_len as u32, 1, 1);
                }
            }
        }
        Ok(())
    }
}