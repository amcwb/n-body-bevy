use std::{borrow::Cow, str::FromStr, sync::{Arc, Mutex, RwLock}, panic::AssertUnwindSafe};

use futures::{executor, FutureExt};
use bevy::{prelude::{Component, Resource, info_span, Plugin, App, SystemStage, Query, Mut, Transform, World, ResMut, Res, AssetServer, Handle, Shader, FromWorld, Commands, Entity}, render::{settings, renderer::{RenderDevice, RenderContext, RenderQueue}, RenderStage, RenderApp, render_resource::{PipelineCache, ComputePipelineDescriptor, BindGroupLayout, CachedComputePipelineId, BindGroup, CachedPipelineState, BufferId, AsBindGroup}, render_graph::{self, RenderGraph}, MainWorld, texture::FallbackImage, render_asset::RenderAssets, RenderWorld}, time::Time, utils::hashbrown::HashMap};
use bytemuck::{Pod, Zeroable};
use futures_intrusive::{channel::shared::{OneshotSender, OneshotReceiver, GenericSender, GenericReceiver}, buffer::GrowingHeapBuf};
use wgpu::{util::DeviceExt, Instance, Adapter, Device, Queue, Buffer, ShaderModule, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages, BindingType, StorageTextureAccess, TextureFormat, TextureViewDimension, BindGroupDescriptor, ComputePassDescriptor, BufferBindingType, BufferAsyncError};
use bevy::render::render_resource::*;
use parking_lot::RawMutex;

use crate::TimeScale;

#[derive(Default, Copy, Clone, Debug, Component, AsBindGroup, ShaderType)]
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

unsafe impl Zeroable for Particle {}
unsafe impl Pod for Particle {}

#[derive(Resource)]
struct StagingBuffer(Buffer);

#[derive(Resource)]
struct StagingBufferSlice<'a>(wgpu::BufferSlice<'a>);

#[derive(Resource)]
struct StorageBuffer(Buffer);


#[derive(Resource)]
struct ComputeBindGroup(BindGroup);

#[derive(Resource, Clone)]
pub struct ComputePipeline {
    bind_group_layout: Option<BindGroupLayout>,
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
            // .add_system_to_stage(RenderStage::Render, render);
            
        // Create a oneshot channel for this render
        let (sender, receiver) = futures_intrusive::channel::shared::channel::<Result<(), BufferAsyncError>>(9999);
        let oneshot = Oneshot {
            sender: Arc::new(Mutex::new(sender)),
            receiver: Arc::new(Mutex::new(receiver))
        };

        render_app.insert_resource(oneshot);
        render_app.insert_resource(ProcessNonce);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("compute", ComputeNode::default());
    }
}

#[derive(Resource)]
struct ParticleRenderStorage(Vec<Particle>);

#[derive(Resource)]
struct ReverseParticleRenderStorage(Vec<Particle>);

#[derive(Resource)]
struct ProcessNonce;

#[derive(Resource)]
struct ResolvedDt(f64);

#[derive(Resource)]
struct Oneshot<T> where T: 'static {
    sender: Arc<Mutex<GenericSender<RawMutex, T, GrowingHeapBuf<T>>>>,
    receiver: Arc<Mutex<GenericReceiver<RawMutex, T, GrowingHeapBuf<T>>>>,
}

fn extract_particles(
    mut commands: Commands,
    mut world: ResMut<MainWorld>,
    buffer: Option<Res<StagingBuffer>>,
    oneshot: Res<Oneshot<Result<(), BufferAsyncError>>>,
    time: Res<Time>,

    // Solely to check if we've added this already
    prs: Option<Res<ParticleRenderStorage>>,
) {
    // println!("Extract");
    let mut particles = world.query::<(&mut Particle, Entity)>();

    // While we shouldn't be doing this, we're going to use the extract stage shared between the two worlds
    // to get data from the storage buffer.
    if prs.is_some() && buffer.is_some(){
        // println!("Block");
        // let receiver = executor::block_on(oneshot.receiver.lock().unwrap().receive());
        // println!("Post-block");
        let result = oneshot.receiver.lock().unwrap().receive().now_or_never();
        // println!("Result: {:?}", result);
        if let Some(Some(Ok(()))) = result {
        // if let Some(Ok(())) = receiver {
            let buffer = &buffer.unwrap().0;
            let buffer_slice = buffer.slice(..);
            // println!("Received message");

            
            // Gets contents of buffer
            // println!("Get mapped range");
            let data = std::panic::catch_unwind(AssertUnwindSafe(|| buffer_slice.get_mapped_range()));
            // println!("After get");
            if let Ok(data) = data {
                let _cast_span = info_span!("cast").entered();
                // Since contents are got in bytes, this converts these bytes back to u32
                let result: Vec<Particle> = bytemuck::cast_slice(&data).to_vec();

                // println!("Dataaaaaa {:?}", result[0..3].to_vec());
                // panic!("Exist");

                _cast_span.exit();

                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                drop(data);

                let _set_span = info_span!("set").entered();
                
                let mut i = 0;
                for mut particle in particles.iter_mut(&mut world) {
                    *particle.0 = result[i];
                    i += 1;
                }
                _set_span.exit();
                
            } else {
                // println!("Unable to get mapped range");
            }

            commands.insert_resource(ProcessNonce);
            // println!("Insert nonce");
        };
    }

    commands.insert_resource(ParticleRenderStorage(particles
        .iter(&world)
        .map(|(particle, _e)| particle)
        .cloned()
        .collect::<Vec<_>>()
    ));

    let time_scale = world.resource::<TimeScale>();
    commands.insert_resource(ResolvedDt(time_scale.0 * time.delta_seconds_f64()));
}

fn queue_bind_group(
    mut commands: Commands,
    oneshot: Res<Oneshot<Result<(), BufferAsyncError>>>,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    particles: Res<ParticleRenderStorage>,
    pipeline_cache: Res<PipelineCache>,
    nonce: Option<Res<ProcessNonce>>,
    dt: Res<ResolvedDt>
) {
    if nonce.is_none() {
        return;
    }

    // println!("Test 5");
    // let _expand_span = info_span!("expand_particles").entered();
    // let expanded_particles = particles.0.iter().map(|p| p.expand(time.elapsed_seconds())).collect::<Vec<ParticleWithDt>>();
    // _expand_span.exit();
    let slice_size = particles.0.iter().len() * std::mem::size_of::<Particle>();
    let size = slice_size as wgpu::BufferAddress;

    if size == 0 {
        return;
    }

    // println!("Dataaaa 2 {:?}", particles.0.iter().collect::<Vec<_>>()[0..3].to_vec());

    if let Some(bind_group_layout) = &pipeline.bind_group_layout {
        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a #up and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        // println!("Test 4");
        let _storage_span = info_span!("create_storage_buffer").entered();
        let storage_buffer = render_device.wgpu_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&particles.0.iter().cloned().collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // println!("{:?} {:?}", storage_buffer.size(), size);
        _storage_span.exit();

        // We don't care about the DT buffer
        let _storage_span = info_span!("create_dt_buffer").entered();
        let dt_buffer = render_device.wgpu_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DT Buffer"),
            contents: bytemuck::bytes_of::<f32>(&(dt.0 as f32)),
            usage: wgpu::BufferUsages::UNIFORM,
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

        let _bind_group_span = info_span!("create_bind_group").entered();
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dt_buffer.as_entire_binding(),
                }
                ],
        });
        _bind_group_span.exit();
        
        // commands.insert_resource(ComputeBindGroup(bind_group));
        // println!("Test 3");

        // let pipeline_cache = world.resource::<PipelineCache>();
        // let pipeline = world.resource::<ComputePipeline>();

        if let CachedPipelineState::Ok(_) = pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
            let compute_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .expect("Pipeline was promised, but not found");


            let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
            
            // Re-insert resource
            let pipeline = ComputePipeline {
                pipeline: pipeline.pipeline,
                bind_group_layout: Some(bind_group_layout.into())
            };

            // println!("Test 2");
            
            // Queue data here
            // let command_bind_group = world.get_resource::<ComputeBindGroup>();
            commands.remove_resource::<ProcessNonce>();
            // let storage_buffer = &world.resource::<StorageBuffer>().0;
            // let staging_buffer = &world.resource::<StagingBuffer>().0;
            
            // let pipeline_cache = world.resource::<PipelineCache>();
            // let pipeline = world.resource::<ComputePipeline>();

            // Read only access requires us to use iter_entities
            // let particles = world.iter_entities()
            //     .map(|e| world.get_entity(e).unwrap().get::<Particle>())
            //     .filter(|e| e.is_some())
            //     .map(|e| e.unwrap())
            //     .collect::<Vec<_>>();

            let _span = info_span!("create_encoder").entered();
            let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default()); 
                pass.set_bind_group(0, &bind_group, &[]);

                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
                    .unwrap();
                    
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(particles.0.iter().len() as u32, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size as u64);
            _span.exit();

            let sender = oneshot.sender.clone();
            
            render_queue.0.submit(Some(encoder.finish()));

            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                // println!("Send!");
                // This is technically possible to be a channelsenderror, but we can ignore it
                let res = sender.lock().expect("Unable to obtain lock").try_send(v);
                // println!("Result of send: {:?}", res);
            });

            render_device.poll(wgpu::Maintain::Poll);
            
            // println!("Test 1");
            // println!("Dispatch");
            // println!("Test 1 backup");

            commands.insert_resource(StagingBuffer(staging_buffer));
            commands.insert_resource(StorageBuffer(storage_buffer));
            // println!("Insert new staging buffer");
        }
        
    }
}

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> ComputePipeline {
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
            // bind_group_layout
            bind_group_layout: None
        }
    }
}

impl render_graph::Node for ComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipeline>();
        
        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            ComputeNodeState::Loading => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.pipeline)
                {
                    let compute_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.pipeline)
                        .expect("Pipeline was promised, but not found");

                    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
                    // Re-insert resource

                    // println!("Insert pipeline");
                    world.insert_resource(ComputePipeline {
                        pipeline: pipeline.pipeline,
                        bind_group_layout: Some(bind_group_layout.into())
                    });

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
        // let command_bind_group = world.get_resource::<ComputeBindGroup>();
        // if let Some(command_bind_group) = command_bind_group {
        //     let storage_buffer = &world.resource::<StorageBuffer>().0;
        //     let staging_buffer = &world.resource::<StagingBuffer>().0;
            
        //     let pipeline_cache = world.resource::<PipelineCache>();
        //     let pipeline = world.resource::<ComputePipeline>();
    
        //     // Read only access requires us to use iter_entities
        //     let particles = world.iter_entities()
        //         .map(|e| world.get_entity(e).unwrap().get::<Particle>())
        //         .filter(|e| e.is_some())
        //         .map(|e| e.unwrap())
        //         .collect::<Vec<_>>();
    
        //     let particles_len = particles.len();
    
        //     let encoder = &mut render_context.command_encoder;
        //     encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, (particles_len * std::mem::size_of::<ParticleWithDt>()) as u64);
        //     let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        

        //     let buffer_slice = staging_buffer.slice(..);
        //     let sender = world.resource::<Oneshot<Result<(), BufferAsyncError>>>().sender.clone();
            
        //     buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.lock().expect("Can't lock sender").send(v).expect("Can't send"));

        //     pass.set_bind_group(0, &command_bind_group.0, &[]);
    
        //     // select the pipeline based on the current state
        //     match self.state {
        //         // When loading, we don't have a bind group layout yet
        //         ComputeNodeState::Loading => {}
        //         ComputeNodeState::Update => {
        //             let update_pipeline = pipeline_cache
        //                 .get_compute_pipeline(pipeline.pipeline)
        //                 .unwrap();
                        
        //             pass.set_pipeline(update_pipeline);
        //             pass.dispatch_workgroups(particles_len as u32, 1, 1);
        //             // In case parent doesn't do this
        //             render_context.render_device.poll(wgpu::Maintain::Wait);
        //         }
        //     }
        // }
        Ok(())
    }
}