use std::{borrow::Cow, str::FromStr};

use futures::executor;
use bevy::prelude::{Component, Resource, info_span};
use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Instance, Adapter, Device, Queue};

#[derive(Default, Copy, Clone, Debug, Component)]
pub struct Particle {
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


#[derive(Debug, Resource)]
pub struct Application {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl Application {
    pub fn new_sync() -> Application {
        executor::block_on(Application::new())
    }

    /// This is staying using async, we are going to use
    /// ```rust
    /// use futures::executor; // 0.3.1
    /// fn main() {
    ///     let v = executor::block_on(example());
    ///     println!("{}", v);
    /// }    
    /// ```
    /// Within the systems added to bevy
    pub async fn new() -> Application {
        // Instantiates instance of WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Expected an adapter");

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let info = adapter.get_info();
        // skip this on LavaPipe temporarily
        if info.vendor == 0x10005 {
            panic!("Invalid Vendor");
        }

        Application {
            instance,
            adapter,
            device,
            queue
        }
    }

    pub fn run_sync(&mut self, dt: f32, particles: Vec<Particle>) -> Option<Vec<Particle>> {
        executor::block_on(self.run(dt, particles))
    }

    pub async fn run(&mut self, dt: f32, particles: Vec<Particle>) -> Option<Vec<Particle>> {
        let root_span = info_span!("calculate_force_from").entered();
        let _expand_span = info_span!("expand_particles").entered();
        let expanded_particles = particles.iter().map(|p| p.expand(dt)).collect::<Vec<ParticleWithDt>>();
        _expand_span.exit();

        let _shader_span = info_span!("load_shader").entered();
        // Loads the shader from WGSL
        let cs_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wsgl"))),
        });
        _shader_span.exit();

        // Gets the size in bytes of the buffer.
        let slice_size = particles.len() * std::mem::size_of::<ParticleWithDt>();
        let size = slice_size as wgpu::BufferAddress;

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let _staging_span = info_span!("create_staging_buffer").entered();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        _staging_span.exit();

        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let _storage_span = info_span!("create_storage_buffer").entered();
        let storage_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&expanded_particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        _storage_span.exit();

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let _pipeline_span = info_span!("create_pipeline").entered();
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });
        _pipeline_span.exit();

        let _bind_group_span = info_span!("create_bind_group").entered();
        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
        });
        _bind_group_span.exit();

        let _encoder_span = info_span!("command_encoder").entered();
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute gravity");
            cpass.dispatch_workgroups(particles.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        _encoder_span.exit();
        
        let _clone_span = info_span!("clone_buffer").entered();
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
        _clone_span.exit();

        // Submits command encoder for processing
        self.queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _poll_span = info_span!("poll").entered();
        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);
        _poll_span.exit();

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = receiver.receive().await {
            let _cast_span = info_span!("cast").entered();
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<ParticleWithDt> = bytemuck::cast_slice(&data).to_vec();
            _cast_span.exit();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            let _truncate_span = info_span!("truncate").entered();
            // Returns data from buffer
            let res = result.iter().map(|p| p.truncate()).collect::<Vec<Particle>>();
            _truncate_span.exit();
            root_span.exit();
            Some(res)
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
