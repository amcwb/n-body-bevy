use bevy::prelude::{Component, info_span};
use glw::gl::types::{GLenum, GLchar};
use glw::glfw::{Window, WindowEvent, Context};
use glw::{glfw, gl, GLContext};

use glfw::{WindowHint};
use glw::program::{CommandList, Pipeline};
use glw::shader::ShaderType;
use glw::{Shader, Uniform, MemoryBarrier};
use glw::buffers::{StructuredBuffer, BufferResource};
use std::sync::mpsc::Receiver;



#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Default, Copy, Clone, Debug, Component)]
pub struct ParticleWithForceMarker {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
    pub mass: f32,
}

impl From<Particle> for ParticleWithForceMarker {
    fn from(x: Particle) -> Self {
        Self {
            mass: x.mass,
            x: x.x,
            y: x.y,
            z: x.z,
            vx: x.vx,
            vy: x.vy,
            vz: x.vz,
            fx: 0.0,
            fy: 0.0,
            fz: 0.0
        }
    }
}

impl From<ParticleWithForceMarker> for Particle {
    fn from(x: ParticleWithForceMarker) -> Self {
        Self {
            mass: x.mass,
            x: x.x,
            y: x.y,
            z: x.z,
            vx: x.vx,
            vy: x.vy,
            vz: x.vz
        }
    }
}
struct WindowPtr(Window);
struct ReceiverPtr(Receiver<(f64, WindowEvent)>);

unsafe impl Send for WindowPtr {}
unsafe impl Sync for WindowPtr {}
unsafe impl Send for ReceiverPtr {}
unsafe impl Sync for ReceiverPtr {}
pub struct Application {
    /// GLFW specific things
    glfw: glfw::Glfw,

    compute_program: glw::GraphicsPipeline,

    // Program state
    is_paused: bool,
    gl_ctx: glw::GLContext,
    window: WindowPtr,
    receiver: ReceiverPtr,

    prev_sb: StructuredBuffer<Particle>,
    curr_sb: StructuredBuffer<Particle>,
}

impl Application {
    pub fn new() -> Result<Application, Box<dyn std::error::Error>> {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS)?;
        glfw.window_hint(WindowHint::ContextVersion(4, 5));
        glfw.window_hint(WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

        // TODO: Use bevy window
        let (mut window, events) = glfw.with_primary_monitor(|instance, _mon| {
            instance
                .create_window(
                    100,
                    100,
                    "Util window for physics",
                    glfw::WindowMode::Windowed,
                ).unwrap()
        });
        let ctx = glw::GLContext::new(&mut window);

        let compute_program = {
            let mut c_shader = Shader::new(ShaderType::Compute);
            c_shader.load_from_file("Shaders/shader.compute").unwrap();

            glw::PipelineBuilder::new()
                .with_compute_shader(c_shader)
                .build()
        };
        // TODO: Does this help?
        // window.hide();

        Ok(Application {
            glfw,
            is_paused: false,
            compute_program,
            gl_ctx: ctx,
            window: WindowPtr(window),
            receiver: ReceiverPtr(events),
            curr_sb: StructuredBuffer::default(),
            prev_sb: StructuredBuffer::default()
        })
    }

    pub fn run(&mut self, dt: f32, particles: Vec<Particle>) -> Option<Vec<Particle>> {
        // unsafe {
        //     glfwMakeContextCurrent(self.window.0.window_ptr());
        // }
        

        let my_span = info_span!("configure-opengl").entered();
        let mut ctx = GLContext {};
        // println!("aaaa");

        // gl::load_with(|s| {
            // println!("{} {:?}", s, self.window.0.get_proc_address(s) as *const _);
        //     self.window.0.get_proc_address(s) as *const _
        // });
        // let mut particles: Vec<Particle> = Vec::from_iter(particles.iter().map(|p| Particle::from(*p)));
        // get_error(true);
        // println!("aaaa");

        let my_span2 = info_span!("configure-context").entered();
        self.glfw.make_context_current(Some(&self.window.0));
        my_span2.exit();
        // println!("aaaa");

        // self.glfw.set_swap_interval(glfw::SwapInterval::Sync(1));
        // println!("aaaa");

        
        let my_span2 = info_span!("configure-command-list").entered();
        let mut cmd_list = ctx.create_command_list();
        // println!("bbbbb");
        ctx.set_debug();
        my_span2.exit();

        let my_span2 = info_span!("bind-pipeline-and-set-uniform").entered();
        cmd_list.bind_pipeline(&self.compute_program);
        // println!("bbbbb");

        let length = particles.len();
        cmd_list.set_uniform("u_particle_count", Uniform::Int(length as i32));
        // println!("bbbbb");
        cmd_list.set_uniform("u_dt", Uniform::Float(dt));
        // println!("bbbbb");
        my_span2.exit();
        my_span.exit();

        unsafe {
            // let cname = std::ffi::CString::new("u_particle_count").expect("CString::new failed");
            
            let d = gl::GetUniformLocation(self.compute_program.get() as u32, "u_particle_count".to_string().as_ptr() as *const GLchar);
            if d == 0 {
                return None
            }
        }

        if self.prev_sb.get_buffer_size() != particles.len() * self.prev_sb.get_structure_size() {
            // println!("ccccc");
            self.prev_sb = StructuredBuffer::from(particles);
            self.curr_sb = StructuredBuffer::<Particle>::new(length as usize);
        } else {
            // println!("ddddd");
            // unsafe {
            //     gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, self.curr_sb.get_id());
            //     let d : *mut Particle = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_WRITE) as *mut Particle;
            //     println!("{:?}", d);
            //     std::ptr::copy(particles.as_ptr(), d , length);
            //     // for i in 0..length {
            //     //     // if p1.mass > 10000.0 { println!("{:?}", p1); }
            //     //     if d.offset(i as isize).is_null() {
            //     //         gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            //     //         break
            //     //     }
            //     //     *d.offset(i as isize) = particles[i];
            //     //     // println!("{:?}", p1);;
            //     // }
            //     gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            // }
            // self.prev_sb.map_data(&particles);
            // println!("eeeee");
            let my_span = info_span!("recreate-buffer").entered();
            self.prev_sb = StructuredBuffer::from(particles);
            self.curr_sb = StructuredBuffer::<Particle>::new(length as usize);
            my_span.exit();
            // println!("eeeee");
        }

        let my_span = info_span!("bind-and-dispatch").entered();
        cmd_list.bind_buffer(&mut self.curr_sb,0);
        cmd_list.bind_buffer(&mut self.prev_sb,1);

        cmd_list.dispatch(
            length as u32,
            1,
            1,
        );
        
        cmd_list.memory_barrier(MemoryBarrier::ShaderStorage);


        std::mem::swap(&mut self.curr_sb, &mut self.prev_sb);
        my_span.exit();
        let buffer: Vec<Particle>;

        unsafe {
            // println!("one");
            // buffer.set_len(length);
            // get_error(true);
            let my_span = info_span!("collect-results").entered();
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, self.prev_sb.get_id());  // gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, prev_sb.get_id());
            // println!("two");
            let d : *const Particle = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const Particle;
            // println!("three");
    
            if d.is_null() {
                gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
                return None
            }
            // for i in 0..length {
            //     // if p1.mass > 10000.0 { println!("{:?}", p1); }
            //     if d.offset(i as isize).is_null() {
            //         gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            //         return None;
            //     }
            //     let p1 = *d.offset(i as isize);
            //     // println!("{:?}", p1);
            //     buffer.push(Particle::from(p1));
            // }
            // println!("{:?}", buffer);
            buffer = std::slice::from_raw_parts(d, length).to_vec();
            // std::ptr::copy_nonoverlapping(d, buffer.as_mut_ptr(), length);
            // println!("four");
            // println!("{:?}", buffer);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            my_span.exit();
            // println!("five");
        }

        // get_error(false);
        Some(buffer)
    }
    
}

fn get_error(log: bool) {
    unsafe {
        let mut err = gl::GetError();
        while err != gl::NO_ERROR {
            if log { println!("{}", err.to_string()) };
            err = gl::GetError();
        }
    }
}