use bevy::prelude::Component;
use glw::glfw::ffi::{glfwMakeContextCurrent, GLFWwindow, glfwGetCurrentContext};
use glw::glfw::{make_context_current, Window, WindowEvent};
use glw::{glfw, gl, GLContext};

use glfw::{Context, WindowHint};
use bevy::math::Vec3A;
use glw::program::CommandList;
use glw::shader::ShaderType;
use glw::{Color, RenderTarget, Shader, Uniform, Vec2, MemoryBarrier};
use glw::buffers::{StructuredBuffer, BufferResource};
use std::error::Error;

use std::mem::MaybeUninit;
use std::sync::mpsc::Receiver;

use rand::*;
use std::os::raw::c_void;

#[allow(dead_code)]
#[derive(Default, Copy, Clone, Debug, Component)]
pub struct Particle {
    pub mass: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64
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
                    1,
                    1,
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
            receiver: ReceiverPtr(events)
        })
    }

    pub fn run(&mut self, dt: f32, particles: Vec<Particle>) -> Option<Vec<Particle>> {
        // self.glfw.set_swap_interval(glfw::SwapInterval::Sync(1))
        // unsafe {
        //     glfwMakeContextCurrent(self.window.0.window_ptr());
        // }
        let ctx = GLContext {};

        // gl::load_with(|s| {
        //     // println!("{} {:?}", s, self.window.0.get_proc_address(s) as *const _);
        //     self.window.0.get_proc_address(s) as *const _
        // });


        self.glfw.make_context_current(Some(&self.window.0));
        let mut cmd_list = ctx.create_command_list();

        cmd_list.bind_pipeline(&self.compute_program);

        let length = particles.len();
        cmd_list.set_uniform("u_particle_count", Uniform::Float(length as f32));
        cmd_list.set_uniform("u_dt", Uniform::Float(dt));

        let mut prev_sb = StructuredBuffer::from(particles);
        let mut curr_sb = StructuredBuffer::<Particle>::new(length as usize);

        cmd_list.bind_buffer(&curr_sb,0);
        cmd_list.bind_buffer(&prev_sb,1);

        cmd_list.dispatch(
            1,
            1,
            1,
        );

        cmd_list.memory_barrier(MemoryBarrier::ShaderStorage);

        // std::mem::swap(&mut curr_sb, &mut prev_sb);
        let mut buffer: Vec<Particle> = Vec::new();

        
        unsafe {
            // println!("one");
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, prev_sb.get_id());
            // println!("two");
            let d : *const Particle = gl::MapBuffer(gl::SHADER_STORAGE_BUFFER, gl::READ_ONLY) as *const Particle;
            if d.is_null() { return None }
            // println!("{}", d.is_null());
            for i in 0..length {
                // println!("{:?}", *d.offset(i as isize));
                if d.offset(i as isize).is_null() { return None }
                // println!("after")
                buffer.push(*d.offset(i as isize));
            }
            // println!("{:?}", buffer);
            // std::ptr::copy_nonoverlapping(d, buffer.as_mut_ptr(), length);
            // println!("four");
            // println!("{:?}", buffer);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            // println!("five");
        }
        // self.ctx.execute_command_list(&cmd_list);
        // println!("a");

        Some(buffer)
    }
    
}