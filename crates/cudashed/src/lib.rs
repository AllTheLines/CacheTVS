use color_eyre::eyre::Result;
use cudarc::driver::PushKernelArg;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, HostSlice, LaunchConfig};
use cudarc::nvrtc;
use cudarc::nvrtc::CompileOptions;
use itertools::Itertools;
use std::sync::Arc;
use std::time::Instant;

pub struct CudaKernel {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    angle_kernel: CudaFunction,
    rotate_kernel: CudaFunction,
}

impl CudaKernel {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)?;

        let angle_kernel: CudaFunction;
        {
            let kernel = nvrtc::compile_ptx_with_opts(
                include_str!("kernels/angles.cu"),
                CompileOptions {
                    options: vec![
                        "-use_fast_math".to_owned(),
                    ],
                    ..CompileOptions::default()
                },
            )?;
            let module = ctx.load_module(kernel)?;
            angle_kernel = module.load_function("angle_kernel")?;
        }

        let rotate_kernel: CudaFunction;
        {
            let kernel = nvrtc::compile_ptx_with_opts(
                include_str!("kernels/rotate.cu"),
                CompileOptions {
                    options: vec![
                        "-use_fast_math".to_owned(),
                    ],
                    ..CompileOptions::default()
                },
            )?;
            let module = ctx.load_module(kernel)?;
            rotate_kernel = module.load_function("rotate_kernel")?;
        }

        // JIT the kernel
        let stream = ctx.default_stream();

        Ok(Self {
            ctx,
            stream,
            angle_kernel,
            rotate_kernel,
        })
    }

    fn rotate(
        &self,
        elevations: &CudaSlice<i16>,
        rot_elevation_buf: &CudaSlice<i16>,
        rot_index_buf: &CudaSlice<i32>,
        angle: u32,
        num_angles: u32,
    ) -> Result<()> {
        let launch_cfg = LaunchConfig {
            block_dim: (32, 32, 1),
            grid_dim: (188, 375, num_angles),
            shared_mem_bytes: 0,
        };

        // rotate_kernel(
        //     const short* __restrict__ elevations,
        //     short* __restrict__ elevations_out,
        //     const int* __restrict__ index_out,
        //     int angle_off
        // )
        let mut builder = self.stream.launch_builder(&self.rotate_kernel);
        builder.arg(elevations);
        builder.arg(rot_elevation_buf);
        builder.arg(rot_index_buf);
        builder.arg(&angle);

        unsafe {
            builder.launch(launch_cfg)?;
        }

        Ok(())
    }

    fn calculate_angles(
        &self,
        elevations: &CudaSlice<i16>,
        idxs: &CudaSlice<i32>,
        num_angles: u32,
        angle_buf: &CudaSlice<f32>,
    ) -> Result<()> {


        // TODO: this is extremely tuned for Everest, maybe make this a bit more general?
        let launch_cfg = LaunchConfig {
            block_dim: (1000, 1, 1),
            grid_dim: (6000, num_angles, 1),
            shared_mem_bytes: 0,
        };

        // void angle_kernel(
        //     const short* __restrict__ elevations,
        //     const int* __restrict__ idxs,
        //     float* __restrict__ result
        // )
        let mut builder = self.stream.launch_builder(&self.angle_kernel);
        builder.arg(elevations);
        builder.arg(idxs);
        builder.arg(angle_buf);

        unsafe {
            builder.launch(launch_cfg)?;
        }

        Ok(())
    }

    /// `line_of_sight` calculates all lines of sights for
    pub fn line_of_sight(
        &self,
        max_los_points: usize,
        elevs: &[f32],
        cumulative_surfaces: usize,
    ) -> Result<Vec<f32>> {
        let timed = Instant::now();

        let half_elevs = elevs
            .iter()
            .map(|&x| (x as i32) as i16)
            .collect_vec();

        let result = self.stream.alloc_zeros::<f32>(cumulative_surfaces)?;
        let elev_buffer = self.stream.memcpy_stod(&half_elevs)?;

        const STEP: usize = 1;

        // TODO: fix uint32_t overflow in rotation kernel.
        //       The real limit is likely 59 but that is an uneven number.
        //       It is likely because 2^32 < 60*12000*6000.
        assert!(STEP <= 45);

        let rotated_elevs = self.stream.alloc_zeros::<i16>(STEP * max_los_points * 2*max_los_points)?;
        let rotated_indexes = self.stream.alloc_zeros::<i32>(STEP * max_los_points * max_los_points)?;

        for angle in (0..360).step_by(STEP) {
            self.rotate(&elev_buffer, &rotated_elevs, &rotated_indexes, angle, STEP as u32)?;
            self.calculate_angles(&rotated_elevs, &rotated_indexes, STEP as u32, &result)?;
        }

        let res = self.stream.memcpy_dtov(&result)?;

        println!("Total kernel took {:?}", timed.elapsed());

        Ok(res)
    }
}
