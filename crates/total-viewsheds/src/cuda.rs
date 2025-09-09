use crate::axes;
use crate::compute::Angle;
use color_eyre::eyre::{ContextCompat as _, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::{DeviceRepr, PushKernelArg};
use cudarc::nvrtc;
use cudarc::nvrtc::CompileOptions;
use cudarc::runtime::result::get_mem_info;
use half::f16;
use itertools::Itertools;
use radsort::sort;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::mem;
use std::mem::forget;
use std::sync::Arc;
use std::time::Instant;
use total_viewsheds_kernel::kernel::Constants;
use wgpu::hal::Device;

pub struct CudaKernel {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    angle_kernel: CudaFunction,
}

const MB: usize = 1_000_000;

#[repr(C)]
struct Dimensions {
    angles: u32,
    total_bands: u32,
    max_los_as_points: u32,
    dem_width: u32,
    tvs_width: u32,
    observer_height: f32,
}

unsafe impl DeviceRepr for Dimensions {}

impl CudaKernel {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)?;

        let angle_kernel: CudaFunction;
        {
            let kernel = nvrtc::compile_ptx_with_opts(
                include_str!("angles.cu"),
                CompileOptions {
                    options: vec![
                        // "-G".to_string(),
                        "--generate-line-info".to_owned(),
                        "--include-path=/usr/local/cuda/include/".to_owned(),
                        "--include-path=/usr/local/cuda/include/cccl/".to_owned(),
                        "--extra-device-vectorization".to_owned(),
                    ],
                    ..CompileOptions::default()
                },
            )?;
            let module = ctx.load_module(kernel)?;
            angle_kernel = module.load_function("angle_kernel")?;
        }

        // JIT the kernel

        let stream = ctx.default_stream();

        Ok(Self {
            ctx,
            stream,
            angle_kernel,
        })
    }

    fn calculate_angles(
        &self,
        elevations: &CudaSlice<i16>,
        angle_buf: &CudaSlice<f32>,
    ) -> Result<()> {
        // TODO: this is extremely tuned for Everest, maybe make this a bit more general?
        let launch_cfg = LaunchConfig {
            block_dim: (1000, 1, 1),
            grid_dim: (6000, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.angle_kernel);
        builder.arg(elevations);
        builder.arg(angle_buf);

        unsafe {
            builder.launch(launch_cfg)?;
        }

        Ok(())
    }


    /// `line_of_sight` calculates all lines of sights for
    pub fn line_of_sight(
        &self,
        elevs: &[f32],
        cumulative_surfaces: usize,
    ) -> Result<Vec<f32>> {
        let half_elevs = elevs.iter().map(|&x| { (x as i32) as i16 } ).collect_vec();

        let elevations = self.stream.memcpy_stod(&half_elevs)?;
        let result = self.stream.alloc_zeros::<f32>(cumulative_surfaces)?;

        // Use the above "overhead" to calculate about how much space we have left
        // on the GPU so that we can use the maximum
        let overhead = elevations.num_bytes()
            + result.num_bytes();

        tracing::info!("data overhead: {}MB", overhead/MB);

        // we'll have |deltas| f32s, all future calculations are in bytes so we need size_of
        let (free_bytes, total) = get_mem_info()?;
        tracing::info!(
            "{}MB free / {}MB total",
            (free_bytes - overhead) / MB,
            total / MB
        );

        // let angles = axes::SECTOR_STEPS as u32;
        let angles = 1;

        let mut time = Instant::now();

        self.calculate_angles(
            &elevations,
            &result,
        )?;

        let res = self.stream.memcpy_dtov(&result)?;
        println!("Took {:?} to process angles: {}", time.elapsed(), angles);

        Ok(res)
    }
}
