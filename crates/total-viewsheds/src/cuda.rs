use color_eyre::eyre::{ContextCompat as _, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, HostSlice, LaunchConfig};
use cudarc::driver::{DeviceRepr, PushKernelArg};
use cudarc::nvrtc;
use cudarc::nvrtc::CompileOptions;
use cudarc::runtime::result::get_mem_info;
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, mpsc};
use std::time::Instant;
use std::{f64, thread};
use crate::axes::SECTOR_STEPS;

pub struct CudaKernel {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    angle_kernel: CudaFunction,
}

const MB: usize = 1_000_000;

#[inline(always)]
fn calculate_point(
    y: isize,
    x_center: isize,
    y_center: isize,
    x_sin: f64,
    x_cos: f64,
    sin: f64,
    cos: f64,
    width: isize,
) -> isize {
    let y_sin = (y - y_center) as f64 * sin;
    let y_cos = (y - y_center) as f64 * cos;

    let x_rot = (x_cos - y_sin).round() as isize + y_center;
    let y_rot = (y_cos + x_sin).round() as isize + x_center;

    x_rot * width + y_rot
}

/// `generate_rotation` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
fn generate_rotation(elevation_count: usize, angle: f64, max_los: usize) -> Vec<u32> {
    let width = (max_los * 3) as isize;

    assert_eq!(width % 2, 0);
    assert!(elevation_count as isize % width == 0 && elevation_count as isize / width == width);

    let (sin, cos) = (f64::sin(angle.to_radians()), f64::cos(angle.to_radians()));
    let (x_center, y_center) = (width / 2, width / 2);

    let mut res = Vec::with_capacity(width as usize * max_los);
    for x in (max_los as isize)..(max_los as isize) * 2 {
        let x_sin = (x - x_center) as f64 * sin;
        let x_cos = (x - x_center) as f64 * cos;
        for y in 0..width {
            let y_sin = (y - y_center) as f64 * sin;
            let y_cos = (y - y_center) as f64 * cos;

            let x_rot = (x_cos - y_sin).round() as isize + y_center;
            let y_rot = (y_cos + x_sin).round() as isize + x_center;

            let new_idx = x_rot * width + y_rot;
            debug_assert!(new_idx < elevation_count as isize, "{new_idx}");
            debug_assert!(new_idx >= 0, "{new_idx}");

            res.push(new_idx as u32);
        }
    }

    res
}

// TODO: this currently eats over half my RAM on my big machine.
//       Good news is that this takes a total of 20s for all 180 angles, giving us
//       An overhead of .1s/angle. Funnily enough, this does very well multithreaded,
//       Single threaded it seems to take quite a bit longer??
fn multithread_rotations(elevation_count: usize, max_los_points: usize) -> Result<Vec<Vec<u32>>> {
    let threads = (0..crate::axes::SECTOR_STEPS)
        .map(|angle| thread::spawn(move || {
            generate_rotation(elevation_count, angle as f64, max_los_points)
        }))
        .collect::<Vec<_>>();

    let mut res = vec![];
    for thread in threads {
        res.push(thread.join().unwrap());
    }

    Ok(res)
}

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
                        "-use_fast_math".to_owned(),
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
        max_los_points: usize,
        elevs: &[f32],
        cumulative_surfaces: usize,
    ) -> Result<Vec<f32>> {
        let half_elevs = elevs.iter().map(|&x| (x as i32) as i16).collect_vec();

        let elevations = self.stream.memcpy_stod(&half_elevs)?;
        let result = self.stream.alloc_zeros::<f32>(cumulative_surfaces)?;

        // Use the above "overhead" to calculate about how much space we have left
        // on the GPU so that we can use the maximum
        let overhead = elevations.num_bytes() + result.num_bytes();

        tracing::info!("data overhead: {}MB", overhead / MB);

        // we'll have |deltas| f32s, all future calculations are in bytes so we need size_of
        let (free_bytes, total) = get_mem_info()?;
        tracing::info!(
            "{}MB free / {}MB total",
            (free_bytes - overhead) / MB,
            total / MB
        );

        let angles = crate::axes::SECTOR_STEPS as u32;
        // let angles = 1;


        let rot_time = Instant::now();
        multithread_rotations(half_elevs.len(), max_los_points)?;
        println!("Took {:?} to process rotation", rot_time.elapsed());

        let time = Instant::now();

        self.calculate_angles(&elevations, &result)?;

        let res = self.stream.memcpy_dtov(&result)?;
        println!("Took {:?} to process angles: {}", time.elapsed(), angles);

        Ok(res)
    }
}
