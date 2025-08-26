use color_eyre::eyre::{ContextCompat as _, Result};
use std::mem;

use crate::compute::Angle;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::{DeviceRepr, PushKernelArg};
use cudarc::nvrtc;
use cudarc::nvrtc::CompileOptions;
use cudarc::runtime::result::get_mem_info;
use std::sync::Arc;
use std::time::Instant;
use total_viewsheds_kernel::kernel::Constants;
use crate::axes;

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
        let max_kernel: CudaFunction;

        {
            let kernel = nvrtc::compile_ptx_with_opts(
                include_str!("angles.cu"),
                CompileOptions {
                    options: vec![
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
        constants: &Dimensions,
        elevations: &CudaSlice<f32>,
        distances: &CudaSlice<f32>,
        forwards_deltas: &CudaSlice<i32>,
        backward_deltas: &CudaSlice<i32>,
        angle_buf: &CudaSlice<f32>,
        total_sectors: u32,
        angles: u32,
    ) -> Result<()> {
        // tracing::debug!("Calculating angles for {}", n);

        // TODO: this is extremely tuned for Everest, maybe make this a bit more general?
        let launch_cfg = LaunchConfig {
            block_dim: (750, 1, 1),
            grid_dim: (total_sectors, angles, 1),
            shared_mem_bytes: 0,
        };


        // assert_eq!(
        //     launch_cfg.block_dim.1 * launch_cfg.grid_dim.1,
        //     deltas_size as u32
        // );

        let mut builder = self.stream.launch_builder(&self.angle_kernel);
        builder.arg(constants);
        builder.arg(elevations);
        builder.arg(distances);
        builder.arg(forwards_deltas);
        builder.arg(backward_deltas);
        builder.arg(angle_buf);

        unsafe {
            builder.launch(launch_cfg)?;
        }

        Ok(())
    }

    // [10, -2, 10, 5]
    // kernel_id(forward):  0 + ineach(10, 8, 18, 23)
    // kernel_id(backward): 0 + ineach(-10, -8, -18, -23)
    //
    // dem_id = tvs_id
    // for (delta in deltas)
    //   dem_id += delta
    //   distance[dem_id]
    //   elevation[dem_id]
    //
    // dem_id = tvs_id
    // dem_id += forward[delta_index]
    //

    // fn get_pov_id(line_num: usize, total_bands: usize) {
    //
    //     let mut tvs_id = line_num % total_bands;
    //     let angle = line_num / total_bands;
    //
    //     // determine whether we are forwards or backwards facing
    //     let half_total_bands = total_bands/2;
    //
    //     tvs_id %= half_total_bands;
    //     bool forward = line_num < half_total_bands;
    //
    //     int pov_x = (tvs_id % constants.tvs_width) + constants.max_los_as_points;
    //     int pov_y = (tvs_id / constants.tvs_width) + constants.max_los_as_points;
    // }

    pub fn line_of_sight(
        &self,
        constants: &Constants,
        angles: &[Angle],
        elevations: &[f32],
        cumulative_surfaces: usize,
    ) -> Result<Vec<f32>> {
        let all_angles = struct_of_arrays(angles);

        // allocate all the forward deltas/backward deltas, elevations, and the cumulative
        // surfaces result
        let forward_deltas = self.stream.memcpy_stod(&all_angles.forward_deltas)?;
        let backward_deltas = self.stream.memcpy_stod(&all_angles.forward_deltas)?;
        let band_distances = self.stream.memcpy_stod(&all_angles.band_distances)?;

        let elevations = self.stream.memcpy_stod(elevations)?;
        let result = self.stream.alloc_zeros::<f32>(cumulative_surfaces)?;

        // Use the above "overhead" to calculate about how much space we have left
        // on the GPU so that we can use the maximum
        let overhead = forward_deltas.num_bytes()
            + backward_deltas.num_bytes()
            + band_distances.num_bytes()
            + elevations.num_bytes()
            + result.num_bytes();

        tracing::debug!("data overhead: {}MB", overhead / MB);

        // we'll have |deltas| f32s, all future calculations are in bytes so we need size_of
        let line_size = angles[0].forward_deltas.len() * size_of::<f32>();
        tracing::debug!("line size: {}KB", line_size / 1000);

        let (free_bytes, total) = get_mem_info()?;
        tracing::debug!(
            "{}MB free / {}MB total",
            (free_bytes - overhead) / MB,
            total / MB
        );

        let dims = Dimensions {
            angles: crate::axes::SECTOR_STEPS as u32,
            total_bands: constants.total_bands,
            max_los_as_points: constants.max_los_as_points,
            dem_width: constants.dem_width,
            tvs_width: constants.tvs_width,
            observer_height: constants.observer_height,
        };

        dbg!(constants.total_bands);
        dbg!(constants.tvs_width);
        dbg!(constants.max_los_as_points);

        let mut time = Instant::now();
        self.calculate_angles(
            &dims,
            &elevations,
            &band_distances,
            &forward_deltas,
            &backward_deltas,
            &result,
            constants.total_bands,
            axes::SECTOR_STEPS as u32,
        )?;

        let res = self.stream.memcpy_dtov(&result)?;
        println!("Took {:?} to process angles: {}", time.elapsed(), 1);

        Ok(res)
    }
}

fn struct_of_arrays(angles: &[Angle]) -> Angle {
    let forward = angles
        .iter()
        .map(|a| a.forward_deltas.clone())
        .flatten()
        .collect::<Vec<_>>();

    let backwards = angles
        .iter()
        .map(|a| a.backward_deltas.clone())
        .flatten()
        .collect::<Vec<_>>();

    let band_distances = angles
        .iter()
        .map(|a| a.band_distances.clone())
        .flatten()
        .collect::<Vec<_>>();

    Angle {
        forward_deltas: forward,
        backward_deltas: backwards,
        band_distances,
    }
}
