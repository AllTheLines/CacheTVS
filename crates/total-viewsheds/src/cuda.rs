use crate::axes::SECTOR_STEPS;
use color_eyre::eyre::Result;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, HostSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc;
use cudarc::nvrtc::CompileOptions;
use itertools::Itertools;
use std::sync::Arc;
use std::time::Instant;
use std::{f64, thread};

pub struct CudaKernel {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    angle_kernel: CudaFunction,
}

const MB: usize = 1_000_000;


/// `generate_rotation` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
fn generate_rotation(elevation_count: usize, angle: f64, max_los: usize) -> Vec<i32> {
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

            let new_idx = x_rot.clamp(0, width-1) * width + y_rot.clamp(0, width-1);
            let normalized = if new_idx >= 0 && new_idx < elevation_count as isize {
                new_idx
            } else {
                panic!("bad idx: {new_idx}")
            };

            res.push(normalized as i32);
        }
    }

    res
}

// TODO: this currently eats over half my RAM on my big machine.
//       Good news is that this takes a total of 20s for all 180 angles, giving us
//       An overhead of .1s/angle. Funnily enough, this does very well multithreaded,
//       Single threaded it seems to take quite a bit longer??
fn multithread_rotations(
    elevation_count: usize,
    max_los_points: usize,
    count: usize,
    offset: usize,
) -> Result<Vec<Vec<i32>>> {
    let threads = (offset..offset + count)
        .map(|angle| {
            thread::spawn(move || generate_rotation(elevation_count, angle as f64, max_los_points))
        })
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

    fn calculate_angles(&self, elevations: &[i16], idxs: &[i32], angle_buf: &CudaSlice<f32>) -> Result<()> {
        let elevs = self.stream.memcpy_stod(elevations)?;
        let idxs = self.stream.memcpy_stod(idxs)?;

        // TODO: this is extremely tuned for Everest, maybe make this a bit more general?
        let launch_cfg = LaunchConfig {
            block_dim: (1000, 1, 1),
            grid_dim: (6000, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = self.stream.launch_builder(&self.angle_kernel);
        builder.arg(&elevs);
        builder.arg(&idxs);
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

        let result = self.stream.alloc_zeros::<f32>(cumulative_surfaces)?;

        let angles = u32::from(SECTOR_STEPS);
        // let angles = 1;

        let mut time = Instant::now();

        for angle in (0..360).step_by(180) {
            let rotated: Vec<Vec<i32>> =
                multithread_rotations(half_elevs.len(), max_los_points, 180, angle)?;

            println!("Took {:?} to process angles: 180", time.elapsed());

            for rotation in rotated {
                time = Instant::now();
                let elevations = rotation
                    .iter()
                    .map(|&idx| {
                        if idx < 0i32 {
                            i16::MIN
                        } else {
                            *unsafe { half_elevs.get_unchecked(idx as usize) }
                        }
                    })
                    .collect::<Vec<i16>>();

                let idxs = rotation
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| {
                        let col = *idx % 18000;
                        (6000..12000).contains(&col)
                    })
                    .map(|(_, val)| {
                        let x = (val / 18000) - 6000;
                        let y = (val % 18000) - 6000;
                        if (0..6000).contains(&x) && (0..6000).contains(&y) {
                            x*6000+y
                        } else {
                            -1i32
                        }
                    })
                    .collect_vec();

                self.calculate_angles(&elevations, &idxs, &result)?;

                println!("Took {:?} to process angle", time.elapsed());
                time = Instant::now();
            }
        }

        let res = self.stream.memcpy_dtov(&result)?;
        Ok(res)
    }
}
