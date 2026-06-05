//! For kernels that run each angle in parallel.

use crate::compute::kernel;
use crate::compute::kernel::OutputData;
use crate::los_pack::LineOfSightPacked;
use color_eyre::Result;
use std::path::Path;
use std::sync::{mpmc, mpsc};
use std::{thread, time};

/// `Work` holds all data needed for a worker to do a line of sight
/// calculation
struct Work {
    /// angle of rotation for line of sight
    angle: u16,
}

/// "pinned" thread that "pops" work from `storage_worker`
/// aggregates the results as long as there is work to do,
/// and finally sends it to `res` to aggregate in the main thread
#[expect(clippy::expect_used, reason = "invariants broken if errors returned")]
fn kernel_worker(
    storage_worker: &crate::storage::worker::Worker,
    elevations: &[i16],
    work_todo: mpmc::Receiver<Work>,
    res: &mpsc::Sender<OutputData>,
    config: &crate::run::Config,
) {
    #[expect(clippy::as_conversions, reason = "usize will be u64")]
    let size = config.dem_metadata.max_line_of_sight.pow(2) as usize;
    let mut output = OutputData {
        surfaces: vec![0.0f32; size],
        longest: vec![LineOfSightPacked::new(0, 0).expect("0, 0 is a valid packed los"); size],
    };

    for work in work_todo {
        tracing::info!("starting work on {}", work.angle);
        let now = time::Instant::now();
        kernel(
            storage_worker,
            elevations,
            &mut output,
            work.angle,
            config.angle_subdivisions.into(),
            config,
        );
        tracing::info!("finished {} in {:?}", work.angle, now.elapsed());
    }
    res.send(output).expect("unable to send to ");
}

/// final aggregation step to aggregate the `kernel_worker`s results
fn compilation_worker(
    work: mpsc::Receiver<OutputData>,
    max_los: usize,
) -> (Vec<f32>, Vec<LineOfSightPacked>) {
    let mut surfaces = vec![0.0f32; max_los * max_los];
    let mut longest = vec![LineOfSightPacked::default(); max_los * max_los];

    for data in work {
        surfaces
            .iter_mut()
            .zip(data.surfaces)
            .for_each(|(to, from)| {
                *to += from;
            });

        longest.iter_mut().zip(data.longest).for_each(|(to, from)| {
            *to = to.max(from);
        });
    }

    (surfaces, longest)
}

/// create a db at path P only if `is_db_worker`, otherwise return
/// a noop worker
pub fn init_worker<P: AsRef<Path>>(
    path: P,
    meta_data: &tvs_lib::metadata::MetaData,
    is_db_worker: bool,
) -> Result<crate::storage::worker::Worker> {
    if !is_db_worker {
        return Ok(crate::storage::worker::Worker::new_noop());
    }

    let db = crate::storage::db::DB::new(&path)?;
    db.save_metadata(meta_data)?;

    Ok(crate::storage::worker::Worker::new(&path))
}

impl super::run::Compute<'_> {
    /// `run_parallel` runs the CPU kernel in parallel
    #[expect(clippy::expect_used, reason = "We need to panic on failure")]
    pub fn run(&mut self) -> Result<()> {
        let max_los = usize::try_from(self.dem.max_los_as_points)?;

        let elevations = &self.dem.elevations;
        let config = &self.config;

        let (local_send, kernel_receive) = mpmc::channel();
        let (kernel_send, compile_receive) = mpsc::channel();

        let borrow_elevations = &elevations;

        let should_init_global_db =
            Self::is_process_viewsheds(&self.config.process) && !self.config.database_per_thread;

        let should_init_local_db =
            Self::is_process_viewsheds(&self.config.process) && self.config.database_per_thread;

        let global_worker = &init_worker(
            &self.config.viewsheds_db_path,
            &self.config.dem_metadata,
            should_init_global_db,
        )?;

        let (out_surfaces, out_longest) = thread::scope(|scope| {
            let worker_handles: Result<Vec<_>> = (0..self.config.thread_count)
                .map(|id| {
                    let db_path = self.config.viewsheds_db_path.join(format!("{id}.db"));
                    let local_worker =
                        init_worker(db_path, &self.config.dem_metadata, should_init_local_db)?;

                    let copied_kernel_send = kernel_send.clone();
                    let copied_kernel_receive = kernel_receive.clone();

                    Ok(scope.spawn(move || {
                        kernel_worker(
                            if should_init_global_db {
                                global_worker
                            } else {
                                &local_worker
                            },
                            borrow_elevations,
                            copied_kernel_receive,
                            &copied_kernel_send,
                            config,
                        );
                    }))
                })
                .collect::<Result<Vec<_>>>();

            let handles = worker_handles.expect("unable to spawn workers");

            for angle in 0u16..(360u16 * u16::from(config.angle_subdivisions)) {
                local_send
                    .send(Work { angle })
                    .expect("unable to send work ro workers");
            }

            drop(local_send);
            drop(kernel_receive);
            drop(kernel_send);

            let (surfaces, longest) = compilation_worker(compile_receive, max_los);

            for handle in handles {
                handle.join().expect("unable to join worker thread");
            }

            (surfaces, longest)
        });

        self.total_surfaces = out_surfaces;
        self.longest_lines = out_longest;

        self.render_total_surfaces()?;
        self.render_longest_lines()?;

        Ok(())
    }
}
