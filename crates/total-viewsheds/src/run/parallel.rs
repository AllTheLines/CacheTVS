//! For kernels that run each angle in parallel.

use crate::cpu::kernel::OutputData;
use crate::cpu::{kernel, storage};
use crate::los_pack::LineOfSightPacked;
use color_eyre::Result;
use std::sync::{mpmc, mpsc};
use std::{thread, time};

/// `Work` holds all data needed for a worker to do a line of sight
/// calculation
struct Work {
    /// angle of rotation for line of sight
    angle: u16,
}

fn kernel_worker(
    storage_worker: &storage::worker::Worker,
    elevations: &[i16],
    work_todo: mpmc::Receiver<Work>,
    res: &mpsc::Sender<OutputData>,
    config: &crate::run::compute::Config,
) {
    let size = config.metadata.max_line_of_sight.pow(2) as usize;
    let mut output = OutputData {
        surfaces: vec![0.0f32; size],
        longest: vec![LineOfSightPacked::new(0, 0).unwrap(); size],
    };

    for work in work_todo {
        tracing::info!("starting work on {}", work.angle);
        let now = time::Instant::now();
        kernel(
            storage_worker,
            elevations,
            &mut output,
            f32::from(work.angle),
            config,
        );
        tracing::info!("finished {} in {:?}", work.angle, now.elapsed());
    }
    res.send(output).expect("");
}

fn compilation_worker(
    work: mpsc::Receiver<OutputData>,
    max_los: usize,
) -> (Vec<f32>, Vec<LineOfSightPacked>) {
    let mut surfaces = vec![0.0f32; max_los * max_los];
    let mut longest = vec![LineOfSightPacked::new(0, 0).unwrap(); max_los * max_los];

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

impl super::compute::Compute<'_> {
    /// `run_parallel` runs the CPU kernel in parallel
    pub fn run_parallel(&mut self) -> Result<()> {
        let max_los = usize::try_from(self.dem.max_los_as_points)?;

        let elevations = &self.dem.elevations;
        let config = &self.config;

        let (local_send, kernel_receive) = mpmc::channel();
        let (kernel_send, compile_receive) = mpsc::channel();

        let borrow_elevations = &elevations;

        let (out_surfaces, out_longest) = thread::scope(|s| {
            let worker_handles: Result<Vec<_>> = (0..self.config.thread_count)
                .map(|id| {
                    let db_worker = if Self::is_process_viewsheds(&self.config.process) {
                        let db_path = self.config.viewsheds_db_path.join(format!("{id}.db"));

                        let db = crate::cpu::storage::db::DB::new(&db_path)?;
                        db.save_metadata(&config.metadata)?;

                        crate::cpu::storage::worker::Worker::new(&db_path)
                    } else {
                        crate::cpu::storage::worker::Worker::new_noop()
                    };

                    let kernel_send = kernel_send.clone();
                    let kernel_receive = kernel_receive.clone();

                    Ok(s.spawn(move || {
                        kernel_worker(
                            &db_worker,
                            borrow_elevations,
                            kernel_receive,
                            &kernel_send,
                            config,
                        );
                    }))
                })
                .collect::<Result<Vec<_>>>();

            let handles = worker_handles.expect("");

            for angle in 0u16..360u16 {
                local_send.send(Work { angle }).unwrap();
            }

            drop(local_send);
            drop(kernel_receive);
            drop(kernel_send);

            let (surfaces, longest) = compilation_worker(compile_receive, max_los);

            for handle in handles {
                handle.join().unwrap();
            }

            (surfaces, longest)
        });

        self.total_surfaces = out_surfaces;
        self.longest_lines = out_longest;

        self.render_total_surfaces()?;
        self.render_longest_lines()?;

        // if Self::is_process_viewsheds(&self.config.process) {
        //     crate::cpu::storage::db::DB::new(&self.config.viewsheds_db_path)?.create_indexes()?;
        // }

        Ok(())
    }
}
