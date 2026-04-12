//! For kernels that run each angle in parallel.

use crate::cpu::{kernel, storage};
use crate::cpu::kernel::OutputData;
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
    max_los: usize,
) {
    let mut output = OutputData {
        surfaces: vec![0.0f32; max_los*max_los],
        longest: vec![LineOfSightPacked::new(0, 0).unwrap(); max_los*max_los]
    };

    for work in work_todo {
        tracing::info!("starting work on {}", work.angle);
        let now = time::Instant::now();
        kernel(
            storage_worker,
            elevations,
            &mut output,
            max_los,
            f32::from(work.angle),
            config,
        );
        tracing::info!("finished {} in {:?}", work.angle, now.elapsed());
    }
    res.send(output).expect("");

}

fn compilation_worker(work: mpsc::Receiver<OutputData>, max_los: usize) -> (Vec<f32>, Vec<LineOfSightPacked>) {
    let mut surfaces = vec![0.0f32; max_los * max_los];
    let mut longest = vec![LineOfSightPacked::new(0, 0).unwrap(); max_los * max_los];


    for data in work {
        surfaces
            .iter_mut()
            .zip(data.surfaces)
            .for_each(|(to, from)| {
                *to += from;
            });

        longest
            .iter_mut()
            .zip(data.longest)
            .for_each(|(to, from)| {
                *to = to.max(from);
            });
    }

    (surfaces, longest)
}

impl super::compute::Compute<'_> {
    /// `run_parallel` runs the CPU kernel in parallel
    pub fn run_parallel(&mut self) -> Result<()> {
        let max_los = usize::try_from(self.dem.max_los_as_points)?;

        let db_worker = &if Self::is_process_viewsheds(&self.config.process) {
            if std::fs::exists(&self.config.viewsheds_db_path)? {
                std::fs::remove_file(&self.config.viewsheds_db_path)?;
            }

            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "Should always fit in `u32`"
            )]
            let max_los_metric = (self.dem.max_los_as_points as f32 * self.dem.scale) as u32;

            let db = crate::cpu::storage::db::DB::new(&self.config.viewsheds_db_path)?;
            db.save_metadata(&crate::cpu::storage::metadata::MetaData {
                width: self.dem.width,
                scale: self.dem.scale,
                max_line_of_sight: max_los_metric,
                reserved_ring_size: 0,
                centre: self.dem.centre,
            })?;

            crate::cpu::storage::worker::Worker::new(&self.config.viewsheds_db_path)
        } else {
            crate::cpu::storage::worker::Worker::new_noop()
        };

        let elevations = &self.dem.elevations;
        let thread_count = self.config.thread_count;
        let config = &self.config;

        let (local_send, kernel_receive) = mpmc::channel();
        let (kernel_send, compile_receive) = mpsc::channel();

        let borrow_elevations = &elevations;

        let (out_surfaces, out_longest) = thread::scope(|s| {

            let worker_handles = std::iter::repeat_with(|| {
                    let kernel_send = kernel_send.clone();
                    let kernel_receive = kernel_receive.clone();
                    s.spawn(move || {
                        kernel_worker(
                            db_worker,
                            borrow_elevations,
                            kernel_receive,
                            &kernel_send,
                            config,
                            max_los,
                        );
                    })
                }).take(thread_count)
                .collect::<Vec<_>>();

            for angle in 0u16..360u16 {
                local_send
                    .send(Work { angle })
                    .unwrap();
            }

            drop(local_send);
            drop(kernel_receive);
            drop(kernel_send);

            let (surfaces, longest) = compilation_worker(compile_receive, max_los);

            for handle in worker_handles {
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
