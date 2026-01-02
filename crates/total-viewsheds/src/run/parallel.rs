//! For kernels that run each angle in parallel.

use color_eyre::{
    eyre::{eyre, ContextCompat as _},
    Result,
};
use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};


#[cfg(any(test, feature = "ring_data"))]
const PROCESS_RING_DATA: bool = true;

#[cfg(not(any(test, feature = "ring_data")))]
const PROCESS_RING_DATA: bool = false;

impl super::compute::Compute<'_> {

    #[expect(
        clippy::panic_in_result_fn,
        reason = "It's too complicated and of no benefit to get the errors from the threads"
    )]
    /// `run_parallel` runs the CPU kernel in parallel
    pub fn run_parallel(&mut self) -> Result<()> {
        let max_los = usize::try_from(self.dem.max_los_as_points)?;
        let tvs_size = max_los * max_los;
        let is_process_ring_data = Self::is_process_viewsheds(&self.config.process);
        let reserved_ring_data_size = if is_process_ring_data {
            usize::try_from(self.constants.reserved_rings_per_band)?
        } else {
            0
        };

        let mut surfaces = vec![0.0f32; tvs_size];
        let mut longest = vec![(0u16, 0.0f32); tvs_size];
        let mut ring_data = vec![vec![0u32; tvs_size * reserved_ring_data_size]; 360];

        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;

        {
            let accumulating = AccumulatingData {
                constants: self.constants,
                surfaces: std::sync::Mutex::new(&mut surfaces),
                longest: std::sync::Mutex::new(&mut longest),
                visibility: std::sync::Mutex::new(&mut ring_data),
            };

            let elevations = &self.dem.elevations;

            pool.install(move || {
                (0u16..360u16)
                    .into_par_iter()
                    .map(|angle| {
                        let start = std::time::Instant::now();
                        tracing::info!("starting angle: {angle}");

                        let output = crate::cpu::kernel(
                            elevations,
                            max_los,
                            f32::from(angle),
                            is_process_ring_data,
                        );
                        tracing::info!("finished angle in {:?}", start.elapsed());
                        (angle, output)
                    })
                    .for_each(|(angle, output)| {
                        let result = accumulating.handle_parallel_per_angle_output(angle, output);
                        #[expect(
                            clippy::panic,
                            reason = "No point accumulating errors and returning them"
                        )]
                        if let Err(error) = result {
                            panic!("{error:?}");
                        }
                    });
            });
        };

        self.total_surfaces = surfaces;
        let packed: Result<Vec<crate::los_pack::LineOfSightPacked>> = longest
            .iter()
            .map(|&(angle, distance): &(u16, f32)| {
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "distances always fit in u32"
                )]
                crate::los_pack::LineOfSightPacked::new(distance as u32, angle)
            })
            .collect();
        self.longest_lines = packed?;
        self.ring_data = ring_data;

        self.render_total_surfaces()?;
        self.render_longest_lines()?;

        if Self::is_process_viewsheds(&self.config.process)
            && self.config.output_directory.is_some()
        {
            for sector in 0..crate::run::compute::SECTOR_STEPS {
                self.save_sector_ring_data(
                    sector,
                    self.ring_data
                        .get(usize::from(sector))
                        .context("Sector not found in final ring data")?,
                )?;
            }
        }

        Ok(())
    }
}

/// A struct to accumulate data as it comes from the angle compute threads.
struct AccumulatingData<'accumulating> {
    /// Various common kernel constants.
    constants: kernel::constants::Constants,
    /// Total surfaces.
    surfaces: std::sync::Mutex<&'accumulating mut Vec<f32>>,
    /// Longest lines.
    longest: std::sync::Mutex<&'accumulating mut Vec<(u16, f32)>>,
    /// Ring data to reconstruct individual viewsheds.
    visibility: std::sync::Mutex<&'accumulating mut Vec<Vec<u32>>>,
}

impl AccumulatingData<'_> {
    /// Handle output from angle threads.
    fn handle_parallel_per_angle_output(
        &self,
        angle: u16,
        output: crate::cpu::kernel::OutputData,
    ) -> Result<()> {
        self.surfaces
            .lock()
            .map_err(|err| eyre!("{err:?}"))?
            .iter_mut()
            .zip(output.surfaces)
            .for_each(|(to, from)| {
                *to += from;
            });

        self.longest
            .lock()
            .map_err(|err| eyre!("{err:?}"))?
            .iter_mut()
            .zip(output.longest)
            .for_each(|(to, from)| {
                if from > to.1 {
                    *to = (angle, from);
                }
            });

        if self.constants.is_ring_data() {
            self.convert_bitmap_to_ids(&output.visibility, angle)?;
        }

        Ok(())
    }

    /// Convert CPU visibilty bitmap to GPU sector ring data.
    fn convert_bitmap_to_ids(&self, bitmap: &[Vec<bool>], angle: u16) -> Result<()> {
        let max_los = self.constants.max_los_as_points;
        let tvs_size = max_los * max_los;
        let sector = usize::from(angle.rem_euclid(super::compute::SECTOR_STEPS));
        let reserved_ring_space = usize::try_from(self.constants.reserved_rings_per_band)?;
        let reserved_per_tvs = reserved_ring_space * usize::try_from(tvs_size)?;
        let angle_ring_data =
            crate::output::ring_data::convert_bitmap_to_ids(bitmap, reserved_ring_space, max_los)?;

        let mut visibility = self.visibility.lock().map_err(|err| eyre!("{err:?}"))?;
        let item = visibility
            .get_mut(sector)
            .context("Couldn't find sector slice to store visibilty")?;

        if angle < 180 {
            // Forward lines of sight
            item.splice(0..reserved_per_tvs, angle_ring_data);
        } else {
            // Backward lines of sight
            item.splice(reserved_per_tvs.., angle_ring_data);
        }
        drop(visibility);

        Ok(())
    }
}
