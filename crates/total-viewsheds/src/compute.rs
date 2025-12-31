//! The main entrypoint for running computations.

use crate::los_pack::LineOfSightPacked;
use color_eyre::Result;
use rayon::iter::IntoParallelIterator as _;
use rayon::iter::ParallelIterator as _;
use rayon::ThreadPoolBuilder;
use std::sync::Mutex;
use std::time::Instant;

/// The number of angles we rotate through. The other half are done via "backwards" lines of sight.
pub const SECTOR_STEPS: u16 = 180;

/// Handles all the computations.
pub struct Compute<'compute> {
    /// User configuration.
    config: ComputeConfig,
    /// Vulkan GPU manager
    vulkan: Option<super::vulkan::Vulkan>,
    /// Storage interface for conputed ring (viewshed) data.
    storage: Option<crate::output::ring_data::Storage>,
    /// The Digital Elevation Model that we're computing.
    dem: &'compute mut crate::dem::DEM,
    /// The constants for each kernel computation.
    pub constants: kernel::constants::Constants,
    /// The amount of reserved memory for ring data.
    total_reserved_rings: usize,
    /// Keeps track of the cumulative surfaces from every angle.
    pub total_surfaces: Vec<f32>,
    /// Keeps track of the ring (viewshed) data.
    pub ring_data: Vec<Vec<u32>>,
    /// Keeps track of the longest lines of sight.
    pub longest_lines: Vec<crate::los_pack::LineOfSightPacked>,
}

/// Configuration for computing.
pub struct ComputeConfig {
    /// The height of the observer that views viewsheds.
    pub observer_height: f32,
    /// The size of each elevation point in meters.
    pub scale: f32,
    /// Where to run the kernel computations
    pub backend: crate::config::Backend,
    /// What to compute.
    pub process: Vec<crate::config::Process>,
    /// Output directory
    pub output_directory: Option<std::path::PathBuf>,
    /// The number of reserved rings per km.
    pub rings_per_km: f32,
    /// How to normalise the heatmap data.
    pub heatmap: crate::config::HeatmapNormalisation,
    /// Refractoin coefficient
    pub refraction: f32,
}

impl<'compute> Compute<'compute> {
    /// Instantiate.
    pub fn new(config: ComputeConfig, dem: &'compute mut crate::dem::DEM) -> Result<Self> {
        let total_bands = dem.computable_points_count * 2;

        let rings_per_band = if Self::is_process_viewsheds(&config.process) {
            Self::ring_count_per_band(config.rings_per_km, dem.max_los_as_points * dem.scale_u32())
        } else {
            1
        };
        let total_reserved_rings = if Self::is_process_viewsheds(&config.process) {
            usize::try_from(total_bands)? * rings_per_band
        } else {
            1
        };

        let storage = if Self::is_process_viewsheds(&config.process) {
            match &config.output_directory {
                Some(output_directory) => {
                    Some(crate::output::ring_data::Storage::new(output_directory)?)
                }
                None => None,
            }
        } else {
            None
        };

        let constants = kernel::constants::Constants {
            total_bands,
            max_los_as_points: dem.max_los_as_points,
            dem_width: dem.width,
            tvs_width: dem.tvs_width,
            observer_height: config.observer_height,
            reserved_rings_per_band: u32::try_from(rings_per_band)?,
            process: Self::bitmask_flags_for_kernel(&config.process),
            scale: config.scale,
            refraction: config.refraction,
            ..Default::default()
        };

        // We only need the "chocolate box" section of rotations to do visibility calculations.
        let rotations_size = kernel::chocolate_box::size(dem.width, dem.tvs_width);

        let vulkan = if matches!(config.backend, crate::config::Backend::Vulkan) {
            let elevations = dem.elevations.clone();
            dem.elevations = Vec::new(); // Free up some RAM.
            Some(super::vulkan::Vulkan::new(
                constants,
                elevations,
                usize::try_from(rotations_size)?,
                total_reserved_rings,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            vulkan,
            storage,
            dem,
            constants,
            total_reserved_rings,
            total_surfaces: Vec::default(),
            ring_data: Vec::default(),
            longest_lines: Vec::default(),
        })
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Accuracy isn't needed, we're just calculating a value to help find minimum RAM usage."
    )]
    /// Calculate the expected number of rings per band of sight.
    pub const fn ring_count_per_band(rings_per_km: f32, max_line_of_sight: u32) -> usize {
        let meters_per_km = 1000.0;
        let band_length_in_km = (max_line_of_sight as f32) / meters_per_km;
        (band_length_in_km * rings_per_km) as usize
    }

    /// Are we computing everything?
    fn is_process_everything(process: &[crate::config::Process]) -> bool {
        process.contains(&crate::config::Process::All)
    }

    /// Are we computing total surface areas?
    pub fn is_process_surfaces(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process)
            || process.contains(&crate::config::Process::TotalSurfaces)
    }

    /// Are we computing viewsheds?
    pub fn is_process_viewsheds(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process) || process.contains(&crate::config::Process::Viewsheds)
    }

    /// Are we computing total surface areas?
    pub fn is_process_longest_lines(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process)
            || process.contains(&crate::config::Process::LongestLines)
    }

    /// Create a GPU-friendly bitmask of flags to use in the kernel.
    pub fn bitmask_flags_for_kernel(processes: &[crate::config::Process]) -> u32 {
        use kernel::constants as kernel;
        let mut flags = 0u32;
        for process in processes {
            match process {
                crate::config::Process::All => {
                    flags |= kernel::Flag::TotalSurfaces.bit() | kernel::Flag::RingData.bit();
                }
                crate::config::Process::TotalSurfaces => {
                    flags |= kernel::Flag::TotalSurfaces.bit();
                }
                crate::config::Process::Viewsheds => flags |= kernel::Flag::RingData.bit(),
                crate::config::Process::LongestLines => {
                    flags |= kernel::Flag::LongestLines.bit();
                }
            }
        }
        flags
    }

    /// Do all computations.
    pub fn run(&mut self) -> Result<()> {
        if matches!(self.config.backend, crate::config::Backend::CPU) {
            self.run_parallel()?;
        } else {
            self.run_sequential()?;
        }

        Ok(())
    }

    /// `run_sequential` runs a sequential GPU or CPU kernel
    fn run_sequential(&mut self) -> Result<()> {
        if Self::is_process_surfaces(&self.config.process) {
            self.total_surfaces = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
        }

        if Self::is_process_viewsheds(&self.config.process)
            && self.config.output_directory.is_some()
        {
            self.save_ring_metadata()?;
        }

        let mut longest_lines = if Self::is_process_longest_lines(&self.config.process) {
            self.longest_lines = vec![
                crate::los_pack::LineOfSightPacked::default();
                usize::try_from(self.dem.computable_points_count)?
            ];
            vec![0.0; usize::try_from(self.dem.computable_points_count)?]
        } else {
            Vec::new()
        };

        for angle in 0..SECTOR_STEPS {
            let mut sector_ring_data = vec![0; self.total_reserved_rings];
            let trig = kernel::rotation::Rotator::calculate_trig(f32::from(angle));
            self.constants.sine = trig.0;
            self.constants.cosine = trig.1;
            self.compute_sector(angle, &mut sector_ring_data, &mut longest_lines)?;

            if Self::is_process_viewsheds(&self.config.process) {
                match &self.config.output_directory {
                    Some(_) => {
                        self.save_sector_ring_data(angle, &sector_ring_data)?;
                    }
                    None => self.ring_data.push(sector_ring_data.clone()),
                }
            }

            if Self::is_process_longest_lines(&self.config.process) {
                self.increment_longest_lines(&longest_lines, angle)?;
                if angle == SECTOR_STEPS - 1 {
                    self.render_longest_lines()?;
                }
            }
        }

        if Self::is_process_surfaces(&self.config.process) {
            self.render_total_surfaces()?;
        }

        Ok(())
    }

    /// `run_parallel` runs the CPU kernel in parallel
    fn run_parallel(&mut self) -> Result<()> {
        let max_los = usize::try_from(self.dem.max_los_as_points)?;
        let mut surfaces = vec![0.0f32; max_los * max_los];
        let mut longest = vec![(0u16, 0.0f32); max_los * max_los];

        let pool = ThreadPoolBuilder::new().num_threads(8).build()?;

        {
            let angle_mu = &Mutex::new(&mut surfaces);
            let longest_mu = &Mutex::new(&mut longest);

            let elevations = &self.dem.elevations;

            pool.install(move || {
                (0u16..360u16)
                    .into_par_iter()
                    .map(|angle| {
                        let start = Instant::now();
                        tracing::info!("starting angle: {angle}");
                        let (heatmap, long, _) =
                            crate::cpu::kernel(elevations, max_los, f32::from(angle), false);
                        tracing::info!("finished angle in {:?}", start.elapsed());
                        (angle, heatmap, long)
                    })
                    .for_each(|(angle, heatmap, long)| {
                        #[expect(clippy::expect_used, reason = "a poisoned mutex should crash")]
                        angle_mu
                            .lock()
                            .expect("mutex poisoned")
                            .iter_mut()
                            .zip(heatmap)
                            .for_each(|(to, from)| {
                                *to += from;
                            });

                        #[expect(clippy::expect_used, reason = "a poisoned mutex should crash")]
                        longest_mu
                            .lock()
                            .expect("mutex poisoned")
                            .iter_mut()
                            .zip(long)
                            .for_each(|(to, from)| {
                                if from > to.1 {
                                    *to = (angle, from);
                                }
                            });
                    });
            });
        };

        self.total_surfaces = surfaces;
        let packed: Result<Vec<LineOfSightPacked>> = longest
            .iter()
            .map(|&(angle, distance): &(u16, f32)| {
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "distances always fit in u32"
                )]
                LineOfSightPacked::new(distance as u32, angle)
            })
            .collect();
        self.longest_lines = packed?;

        self.render_total_surfaces()?;
        self.render_longest_lines()?;

        Ok(())
    }

    /// Add the accumulated total surface areas for the current sector to the running total.
    fn add_sector_surfaces_to_running_total(&mut self, cumulative_surfaces: &[f32]) {
        for (left, right) in self
            .total_surfaces
            .iter_mut()
            .zip(cumulative_surfaces.iter())
        {
            *left += right;
        }
    }

    /// Check to see if this angle increases the current longest line of sight for the point.
    fn increment_longest_lines(&mut self, longest_lines: &[f32], sector: u16) -> Result<()> {
        for (left, right) in self.longest_lines.iter_mut().zip(longest_lines.iter()) {
            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "Distances always fit in u32"
            )]
            let current = right.abs() as u32;
            if current > left.distance() {
                let angle = if *right >= 0.0 { sector } else { sector + 180 };
                let packed = crate::los_pack::LineOfSightPacked::new(current, angle)?;
                *left = packed;
            }
        }

        Ok(())
    }

    /// The metadata needed to reconstruct viewsheds based on the DEM and reserved rings.
    pub fn metadata(&self) -> Result<crate::output::ring_data::MetaData> {
        Ok(crate::output::ring_data::MetaData {
            width: self.dem.width,
            scale: self.dem.scale,
            max_line_of_sight: self.dem.max_los_as_points * self.dem.scale_u32(),
            reserved_ring_size: usize::try_from(self.constants.reserved_rings_per_band)?,
            centre: self.dem.centre,
        })
    }

    /// Save band deltas to cache.
    pub fn save_sector_ring_data(&self, sector: u16, ring_data: &[u32]) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            color_eyre::eyre::bail!("Tried to save sector ring data without any active storage.");
        };

        storage.save_sector(sector, ring_data)?;
        Ok(())
    }

    /// Save the metadata for the ring data (aka viewsheds).
    pub fn save_ring_metadata(&self) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            color_eyre::eyre::bail!("Tried to save ring metadata without any active storage.");
        };

        storage.save_metadata(&self.metadata()?)?;
        Ok(())
    }

    /// Render a heatmap and `.bt` file of the total surface areas for each point within the computable area of the
    /// DEM.
    fn render_total_surfaces(&self) -> Result<()> {
        let Some(output_dir) = &self.config.output_directory else {
            return Ok(());
        };

        crate::output::png::save(
            &self.total_surfaces,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("total_surfaces.png"),
            self.config.heatmap,
        )?;

        crate::output::bt::save(
            self.dem,
            &self.total_surfaces,
            &output_dir.join("total_surfaces.bt"),
        )?;

        Ok(())
    }

    /// Render a heatmap and `.bt` of the longest lines of sight for each point within the computable area of the
    /// DEM.
    fn render_longest_lines(&self) -> Result<()> {
        let Some(output_dir) = &self.config.output_directory else {
            return Ok(());
        };

        let distances = self
            .longest_lines
            .iter()
            .map(|los| {
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "Distances always fit in u32"
                )]
                {
                    los.distance() as f32
                }
            })
            .collect::<Vec<_>>();
        crate::output::png::save(
            &distances,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("longest_lines.png"),
            self.config.heatmap,
        )?;

        let packed_lines = self
            .longest_lines
            .iter()
            .map(crate::los_pack::LineOfSightPacked::as_f32)
            .collect::<Vec<_>>();
        crate::output::bt::save(
            self.dem,
            &packed_lines,
            &output_dir.join("longest_lines.bt"),
        )?;

        Ok(())
    }

    /// Compute a single sector.
    fn compute_sector(
        &mut self,
        angle: u16,
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        tracing::info!("Running kernel for {angle}°");
        match self.config.backend {
            crate::config::Backend::VulkanCPU => {
                self.compute_sector_cpu(ring_data, longest_lines)?;
            }
            crate::config::Backend::Vulkan => {
                self.compute_sector_vulkan(ring_data, longest_lines)?;
            }
            #[expect(clippy::unimplemented, reason = "CPU kernel is only multithreaded")]
            crate::config::Backend::CPU => {
                unimplemented!();
            }

            #[expect(clippy::unimplemented, reason = "Coming Soon!")]
            crate::config::Backend::Cuda => unimplemented!(),
        }

        Ok(())
    }

    /// Do a whole sector calculation on the GPU using Vulkan.
    fn compute_sector_vulkan(
        &mut self,
        rings: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        let Some(gpu) = self.vulkan.as_mut() else {
            color_eyre::eyre::bail!("`self.gpu` not instantiated yet.");
        };

        let (surfaces_data, rings_data, longest_lines_data) = gpu.run(self.constants)?;
        if Self::is_process_surfaces(&self.config.process) {
            self.total_surfaces
                .copy_from_slice(surfaces_data.as_slice());
        }
        if Self::is_process_viewsheds(&self.config.process) {
            rings.copy_from_slice(rings_data.as_slice());
        }
        if Self::is_process_longest_lines(&self.config.process) {
            longest_lines.copy_from_slice(longest_lines_data.as_slice());
        }
        Ok(())
    }

    /// Do a whole sector calculation on the CPU.
    fn compute_sector_cpu(
        &mut self,
        ring_data: &mut [u32],
        longest_lines: &mut [f32],
    ) -> Result<()> {
        let chocolate_box_size = kernel::chocolate_box::size(self.dem.width, self.dem.tvs_width);
        let mut rotated_elevations = vec![0.0; usize::try_from(chocolate_box_size)?];
        for chocolate_id in 0..(self.dem.computable_points_count * 2) {
            let chocolate = kernel::chocolate_box::Rotator::new_from_cached_trig(
                chocolate_id,
                self.dem.width,
                self.dem.tvs_width,
                self.constants.sine,
                self.constants.cosine,
            );
            // Note that we _anti_ rotate because anti-rotating the DEM grid has the effect of normally
            // rotating the line of sight. Which is just more intuitive to work with when debugging.
            chocolate
                .anti_rotate_value_nearest_neighbour(&self.dem.elevations, &mut rotated_elevations);
        }

        let mut buffers = kernel::kernel::Buffers {
            constants: &self.constants,
            elevations: &rotated_elevations,
            cumulative_surfaces: &mut self.total_surfaces,
            longest_lines,
            ring_data,
        };

        for tvs_id in 0..self.constants.total_bands {
            kernel::kernel::Kernel::run(tvs_id, &mut buffers);
        }

        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use googletest::prelude::*;

    pub fn make_dem(elevations: &[i16]) -> crate::dem::DEM {
        let width = elevations.len().isqrt() as u32;
        let mut dem = crate::dem::DEM::new(
            crate::projection::LatLonCoord((33.33, 33.33).into()),
            width,
            1.0,
            width / 3,
        )
        .unwrap();
        dem.elevations = elevations.into();
        dem
    }

    pub fn compute(dem: &mut crate::dem::DEM) -> Compute<'_> {
        let config = ComputeConfig {
            observer_height: 0.8,
            scale: 1.0,
            backend: crate::config::Backend::VulkanCPU,
            process: vec![
                crate::config::Process::TotalSurfaces,
                crate::config::Process::Viewsheds,
                crate::config::Process::LongestLines,
            ],
            output_directory: None,
            rings_per_km: 5000.0,
            heatmap: crate::config::HeatmapNormalisation::UnitScale,
            refraction: 0.13,
        };

        let mut compute = Compute::new(config, dem).unwrap();
        compute.run().unwrap();
        compute
    }

    #[test]
    fn total_surfaces() {
        let mut dem = make_dem(&kernel::tests::dems::bigger_dem());
        let compute = compute(&mut dem);
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                0.0, 0.0,      0.0,       0.0,
                0.0, 6.283163, 38.920944, 0.0,
                0.0, 70.75571, 94.24808,  0.0,
                0.0, 0.0,      0.0,       0.0
            ]
        );
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Distances always fit in u32"
    )]
    #[gtest]
    fn longest_lines() {
        let mut dem = make_dem(&kernel::tests::dems::bigger_dem());
        let compute = compute(&mut dem);

        #[rustfmt::skip]
        expect_eq!(
            compute.longest_lines.iter()
            .map(|los| los.distance() as f32)
            .collect::<Vec<_>>(),
            [
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 5.0, 0.0,
                0.0, 5.0, 5.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ]
        );

        #[rustfmt::skip]
        expect_eq!(
            compute.longest_lines.iter()
            .map(|los| los.angle().unwrap())
            .collect::<Vec<_>>(),
            [
                0, 0,   0,   0,
                0, 0,   0,   0,
                0, 180, 0,   0,
                0, 0,   0,   0
            ]
        );
    }
}
