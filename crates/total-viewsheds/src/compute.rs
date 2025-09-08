//! The main entrypoint for running computations.

use color_eyre::{eyre::Ok, Result};

/// Handles all the computations.
pub struct Compute<'compute> {
    /// Where to run the kernel computations
    backend: crate::config::Backend,
    /// What to compute.
    process: Vec<crate::config::Process>,
    /// Vulkan GPU manager
    vulkan: Option<super::vulkan::Vulkan>,
    /// The OS's state directory for saving our cache into.
    state_directory: Option<std::path::PathBuf>,
    /// Output directory
    output_directory: Option<std::path::PathBuf>,
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
}

impl<'compute> Compute<'compute> {
    /// Instantiate.
    pub fn new(
        backend: crate::config::Backend,
        process: Vec<crate::config::Process>,
        state_directory: Option<std::path::PathBuf>,
        maybe_output_directory: Option<std::path::PathBuf>,
        dem: &'compute mut crate::dem::DEM,
        rings_per_km: f32,
        observer_height: f32,
    ) -> Result<Self> {
        let total_bands = dem.computable_points_count * 2;

        let rings_per_band = if Self::is_process_viewsheds(&process) {
            Self::ring_count_per_band(rings_per_km, dem.max_line_of_sight)
        } else {
            1
        };
        let total_reserved_rings = if Self::is_process_viewsheds(&process) {
            usize::try_from(total_bands)? * rings_per_band
        } else {
            1
        };

        let storage = if Self::is_process_viewsheds(&process) {
            match &maybe_output_directory {
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
            observer_height,
            reserved_rings_per_band: u32::try_from(rings_per_band)?,
            process: Self::bitmask_flags_for_kernel(&process),
            ..Default::default()
        };

        #[expect(
            clippy::if_then_some_else_none,
            reason = "The `?` is hard to use in the closure"
        )]
        let vulkan = if matches!(backend, crate::config::Backend::Vulkan) {
            let elevations = dem.elevations.clone();
            dem.elevations = Vec::new(); // Free up some RAM.
            Some(super::vulkan::Vulkan::new(
                constants,
                elevations,
                usize::try_from(dem.size)?,
                usize::try_from(dem.band_deltas_size())?,
                total_reserved_rings,
            )?)
        } else {
            None
        };

        Ok(Self {
            backend,
            process,
            vulkan,
            state_directory,
            output_directory: maybe_output_directory,
            storage,
            dem,
            constants,
            total_reserved_rings,
            total_surfaces: Vec::default(),
            ring_data: Vec::default(),
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
            }
        }
        flags
    }

    /// Do all computations.
    pub fn run(&mut self) -> Result<()> {
        let mut sector_surfaces = if Self::is_process_surfaces(&self.process) {
            let blank = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
            self.total_surfaces.clone_from(&blank);
            blank
        } else {
            Vec::new()
        };

        if Self::is_process_viewsheds(&self.process) && self.output_directory.is_some() {
            self.save_ring_metadata()?;
        }

        for angle in 0..crate::axes::SECTOR_STEPS {
            self.load_or_compute_cache(angle)?;
            let mut sector_ring_data = vec![0; self.total_reserved_rings];
            self.compute_sector(angle, &mut sector_surfaces, &mut sector_ring_data)?;

            if Self::is_process_viewsheds(&self.process) {
                match &self.output_directory {
                    Some(_) => {
                        self.save_sector_ring_data(angle, &sector_ring_data)?;
                    }
                    None => self.ring_data.push(sector_ring_data.clone()),
                }
            }

            if Self::is_process_surfaces(&self.process) {
                self.add_sector_surfaces_to_running_total(&sector_surfaces);
                self.render_total_surfaces()?;
            }
        }

        Ok(())
    }

    /// Either load cache from the filesystem or create and save it.
    fn load_or_compute_cache(&mut self, angle: u16) -> Result<()> {
        let maybe_cache = if let Some(state_directory) = self.state_directory.clone() {
            Some(crate::cache::Cache::new(
                &state_directory,
                self.dem.width,
                angle,
            ))
        } else {
            None
        };

        if let Some(cache) = &maybe_cache {
            cache.ensure_directories_exists()?;

            if cache.is_cache_exists {
                tracing::debug!(
                    "Loading cache from: {}/*/{}",
                    cache.base_directory.display(),
                    angle
                );
                self.dem.band_deltas = cache.load_band_deltas()?;
                self.dem.band_distances = cache.load_distances()?;
                return Ok(());
            }

            tracing::warn!(
                "Cached data not found at: {}/*/{}. So computing now...",
                cache.base_directory.display(),
                angle
            );
        } else {
            tracing::warn!("Forcing computation of cache for angle {angle}°...");
        }

        self.dem.calculate_axes(f32::from(angle))?;
        self.dem.compile_band_data()?;

        if let Some(cache) = &maybe_cache {
            cache.save_band_deltas(&self.dem.band_deltas)?;
            cache.save_distances(&self.dem.band_distances)?;
        }

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

    /// The metadata needed to reconstruct viewsheds based on the DEM and reserved rings.
    pub fn metadata(&self) -> Result<crate::output::ring_data::MetaData> {
        Ok(crate::output::ring_data::MetaData {
            width: self.dem.width,
            scale: self.dem.scale,
            max_line_of_sight: self.dem.max_line_of_sight,
            reserved_ring_size: usize::try_from(self.constants.reserved_rings_per_band)?,
            sector_shift: crate::axes::SECTOR_SHIFT,
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

    /// Render a heatmap of the total surface areas of each point within the computable area of the
    /// DEM.
    fn render_total_surfaces(&self) -> Result<()> {
        let Some(output_dir) = &self.output_directory else {
            return Ok(());
        };

        crate::output::png::save(
            &self.total_surfaces,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("heatmap.png"),
        )?;

        Ok(())
    }

    /// Compute a single sector.
    fn compute_sector(
        &mut self,
        angle: u16,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
    ) -> Result<()> {
        tracing::info!("Running kernel for {angle}°");
        match self.backend {
            crate::config::Backend::CPU => {
                self.compute_sector_cpu(cumulative_surfaces, ring_data);
            }
            crate::config::Backend::Vulkan => {
                self.compute_sector_vulkan(cumulative_surfaces, ring_data)?;
            }

            #[expect(clippy::unimplemented, reason = "Coming Soon!")]
            crate::config::Backend::Cuda => unimplemented!(),
        }

        Ok(())
    }

    /// Do a whole sector calculation on the GPU using Vulkan.
    fn compute_sector_vulkan(
        &mut self,
        cumulative_surfaces: &mut [f32],
        ring_data: &mut [u32],
    ) -> Result<()> {
        let Some(gpu) = self.vulkan.as_mut() else {
            color_eyre::eyre::bail!("`self.gpu` not instantiated yet.");
        };

        let (surfaces, rings) = gpu.run(&self.dem.band_distances, &self.dem.band_deltas)?;
        cumulative_surfaces.copy_from_slice(surfaces.as_slice());
        ring_data.copy_from_slice(rings.as_slice());
        Ok(())
    }

    /// Do a whole sector calculation on the CPU.
    fn compute_sector_cpu(&self, cumulative_surfaces: &mut [f32], ring_data: &mut [u32]) {
        for kernel_id in 0..self.constants.total_bands {
            kernel::kernel::kernel(
                kernel_id,
                &self.constants,
                &self.dem.elevations,
                &self.dem.band_distances,
                &self.dem.band_deltas,
                cumulative_surfaces,
                ring_data,
            );
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    pub fn make_dem() -> crate::dem::DEM {
        crate::dem::DEM::new(
            crate::projection::LatLonCoord((33.33, 33.33).into()),
            9,
            1.0,
            3,
        )
        .unwrap()
    }

    pub fn compute<'dem>(dem: &'dem mut crate::dem::DEM, elevations: &[i16]) -> Compute<'dem> {
        dem.elevations = elevations.iter().map(|&x| f32::from(x)).collect();
        let mut compute = Compute::new(
            crate::config::Backend::CPU,
            vec![
                crate::config::Process::TotalSurfaces,
                crate::config::Process::Viewsheds,
            ],
            None,
            None,
            dem,
            5000.0,
            1.8,
        )
        .unwrap();
        compute.run().unwrap();
        compute
    }

    fn create_viewshed(elevations: &[i16], pov_id: u32) -> Vec<String> {
        let mut dem = make_dem();
        let compute = compute(&mut dem, elevations);
        let ring_data = compute.ring_data;
        crate::output::ascii::OutputASCII::convert(
            &dem,
            pov_id,
            &ring_data,
            crate::compute::Compute::ring_count_per_band(5000.0, 3),
        )
        .unwrap()
    }

    #[rustfmt::skip]
    pub fn single_peak_dem() -> Vec<i16> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 3, 6, 6, 6, 3, 1, 0,
            0, 1, 3, 6, 9, 6, 3, 1, 0,
            0, 1, 3, 6, 6, 6, 3, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 1, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    }

    #[rustfmt::skip]
    pub fn double_peak_dem() -> Vec<i16> {
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 1, 3, 3, 3, 3, 3, 3, 4,
            0, 1, 3, 4, 4, 4, 4, 4, 3,
            0, 1, 3, 4, 6, 4, 4, 4, 3,
            0, 1, 3, 4, 4, 4, 5, 5, 3,
            0, 1, 3, 4, 4, 5, 9, 5, 3,
            0, 1, 1, 4, 4, 5, 5, 5, 3,
            0, 0, 4, 1, 3, 3, 3, 3, 3
        ]
    }

    #[test]
    fn single_peak_totals() {
        let mut dem = make_dem();
        let compute = compute(&mut dem, &single_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                2543.4292, 1649.654, 2696.0396,
                1641.8002, 3197.66, 1641.8002,
                2696.0396, 1649.6539, 2543.4292
            ]
        );
    }

    #[test]
    fn double_peak_totals() {
        let mut dem = make_dem();
        let compute = compute(&mut dem, &double_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                2687.689, 2546.9956, 2622.3494,
                2564.7678, 3231.647, 2239.714, 
                2604.2551, 2186.5012, 1768.3433
            ]
        );
    }

    // Here we use a simple ASCII representation of the opening and closing of ring
    // sectors:
    //   A '.' can be either inside or outisde a ring sector.
    //   A '+' is the opening and a '-' is the closing.
    //   A '±' represents both an opening and closing (from different sectors).
    //
    // Eg; Given this ASCII art profile of 2 mountain peaks, where the observer is
    // 'X':
    //
    //                     .*`*.
    //                  .*`  |  `*.
    //       .      X.*`     |     `*.
    //    .*`|`*. .*`        |        `*.
    // .*`   |   `  |        |           `*.
    //       |      |        |
    // there would be 2 ring sectors, both opening at the same point but looking
    // in different directions:
    //       |      |        |
    // ......-......+........-..............
    //
    // Or to use 0s and 1s to show the surfaces seen by the observer:
    //
    // 0000001111111111111111100000000000000
    mod viewsheds {
        #[test]
        fn summit() {
            let viewshed = super::create_viewshed(&super::single_peak_dem(), 40);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    ". . . . . . . . .",
                    ". ± ± ± ± ± ± ± .",
                    ". ± ± ± . ± ± ± .",
                    ". ± ± . . . ± ± .",
                    ". ± . . o . . ± .",
                    ". ± ± . . . ± ± .",
                    ". ± ± ± . ± ± ± .",
                    ". ± ± ± ± ± ± ± .",
                    ". . . . . . . . ."]
            );
        }

        #[test]
        fn off_summit() {
            let viewshed = super::create_viewshed(&super::single_peak_dem(), 30);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    "± ± ± ± ± ± ± . .",
                    "± ± ± . ± ± ± . .",
                    "± ± . . ± ± ± . .",
                    "± . . o . . ± . .",
                    "± ± ± . . ± ± . .",
                    "± ± ± . ± ± . . .",
                    "± ± ± ± ± . . . .",
                    ". . . . . . . . .",
                    ". . . . . . . . ."
                ]
            );
        }

        #[test]
        fn double_peak() {
            let viewshed = super::create_viewshed(&super::double_peak_dem(), 30);
            #[rustfmt::skip]
            assert_eq!(
                viewshed,
                [
                    "± ± ± ± ± ± ± . .",
                    "± ± ± . ± ± ± . .",
                    "± ± . . ± ± ± . .",
                    "± . . o . . ± . .",
                    "± ± ± . . ± ± . .",
                    "± ± ± . ± ± . . .",
                    "± ± ± ± ± . ± . .",
                    ". . . . . . . . .",
                    ". . . . . . . . ."
                ]
            );
        }
    }
}
