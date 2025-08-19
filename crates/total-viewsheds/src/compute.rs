//! The main entrypoint for running computations.

use color_eyre::Result;

/// Handles all the computations.
pub struct Compute<'compute> {
    /// Where to run the kernel computations
    method: crate::config::ComputeType,
    /// The GPU manager
    gpu: Option<super::gpu::GPU>,
    /// The OS's state directory for saving our cache into.
    state_directory: Option<std::path::PathBuf>,
    /// Output directory
    output_dir: Option<std::path::PathBuf>,
    /// The Digital Elevation Model that we're computing.
    dem: &'compute mut crate::dem::DEM,
    /// The constants for each kernel computation.
    pub constants: total_viewsheds_kernel::kernel::Constants,
    /// The amount of reserved memory for ring data.
    total_reserved_rings: usize,
    /// Keeps track of the cumulative surfaces from every angle.
    pub total_surfaces: Vec<f32>,
}

impl<'compute> Compute<'compute> {
    /// Instantiate.
    pub fn new(
        method: crate::config::ComputeType,
        state_directory: Option<std::path::PathBuf>,
        output_dir: Option<std::path::PathBuf>,
        dem: &'compute mut crate::dem::DEM,
        rings_per_km: f32,
    ) -> Result<Self> {
        let total_bands = dem.computable_points_count * 2;
        let rings_per_band = Self::ring_count_per_band(rings_per_km, dem.max_line_of_sight);
        let constants = total_viewsheds_kernel::kernel::Constants {
            total_bands,
            max_los_as_points: dem.max_los_as_points,
            dem_width: dem.width,
            tvs_width: dem.tvs_width,
            observer_height: 1.8,
            reserved_rings_per_band: u32::try_from(rings_per_band)?,
            ..Default::default()
        };

        #[cfg(test)]
        let total_reserved_rings = usize::try_from(total_bands)? * rings_per_band;
        #[expect(
            clippy::cfg_not_test,
            reason = "
              No-op for now, it can be incredibly memory intensive, so we need to be explicit about
              how we handle it.
            "
        )]
        #[cfg(not(test))]
        let total_reserved_rings = 1;

        #[expect(
            clippy::if_then_some_else_none,
            reason = "The `?` is hard to use in the closure"
        )]
        let gpu = if matches!(method, crate::config::ComputeType::Vulkan) {
            let elevations = dem.elevations.clone();
            dem.elevations = Vec::new(); // Free up some RAM.
            Some(super::gpu::GPU::new(
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
            method,
            gpu,
            state_directory,
            output_dir,
            dem,
            constants,
            total_reserved_rings,
            total_surfaces: Vec::default(),
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
    fn ring_count_per_band(rings_per_km: f32, max_line_of_sight: u32) -> usize {
        let meters_per_km = 1000.0;
        let band_length_in_km = (max_line_of_sight as f32) / meters_per_km;
        (band_length_in_km * rings_per_km) as usize
    }

    /// Do alld computations.
    pub fn run(&mut self) -> Result<(Vec<f32>, Vec<Vec<u32>>)> {
        let mut cumulative_surfaces = vec![0.0; usize::try_from(self.dem.computable_points_count)?];
        self.total_surfaces.clone_from(&cumulative_surfaces);
        let mut all_ring_data = Vec::new();

        for angle in 0..crate::axes::SECTOR_STEPS {
            self.load_or_compute_cache(angle)?;
            let mut sector_ring_data = vec![0; self.total_reserved_rings];
            self.compute_sector(angle, &mut cumulative_surfaces, &mut sector_ring_data)?;
            all_ring_data.push(sector_ring_data.clone());
            self.render_total_surfaces()?;
        }

        Ok((cumulative_surfaces, all_ring_data))
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

    /// Render a heatmap of the total surface areas of each point within the computable area of the
    /// DEM.
    fn render_total_surfaces(&self) -> Result<()> {
        let Some(output_dir) = &self.output_dir else {
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
        match self.method {
            crate::config::ComputeType::CPU => {
                self.compute_sector_cpu(cumulative_surfaces, ring_data);
            }
            crate::config::ComputeType::Vulkan => {
                self.compute_sector_vulkan(cumulative_surfaces)?;
            }

            #[expect(clippy::unimplemented, reason = "Coming Soon!")]
            crate::config::ComputeType::Cuda => unimplemented!(),
        }

        self.add_sector_surfaces_to_running_total(cumulative_surfaces);

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

    /// Do a whole sector calculation on the GPU using Vulkan.
    fn compute_sector_vulkan(&mut self, cumulative_surfaces: &mut [f32]) -> Result<()> {
        let Some(gpu) = self.gpu.as_mut() else {
            color_eyre::eyre::bail!("`self.gpu` not instantiated yet.");
        };

        let result = gpu.run(&self.dem.band_distances, &self.dem.band_deltas)?;
        cumulative_surfaces.copy_from_slice(result.as_slice());
        Ok(())
    }

    /// Do a whole sector calculation on the CPU.
    fn compute_sector_cpu(&self, cumulative_surfaces: &mut [f32], ring_data: &mut [u32]) {
        for kernel_id in 0..self.constants.total_bands {
            total_viewsheds_kernel::kernel::kernel(
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

#[expect(clippy::unreadable_literal, reason = "It's just for the tests")]
#[cfg(test)]
mod test {
    use super::*;

    fn make_dem() -> crate::dem::DEM {
        crate::dem::DEM::new(9, 1.0, 3).unwrap()
    }

    fn compute_tvs(dem: &mut crate::dem::DEM, elevations: &[i16]) -> (Vec<f32>, Vec<Vec<u32>>) {
        dem.elevations = elevations.iter().map(|&x| f32::from(x)).collect();
        let mut compute =
            Compute::new(crate::config::ComputeType::CPU, None, None, dem, 5000.0).unwrap();
        compute.run().unwrap()
    }

    fn create_tvs(elevations: &[i16]) -> (Vec<f32>, Vec<Vec<u32>>) {
        let mut dem = make_dem();
        compute_tvs(&mut dem, elevations)
    }

    fn create_viewshed(elevations: &[i16], pov_id: u32) -> Vec<String> {
        let mut dem = make_dem();
        let (_, ring_data) = compute_tvs(&mut dem, elevations);
        crate::output::ascii::OutputASCII::convert(
            &dem,
            pov_id,
            &ring_data,
            crate::compute::Compute::ring_count_per_band(5000.0, 3),
        )
        .unwrap()
    }

    #[rustfmt::skip]
    fn single_peak_dem() -> Vec<i16> {
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
    fn double_peak_dem() -> Vec<i16> {
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
        let (cumulative_surfaces, _) = create_tvs(&single_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            cumulative_surfaces,
            [
                28.95109,  18.207317, 28.951097,
                18.207323, 35.32013,  18.207323,
                28.951097, 18.207315, 28.95109
            ]
        );
    }

    #[test]
    fn double_peak_totals() {
        let (cumulative_surfaces, _) = create_tvs(&double_peak_dem());
        #[rustfmt::skip]
        assert_eq!(
            cumulative_surfaces,
            [
                30.305561, 27.532038, 27.445093,
                27.366535, 35.86692,  24.969402,
                27.101337, 24.085308, 22.368181
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
