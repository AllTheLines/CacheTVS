//! The main entrypoint for running computations.

use color_eyre::Result;

/// The number of angles we rotate through. The other half are done via "backwards" lines of sight.
pub const SECTOR_STEPS: u16 = 180;

/// Handles all the computations.
pub struct Compute<'compute> {
    /// User configuration.
    pub config: Config,
    /// Vulkan GPU manager
    pub vulkan: Option<crate::vulkan::Vulkan>,
    /// Storage interface for conputed ring (viewshed) data.
    storage: Option<crate::output::ring_data::Storage>,
    /// The Digital Elevation Model that we're computing.
    pub dem: &'compute mut crate::dem::DEM,
    /// The constants for each kernel computation.
    pub constants: kernel::constants::Constants,
    /// The amount of reserved memory for ring data.
    pub total_reserved_rings: usize,
    /// Keeps track of the cumulative surfaces from every angle.
    pub total_surfaces: Vec<f32>,
    /// Keeps track of the ring (viewshed) data.
    pub ring_data: Vec<Vec<u32>>,
    /// Keeps track of the longest lines of sight.
    pub longest_lines: Vec<crate::los_pack::LineOfSightPacked>,
}

/// Configuration for computing.
pub struct Config {
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
    pub fn new(config: Config, dem: &'compute mut crate::dem::DEM) -> Result<Self> {
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
            Some(crate::vulkan::Vulkan::new(
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

    /// Do all computations.
    pub fn run(&mut self) -> Result<()> {
        if Self::is_process_viewsheds(&self.config.process)
            && self.config.output_directory.is_some()
        {
            self.save_ring_metadata()?;
        }

        if matches!(self.config.backend, crate::config::Backend::CPU) {
            self.run_parallel()?;
        } else {
            self.run_sequential()?;
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
    pub fn render_total_surfaces(&self) -> Result<()> {
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
    pub fn render_longest_lines(&self) -> Result<()> {
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

    pub fn compute(dem: &mut crate::dem::DEM, backend: crate::config::Backend) -> Compute<'_> {
        let config = Config {
            observer_height: 0.8,
            scale: 1.0,
            backend,
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

    fn total_surfaces(backend: crate::config::Backend) {
        let mut dem = make_dem(&kernel::tests::dems::bigger_dem());
        let compute = compute(&mut dem, backend);
        #[rustfmt::skip]
        assert_eq!(
            compute.total_surfaces,
            [
                0.0, 0.0,      0.0,       0.0,
                0.0, 568.6271, 3996.5193, 0.0,
                0.0, 6310.845, 8529.429,  0.0,
                0.0, 0.0,      0.0,       0.0
            ]
        );
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Distances always fit in u32"
    )]
    fn longest_lines(backend: crate::config::Backend) {
        let mut dem = make_dem(&kernel::tests::dems::bigger_dem());
        let compute = compute(&mut dem, backend);

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

    mod gpu {
        use googletest::prelude::*;

        #[test]
        fn total_surfaces() {
            super::total_surfaces(crate::config::Backend::VulkanCPU);
        }

        #[gtest]
        fn longest_lines() {
            super::longest_lines(crate::config::Backend::VulkanCPU);
        }
    }

    mod cpu {
        use googletest::prelude::*;

        #[test]
        #[ignore = "TODO@ryan: Enable once viewshed tests are settled"]
        fn total_surfaces() {
            super::total_surfaces(crate::config::Backend::CPU);
        }

        #[gtest]
        #[ignore = "TODO@ryan: Enable once viewshed tests are settled"]
        fn longest_lines() {
            super::longest_lines(crate::config::Backend::CPU);
        }

        #[gtest]
        #[ignore = "TODO@ryan: Enable once you've added refraction"]
        fn refraction_affects_visibility() {
            // Set your refraction constant to this so that the effect is so dramatic that it shows
            // up in our tiny test DEMS.
            // let refraction = -kernel::kernel::EARTH_DIAMETER;
        }
    }
}
