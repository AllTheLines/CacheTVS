//! The main entrypoint for running computations.

use color_eyre::Result;
use std::path::PathBuf;

/// Handles all the computations.
pub(crate) struct Compute<'compute>
where
    'static: 'compute,
{
    /// User configuration.
    pub config: Config,
    /// The Digital Elevation Model that we're computing.
    pub dem: &'compute mut tvs_lib::dem::DEM,
    /// Keeps track of the cumulative surfaces from every angle.
    pub total_surfaces: Vec<f32>,
    /// Keeps track of the longest lines of sight.
    pub longest_lines: Vec<crate::los_pack::LineOfSightPacked>,
}

#[derive(Clone)]
/// Configuration for computing.
pub(crate) struct Config {
    /// The height of the observer that views viewsheds.
    pub observer_height: f32,
    /// What to compute.
    pub process: Vec<crate::config::Process>,
    /// Output directory
    pub output_directory: Option<std::path::PathBuf>,
    /// How to normalise the heatmap data.
    pub heatmap: crate::config::HeatmapNormalisation,
    /// Refraction coefficient
    pub refraction: f32,
    /// Number of threads for computation
    pub thread_count: usize,
    /// Disables the rendering of PNG images (good for long runs)
    pub disable_render_image: bool,
    /// Where to store the viewshed
    pub viewsheds_db_path: PathBuf,
    /// Metadata about the DEM and compute run.
    pub dem_metadata: tvs_lib::metadata::MetaData,
    /// Polygon for Area of Interest
    pub area_of_interest: geo::Polygon,
    /// Should a database aggregate per thread?
    pub database_per_thread: bool,
    /// DEM IDs to save viewsheds for.
    pub viewsheds_to_save: Option<std::collections::HashMap<i64, i64>>,
    /// Subdivides 360 degrees into a `angle_subdivisions` number of subdivisions
    pub angle_subdivisions: u8,
}

impl<'compute> Compute<'compute> {
    /// Instantiate.
    pub(crate) fn new(config: Config, dem: &'compute mut tvs_lib::dem::DEM) -> Result<Self> {
        if Self::is_process_viewsheds(&config.process) && !cfg!(any(test, feature = "ring_data")) {
            color_eyre::eyre::bail!(
                "Viewshed storage is only supported with the ring_data feature, \
                please recompile with --features=ring_data"
            );
        }

        Ok(Self {
            config,
            dem,
            total_surfaces: Vec::default(),
            longest_lines: Vec::default(),
        })
    }

    /// Are we computing everything?
    fn is_process_everything(process: &[crate::config::Process]) -> bool {
        process.contains(&crate::config::Process::All)
    }

    /// Are we computing viewsheds?
    pub(crate) fn is_process_viewsheds(process: &[crate::config::Process]) -> bool {
        Self::is_process_everything(process) || process.contains(&crate::config::Process::Viewsheds)
    }

    /// Render a heatmap and `.tiff` file of the total surface areas for each point within the computable area of the
    /// DEM.
    pub(crate) fn render_total_surfaces(&self) -> Result<()> {
        let Some(output_dir) = &self.config.output_directory else {
            return Ok(());
        };

        crate::output::tiff::save(
            self.dem,
            &self.total_surfaces,
            &output_dir.join("total_surfaces.tiff"),
        )?;

        if self.config.disable_render_image {
            return Ok(());
        }

        crate::output::png::save(
            &self.total_surfaces,
            self.dem.tvs_width,
            self.dem.tvs_width,
            output_dir.join("total_surfaces.png"),
            self.config.heatmap,
        )?;

        Ok(())
    }

    /// Render a heatmap and `.tiff` of the longest lines of sight for each point within the computable area of the
    /// DEM.
    pub(crate) fn render_longest_lines(&self) -> Result<()> {
        let Some(output_dir) = &self.config.output_directory else {
            return Ok(());
        };

        let packed_lines = self
            .longest_lines
            .iter()
            .map(|&packed| packed.as_f32())
            .collect::<Vec<_>>();

        crate::output::tiff::save(
            self.dem,
            &packed_lines,
            &output_dir.join("longest_lines.tiff"),
        )?;

        if self.config.disable_render_image {
            return Ok(());
        }

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

        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use googletest::prelude::*;

    pub(crate) fn make_dem(elevations: &[i16]) -> tvs_lib::dem::DEM {
        let width = elevations.len().isqrt() as u32;
        let mut dem = tvs_lib::dem::DEM::new(
            tvs_lib::projector::LonLatCoord((33.33, 33.33).into()),
            width,
            1.0,
            width / 3,
        )
        .unwrap();
        dem.elevations = elevations.into();
        dem
    }

    pub(crate) fn default_metadata() -> tvs_lib::metadata::MetaData {
        tvs_lib::metadata::MetaData {
            scale: 1.0,
            ..Default::default()
        }
    }

    pub(crate) fn big_dem_metadata() -> tvs_lib::metadata::MetaData {
        tvs_lib::metadata::MetaData {
            width: 12,
            max_line_of_sight: 4,
            centre: tvs_lib::projector::LonLatCoord((33.33, 33.33).into()),
            ..crate::run::test::default_metadata()
        }
    }

    pub(crate) fn default_config(temp_db: &tempfile::NamedTempFile) -> Config {
        Config {
            observer_height: 0.8,
            process: vec![crate::config::Process::Viewsheds],
            output_directory: None,
            heatmap: crate::config::HeatmapNormalisation::UnitScale,
            refraction: 0.13f32,
            thread_count: 1, // single thread it for consistency
            disable_render_image: false,
            viewsheds_db_path: temp_db.path().into(),
            dem_metadata: default_metadata(),
            area_of_interest: geo::Polygon::empty(),
            database_per_thread: false,
            viewsheds_to_save: None,
            angle_subdivisions: 1,
        }
    }

    pub(crate) fn compute(dem: &mut tvs_lib::dem::DEM, config: Config) -> Compute<'_> {
        let mut compute = Compute::new(config, dem).unwrap();
        compute.run().unwrap();
        compute
    }

    #[rustfmt::skip]
    const EXPECTED_SURFACES: [f32; 16] = [
         0.0, 0.0,       0.0,      0.0,
         0.0, 6.283163,  21.118624,0.0,
         0.0, 47.909348, 62.832096,0.0,
         0.0, 0.0,       0.0,      0.0,
    ];

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Distances always fit in u32"
    )]
    #[gtest]
    fn longest_lines() {
        let mut dem = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let config = Config {
            dem_metadata: big_dem_metadata(),
            ..crate::run::test::default_config(&temp_db)
        };
        let compute = compute(&mut dem, config);

        #[rustfmt::skip]
        expect_eq!(
            compute.longest_lines.iter()
            .map(|los| los.distance() as f32)
            .collect::<Vec<_>>(),
            [
                0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 4.0, 0.0,
                0.0, 4.0, 4.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ]
        );

        #[rustfmt::skip]
        let angles = [
            0,  0,  0,  0,
            0,  0,  29, 0,
            0,  1,  0,  0,
            0,  0,  0,  0
        ];

        expect_eq!(
            compute
                .longest_lines
                .iter()
                .map(|packed| packed.angle())
                .collect::<Vec<_>>(),
            angles
        );
    }

    #[test]
    fn total_surfaces() {
        let mut dem = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let config = super::Config {
            dem_metadata: crate::run::test::big_dem_metadata(),
            ..default_config(&temp_db)
        };
        let compute = compute(&mut dem, config);
        assert_eq!(compute.total_surfaces, EXPECTED_SURFACES);
    }

    #[gtest]
    fn refraction_affects_visibility() {
        let mut dem_for_no_refraction = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db_for_no_refraction = tempfile::NamedTempFile::new().unwrap();
        let compute_no_refraction = compute(
            &mut dem_for_no_refraction,
            super::Config {
                // Our "bigger_dem" isn't actually big enough for a 0.0 refraction to have an
                // affect. We already test for default refraction above, so may as well test for
                // 0.0 refraction here just in case there's some unexpected divergence.
                refraction: 0.0,
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..default_config(&temp_db_for_no_refraction)
            },
        );
        expect_eq!(compute_no_refraction.total_surfaces, EXPECTED_SURFACES);

        let mut dem_for_very_refraction = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db_for_very_refraction = tempfile::NamedTempFile::new().unwrap();
        let compute_very_refraction = compute(
            &mut dem_for_very_refraction,
            super::Config {
                refraction: -tvs_lib::projector::EARTH_DIAMETER,
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..default_config(&temp_db_for_very_refraction)
            },
        );
        #[rustfmt::skip]
            expect_eq!(
                compute_very_refraction.total_surfaces,
                [
                    0.0, 0.0,      0.0,       0.0,
                    0.0, 6.283163, 9.424768,  0.0,
                    0.0, 14.468756,38.083088, 0.0,
                    0.0, 0.0,      0.0,       0.0
                ]
            );
    }

    #[gtest]
    fn scale_affects_visibility() {
        let mut dem_for_small_scale = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db_for_small_scale = tempfile::NamedTempFile::new().unwrap();
        let compute_small_scale = compute(
            &mut dem_for_small_scale,
            super::Config {
                dem_metadata: tvs_lib::metadata::MetaData {
                    scale: 0.01,
                    ..crate::run::test::big_dem_metadata()
                },
                ..default_config(&temp_db_for_small_scale)
            },
        );
        #[rustfmt::skip]
            expect_eq!(
                compute_small_scale.total_surfaces,
                [
                    0.0, 0.0,       0.0,       0.0,
                    0.0, 0.06283202,0.21118537, 0.0,
                    0.0, 0.4790932, 0.6283214, 0.0,
                    0.0, 0.0,       0.0,       0.0
                ]
            );

        let mut dem_for_big_scale = make_dem(&crate::tests::fixtures::bigger_dem());
        let temp_db_for_big_scale = tempfile::NamedTempFile::new().unwrap();
        let compute_big_scale = compute(
            &mut dem_for_big_scale,
            super::Config {
                dem_metadata: tvs_lib::metadata::MetaData {
                    scale: 100.0,
                    ..crate::run::test::big_dem_metadata()
                },
                ..default_config(&temp_db_for_big_scale)
            },
        );
        #[rustfmt::skip]
            expect_eq!(
                compute_big_scale.total_surfaces,
                [
                    0.0, 0.0,      0.0,       0.0,
                    0.0, 628.317,  2111.8528, 0.0,
                    0.0, 4790.9106,6283.1714, 0.0,
                    0.0, 0.0,      0.0,       0.0
                ]
            );
    }
}
