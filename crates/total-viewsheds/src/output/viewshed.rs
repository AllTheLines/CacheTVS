//! Reconstruct _individual_ viewsheds, not total viewsheds.

use color_eyre::eyre::{ContextCompat as _, Result};
use geo::{BooleanOps as _, HasDimensions as _};

/// A viewshed-based coordinate is projected to a metric system where the anchor is the viewshed's
/// point of view. The other option would be a metric projection with an anchor in the DEM centre,
/// but metric projections are not globally correct. So reprojecting to the _viewshed's_ centre
/// just gives us that little bit more accuracy, especially for larger DEMs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate(pub geo::Coord);

/// `Viewshed`
pub struct Viewshed<'viewshed> {
    /// The DEM used to compute the final data.
    pub dem: &'viewshed crate::dem::DEM,
    /// Coordinate of the observer for the viewshed we want to reconstruct.
    pub pov_coord: crate::dem::Coordinate,
}

impl Viewshed<'_> {
    /// Reconstruct a viewshed.
    pub fn reconstruct(
        source: &super::ring_data::Source,
        pov_coord_latlon: crate::projection::LatLonCoord,
    ) -> Result<geo::MultiPolygon> {
        let ring_data = match source {
            crate::output::ring_data::Source::Directory(directory) => {
                &super::ring_data::AllData::new_from_fjall(directory)?
            }
            crate::output::ring_data::Source::SQLite(db_path) => {
                &super::ring_data::AllData::new_from_sqlite(db_path)?
            }
            #[cfg(test)]
            crate::output::ring_data::Source::RAM(data) => data,
        };
        tracing::debug!("Using metadata for ring data: {:?}", ring_data.metadata);

        let mut polygon = geo::MultiPolygon::empty();
        let dem = crate::dem::DEM::new(
            ring_data.metadata.centre,
            ring_data.metadata.width,
            ring_data.metadata.scale,
            ring_data.metadata.max_line_of_sight,
        )?;

        let pov_dem_coord = dem.latlon_to_dem_coord(pov_coord_latlon)?;
        tracing::info!(
            "Reconstructing viewshed for DEM-relative coord: {:?}.",
            pov_dem_coord
        );

        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: pov_dem_coord,
        };

        let constructor_by_sectors = || -> Result<geo::MultiPolygon> {
            for angle_integer in 0..crate::run::compute::SECTOR_STEPS {
                let angle = f32::from(angle_integer);
                let mut reconstructor =
                    Reconstructor::new(&viewshed, ring_data.metadata.reserved_ring_size, angle)?;
                reconstructor.sector_ring_data = ring_data.get_sector(angle_integer)?;
                polygon = reconstructor.reconstruct_sector(polygon)?;
            }

            Ok(polygon)
        };

        polygon = match source {
            crate::output::ring_data::Source::Directory(_) => constructor_by_sectors()?,
            crate::output::ring_data::Source::SQLite(db_path) => {
                let db = crate::cpu::storage::db::DB::new(db_path)?;
                let mut reconstructor =
                    Reconstructor::new(&viewshed, ring_data.metadata.reserved_ring_size, 0.0)?;

                let segments = db.load_segments_for_tvs_id(reconstructor.pov_id)?;
                reconstructor.parse_polar_segments(&segments)
            }
            #[cfg(test)]
            crate::output::ring_data::Source::RAM(_) => constructor_by_sectors()?,
        };

        Ok(polygon)
    }

    /// Convert from the viewshed projection to DEM coordinates.
    #[cfg(test)]
    pub fn convert_viewshed_coord_to_dem_coord(
        &self,
        viewshed_coord: Coordinate,
    ) -> Result<geo::Coord> {
        let scale = f64::from(self.dem.scale);
        let origin = crate::projection::Converter {
            base: self.dem.centre,
        }
        .to_degrees((self.pov_coord.0.x, self.pov_coord.0.y).into())?;
        let flipped = Coordinate(geo::Coord {
            x: viewshed_coord.0.x,
            y: -viewshed_coord.0.y,
        });
        let projected_coord = crate::projection::Converter::change_metric_origin(
            origin,
            // The path back to (0,0) is exactly the opposite of the viewshed's point of view.
            -self.pov_coord.0 * scale,
            flipped.0 * scale,
        )?;
        Ok(projected_coord / scale)
    }
}

/// `Reconstructor`
// TODO: Find a way to make this part of [`Viewshed`].
pub struct Reconstructor<'viewshed> {
    /// Data for the entire viewshed.
    viewshed: &'viewshed Viewshed<'viewshed>,
    /// Data about the visibility regions of each computed band of sight.
    sector_ring_data: Vec<u32>,
    /// Where we're currently reading sector data from.
    cursor: usize,
    /// Amount of reserved ring data space.
    reserved_ring_size: usize,
    /// The DEM id of the observer.
    pov_id: u32,
    /// The current sector angle
    current_angle: f32,
}

impl<'viewshed> Reconstructor<'viewshed> {
    /// Instantiate the reconstructor for a single angle.
    ///
    /// The reason we only reconstruct for a single angle is that the sector data (for the angle)
    /// is most useful exposed as an iterator. And it's not so easy with lifetimes to keep around
    /// the raw data and an iterator for each angle.
    fn new(
        viewshed: &'viewshed Viewshed<'viewshed>,
        reserved_ring_size: usize,
        angle: f32,
    ) -> Result<Self> {
        let pov_id = viewshed.dem.dem_coord_to_id(viewshed.pov_coord);
        let reconstructor = Self {
            viewshed,
            sector_ring_data: Vec::default(),
            cursor: 0,
            reserved_ring_size,
            pov_id,
            current_angle: angle,
        };

        if !viewshed.dem.is_point_computable(pov_id) {
            color_eyre::eyre::bail!(
                "Point of view ({:?}) is not calculable",
                reconstructor.viewshed.pov_coord
            );
        }

        Ok(reconstructor)
    }

    /// Extract and reconstruct a single viewshed from all the ring data for all possible viewsheds.
    pub fn reconstruct_sector(&mut self, viewshed: geo::MultiPolygon) -> Result<geo::MultiPolygon> {
        tracing::debug!(
            "Building viewshed for sector {} using ring data of length {}",
            self.current_angle,
            self.sector_ring_data.len()
        );
        self.parse_fjall_sector(viewshed)
    }

    /// Read the next value in the ring data array.
    fn read_next_value(&mut self) -> Result<u32> {
        let value = *self
            .sector_ring_data
            .get(self.cursor)
            .context("Couldn't get next ring in ring data")?;
        self.cursor += 1;
        Ok(value)
    }

    /// Parse an entire sector (angle) of ring data.
    fn parse_fjall_sector(
        &mut self,
        mut viewshed_so_far: geo::MultiPolygon,
    ) -> Result<geo::MultiPolygon> {
        let tvs_id = self.viewshed.dem.pov_id_to_tvs_id(u64::from(self.pov_id));
        let rotated_tvs_id = kernel::rotation::Rotator::rotate_index(
            u32::try_from(tvs_id)?,
            self.viewshed.dem.tvs_width,
            self.current_angle,
        );

        self.cursor = rotated_tvs_id * self.reserved_ring_size;

        let computable_points = usize::try_from(self.viewshed.dem.computable_points_count)?;
        let wrap_to_backwards_ids = (rotated_tvs_id + computable_points) * self.reserved_ring_size;

        let max_rings = u32::try_from((self.reserved_ring_size - 2).div_euclid(2))?;
        for direction in [
            kernel::elevations::Direction::Forward,
            kernel::elevations::Direction::Backward,
        ] {
            // We divide by 2 because every ring must have both an opening and a closing.
            let mut no_of_ring_values = self.read_next_value()?.div_euclid(2);

            if no_of_ring_values == 0 {
                // These are most common outside the circle area of the TVS region.
                self.cursor = wrap_to_backwards_ids;
                continue;
            }
            if no_of_ring_values > max_rings {
                tracing::warn!(
                    "More rings in band than reserved rings ({} > {}) for point {:?}",
                    no_of_ring_values,
                    max_rings,
                    direction
                );
                no_of_ring_values = max_rings;
            }

            for ring in 0..no_of_ring_values {
                let opening = if ring == 0 {
                    0
                } else {
                    self.read_next_value()?
                };
                let closing = self.read_next_value()?;

                let polygon = self.make_visible_polygon(opening, closing, &direction);
                viewshed_so_far = viewshed_so_far.union(&polygon);
                if viewshed_so_far.is_empty() {
                    color_eyre::eyre::bail!("Invalid polygon: {polygon:?}");
                }
            }

            self.cursor = wrap_to_backwards_ids;
        }

        Ok(viewshed_so_far)
    }

    /// Convert polar segements to `GeoJson`.
    pub fn parse_polar_segments(
        &mut self,
        data: &[Vec<crate::cpu::storage::segments::Segment>],
    ) -> geo::MultiPolygon {
        let mut viewshed_so_far = geo::MultiPolygon::empty();
        for (angle, segments) in data.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "The angle will always fit in `f32`"
            )]
            {
                self.current_angle = angle as f32;
            };
            for segment in segments {
                let opening = u32::from(segment.start());
                let closing = u32::from(segment.start() + segment.distance());
                let polygon = self.make_visible_polygon(
                    opening,
                    closing,
                    &kernel::elevations::Direction::Forward,
                );
                viewshed_so_far = viewshed_so_far.union(&polygon);
            }
        }

        viewshed_so_far
    }

    /// Convert an index along a line of sight into a coordinate.
    fn index_to_coordinate(
        &self,
        index: u32,
        direction: &kernel::elevations::Direction,
    ) -> Coordinate {
        let angle = match direction {
            kernel::elevations::Direction::Forward => self.current_angle.to_radians(),
            kernel::elevations::Direction::Backward => (self.current_angle + 180.0).to_radians(),
        };
        let distance = f64::from(index);

        Coordinate(geo::coord! {
            x: distance * f64::from(angle.cos()),
            y: distance * f64::from(angle.sin())
        })
    }

    /// Rotate a point about the centre of the viewshed.
    #[expect(
        clippy::suboptimal_flops,
        reason = "I think readability is more important?"
    )]
    fn rotate_by(point: Coordinate, angle: f64) -> geo::Coord {
        let dx = point.0.x;
        let dy = point.0.y;
        let cos = angle.to_radians().cos();
        let sin = angle.to_radians().sin();
        geo::coord! {
            x: dx * cos - dy * sin,
            y: dx * sin + dy * cos
        }
    }

    /// Make a single polygon representing a visible region of the planet.
    fn make_visible_polygon(
        &self,
        opening_index: u32,
        closing_index: u32,
        direction: &kernel::elevations::Direction,
    ) -> geo::Polygon {
        let opening_coord = self.index_to_coordinate(opening_index, direction);
        let closing_coord = self.index_to_coordinate(closing_index, direction);

        let spread = 0.5001f64;
        let bottom_left = Self::rotate_by(opening_coord, spread);
        let bottom_right = Self::rotate_by(opening_coord, -spread);
        let top_left = Self::rotate_by(closing_coord, spread);
        let top_right = Self::rotate_by(closing_coord, -spread);

        let scale = f64::from(self.viewshed.dem.scale);

        geo::Polygon::new(
            geo::LineString(vec![
                bottom_left * scale,
                bottom_right * scale,
                top_right * scale,
                top_left * scale,
                bottom_left * scale,
            ]),
            vec![],
        )
    }

    /// Save the viewshed to disk.
    #[expect(
        clippy::panic_in_result_fn,
        clippy::panic,
        reason = "The closures expects () so I don't think there's any other way?"
    )]
    pub fn save(
        mut viewshed: geo::MultiPolygon,
        output_directory: &std::path::Path,
        viewshed_latlon: crate::projection::LatLonCoord,
    ) -> Result<()> {
        let filename = format!("{}-{}.json", viewshed_latlon.0.x, viewshed_latlon.0.y);
        let directory = output_directory.join("viewsheds");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join(filename);
        let projector = crate::projection::Converter {
            base: viewshed_latlon,
        };

        for point in viewshed.iter_mut() {
            point.exterior_mut(|line| {
                for coordinate in line.coords_mut() {
                    let projected = projector
                        .to_degrees(geo::Coord {
                            x: coordinate.x,
                            y: coordinate.y,
                        })
                        .unwrap_or_else(|_| {
                            panic!(
                                "Couldn't project viewshed coordinate to degrees: {coordinate:?}",
                            )
                        });
                    *coordinate = projected.0;
                }
            });

            point.interiors_mut(|lines| {
                for line in lines {
                    for coordinate in line.coords_mut() {
                        let projected = projector
                            .to_degrees(geo::Coord {
                                x: coordinate.x,
                                y: coordinate.y,
                            })
                            .unwrap_or_else(|_| {
                                panic!(
                                "Couldn't project viewshed coordinate to degrees: {coordinate:?}",
                            )
                            });
                        *coordinate = projected.0;
                    }
                }
            });
        }
        let json = geojson::GeoJson::from(&viewshed).to_string();
        std::fs::write(path, json)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{output::ascii::assert_viewshed, run};

    use super::*;

    const RESERVED_RING_SIZE: usize = crate::run::compute::Compute::ring_count_per_band(5000.0, 3);

    fn builder<'viewshed>(viewshed: &'viewshed Viewshed, angle: f32) -> Reconstructor<'viewshed> {
        Reconstructor::new(viewshed, RESERVED_RING_SIZE, angle).unwrap()
    }

    #[derive(Debug)]
    struct VisiblePolygonFor {
        pov: geo::Coord,
        angle: f32,
        opening_index: u32,
        closing_index: u32,
    }

    fn make_visible_polygon_for(setup: &VisiblePolygonFor) -> Vec<geo::Coord> {
        let dem = crate::run::compute::test::make_dem(&kernel::tests::dems::single_peak_dem());
        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: crate::dem::Coordinate(setup.pov),
        };
        let direction = if setup.angle < 180.0 {
            kernel::elevations::Direction::Forward
        } else {
            kernel::elevations::Direction::Backward
        };
        let angle = match direction {
            kernel::elevations::Direction::Forward => setup.angle,
            kernel::elevations::Direction::Backward => setup.angle - 180.0,
        };
        let viewsheder = builder(&viewshed, angle);
        let polygon =
            viewsheder.make_visible_polygon(setup.opening_index, setup.closing_index, &direction);

        let mut polygon_as_dem_coords = Vec::new();
        for coord in &polygon.exterior().0 {
            let converted_coord = viewsheder
                .viewshed
                .convert_viewshed_coord_to_dem_coord(Coordinate(*coord))
                .unwrap();
            polygon_as_dem_coords.push(round_coordinate(converted_coord));
        }
        polygon_as_dem_coords
    }

    fn round(float: f64) -> f64 {
        let factor = 10f64.powi(7);
        (float * factor).round() / factor
    }

    fn round_coordinate(coordinate: geo::Coord) -> geo::Coord {
        geo::coord! {
          x: round(coordinate.x),
          y: round(coordinate.y),
        }
    }

    // Guide for the following tests:
    //
    //    0  1  2  3  4  5  6  7  8
    // 0  .  .  .  .  .  .  .  .  .
    // 1  .  .  .  .  .  .d .  .  .
    // 2  .  .  .  .  .a .  )  .  .
    // 3  .  .  .  .  .  (  . c.  .
    // 4  .  .  .  .  o  . b.  .  .
    // 5  .  .  .  .  .  .  .  .  .
    // 6  .  .  .  .  .  .  .  .  .
    // 7  .  .  .  .  .  .  .  .  .
    // 8  .  .  .  .  .  .  .  .  .
    //
    mod from_centre_to_top_right {
        use super::*;

        const POV: geo::Coord = geo::coord! {x: 4.0, y: 4.0};
        const ANGLE: f32 = 45.0;

        // The polygon we're making is `abcd` from the above guide.
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    pov: POV,
                    angle: ANGLE,
                    opening_index: 1,
                    closing_index: 2,
                }),
                vec![
                    (4.7009067, 3.2867503),
                    (4.7132503, 3.2990939),
                    (5.4265022, 2.5981862),
                    (5.401815, 2.5734989),
                    (4.7009067, 3.2867503)
                ]
                .into_iter()
                .map(Into::into)
                .collect::<Vec<geo::Coord>>()
            );
        }
    }

    // Guide for the following tests:
    //
    //    0  1  2  3  4  5  6  7  8
    // 0  .  .  .  .  .  .  .  .  .
    // 1  .  .  .  .  .  .  .  .  .
    // 2  .  .  .  .  .  .  .  .  .
    // 3  .  .  .  .  .  .  .  .  .
    // 4  .  .  .  .  .  .  .  .  .
    // 5  .  .  .  o  .a .  .  .  .
    // 6  .  .  .  .  (  .d .  .  .
    // 7  .  .  .  . b.  )  .  .  .
    // 8  .  .  .  .  . c.  .  .  .
    //
    mod from_bottom_left_to_bottom_right {
        use super::*;

        const POV: geo::Coord = geo::coord! {x: 3.0, y: 5.0};
        const ANGLE: f32 = 135.0 + 180.0;

        // The polygon we're making is `abcd` from the above guide.
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    pov: POV,
                    angle: ANGLE,
                    opening_index: 1,
                    closing_index: 2,
                }),
                vec![
                    (3.7132498, 5.7009093),
                    (3.7009061, 5.7132529),
                    (4.4018138, 6.4265049),
                    (4.4265011, 6.4018176),
                    (3.7132498, 5.7009093)
                ]
                .into_iter()
                .map(Into::into)
                .collect::<Vec<geo::Coord>>()
            );
        }
    }

    fn viewshed_in_hole(backend: &crate::config::Backend) {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &kernel::tests::dems::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            crate::run::compute::test::default_config(backend.clone(), &temp_db),
        );

        let expected = &[
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████▀ ▄ ▀███████████",
            "████████ ▀█▀ ███████████",
            "█████████▄▄▄████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];

        match backend {
            crate::config::Backend::VulkanCPU | crate::config::Backend::CPU => {
                assert_viewshed(&viewshed, expected);
            }
            crate::config::Backend::Vulkan | crate::config::Backend::Cuda => {
                panic!("We're not testing these.")
            }
        }
    }

    fn viewshed_on_summit(backend: crate::config::Backend) {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &kernel::tests::dems::bigger_dem(),
            geo::Coord { x: 6.0, y: 6.0 },
            crate::run::compute::test::default_config(backend, &temp_db),
        );

        assert_viewshed(
            &viewshed,
            &[
                "████████████████████████",
                "████████████████████████",
                "███████▀▀ ▄▄▄▄▄ ▀▀██████",
                "█████▀ ▄█████████▄ ▀████",
                "████▀ █████████████ ▀███",
                "████ ███████████████ ███",
                "████ ███████████████ ███",
                "████ ▀█████████████▀ ███",
                "█████ ▀███████████▀ ████",
                "██████▄ ▀▀█████▀▀ ▄█████",
                "█████████▄▄▄▄▄▄▄████████",
                "████████████████████████",
            ],
        );
    }

    fn viewshed_near_summit(backend: &crate::config::Backend) {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &kernel::tests::dems::bigger_dem(),
            geo::Coord { x: 5.0, y: 6.0 },
            crate::run::compute::test::default_config(backend.clone(), &temp_db),
        );

        let expected = &[
            "████████████████████████",
            "████████████████████████",
            "█████▀▀ ▄▄▄▄▄ ▀▀████████",
            "███▀ ▄█████████ ▄███████",
            "██▀ █████████▀▄█████████",
            "██ ████████▀ ███████████",
            "██ ████████▀ ███████████",
            "██ ▀████████▄▀██████████",
            "███ ▀█████████▄▀████████",
            "████▄ ▀▀█████▀▀ ▄███████",
            "███████▄▄▄▄▄▄▄██████████",
            "████████████████████████",
        ];

        match backend {
            crate::config::Backend::VulkanCPU | crate::config::Backend::CPU => {
                assert_viewshed(&viewshed, expected);
            }
            crate::config::Backend::Vulkan | crate::config::Backend::Cuda => {
                panic!("We're not testing these.")
            }
        }
    }

    fn viewshed_in_hole_tall_observer(backend: crate::config::Backend) {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &kernel::tests::dems::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            run::compute::Config {
                observer_height: 20.0,
                ..crate::run::compute::test::default_config(backend, &temp_db)
            },
        );

        assert_viewshed(
            &viewshed,
            &[
                "████████████████████████",
                "█████▀▀ ▄▄▄▄▄ ▀▀████████",
                "███▀ ▄█████████▄ ▀██████",
                "██▀ █████████████ ▀█████",
                "██ ███████████████ █████",
                "██ ███████████████ █████",
                "██ ▀█████████████▀ █████",
                "███ ▀███████████▀ ██████",
                "████▄ ▀▀█████▀▀ ▄███████",
                "███████▄▄▄▄▄▄▄██████████",
                "████████████████████████",
                "████████████████████████",
            ],
        );
    }

    mod gpu {
        #[test]
        fn viewshed_in_hole() {
            super::viewshed_in_hole(&crate::config::Backend::VulkanCPU);
        }

        #[test]
        fn viewshed_on_summit() {
            super::viewshed_on_summit(crate::config::Backend::VulkanCPU);
        }

        #[test]
        fn viewshed_near_summit() {
            super::viewshed_near_summit(&crate::config::Backend::VulkanCPU);
        }

        #[test]
        fn viewshed_in_hole_tall_observer() {
            super::viewshed_in_hole_tall_observer(crate::config::Backend::VulkanCPU);
        }
    }

    mod cpu {
        #[test]
        fn viewshed_in_hole() {
            super::viewshed_in_hole(&crate::config::Backend::CPU);
        }

        #[test]
        fn viewshed_on_summit() {
            super::viewshed_on_summit(crate::config::Backend::CPU);
        }

        #[test]
        fn viewshed_near_summit() {
            super::viewshed_near_summit(&crate::config::Backend::CPU);
        }

        #[test]
        fn viewshed_in_hole_tall_observer() {
            super::viewshed_in_hole_tall_observer(crate::config::Backend::CPU);
        }
    }
}
