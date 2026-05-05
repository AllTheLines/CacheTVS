//! Reconstruct _individual_ viewsheds, not total viewsheds.

use color_eyre::eyre::Result;
use geo::BooleanOps as _;

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
        db_path: std::path::PathBuf,
        pov_coord_lonlat: crate::projection::LonLatCoord,
    ) -> Result<geo::MultiPolygon> {
        let db = crate::storage::db::DB::new(db_path)?;
        let metadata = db.load_metadata()?;
        tracing::debug!("Using metadata for ring data: {:?}", metadata);

        let dem = crate::dem::DEM::new(
            metadata.centre,
            metadata.width,
            metadata.scale,
            metadata.max_line_of_sight,
        )?;

        let pov_dem_coord =
            crate::projection::Converter::lonlat_to_dem_coord(&metadata, pov_coord_lonlat)?;
        tracing::info!(
            "Reconstructing viewshed for DEM-relative coord: {:?}.",
            pov_dem_coord
        );

        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: pov_dem_coord,
        };

        let mut reconstructor = Reconstructor::new(&viewshed, 0.0)?;

        let segments = db.load_segments_for_tvs_id(reconstructor.pov_id)?;
        let polygon = reconstructor.parse_polar_segments(&segments);

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
    fn new(viewshed: &'viewshed Viewshed<'viewshed>, angle: f32) -> Result<Self> {
        let pov_id = viewshed.dem.dem_coord_to_id(viewshed.pov_coord);
        let reconstructor = Self {
            viewshed,
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

    /// Convert polar segements to `GeoJson`.
    pub fn parse_polar_segments(
        &mut self,
        data: &[Vec<crate::storage::segments::Segment>],
    ) -> geo::MultiPolygon {
        let mut viewshed_so_far = geo::MultiPolygon::empty();
        for (angle, segments) in data.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "The angle will always fit in `f32`"
            )]
            {
                // -45?? This was needed after moving to skew rotation 😬
                self.current_angle = angle as f32 - 45.0;
            };
            for segment in segments {
                let opening = u32::from(segment.start());
                let closing = u32::from(segment.start() + segment.distance());
                let polygon = self.make_visible_polygon(opening, closing);
                viewshed_so_far = viewshed_so_far.union(&polygon);
            }
        }

        viewshed_so_far
    }

    /// Convert an index along a line of sight into a coordinate.
    fn index_to_coordinate(&self, index: u32) -> Coordinate {
        let radians = self.current_angle.to_radians();
        let distance = f64::from(index);

        Coordinate(geo::coord! {
            x: distance * f64::from(radians.cos()),
            y: distance * f64::from(radians.sin())
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
    fn make_visible_polygon(&self, opening_index: u32, closing_index: u32) -> geo::Polygon {
        let opening_coord = self.index_to_coordinate(opening_index);
        let closing_coord = self.index_to_coordinate(closing_index);

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
        viewshed_latlon: crate::projection::LonLatCoord,
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
    use super::*;
    use crate::{output::ascii::assert_viewshed, run};

    fn builder<'viewshed>(viewshed: &'viewshed Viewshed, angle: f32) -> Reconstructor<'viewshed> {
        Reconstructor::new(viewshed, angle).unwrap()
    }

    #[derive(Debug)]
    struct VisiblePolygonFor {
        pov: geo::Coord,
        angle: f32,
        opening_index: u32,
        closing_index: u32,
    }

    fn make_visible_polygon_for(setup: &VisiblePolygonFor) -> Vec<geo::Coord> {
        let dem = crate::run::test::make_dem(&crate::tests::fixtures::single_peak_dem());
        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: crate::dem::Coordinate(setup.pov),
        };
        let viewsheder = builder(&viewshed, setup.angle);
        let polygon = viewsheder.make_visible_polygon(setup.opening_index, setup.closing_index);

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

    #[test]
    fn viewshed_in_hole() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            run::Config {
                dem_metadata: run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
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
        assert_viewshed(&viewshed, expected);
    }

    #[test]
    fn viewshed_on_summit() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 6.0, y: 6.0 },
            run::Config {
                dem_metadata: run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );

        let expected = [
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
        ];
        assert_viewshed(&viewshed, &expected);
    }

    #[test]
    fn viewshed_near_summit() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 6.0 },
            run::Config {
                dem_metadata: run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
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
        assert_viewshed(&viewshed, expected);
    }

    #[test]
    fn viewshed_in_hole_tall_observer() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            run::Config {
                observer_height: 20.0,
                dem_metadata: run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );

        let expected = &[
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
        ];
        assert_viewshed(&viewshed, expected);
    }
}
