//! Reconstruct _individual_ viewsheds, not total viewsheds.
//!
//! Here is a crude diagram of the raw visibility data returned from the kernel:
//!
//!    . . .C/ . / . / .
//!    . . . . . ) . . .
//!    . . / . / . /D. .
//!    . . . . . . . . .
//!    .A/ . / . / . . .
//!    . . ( . . . . . .
//!    / . / . /B. . . .
//!    . . . . . . . . .
//!    . o . / . . . . .
//!
//! Key:
//!  * `o`: The point of view of the observer.
//!  * `/`: The left edge, centre, and right edge of the band of sight.
//!  * `(`: Opening of a visible "ring" within the band of sight.
//!  * `)`: Closing of a visible "ring" within the band of sight.
//!  * The line between `A` and `B` is the beginning of a visible region.
//!  * The line between `C` and `D` is the end of a visible region.

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
    dem: &'viewshed crate::dem::DEM,
    /// Coordinate of the observer for the viewshed we want to reconstruct.
    pov_coord: crate::dem::Coordinate,
}

impl Viewshed<'_> {
    /// Reconstruct a viewshed.
    pub fn reconstruct(
        source: &super::ring_data::Source,
        pov_coord_latlon: crate::projection::LatLonCoord,
    ) -> Result<geo::MultiPolygon> {
        let ring_data = match source {
            crate::output::ring_data::Source::Directory(directory) => {
                &super::ring_data::AllData::new_from_storage(directory)?
            }
            #[cfg(test)]
            crate::output::ring_data::Source::RAM(data) => data,
        };
        tracing::debug!("Using metadata for ring data: {:?}", ring_data.metadata);

        let sector_shift = f64::from(ring_data.metadata.sector_shift);
        let mut polygon = geo::MultiPolygon::empty();
        let mut current_angle = sector_shift;
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

        for angle in 0..crate::axes::SECTOR_STEPS {
            let mut reconstructor = Reconstructor::new(
                &viewshed,
                ring_data.metadata.reserved_ring_size,
                current_angle,
            )?;
            let sector_ring_data = ring_data.get_sector(angle)?;
            reconstructor.sector_ring_data = sector_ring_data.iter();
            polygon = reconstructor.reconstruct_sector(polygon)?;
            current_angle += 1.0f64;
        }

        Ok(polygon)
    }

    /// Convert from DEM coordinates (used by the GPU and tests) to the true geometric coordinate
    /// projection of the viewshed.
    fn convert_dem_coord_to_viewshed_coord(
        &self,
        dem_coord: crate::dem::Coordinate,
    ) -> Result<Coordinate> {
        let scale = f64::from(self.dem.scale);
        let projected_coord = crate::projection::Converter::change_metric_origin(
            self.dem.centre,
            geo::coord! {
                x: self.pov_coord.0.x,
                y: -self.pov_coord.0.y
            } * scale,
            geo::coord! {
                x: dem_coord.0.x,
                y: -dem_coord.0.y
            } * scale,
        )?;
        let viewshed_coord = Coordinate(
            geo::coord! {
                x: projected_coord.x,
                y: -projected_coord.y
            } / scale,
        );
        Ok(viewshed_coord)
    }

    /// Convert from the viewshed projection to DEM coordinates.
    #[cfg(test)]
    fn convert_viewshed_coord_to_dem_coord(
        &self,
        viewshed_coord: Coordinate,
    ) -> Result<geo::Coord> {
        let scale = f64::from(self.dem.scale);
        let origin = crate::projection::Converter {
            base: self.dem.centre,
        }
        .to_degrees((self.pov_coord.0.x, self.pov_coord.0.y).into())?;
        let projected_coord = crate::projection::Converter::change_metric_origin(
            origin,
            // The path back to (0,0) is exactly the opposite of the viewshed's point of view.
            -self.pov_coord.0 * scale,
            viewshed_coord.0 * scale,
        )?;
        Ok(projected_coord / scale)
    }
}

/// `Reconstructor`
// TODO: Find a way to make this part of [`Viewshed`].
pub struct Reconstructor<'viewshed, 'sector> {
    /// Data for the entire viewshed.
    viewshed: &'viewshed Viewshed<'viewshed>,
    /// Data about the visibility regions of each computed band of sight.
    sector_ring_data: std::slice::Iter<'sector, u32>,
    /// Amount of reserved ring data space.
    reserved_ring_size: usize,
    /// The DEM id of the observer.
    pov_id: u32,
    /// The current sector angle
    current_angle: f64,
}

impl<'viewshed> Reconstructor<'viewshed, '_> {
    /// Instantiate the reconstructor for a single angle.
    ///
    /// The reason we only reconstruct for a single angle is that the sector data (for the angle)
    /// is most useful exposed as an iterator. And it's not so easy with lifetimes to keep around
    /// the raw data and an iterator for each angle.
    fn new(
        viewshed: &'viewshed Viewshed<'viewshed>,
        reserved_ring_size: usize,
        angle: f64,
    ) -> Result<Self> {
        let pov_id = viewshed.dem.dem_coord_to_id(viewshed.pov_coord);
        let reconstructor = Self {
            viewshed,
            sector_ring_data: std::slice::Iter::default(),
            reserved_ring_size,
            pov_id,
            current_angle: angle - 90.0f64,
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
        self.parse_sector(viewshed)
    }

    /// Read the next value in the ring data array.
    fn read_next_value(&mut self) -> Result<u32> {
        let value = *self
            .sector_ring_data
            .next()
            .context("Couldn't get next ring in ring data")?;
        Ok(value)
    }

    /// Parse an entire sector (angle) of ring data.
    fn parse_sector(
        &mut self,
        mut viewshed_so_far: geo::MultiPolygon,
    ) -> Result<geo::MultiPolygon> {
        let max_rings = u32::try_from((self.reserved_ring_size - 2).div_euclid(2))?;
        for point in 0..self.viewshed.dem.computable_points_count * 2 {
            // We divide by 2 because every ring must have both an opening and a closing.
            let mut no_of_ring_values = self.read_next_value()?.div_euclid(2);

            if no_of_ring_values == 0 {
                tracing::warn!("No rings for point {point}.");
                self.sector_ring_data.nth(usize::try_from(max_rings * 2)?);
                continue;
            }
            if no_of_ring_values > max_rings {
                tracing::warn!(
                    "More rings in band than reserved rings ({} > {}) for point {}",
                    no_of_ring_values,
                    max_rings,
                    point
                );
                no_of_ring_values = max_rings;
            }

            // Assume that every DEM point has an opening at the PoV.
            let pov_id = self.read_next_value()?;

            for index in 0..no_of_ring_values {
                let opening = if index == 0 {
                    pov_id
                } else {
                    self.read_next_value()?
                };
                let closing = self.read_next_value()?;

                // TODO: jump straight to this, rather than looping ever point in between. Should
                // get a huge speedup.
                if pov_id == self.pov_id {
                    let polygon = self.make_visible_polygon(opening, closing)?;
                    viewshed_so_far = viewshed_so_far.union(&polygon);
                    if viewshed_so_far.is_empty() {
                        color_eyre::eyre::bail!("Invalid polygon: {polygon:?}");
                    }
                }
            }

            let skip = self.reserved_ring_size - ((usize::try_from(no_of_ring_values)?) * 2) - 2;
            self.sector_ring_data.nth(skip);
        }

        if viewshed_so_far.is_empty() {
            color_eyre::eyre::bail!("No polygon rings added.");
        }

        Ok(viewshed_so_far)
    }

    /// Find the intersection between the given point and its shortest path to the current band of
    /// sight's centre.
    #[expect(
        clippy::suboptimal_flops,
        reason = "I think readability is more important?"
    )]
    fn intersection_with_band_centre(&self, point: Coordinate) -> Coordinate {
        // TODO: Is there not a form of the equation where we don't need to flip here?
        let flip = f64::from(self.viewshed.dem.width - 1);
        let flipped_point_y = flip - point.0.y;
        let flipped_pov_y = flip;

        let angle = self.current_angle.to_radians();
        let cos = angle.cos();
        let sin = angle.sin();
        let factor = point.0.x * cos + (flipped_point_y - flipped_pov_y) * sin;
        Coordinate(geo::coord! {
            x: cos * factor,
            y: flip - (flipped_pov_y + sin * factor)
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
        opening_dem_id: u32,
        closing_dem_id: u32,
    ) -> Result<geo::Polygon> {
        let opening_dem_coord = self.viewshed.dem.convert_dem_id_to_coord(opening_dem_id);
        let closing_dem_coord = self.viewshed.dem.convert_dem_id_to_coord(closing_dem_id);
        let opening = self
            .viewshed
            .convert_dem_coord_to_viewshed_coord(opening_dem_coord)?;
        let closing = self
            .viewshed
            .convert_dem_coord_to_viewshed_coord(closing_dem_coord)?;
        let opening_intersection = self.intersection_with_band_centre(opening);
        let closing_intersection = self.intersection_with_band_centre(closing);

        let spread = 0.5001f64;
        let bottom_left = Self::rotate_by(opening_intersection, spread);
        let bottom_right = Self::rotate_by(opening_intersection, -spread);
        let top_left = Self::rotate_by(closing_intersection, spread);
        let top_right = Self::rotate_by(closing_intersection, -spread);

        let scale = f64::from(self.viewshed.dem.scale);

        Ok(geo::Polygon::new(
            geo::LineString(vec![
                bottom_left * scale,
                bottom_right * scale,
                top_right * scale,
                top_left * scale,
                bottom_left * scale,
            ]),
            vec![],
        ))
    }

    /// Save the viewshed to disk.
    #[expect(
        clippy::panic_in_result_fn,
        clippy::panic,
        reason = "The closures expect () so I don't think there's any other way?"
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
                            y: -coordinate.y,
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
                                y: -coordinate.y,
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

#[expect(
    clippy::unreadable_literal,
    clippy::default_numeric_fallback,
    reason = "It's just for the tests"
)]
#[cfg(test)]
mod test {
    use geo::Extremes as _;

    use super::*;

    const SIGHT_OFFSET: f64 = 90.0;
    const RESERVED_RING_SIZE: usize = crate::compute::Compute::ring_count_per_band(5000.0, 3);

    fn builder<'viewshed, 'sector>(
        viewshed: &'viewshed Viewshed,
        angle: f64,
    ) -> Reconstructor<'viewshed, 'sector> {
        Reconstructor::new(viewshed, RESERVED_RING_SIZE, angle + SIGHT_OFFSET).unwrap()
    }

    struct IntersectFor {
        pov: geo::Coord,
        angle: f64,
        point: geo::Coord,
    }

    fn intersect_for(setup: &IntersectFor) -> geo::Coord {
        let mut dem = crate::compute::test::make_dem();
        crate::compute::test::compute(&mut dem, &crate::compute::test::single_peak_dem());
        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: crate::dem::Coordinate(setup.pov),
        };
        let viewsheder = builder(&viewshed, setup.angle);

        let viewshed_coord = viewsheder
            .viewshed
            .convert_dem_coord_to_viewshed_coord(crate::dem::Coordinate(setup.point))
            .unwrap();
        let projected = viewsheder.intersection_with_band_centre(viewshed_coord);
        let coordinate = viewsheder
            .viewshed
            .convert_viewshed_coord_to_dem_coord(projected)
            .unwrap();
        round_coordinate(coordinate)
    }

    struct VisiblePolygonFor {
        pov: geo::Coord,
        angle: f64,
        opening_coord: geo::Coord,
        closing_coord: geo::Coord,
    }

    fn make_visible_polygon_for(setup: &VisiblePolygonFor) -> Vec<geo::Coord> {
        let mut dem = crate::compute::test::make_dem();
        crate::compute::test::compute(&mut dem, &crate::compute::test::single_peak_dem());
        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: crate::dem::Coordinate(setup.pov),
        };
        let viewsheder = builder(&viewshed, setup.angle);
        let opening_dem_id = dem.dem_coord_to_id(crate::dem::Coordinate(setup.opening_coord));
        let closing_dem_id = dem.dem_coord_to_id(crate::dem::Coordinate(setup.closing_coord));
        let polygon = viewsheder
            .make_visible_polygon(opening_dem_id, closing_dem_id)
            .unwrap();

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
    // 1  .  .  .  .  .  .c .  .  .
    // 2  .  .  .  .  .a .  )  .  .
    // 3  .  .  .  .  .  (  . d.  .
    // 4  .  .  .  .  o  . b.  .  .
    // 5  .  .  .  .  .  .  .  .  .
    // 6  .  .  .  .  .  .  .  .  .
    // 7  .  .  .  .  .  .  .  .  .
    // 8  .  .  .  .  .  .  .  .  .
    //
    mod from_centre_to_top_right {
        use super::*;
        use googletest::prelude::*;

        const POV: geo::Coord = geo::coord! {x: 4.0, y: 4.0};
        const ANGLE: f64 = 45.0;

        #[gtest]
        fn intersection_with_band_centre() {
            expect_eq!(
                intersect_for(&IntersectFor {
                    pov: POV,
                    angle: ANGLE,
                    point: geo::Coord { x: 4.0, y: 2.0 },
                }),
                geo::coord! { x: 4.9999992, y: 3.0000016 }
            );

            expect_eq!(
                intersect_for(&IntersectFor {
                    pov: POV,
                    angle: ANGLE,
                    point: geo::Coord { x: 8.0, y: 2.0 },
                }),
                geo::coord! { x: 6.9999992, y: 1.0000033 }
            );

            expect_eq!(
                intersect_for(&IntersectFor {
                    pov: POV,
                    angle: ANGLE,
                    point: geo::Coord { x: 8.0, y: 3.0 },
                }),
                geo::coord! { x: 6.4999988, y: 1.5000033 }
            );
        }

        // The polygon we're making is `abcd` from the above guide.
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    pov: POV,
                    angle: ANGLE,
                    opening_coord: geo::coord! {x: 5.0, y: 3.0},
                    closing_coord: geo::coord! {x: 6.0, y: 2.0},
                }),
                vec![
                    (5.0086889, 3.0087684),
                    (4.9912324, 2.9913119),
                    (5.9824664, 1.9826221),
                    (6.0173795, 2.0175352),
                    (5.0086889, 3.0087684)
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
    // 5  .  .  .  o  .b .  .  .  .
    // 6  .  .  .  .  (  .c .  .  .
    // 7  .  .  .  . a.  )  .  .  .
    // 8  .  .  .  .  . d.  .  .  .
    //
    mod from_bottom_left_to_bottom_right {
        use super::*;
        use googletest::prelude::*;

        const POV: geo::Coord = geo::coord! {x: 3.0, y: 5.0};
        const ANGLE: f64 = 135.0;

        #[gtest]
        fn intersection_with_band_centre() {
            expect_eq!(
                intersect_for(&IntersectFor {
                    pov: POV,
                    angle: ANGLE,
                    point: geo::Coord { x: 4.5, y: 5.0 },
                }),
                geo::coord! { x: 3.7499985, y: 5.7500014 }
            );

            expect_eq!(
                intersect_for(&IntersectFor {
                    pov: POV,
                    angle: ANGLE,
                    point: geo::Coord { x: 5.0, y: 7.5 },
                }),
                geo::coord! { x: 5.2499977, y: 7.2500015 }
            );
        }

        // The polygon we're making is `abcd` from the above guide.
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    pov: POV,
                    angle: ANGLE,
                    opening_coord: geo::coord! {x: 4.0, y: 6.0},
                    closing_coord: geo::coord! {x: 5.0, y: 7.0},
                }),
                vec![
                    (3.9912318, 6.0086914),
                    (4.0086883, 5.9912349),
                    (5.0173782, 6.9824688),
                    (4.9824651, 7.0173819),
                    (3.9912318, 6.0086914),
                ]
                .into_iter()
                .map(Into::into)
                .collect::<Vec<geo::Coord>>()
            );
        }
    }

    #[test]
    fn final_viewshed() {
        let mut dem = crate::compute::test::make_dem();
        let compute =
            crate::compute::test::compute(&mut dem, &crate::compute::test::single_peak_dem());

        let mut viewshed = Viewshed::reconstruct(
            &super::super::ring_data::Source::RAM(crate::output::ring_data::AllData {
                metadata: compute.metadata().unwrap(),
                ring_data: crate::output::ring_data::SectorData::AllSectors(compute.ring_data),
            }),
            dem.centre,
        )
        .unwrap();

        let viewsheder = Viewshed {
            dem: &dem,
            // TODO: It would be good to have a function that derives the DEM coord from the
            // lat/lon. Then we could use it here like: `latlon_to_dem_coord(dem.centre)`.
            pov_coord: crate::dem::Coordinate(geo::coord! {x: 4.0, y: 4.0}),
        };

        // Convert the viewshed coordinates to DEM-based. This helps give some intuition whilst
        // debugging because you can use the littel ASCII grids above.
        for point in viewshed.iter_mut() {
            point.exterior_mut(|line| {
                for coordinate in line.coords_mut() {
                    let projected = viewsheder
                        .convert_viewshed_coord_to_dem_coord(Coordinate(*coordinate))
                        .unwrap();
                    *coordinate = round_coordinate(projected);
                }
            });
        }

        assert_eq!(viewshed.0.len(), 1);

        let extent = geo::extremes::Outcome {
            x_min: geo::extremes::Extreme {
                index: 0,
                coord: (0.7816801, 1.2514028).into(),
            },
            y_min: geo::extremes::Extreme {
                index: 180,
                coord: (6.7485972, 0.7816834).into(),
            },
            x_max: geo::extremes::Extreme {
                index: 360,
                coord: (7.2183166, 6.7486005).into(),
            },
            y_max: geo::extremes::Extreme {
                index: 540,
                coord: (1.2513995, 7.2183199).into(),
            },
        };
        assert_eq!(viewshed.extremes().unwrap(), extent);
    }
}
