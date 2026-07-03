//! Create euclidean polygons from polar segments.

/// The 5 vertices of a segment and their distances from the centre.
#[derive(Debug)]
pub(crate) struct Vertices {
    /// The 5 vertices of a segment. The 4 corners plus a copy of the first vertex to close the
    /// polygon.
    pub vertices: [geo::Coord; 5],
    /// The distances of the top and bottom of the segment from the centre.
    pub distances: std::ops::Range<u32>,
}

/// `SegmentPolygon`
pub(crate) struct SegmentPolygon {
    /// Scale of DEM data.
    pub dem_scale: f32,
    /// The current sector angle.
    pub angle: f32,
    /// The number of angles per degree.
    pub angle_scale: f32,
}

impl SegmentPolygon {
    /// Make a single polygon representing a visible region of the planet.
    pub(crate) fn make(&self, opening_index: u32, closing_index: u32) -> Vertices {
        let opening_coord = self.index_to_coordinate(opening_index);
        let closing_coord = self.index_to_coordinate(closing_index);

        let spread = 1.0f64 / f64::from(self.angle_scale) / 2.0f64;
        let bottom_left = Self::rotate_by(opening_coord, spread);
        let bottom_right = Self::rotate_by(opening_coord, -spread);
        let top_left = Self::rotate_by(closing_coord, spread);
        let top_right = Self::rotate_by(closing_coord, -spread);

        let scale = f64::from(self.dem_scale);

        let vertices = [
            bottom_left * scale,
            bottom_right * scale,
            top_right * scale,
            top_left * scale,
            bottom_left * scale,
        ];

        Vertices {
            vertices,
            distances: opening_index..closing_index,
        }
    }

    /// Convert an index along a line of sight into a coordinate.
    fn index_to_coordinate(&self, index: u32) -> super::viewshed::Coordinate {
        let radians = self.angle.to_radians();
        let distance = f64::from(index);

        super::viewshed::Coordinate(geo::coord! {
            x: distance * f64::from(radians.cos()),
            y: distance * f64::from(radians.sin())
        })
    }

    /// Rotate a point about the centre of the viewshed.
    #[expect(
        clippy::suboptimal_flops,
        reason = "I think readability is more important?"
    )]
    fn rotate_by(point: super::viewshed::Coordinate, angle: f64) -> geo::Coord {
        let dx = point.0.x;
        let dy = point.0.y;
        let cos = angle.to_radians().cos();
        let sin = angle.to_radians().sin();
        geo::coord! {
            x: dx * cos - dy * sin,
            y: dx * sin + dy * cos
        }
    }
}

#[cfg(test)]
mod test {
    fn builder(
        viewshed: &crate::output::viewsheds::viewshed::Viewshed,
        angle: f32,
    ) -> crate::output::viewsheds::segment_polygon::SegmentPolygon {
        crate::output::viewsheds::segment_polygon::SegmentPolygon {
            dem_scale: viewshed.dem.scale,
            angle,
            angle_scale: 1.0,
        }
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
        let viewshed = crate::output::viewsheds::viewshed::Viewshed {
            dem: &dem,
            pov_coord: tvs_lib::dem::Coordinate(setup.pov),
        };
        let viewsheder = builder(&viewshed, setup.angle);
        let vertices = viewsheder
            .make(setup.opening_index, setup.closing_index)
            .vertices;
        let polygon = geo::Polygon::new(geo::LineString(vertices.into()), vec![]);

        let mut polygon_as_dem_coords = Vec::new();
        for coord in &polygon.exterior().0 {
            let converted_coord = viewshed
                .convert_viewshed_coord_to_dem_coord(
                    crate::output::viewsheds::viewshed::Coordinate(*coord),
                )
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
                    (4.7009079, 3.2867515),
                    (4.7132491, 3.2990927),
                    (5.4264998, 2.5981837),
                    (5.4018174, 2.5735014),
                    (4.7009079, 3.2867515)
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
                    (3.7132486, 5.7009105),
                    (3.7009074, 5.7132517),
                    (4.4018163, 6.4265025),
                    (4.4264987, 6.4018201),
                    (3.7132486, 5.7009105)
                ]
                .into_iter()
                .map(Into::into)
                .collect::<Vec<geo::Coord>>()
            );
        }
    }
}
