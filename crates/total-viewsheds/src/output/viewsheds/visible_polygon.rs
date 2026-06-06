//! Create euclidean polygons from polar segments.

/// `VisiblePolygon`
pub struct VisiblePolygon {
    /// Scale of DEM data.
    pub scale: f32,
    /// The current sector angle.
    pub current_angle: f32,
}

impl VisiblePolygon {
    /// Convert an index along a line of sight into a coordinate.
    fn index_to_coordinate(&self, index: u32) -> super::viewshed::Coordinate {
        let radians = self.current_angle.to_radians();
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

    /// Make a single polygon representing a visible region of the planet.
    pub fn make_visible_polygon(&self, opening_index: u32, closing_index: u32) -> geo::Polygon {
        let opening_coord = self.index_to_coordinate(opening_index);
        let closing_coord = self.index_to_coordinate(closing_index);

        let spread = 0.5001f64;
        let bottom_left = Self::rotate_by(opening_coord, spread);
        let bottom_right = Self::rotate_by(opening_coord, -spread);
        let top_left = Self::rotate_by(closing_coord, spread);
        let top_right = Self::rotate_by(closing_coord, -spread);

        let scale = f64::from(self.scale);

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
}

#[cfg(test)]
mod test {
    fn builder(
        viewshed: &crate::output::viewsheds::viewshed::Viewshed,
        angle: f32,
    ) -> crate::output::viewsheds::visible_polygon::VisiblePolygon {
        crate::output::viewsheds::visible_polygon::VisiblePolygon {
            scale: viewshed.dem.scale,
            current_angle: angle,
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
        let polygon = viewsheder.make_visible_polygon(setup.opening_index, setup.closing_index);

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
}
