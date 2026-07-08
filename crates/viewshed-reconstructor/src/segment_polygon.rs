//! Create euclidean polygons from polar segments.

/// The 5 vertices of a segment and their distances from the centre.
#[derive(Debug)]
pub(crate) struct Vertices {
    /// The 5 vertices of a segment. The 4 corners plus a copy of the first vertex to close the
    /// polygon.
    pub vertices: [crate::polygon::Coordinate; 5],
    /// The distances of the top and bottom of the segment from the centre.
    pub distances: std::ops::Range<u32>,
}

impl Vertices {
    /// Make a single polygon representing a visible region of the planet.
    pub(crate) fn new(segment: &SegmentPolygon, opening_index: u32, closing_index: u32) -> Self {
        let opening_coord = segment.index_to_coordinate(opening_index);
        let closing_coord = segment.index_to_coordinate(closing_index);

        let spread = 1.0f64 / f64::from(segment.angle_scale) / 2.0f64;
        let bottom_left = SegmentPolygon::rotate_by(opening_coord, spread);
        let bottom_right = SegmentPolygon::rotate_by(opening_coord, -spread);
        let top_left = SegmentPolygon::rotate_by(closing_coord, spread);
        let top_right = SegmentPolygon::rotate_by(closing_coord, -spread);

        let scale = f64::from(segment.dem_scale);

        let vertices = [
            bottom_left.scale(scale),
            bottom_right.scale(scale),
            top_right.scale(scale),
            top_left.scale(scale),
            bottom_left.scale(scale),
        ];

        Self {
            vertices,
            distances: opening_index..closing_index,
        }
    }
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
    /// Convert an index along a line of sight into a coordinate.
    fn index_to_coordinate(&self, index: u32) -> crate::polygon::Coordinate {
        let radians = self.angle.to_radians();
        let distance = f64::from(index);

        crate::polygon::Coordinate {
            x: distance * f64::from(radians.cos()),
            y: distance * f64::from(radians.sin()),
        }
    }

    /// Rotate a point about the centre of the viewshed.
    #[expect(
        clippy::suboptimal_flops,
        reason = "I think readability is more important?"
    )]
    fn rotate_by(point: crate::polygon::Coordinate, angle: f64) -> crate::polygon::Coordinate {
        let dx = point.x;
        let dy = point.y;
        let cos = angle.to_radians().cos();
        let sin = angle.to_radians().sin();
        crate::polygon::Coordinate {
            x: dx * cos - dy * sin,
            y: dx * sin + dy * cos,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn builder(angle: f32) -> SegmentPolygon {
        SegmentPolygon {
            dem_scale: 1.0,
            angle,
            angle_scale: 1.0,
        }
    }

    #[derive(Debug)]
    struct VisiblePolygonFor {
        angle: f32,
        opening_index: u32,
        closing_index: u32,
    }

    fn make_visible_polygon_for(setup: &VisiblePolygonFor) -> Vec<crate::polygon::Coordinate> {
        let segment = builder(setup.angle);
        let vertices =
            super::Vertices::new(&segment, setup.opening_index, setup.closing_index).vertices;

        let mut polygon_as_dem_coords = Vec::new();
        for vertex in &vertices {
            let dem_coord = crate::polygon::Coordinate {
                x: vertex.x + 4.0,
                y: -vertex.y + 4.0,
            };
            polygon_as_dem_coords.push(round_coordinate(dem_coord));
        }
        polygon_as_dem_coords
    }

    pub(crate) fn coord(x: f64, y: f64) -> crate::polygon::Coordinate {
        crate::polygon::Coordinate { x, y }
    }

    fn round(float: f64) -> f64 {
        let factor = 10f64.powi(7);
        (float * factor).round() / factor
    }

    fn round_coordinate(coordinate: crate::polygon::Coordinate) -> crate::polygon::Coordinate {
        crate::polygon::Coordinate {
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

        const ANGLE: f32 = 45.0;

        // The polygon we're making is `abcd` from the above guide.
        #[expect(clippy::unreadable_literal, reason = "It's just a test")]
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    angle: ANGLE,
                    opening_index: 1,
                    closing_index: 2,
                }),
                vec![
                    coord(4.7009093, 3.2867496),
                    coord(4.7132504, 3.2990907),
                    coord(5.4265009, 2.5981815),
                    coord(5.4018185, 2.5734991),
                    coord(4.7009093, 3.2867496),
                ]
                .into_iter()
                .collect::<Vec<crate::polygon::Coordinate>>()
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
    // 5  .  .  .  .  o  .a .  .  .
    // 6  .  .  .  .  .  (  .d .  .
    // 7  .  .  .  .  . b.  )  .  .
    // 8  .  .  .  .  .  . c.  .  .
    // 4  .  .  .  .  .  .  .  .  .
    //
    mod from_bottom_left_to_bottom_right {
        use super::*;

        const ANGLE: f32 = 135.0 + 180.0;

        // The polygon we're making is `abcd` from the above guide.
        #[expect(clippy::unreadable_literal, reason = "It's just a test")]
        #[test]
        fn making_a_visible_polygon() {
            assert_eq!(
                make_visible_polygon_for(&VisiblePolygonFor {
                    angle: ANGLE,
                    opening_index: 1,
                    closing_index: 2,
                }),
                vec![
                    coord(4.7132503, 4.7009094),
                    coord(4.7009091, 4.7132506),
                    coord(5.4018183, 5.4265011),
                    coord(5.4265006, 5.4018187),
                    coord(4.7132503, 4.7009094),
                ]
                .into_iter()
                .collect::<Vec<crate::polygon::Coordinate>>()
            );
        }
    }
}
