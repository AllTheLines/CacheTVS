//! Join segments and polygons to existing growable polygons.

mod last;
mod normal;

/// Keeps track of active and completed polygons within a viewshed.
#[derive(Default)]
pub struct Joiner {
    /// Polygons that don't intersect with the current angle. They don't need to be checked against
    /// new segments.
    completed: Vec<super::growable_polygon::GrowablePolygon>,
    /// Polygons that intersect with the current angle. They must be checked to see if any of their
    /// openings touch any of the segments of the current angle.
    active: Vec<super::growable_polygon::GrowablePolygon>,
}

impl Joiner {
    /// Join segments into a collection of polygons.
    ///
    /// # Errors
    ///   If joining segments fails.
    #[inline]
    #[must_use]
    pub fn join(
        data: &[Vec<crate::segment::Segment>],
        dem_scale: f32,
    ) -> Vec<super::growable_polygon::GrowablePolygon> {
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "The angle count should never strain the f32 mantissa"
        )]
        let angle_count = data.len() as f32;
        let angle_scale = angle_count / 360.0;

        let mut joiner = Self {
            completed: Vec::new(),
            active: Vec::new(),
        };

        for (anglish, segments) in data.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "The angle count should never strain the f32 mantissa"
            )]
            let angle = anglish as f32 / angle_scale;

            joiner.build_angle(angle, angle_scale, segments, dem_scale);
        }

        joiner.build_final_angle();
        joiner.move_all_active_polygons_to_completed();
        for polygon in &mut joiner.completed {
            polygon.dedup_vertices_ignore_openings();
        }

        joiner.completed
    }
}

#[cfg(test)]
fn rasterise_multi_polygon(
    multi_polygon: Vec<crate::growable_polygon::GrowablePolygon>,
) -> Vec<String> {
    let width = 12u32;
    let centre = f64::from(width.div_euclid(2));

    let mut multi_polygons_geo = Vec::new();

    for polygon in multi_polygon {
        let mut line_exterior = Vec::new();
        for coordinate in polygon.to_polygon().exterior {
            let foo = geo::Coord {
                x: coordinate.x + centre,
                y: coordinate.y + centre,
            };
            line_exterior.push(foo);
        }

        let mut holes = Vec::new();
        for hole in polygon.to_polygon().interior {
            let mut line = Vec::new();
            for coordinate in hole {
                let foo = geo::Coord {
                    x: coordinate.x + centre,
                    y: coordinate.y + centre,
                };
                line.push(foo);
            }
            holes.push(geo::LineString::from(line));
        }

        let exterior = geo::LineString::from(line_exterior);
        multi_polygons_geo.push(geo::Polygon::new(exterior, holes));
    }

    tvs_lib::ascii::rasterise_multi_polygon_geo(&geo::MultiPolygon::new(multi_polygons_geo), width)
}
