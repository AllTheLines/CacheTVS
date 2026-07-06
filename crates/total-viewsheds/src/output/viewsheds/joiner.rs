//! Join segments and polygons to existing growable polygons.

mod last;
mod normal;

use color_eyre::Result;

/// Keeps track of active and completed polygons within a viewshed.
#[derive(Default)]
struct Joiner {
    /// Polygons that don't intersect with the current angle. They don't need to be checked against
    /// new segments.
    completed: Vec<super::growable_polygon::GrowablePolygon>,
    /// Polygons that intersect with the current angle. They must be checked to see if any of their
    /// openings touch any of the segments of the current angle.
    active: Vec<super::growable_polygon::GrowablePolygon>,
}

impl Joiner {
    /// Join segments into a collection of polygons.
    fn join(
        data: &[Vec<crate::storage::segments::Segment>],
        dem_scale: f32,
    ) -> Result<Vec<super::growable_polygon::GrowablePolygon>> {
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

            joiner.build_angle(angle, angle_scale, segments, dem_scale)?;
        }

        joiner.build_final_angle()?;
        joiner.move_all_active_polygons_to_completed();

        Ok(joiner.completed)
    }
}

/// Build a viewshed of euclidean polygons from polar segments.
pub(crate) fn build_viewshed_polygon(
    data: &[Vec<crate::storage::segments::Segment>],
    dem_scale: f32,
) -> Result<geo::MultiPolygon> {
    let raw_polygons = Joiner::join(data, dem_scale)?;
    let mut geo_polygons = Vec::new();
    for mut raw_polygon in raw_polygons {
        raw_polygon.dedup_vertices_ignore_openings();
        let polygon = raw_polygon.to_geo_polygon();
        tracing::trace!("Final viewshed, adding polygon: {polygon:?}");
        geo_polygons.push(polygon);
    }

    Ok(geo::MultiPolygon::new(geo_polygons))
}
