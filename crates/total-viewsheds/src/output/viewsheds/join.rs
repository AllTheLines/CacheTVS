//! Join the visible segments from each computed angle into the viewshed.

impl super::visible_polygon::VisiblePolygon {
    /// Convert polar segments to euclidean polygons.
    pub fn parse_polar_segments(
        data: &[Vec<crate::storage::segments::Segment>],
        scale: f32,
    ) -> geo::MultiPolygon {
        let angle_count = data.len();
        let mut polygons = Vec::new();
        for (anglish, segments) in data.iter().enumerate() {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "The angle count should never strain the f32 mantissa"
            )]
            let (anglish_f32, angle_count_f32) = { (anglish as f32, angle_count as f32) };
            let polygoner = Self {
                scale,
                current_angle: (anglish_f32 / angle_count_f32) * 360.0,
            };
            for segment in segments {
                let opening = u32::from(segment.start());
                let closing = u32::from(segment.start() + segment.distance());
                let polygon = polygoner.make_visible_polygon(opening, closing);
                polygons.push(polygon);
            }
        }

        geo::unary_union(polygons.iter())
    }
}
