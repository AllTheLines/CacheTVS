//! Join the visible segments from each computed angle into the viewshed. These "common" joinings
//! are the ones the occur for every angle other than the final one. Therefore these should simpler
//! and faster.

use color_eyre::eyre::{Result, bail};

/// Keeps track of active and completed polygons within a viewshed.
#[derive(Default)]
pub struct Joiner {
    /// Polygons that don't intersect with the current angle. They don't need to be checked against
    /// new segments.
    pub completed: Vec<super::growable_polygon::GrowablePolygon>,
    /// Polygons that intersect with the current angle. They must be checked to see if any of their
    /// openings touch any of the segments of the current angle.
    pub active: Vec<super::growable_polygon::GrowablePolygon>,
}

impl Joiner {
    /// Build a viewshed of euclidean polygons from polar segments.
    pub fn build(
        data: &[Vec<crate::storage::segments::Segment>],
        dem_scale: f32,
    ) -> Result<geo::MultiPolygon> {
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

        let mut geo_polygons = Vec::new();
        for mut raw_polygon in joiner.completed {
            raw_polygon.dedup_vertices_ignore_openings();
            let polygon = raw_polygon.to_geo_polygon();
            tracing::trace!("Final viewshed, adding polygon: {polygon:?}");
            geo_polygons.push(polygon);
        }

        Ok(geo::MultiPolygon::new(geo_polygons))
    }

    /// Build the viewshed for a single angle.
    fn build_angle(
        &mut self,
        angle: f32,
        angle_scale: f32,
        segments: &[crate::storage::segments::Segment],
        dem_scale: f32,
    ) -> Result<()> {
        tracing::debug!("");
        tracing::debug!("Building viewshed for angle: {angle}");
        tracing::debug!(
            "Polygon counts, completed: {}, active: {}. Segments: {}",
            self.completed.len(),
            self.active.len(),
            segments.len()
        );

        self.prepare_active_polygons();

        let mut new_polygons = Vec::new();

        let polygoner = super::segment_polygon::SegmentPolygon {
            dem_scale,
            angle,
            angle_scale,
        };

        let timing = std::time::Instant::now();
        let last_segment_index = segments.len().saturating_sub(1);
        for (segment_index, polar_segment) in segments.iter().enumerate() {
            let is_last_segment = segment_index == last_segment_index;
            let start = u32::from(polar_segment.start());
            let end = u32::from(polar_segment.start() + polar_segment.distance());
            tracing::debug!("Segment {segment_index}, distances range: {:?}", start..end);
            let polygon_segment = polygoner.make(start, end);
            let mut is_segment_touching_anything = false;
            let mut maybe_joining_polygon_index = None;
            let mut joined_polygons_to_remove = Vec::new();

            for active_polygon_index in 0..self.active.len() {
                let touchging_timing = std::time::Instant::now();

                let is_segment_touching_this_polygon = self.join_segment(
                    active_polygon_index,
                    &polygon_segment,
                    maybe_joining_polygon_index,
                    angle,
                )?;

                #[expect(
                    clippy::indexing_slicing,
                    reason = "We're getting the indexes from `.len()`"
                )]
                let active_polygon = &mut self.active[active_polygon_index];

                if is_segment_touching_this_polygon {
                    active_polygon.is_touched = true;
                    is_segment_touching_anything = true;
                    if let Some(joining_polygon_index) = maybe_joining_polygon_index {
                        joined_polygons_to_remove.push(joining_polygon_index);
                    } else {
                        maybe_joining_polygon_index = Some(active_polygon_index);
                    }
                } else {
                    maybe_joining_polygon_index = None;
                }

                tracing::debug!(
                    "Segment checked against active {active_polygon_index} in {:?}",
                    touchging_timing.elapsed()
                );
            }

            tracing::debug!("Removing joined polygons: {joined_polygons_to_remove:?}");
            for joined_polygon_index in joined_polygons_to_remove {
                self.active.remove(joined_polygon_index);
            }

            if is_last_segment {
                for polygon in &mut self.active {
                    polygon.downgrade_openings()?;
                }
            }

            if !is_segment_touching_anything {
                new_polygons.push(Self::create_new_polygon_from_untouched_segment(
                    &polygon_segment,
                    angle,
                )?);
            }
        }
        tracing::debug!("Angle {angle} done in {:?}", timing.elapsed());

        self.move_untouched_active_polygons_to_completed();
        self.active.extend(new_polygons);

        Ok(())
    }

    /// When a segment doesn't touch anything it becomes its own independent polygon.
    fn create_new_polygon_from_untouched_segment(
        polygon_segment: &super::segment_polygon::Vertices,
        angle: f32,
    ) -> Result<super::growable_polygon::GrowablePolygon> {
        tracing::debug!(
            "Segment not touching anything at angle {angle}, \
             so making it its own polygon: {polygon_segment:?}"
        );
        let mut polygon = super::growable_polygon::GrowablePolygon::new(polygon_segment, angle);
        if angle == 0.0 {
            polygon.is_created_at_angle_0 = true;
        }
        polygon.downgrade_openings()?;

        Ok(polygon)
    }

    /// Set all active polygons to untouched and order them based on their openings' distance from
    /// the centre.
    pub(crate) fn prepare_active_polygons(&mut self) {
        self.active.sort_by_key(|polygon| polygon.furthest_opening);
        self.active.iter_mut().for_each(|active| {
            active.is_touched = false;
        });
    }

    /// If a polygon wasn't touched by a segment from the current angle then it is moved to `self.completed`.
    fn move_untouched_active_polygons_to_completed(&mut self) {
        let completed = self
            .active
            .extract_if(.., |polygon| !polygon.is_touched)
            .collect::<Vec<_>>();
        self.completed.extend(completed);
    }

    /// Once all angles have been checked, we can move all active polygons to "completed".
    fn move_all_active_polygons_to_completed(&mut self) {
        let active = self.active.drain(..);
        self.completed.extend(active);
    }

    /// This is the essential case for joining a new segment (B) to a single existing viewshed
    /// polygon (A):
    ///
    ///     ┌───────┐
    /// ┌───┐c      │
    /// │   │───┐   │
    /// │ B │ h │   │
    /// │   │───┘   │
    /// └───┘b   A  │
    ///     └───┐   │
    ///         │   │
    ///     ┌───┘   │
    ///     │a      │
    ///     └───────┘
    ///
    /// * Polygon A is an existing "active", complex [`GrowablePolygon`].
    /// * Polygon B is a new simple segment, represented by 5 vertices ([`SegmentPolygonVertices`]).
    /// * Polygon A has 1 or more "openings" (represented here as "a", "b" and "c"). If it has
    ///   more than 1 opening there will be holes, that need to be created separately. A hole-to-be
    ///   is represented here by "h".
    /// * Polygon B's vertices replace polygon A's vertices starting at the end of the first
    ///   touching opening and ending at the start of the last touching opening.
    /// * The replaced vertices are used to make holes.
    ///
    /// This is the result of the join:
    ///
    ///      ┌───────┐
    ///  ┌───┘       │
    ///  │   ┌───┐   │
    ///  │b  │ h │   │
    ///  │   └───┘   │
    ///  └───┐       │
    ///      └───┐   │
    ///          │   │
    ///      ┌───┘   │
    ///      │a      │
    ///      └───────┘
    ///
    /// * Only 1 polygon remains, therefore polygon A has grown.
    /// * A hole or holes ("h") have been created.
    /// * There is one less opening (just "a" and "b").
    fn join_segment(
        &mut self,
        growable_polygon_index: usize,
        segment: &super::segment_polygon::Vertices,
        maybe_joining_polygon_index: Option<usize>,
        angle: f32,
    ) -> Result<bool> {
        let mut maybe_touching_start = None;
        let mut maybe_touching_end = None;

        let (base_polygon, maybe_joining_polygon) =
            self.get_involved_polygons(growable_polygon_index, maybe_joining_polygon_index);

        tracing::trace!(
            "Polygon {growable_polygon_index} BEFORE: {:#?}",
            base_polygon
        );

        let mut iterator = base_polygon.vertices.iter_mut().enumerate().rev();
        while let Some((index, vertex)) = iterator.next() {
            match vertex.opening {
                super::growable_polygon::Opening::Start(opening_start) => {
                    let Some((index_next, vertex_next)) = iterator.next() else {
                        bail!("Opening without following vertex");
                    };

                    let super::growable_polygon::Opening::End(opening_end) = vertex_next.opening
                    else {
                        bail!("Opening start not followed by opening end");
                    };

                    let opening_range = opening_start..opening_end;

                    tracing::debug!(
                        "Checking touch for polygon {growable_polygon_index} at opening index: \
                         {index}, distances: {opening_range:?}",
                    );
                    if super::growable_polygon::GrowablePolygon::is_touching(
                        &segment.distances,
                        &opening_range,
                    ) {
                        tracing::debug!(
                            "🟢 Polygon {growable_polygon_index} touches segment opening"
                        );

                        if maybe_touching_start.is_none() {
                            maybe_touching_start = Some(index);
                        }
                        maybe_touching_end = Some(index_next + 1);
                    } else {
                        tracing::debug!(
                            "🟡 Polygon {growable_polygon_index} does not touch segment opening"
                        );
                    }
                }
                super::growable_polygon::Opening::End(_) => {
                    bail!("Unexpected opening end reached");
                }
                super::growable_polygon::Opening::Null
                | super::growable_polygon::Opening::GenesisStart(_)
                | super::growable_polygon::Opening::GenesisEnd(_)
                | super::growable_polygon::Opening::NewStart(_)
                | super::growable_polygon::Opening::NewEnd(_) => (),
            }
        }

        if let Some(touching_start) = maybe_touching_start
            && let Some(touching_end) = maybe_touching_end
        {
            let base_vertices_range = touching_end..touching_start;
            match maybe_joining_polygon {
                Some(joining_polygon) => {
                    base_polygon.join_non_starting_polygon(base_vertices_range, joining_polygon)?;
                    tracing::trace!(
                        "Polygon {growable_polygon_index} AFTER joined polygon: {:#?}",
                        base_polygon
                    );
                }
                None => {
                    base_polygon.join_segment(segment, base_vertices_range, angle)?;
                    tracing::trace!(
                        "Polygon {growable_polygon_index} AFTER joined segment: {:#?}",
                        base_polygon
                    );
                }
            }

            return Ok(true);
        }

        Ok(false)
    }

    /// Get the polygons involved in a touch check.
    ///
    /// It's in a function purely to hide the ugliness.
    fn get_involved_polygons(
        &mut self,
        growable_polygon_index: usize,
        maybe_joining_polygon_index: Option<usize>,
    ) -> (
        &mut super::growable_polygon::GrowablePolygon,
        Option<&mut super::growable_polygon::GrowablePolygon>,
    ) {
        if let Some(joining_polygon_index) = maybe_joining_polygon_index {
            #[expect(
                clippy::expect_used,
                reason = "An error here is a broken invariant. Better to panic."
            )]
            // This `disjoint` shenanigans is to get 2 mutable references at once, otherwise
            // the borrow checker complains.
            let [growable_polygon, joining_polygon] = self
                .active
                .get_disjoint_mut([growable_polygon_index, joining_polygon_index])
                .expect("Polygon indexes not found");
            (growable_polygon, Some(joining_polygon))
        } else {
            #[expect(
                clippy::indexing_slicing,
                reason = "We're getting the indexes from `.len()`"
            )]
            (&mut self.active[growable_polygon_index], None)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn multiple_angles() {
        crate::setup_logging().unwrap();
        let segment = vec![crate::storage::segments::Segment::new(0, 5)];
        let joined = Joiner::build(&[segment.clone(), segment.clone(), segment], 1.0).unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "█████████████████▀██████",
            "█████████████▀▀▄▄ ██████",
            "██████████▀▀▄████ ██████",
            "███████▀▄▄███████ ██████",
            "███▀▀▄▄██████████ ██████",
            "██▄▀▀████████████ ██████",
            "█████▄▄▀█████████ ██████",
            "████████▄▄▀▀█████ ██████",
            "████████████▄▀▀██ ██████",
            "███████████████▄▄ ██████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_segments() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 5)];
        let multiple = vec![
            crate::storage::segments::Segment::new(0, 2),
            crate::storage::segments::Segment::new(3, 1),
        ];
        let joined = Joiner::build(
            &[
                main.clone(),
                main.clone(),
                main.clone(),
                main.clone(),
                multiple,
                main,
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████ ▀▀█████████",
            "██████████▀▀▄██▄▄▀▀█████",
            "███▀▀█▀▀▄▄▀▀ ██████▄▄▀██",
            "███ █▄█▀ ▄▀▀▄████████ ██",
            "███ █████▄███████████ ██",
            "███ █████████████████ ██",
            "███ █████████████████ ██",
            "███ █████████████████ ██",
            "████▄▄▀▀█████████▀▀▄▄███",
            "████████▄▄▀▀█▀▀▄▄███████",
            "████████████▄███████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_polygons() {
        crate::setup_logging().unwrap();
        let segment = vec![crate::storage::segments::Segment::new(0, 5)];
        let joined = Joiner::build(
            &[
                segment.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                segment,
                vec![],
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "█████████▀▀▀▀▄ █████████",
            "██████▄ ▄████▀▄█████████",
            "████████ ████ ██████████",
            "█████████▄▀██ ████▀▀▄ ██",
            "███████████  █▀▀▄▄███ ██",
            "████████████▄▄▀▀█████ ██",
            "████████████████▄▄▀▀█ ██",
            "████████████████████▄▄██",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_polygons_not_touching() {
        crate::setup_logging().unwrap();
        let segment = vec![crate::storage::segments::Segment::new(2, 2)];
        let joined = Joiner::build(
            &[
                segment.clone(),
                vec![],
                vec![],
                vec![],
                vec![],
                segment,
                vec![],
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▀▀▀▄ █████████",
            "███████▄ ████▀▄█████████",
            "█████████▄▀▀▄▄████▀▀████",
            "████████████████ ▄█ ████",
            "████████████████ ██ ████",
            "████████████████▄▄▀ ████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn varying_sized_segments() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 4)];
        let variance = vec![crate::storage::segments::Segment::new(0, 2)];
        let joined = Joiner::build(
            &[main.clone(), main.clone(), main.clone(), main, variance],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀ █████████████",
            "████████ █▄▀██████▀█████",
            "██████▀▄███▄▄▀▀█▀▄ █████",
            "█████ █████████▄██ █████",
            "████▄▀████████████ █████",
            "██████ ███████████ █████",
            "███████▄▀████████▀ █████",
            "█████████ ██▀▀▄▄▄███████",
            "██████████▄▄████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn untouched_openings_get_closed() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 4)];
        let pair = vec![
            crate::storage::segments::Segment::new(0, 2),
            crate::storage::segments::Segment::new(3, 1),
        ];
        let bottom = vec![crate::storage::segments::Segment::new(0, 2)];
        let joined = Joiner::build(&[main.clone(), pair, bottom, main.clone(), main], 1.0).unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▄▄▀▀▀█████████",
            "████████ ██████▄▄▀▀█████",
            "██████▀▄██████████ █████",
            "█████ ████████████ █████",
            "████▄▄▄▄▄▀████████ █████",
            "██████████ █▀▀▄▄▀█ █████",
            "███████████▄▀▀▀▀▄  █████",
            "██████████ ▄▀▀▄▄▄███████",
            "██████████▄▄████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn centre_is_touched_after_more_than_one_angle() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 4)];
        let top = vec![crate::storage::segments::Segment::new(2, 2)];
        let joined = Joiner::build(
            &[
                main.clone(),
                top.clone(),
                top.clone(),
                top.clone(),
                top,
                main,
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "██████████▀▀▄▀▀█████████",
            "██████▀▀▄▄█████▄▄▀▀█████",
            "█████ ████▀▀ ██████ ████",
            "█████ ███ ██ ██████ ████",
            "█████ ███ ██▄▀▀████ ████",
            "█████ ███▄▀▀█▀▀▄███ ████",
            "█████▄▀▀████▄████▀▀▄████",
            "████████▄▄▀▀█▀▀▄▄███████",
            "████████████▄███████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn a_hole() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 4)];
        let lid = vec![crate::storage::segments::Segment::new(2, 1)];
        let joined =
            Joiner::build(&[main.clone(), main.clone(), lid, main.clone(), main], 1.0).unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████████▀▄▄▀▀▀█████████",
            "████████ ██████▄▄▀▀█████",
            "██████▀▄██████████ █████",
            "█████ ████████████ █████",
            "████▄▄▄▀▄ ▄▄ █████ █████",
            "███████▄▀▄▀ ██████ █████",
            "█████████ █▄█████▀ █████",
            "██████████ █▀▀▄▄▄███████",
            "██████████▄▄████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_holes() {
        crate::setup_logging().unwrap();
        let main = vec![crate::storage::segments::Segment::new(0, 6)];
        let struts = vec![
            crate::storage::segments::Segment::new(0, 1),
            crate::storage::segments::Segment::new(2, 1),
            crate::storage::segments::Segment::new(4, 1),
        ];
        let joined = Joiner::build(
            &[main.clone(), main.clone(), main.clone(), struts, main],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████▀▀██████████████",
            "████████▀▄▄▄▄▄▀▀▀███████",
            "███████▀▄▀ ██████▄▄▄▄▀▀█",
            "█████▀▄█▀ ▄███████████ █",
            "████▀▄▀▄▀▄▀ ██████████ █",
            "███ █▀▄ █▀ ▄██████████ █",
            "▄ ▄█▄▄▄█▄▄▄███████████ █",
            "█▄▀███████████████████ █",
            "███▄▀█████████████████ █",
            "████▄▀███████████████▀ █",
            "██████▄▀██████▀▀▀▄▄▄▄███",
            "███████▄▀▀▄▄▄▄██████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn segment_joins_to_multiple_polygons() {
        crate::setup_logging().unwrap();
        let polygons = vec![
            crate::storage::segments::Segment::new(2, 1),
            crate::storage::segments::Segment::new(4, 1),
        ];
        let long = vec![crate::storage::segments::Segment::new(0, 5)];
        let joined = Joiner::build(
            &[
                polygons.clone(),
                polygons.clone(),
                polygons.clone(),
                long,
                polygons,
                vec![],
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "██████████▀▀ ███████████",
            "██████▀▀▄▄▀▀▄███████████",
            "███▀▄▄▀▀▄▄▀▀ ████████▀██",
            "███ █▄▄▀▄▄▀▀▄████▀█ ▄ ██",
            "███ █████▄▀▀███ ▄ █ █ ██",
            "███ ██████▀▀▄██ █ █ █ ██",
            "███ ██▀▀█▄▀▀█▀▀▄█ █ █ ██",
            "███ █▄▀▀▄▄▀▀▄▀▀▄▄▀▀▄█ ██",
            "████▄▄▀▀▄▄▀▀▄▀▀▄▄▀▀▄▄███",
            "████████▄▄▀▀▄▀▀▄▄███████",
            "████████████▄███████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn join_in_distance_order() {
        crate::setup_logging().unwrap();

        // Even though this polygon is made first and so naturally appears first in the `self.active`
        // list of polygons, it is in fact _nearer_ the centre and so must be joined before newer but
        // higher polygons.
        let higher_but_made_first = crate::storage::segments::Segment::new(2, 2);

        let lower = crate::storage::segments::Segment::new(0, 1);
        let first = vec![higher_but_made_first.clone()];
        let second = vec![lower, higher_but_made_first];
        let long = vec![crate::storage::segments::Segment::new(0, 5)];
        let joined = Joiner::build(&[vec![], first, second, long, vec![]], 1.0).unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████▀ ██████████████",
            "███████▀▄█ █████████████",
            "█████▀▄███ █████████████",
            "████▀▄█████ ████████████",
            "███ ███████▄▀███████████",
            "██▄▄▄▀██▄ ▄▀ ███████████",
            "██████ ██▄▀▄█▀▀▄▀███████",
            "███████▄▀██▄▄████ ▀█████",
            "█████████ ██▀▀▄▄▄███████",
            "██████████▄▄████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }

    #[test]
    fn inherit_holes_common() {
        crate::setup_logging().unwrap();

        let joined = Joiner::build(
            &[
                vec![],
                vec![crate::storage::segments::Segment::new(0, 3)],
                vec![
                    crate::storage::segments::Segment::new(0, 1),
                    crate::storage::segments::Segment::new(2, 1),
                ],
                vec![
                    crate::storage::segments::Segment::new(0, 3),
                    crate::storage::segments::Segment::new(4, 1),
                ],
                vec![crate::storage::segments::Segment::new(0, 5)],
                vec![],
                vec![],
            ],
            1.0,
        )
        .unwrap();
        let actual = crate::output::ascii::rasterise(joined);
        let expected = [
            "████████████████████████",
            "████████████████████████",
            "█████▀▄▀████████████████",
            "████▀▄█▄▀███████████████",
            "███▀▄████▄▀█████████████",
            "██▀▄██████▄▀████████████",
            "██▄▀ ▄ █████▄▀▀█████████",
            "███▄▀ █ ██▀ ▄▀█▄▄▀██████",
            "████▄▀ █ ▀▀█▄▄▀▄▄███████",
            "█████▄▀ ███▄▄▄██████████",
            "██████▄█████████████████",
            "████████████████████████",
        ];
        crate::output::ascii::assert_rasterised(&actual, &expected);
    }
}
