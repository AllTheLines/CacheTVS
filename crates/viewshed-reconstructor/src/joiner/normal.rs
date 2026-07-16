//! Join the visible segments from each computed angle into the viewshed. These "normal" joinings
//! are the ones that occur for every angle other than the final one. Therefore these should be
//! simpler and faster.

impl super::Joiner {
    /// Build the viewshed for a single angle.
    pub(crate) fn build_angle(
        &mut self,
        angle: f32,
        angle_scale: f32,
        segments: &[crate::segment::Segment],
        dem_scale: f32,
    ) -> bool {
        tracing::debug!("");
        tracing::debug!("🏳️ Building viewshed for angle: {angle}");
        tracing::debug!(
            "Polygon counts, completed: {}, active: {}. Segments: {}",
            self.completed.len(),
            self.active.len(),
            segments.len()
        );

        self.set_all_active_polygons_to_untouched();

        let mut new_polygons = Vec::new();

        let segment = crate::segment_polygon::SegmentPolygon {
            dem_scale,
            angle,
            angle_scale,
        };

        for (segment_index, polar_segment) in segments.iter().enumerate() {
            let start = u32::from(polar_segment.start());
            let end = u32::from(polar_segment.start() + polar_segment.distance());
            let polygon_segment = crate::segment_polygon::Vertices::new(&segment, start, end);
            let mut is_segment_touching_anything = false;
            let mut maybe_joining_polygon_index = None;
            let mut joined_polygons_to_remove = Vec::new();
            let mut running_highest = 0;
            tracing::debug!("Segment {segment_index}, distances range: {:?}", start..end);

            self.order_active_polygons_by_distance_from_centre();

            for active_polygon_index in 0..self.active.len() {
                if self.is_polygon_surrounded(active_polygon_index, running_highest) {
                    // TODO: This is a hack. Thankfully it's rare, so I've never actually seen a
                    // touching but unjoined polygon in the wild. Without the hack though what you
                    // do see is lots of unsightly artifacting eminating from the affected polygon.
                    tracing::warn!(
                        "Polygon {active_polygon_index} falls inside previous polygon, skipping join checks."
                    );
                    continue;
                }

                let Some(is_segment_touching_this_polygon) = self.join_segment(
                    active_polygon_index,
                    &polygon_segment,
                    maybe_joining_polygon_index,
                    angle,
                ) else {
                    self.remove_all_polygons_apart_from_bad_ones(
                        active_polygon_index,
                        maybe_joining_polygon_index,
                    );
                    return false;
                };

                let polygon_highest = self
                    .get_active_polygon(active_polygon_index)
                    .lowest_openings
                    .iter()
                    .max()
                    .expect("Polygon without any openings");
                running_highest = *polygon_highest;

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
                    }
                    maybe_joining_polygon_index = Some(active_polygon_index);
                } else {
                    maybe_joining_polygon_index = None;
                }
            }

            tracing::debug!("Removing joined polygons: {joined_polygons_to_remove:?}");
            joined_polygons_to_remove.dedup();
            for joined_polygon_index in joined_polygons_to_remove.iter().rev() {
                self.active.remove(*joined_polygon_index);
            }

            if !is_segment_touching_anything {
                new_polygons.push(Self::create_new_polygon_from_untouched_segment(
                    &polygon_segment,
                    angle,
                ));
            }
        }

        for polygon in &mut self.active {
            polygon.downgrade_openings();
        }

        self.move_untouched_active_polygons_to_completed();
        self.active.extend(new_polygons);

        true
    }

    /// Get a reference to an active polygon.
    fn get_active_polygon(&self, index: usize) -> &crate::growable_polygon::GrowablePolygon {
        #[expect(
            clippy::indexing_slicing,
            reason = "We're getting the indexes from `.len()`"
        )]
        &self.active[index]
    }

    /// Check for an edge case where a polygon is between 2 openings of another polygon.
    fn is_polygon_surrounded(&self, index: usize, running_highest: u32) -> bool {
        let polygon_lowest = self
            .get_active_polygon(index)
            .lowest_openings
            .iter()
            .min()
            .expect("Polygon without any openings");
        polygon_lowest < &running_highest
    }

    /// When a segment doesn't touch anything it becomes its own independent polygon.
    fn create_new_polygon_from_untouched_segment(
        polygon_segment: &crate::segment_polygon::Vertices,
        angle: f32,
    ) -> crate::growable_polygon::GrowablePolygon {
        tracing::debug!(
            "Segment not touching anything at angle {angle}, \
             so making it its own polygon: {polygon_segment:?}"
        );
        let mut polygon = crate::growable_polygon::GrowablePolygon::new(polygon_segment, angle);
        if angle == 0.0 {
            polygon.is_created_at_angle_0 = true;
        }
        polygon.downgrade_openings();

        polygon
    }

    /// Set all active polygons to untouched.
    pub(crate) fn set_all_active_polygons_to_untouched(&mut self) {
        self.active.iter_mut().for_each(|active| {
            active.is_touched = false;
        });
    }

    /// Order active polygons by their lowest opening's distance from the centre.
    pub(crate) fn order_active_polygons_by_distance_from_centre(&mut self) {
        self.active.sort_by_key(|polygon| {
            *polygon
                .lowest_openings
                .iter()
                .min()
                .expect("Active polygon without any openings")
        });
    }

    /// If a polygon wasn't touched by a segment from the current angle then it is moved to `self.completed`.
    fn move_untouched_active_polygons_to_completed(&mut self) {
        let completed = self.active.extract_if(.., |polygon| !polygon.is_touched);
        self.completed.extend(completed);
    }

    /// Once all angles have been checked, we can move all active polygons to "completed".
    pub(crate) fn move_all_active_polygons_to_completed(&mut self) {
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
        segment: &crate::segment_polygon::Vertices,
        maybe_joining_polygon_index: Option<usize>,
        angle: f32,
    ) -> Option<bool> {
        let (base_polygon, maybe_joining_polygon) =
            self.get_involved_polygons(growable_polygon_index, maybe_joining_polygon_index);

        tracing::trace!(
            "Polygon {growable_polygon_index} BEFORE: {:#?}",
            base_polygon
        );

        tracing::debug!(
            "Polygon {growable_polygon_index} ({})",
            base_polygon.vertices.len()
        );

        let Some(base_vertices_range) =
            super::super::vertices::find_contact(&base_polygon.vertices, &segment.distances)
        else {
            return Some(false);
        };

        match maybe_joining_polygon {
            Some(joining_polygon) => {
                base_polygon
                    .join_non_starting_polygon(base_vertices_range.clone(), joining_polygon);
                tracing::trace!(
                    "Polygon {growable_polygon_index} AFTER joined polygon: {:#?}",
                    base_polygon
                );

                #[cfg(debug_assertions)]
                {
                    let base_polygon_original = base_polygon.clone();
                    if base_polygon.self_intersects() {
                        tracing::debug!(
                            "Base polygon ({}) before join: {:#?}",
                            base_polygon_original.vertices.len(),
                            base_polygon_original
                        );
                        tracing::warn!(
                            "Possibly bad base vertices range: {:#?}",
                            base_vertices_range
                        );
                        tracing::warn!(
                            "Possibly bad joining_vertices ({}): {:#?}",
                            joining_polygon.vertices.len(),
                            joining_polygon
                        );
                        tracing::warn!(
                            "Bad base polygon ({}) after join: {:#?}",
                            base_polygon.vertices.len(),
                            base_polygon
                        );
                        *base_polygon = base_polygon_original;
                        return None;
                    }
                }
            }
            None => {
                base_polygon.join_segment(segment, base_vertices_range, angle);
                tracing::trace!(
                    "Polygon {growable_polygon_index} AFTER joined segment: {:#?}",
                    base_polygon
                );
            }
        }

        Some(true)
    }

    /// Get the polygons involved in a touch check.
    ///
    /// It's in a function purely to hide the ugliness.
    fn get_involved_polygons(
        &mut self,
        growable_polygon_index: usize,
        maybe_joining_polygon_index: Option<usize>,
    ) -> (
        &mut crate::growable_polygon::GrowablePolygon,
        Option<&mut crate::growable_polygon::GrowablePolygon>,
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

    /// When an invalid polygon is detected we want to isolate them for rendering.
    fn remove_all_polygons_apart_from_bad_ones(
        &mut self,
        active_polygon_index: usize,
        maybe_joining_polygon_index: Option<usize>,
    ) {
        self.completed.drain(..);
        let base = self.get_active_polygon(active_polygon_index).clone();
        self.completed.push(base);

        if let Some(joining_polygon_index) = maybe_joining_polygon_index {
            let joining = self.get_active_polygon(joining_polygon_index).clone();
            self.active.drain(..);
            self.completed.push(joining);
        } else {
            self.active.drain(..);
        }
    }
}

#[cfg(test)]
mod test {
    use tvs_lib::ascii::assert_rasterised;

    use crate::joiner::{Joiner, rasterise_multi_polygon};

    #[test]
    fn multiple_angles() {
        crate::setup_logging();
        let segment = vec![crate::segment::Segment::new(0, 5)];
        let joined = Joiner::join(&[segment.clone(), segment.clone(), segment], 1.0);
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_segments() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 5)];
        let multiple = vec![
            crate::segment::Segment::new(0, 2),
            crate::segment::Segment::new(3, 1),
        ];
        let joined = Joiner::join(
            &[
                main.clone(),
                main.clone(),
                main.clone(),
                main.clone(),
                multiple,
                main,
            ],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_polygons() {
        crate::setup_logging();
        let segment = vec![crate::segment::Segment::new(0, 5)];
        let joined = Joiner::join(
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
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_polygons_not_touching() {
        crate::setup_logging();
        let segment = vec![crate::segment::Segment::new(2, 2)];
        let joined = Joiner::join(
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
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn varying_sized_segments() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 4)];
        let variance = vec![crate::segment::Segment::new(0, 2)];
        let joined = Joiner::join(
            &[main.clone(), main.clone(), main.clone(), main, variance],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn untouched_openings_get_closed() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 4)];
        let pair = vec![
            crate::segment::Segment::new(0, 2),
            crate::segment::Segment::new(3, 1),
        ];
        let bottom = vec![crate::segment::Segment::new(0, 2)];
        let joined = Joiner::join(&[main.clone(), pair, bottom, main.clone(), main], 1.0);
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn centre_is_touched_after_more_than_one_angle() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 4)];
        let top = vec![crate::segment::Segment::new(2, 2)];
        let joined = Joiner::join(
            &[
                main.clone(),
                top.clone(),
                top.clone(),
                top.clone(),
                top,
                main,
            ],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn a_hole() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 4)];
        let lid = vec![crate::segment::Segment::new(2, 1)];
        let joined = Joiner::join(&[main.clone(), main.clone(), lid, main.clone(), main], 1.0);
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn multiple_holes() {
        crate::setup_logging();
        let main = vec![crate::segment::Segment::new(0, 6)];
        let struts = vec![
            crate::segment::Segment::new(0, 1),
            crate::segment::Segment::new(2, 1),
            crate::segment::Segment::new(4, 1),
        ];
        let joined = Joiner::join(
            &[main.clone(), main.clone(), main.clone(), struts, main],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn segment_joins_to_multiple_polygons() {
        crate::setup_logging();
        let polygons = vec![
            crate::segment::Segment::new(2, 1),
            crate::segment::Segment::new(4, 1),
        ];
        let long = vec![crate::segment::Segment::new(0, 5)];
        let joined = Joiner::join(
            &[
                polygons.clone(),
                polygons.clone(),
                polygons.clone(),
                long,
                polygons,
                vec![],
            ],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn surrounded_polygon() {
        crate::setup_logging();
        let surroundings = vec![
            crate::segment::Segment::new(0, 1),
            crate::segment::Segment::new(4, 1),
        ];
        let with_inner = vec![
            crate::segment::Segment::new(0, 1),
            crate::segment::Segment::new(2, 1),
            crate::segment::Segment::new(4, 1),
        ];
        let long = vec![crate::segment::Segment::new(0, 5)];
        let joined = Joiner::join(
            &[
                long.clone(),
                surroundings.clone(),
                surroundings,
                with_inner.clone(),
                with_inner,
                long,
                vec![],
            ],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
        // Strictly speaking this is incorrect as it doesn't join. But joining would be very
        // complicated. At least it is not distorted and doesn't generate self-intersections.
        let expected = [
            "████████████████████████",
            "█████████▀▀▀▀▄ █████████",
            "█████▀▄ ▄████▀▄█████████",
            "████▀▄█  ████ ██████████",
            "███▀▄█ ▀▄▄▀██ ████▀▀▄ ██",
            "██▀▄█ ▀▄█ ▀▄ █▀▀▄▄███ ██",
            "██▄▀ █▄▄▄█▄▀▄ ▀▀█████ ██",
            "███▄▀ █████▄▄███▄▄▀▀█ ██",
            "████▄▀ ██████████▀▄█▀▄██",
            "█████▄▀▄▄▀▀▀▀██▀▄▀▀▄████",
            "██████▄▄▀▀▀▀█▄▄▀▄███████",
            "████████████▄▄▄█████████",
        ];
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn join_in_distance_order() {
        crate::setup_logging();

        // Even though this polygon is made first and so naturally appears first in the `self.active`
        // list of polygons, it is in fact _nearer_ the centre and so must be joined before newer but
        // higher polygons.
        let higher_but_made_first = crate::segment::Segment::new(2, 2);

        let lower = crate::segment::Segment::new(0, 1);
        let first = vec![higher_but_made_first.clone()];
        let second = vec![lower, higher_but_made_first];
        let long = vec![crate::segment::Segment::new(0, 5)];
        let joined = Joiner::join(&[vec![], first, second, long, vec![]], 1.0);
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }

    #[test]
    fn inherit_holes_common() {
        crate::setup_logging();

        let joined = Joiner::join(
            &[
                vec![],
                vec![crate::segment::Segment::new(0, 3)],
                vec![
                    crate::segment::Segment::new(0, 1),
                    crate::segment::Segment::new(2, 1),
                ],
                vec![
                    crate::segment::Segment::new(0, 3),
                    crate::segment::Segment::new(4, 1),
                ],
                vec![crate::segment::Segment::new(0, 5)],
                vec![],
                vec![],
            ],
            1.0,
        );
        let actual = rasterise_multi_polygon(joined);
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
        assert_rasterised(&actual, &expected);
    }
}
