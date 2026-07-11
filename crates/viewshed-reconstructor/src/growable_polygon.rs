//! Growable polygons are polygons constructed one angle at a time from polar segments.

use crate::vertices::Opening;

/// A polygon that grows, but only from its anti-clockwise facing side.
#[derive(Debug, Clone)]
pub struct GrowablePolygon {
    /// The main exterior vertices.
    pub vertices: Vec<super::vertices::Vertex>,
    /// Interiore holes within the polygon.
    pub holes: Vec<Vec<super::vertices::Vertex>>,
    /// The distance of the opening furthest from the centre of the viewshed. Used for ordering
    /// active polygons. Polygons must be joined to new segments (and other polygons) in increasing
    /// order to prevent overlaps.
    pub furthest_opening: u32,
    /// Has the polygon been touched by anything in the current angle? If not, then it is marked as
    /// an isolated polygon that doesn't need to be checked for touches on subsequent angles.
    pub is_touched: bool,
    /// Was the polygon created at 0 degrees? This is needed because the polygon may need to be
    /// completed by joining another polygon that wraps into it at 360 degrees.
    pub is_created_at_angle_0: bool,
}

impl GrowablePolygon {
    /// Create a new growable polygon. It always begins with a single polar segment converted to
    /// its euclidean coordinates.
    pub(crate) fn new(segment_vertices: &super::segment_polygon::Vertices, angle: f32) -> Self {
        let mut vertices = Vec::new();
        for (segment_index, segment_vertex) in segment_vertices.vertices.iter().enumerate() {
            let opening = match segment_index {
                1 => {
                    if angle == 0.0 {
                        Opening::GenesisStart(segment_vertices.distances.start)
                    } else {
                        Opening::Null
                    }
                }
                2 => {
                    if angle == 0.0 {
                        Opening::GenesisEnd(segment_vertices.distances.end)
                    } else {
                        Opening::Null
                    }
                }
                3 => Opening::NewEnd(segment_vertices.distances.end),
                4 => Opening::NewStart(segment_vertices.distances.start),
                _ => Opening::Null,
            };
            vertices.push(super::vertices::Vertex {
                coordinate: *segment_vertex,
                opening: opening.clone(),
            });
        }

        Self {
            vertices,
            holes: Vec::new(),
            furthest_opening: segment_vertices.distances.start,
            is_touched: false,
            is_created_at_angle_0: false,
        }
    }

    /// Create a new polygon ready to be inserted into an existing polygon.
    ///
    /// Note that we wind our polygons anti-clockwise. The normal new polygon is on the left with
    /// indices (abcde). And the inserted polygon is on the right with vertices (cdeb). Note how the
    /// inserted polygon can't wind in the same way. It's been opened up into a straight line. The
    /// break happens between "c" and "b" such that they then become the ends of the line.
    ///
    ///              z┌──┐y       z┌──┐y
    ///   d┌──┐c      │  │      d┌─┘c │
    ///    │  │   +   │  │   =   │    │
    /// a/e└──┘b      │  │      e└─┐b │
    ///              w└──┘x       w└──┘x
    ///
    ///  (abcde)  +  (wxyz) =  (wxyzcdeb)
    fn new_for_insertion(segment_vertices: &super::segment_polygon::Vertices, angle: f32) -> Self {
        let mut polygon = Self::new(segment_vertices, angle);

        #[expect(
            clippy::indexing_slicing,
            reason = "An new polygon MUST have 5 vertices by definition"
        )]
        let second = { polygon.vertices[1].clone() };

        // Removing the first vertex doesn't destroy any data, because the first and last vertices
        // are duplicates.
        polygon.vertices.remove(0);
        // Removing the second vertex also doesn't destroy any data because we copied it earlier.
        polygon.vertices.remove(0);

        polygon.vertices.push(second);
        polygon
    }

    /// Once all the segments for an angle have been checked we convert the openings in the new
    /// polygon as follows:
    ///   * `Opening::NewStart/NewEnd` become `Opening::Start/End`.
    ///   * `Opening::Start/End` become `Opening::Null`.
    pub(crate) fn downgrade_openings(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Downgrading openings");
        let mut iterator = self.vertices.iter_mut().rev();
        while let Some(vertex) = iterator.next() {
            match vertex.opening {
                Opening::Null | Opening::GenesisStart(_) | Opening::GenesisEnd(_) => {}
                Opening::End(_) => {
                    panic!("Dangling `Opening`");
                }
                Opening::Start(_) => {
                    let Some(next) = iterator.next() else {
                        panic!("`Opening::Start` without adjacent end");
                    };

                    vertex.opening = Opening::Null;
                    next.opening = Opening::Null;
                }
                Opening::NewStart(at) => {
                    vertex.opening = Opening::Start(at);
                }
                Opening::NewEnd(at) => {
                    vertex.opening = Opening::End(at);
                }
            }
        }
    }

    /// Insert new vertices into the polygon. Can be either a segment or an entire other growable
    /// polygon that has _just_ been joined to the segment. Note that even though we could be
    /// joinging a whole new polygon, it will still only ever be limited to the known case of the
    /// freshly joined _segment_ touching this polygon. Therefore, we can assume things like, no
    /// holes will ever be created on the _incoming_ polygon because only the segment section of it
    /// is touching this currently instantiated polygon.
    fn join(
        &mut self,
        base_vertices_range: std::ops::Range<usize>,
        joining_vertices: Vec<super::vertices::Vertex>,
    ) {
        let old_start = self.extract_old_start(base_vertices_range.end);

        let removed_vertices: Vec<super::vertices::Vertex> = self
            .vertices
            .splice(base_vertices_range.clone(), joining_vertices)
            .collect();

        self.vertices
            .get_mut(base_vertices_range.start)
            .expect("Couldn't get new opening start index")
            .opening = old_start;

        self.dedup_vertices_but_keep_openings();
        self.create_holes(&removed_vertices);
    }

    /// Insert a segment into the polygon.
    pub(crate) fn join_segment(
        &mut self,
        segment_vertices: &super::segment_polygon::Vertices,
        vertices_range: std::ops::Range<usize>,
        angle: f32,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!(
            "Splicing new segment ({:?}) at: {vertices_range:?}",
            segment_vertices.distances
        );

        let segment = Self::new_for_insertion(segment_vertices, angle);
        self.join(vertices_range, segment.vertices);

        if segment.furthest_opening < self.furthest_opening {
            self.furthest_opening = segment.furthest_opening;
        }
    }

    /// Insert another polygon into the `self` polygon, where `self` isn't the final polygon.
    ///
    /// Consider this situation. "A" and "B" are _existing_ active polygons. "s" is an incoming
    /// segment. "s" joins to polygon "A" creating a new polygon "A+s". This new polygon then has to
    /// also connect to polygon "B".
    ///
    ///    `NewEnd`┌─┐x┌───────┐
    ///            │ │ │   B   │ Base polygon
    ///            │ │ └───────┘
    ///            │s│
    ///            │ │────────┐
    ///            │ │    A   │  Joining polygon
    ///  `NewStart`└─┘────────┘
    ///
    /// We know that a segment always has `NewEnd` and `NewStart` openings. Therefore we can
    /// calculate the index "x" at which the new polygon should be inserted by finding `NewEnd` and
    /// substracting 1.
    pub(crate) fn join_non_starting_polygon(
        &mut self,
        base_vertices_range: std::ops::Range<usize>,
        joining_polygon: &mut Self,
    ) {
        let mut maybe_joining_new_end_index = None;
        for (index, vertex) in joining_polygon.vertices.iter_mut().enumerate().rev() {
            // Find where in the joining polygon we are opening up to be joined by the base polygon.
            if matches!(vertex.opening, Opening::NewEnd(_)) {
                maybe_joining_new_end_index = Some(index);
            }

            // Nullify all openings so we don't try to join them in the future.
            // TODO: why can't this be done via the `downgrade_openings()` method?
            let is_start = matches!(vertex.opening, Opening::Start(_));
            let is_end = matches!(vertex.opening, Opening::End(_));
            if is_start || is_end {
                vertex.opening = Opening::Null;
            }
        }
        let Some(joining_new_end_index) = maybe_joining_new_end_index else {
            panic!("Couldn't find `Opening::NewEnd` for joining polygon");
        };

        // We can assume that the vertex at which we join the incoming polygon is always 1 before
        // the base polygon's `NewEnd` opening.
        let joining_vertices_entry = joining_new_end_index - 1;

        // Rotating achieves the effect of unlooping the polygon at the point at which the polygon
        // is joined.
        joining_polygon.vertices.rotate_left(joining_vertices_entry);

        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!(
            "Splicing existing polygon at {base_vertices_range:?}, \
            joining polygon: {joining_polygon:#?}, rotated at {joining_vertices_entry}",
        );

        self.join(base_vertices_range, joining_polygon.vertices.clone());

        self.holes.extend(joining_polygon.holes.clone());

        if joining_polygon.furthest_opening < self.furthest_opening {
            self.furthest_opening = joining_polygon.furthest_opening;
        }

        if joining_polygon.is_created_at_angle_0 {
            self.is_created_at_angle_0 = true;
        }
    }

    /// Insert a starting polygon (from angle 0) into a final polygon (from angle ~360).
    pub(crate) fn join_starting_polygon(
        &mut self,
        base_vertices_range: std::ops::Range<usize>,
        joining_vertices_range: std::ops::Range<usize>,
        joining_polygon: &mut Self,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!(
            "Splicing final polygon at {base_vertices_range:?}, \
             with: {:#?} at {joining_vertices_range:?}",
            joining_polygon.vertices
        );

        // Nullify the Genesis markers
        joining_polygon
            .vertices
            .get_mut(joining_vertices_range.start)
            .expect("Bad joining vertex index")
            .opening = Opening::Null;
        joining_polygon
            .vertices
            .get_mut(joining_vertices_range.end)
            .expect("Bad joining vertex index")
            .opening = Opening::Null;

        // Rotating achieves the effect of unlooping the polygon at the point at which the polygon
        // is joined.
        joining_polygon
            .vertices
            .rotate_left(joining_vertices_range.start);

        self.join(base_vertices_range, joining_polygon.vertices.clone());

        self.holes.extend(joining_polygon.holes.clone());
    }

    /// Join a polygon into itself.
    pub(crate) fn join_self(
        &mut self,
        left_range: std::ops::Range<usize>,
        right_range: std::ops::Range<usize>,
    ) {
        let range = if right_range.start > left_range.start {
            left_range.start..right_range.start
        } else {
            right_range.start..left_range.start
        };

        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!(
            "Splicing polygon into itself at: {range:?} ({left_range:?}/{right_range:?})"
        );

        let removed_vertices: Vec<super::vertices::Vertex> =
            self.vertices.splice(range, vec![]).collect();
        self.create_holes(&removed_vertices);
    }

    /// Extract the starting vertex of an opening that has just been joined to.
    pub(crate) fn extract_old_start(&mut self, index: usize) -> Opening {
        let vertex = self
            .vertices
            .get_mut(index)
            .expect("Bad index for old opening start");

        let old_start = vertex.opening.clone();
        vertex.opening = Opening::Null;

        old_start
    }

    /// Create interior holes from the vertices that were removed for the join.
    fn create_holes(&mut self, vertices: &[super::vertices::Vertex]) {
        let holes = vertices.split_inclusive(|vertex| matches!(vertex.opening, Opening::End(_)));

        for hole in holes {
            if hole.iter().any(|vertex| !vertex.is_centre()) {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::trace!("Hole: {hole:?}");
                self.holes.push(hole.to_vec());
            }
        }
    }

    /// Is a segment and a polygon, or 2 polygons, touching? We decide by whether their openings are
    /// overlapping.
    pub(crate) const fn is_touching(
        left: &std::ops::Range<u32>,
        right: &std::ops::Range<u32>,
    ) -> bool {
        left.start < right.end && right.start < left.end
    }

    /// Convert the polygon to the `geo` crate's representation. Ready for exporting to `GeoJSON`.
    pub fn to_polygon(&self) -> crate::polygon::Polygon {
        let holes: Vec<Vec<crate::polygon::Coordinate>> = self
            .holes
            .iter()
            .map(|hole| {
                hole.iter()
                    .map(|vertex| vertex.coordinate)
                    .collect::<Vec<crate::polygon::Coordinate>>()
            })
            .collect();

        let vertices: Vec<crate::polygon::Coordinate> = self
            .vertices
            .iter()
            .map(|vertex| vertex.coordinate)
            .collect();

        crate::polygon::Polygon {
            exterior: vertices,
            interior: holes,
        }
    }

    /// Remove adjacent vertices that are either identical or extremely similar. This reduces the
    /// byte size of the final polygon and speeds up the algorithm by about 10%.
    //
    // TODO: Could the algorithm be improved so that this wasn't needed?
    fn dedup_vertices_but_keep_openings(&mut self) {
        self.vertices.dedup_by(|left, right| {
            if !matches!(left.opening, Opening::Null) || !matches!(right.opening, Opening::Null) {
                return false;
            }
            Self::are_coordinates_within_tolerance(&left.coordinate, &right.coordinate)
        });
    }

    /// Dedupe vertices, but allow destroying opening metadata. This is useful right at the very end
    /// of the reconstruction.
    pub(crate) fn dedup_vertices_ignore_openings(&mut self) {
        self.vertices.dedup_by(|left, right| {
            Self::are_coordinates_within_tolerance(&left.coordinate, &right.coordinate)
        });
    }

    /// Are the two coordinates identical or as good as identical.
    fn are_coordinates_within_tolerance(
        left: &crate::polygon::Coordinate,
        right: &crate::polygon::Coordinate,
    ) -> bool {
        let tolerance = 1e-6f64;
        (left.x - right.x).abs() < tolerance && (left.y - right.y).abs() < tolerance
    }
}
