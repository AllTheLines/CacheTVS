//! The vertices of a viewshed polygon. They have both real-world coordinates and "openings", which
//! are vertices that can join with other "openings".

/// A vertex is a single point in a polygon.
#[derive(Debug, Clone)]
pub(crate) struct Vertex {
    /// The coordinate of the vertex in euclidian space.
    pub coordinate: geo::Coord,
    /// The kind of opening that this vertex represents.
    pub opening: Opening,
}

impl Vertex {
    /// Is the vertex at the centre of the viewshed? The centre is a special place, it is a
    /// zero-length opening that isn't nulled if no segments/polygons touch it for a given angle.
    pub(crate) fn is_centre(&self) -> bool {
        self.coordinate.x.abs() == 0.0 && self.coordinate.y.abs() == 0.0
    }
}

/// An opening is where one polygon can join a segment, or even another polygon.
/// The `u32` values in the variants are for storing the polar distance from the centre. This saves
/// having to do trigonometry to figure out if two openings are touching.
#[derive(Eq, PartialEq, Debug, Clone)]
pub(crate) enum Opening {
    /// No opening. We could also just use `Option::None`, but unwrapping it isn't so ergonomic.
    Null,
    /// This i for polygons that start at 0 degrees. It's possible that another polygon (or even
    /// itself), could connect to it at 360 degrees.
    GenesisStart(u32),
    /// The closing of the genesis opening at 0 degrees.
    GenesisEnd(u32),
    /// An active opening's start.
    Start(u32),
    /// An active opening's end.
    End(u32),
    /// When a segment or polygon is successfully attached to an existing polygon, but we still want
    /// to carry on looking for other attachments in the given angle, we don't want to mistakenly
    /// attach new segments/polygons to something that's just been attached. Once the whole angle is
    /// completed, then these get downgraded to the common `Start/End` opendings.
    NewStart(u32),
    /// A `New` openings end.
    NewEnd(u32),
}

/// Iterator for finding start/open vertex pairs for openings.
pub(crate) struct OpeningsIterator<'vertices> {
    /// The vertices to scan.
    vertices: &'vertices [Vertex],
    /// The current index of the iterator.
    index: usize,
    /// Keep track of the previously scanned opening.
    previous_opening: Opening,
}

impl<'vertices> OpeningsIterator<'vertices> {
    /// Instantiate.
    pub(crate) const fn new(vertices: &'vertices [Vertex]) -> Self {
        Self {
            vertices,
            index: vertices.len() - 1,
            previous_opening: Opening::Null,
        }
    }
}

impl Iterator for OpeningsIterator<'_> {
    type Item = (usize, std::ops::Range<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index <= 1 {
                return None;
            }

            #[expect(clippy::indexing_slicing, reason = "Bounds are enforced elsewhere")]
            let (left, right) = { (&self.vertices[self.index], &self.vertices[self.index - 1]) };

            let is_currently_end = matches!(left.opening, Opening::End(_));
            let is_previously_start = matches!(self.previous_opening, Opening::Start(_));
            let is_end_without_a_start = is_currently_end && !is_previously_start;
            assert!(!is_end_without_a_start, "Dangling Opening::End");
            if let Opening::Start(start) = left.opening {
                #[expect(
                    clippy::panic,
                    reason = "Opening ordering is a requirement of the algorithm"
                )]
                let Opening::End(end) = right.opening else {
                    panic!("Opening::Start not followed by Opening::End");
                };

                self.previous_opening = right.opening.clone();
                let opening_range = start..end;
                let opening_index = self.index;
                self.index -= 2;
                return Some((opening_index, opening_range));
            }

            self.previous_opening = left.opening.clone();
            self.index -= 1;
        }
    }
}

/// Check if 2 openings touch.
pub(crate) fn find_contact(
    vertices: &[Vertex],
    joining_opening_range: &std::ops::Range<u32>,
) -> Option<std::ops::Range<usize>> {
    let mut maybe_touching_start = None;
    let mut maybe_touching_end = None;

    let openings = OpeningsIterator::new(vertices);

    for (base_opening_index, base_opening_range) in openings {
        tracing::debug!(
            "Checking if openings touch: \
             index: {base_opening_index:?}, \
             base distances: {base_opening_range:?}, \
            joining distances: {joining_opening_range:?}",
        );

        if !crate::output::viewsheds::growable_polygon::GrowablePolygon::is_touching(
            &base_opening_range,
            joining_opening_range,
        ) {
            tracing::debug!("🟡 Openings don't touch");
            continue;
        }

        tracing::debug!("🟢 Openings touch");

        if maybe_touching_start.is_none() {
            maybe_touching_start = Some(base_opening_index);
        }
        maybe_touching_end = Some(base_opening_index);
    }

    match (maybe_touching_start, maybe_touching_end) {
        (Some(start), Some(end)) => Some(end..start),
        _ => None,
    }
}
