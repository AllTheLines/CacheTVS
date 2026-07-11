//! The vertices of a viewshed polygon. They have both real-world coordinates and "openings", which
//! are vertices that can join with other "openings".

/// A vertex is a single point in a polygon.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// The coordinate of the vertex in euclidian space.
    pub coordinate: crate::polygon::Coordinate,
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
pub enum Opening {
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

    #[expect(
        clippy::panic,
        reason = "Opening ordering is a requirement of the algorithm"
    )]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.index == 0 {
                #[expect(
                    clippy::indexing_slicing,
                    reason = "There always has to be a first element"
                )]
                if matches!(self.vertices[0].opening, Opening::Start(_)) {
                    panic!("Dangling Opening::Start");
                };
                return None;
            }

            #[expect(clippy::indexing_slicing, reason = "Bounds are enforced elsewhere")]
            let (left, right) = { (&self.vertices[self.index], &self.vertices[self.index - 1]) };

            let is_currently_end = matches!(left.opening, Opening::End(_));
            let is_previously_start = matches!(self.previous_opening, Opening::Start(_));
            let is_end_without_a_start = is_currently_end && !is_previously_start;
            assert!(!is_end_without_a_start, "Dangling Opening::End");

            if let Opening::Start(start) = left.opening {
                let Opening::End(end) = right.opening else {
                    panic!("Opening::Start not followed by Opening::End");
                };

                self.previous_opening = right.opening.clone();
                let opening_range = start..end;
                let opening_index = self.index;
                self.index = self.index.saturating_sub(2);
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
    OpeningsIterator::new(vertices)
        .filter(|(base_opening_index, base_opening_range)| {
            #[cfg(not(target_arch = "wasm32"))]
            tracing::debug!(
                "Checking if openings touch: \
                 index: {base_opening_index:?}, \
                 base distances: {base_opening_range:?}, \
                 joining distances: {joining_opening_range:?}",
            );

            let is_touching = crate::growable_polygon::GrowablePolygon::is_touching(
                base_opening_range,
                joining_opening_range,
            );

            if is_touching {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::debug!("🟢 Openings touch");
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                tracing::debug!("🟡 Openings don't touch");
            }

            is_touching
        })
        .fold(None, |accumulator, (index, _)| {
            let range_of_touch = accumulator.map_or(index..index, |range| {
                std::cmp::min(range.start, index)..std::cmp::max(range.end, index)
            });

            Some(range_of_touch)
        })
}

#[cfg(test)]
mod test {
    use super::*;

    fn vertex(opening: super::Opening) -> Vertex {
        super::Vertex {
            coordinate: crate::polygon::Coordinate::zero(),
            opening,
        }
    }

    fn run(mut vertices: Vec<super::Vertex>) -> Vec<(usize, std::ops::Range<u32>)> {
        vertices.reverse();
        OpeningsIterator::new(&vertices).collect::<Vec<_>>()
    }

    // TODO:
    //   I actually think side by side openings would themselves be a bug, but I haven't
    //   verified that so better test for it just in case.
    #[test]
    fn openings_side_by_side_should_not_panic() {
        let vertices = vec![
            vertex(super::Opening::Start(0)),
            vertex(super::Opening::End(1)),
            vertex(super::Opening::Start(3)),
            vertex(super::Opening::End(4)),
        ];
        assert_eq!(run(vertices), vec![(3, 0..1), (1, 3..4)]);
    }

    #[test]
    #[should_panic(expected = "Opening::Start not followed by Opening::End")]
    fn start_not_followed_by_end_panics() {
        let vertices = vec![
            vertex(super::Opening::Start(0)),
            vertex(super::Opening::Null),
            vertex(super::Opening::End(0)),
        ];
        run(vertices);
    }

    #[test]
    #[should_panic(expected = "Dangling Opening::Start")]
    fn start_in_final_position_panics() {
        let vertices = vec![
            vertex(super::Opening::Null),
            vertex(super::Opening::Null),
            vertex(super::Opening::Start(0)),
        ];
        run(vertices);
    }

    #[test]
    #[should_panic(expected = "Dangling Opening::End")]
    fn end_in_initial_position_panics() {
        let vertices = vec![
            vertex(super::Opening::End(0)),
            vertex(super::Opening::Start(0)),
            vertex(super::Opening::End(0)),
        ];
        run(vertices);
    }

    #[test]
    #[should_panic(expected = "Dangling Opening::End")]
    fn dangling_end_panics() {
        let vertices = vec![
            vertex(super::Opening::Null),
            vertex(super::Opening::End(0)),
            vertex(super::Opening::Start(0)),
            vertex(super::Opening::End(0)),
        ];
        run(vertices);
    }
}
