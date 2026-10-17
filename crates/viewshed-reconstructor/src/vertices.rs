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

/// The kind of opening.
#[derive(Debug, Clone)]
pub enum OpeningKind {
    /// An opening on a starting angle 0 polygon.
    Genesis,
    /// A normal opening.
    Normal,
    /// An opening that has been added before the end of parsing the current angle.
    New,
}

/// An opening/closing on a polygon.
#[derive(Debug, Clone)]
pub struct OpeningPair {
    /// The vertex index.
    pub index: usize,
    /// The index range of the opneing.
    pub range: std::ops::Range<u32>,
    /// The kind of opening.
    pub kind: OpeningKind,
}

/// Iterator for finding start/open vertex pairs for openings.
pub(crate) struct OpeningPairs<'vertices> {
    /// The vertices to scan.
    vertices: &'vertices [Vertex],
    /// The current index of the iterator.
    index: usize,
    /// Keep track of the previously scanned opening.
    previous_opening: Opening,
}

impl<'vertices> OpeningPairs<'vertices> {
    /// Instantiate.
    const fn new(vertices: &'vertices [Vertex]) -> Self {
        Self {
            vertices,
            index: vertices.len() - 1,
            previous_opening: Opening::Null,
        }
    }

    /// Find all the opening pairs on a polygon.
    pub(crate) fn find(vertices: &'vertices [Vertex]) -> Vec<OpeningPair> {
        let mut finder = Self::new(vertices);
        let mut opening_pairs = Vec::new();
        while let Some(opening_pair) = finder.next() {
            opening_pairs.push(opening_pair);
        }
        opening_pairs.sort_by_key(|opening_pair| opening_pair.range.start);
        opening_pairs
    }

    #[expect(
        clippy::panic,
        reason = "Opening ordering is a requirement of the algorithm"
    )]
    /// Loop over indexes to find opening pairs.
    fn next(&mut self) -> Option<OpeningPair> {
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
            if is_currently_end {
                let is_previously_start = matches!(self.previous_opening, Opening::Start(_));
                assert!(is_previously_start, "Dangling Opening::End");
            }

            if let Opening::NewStart(start) = left.opening {
                let Opening::NewEnd(end) = right.opening else {
                    panic!("Opening::NewStart not followed by Opening::NewEnd");
                };

                let pair =
                    self.opening_pair_found(right.opening.clone(), start..end, OpeningKind::New);
                return Some(pair);
            }

            if let Opening::Start(start) = left.opening {
                let Opening::End(end) = right.opening else {
                    panic!("Opening::Start not followed by Opening::End");
                };

                let pair =
                    self.opening_pair_found(right.opening.clone(), start..end, OpeningKind::Normal);
                return Some(pair);
            }

            self.previous_opening = left.opening.clone();
            self.index -= 1;
        }
    }

    /// Handle the finding of an opening pair.
    const fn opening_pair_found(
        &mut self,
        previous: Opening,
        range: std::ops::Range<u32>,
        kind: OpeningKind,
    ) -> OpeningPair {
        self.previous_opening = previous;
        let opening_index = self.index;
        self.index = self.index.saturating_sub(2);
        OpeningPair {
            index: opening_index,
            range,
            kind,
        }
    }
}

/// Check if 2 openings touch.
pub(crate) fn find_contact(
    growable_polygon: &crate::growable_polygon::GrowablePolygon,
    joining_opening_range: &std::ops::Range<u32>,
) -> Option<std::ops::Range<usize>> {
    growable_polygon
        .openings
        .iter()
        .filter(|opening_pair| matches!(opening_pair.kind, OpeningKind::Normal))
        .filter(|opening_pair| {
            let is_touching = crate::growable_polygon::GrowablePolygon::is_touching(
                &opening_pair.range,
                joining_opening_range,
            );

            let message = if is_touching {
                "🟢 Openings touch"
            } else {
                "🟡 Openings don't touch"
            };

            tracing::debug!(
                "{message}: \
                 index: {:?}, \
                 base distances: {:?}, \
                 joining distances: {joining_opening_range:?}",
                opening_pair.index,
                opening_pair.range
            );

            is_touching
        })
        .fold(None, |accumulator, opening_pair| {
            let index = opening_pair.index;
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
        OpeningPairs::find(&vertices)
            .into_iter()
            .map(|opening| (opening.index, opening.range))
            .collect::<Vec<_>>()
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
