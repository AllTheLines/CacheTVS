#![cfg(test)]

#[derive(Clone, Copy)]
/// A raster coodinate.
pub struct RasterCoord {
    /// The x coordinate.
    pub x: i32,
    /// The y coordinate.
    pub y: i32,
}

impl From<(i32, i32)> for RasterCoord {
    fn from(value: (i32, i32)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

/// Iterator that yields (x, y) pixels on a line using Bresenham's algorithm.
/// <https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm>
pub struct Bresenham {
    /// The current start of the line. Updates as we move through the rasterisation.
    from: RasterCoord,
    /// Where the line ends.
    to: RasterCoord,
    /// The absolute distance that each component of the line has to move.
    delta: RasterCoord,
    /// How to step through each component of the line along the raster.
    incrementor: RasterCoord,
    /// The difference between delta.x and delta.y. Defines the shortest path between the beginning
    /// and end of the line.
    error: i32,
    /// Tell the Iterator that we've finished rasterising.
    is_finished: bool,
}

impl Bresenham {
    /// Instantiate.
    pub fn new(from: RasterCoord, to: RasterCoord) -> Self {
        let delta = RasterCoord {
            x: (to.x - from.x).abs(),
            y: (to.y - from.y).abs(),
        };
        Self {
            from,
            to,
            delta,
            incrementor: RasterCoord {
                x: if from.x < to.x { 1 } else { -1 },
                y: if from.y < to.y { 1 } else { -1 },
            },
            error: delta.x - delta.y,
            is_finished: false,
        }
    }
}

impl Iterator for Bresenham {
    type Item = RasterCoord;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_finished {
            return None;
        }
        let current_x = self.from.x;
        let current_y = self.from.y;
        if current_x == self.to.x && current_y == self.to.y {
            self.is_finished = true;
            return Some((self.to.x, self.to.y).into());
        }
        let doubled_error = 2i32 * self.error;
        if doubled_error > -self.delta.y {
            self.error -= self.delta.y;
            self.from.x += self.incrementor.x;
        }
        if doubled_error < self.delta.x {
            self.error += self.delta.x;
            self.from.y += self.incrementor.y;
        }
        Some((current_x, current_y).into())
    }
}

#[expect(clippy::default_numeric_fallback, reason = "Just tests")]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn positive_line() {
        let line: Vec<RasterCoord> =
            Bresenham::new(RasterCoord { x: 0, y: 0 }, RasterCoord { x: 3, y: 3 }).collect();
        assert_eq!(
            line.iter()
                .map(|coord| (coord.x, coord.y))
                .collect::<Vec<(i32, i32)>>(),
            vec![(0, 0), (1, 1), (2, 2), (3, 3)]
        );
    }

    #[test]
    fn negative_line() {
        let line: Vec<RasterCoord> =
            Bresenham::new(RasterCoord { x: -3, y: -3 }, RasterCoord { x: 1, y: 1 }).collect();
        assert_eq!(
            line.iter()
                .map(|coord| (coord.x, coord.y))
                .collect::<Vec<(i32, i32)>>(),
            vec![(-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1)]
        );
    }
}
