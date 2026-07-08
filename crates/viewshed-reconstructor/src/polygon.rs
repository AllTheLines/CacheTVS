//! A polygon represented by simple types. Avoids having to compile the `geo` crate.

/// A viewshed-based coordinate.
#[expect(clippy::exhaustive_structs, reason = "It will never change")]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Coordinate {
    /// The x coordinate.
    pub x: f64,
    /// The y coordinate.
    pub y: f64,
}

impl Coordinate {
    /// Just a shortcut for a coordinate at 0,0.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Scale the x and y compenents by a factor.
    #[inline]
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }
}

/// A polygon with a single exterior and multiple interior "holes".
#[expect(clippy::exhaustive_structs, reason = "It will never change")]
pub struct Polygon {
    /// Coordinates representing the polygon's exterior boundary.
    pub exterior: Vec<Coordinate>,
    /// Coordinates representing the polygon's interior boundaries.
    pub interior: Vec<Vec<Coordinate>>,
}
