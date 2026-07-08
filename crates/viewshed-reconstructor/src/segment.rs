//! A "segment" is a single area of visibility for a given angle on a viewshed. There may be many
//! segments per angle.

/// `Segment` is the rho portion of a line segment in polar coordinates
/// as (`rho`: u16, `delta_rho`: u16) which are packed into a single u32 for storage.
#[expect(clippy::exhaustive_structs, reason = "This will never change.")]
#[derive(Clone, Default)]
pub struct Segment(pub u32);

impl Segment {
    /// `new` creates a `Segment` the segment's start point and the distance
    #[inline]
    #[must_use]
    pub fn new(start: u16, distance: u16) -> Self {
        // pack start/distsance into a u32 in the format of (start|distance)
        let wide_start: u32 = start.into();
        let wide_distance: u32 = distance.into();
        Self((wide_start << 16) | wide_distance)
    }

    /// `start` returns the starting point of the `Segment`
    #[expect(
        clippy::as_conversions,
        reason = "the top 16 bits are guaranteed to be 0"
    )]
    #[inline]
    #[must_use]
    pub const fn start(&self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// `distance` returns the distance the `Segment` takes
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "the top 16 bits are guaranteed to be 0"
    )]
    #[inline]
    #[must_use]
    pub const fn distance(&self) -> u16 {
        self.0 as u16
    }
}
