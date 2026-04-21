//! A "segment" is a single area of visibility for a given angle on a viewshed. There may be many
//! segments per angle.

/// `Segment` is the rho portion of a line segment in polar coordinates
/// as (`rho`: u16, `delta_rho`: u16) which are packed into a single u32 for storage
#[derive(Clone, Default)]
pub struct Segment(pub u32);

impl Segment {
    /// `new` creates a `Segment` the segment's start point and the distance
    fn new(start: u16, distance: u16) -> Self {
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
    pub const fn start(&self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// `distance` returns the distance the `Segment` takes
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "the top 16 bits are guaranteed to be 0"
    )]
    pub const fn distance(&self) -> u16 {
        self.0 as u16
    }
}

#[expect(
    clippy::module_name_repetitions,
    reason = "That's the name we use in the DB"
)]
/// `PolarSegments` holds the degree of a line of sight and the list
/// of visible `Segments` which is constucted through a Run Length
/// Encoding algorithm.
///
/// Each `dem_id` will usually ~360 degrees. This may be more for 2 reasons:
///   1. The kernel is run with higher angular resolutions.
///   2. The rotation maths causes some `dem_id`/angles to have multiple pairs.
#[derive(Clone)]
pub struct PolarSegments {
    /// `degree` is a whole degree in the range [0, 359]
    pub degree: u16,
    /// `visible_segments` is a list of segments visible for a given
    /// angle and tvs id
    pub visible_segments: Vec<Segment>,
}

impl PolarSegments {
    /// `from_bools` constructs an `PolarSegments` from a visibility bitmap.
    /// It does so by implementing a binary Run Lenght Encoding.
    #[expect(
        clippy::expect_used,
        reason = "We assume everything fits in a u16, we want a panic if it doesn't"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "we want to panic if out of indexes are oob"
    )]
    pub fn from_bools(degree: u16, bitmap: &[bool]) -> Self {
        let mut visible_segments: Vec<Segment> = Vec::with_capacity(1);

        let char_slice: &[u8] = bytemuck::cast_slice(bitmap);

        let mut cur_index = 0;
        while char_slice.get(cur_index).is_some() {
            let first_zero_index =
                memchr::memchr(0, &char_slice[cur_index..]).unwrap_or(char_slice.len() - cur_index);

            visible_segments.push(Segment::new(
                u16::try_from(cur_index).expect("cur_index overflowed"),
                u16::try_from(first_zero_index).expect("first_zero_index overflowed"),
            ));

            cur_index += first_zero_index;

            let next =
                memchr::memchr(1, &char_slice[cur_index..]).unwrap_or(char_slice.len() - cur_index);

            cur_index += next;
        }

        Self {
            degree,
            visible_segments,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::cpu::storage::segments::PolarSegments;

    #[expect(
        clippy::indexing_slicing,
        reason = "testing, slicing/panicking is okay"
    )]
    #[test]
    fn bitmap_to_angle() {
        {
            let test_visibility = vec![true, true, true, false, true];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 2);
            assert_eq!(angles.visible_segments[0].distance(), 3);
            assert_eq!(angles.visible_segments[1].distance(), 1);
        };

        {
            let test_visibility = vec![true, false, false];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 1);
            assert_eq!(angles.visible_segments[0].distance(), 1);
            assert_eq!(angles.visible_segments[0].start(), 0);
        };

        {
            let test_visibility = vec![true, false, true, true, true, true, false];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 2);
            assert_eq!(angles.visible_segments[1].distance(), 4);
        }
    }
}
