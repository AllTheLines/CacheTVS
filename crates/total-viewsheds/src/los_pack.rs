//! Pack line of sight distance and angle into 32 bits. This makes storing and transfering the data
//! easier for viewing on the web.

use color_eyre::Result;

/// The max and bitmask for a u22: 4,194,303.
const U22_MAX: u32 = (1 << 22) - 1;
/// The max and bitmask for a u10: 1023.
const U10_MAX: u32 = (1 << 10) - 1;

#[derive(Default, Debug, Clone)]
/// Line of sight data that can be packed within 32 bits.
pub struct LineOfSightPacked(f32);

#[expect(
    clippy::big_endian_bytes,
    reason = "We don't care about the host's endedness, we're just transmuting"
)]
impl LineOfSightPacked {
    /// Pack line of sight data into f32.
    ///
    /// f32 is better than u32 because the `.bt` format only supports `i16` and `f32`. And I imagine
    /// that floats are just more widely recognised in general. At the end of the day, they're just
    /// bits that are neither valid f32 nor u32.
    pub fn new(distance: u32, angle: u16) -> Result<Self> {
        if distance > U22_MAX {
            color_eyre::eyre::bail!("{} is greater than u22 max {U22_MAX}", distance);
        }
        if u32::from(angle) > U10_MAX {
            color_eyre::eyre::bail!("{} is greater than u10 max {U10_MAX}", angle);
        }

        let packed = unsafe {
            Self::new_unchecked(distance, angle)
        };

        Ok(packed)
    }

    pub unsafe fn new_unchecked(distance: u32, angle: u16) -> Self {
        let angle_u32 = u32::from(angle);

        let distance_bits = (distance & U22_MAX) << 10u32;
        let angle_bits = angle_u32 & U10_MAX;
        let packed_integer_u32 = distance_bits | angle_bits;
        let packed_float_f32 = f32::from_be_bytes(packed_integer_u32.to_be_bytes());

        Self(packed_float_f32)

    }

    /// Transmute to `u32` in order to do the bitshifting.
    const fn to_u32(&self) -> u32 {
        u32::from_be_bytes(self.as_f32().to_be_bytes())
    }

    /// Distance in meters from the point of view to the visible point.
    pub const fn distance(&self) -> u32 {
        (self.to_u32() >> 10u32) & U22_MAX
    }

    /// The angle of the line of sight from the point of view.
    pub fn angle(&self) -> u16 {
        (self.to_u32() & U10_MAX) as u16
    }

    /// Get the raw f32 value. It doesn't represent any useful data, it's just the f32 view of the
    /// packed data.
    pub const fn as_f32(&self) -> f32 {
        self.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use googletest::prelude::*;

    #[test]
    fn pack_unpack_small() {
        let line_of_sight = LineOfSightPacked::new(123, 45).unwrap();

        assert_eq!(line_of_sight.distance(), 123);
        assert_eq!(line_of_sight.angle(), 45);
    }

    #[test]
    fn pack_unpack_big() {
        let max_angle = U10_MAX.try_into().unwrap();
        let line_of_sight = LineOfSightPacked::new(U22_MAX, max_angle).unwrap();

        assert_eq!(line_of_sight.distance(), U22_MAX);
        assert_eq!(line_of_sight.angle(), max_angle);
    }

    #[gtest]
    fn pack_unpack_oversize_distance() {
        expect_that!(
            LineOfSightPacked::new(U22_MAX + 1, 0)
                .unwrap_err()
                .to_string(),
            contains_substring("4194304 is greater than u22 max")
        );
    }

    #[gtest]
    fn pack_unpack_oversize_angle() {
        expect_that!(
            LineOfSightPacked::new(0, (U10_MAX + 1).try_into().unwrap())
                .unwrap_err()
                .to_string(),
            contains_substring("1024 is greater than u10 max")
        );
    }
}
