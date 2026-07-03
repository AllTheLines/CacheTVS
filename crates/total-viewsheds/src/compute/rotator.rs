//! A "bijective" rotator that ensures a 1:1 mapping of original DEM IDs to rotated IDs.
//!
//! We use the "3 Shear" method, aka Paeth Rotation.
//! <https://web.archive.org/web/20241121043312/https://datagenetics.com/blog/august32013/index.html>

/// Shear rotation is not stable beyond rotations of ±90°. So we have to use other methods to
/// rotate to within ±90°. Rotating in units of 90° is always stable because you can use
/// axis-switching and coordinate inversion:
///   * 90°:  (`width_index` - y, x)
///   * 180°: (`width_index` - x, `width_index` - y)
//    * 270°: (y, `width_index` - x)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Quadrant {
    /// Top-right quadrant: 0° to 90°
    TopRight,
    /// Top-left quadrant: 90° to 180°
    TopLeft,
    /// Bottom-left quadrant: 180° to 270°
    BottomLeft,
    /// Bottom-right quadrant: 270° to 0°
    BottomRight,
}

impl Quadrant {
    /// What's the upper angular limit of each quadrant?
    const fn upper_limit(self) -> f64 {
        match self {
            Self::BottomRight => 0.0,
            Self::TopRight => 90.0,
            Self::TopLeft => 180.0,
            Self::BottomLeft => 270.0,
        }
    }
}

impl From<f64> for Quadrant {
    fn from(angle: f64) -> Self {
        let quadrant_size = 90.0f64;

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "Our enum definition shows that we can only be between 0 and 3"
        )]
        let quadrant = (angle / quadrant_size).floor() as u8;

        match quadrant {
            1 => Self::TopRight,
            2 => Self::TopLeft,
            3 => Self::BottomLeft,
            _ => Self::BottomRight,
        }
    }
}

/// Rotator struct.
pub(crate) struct Rotator {
    /// The centre of the DEM in relative coordinates.
    centre: (f64, f64),
    /// The index of the last point in a row/column of the DEM.
    width_index: isize,
    /// The 3-Shear method is only stable between -90° and +90°. So we have to use basic flipping
    /// to move the pixels for angles outside of this range. Each quadrant has its own quick way
    /// of rotating to its upper limit.
    quadrant: Quadrant,
    /// Cached constant for horizontal shearing.
    alpha: f64,
    /// Cached constant for vertical shearing.
    beta: f64,
}

impl Rotator {
    /// Instantiate.
    pub(crate) fn new(angle: f64, width: isize) -> Self {
        assert!(
            (0.0f64..360.0f64).contains(&angle),
            "Angle {angle} must be in range 0..360"
        );

        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "
              No DEM will be wider than the largest representable integer for a f64 (2^53)
            "
        )]
        let middle = (width - 1) as f64 / 2.0f64;
        let (x_center, y_center) = (middle, middle);
        let angle_inverted = Self::invert_angle(angle);

        let quadrant = Quadrant::from(angle_inverted);
        let remaining_angle = angle_inverted - quadrant.upper_limit();

        let theta = remaining_angle.to_radians();
        let alpha = -(theta / 2.0).tan();
        let beta = theta.sin();

        Self {
            centre: (x_center, y_center),
            width_index: width - 1,
            quadrant,
            alpha,
            beta,
        }
    }

    /// In order to follow the convention that rotation is measured anti-clockwise, typically
    /// we'd just make the angle negative here. But seeing as the rest of the maths involved in
    /// skew rotation is based in positive angles it's better to achieve the same rotation
    /// through a positive but inverted angle.
    pub(crate) fn invert_angle(angle: f64) -> f64 {
        if angle > 0.0f64 {
            360.0f64 - angle
        } else {
            0.0
        }
    }

    /// Rotate using the 3 shear method. We got the idea from Matt Parker:
    ///   <https://www.youtube.com/watch?v=1LCEiVDHJmc>
    ///
    /// Note that each shear step _must_ be rounded. This is essential to achieving a bijective
    /// mapping.
    pub(crate) fn rotate(&self, x: isize, y: isize) -> (isize, isize) {
        let (x_centre, y_centre) = self.centre;
        let (quadrant_x, quadrant_y) = self.rotate_to_quadrant(x, y);

        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "
              No DEM will be wider than the largest representable integer for a f64 (2^53)
            "
        )]
        let (x_offset, y_offset) = (quadrant_x as f64 - x_centre, quadrant_y as f64 - y_centre);

        // First horizontal shear
        let x1 = x_offset + (self.alpha * y_offset).round();
        let y1 = y_offset;

        // Vertical shear
        let x2 = x1;
        let y2 = y1 + (self.beta * x1).round();

        // Second horizontal shear
        let x3 = x2 + (self.alpha * y2).round();
        let y3 = y2;

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "No DEM will be wider then i64:MAX"
        )]
        let (x_rotated, y_rotated) = ((x3 + x_centre) as isize, (y3 + y_centre) as isize);

        (
            x_rotated.clamp(0, self.width_index),
            y_rotated.clamp(0, self.width_index),
        )
    }

    /// Skew rotation is only stable for ±90°. So we use other stabler methods to get within ±90°.
    const fn rotate_to_quadrant(&self, x: isize, y: isize) -> (isize, isize) {
        match self.quadrant {
            Quadrant::BottomRight => (x, y),
            Quadrant::TopRight => self.rotate_90(x, y),
            Quadrant::TopLeft => self.rotate_180(x, y),
            Quadrant::BottomLeft => self.rotate_270(x, y),
        }
    }

    /// Rotate by 90° using axis exchange and flipping.
    const fn rotate_90(&self, x: isize, y: isize) -> (isize, isize) {
        (self.width_index - y, x)
    }

    /// Rotate by 180° using flipping.
    const fn rotate_180(&self, x: isize, y: isize) -> (isize, isize) {
        (self.width_index - x, self.width_index - y)
    }

    /// Rotate by 270° using axis exchange and flipping.
    const fn rotate_270(&self, x: isize, y: isize) -> (isize, isize) {
        (y, self.width_index - x)
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    reason = "These are just tests"
)]
#[cfg(test)]
mod test {
    use super::*;
    use googletest::prelude::*;

    #[rustfmt::skip]
    const DEM: [i16; 36] = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,11,
        12,13,14,15,16,17,
        18,19,20,21,22,23,
        24,25,26,27,28,29,
        30,31,32,33,34,35,
    ];

    fn rotate(angle: f64) -> Vec<i16> {
        let width = DEM.len().isqrt() as isize;
        let rotator = Rotator::new(angle, width);
        let mut rotated = Vec::new();
        for y in 0..width {
            for x in 0..width {
                let (x_rotated, y_rotated) = rotator.rotate(x, y);
                let index = y_rotated * width + x_rotated;
                let value = DEM[index as usize];
                rotated.push(value);
            }
        }

        rotated
    }

    #[gtest]
    fn remainder_always_less_than_90() {
        for angle in 0..360 {
            let rotator = Rotator::new(f64::from(angle), 6);
            let inverted_angle = Rotator::invert_angle(f64::from(angle));
            let remainder = inverted_angle - rotator.quadrant.upper_limit();
            expect_that!(remainder, lt(90.0));
        }
    }

    #[gtest]
    fn rotate_dem_0() {
        expect_eq!(rotate(0.0), DEM);
    }

    #[gtest]
    fn rotate_dem_45() {
        // Repeating IDs are because of "dolphins" jumping out of the DEM square and needing to be
        // clamped inside.
        #[rustfmt::skip]
        let expected: [i16; 36] = [
            18, 12, 7,  1,  2,  2,
            18, 12, 13, 8,  9,  3,
            24, 19, 20, 14, 10, 4,
            31, 25, 21, 15, 16, 11,
            32, 26, 27, 22, 23, 17,
            33, 33, 34, 28, 23, 17,
        ];
        expect_eq!(rotate(45.0), expected);
    }

    #[gtest]
    fn rotate_dem_180() {
        #[rustfmt::skip]
        let expected: [i16; 36] = [
            35, 34, 33, 32, 31, 30,
            29, 28, 27, 26, 25, 24,
            23, 22, 21, 20, 19, 18,
            17, 16, 15, 14, 13, 12,
            11, 10, 9,  8,  7,  6,
            5,  4,  3,  2,  1,  0,
        ];
        expect_eq!(rotate(180.0), expected);
    }

    #[gtest]
    fn rotate_dem_225() {
        #[rustfmt::skip]
        let expected: [i16; 36] = [
            17, 23, 28, 34, 33, 33,
            17, 23, 22, 27, 26, 32,
            11, 16, 15, 21, 25, 31,
            4,  10, 14, 20, 19, 24,
            3,  9,  8,  13, 12, 18,
            2,  2,  1,  7,  12, 18,
        ];
        expect_eq!(rotate(225.0), expected);
    }

    #[gtest]
    fn rotate_dem_315() {
        #[rustfmt::skip]
        let expected: [i16; 36] = [
            2,  3,  4,  11, 17, 17,
            2,  9,  10, 16, 23, 23,
            1,  8,  14, 15, 22, 28,
            7,  13, 20, 21, 27, 34,
            12, 12, 19, 25, 26, 33,
            18, 18, 24, 31, 32, 33,
        ];
        expect_eq!(rotate(315.0), expected);
    }

    #[gtest]
    fn rotate_dem_359() {
        expect_eq!(rotate(359.0), DEM);
    }
}
