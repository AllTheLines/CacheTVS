//! Rotate DEM data so that it is better aligned for memory access.

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A coordinate for which a rotation cannot be made.
pub const NOOP_COORDINATE: glam::Vec2 = glam::Vec2::new(f32::NAN, f32::NAN);
/// A DEM ID for which a rotation cannot be made.
pub const NOOP_DEM_ID: usize = usize::MAX;

/// This is to overcome float rounding for edge case angles.
///
/// For example at 45° some coordinates rotate to .5, which rounds to the same
/// coordinate it came from. Adding a little shift ensures that the desired
/// rounding occurs and that rasterisation is correct.
pub const ANGLE_SHIFT: f32 = 0.0001;

/// Cached data for rotating points.
pub struct Rotator {
    /// The ID of the elevation point in the DEM.
    dem_id: u32,
    /// The number of points in the width of the DEM.
    width: u32,
    /// The distance each point has to move so that its coordinate is relative to the centre of the
    /// DEM.
    offset_from_centre: f32,
    /// The cached sine of the angle we're rotating.
    sine: f32,
    /// The cached cosine of the angle we're rotating.
    cosine: f32,
}

impl Rotator {
    #[inline]
    #[must_use]
    /// Instantiate with precalculated sine and cosine. Used by GPU to shave off some cycles from
    /// not having to do the trigonometry.
    pub fn new_from_cached_trig(dem_id: u32, width: u32, sine: f32, cosine: f32) -> Self {
        Self {
            dem_id,
            width,
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "Width will always fit in `f32`"
            )]
            offset_from_centre: (width as f32 - 1.0) / 2.0,
            sine,
            cosine,
        }
    }

    #[inline]
    #[must_use]
    /// Instantiate with an angle.
    pub fn new_from_angle(dem_id: u32, width: u32, angle: f32) -> Self {
        let trig = Self::calculate_trig(angle);
        Self::new_from_cached_trig(dem_id, width, trig.0, trig.1)
    }

    #[inline]
    #[must_use]
    /// Calculate the sine and cosine of the angle with small angle shift for better rasterisation.
    pub fn calculate_trig(angle: f32) -> (f32, f32) {
        (
            (angle + ANGLE_SHIFT).to_radians().sin(),
            (angle + ANGLE_SHIFT).to_radians().cos(),
        )
    }

    #[inline]
    /// Rotate a single point.
    pub fn rotate_value_nearest_neighbour(
        &self,
        elevations_in: &[f32],
        elevations_out: &mut [f32],
    ) {
        let rotated_id = self.rotate_dem_id();
        let elevation = if rotated_id == NOOP_DEM_ID {
            f32::NAN
        } else {
            elevations_in[rotated_id]
        };

        #[expect(clippy::as_conversions, reason = "DEM ID will always fit in `usize`")]
        {
            elevations_out[self.dem_id as usize] = elevation;
        }
    }

    #[inline]
    /// Rotate a single point using sampled interpolation
    pub fn rotate_value_bilinear(&self, elevations_in: &[f32], elevations_out: &mut [f32]) {
        let coordinate = self.dem_id_to_coord();
        let rotated = self.rotate_coordinate(coordinate);
        let interpolated = if rotated == NOOP_COORDINATE {
            f32::NAN
        } else {
            self.bilinear_sample(elevations_in, rotated)
        };

        #[expect(clippy::as_conversions, reason = "DEM ID will always fit in `usize`")]
        {
            elevations_out[self.dem_id as usize] = interpolated;
        }
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "We're interpolating, it has to be like this."
    )]
    #[inline]
    /// Get a pixel for sampling.
    fn get_pixel(&self, elevations: &[f32], x: f32, y: f32) -> f32 {
        let width_f32 = self.width as f32;
        let clamped_x = x.clamp(0.0, width_f32 - 1.0) as usize;
        let clamped_y = y.clamp(0.0, width_f32 - 1.0) as usize;
        elevations[clamped_y * self.width as usize + clamped_x]
    }

    #[inline]
    #[must_use]
    /// Bilinear interpolation.
    ///
    /// This seems to give clearly worse results. Perhaps try larger interpolation kernels?
    pub fn bilinear_sample(&self, elevations: &[f32], coordinate: glam::Vec2) -> f32 {
        // Integer pixel indices for the top-left corner
        let x_base = coordinate.x.floor();
        let y_base = coordinate.y.floor();

        // Fractional offsets within the pixel cell
        let x_fraction = coordinate.x - x_base;
        let y_fraction = coordinate.y - y_base;

        // Weights for interpolation along each axis
        let x0_weight = 1.0 - x_fraction;
        let x1_weight = x_fraction;
        let y0_weight = 1.0 - y_fraction;
        let y1_weight = y_fraction;

        // Sample the four neighboring pixels
        let top_left = self.get_pixel(elevations, x_base, y_base);
        let top_right = self.get_pixel(elevations, x_base + 1.0, y_base);
        let bottom_left = self.get_pixel(elevations, x_base, y_base + 1.0);
        let bottom_right = self.get_pixel(elevations, x_base + 1.0, y_base + 1.0);

        // Interpolate
        let interp_top = top_left.mul_add(x0_weight, top_right * x1_weight);
        let interp_bottom = bottom_left * x0_weight + bottom_right * x1_weight;
        interp_top * y0_weight + interp_bottom * y1_weight
    }

    #[inline]
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "We're just casting for the maths. And the tests should catch any issues."
    )]
    /// Rotate a coordinate about the centre of the DEM.
    fn private_rotate_coordinate(&self, point: glam::UVec2, direction: f32) -> glam::Vec2 {
        let sine = self.sine * direction;
        let x_from = point.x as f32 - self.offset_from_centre;
        let y_from = point.y as f32 - self.offset_from_centre;
        let distance = glam::Vec2::new(x_from, y_from).distance(glam::Vec2::ZERO);
        if distance > self.offset_from_centre {
            // TODO: Explore better ways of handling this. On the CPU we could just return `None`,
            // but the GPU doesn't support `Option`.
            return NOOP_COORDINATE;
        }

        let x_to = x_from.mul_add(self.cosine, -(y_from * sine));
        let dem_x = x_to + self.offset_from_centre;

        let y_to = x_from.mul_add(sine, y_from * self.cosine);
        let dem_y = y_to + self.offset_from_centre;

        glam::Vec2 { x: dem_x, y: dem_y }
    }

    #[inline]
    #[must_use]
    /// Rotate an x,y coordinate about the centre of the DEM.
    pub fn rotate_coordinate(&self, point: glam::UVec2) -> glam::Vec2 {
        self.private_rotate_coordinate(point, 1.0)
    }

    #[inline]
    #[must_use]
    /// Rotate an x,y coordinate about the centre of the DEM, but in the opposite direction.
    pub fn anti_rotate_coordinate(&self, point: glam::UVec2) -> glam::Vec2 {
        self.private_rotate_coordinate(point, -1.0)
    }

    #[inline]
    #[must_use]
    /// Rotate or anti rotate a DEM ID.
    fn private_rotate_dem_id(&self, anti_rotate: bool) -> usize {
        let coordinate = self.dem_id_to_coord();
        let rotated_coord = if anti_rotate {
            self.anti_rotate_coordinate(coordinate)
        } else {
            self.rotate_coordinate(coordinate)
        };
        if rotated_coord.x.is_nan() || rotated_coord.y.is_nan() {
            return NOOP_DEM_ID;
        }
        let rounded = rotated_coord.round();

        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "We're rasterising"
        )]
        {
            let cast = glam::UVec2::new(rounded.x as u32, rounded.y as u32);
            self.coord_to_dem_id(cast) as usize
        }
    }

    #[inline]
    #[must_use]
    /// Rotate a DEM ID.
    pub fn rotate_dem_id(&self) -> usize {
        self.private_rotate_dem_id(false)
    }

    #[inline]
    #[must_use]
    /// Rotate a DEM ID by the same angle but in the opposite direction.
    pub fn anti_rotate_dem_id(&self) -> usize {
        self.private_rotate_dem_id(true)
    }

    #[inline]
    #[must_use]
    /// Convert a DEM ID index to an x,y coordinate.
    pub const fn dem_id_to_coord(&self) -> glam::UVec2 {
        let x = self.dem_id.rem_euclid(self.width);
        let y = self.dem_id.div_euclid(self.width);
        glam::UVec2 { x, y }
    }

    #[inline]
    #[must_use]
    /// Convert an x,y coordinate to a DEM ID.
    pub const fn coord_to_dem_id(&self, coordinate: glam::UVec2) -> u32 {
        coordinate.y * self.width + coordinate.x
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    reason = "These are just tests"
)]
#[cfg(test)]
mod test {
    use core::f32;

    use crate::tests::matchers::good_coordinate;

    use super::*;
    use googletest::prelude::*;

    const NAN: f32 = f32::NAN;

    #[rustfmt::skip]
    const DEM3: [f32; 9] = [
        0.0, 1.0, 2.0,
        3.0, 4.0, 5.0,
        6.0, 7.0, 8.0
    ];

    #[rustfmt::skip]
    const DEM4: [f32; 16] = [
        0.0,  1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,  7.0,
        8.0,  9.0,  10.0, 11.0,
        12.0, 13.0, 14.0, 15.0
    ];

    fn run_dem3(angle: f32) -> [f32; 9] {
        let mut rotated = [0.0; 9];
        for dem_id in 0..DEM3.len() {
            let rotator = Rotator::new_from_angle(dem_id as u32, 3, angle);
            rotator.rotate_value_nearest_neighbour(&DEM3, &mut rotated);
        }

        rotated
    }

    fn run_dem4(angle: f32) -> [f32; 16] {
        let mut rotated = [0.0; 16];
        for dem_id in 0..DEM4.len() {
            let rotator = Rotator::new_from_angle(dem_id as u32, 4, angle);
            rotator.rotate_value_nearest_neighbour(&DEM4, &mut rotated);
        }

        rotated
    }

    #[expect(clippy::print_stderr, reason = "Needed for test output")]
    fn assert_dem(actual: &[f32], expected: &[f32]) {
        let mut is_failed = false;
        for (id, value) in expected.iter().enumerate() {
            if value.is_nan() {
                expect_that!(actual[id], is_nan());
            } else {
                if verify_float_eq!(actual[id], *value).is_err() {
                    is_failed = true;
                }
                expect_float_eq!(actual[id], *value);
            }
        }

        if is_failed {
            eprint!("Actual: ");
            crate::tests::dems::print_dem(actual);
        }
    }

    #[gtest]
    fn rotate_by_0_dem3() {
        #[rustfmt::skip]
        let expected = [
            NAN, 1.0, NAN,
            3.0, 4.0, 5.0,
            NAN, 7.0, NAN
        ];
        assert_dem(&run_dem3(0.0), &expected);
    }

    #[gtest]
    fn rotate_by_0_dem4() {
        #[rustfmt::skip]
        let expected = [
            NAN, NAN, NAN,  NAN,
            NAN, 5.0, 6.0,  NAN,
            NAN, 9.0, 10.0, NAN,
            NAN, NAN, NAN,  NAN
        ];
        assert_dem(&run_dem4(0.0), &expected);
    }

    #[gtest]
    fn rotate_by_45_dem3() {
        #[rustfmt::skip]
        let expected = [
            NAN, 2.0, NAN,
            0.0, 4.0, 8.0,
            NAN, 6.0, NAN
        ];

        assert_dem(&run_dem3(45.0), &expected);
    }

    #[gtest]
    fn rotate_by_45_dem4() {
        #[rustfmt::skip]
        let expected = [
            NAN, NAN, NAN,  NAN,
            NAN, 6.0, 10.0, NAN,
            NAN, 5.0, 9.0,  NAN,
            NAN, NAN, NAN,  NAN
        ];
        assert_dem(&run_dem4(45.0001), &expected);
    }

    #[gtest]
    fn rotate_by_90_dem3() {
        #[rustfmt::skip]
        let expected = [
            NAN, 5.0, NAN,
            1.0, 4.0, 7.0,
            NAN, 3.0, NAN
        ];

        assert_dem(&run_dem3(90.0), &expected);
    }

    #[gtest]
    fn rotate_by_135_dem3() {
        #[rustfmt::skip]
        let expected = [
            NAN, 8.0, NAN,
            2.0, 4.0, 6.0,
            NAN, 0.0, NAN
        ];

        assert_dem(&run_dem3(135.0), &expected);
    }

    #[gtest]
    fn rotate_dem3_coordinate() {
        let mut rotator = Rotator::new_from_angle(0, 3, 1.0);
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 1, y: 1 }),
            good_coordinate(glam::Vec2 { x: 1.0, y: 1.0 })
        );
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 1, y: 0 }),
            good_coordinate(glam::Vec2 {
                x: 1.0174541,
                y: 0.00015234947,
            })
        );

        rotator = Rotator::new_from_angle(0, 3, 45.0);
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 1, y: 1 }),
            good_coordinate(glam::Vec2 { x: 1.0, y: 1.0 })
        );
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 0, y: 1 }),
            good_coordinate(glam::Vec2 {
                x: 0.29289448,
                y: 0.29289448
            })
        );
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 0, y: 0 }),
            good_coordinate(glam::Vec2 {
                x: f32::NAN,
                y: f32::NAN,
            }),
        );

        rotator = Rotator::new_from_angle(0, 3, 180.0);
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 1, y: 1 }),
            good_coordinate(glam::Vec2 { x: 1.0, y: 1.0 })
        );
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 2, y: 1 }),
            good_coordinate(glam::Vec2 {
                x: 0.0,
                y: 0.9999983
            })
        );
        expect_that!(
            rotator.rotate_coordinate(glam::UVec2 { x: 2, y: 2 }),
            good_coordinate(glam::Vec2 {
                x: f32::NAN,
                y: f32::NAN,
            }),
        );
    }

    #[gtest]
    fn rotate_dem4_coordinate() {
        let mut rotator = Rotator::new_from_angle(0, 4, 45.0);
        expect_eq!(
            rotator.rotate_coordinate(glam::UVec2 { x: 2, y: 2 }),
            glam::Vec2 {
                x: 1.4999988,
                y: 2.2071068,
            }
        );

        // Note how that without the shift the coordinate would be rasterised differently. Namely,
        // 1.5 is rounded to 2.0, so the rasterised coordinate doesn't actually change.
        rotator = Rotator::new_from_angle(0, 4, 45.0 - ANGLE_SHIFT);
        expect_eq!(
            rotator.rotate_coordinate(glam::UVec2 { x: 2, y: 2 }),
            glam::Vec2 {
                x: 1.5,
                y: 2.2071068,
            }
        );
    }

    #[gtest]
    fn rotate_dem3_id() {
        let mut rotator = Rotator::new_from_angle(0, 3, 0.0);
        assert_eq!(rotator.rotate_dem_id(), NOOP_DEM_ID);

        rotator = Rotator::new_from_angle(1, 3, 0.0);
        assert_eq!(rotator.rotate_dem_id(), 1);

        rotator = Rotator::new_from_angle(1, 3, 45.0);
        assert_eq!(rotator.rotate_dem_id(), 2);

        rotator = Rotator::new_from_angle(4, 3, 90.0);
        assert_eq!(rotator.rotate_dem_id(), 4);

        rotator = Rotator::new_from_angle(5, 3, 90.0);
        assert_eq!(rotator.rotate_dem_id(), 7);

        rotator = Rotator::new_from_angle(0, 3, 135.0);
        assert_eq!(rotator.rotate_dem_id(), NOOP_DEM_ID);

        rotator = Rotator::new_from_angle(1, 3, 135.0);
        assert_eq!(rotator.rotate_dem_id(), 8);
    }

    #[gtest]
    fn rotate_dem4_id() {
        let mut rotator = Rotator::new_from_angle(0, 4, 0.0);
        assert_eq!(rotator.rotate_dem_id(), NOOP_DEM_ID);

        rotator = Rotator::new_from_angle(10, 4, 0.0);
        assert_eq!(rotator.rotate_dem_id(), 10);

        rotator = Rotator::new_from_angle(10, 4, 45.0);
        assert_eq!(rotator.rotate_dem_id(), 9);
    }
}
