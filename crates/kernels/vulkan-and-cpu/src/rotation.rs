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
pub const ANGLE_SHIFT: f32 = -0.0001;

/// Cached data for rotating points.
pub struct Rotator {
    /// The index of the elevation point, whether in TVS, Chocolate Box or DEM.
    pub index: u32,
    /// The number of points in the width of the DEM/TVS.
    pub width: u32,
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
    pub fn new_from_cached_trig(index: u32, width: u32, sine: f32, cosine: f32) -> Self {
        Self {
            index,
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
    pub fn new_from_angle(index: u32, dem_width: u32, angle: f32) -> Self {
        let trig = Self::calculate_trig(angle);
        Self::new_from_cached_trig(index, dem_width, trig.0, trig.1)
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
    /// Rotate or anti rotate an index.
    fn private_rotate_index(&self, index: u32, is_anti_rotate: bool) -> usize {
        let coordinate = self.index_to_coord(index);
        self.rotate_coordinate_to_index(coordinate, is_anti_rotate)
    }

    #[inline]
    #[must_use]
    /// Rotate an index.
    pub fn rotate_index(index: u32, width: u32, angle: f32) -> usize {
        let rotator = Self::new_from_angle(index, width, angle);
        rotator.private_rotate_index(index, false)
    }

    #[inline]
    #[must_use]
    /// Anti-rotate an index.
    pub fn anti_rotate_index(index: u32, width: u32, angle: f32) -> usize {
        let rotator = Self::new_from_angle(index, width, angle);
        rotator.private_rotate_index(index, true)
    }

    #[inline]
    #[must_use]
    /// Rotate an index.
    pub fn anti_rotate_index_from_cached_trig(
        index: u32,
        width: u32,
        sine: f32,
        cosine: f32,
    ) -> usize {
        let rotator = Self::new_from_cached_trig(index, width, sine, cosine);
        rotator.private_rotate_index(index, true)
    }

    #[inline]
    #[must_use]
    /// Rotate or anti rotate a coordinate to its scalar index.
    pub fn rotate_coordinate_to_index(
        &self,
        coordinate: glam::UVec2,
        is_anti_rotate: bool,
    ) -> usize {
        let rotated_coord = if is_anti_rotate {
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
            self.coord_to_index(cast) as usize
        }
    }

    #[inline]
    #[must_use]
    /// Convert an index to an x,y coordinate.
    pub const fn index_to_coord(&self, index: u32) -> glam::UVec2 {
        let x = index.rem_euclid(self.width);
        let y = index.div_euclid(self.width);
        glam::UVec2 { x, y }
    }

    #[inline]
    #[must_use]
    /// Convert an x,y coordinate to a DEM ID.
    pub const fn coord_to_index(&self, coordinate: glam::UVec2) -> u32 {
        coordinate.y * self.width + coordinate.x
    }
}

#[expect(clippy::unreadable_literal, reason = "These are just tests")]
#[cfg(test)]
mod test {
    use core::f32;

    use crate::tests::matchers::good_coordinate;

    use super::*;
    use googletest::prelude::*;

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
                x: 1.5000012,
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
}
