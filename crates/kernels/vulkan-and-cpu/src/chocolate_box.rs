//! For every sector (angle of the viewshed) we only need to rotate a horizontal rectangular
//! region of the DEM. We refer to this as the "Chocolate Box". This saves both computation and
//! memory.
//!
//! Consider this _rotated_ DEM, where:
//!   `·` is unused elevation data
//!   `x` is auxilliary data
//!   `o` is the TVS
//!
//!   · · · · · · · · ·
//!   · · · · · · · · ·
//!   · · · · · · · · ·
//!   x x x o o o x x x
//!   x x x o o o x x x
//!   x x x o o o x x x
//!   · · · · · · · · ·
//!   · · · · · · · · ·
//!   · · · · · · · · ·
//!
//! The Chocolate Box is the region made up of the `x` and `o`. Note that this is how it looks for
//! every single angle. The difference is that for each angle the source of the elevations is
//! different.

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Rotate elevations to the Chocolate Box.
pub struct Rotator {
    /// A normal rotator.
    rotator: super::rotation::Rotator,
    /// The number of points before reaching the Chocolate Box.
    offset: u32,
}

impl Rotator {
    #[inline]
    #[must_use]
    /// Instantiate with precalculated sine and cosine. Used by GPU to shave off some cycles from
    /// not having to do the trigonometry.
    pub fn new_from_cached_trig(
        index: u32,
        dem_width: u32,
        tvs_width: u32,
        sine: f32,
        cosine: f32,
    ) -> Self {
        let rotator =
            super::rotation::Rotator::new_from_cached_trig(index, tvs_width * 3, sine, cosine);
        Self {
            rotator,
            offset: offset(dem_width, tvs_width),
        }
    }

    #[inline]
    #[must_use]
    /// Instantiate with an angle.
    pub fn new_from_angle(index: u32, dem_width: u32, tvs_width: u32, angle: f32) -> Self {
        let rotator = super::rotation::Rotator::new_from_angle(index, tvs_width * 3, angle);
        Self {
            rotator,
            offset: offset(dem_width, tvs_width),
        }
    }

    #[inline]
    /// Rotate a single point using nearest neighbour.
    pub fn anti_rotate_value_nearest_neighbour(
        &self,
        elevations_in: &[i16],
        elevations_out: &mut [f32],
    ) {
        let rotated_dem_id = self.anti_rotate_chocolate_id_to_dem_id();
        let elevation = if rotated_dem_id == crate::rotation::NOOP_DEM_ID {
            f32::NAN
        } else {
            Self::get_elevation(elevations_in, rotated_dem_id)
        };

        #[expect(clippy::as_conversions, reason = "Index will always fit in `usize`")]
        {
            elevations_out[self.rotator.index as usize] = elevation;
        }
    }

    #[inline]
    /// Rotate a single point using sampled interpolation
    pub fn anti_rotate_value_bilinear(&self, elevations_in: &[i16], elevations_out: &mut [f32]) {
        let coordinate = self.chocolate_id_to_coord();
        let rotated = self.rotator.anti_rotate_coordinate(coordinate);
        let interpolated = if rotated == crate::rotation::NOOP_COORDINATE {
            f32::NAN
        } else {
            self.bilinear_sample(elevations_in, rotated)
        };

        #[expect(clippy::as_conversions, reason = "Index will always fit in `usize`")]
        {
            elevations_out[self.rotator.index as usize] = interpolated;
        }
    }

    /// Get an elevation from the main DEM array.
    fn get_elevation(elevations_in: &[i16], dem_id: usize) -> f32 {
        let elevation = f32::from(elevations_in[dem_id]);
        if elevation > crate::elevations::NODATA {
            elevation
        } else {
            f32::NAN
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
    fn get_pixel(&self, elevations: &[i16], x: f32, y: f32) -> f32 {
        let width_f32 = self.rotator.width as f32;
        let clamped_x = x.clamp(0.0, width_f32 - 1.0) as usize;
        let clamped_y = y.clamp(0.0, width_f32 - 1.0) as usize;
        let id = clamped_y * self.rotator.width as usize + clamped_x;
        Self::get_elevation(elevations, id)
    }

    #[inline]
    #[must_use]
    /// Bilinear interpolation.
    ///
    /// This seems to give clearly worse results. Perhaps try larger interpolation kernels?
    pub fn bilinear_sample(&self, elevations: &[i16], coordinate: glam::Vec2) -> f32 {
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
    /// Rotate or anti rotate a Chocolate ID to its DEM ID.
    fn private_rotate_chocolate_id_to_dem_id(&self, anti_rotate: bool) -> usize {
        let coordinate = self.chocolate_id_to_coord();
        self.rotator
            .rotate_coordinate_to_index(coordinate, anti_rotate)
    }

    #[inline]
    #[must_use]
    /// Rotate a DEM ID.
    pub fn anti_rotate_chocolate_id_to_dem_id(&self) -> usize {
        self.private_rotate_chocolate_id_to_dem_id(true)
    }

    #[inline]
    #[must_use]
    /// Convert a Chocolate ID index to an x,y coordinate.
    pub const fn chocolate_id_to_coord(&self) -> glam::UVec2 {
        let index = self.offset + self.rotator.index;
        self.rotator.index_to_coord(index)
    }
}

#[inline]
#[must_use]
/// The size of the Chocolate Box.
pub const fn size(dem_width: u32, tvs_width: u32) -> u32 {
    dem_width * tvs_width
}

#[inline]
#[must_use]
/// Where the chocolate box starts in the DEM.
pub const fn offset(dem_width: u32, tvs_width: u32) -> u32 {
    let x = 0;
    let y_centre = dem_width.div_euclid(2);
    let y = y_centre - tvs_width.div_euclid(2);
    (y * dem_width) + x
}

#[expect(
    clippy::as_conversions,
    reason = "This needs to run on the GPU where fallibility isn't possible"
)]
#[inline]
#[must_use]
/// Calculate where we are in the Chocolate Box elevations array.
pub const fn chocolate_id_from_tvs_id(
    rotated_tvs_id: u32,
    constants: &crate::constants::Constants,
) -> usize {
    let tvs_x = rotated_tvs_id.rem_euclid(constants.tvs_width);
    let tvs_y = rotated_tvs_id.div_euclid(constants.tvs_width);
    ((tvs_y * constants.dem_width) + (tvs_x + constants.tvs_width)) as usize
}

#[cfg(test)]
mod test {
    use core::f32;

    use super::*;
    use googletest::prelude::*;

    const NAN: f32 = f32::NAN;

    #[rustfmt::skip]
    const DEM3: [i16; 9] = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 8
    ];

    #[rustfmt::skip]
    const DEM6: [i16; 36] = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,11,
        12,13,14,15,16,17,
        18,19,20,21,22,23,
        24,25,26,27,28,29,
        30,31,32,33,34,35
    ];

    fn run_dem3(angle: f32) -> [f32; 3] {
        let dem_width = u32::try_from(DEM3.len().isqrt()).unwrap();
        let tvs_width = dem_width.div_euclid(3);
        let mut rotated = [0.0; 3];
        for chocolate_id in 0..super::size(dem_width, tvs_width) {
            let rotator = Rotator::new_from_angle(chocolate_id, dem_width, tvs_width, angle);
            rotator.anti_rotate_value_nearest_neighbour(&DEM3, &mut rotated);
        }

        rotated
    }

    fn run_dem6(angle: f32) -> [f32; 12] {
        let dem_width = u32::try_from(DEM6.len().isqrt()).unwrap();
        let tvs_width = dem_width.div_euclid(3);
        let mut rotated = [0.0; 12];
        for chocolate_id in 0..super::size(dem_width, tvs_width) {
            let rotator = Rotator::new_from_angle(chocolate_id, dem_width, tvs_width, angle);
            rotator.anti_rotate_value_nearest_neighbour(&DEM6, &mut rotated);
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
            3.0, 4.0, 5.0,
        ];
        assert_dem(&run_dem3(0.0), &expected);
    }

    #[gtest]
    fn rotate_by_0_dem6() {
        #[rustfmt::skip]
        let expected = [
            NAN, 13.0, 14.0, 15.0, 16.0, NAN,
            NAN, 19.0, 20.0, 21.0, 22.0, NAN
        ];
        assert_dem(&run_dem6(0.0), &expected);
    }

    #[gtest]
    fn rotate_by_45_dem3() {
        #[rustfmt::skip]
        let expected = [
            6.0, 4.0, 2.0,
        ];

        assert_dem(&run_dem3(45.0), &expected);
    }

    #[gtest]
    fn rotate_by_45_dem6() {
        #[rustfmt::skip]
        let expected = [
            NAN, 19.0, 20.0, 15.0, 9.0,  NAN,
            NAN, 26.0, 21.0, 21.0, 16.0, NAN
        ];
        assert_dem(&run_dem6(45.0001), &expected);
    }

    #[gtest]
    fn rotate_by_90_dem3() {
        #[rustfmt::skip]
        let expected = [
            7.0, 4.0, 1.0,
        ];

        assert_dem(&run_dem3(90.0), &expected);
    }

    #[gtest]
    fn rotate_by_135_dem3() {
        #[rustfmt::skip]
        let expected = [
            8.0, 4.0, 0.0,
        ];

        assert_dem(&run_dem3(135.0), &expected);
    }

    #[gtest]
    fn rotate_dem3_id() {
        let dem_width = 3u32;
        let tvs_width = dem_width.div_euclid(3);

        let mut chocolate = Rotator::new_from_angle(0, dem_width, tvs_width, 0.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 3);

        chocolate = Rotator::new_from_angle(1, dem_width, tvs_width, 0.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 4);

        chocolate = Rotator::new_from_angle(1, dem_width, tvs_width, 45.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 4);

        chocolate = Rotator::new_from_angle(4, dem_width, tvs_width, 90.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 5);

        chocolate = Rotator::new_from_angle(0, dem_width, tvs_width, 135.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 8);

        chocolate = Rotator::new_from_angle(1, dem_width, tvs_width, 135.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 4);

        chocolate = Rotator::new_from_angle(5, dem_width, tvs_width, 90.0);
        assert_eq!(
            chocolate.anti_rotate_chocolate_id_to_dem_id(),
            crate::rotation::NOOP_DEM_ID
        );
    }

    #[gtest]
    fn rotate_dem6_id() {
        let dem_width = 6u32;
        let tvs_width = dem_width.div_euclid(3);
        let mut chocolate = Rotator::new_from_angle(0, dem_width, tvs_width, 0.0);
        assert_eq!(
            chocolate.anti_rotate_chocolate_id_to_dem_id(),
            crate::rotation::NOOP_DEM_ID
        );

        chocolate = Rotator::new_from_angle(10, dem_width, tvs_width, 0.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 22);

        chocolate = Rotator::new_from_angle(1, dem_width, tvs_width, 0.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 13);

        chocolate = Rotator::new_from_angle(2, dem_width, tvs_width, 45.0);
        assert_eq!(chocolate.anti_rotate_chocolate_id_to_dem_id(), 14);
    }
}
