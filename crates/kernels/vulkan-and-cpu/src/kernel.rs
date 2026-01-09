//! Total Viewsheds kernel. The heart of the calculations.

#[cfg(not(target_arch = "spirv"))]
use crate::rotation::ANGLE_SHIFT;

use crate::{elevations::Elevations, ring_data::RingData, rotation::NOOP_DEM_ID};

/// Ensure that the first point from the point of view is always visible
const MAX_ANGLE: f32 = -2000.0;

#[expect(clippy::doc_markdown, reason = "PoV is just a shorthand")]
/// Normalisation constant: due to repeated exposure of closer points and
/// infrequent exposure of points further away.
///
/// This normalises surfaces due to repeated visibility over sectors.
/// For example points near the PoV may be counted 180 times giving
/// an unrealistic aggregation of total surfaces. This trig operation
/// (0.017 is tan(1 degree) where 1 degree is the size of each sector)
/// considers adjacent points as 1/57th, points that are about 57 cells
/// away as 1 and points further away as increasingly over 1.
/// Bear in my mind that due to the parallel nature of the band of sight,
/// distant points may not even be swept for a given PoV, therefore it
/// is an approximation to over-compensate the surface area of very
/// distant points.
///
/// Ultimately the reasoning for such normalisation is not so much to
/// measure true surface areas, but to ensure as much as possible that all
/// PoVs are treated consistently so that overlapping TVS raster map tiles
/// do not have visual artefacts.
const TAN_ONE_RADIAN: f32 = 0.017_453_3;

/// Diameter of the Earth in meters. So that some points are not visible simply
/// by virtue of the earth's spherical shape.
pub const EARTH_DIAMETER: f32 = 12_742_000.0;

#[expect(
    clippy::exhaustive_structs,
    reason = "These are only intended to be used within our workspace"
)]
/// GPU buffers
pub struct Buffers<'buffers> {
    /// Constants for the calculations.
    pub constants: &'buffers crate::constants::Constants,
    /// Every single rotated DEM point's elevation.
    pub elevations: &'buffers [f32],
    /// Array for final TVS values. Usually 1/8th the size of DEM.
    pub cumulative_surfaces: &'buffers mut [f32],
    /// Array for recording longest lines of sight.
    pub longest_lines: &'buffers mut [f32],
    /// Array for recording viewshed visibility regions.
    pub ring_data: &'buffers mut [u32],
}

/// Kernel state
pub struct Kernel {
    /// The TVS ID in its non-rotated coordinates.
    original_tvs_id: usize,
    /// The TVS ID in rotated coordinates.
    rotated_tvs_id: u32,
    /// The current maximum angle between the observer and an elevation point.
    max_angle: f32,
    /// Is the current elevation visible.
    is_currently_visible: bool,
    /// Is the previous elevation visible.
    is_previously_visible: bool,
    /// Is region of visibility closing.
    is_closing: bool,
    /// Keep track of the amount of the earth visible from this particular band
    band_surface: f32,
    /// Keep track of the longest line.
    longest_line: f32,
    /// Keep track of the visibility regions needed to reconstruct individual viewsheds.
    ring_data: RingData,
    /// Keep track of where to get the next elevation in elevations buffer.
    elevations: Elevations,
    /// Refraction constant.
    ///
    /// A good brief overview of refraction in viewshed analysis:
    /// <https://pro.arcgis.com/en/pro-app/latest/tool-reference/3d-analyst/how-line-of-sight-works.htm>
    /// A more in-depth analysis of how refraction can vary with altitude:
    /// <https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2010JD014067>
    refraction: f32,
}

impl Kernel {
    #[inline]
    /// Instantiate.
    fn new(kernel_id: u32, buffers: &Buffers) -> Self {
        let half_total_bands = buffers.constants.total_bands.div_euclid(2);

        // Translate a kernel ID to a TVS ID
        let rotated_tvs_id: u32;
        let direction = if kernel_id < half_total_bands {
            rotated_tvs_id = kernel_id;
            crate::elevations::Direction::Forward
        } else {
            rotated_tvs_id = kernel_id - half_total_bands;
            crate::elevations::Direction::Backward
        };

        let chocolate_id =
            crate::chocolate_box::chocolate_id_from_tvs_id(rotated_tvs_id, buffers.constants);

        let original_tvs_id = crate::rotation::Rotator::anti_rotate_index_from_cached_trig(
            rotated_tvs_id,
            buffers.constants.tvs_width,
            buffers.constants.sine,
            buffers.constants.cosine,
        );

        Self {
            original_tvs_id,
            max_angle: MAX_ANGLE,
            rotated_tvs_id,
            is_currently_visible: true,
            is_previously_visible: true,
            is_closing: false,
            band_surface: 0.0,
            ring_data: RingData::new(kernel_id, buffers.constants.reserved_rings_per_band),
            elevations: Elevations::new(
                buffers.elevations,
                direction,
                chocolate_id,
                buffers.constants.observer_height,
            ),
            longest_line: 0.0,
            refraction: buffers.constants.refraction,
        }
    }

    /// The kernel
    #[inline]
    pub fn run(kernel_id: u32, buffers: &mut Buffers) {
        let mut runner = Self::new(kernel_id, buffers);
        if runner.original_tvs_id == NOOP_DEM_ID {
            return;
        }

        for index in 0..buffers.constants.max_los_as_points {
            runner.kernel(buffers, index);
        }

        // Close any ring sectors prematurely cut off by a restricted line of sight.
        if buffers.constants.is_ring_data() && runner.is_currently_visible && !runner.is_closing {
            runner
                .ring_data
                .save(buffers.ring_data, buffers.constants.max_los_as_points);
        }

        if buffers.constants.is_ring_data() {
            runner.ring_data.finish(buffers.ring_data);
        }

        // TODO: Not thread safe because forward and backward invocations for the same point could
        // theoretically update the value at the same time.
        {
            // Accumulate surfaces for a given TVS ID.
            if buffers.constants.is_total_surfaces() {
                buffers.cumulative_surfaces[runner.original_tvs_id] += runner.band_surface;
            }

            // Save the longest line of sight for the given TVS ID.
            if buffers.constants.is_longest_lines() {
                let current_longest = buffers.longest_lines[runner.original_tvs_id];
                if runner.longest_line > current_longest.abs() {
                    if runner.elevations.is_backward() {
                        runner.longest_line = -runner.longest_line;
                    }

                    buffers.longest_lines[runner.original_tvs_id] = runner.longest_line;
                }

                #[expect(
                    clippy::float_cmp,
                    reason = "They are both from the same bits in memory"
                )]
                // For consistency with the CPU kernel we always want the first ever occurrence
                // of a longest line to take precedence. However in this kernel we interleave the
                // forward lines (0-179°) angles with the backward lines (180-359°). This means we
                // have to have this awkward check here where a forward lines takes precedence over
                // an equally long backward line.
                let is_same_length_line = runner.longest_line == current_longest.abs();
                if runner.elevations.is_forward() && is_same_length_line {
                    buffers.longest_lines[runner.original_tvs_id] = runner.longest_line;
                }
            }
        }
    }

    /// The kernel
    #[inline]
    fn kernel(&mut self, buffers: &mut Buffers, index: u32) {
        // TODO: does getting these all at once before the loop give a speed up?
        let elevation_delta = self.elevations.next(buffers.elevations);

        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "⚠️ Need to verify that distance never reaches the limits of f32."
        )]
        let distance = (index + 1) as f32 * buffers.constants.scale;

        // The actual visibility calculation.
        // TODO:
        //   * Currently it's an approximation by not using arctan.
        //   * Is there a performance gain to be had from only checking for an
        //     increase in elevation as a trigger for the full angle calculation?
        //   * Is this safe for `f32`? At what point does it break down?
        let curvature_correction = (distance * distance * (self.refraction - 1.0)) / EARTH_DIAMETER;
        let angle = (elevation_delta + curvature_correction) / distance;

        //                            5              |-
        //                        4 .-`-. 6          |-
        //            1       3  .-`     `-.  7      |-
        //   o    0 .-`-. 2   .-`           `-.      |- Elevation deltas
        //   /\  .-`     `-.-`                 `-.8  |-
        //    |                                      |-
        //    |---|---|---|---|---|---|---|---|---|
        //    |       Distance deltas
        //    |
        //   PoV  --------> direction of band point iterations
        //
        // Notice how only points 1, 4 and 5 increase the angle between the viewer and
        // the land and therefore can be considered visible.
        self.is_currently_visible = angle > self.max_angle;

        if buffers.constants.is_total_surfaces() && self.is_currently_visible {
            // TODO: Can this be refactored into a single calculation at the closing
            //       of a ring sector?
            self.band_surface += distance * TAN_ONE_RADIAN;
        }

        if buffers.constants.is_longest_lines() && self.is_currently_visible {
            self.longest_line = distance;
        }

        if buffers.constants.is_ring_data() {
            let is_opening = self.is_currently_visible && !self.is_previously_visible;
            self.is_closing = self.is_previously_visible && !self.is_currently_visible;

            if is_opening {
                self.ring_data.save(buffers.ring_data, index);
            }

            if self.is_closing {
                self.ring_data.save(buffers.ring_data, index);
            }
        }

        // Prepare for the next iteration.
        self.is_previously_visible = self.is_currently_visible;
        self.max_angle = f32::max(angle, self.max_angle);
    }

    #[cfg(not(target_arch = "spirv"))]
    #[expect(
        dead_code,
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "Used for debugging"
    )]
    /// Check if the current invocation meets the given criteria.
    fn is_debug_state(&self, buffers: &Buffers, angle: f32, pov_id: usize) -> bool {
        let angle_to_debug = angle + ANGLE_SHIFT;
        let chocolate_id =
            crate::chocolate_box::chocolate_id_from_tvs_id(self.rotated_tvs_id, buffers.constants);
        let original_pov_id = crate::chocolate_box::Rotator::new_from_cached_trig(
            chocolate_id as u32,
            buffers.constants.dem_width,
            buffers.constants.tvs_width,
            buffers.constants.sine,
            buffers.constants.cosine,
        )
        .anti_rotate_chocolate_id_to_dem_id();

        angle_to_debug.to_radians().cos() == buffers.constants.cosine
            && angle_to_debug.to_radians().sin() == buffers.constants.sine
            && original_pov_id == pov_id
    }

    #[cfg(not(target_arch = "spirv"))]
    #[expect(dead_code, clippy::float_cmp, reason = "Used for debugging")]
    /// Get the angle from the trigonometry values.
    fn debug_angle(buffers: &Buffers) -> f32 {
        for sector in 0..180u16 {
            let angle = f32::from(sector) + ANGLE_SHIFT;
            if angle.to_radians().cos() == buffers.constants.cosine
                && angle.to_radians().sin() == buffers.constants.sine
            {
                return angle;
            }
        }
        f32::NAN
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    reason = "These are tests"
)]
#[cfg(test)]
mod test {
    use crate::rotation;

    use super::*;
    use googletest::prelude::*;

    enum TvsId {
        /// A TVS ID for a forward invocation.
        Forward(u32),
        /// A TVS ID for a backward invocation.
        Backward(u32),
    }

    fn invoke_default(directed_tvs_id: &TvsId, angle: f32) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        let constants = constants(angle);
        invoke(directed_tvs_id, angle, Some(constants))
    }

    fn invoke(
        directed_tvs_id: &TvsId,
        angle: f32,
        maybe_constants: Option<crate::constants::Constants>,
    ) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        let constants = maybe_constants.unwrap_or_else(|| constants(angle));
        let tvs_id = match directed_tvs_id {
            TvsId::Forward(id) | TvsId::Backward(id) => id,
        };

        let elevations = crate::tests::dems::bigger_dem();
        let dem_size = constants.dem_width.pow(2);
        let chocolate_box_size = dem_size.div_euclid(3);
        let mut rotated_elevations = vec![0.0; chocolate_box_size as usize];
        for chocolate_id in 0..chocolate_box_size {
            let rotator = crate::chocolate_box::Rotator::new_from_cached_trig(
                chocolate_id,
                constants.dem_width,
                constants.tvs_width,
                constants.sine,
                constants.cosine,
            );
            rotator.anti_rotate_value_nearest_neighbour(&elevations, &mut rotated_elevations);
        }

        let tvs_size = constants.tvs_width.pow(2) as usize;
        let mut cumulative_surfaces = vec![0.0; tvs_size];
        let mut ring_data = vec![0; reserved_ring_data()];
        let mut longest_lines = vec![0.0; tvs_size];

        let rotated_tvs_id =
            crate::rotation::Rotator::rotate_index(*tvs_id, constants.tvs_width, angle);

        let offset = match directed_tvs_id {
            TvsId::Forward(_) => 0,
            TvsId::Backward(_) => tvs_size as u32,
        };

        let mut buffers = Buffers {
            constants: &constants,
            elevations: &rotated_elevations,
            cumulative_surfaces: &mut cumulative_surfaces,
            longest_lines: &mut longest_lines,
            ring_data: &mut ring_data,
        };

        Kernel::run(rotated_tvs_id as u32 + offset, &mut buffers);

        (
            cumulative_surfaces.clone(),
            ring_data.clone(),
            longest_lines.clone(),
        )
    }

    fn reserved_ring_data() -> usize {
        (constants(0.0).total_bands * constants(0.0).reserved_rings_per_band) as usize
    }

    fn constants(angle: f32) -> crate::constants::Constants {
        let width = 4;
        let trig = rotation::Rotator::calculate_trig(angle);
        crate::constants::Constants {
            total_bands: (width * width) * 2,
            max_los_as_points: width,
            dem_width: width * 3,
            tvs_width: width,
            scale: 1.0,
            observer_height: 1.65,
            reserved_rings_per_band: 5,
            process: crate::constants::Flag::TotalSurfaces.bit()
                | crate::constants::Flag::RingData.bit()
                | crate::constants::Flag::LongestLines.bit(),
            sine: trig.0,
            cosine: trig.1,
            ..Default::default()
        }
    }

    fn empty_tvs() -> Vec<f32> {
        vec![0.0; (constants(0.0).tvs_width.pow(2)) as usize]
    }

    fn empty_ring_data() -> Vec<u32> {
        vec![0; reserved_ring_data()]
    }

    fn expect_tvs(directed_tvs_id: &TvsId, tvs: &[f32], expected: f32) {
        let tvs_id = *match directed_tvs_id {
            TvsId::Forward(id) | TvsId::Backward(id) => id,
        };
        let mut tvs_expected = empty_tvs();
        tvs_expected[tvs_id as usize] = expected;
        expect_eq!(tvs, tvs_expected);
    }

    fn expect_ring_data(
        directed_tvs_id: &TvsId,
        angle: f32,
        ring_data: &[u32],
        expected_dem_ids: Vec<u32>,
    ) {
        let constants = constants(angle);
        let tvs_id = match directed_tvs_id {
            TvsId::Forward(id) | TvsId::Backward(id) => id,
        };
        let tvs_size = constants.tvs_width.pow(2) as usize;
        let rotated_tvs_id =
            crate::rotation::Rotator::rotate_index(*tvs_id, constants.tvs_width, angle);
        let mut ring_expected = empty_ring_data();
        let offset = match directed_tvs_id {
            TvsId::Forward(_) => 0,
            TvsId::Backward(_) => tvs_size,
        };
        let start = (rotated_tvs_id + offset) * constants.reserved_rings_per_band as usize;
        let rings = expected_dem_ids.len() + 1;
        let mut expected = vec![rings as u32];
        expected.extend(expected_dem_ids);
        ring_expected.splice(start..start + rings, expected);
        expect_eq!(ring_data, ring_expected);
    }

    #[gtest]
    fn invocation_at_id5_0_degrees_forward() {
        let tvs_id = TvsId::Forward(5);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id10_0_degrees_forward() {
        let tvs_id = TvsId::Forward(10);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, 4.0);
    }

    #[gtest]
    fn invocation_at_id5_45_degrees_forward() {
        let tvs_id = TvsId::Forward(5);
        let angle = 45.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id10_90_degrees_forward() {
        let tvs_id = TvsId::Forward(10);
        let angle = 90.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, 4.0);
    }

    #[gtest]
    fn invocation_at_id5_135_degrees_forward() {
        let tvs_id = TvsId::Forward(5);
        let angle = 135.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id6_46_degrees_forward() {
        let tvs_id = TvsId::Forward(6);
        let angle = 46.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, 4.0);
    }

    #[gtest]
    fn invocation_at_id5_0_degrees_backward() {
        let tvs_id = TvsId::Backward(5);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, -1.0);
    }

    #[gtest]
    fn invocation_at_id10_0_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        // TODO: I think this result clearly shows that we should be closing the ring sector for
        // the _previous_ DEM ID?
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -4.0);
    }

    #[gtest]
    fn invocation_at_id10_45_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 45.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -4.0);
    }

    #[gtest]
    fn invocation_at_id5_90_degrees_backward() {
        let tvs_id = TvsId::Backward(5);
        let angle = 90.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, -1.0);
    }

    #[gtest]
    fn invocation_at_id10_135_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 135.0;
        let (surfaces, rings, lines) = invoke_default(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -4.0);
    }

    #[gtest]
    fn refraction_affects_visibility() {
        let tvs_id = TvsId::Backward(10);
        let angle = 135.0;
        let mut constants = constants(angle);
        constants.refraction = -EARTH_DIAMETER;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle, Some(constants));

        expect_tvs(&tvs_id, &surfaces, 0.17453301);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -4.0);
    }
}
