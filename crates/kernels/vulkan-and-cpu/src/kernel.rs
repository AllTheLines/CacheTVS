//! Total Viewsheds kernel. The heart of the calculations.

#[cfg(not(target_arch = "spirv"))]
use crate::rotation::ANGLE_SHIFT;

use crate::{ring_data::RingData, rotation::NOOP_DEM_ID};

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

/// So that some points are not visible simply by virtue of the earth's spherical
/// shape.
const EARTH_RADIUS_DOUBLED: f32 = 12_742_000.0;

#[cfg_attr(not(target_arch = "spirv"), derive(Debug, PartialEq, Eq))]
#[expect(
    clippy::exhaustive_enums,
    reason = "There's never going to be more directions"
)]
/// The direction of a band from the observer's point of view. Whether it points North or South is
/// not relevant, they're just opposite to each other.
pub enum BandDirection {
    /// A band facing forward from the observer.
    Forward,
    /// A band facing behind the observer.
    Backward,
}

struct Elevations<'kernel> {
    /// Every single DEM point's elevation.
    elevations: &'kernel [f32],
    /// A record of the last valid elevation. Used to fill "nodata" regions.
    ///
    /// Note that even though most, if not all, of these "nodata" regions occur at sea, that
    /// doesn't necessarily mean that a default elevation of 0 is best. There are various reasons
    /// why the sea isn't always at a perfect 0 elevation, which we won't go into here. The
    /// point being that we need to smoothly transition into "nodata" regions to avoid visual
    /// artefacts.
    last_valid_elevation: f32,
}

impl Elevations<'_> {
    /// Get a single elevation from the rotated DEM.
    fn get(&mut self, rotated_dem_id: usize) -> f32 {
        let elevation = self.elevations[rotated_dem_id];
        let is_invalid = elevation < -1000.0 || elevation.is_nan();
        if is_invalid {
            self.last_valid_elevation
        } else {
            self.last_valid_elevation = elevation;
            elevation
        }
    }
}

/// Kernel state
pub struct Kernel<'kernel> {
    /// The identifier that decides which point of view and band direction to calculate.
    /// The current known longest line of sight is 538km. Let's say that the actual longest could reach
    /// 600km. So for a DEM of 30m resolution, the maximum number of points we could be dealing with
    /// is: `(500,000 / 30)^2 = 400,000,000`. The max of `u32` is 4,294,967,295.
    kernel_id: u32,
    /// Constants for the calculations.
    pub constants: &'kernel crate::constants::Constants,
    /// The TVS ID in rotated coordinates.
    rotated_tvs_id: u32,
    /// Whether going forwards or backwards along the band of sight.
    band_direction: BandDirection,
    /// Array for final TVS values. Usually 1/8th the size of DEM.
    cumulative_surfaces: &'kernel mut [f32],
    /// Array for recording longest lines of sight.
    longest_lines: &'kernel mut [f32],
}

impl<'kernel> Kernel<'kernel> {
    #[inline]
    /// Instantiate.
    const fn new(
        kernel_id: u32,
        constants: &'kernel crate::constants::Constants,
        cumulative_surfaces: &'kernel mut [f32],
        longest_lines: &'kernel mut [f32],
    ) -> Self {
        let half_total_bands = constants.total_bands.div_euclid(2);

        // Translate a kernel ID to a TVS ID
        let rotated_tvs_id: u32;
        let band_direction = if kernel_id < half_total_bands {
            rotated_tvs_id = kernel_id;
            BandDirection::Forward
        } else {
            rotated_tvs_id = kernel_id - half_total_bands;
            BandDirection::Backward
        };

        Self {
            kernel_id,
            constants,
            rotated_tvs_id,
            band_direction,
            cumulative_surfaces,
            longest_lines,
        }
    }

    /// Calculate the Point of View coordinate in rotated space.
    const fn rotated_pov_id(&self) -> usize {
        let pov_x = self.rotated_tvs_id.rem_euclid(self.constants.tvs_width)
            + self.constants.max_los_as_points;
        let pov_y = self.rotated_tvs_id.div_euclid(self.constants.tvs_width)
            + self.constants.max_los_as_points;
        #[expect(
            clippy::as_conversions,
            reason = "This needs to run on the GPU where fallibility isn't possible"
        )]
        let rotated_pov_id = ((pov_y * self.constants.dem_width) + pov_x) as usize;

        rotated_pov_id
    }

    /// The kernel
    #[inline]
    pub fn run(
        kernel_id: u32,
        constants: &'kernel crate::constants::Constants,
        elevations: &'kernel [f32],
        rings_data: &'kernel mut [u32],
        cumulative_surfaces: &'kernel mut [f32],
        longest_lines: &'kernel mut [f32],
    ) {
        let mut runner = Self::new(kernel_id, constants, cumulative_surfaces, longest_lines);
        runner.kernel(elevations, rings_data);
    }

    /// The kernel
    #[inline]
    fn kernel(&mut self, elevations: &'kernel [f32], rings_data: &'kernel mut [u32]) {
        // This can't be placed on `Self` because on Vulkan any writes to it then cause an error
        // about access out of bounds memory. It's a `rust-gpu` thing, I should make an issue for
        // it.
        let mut rings = RingData::new(
            rings_data,
            self.kernel_id,
            self.constants.reserved_rings_per_band,
        );

        // This can't be placed on `Self` because on Vulkan any writes to it then cause an error
        // about access out of bounds memory. It's a `rust-gpu` thing, I should make an issue for
        // it.
        let mut elevation = Elevations {
            elevations,
            last_valid_elevation: 0.0,
        };

        let rotator = crate::rotation::Rotator::new_from_cached_trig(
            self.rotated_tvs_id,
            self.constants.tvs_width,
            self.constants.sine,
            self.constants.cosine,
        );
        let original_tvs_id = rotator.rotate_dem_id();
        if original_tvs_id == NOOP_DEM_ID {
            return;
        }

        let mut max_angle = MAX_ANGLE;
        let mut is_currently_visible = true;
        let mut is_previously_visible = true;
        let mut closing = false;

        // Keep track of the amount of the earth visible from this particular band
        let mut band_surface = 0.0;
        // Keep track of the longest line.
        let mut longest_line = 0.0;

        // The DEM ID will change as we loop, but the PoV won't. For now we need the PoV
        // ID to start the reconstruction of a unique band from the band delta template.
        let rotated_pov_id = self.rotated_pov_id();
        let mut rotated_dem_id = rotated_pov_id;
        let pov_elevation = elevation.get(rotated_dem_id) + self.constants.observer_height;

        // The kernel's kernel. The most critical code of all.
        for index in 0..=self.constants.max_los_as_points {
            // Derive the new DEM ID.
            rotated_dem_id = match self.band_direction {
                BandDirection::Forward => rotated_dem_id + 1,
                BandDirection::Backward => rotated_dem_id - 1,
            };

            // Pull the actual data needed to make a visibility calculation from global memory.
            // TODO: does getting these all at once before the loop give a speed up?
            let elevation_delta = elevation.get(rotated_dem_id) - pov_elevation;

            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "⚠️ Need to verify that distance never reaches the limits of f32."
            )]
            let distance = (index + 1) as f32 * self.constants.scale;

            // The actual visibility calculation.
            // Note the adjustment for curvature of the earth. It is merely a crude
            // approximation using the spherical earth model.
            // TODO:
            //   * Currently it's an approximation by not using arctan.
            //   * Account for refraction.
            //   * Is there a performance gain to be had from only checking for an
            //     increase in elevation as a trigger for the full angle calculation?
            //   * Is this safe for `f32`? At what point does it break down?
            let angle = (elevation_delta / distance) - (distance / EARTH_RADIUS_DOUBLED);

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
            is_currently_visible = angle > max_angle;

            // Here we consider the *previous* visibility to decide whether this is the
            // beginning or ending of a visible region.
            let opening = is_currently_visible && !is_previously_visible;
            closing = is_previously_visible && !is_currently_visible;

            if self.constants.is_total_surfaces() && is_currently_visible {
                // TODO: Can this be refactored into a single calculation at the closing
                //       of a ring sector?
                band_surface += distance * TAN_ONE_RADIAN;
            }

            if self.constants.is_longest_lines() && is_currently_visible {
                longest_line = distance;
            }

            if self.constants.is_ring_data() {
                if opening {
                    rings.save(index);
                }

                if closing {
                    rings.save(index);
                }
            }

            // Prepare for the next iteration.
            is_previously_visible = is_currently_visible;
            max_angle = f32::max(angle, max_angle);
        }

        // Close any ring sectors prematurely cut off by a restricted line of sight.
        if self.constants.is_ring_data() && is_currently_visible && !closing {
            rings.save(self.constants.max_los_as_points);
        }

        if self.constants.is_ring_data() {
            rings.finish();
        }

        // TODO: Not thread safe because forward and backward invocations for the same point could
        // theoretically update the value at the same time.
        {
            // Accumulate surfaces for a given TVS ID.
            if self.constants.is_total_surfaces() {
                self.cumulative_surfaces[original_tvs_id] += band_surface;
            }

            // Save the longest line of sight for the given TVS ID.
            if self.constants.is_longest_lines() {
                #[expect(
                    clippy::as_conversions,
                    reason = "This needs to run on the GPU where fallibility isn't possible"
                )]
                let current_longest = self.longest_lines[self.rotated_tvs_id as usize];
                if longest_line > current_longest.abs() {
                    if matches!(self.band_direction, BandDirection::Backward) {
                        longest_line = -longest_line;
                    }
                    self.longest_lines[original_tvs_id] = longest_line;
                }
            }
        }
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
    fn is_debug_state(&self, angle: f32, pov_id: usize) -> bool {
        let angle_to_debug = angle + ANGLE_SHIFT;
        let rotated_pov_id = self.rotated_pov_id();
        let original_pov_id = crate::rotation::Rotator::new_from_cached_trig(
            rotated_pov_id as u32,
            self.constants.dem_width,
            self.constants.sine,
            self.constants.cosine,
        )
        .rotate_dem_id();
        angle_to_debug.to_radians().cos() == self.constants.cosine
            && angle_to_debug.to_radians().sin() == self.constants.sine
            && original_pov_id == pov_id
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

    fn invoke(directed_tvs_id: &TvsId, angle: f32) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        let constants = constants(angle);
        let tvs_id = match directed_tvs_id {
            TvsId::Forward(id) | TvsId::Backward(id) => id,
        };

        let elevations: Vec<f32> = crate::tests::dems::bigger_dem()
            .iter()
            .map(|elevation| f32::from(*elevation))
            .collect();
        let dem_size = constants.dem_width.pow(2);
        let mut rotated_elevations = vec![0.0; constants.dem_width.pow(2) as usize];
        for dem_id in 0..dem_size {
            let rotator = crate::rotation::Rotator::new_from_cached_trig(
                dem_id,
                constants.dem_width,
                constants.sine,
                constants.cosine,
            );
            rotator.rotate_value_nearest_neighbour(&elevations, &mut rotated_elevations);
        }

        let tvs_size = constants.tvs_width.pow(2) as usize;
        let mut cumulative_surfaces = vec![0.0; tvs_size];
        let mut ring_data = vec![0; reserved_ring_data()];
        let mut longest_lines = vec![0.0; tvs_size];

        let rotator = crate::rotation::Rotator::new_from_angle(*tvs_id, constants.tvs_width, angle);
        let rotated_tvs_id = rotator.anti_rotate_dem_id() as u32;

        let offset = match directed_tvs_id {
            TvsId::Forward(_) => 0,
            TvsId::Backward(_) => tvs_size as u32,
        };

        Kernel::run(
            rotated_tvs_id + offset,
            &constants,
            &rotated_elevations,
            &mut ring_data,
            &mut cumulative_surfaces,
            &mut longest_lines,
        );

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
        let rotator = crate::rotation::Rotator::new_from_angle(*tvs_id, constants.tvs_width, angle);
        let rotated_tvs_id = rotator.anti_rotate_dem_id() as usize;
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
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id10_0_degrees_forward() {
        let tvs_id = TvsId::Forward(10);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.2617995);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, 5.0);
    }

    #[gtest]
    fn invocation_at_id5_45_degrees_forward() {
        let tvs_id = TvsId::Forward(5);
        let angle = 45.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id10_90_degrees_forward() {
        let tvs_id = TvsId::Forward(10);
        let angle = 90.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.2617995);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, 5.0);
    }

    #[gtest]
    fn invocation_at_id5_135_degrees_forward() {
        let tvs_id = TvsId::Forward(5);
        let angle = 135.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, 1.0);
    }

    #[gtest]
    fn invocation_at_id5_0_degrees_backward() {
        let tvs_id = TvsId::Backward(5);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, -1.0);
    }

    #[gtest]
    fn invocation_at_id10_0_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 0.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.2617995);
        // TODO: I think this result clearly shows that we should be closing the ring sector for
        // the _previous_ DEM ID?
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -5.0);
    }

    #[gtest]
    fn invocation_at_id10_45_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 45.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.2617995);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -5.0);
    }

    #[gtest]
    fn invocation_at_id5_90_degrees_backward() {
        let tvs_id = TvsId::Backward(5);
        let angle = 90.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.0174533);
        expect_ring_data(&tvs_id, angle, &rings, vec![1]);
        expect_tvs(&tvs_id, &lines, -1.0);
    }

    #[gtest]
    fn invocation_at_id10_135_degrees_backward() {
        let tvs_id = TvsId::Backward(10);
        let angle = 135.0;
        let (surfaces, rings, lines) = invoke(&tvs_id, angle);

        expect_tvs(&tvs_id, &surfaces, 0.2617995);
        expect_ring_data(&tvs_id, angle, &rings, vec![4]);
        expect_tvs(&tvs_id, &lines, -5.0);
    }
}
