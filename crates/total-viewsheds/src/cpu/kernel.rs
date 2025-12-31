use itertools::izip;

#[cfg(all(target_feature = "sse", target_feature = "sse2"))]
use std::arch::x86_64::{_mm_castps_si128, _mm_cmpgt_ps, _mm_max_ps};

#[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
use std::simd::cmp::SimdPartialOrd;

use std::simd::prelude::{SimdFloat, SimdInt};
use std::simd::{f32x4, Mask, Simd};

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

const TAN_ONE_RAD: f32 = 0.017_453_3;

/// `generate_distances` generates the distance from
fn generate_distances(max_los: usize) -> (Vec<f32>, Vec<f32>) {
    (1..=max_los)
        .map(|step| {
            (
                (step * 100) as f32,
                ((step * 100) as f32) / EARTH_RADIUS_SQUARED,
            )
        })
        .unzip()
}

/// `ViewShed` holds all reusable structures for the total viewshed computation
/// TODO: probably rename this
struct ViewShed {
    max_los: usize,
    angles: Vec<f32>,
    distances: Vec<f32>,
    adjustments: Vec<f32>,
}

#[cfg(all(target_feature = "sse", target_feature = "sse2"))]
fn simd_max(lhs: f32x4, rhs: f32x4) -> f32x4 {
    // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
    // thus allowing this to be non IEEE754 compliant
    unsafe { _mm_max_ps(lhs.into(), rhs.into()).into() }
}

#[inline]
#[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
fn simd_max(lhs: f32x4, rhs: f32x4) -> Simd<f32, 4> {
    lhs.simd_max(rhs)
}

#[inline]
#[cfg(all(target_feature = "sse", target_feature = "sse2"))]
fn simd_cmp(angles: f32x4, prefix: f32x4) -> Mask<i32, 4> {
    let cmp = unsafe { _mm_castps_si128(_mm_cmpgt_ps(angles.into(), prefix.into())) };
    unsafe { Mask::from_int_unchecked(cmp.into()) }
}

#[inline]
#[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
fn simd_cmp(angles: f32x4, prefix: f32x4) -> Mask<i32, 4> {
    angles.simd_gt(prefix)
}

impl ViewShed {
    const VECTOR_WIDTH: usize = 4;

    fn new(max_los: usize) -> Self {
        assert_eq!(max_los % 4, 0);
        assert_ne!(max_los, 0);
        let (distances, adjustments) = generate_distances(max_los);

        Self {
            max_los,
            distances,
            adjustments,
            angles: vec![-2000.0; max_los + 1],
        }
    }

    #[inline]
    /// `line_of_sight` calculates the line of sight given a `pov_height` and the
    /// elevations of the points directly in front of the observer
    fn line_of_sight(&mut self, pov_height: i16, line: &[i16]) -> (f32, Vec<bool>) {
        assert_eq!(line.len(), self.max_los, "the line needs to be los long");

        let (chunked_line, _) = line.as_chunks::<{ Self::VECTOR_WIDTH }>();
        let (distances, _) = self.distances.as_chunks::<{ Self::VECTOR_WIDTH }>();
        let (adjustments, _) = self.adjustments.as_chunks::<{ Self::VECTOR_WIDTH }>();

        {
            let (angles, _) = self.angles[1..].as_chunks_mut::<{ Self::VECTOR_WIDTH }>();
            izip!(angles.iter_mut(), chunked_line, distances, adjustments,).for_each(
                |(angle, &elev, &distance, &adjustment)| {
                    let adjusted: Simd<f32, { Self::VECTOR_WIDTH }> =
                        (Simd::from(elev) - Simd::splat(pov_height)).cast();
                    let res = (adjusted / Simd::from(distance)) - Simd::from(adjustment);
                    res.copy_to_slice(angle);
                },
            );
        };

        let (prefix, _) = self.angles[..self.max_los].as_chunks::<{ Self::VECTOR_WIDTH }>();
        let (angles, _) = self.angles[1..].as_chunks::<{ Self::VECTOR_WIDTH }>();

        let masks = prefix
            .iter()
            .map(|&angle| {
                let start = Simd::from_array(angle);
                let mut v_prefix_max = {
                    let shifted = start.shift_elements_right::<1>(-2000.0f32);
                    simd_max(start, shifted)
                };

                v_prefix_max = {
                    let shifted = v_prefix_max.shift_elements_right::<2>(-2000.0f32);
                    simd_max(v_prefix_max, shifted)
                };

                v_prefix_max
            })
            .scan(Simd::splat(-2000.0f32), |acc, prefix| {
                let new_max = simd_max(prefix, *acc);

                let cur_max = Simd::splat(prefix[3]);
                *acc = simd_max(*acc, cur_max);

                Some(new_max)
            })
            .zip(angles)
            .map(|(prefix, &angle)| simd_cmp(Simd::from_array(angle), prefix))
            .collect::<Vec<_>>();

        let heatmap = masks
            .iter()
            .zip(distances)
            .fold(Simd::splat(0.0f32), |acc, (mask, &dists)| {
                acc + (mask.select(Simd::from_array(dists), Simd::splat(0.0f32))
                    * Simd::splat(TAN_ONE_RAD))
            })
            .reduce_sum();

        (
            heatmap,
            masks.iter().flat_map(|mask| mask.to_array()).collect(),
        )
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "so long as max_los < 2^24, the following as conversions are entirely safe"
)]
#[expect(
    clippy::integer_division,
    reason = "i32 is constructed from (i32, i32) converting back should succeed"
)]
/// `dem_to_pov` turns the `dem_id` to the `pov_id` so that the result can be stored in a heatmap
const fn dem_to_pov(dem_id: i32, width: usize, max_los: usize) -> Option<i32> {
    let dem_x = (dem_id % width as i32) - max_los as i32;
    let dem_y = (dem_id / width as i32) - max_los as i32;
    if dem_x >= max_los as i32 || dem_y >= max_los as i32 {
        return None;
    }

    Some(dem_y * (max_los as i32) + dem_x)
}

/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight mis `max_los`
pub fn kernel(elevation_map: &[i16], max_los: usize, angle: f32) -> (Vec<f32>, Vec<Vec<bool>>) {
    let mut heatmap = vec![0.0f32; max_los * max_los];
    let mut sector_data: Vec<Vec<bool>> = vec![vec![]; max_los * max_los];

    let (indexes, rotated_elevations) =
        super::rotation::generate_rotation(elevation_map, angle as f64, max_los);

    assert_eq!(
        rotated_elevations.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    let chocolate_width = 2 * max_los;

    let mut vs = ViewShed::new(max_los);
    for (line_elevations, line_indexes) in izip!(
        rotated_elevations.chunks_exact(chocolate_width),
        indexes.chunks_exact(chocolate_width),
    ) {
        for (pov, (&pov_height, &result_dem_id)) in izip!(
            line_elevations.iter().take(max_los),
            line_indexes.iter().take(max_los),
        )
        .enumerate()
        {
            // TODO@ryan:
            //   I made this `Option` just because I'm overriding the save location of the
            //   ring data by rotating the TVS ID (or rather unrotating the already rotated
            //   TVS ID). And of course some IDs can't be rotated and remain in the TVS grid,
            //   hence the `Option`. I don't know if you really want this to be returning
            //   `Option`. What we actually need is just the indexes of each TVS point you
            //   compute in order, ie from 0 to TVS size. See the next TODO for more
            //   explanation.
            let Some(result_tvs_id) = dem_to_pov(result_dem_id, 3 * max_los, max_los) else {
                continue;
            };

            // if the line of sight is not within our computable points, do not consider it
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "max_los^2 < 2^31"
            )]
            if result_tvs_id < 0i32 || result_tvs_id >= (max_los * max_los) as i32 {
                continue;
            }

            let neighbor = pov + 1;
            let line_of_sight = &line_elevations[neighbor..neighbor + max_los];
            let (pixel, visibility_bitmap) = vs.line_of_sight(pov_height + 2, line_of_sight);

            heatmap[result_tvs_id as usize] = pixel;

            // TODO@ryan:
            //   This (anti-)rotation of the `result_tvs_id` is just a hack to get the ring data
            //   into the right format for rendering. Ideally we would just fill up the ring data
            //   in the order that each point is processed. Though without skipping any points. The
            //   sector data is just a snapshot of the already rotated TVS grid. The reason for this
            //   is mainly fidelity. We don't want to have to both rotate and unrotate data. Just
            //   the rotation already has all the data we need to reconstruct viewsheds.
            //
            //   In short: either keep the hack or better, just fill the sector data as you process
            //   it, but make sure that any skipped points are also filled with empty bitmaps.
            {
                let sector = angle.rem_euclid(crate::compute::SECTOR_STEPS as f32);
                let rotated_tvs_id = kernel::rotation::Rotator::new_from_angle(
                    result_tvs_id as u32,
                    max_los as u32,
                    sector,
                )
                .anti_rotate_dem_id();

                if rotated_tvs_id != kernel::rotation::NOOP_DEM_ID {
                    sector_data[rotated_tvs_id] = visibility_bitmap;
                }
            }
        }
    }

    (heatmap, sector_data)
}

#[expect(clippy::indexing_slicing, reason = "These are just tests")]
#[cfg(test)]
mod test {
    use googletest::prelude::*;

    use super::*;

    fn run(elevations: &[i16]) -> (f32, Vec<bool>) {
        let mut viewshed = ViewShed::new(elevations.len() - 1);
        let pov = elevations[0];
        viewshed.line_of_sight(pov, &elevations[1..])
    }

    #[gtest]
    fn lines_of_sight() {
        expect_eq!(
            run(&[0, 1000, 4000, 9000, 12000, 3000, 30000, 3000, 3000]),
            (
                20.94396,
                vec![true, true, true, false, false, true, false, false]
            )
        );

        expect_eq!(
            run(&[7, 5, 2, 1, 0]),
            (8.72665, vec![true, false, false, true])
        );
    }
}
