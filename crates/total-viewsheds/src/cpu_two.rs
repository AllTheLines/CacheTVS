use itertools::izip;
use std::arch::x86_64::{_mm256_cmp_ps, _mm_castps_si128, _mm_cmpgt_ps, _mm_cmple_ps, _mm_cmplt_ps, _mm_max_ps};
use std::cmp::max;
use std::iter::zip;
use std::simd::prelude::{SimdFloat, SimdInt};
use std::simd::{f32x4, Mask, Simd};

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

const TAN_ONE_RAD: f32 = 0.017_453_3;

/// `generate_rotation` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "so long as max_los^2 < 2^24, the following `as` conversions are entirely safe"
)]
fn generate_rotation(elevs: &[i16], angle: f64, max_los: usize) -> (Vec<i32>, Vec<i16>) {
    let width = (max_los * 3) as isize;

    #[expect(clippy::integer_division, reason = "we don't need precision here")]
    {
        assert_eq!(
            elevs.len() as isize % width,
            0,
            "Elevations array must be square {}%{width} != 0",
            elevs.len(),
        );
        let elevations_div_width = elevs.len() as isize / width;
        assert_eq!(
            elevations_div_width,
            width,
            "Elevations array must be square {}/{width} (={elevations_div_width}) != {width}",
            elevs.len() as isize
        );
    };

    let (sin, cos) = (f64::sin(angle.to_radians()), f64::cos(angle.to_radians()));

    #[expect(clippy::integer_division, reason = "we don't need precision here")]
    let (x_center, y_center) = (width / 2, width / 2);

    let mut rotation: Vec<i32> = Vec::with_capacity(2 * max_los * max_los);

    for x in (max_los as isize)..(max_los as isize) * 2 {
        let x_sin = (x - x_center) as f64 * sin;
        let x_cos = (x - x_center) as f64 * cos;
        for y in (max_los as isize)..width {
            let y_sin = (y - y_center) as f64 * sin;
            let y_cos = (y - y_center) as f64 * cos;

            let x_rot = (x_cos - y_sin).round() as isize + y_center;
            let y_rot = (y_cos + x_sin).round() as isize + x_center;

            let new_idx = x_rot.clamp(0, width - 1) * width + y_rot.clamp(0, width - 1);

            rotation.push(new_idx as i32);
        }
    }

    debug_assert_eq!(
        rotation.len() as isize,
        max_los as isize * (2 * max_los as isize),
        "the rotation should be 2 * max_los wide, max_los tall"
    );

    // map the indexes to their elevations
    let elevations = rotation
        .iter()
        .map(|&idx| {
            if idx < 0i32 {
                i16::MIN
            } else {
                #[expect(
                    clippy::as_conversions,
                    reason = "elevations start out as i16s, and i16 -> f32 -> i16 is lossless"
                )]
                #[expect(clippy::cast_sign_loss, reason = "idx < 2^31, idx >= 0")]
                // safety: idx is clamped so a get will always be in-bounds
                *unsafe { elevs.get_unchecked(idx as usize) }
            }
        })
        .collect::<Vec<i16>>();

    (rotation, elevations)
}

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

struct ViewShed {
    max_los: usize,
    angles: Vec<f32>,
    distances: Vec<f32>,
    adjustments: Vec<f32>,
}

#[inline]
#[cfg(all(target_feature = "sse", target_feature = "sse2"))]
fn simd_max(lhs: f32x4, rhs: f32x4) -> Simd<f32, 4> {
    // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
    // thus allowing this to be non IEEE754 compliant
    unsafe { _mm_max_ps(lhs.into(), rhs.into()).into() }
}

fn simd_cmp(angles: f32x4, prefix: f32x4) -> Mask<i32, 4> {
    let cmp = unsafe { _mm_castps_si128(_mm_cmpgt_ps(angles.into(), prefix.into())) };
    unsafe { Mask::from_int_unchecked(cmp.into()) }
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
        assert_eq!(line.len(), self.max_los);

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
            .map(|&prefix| {
                let start = Simd::from_array(prefix);
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
                acc + (mask.select(Simd::from_array(dists), Simd::splat(0.0f32)) * Simd::splat(TAN_ONE_RAD))
            })
            .reduce_sum();

        // let most_prefix = &self.prefix_max[..self.max_los];
        //
        // let (prefix_max, _) = most_prefix.as_chunks::<{ Self::VECTOR_WIDTH }>();
        // let (angles, _) = self.angles.as_chunks::<{ Self::VECTOR_WIDTH }>();
        //
        // let res: Simd<f32, { Self::VECTOR_WIDTH }> = izip!(angles.iter(), prefix_max, distances)
        //     .map(|(&angles, &prefix, &dists)| {
        //         let cmp = unsafe {
        //             _mm_castps_si128(_mm_cmpgt_ps(
        //                 Simd::from_array(angles).into(),
        //                 Simd::from_array(prefix).into(),
        //             ))
        //         };
        //         let mask: Mask<i32, { Self::VECTOR_WIDTH }> =
        //             unsafe { Mask::from_int_unchecked(cmp.into()) };
        //         mask.select(Simd::from_array(dists), Simd::splat(0.0f32)) * Simd::splat(TAN_ONE_RAD)
        //     })
        //     .fold(Simd::splat(0.0f32), |acc, dists| acc + dists);

        (heatmap, masks.iter().flat_map(|mask| mask.to_array()).collect())
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
const fn dem_to_pov(dem_id: i32, width: usize, max_los: usize) -> i32 {
    let dem_x = (dem_id / width as i32) - max_los as i32;
    let dem_y = (dem_id % width as i32) - max_los as i32;

    dem_x * (max_los as i32) + dem_y
}

/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight mis `max_los`
pub fn kernel(elevation_map: &[i16], max_los: usize, angle: f32) -> (Vec<f32>, Vec<Vec<bool>>) {
    let mut heatmap = vec![0.0f32; max_los * max_los];
    let mut sector_data: Vec<Vec<bool>> = vec![];

    let (indexes, rotated_elevations) = generate_rotation(elevation_map, angle as f64, max_los);

    assert_eq!(
        rotated_elevations.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    let width = 2 * max_los;

    let mut vs = ViewShed::new(max_los);
    for (line, line_indexes) in izip!(
        rotated_elevations.chunks_exact(width),
        indexes.chunks_exact(width),
    ) {
        for (pov, (&pov_height, &result_dem_id)) in
            izip!(line.iter().take(max_los), line_indexes.iter().take(max_los),).enumerate()
        {
            let result_tvs_id = dem_to_pov(result_dem_id, 3 * max_los, max_los);

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

            let (pixel, sector) = vs.line_of_sight(pov_height, &line[pov..pov + max_los]);


            heatmap[result_tvs_id as usize] = pixel;
            sector_data.push(sector)
        }
    }

    (heatmap, sector_data)
}

#[cfg(test)]
mod test {
    use crate::cpu_two::ViewShed;

    #[test]
    fn line_of_sight() {
        let mut vs = ViewShed::new(8);
        let visibility = vs.line_of_sight(0, &[1000, 4000, 9000, 12000, 3000, 30000, 3000, 3000]);
        println!("{:?}", visibility);
    }
}
