//! `cpu` is a CPU version of the total viewshed calculation

use itertools::{izip, Itertools};

#[cfg(all(target_feature = "avx2", target_feature = "avx", target_feature = "sse", target_feature = "sse2"))]
use std::arch::x86_64::{
    _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_max_ps,
    _mm256_slli_si256, _mm_castps_si128, _mm_cmpge_ps, _mm_max_ps, _CMP_LE_OS,
};
use std::iter::zip;
use std::mem::transmute;
use std::simd::prelude::*;
use std::simd::{LaneCount, Mask, SupportedLaneCount};
use std::time::Instant;
use std::{array, f32, slice, thread};

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

/// `TAN_ONE_RAD` helps normalize the fact that inner points are sampled more often
/// see the TVS paper for reasoning.
const TAN_ONE_RAD: f32 = 0.017_453_3;

struct Vectorized;

trait Viewshed<const WIDTH: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    fn gte(&self, l: Simd<f32, WIDTH>, r: Simd<f32, WIDTH>) -> Mask<i32, WIDTH>;
    fn max(&self, l: Simd<f32, WIDTH>, r: Simd<f32, WIDTH>) -> Simd<f32, WIDTH>;

    fn prefix_max(
        &self,
        angles: &[Simd<f32, WIDTH>],
        prefix_max: &mut [Simd<f32, WIDTH>],
        acc: Simd<f32, WIDTH>,
    ) -> Simd<f32, WIDTH>;
}

impl Viewshed<4> for Vectorized {
    #[inline]
    #[cfg(all(target_feature = "sse", target_feature = "sse2"))]
    fn gte(&self, l: f32x4, r: f32x4) -> Mask<i32, 4> {
        unsafe {
            let mask = _mm_castps_si128(_mm_cmpge_ps(l.into(), r.into()));
            Mask::<i32, 4>::from_int_unchecked(mask.into())
        }
    }

    #[inline]
    #[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
    fn gte(&self, l: f32x4, r: f32x4) -> Mask<i32, 4> {
        l.simd_ge(r)
    }

    #[inline]
    #[cfg(all(target_feature = "sse", target_feature = "sse2"))]
    fn max(&self, l: f32x4, r: f32x4) -> Simd<f32, 4> {
        unsafe { _mm_max_ps(l.into(), r.into()).into() }
    }

    #[inline]
    #[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
    fn max(&self, l: f32x4, r: f32x4) -> Simd<f32, 4> {
        l.simd_max(r)
    }

    #[inline]
    fn prefix_max(&self, angles: &[f32x4], prefix_max: &mut [f32x4], acc: f32x4) -> f32x4 {
        for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
            let mut v_prefix_max = {
                let shifted = angle.shift_elements_right::<1>(-2000.0f32);
                self.max(angle, shifted)
            };

            v_prefix_max = {
                let shifted = v_prefix_max.shift_elements_right::<2>(-2000.0f32);
                self.max(v_prefix_max, shifted)
            };


            *prefix = v_prefix_max;
        }

        let mut local_acc = acc;

        // accumulate the prefix maxes for blocks, re-computing all prefix maxes
        // to include the accumulated value
        for prefix in prefix_max {
            let cur_prefix: f32x4 = *prefix;
            let cur_max: f32x4 = Simd::splat(cur_prefix[3]);

            *prefix = self.max(local_acc, cur_prefix);
            local_acc = self.max(local_acc, cur_max);
        }

        local_acc
    }
}

#[cfg(all(target_feature = "avx2", target_feature = "avx"))]
impl Viewshed<8> for Vectorized {
    #[inline]
    fn gte(&self, l: f32x8, r: f32x8) -> Mask<i32, 8> {
        unsafe {
            let mask = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OS>(r.into(), l.into()));
            Mask::<i32, 8>::from_int_unchecked(mask.into())
        }
    }

    #[inline]
    fn max(&self, l: f32x8, r: f32x8) -> Simd<f32, 8> {
        unsafe { _mm256_max_ps(l.into(), r.into()).into() }
    }

    #[inline]
    fn prefix_max(&self, angles: &[f32x8], prefix_max: &mut [f32x8], acc: f32x8) -> f32x8 {
        // Calculate the 4-wide block prefix max two at a time
        for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
            let mut v_prefix_max = unsafe {
                let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(angle.into()));
                let blended = _mm256_blend_ps::<0b1000_1000>(
                    _mm256_castsi256_ps(shifted),
                    Simd::splat(-2000.0f32).into(),
                );
                self.max(angle, blended.into())
            };

            v_prefix_max = unsafe {
                let shifted = _mm256_slli_si256::<8>(_mm256_castps_si256(v_prefix_max.into()));
                let blended = _mm256_blend_ps::<0b1100_1100>(
                    _mm256_castsi256_ps(shifted),
                    Simd::splat(-2000.0f32).into(),
                );

                self.max(v_prefix_max, blended.into())
            };

            *prefix = v_prefix_max;
        }

        let mut local_acc = f32x4::splat(acc[0]);
        let single_wide_prefx: &mut [f32x4] = unsafe {
            slice::from_raw_parts_mut(
                transmute::<&mut [f32x8], &mut [f32x4]>(prefix_max.as_mut()).as_mut_ptr(),
                prefix_max.len() * 2,
            )
        };

        // accumulate the prefix maxes for blocks, re-computing all prefix maxes
        // to include the accumulated value
        for prefix in single_wide_prefx {
            let cur_prefix: f32x4 = *prefix;
            let cur_max: f32x4 = Simd::splat(cur_prefix[3]);

            *prefix = self.max(local_acc, cur_prefix);
            local_acc = self.max(local_acc, cur_max);
        }

        f32x8::splat(local_acc[0])
    }
}

fn load_elevations<const N: usize>(elev_arr: [i16; N], pov_height: f32) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let elevs = Simd::<i16, N>::from_array(elev_arr);
    let float_elevs: Simd<f32, N> = elevs.cast();
    float_elevs - Simd::splat(pov_height)
}


#[inline]
fn line_of_sight<const N: usize, const UNROLL: usize, VS>(
    vs: &VS,
    elevs: &[[i16; N]],
    distances: &[Simd<f32, N>],
    adjustments: &[Simd<f32, N>],
    prefix_in: Simd<f32, N>,
    pov_height: f32,
) -> ([Simd<f32, N>; UNROLL], Simd<f32, N>)
where
    LaneCount<N>: SupportedLaneCount,
    VS: Viewshed<N>,
{
    let mut sum_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];
    let mut angle_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];
    let mut prefix_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];

    izip!(
        angle_buf.iter_mut(),
        elevs.iter().map(|e| load_elevations(*e, pov_height)),
        distances,
        adjustments
    )
    .for_each(|(angle, elev, dist, adjust)| {
        *angle = elev / dist - adjust;
    });
    angle_buf[0][0] = -2000.1f32;

    let prefix_out = vs.prefix_max(&angle_buf, &mut prefix_buf, prefix_in);

    izip!(
        &mut sum_buf,
        angle_buf.iter(),
        prefix_buf.iter(),
        distances.iter()
    )
    .for_each(|(next_sum, &angle, &pref, &dists)| {
        let mask = vs.gte(angle, pref);

        let selected_distances = mask.select(dists, Simd::splat(0.0));
        let selected_tans = mask.select(Simd::splat(TAN_ONE_RAD), Simd::splat(0.0));

        *next_sum += selected_distances * selected_tans;
    });

    (sum_buf, prefix_out)
}

fn total_viewshed<const WIDTH: usize, const UNROLL: usize, V: Viewshed<WIDTH>>(
    vs: V,
    elevation_map: &[i16],
    indexes: &[i32],
    max_los: usize,
) -> Vec<f32>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    assert_eq!(
        elevation_map.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    assert_eq!(
        max_los % WIDTH,
        0,
        "to help the vectorizer, max_los must be a multiple of {WIDTH}"
    );

    let width = 2 * max_los;

    let mut result = vec![0.0f32; max_los * max_los];

    // precalculate all distances and their spherical earth "adjustments".
    // This saves ~33% of effort inside our hot loop
    let (distances, adjustments): (Vec<Simd<f32, WIDTH>>, Vec<Simd<f32, WIDTH>>) = (0..max_los)
        .step_by(WIDTH)
        .map(|offset| {
            let distance_arr: [i32; WIDTH] = array::from_fn(|i| i as i32);
            let distances = Simd::from_array(distance_arr);

            // x * 100
            let normalized = (distances + Simd::splat(offset as i32)) * Simd::splat(100);

            let floats: Simd<f32, WIDTH> = normalized.cast();

            (floats, floats / Simd::splat(EARTH_RADIUS_SQUARED))
        })
        .unzip();

    let (chunked_distances, rest_distances) = (&distances).as_chunks::<UNROLL>();
    let (chunked_adjustments, rest_adjustments) = (&adjustments).as_chunks::<UNROLL>();

    for line_idx in 0..max_los {
        let elevation_offset = line_idx * width;

        let line: &[i16] = &elevation_map[elevation_offset..(elevation_offset + width)];

        let indexes_offset = line_idx * max_los;

        let line_indexes: &[i32] = &indexes[indexes_offset..(indexes_offset + max_los)];

        // The hottest of the hot loops.
        // Any change inside this loop needs careful benchmarking before committing
        for pov in 0..max_los {
            let result_idx = line_indexes[pov];

            // if the line of sight is not within our computable points, do not consider it
            if result_idx < 0i32 {
                continue;
            }

            // safety: pov is guaranteed to be in bounds since the slice is max_los in size
            let pov_height = f32::from(unsafe { *line.get_unchecked(pov) });

            // convert the max_los-1 elevations ahead of the POV into floats, and adjust
            // for the observer's height
            let elevations: &[[i16; WIDTH]] = unsafe {
                line.get_unchecked(pov..pov + max_los)
                    .as_chunks_unchecked::<WIDTH>()
            };

            let (chunked_elevs, rest_elevs) = elevations.as_chunks::<UNROLL>();

            let (local_sums, prefix) = izip!(chunked_elevs, chunked_distances, chunked_adjustments)
                .fold(
                    ([Simd::splat(0.0); UNROLL], Simd::splat(-2000.0)),
                    |(sum, prefix), (elevs, dists, adjusts)| {
                        let (next_sum, acc) = line_of_sight::<WIDTH, UNROLL, V>(
                            // vs: &VS,
                            &vs,
                            // elevs: &[[i16; N]],
                            elevs,
                            // distances: &[Simd<f32, N>],
                            dists,
                            // adjustments: &[Simd<f32, N>],
                            adjusts,
                            // prefix_in: Simd<f32, N>,
                            prefix,
                            // pov_height: f32,
                            pov_height,
                        );


                        let mut copied = sum;
                        zip(copied.iter_mut(), next_sum)
                            .for_each(|(a, b)| {
                                *a = *a + b;
                            });

                        (copied, acc)
                    },
                );

            let mut sum = local_sums
                .iter()
                .fold(0.0f32, |acc, partial| acc + partial.reduce_sum());

            let (sum_buf, _) = line_of_sight::<WIDTH, UNROLL, V>(
                &vs,
                rest_elevs,
                rest_distances,
                rest_adjustments,
                prefix,
                pov_height,
            );

            sum += sum_buf
                .iter()
                .fold(0.0f32, |acc, partial| acc + partial.reduce_sum());

            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                reason = "result_idx should be in [0, 2^31]"
            )]
            // safety: it is guaranteed by the rotation kernel that if the index is
            // greater than zero that it is in-bounds. This saves ~10% of bounds checks
            unsafe {
                *result.get_unchecked_mut(result_idx as usize) += sum;
            }
        }
    }

    result
}

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
        assert_eq!(elevs.len() as isize % width, 0, "elevs should be square");
        assert_eq!(
            elevs.len() as isize / width,
            width,
            "elevs should be square"
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

    let idxs = (0..max_los)
        .flat_map(|idx| {
            let start = idx * (2 * max_los);
            let end = start + max_los;
            #[expect(clippy::indexing_slicing, reason = "start < rotation-(2*max_los)")]
            &rotation[start..end]
        })
        .map(|&val| {
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
            let x = (val / width as i32) - max_los as i32;
            let y = (val % width as i32) - max_los as i32;
            if (0i32..max_los as i32).contains(&x) && (0i32..max_los as i32).contains(&y) {
                x * (max_los as i32) + y
            } else {
                -1i32
            }
        })
        .collect();

    (idxs, elevations)
}

/// `kernel` is a CPU-based total viewshed kernel. It makes use of image rotation to
/// optimize the cache locality of all lookups for a total viewshed calculation
fn kernel(elevations: &[i16], max_los_points: usize, angle: usize, result: &mut [f32]) {
    assert!(angle < 360, "angle must be [0, 360)");
    let mut start = Instant::now();

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "angle is [0,360), not more than 2^54"
    )]
    let (indexes, rotated_elevations) = generate_rotation(elevations, angle as f64, max_los_points);

    tracing::info!(
        "rotated {:?} in {:?}, calculating kernel",
        angle,
        start.elapsed()
    );

    start = Instant::now();

    let vectorized = Vectorized {};

    let local_result = total_viewshed::<8, 8, Vectorized>(
        vectorized,
        &rotated_elevations,
        &indexes,
        max_los_points,
    );
    for (total, r) in zip(result, local_result) {
        *total += r
    }
    tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
}

/// `multithreaded_kernel` parallelizes CPU kernel calculations for a `core_count` and calculates
/// `num_angles` different angles
pub fn multithreaded_kernel(
    elevations: &[i16],
    max_los_points: usize,
    num_angles: usize,
    core_count: usize,
) -> Vec<f32> {
    thread::scope(|scope| {
        let threads = (0..core_count)
            .map(|start_angle: usize| {
                scope.spawn(move || {
                    let mut res = vec![0.0f32; max_los_points * max_los_points];
                    for angle in (start_angle..num_angles).step_by(core_count) {
                        kernel(elevations, max_los_points, angle, &mut res);
                    }
                    res
                })
            })
            .collect::<Vec<_>>();

        let mut res = vec![0.0f32; max_los_points * max_los_points];

        #[expect(
            clippy::unwrap_used,
            reason = "if the thread doesn't join, the program should terminate"
        )]
        for thread in threads {
            zip(&mut res, thread.join().unwrap()).for_each(|(acc, heatmap)| *acc += heatmap);
        }
        res
    })
}
