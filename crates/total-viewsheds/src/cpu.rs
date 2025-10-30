//! `cpu` is a CPU version of the total viewshed calculation

use itertools::izip;

#[cfg(all(
    target_feature = "avx2",
    target_feature = "avx",
    target_feature = "sse",
    target_feature = "sse2"
))]
use std::arch::x86_64::{
    _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_max_ps,
    _mm256_slli_si256, _mm_castps_si128, _mm_cmpge_ps, _mm_max_ps, _CMP_LE_OS,
};
use std::iter::zip;
use std::simd::prelude::*;
use std::simd::{LaneCount, Mask, StdFloat as _, SupportedLaneCount};
use std::time::Instant;
use std::{array, f32, slice, thread};

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

/// `TAN_ONE_RAD` helps normalize the fact that inner points are sampled more often
/// see the TVS paper for reasoning.
const TAN_ONE_RAD: f32 = 0.017_453_3;

/// `Vectorized` is an empty struct to allow for specializations of the total viewhshed algorithm
/// TODO: maybe we can just use a generic struct?
struct Vectorized;

/// `Viewshed` holds all the platform and vector-width specific methods the CPU kernel
/// needs to operate.
trait Viewshed<const WIDTH: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    /// `gte` takes in a vector of angles and its prefix maximum returns a mask of
    /// i32s which are either -1 or 0 in each lane. This way it can be used to "select"
    /// which lanes of the target vector to use for further calculations
    fn gte(&self, angle: Simd<f32, WIDTH>, prefix: Simd<f32, WIDTH>) -> Mask<i32, WIDTH>;

    /// `max` returns the lane-wise maximum of both vectors. It exists to help platform-specific
    /// and potentially "unsafe" (in floating point terms) and speedier implementations
    fn max(&self, lhs: Simd<f32, WIDTH>, rhs: Simd<f32, WIDTH>) -> Simd<f32, WIDTH>;

    /// `prefix_max` calculates a prefix maximum given all of the `angles` and stores
    /// it in `prefix_max`
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
    fn gte(&self, angle: f32x4, prefix: f32x4) -> Mask<i32, 4> {
        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask = _mm_castps_si128(_mm_cmpge_ps(angle.into(), prefix.into()));
            Mask::<i32, 4>::from_int_unchecked(mask.into())
        }
    }

    #[inline]
    #[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
    fn gte(&self, lhs: f32x4, rhs: f32x4) -> Mask<i32, 4> {
        lhs.simd_ge(rhs)
    }

    #[inline]
    #[cfg(all(target_feature = "sse", target_feature = "sse2"))]
    fn max(&self, lhs: f32x4, rhs: f32x4) -> Simd<f32, 4> {
        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe { _mm_max_ps(lhs.into(), rhs.into()).into() }
    }

    #[inline]
    #[cfg(not(all(target_feature = "sse", target_feature = "sse2")))]
    fn max(&self, lhs: f32x4, rhs: f32x4) -> Simd<f32, 4> {
        lhs.simd_max(r)
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
    fn gte(&self, angle: f32x8, prefix: f32x8) -> Mask<i32, 8> {
        // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask =
                _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OS>(prefix.into(), angle.into()));
            Mask::<i32, 8>::from_int_unchecked(mask.into())
        }
    }

    #[inline]
    fn max(&self, lhs: f32x8, rhs: f32x8) -> Simd<f32, 8> {
        // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe { _mm256_max_ps(lhs.into(), rhs.into()).into() }
    }

    #[inline]
    fn prefix_max(&self, angles: &[f32x8], prefix_max: &mut [f32x8], acc: f32x8) -> f32x8 {
        // Calculate the 4-wide block prefix max two at a time
        for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {

            // safety: all mm256 operations are avx2, and Viewshed<8> has feature guards for both
            let mut v_prefix_max = unsafe {
                let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(angle.into()));
                let blended = _mm256_blend_ps::<0b1000_1000>(
                    _mm256_castsi256_ps(shifted),
                    Simd::splat(-2000.0f32).into(),
                );
                self.max(angle, blended.into())
            };

            // safety: all mm256 operations are avx2, and Viewshed<8> has feature guards for both
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

        // safety: because f32x8s are aligned to exactly sizeof(f32x4) * 2
        // this is well aligned, so the cast is valid
        //
        // This is SUPER MEGA UBER VERY sketchy, and shouldn't be copied
        // unless you _really_, _truly_ understand what the compiler will do
        let single_wide_prefx: &mut [f32x4] = unsafe {
            let ptr = prefix_max.as_mut_ptr();
            slice::from_raw_parts_mut(ptr.cast::<f32x4>(), prefix_max.len() * 2)
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

#[inline]
/// `load_elevations` converts an array of i16 elevations into a height adjusted
fn load_elevations<const N: usize>(elev_arr: [i16; N], pov_height: f32) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let elevs = Simd::<i16, N>::from_array(elev_arr);
    let float_elevs: Simd<f32, N> = elevs.cast();
    float_elevs - Simd::splat(pov_height)
}

#[inline]
/// `line_of_sight` calculates a single line of sight for a given pov, which is passed in via `pov_height`
fn line_of_sight<const N: usize, const UNROLL: usize, VS>(
    vs: &VS,
    elevations: &[[i16; N]],
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
        elevations
            .iter()
            .map(|elev| load_elevations(*elev, pov_height)),
        distances,
        adjustments
    )
    .for_each(|(angle, elev, dist, adjust)| {
        *angle = elev / dist - adjust;
    });

    #[expect(
        clippy::float_cmp,
        reason = "-2000.0f32 is a sentinel value for the first time this accumlative function is run"
    )]
    if prefix_in[0] == -2000.0f32 {
        angle_buf[0][0] = -2000.1f32;
    }

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

        *next_sum = selected_distances.mul_add(selected_tans, *next_sum);
    });

    (sum_buf, prefix_out)
}

/// `total_viewshed` computes a total viewshed heatmap for a given elevation map,
/// and corresponding indexes to store the rotated data
fn total_viewshed<const WIDTH: usize, const UNROLL: usize, V: Viewshed<WIDTH>>(
    vs: &V,
    elevation_map: &[i16],
    indexes: &[i32],
    max_los: usize,
    result: &mut [f32],
) where
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

    // precalculate all distances and their spherical earth "adjustments".
    // This saves ~33% of effort inside our hot loop
    let (distances, adjustments): (Vec<Simd<f32, WIDTH>>, Vec<Simd<f32, WIDTH>>) = (0..max_los)
        .step_by(WIDTH)
        .map(|offset| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "WIDTH < 2^31"
            )]
            let distance_arr: [i32; WIDTH] = array::from_fn(|i| i as i32);
            let distances = Simd::from_array(distance_arr);

            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_wrap,
                clippy::cast_possible_truncation,
                reason = "WIDTH < 2^31"
            )]
            let normalized = (distances + Simd::splat(offset as i32)) * Simd::splat(100i32);

            let floats: Simd<f32, WIDTH> = normalized.cast();

            (floats, floats / Simd::splat(EARTH_RADIUS_SQUARED))
        })
        .unzip();

    let (chunked_distances, rest_distances) = distances.as_chunks::<UNROLL>();
    let (chunked_adjustments, rest_adjustments) = adjustments.as_chunks::<UNROLL>();

    for line_idx in 0..max_los {
        let elevation_offset = line_idx * width;

        #[expect(
            clippy::indexing_slicing,
            reason = "elevation_offset < elevations.len() - width"
        )]
        let line: &[i16] = &elevation_map[elevation_offset..(elevation_offset + width)];

        let indexes_offset = line_idx * max_los;

        // The hottest of the hot loops.
        // Any change inside this loop needs careful benchmarking before committing
        #[expect(
            clippy::indexing_slicing,
            reason = "indexes_offset < elevations.len() - width"
        )]
        for (pov, &result_idx) in indexes[indexes_offset..].iter().enumerate().take(max_los) {
            // if the line of sight is not within our computable points, do not consider it
            if result_idx < 0i32 {
                continue;
            }

            // safety: pov is guaranteed to be in bounds since the slice is max_los in size
            let pov_height = f32::from(unsafe { *line.get_unchecked(pov) });

            // safety: max_los % WIDTH == 0, so [pov..pov+max_los) will also be WIDTH wide
            let (elevations, _): (&[[i16; WIDTH]], _) =
                unsafe { line.get_unchecked(pov..pov + max_los) }.as_chunks::<WIDTH>();

            let (chunked_elevs, rest_elevs) = elevations.as_chunks::<UNROLL>();

            let (local_sums, prefix) = izip!(chunked_elevs, chunked_distances, chunked_adjustments)
                .fold(
                    ([Simd::splat(0.0); UNROLL], Simd::splat(-2000.0)),
                    |(sum, prefix), (elevs, dists, adjusts)| {
                        let (next_sum, acc) = line_of_sight::<WIDTH, UNROLL, V>(
                            // vs: &VS,
                            vs,      // elevs: &[[i16; N]],
                            elevs,   // distances: &[Simd<f32, N>],
                            dists,   // adjustments: &[Simd<f32, N>],
                            adjusts, // prefix_in: Simd<f32, N>,
                            prefix,  // pov_height: f32,
                            pov_height,
                        );

                        let mut copied = sum;
                        zip(copied.iter_mut(), next_sum).for_each(|(old, new)| {
                            *old += new;
                        });

                        (copied, acc)
                    },
                );

            let mut sum = local_sums
                .iter()
                .fold(0.0f32, |acc, partial| acc + partial.reduce_sum());

            let (sum_buf, _) = line_of_sight::<WIDTH, UNROLL, V>(
                vs,
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

/// `kernel` is a CPU-based total viewshed kernel. It makes use of image rotation tof
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

    total_viewshed::<8, 8, Vectorized>(
        &vectorized,
        &rotated_elevations,
        &indexes,
        max_los_points,
        result,
    );
    tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
}

/// `multithreaded_kernel` parallelizes CPU kernel calculations for a `core_count` and calculates
/// `num_angles` different angles
pub fn multithreaded_kernel(
    elevations_original: &[i16],
    max_los_points_original: usize,
    num_angles: usize,
    core_count: usize,
) -> Vec<f32> {
    let max_los_points = max_los_points_original.div_ceil(4) * 4;
    let dem_width = max_los_points * 3;
    let mut elevations_vec = elevations_original.to_vec();
    elevations_vec.resize(dem_width.pow(2), 0);
    let elevations = &elevations_vec;

    if max_los_points != max_los_points_original {
        tracing::warn!("LoS: {max_los_points_original} to {max_los_points}");
    }
    if elevations.len() != elevations_original.len() {
        tracing::warn!(
            "Elevations array length resized: {} to {}",
            elevations_original.len(),
            elevations_vec.len()
        );
    }

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
