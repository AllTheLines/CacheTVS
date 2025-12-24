//! `cpu` is a CPU version of the total viewshed calculation

use itertools::izip;
use rayon::iter::ParallelIterator as _;

#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::{
    __m512, _mm256_alignr_epi32, _mm256_mask_alignr_epi32, _mm512_alignr_epi32,
    _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmple_ps_mask, _mm512_max_ps,
};

use rayon::prelude::IntoParallelIterator as _;
use rayon::ThreadPoolBuilder;
#[cfg(all(
    target_feature = "avx2",
    target_feature = "avx",
    target_feature = "sse",
    target_feature = "sse2"
))]
use std::arch::x86_64::{
    _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_max_ps,
    _mm256_slli_si256, _CMP_GT_OS, _CMP_LE_OS,
};
use std::iter::zip;
use std::simd::prelude::*;
use std::simd::{LaneCount, Mask, SupportedLaneCount};
use std::sync::Mutex;
use std::time::Instant;
use std::{array, f32, mem, slice};

#[cfg(all(target_feature = "sse", target_feature = "sse2"))]
use std::arch::x86_64::{_mm_castps_si128, _mm_cmpge_ps, _mm_max_ps, _mm_cmpgt_ps};

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
        acc: f32,
    ) -> f32;
}

impl Viewshed<4> for Vectorized {
    #[inline]
    #[cfg(all(target_feature = "sse", target_feature = "sse2"))]
    fn gte(&self, angle: f32x4, prefix: f32x4) -> Mask<i32, 4> {
        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask = _mm_castps_si128(_mm_cmpgt_ps(angle.into(), prefix.into()));
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
        lhs.simd_max(rhs)
    }

    #[inline]
    fn prefix_max(&self, angles: &[f32x4], prefix_max: &mut [f32x4], acc: f32) -> f32 {
        zip(prefix_max.iter_mut(), angles.iter())
            .fold(acc, |next: f32, (prefix, &angle)| {
                let start = angle.shift_elements_right::<1>(next);

                let mut v_prefix_max = {
                    let shifted = start.shift_elements_right::<1>(-2000.0f32);
                    self.max(start, shifted)
                };

                v_prefix_max = {
                    let shifted = v_prefix_max.shift_elements_right::<2>(-2000.0f32);
                    self.max(v_prefix_max, shifted)
                };

                *prefix = v_prefix_max;
                start[3].max(v_prefix_max[3])
            })
        // for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
        //     let start = angle.shift_elements_right::<1>(-2000.0f32);
        //
        //     let mut v_prefix_max = {
        //         let shifted = start.shift_elements_right::<1>(-2000.0f32);
        //         self.max(start, shifted)
        //     };
        //
        //     v_prefix_max = {
        //         let shifted = v_prefix_max.shift_elements_right::<2>(-2000.0f32);
        //         self.max(v_prefix_max, shifted)
        //     };
        //
        //     *prefix = v_prefix_max;
        // }
        //
        // let mut local_acc = acc;
        //
        // // accumulate the prefix maxes for blocks, re-computing all prefix maxes
        // // to include the accumulated value
        // for prefix in prefix_max {
        //     let cur_prefix: f32x4 = *prefix;
        //     let cur_max: f32x4 = Simd::splat(cur_prefix[3]);
        //
        //     *prefix = self.max(local_acc, cur_prefix);
        //     local_acc = self.max(local_acc, cur_max);
        // }
        //
        // local_acc
    }
}
//
// #[cfg(all(target_feature = "avx2", target_feature = "avx"))]
// impl Viewshed<8> for Vectorized {
//     #[inline]
//     fn gte(&self, angle: f32x8, prefix: f32x8) -> Mask<i32, 8> {
//         // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
//         // thus allowing this to be non IEEE754 compliant
//         unsafe {
//             let mask =
//                 _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OS>(angle.into(), prefix.into()));
//             Mask::<i32, 8>::from_int_unchecked(mask.into())
//         }
//     }
//
//     #[inline]
//     fn max(&self, lhs: f32x8, rhs: f32x8) -> Simd<f32, 8> {
//         // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
//         // thus allowing this to be non IEEE754 compliant
//         unsafe { _mm256_max_ps(lhs.into(), rhs.into()).into() }
//     }
//
//     #[inline]
//     fn prefix_max(&self, angles: &[f32x8], prefix_max: &mut [f32x8], acc: f32x8) -> f32x8 {
//         // Calculate the 4-wide block prefix max two at a time
//         for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
//             // safety: all mm256 operations are avx2, and Viewshed<8> has feature guards for both
//             let mut v_prefix_max = unsafe {
//                 let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(angle.into()));
//                 let blended = _mm256_blend_ps::<0b1000_1000>(
//                     _mm256_castsi256_ps(shifted),
//                     Simd::splat(-2000.0f32).into(),
//                 );
//                 self.max(angle, blended.into())
//             };
//
//             // safety: all mm256 operations are avx2, and Viewshed<8> has feature guards for both
//             v_prefix_max = unsafe {
//                 let shifted = _mm256_slli_si256::<8>(_mm256_castps_si256(v_prefix_max.into()));
//                 let blended = _mm256_blend_ps::<0b1100_1100>(
//                     _mm256_castsi256_ps(shifted),
//                     Simd::splat(-2000.0f32).into(),
//                 );
//
//                 self.max(v_prefix_max, blended.into())
//             };
//
//             *prefix = v_prefix_max;
//         }
//
//         let mut local_acc = f32x4::splat(acc[3]);
//
//         // safety: because f32x8s are aligned to exactly sizeof(f32x4) * 2
//         // this is well aligned, so the cast is valid
//         //
//         // This is SUPER MEGA UBER VERY sketchy, and shouldn't be copied
//         // unless you _really_, _truly_ understand what the compiler will do
//         let single_wide_prefx: &mut [f32x4] = unsafe {
//             let ptr = prefix_max.as_mut_ptr();
//             slice::from_raw_parts_mut(ptr.cast::<f32x4>(), prefix_max.len() * 2)
//         };
//
//         // accumulate the prefix maxes for blocks, re-computing all prefix maxes
//         // to include the accumulated value
//         for prefix in single_wide_prefx {
//             let cur_prefix: f32x4 = *prefix;
//             let cur_max: f32x4 = Simd::splat(cur_prefix[3]);
//
//             *prefix = self.max(local_acc, cur_prefix);
//             local_acc = self.max(local_acc, cur_max);
//         }
//
//         f32x8::splat(local_acc[3])
//     }
// }

// #[cfg(target_feature = "avx512f")]
// fn _mm512_slli_si512<const K: usize>(elem: __m512) -> __m512
// where
//     [(); { (16 - K) as i32 } as usize]:,
// {
//     unsafe {
//         let zero = f32x16::splat(-2000.0f32);
//         _mm512_castsi512_ps(_mm512_alignr_epi32::<{ (16 - K) as i32 }>(
//             _mm512_castps_si512(elem),
//             _mm512_castps_si512(zero.into()),
//         ))
//     }
// }
//
// #[cfg(target_feature = "avx512f")]
// impl Viewshed<16> for Vectorized {
//     #[inline]
//     fn gte(&self, angle: f32x16, prefix: f32x16) -> Mask<i32, 16> {
//         // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
//         // thus allowing this to be non IEEE754 compliant
//         unsafe {
//             let mask = _mm512_cmple_ps_mask(prefix.into(), angle.into());
//             Mask::<i32, 16>::from_bitmask(mask.into())
//         }
//     }
//
//     #[inline]
//     fn max(&self, lhs: f32x16, rhs: f32x16) -> f32x16 {
//         // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
//         // thus allowing this to be non IEEE754 compliant
//         unsafe { _mm512_max_ps(lhs.into(), rhs.into()).into() }
//     }
//
//     #[inline]
//     fn prefix_max(&self, angles: &[f32x16], prefix_max: &mut [f32x16], acc: f32x16) -> f32x16 {
//         // Calculate the 4-wide block prefix max two at a time
//         for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
//             unsafe {
//                 let mut v_prefix_max =
//                     _mm512_max_ps(angle.into(), _mm512_slli_si512::<1>(angle.into()).into());
//                 v_prefix_max = _mm512_max_ps(
//                     v_prefix_max.into(),
//                     _mm512_slli_si512::<2>(v_prefix_max).into(),
//                 );
//                 v_prefix_max = _mm512_max_ps(
//                     v_prefix_max.into(),
//                     _mm512_slli_si512::<4>(v_prefix_max).into(),
//                 );
//                 v_prefix_max = _mm512_max_ps(
//                     v_prefix_max.into(),
//                     _mm512_slli_si512::<8>(v_prefix_max).into(),
//                 );
//                 *prefix = v_prefix_max.into();
//             }
//         }
//
//         let mut local_acc = f32x16::splat(acc[0]);
//
//         // accumulate the prefix maxes for blocks, re-computing all prefix maxes
//         // to include the accumulated value
//         for prefix in prefix_max {
//             let cur_prefix: f32x16 = *prefix;
//             let cur_max: f32x16 = Simd::splat(cur_prefix[15]);
//
//             *prefix = self.max(local_acc, cur_prefix);
//             local_acc = self.max(local_acc, cur_max);
//         }
//
//         f32x16::splat(local_acc[0])
//     }
// }

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

/// `IndexSIMD` holds the `dem_ids`/"indexes" of the current line of sight,
/// along with where they will be written out to
struct IndexSIMD<'idx, const N: usize> {
    /// `indexes_in` holds a slice of a SIMD-size-wide array of `dem_ids`/"indexes"
    indexes_in: &'idx [[i32; N]],
    /// `indexes_out` is the buffer of a SIMD-size-wide where the visible `dem_ids`/"indexes" will be written out to
    indexes_out: &'idx mut [[i32; N]],
}

#[inline]
/// `line_of_sight` calculates a single line of sight for a given pov, which is passed in via `pov_height`
fn line_of_sight<const N: usize, const UNROLL: usize, VS>(
    vs: &VS,
    elevations: &[[i16; N]],
    distances: &[Simd<f32, N>],
    adjustments: &[Simd<f32, N>],
    prefix_in: f32,
    indexes: Option<IndexSIMD<N>>,
    pov_height: f32,
) -> ([Simd<f32, N>; UNROLL], [Simd<f32, N>; UNROLL], f32)
where
    LaneCount<N>: SupportedLaneCount,
    VS: Viewshed<N>,
{
    let mut sum_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];
    let mut angle_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];
    let mut longest_line_buf: [Simd<f32, N>; UNROLL] = [Simd::splat(0.0); UNROLL];
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
    if prefix_in == -2000.0f32 {
        angle_buf[0][0] = -2000.1f32;
    }

    let prefix_out = vs.prefix_max(&angle_buf, &mut prefix_buf, prefix_in);

    let index_iter = indexes.map(|index_simd| {
        index_simd
            .indexes_in
            .iter()
            .map(|ind| Simd::<i32, N>::from_array(*ind))
            .zip(index_simd.indexes_out.iter_mut())
    });

    izip!(
        &mut sum_buf,
        &mut longest_line_buf,
        angle_buf.iter(),
        prefix_buf.iter(),
        distances.iter(),
        OptionIter::new(index_iter)
    )
        .for_each(|(next_sum, longest_line, &angle, &pref, &dists, inds)| {
            let mask = vs.gte(angle, pref);

            if let Some((inds_in, inds_out)) = inds {
                inds_in.store_select(inds_out, mask);
            }

            let selected_distances = mask.select(dists, Simd::splat(0.0));
            *longest_line = vs.max(*longest_line, selected_distances);

            let selected_tans = mask.select(Simd::splat(TAN_ONE_RAD), Simd::splat(0.0));
            *next_sum = selected_distances * selected_tans;
        });

    (sum_buf, longest_line_buf, prefix_out)
}

/// `dem_to_pov`
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
const fn dem_to_pov(dem_id: i32, width: usize, max_los: usize) -> i32 {
    let dem_x = (dem_id / width as i32) - max_los as i32;
    let dem_y = (dem_id % width as i32) - max_los as i32;

    dem_x * (max_los as i32) + dem_y
}

/// `Indexes` holds the `dem_id`/"indexes" and the output buffer to store them in
struct Indexes<'index> {
    /// `indexes_in` holds the full line of (2 or 3*`max_los`) `dem_ids`/"indexes"
    indexes_in: &'index [i32],
    /// `indexes_out` holds `max_los` indexes of `max_los` length. Zeroes are used if
    indexes_out: &'index mut [i32],
}

/// `OptionIter` holds the state for an optional inner iterator.
/// If passed `None`, it will repeat `None` forever. This comes in handy
/// when working with `izip!`
struct OptionIter<T, Iter>
where
    Iter: Iterator<Item=T>,
{
    /// `iter` holds an optional iterator state which will
    /// call `next()`
    iter: Option<Iter>,
}

impl<T, Iter> OptionIter<T, Iter>
where
    Iter: Iterator<Item=T>,
{
    /// `new` creates a new iter from an Option of the Iter
    const fn new(iter: Option<Iter>) -> Self {
        Self { iter }
    }
}

impl<T, Iter> Iterator for OptionIter<T, Iter>
where
    Iter: Iterator<Item=T>,
{
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut iter) = &mut self.iter {
            iter.next().map(Some)
        } else {
            Some(None)
        }
    }
}

/// `UnrolledAngles` holds the curved earth adjustments
struct UnrolledAngles<'angle, const WIDTH: usize, const UNROLL: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    /// `adjustments` holds a slice of UNROLL sized slices used during loop unrolling, and then the "rest" portion
    adjustments: (
        &'angle [[Simd<f32, WIDTH>; UNROLL]],
        &'angle [Simd<f32, WIDTH>],
    ),
    /// `distances` holds a slice of UNROLL sized slices used during loop unrolling, and then the "rest" portion
    distances: (
        &'angle [[Simd<f32, WIDTH>; UNROLL]],
        &'angle [Simd<f32, WIDTH>],
    ),
}

/// `viewshed` computes the viewshed for a single pov, using its `elevation`, and `max_los`
/// and stores the results in `heatmap` and `longest_line` using the `dem_id`
#[inline]
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "it is what it is for now"
)]
fn viewshed<const WIDTH: usize, const UNROLL: usize, VS>(
    vs: &VS,
    pov_idx: usize,
    elevation: i16,
    dem_id: i32,
    max_los: usize,
    heatmap: &mut [f32],
    longest_line: &mut [f32],
    line: &[i16],
    index_data: Option<Indexes>,
    unrolled_angles: &UnrolledAngles<WIDTH, UNROLL>,
) where
    LaneCount<WIDTH>: SupportedLaneCount,
    VS: Viewshed<WIDTH>,
{
    let result_tvs_id = dem_to_pov(dem_id, 3 * max_los, max_los);

    // if the line of sight is not within our computable points, do not consider it
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        reason = "max_los^2 < 2^31"
    )]
    if result_tvs_id < 0i32 || result_tvs_id >= (max_los * max_los) as i32 {
        return;
    }

    let pov_height = f32::from(elevation);

    // safety: max_los % WIDTH == 0, so [pov_idx..pov_idx+max_los) will also be WIDTH wide
    let (elevations, _): (&[[i16; WIDTH]], _) =
        unsafe { line.get_unchecked(pov_idx..pov_idx + max_los) }.as_chunks::<WIDTH>();

    let (iter, rest) = index_data.map_or_else(
        || (None, None),
        |data| {
            // safety: pov_idx should be between [0, max_los) and len(indexes_in)==2*max_los
            // thus, for any pov_idx pov_idx..pov_idx+max_los is inbounds
            let (indexes, _): (&[[i32; WIDTH]], _) =
                unsafe { data.indexes_in.get_unchecked(pov_idx..pov_idx + max_los) }
                    .as_chunks::<WIDTH>();

            let (indexes_out, _) = data.indexes_out.as_chunks_mut::<WIDTH>();

            let (chunked_indexes, rest_indexes) = indexes.as_chunks::<UNROLL>();
            let (chunked_indexes_out, rest_indexes_out) = indexes_out.as_chunks_mut::<UNROLL>();

            (
                Some(
                    zip(chunked_indexes.iter(), chunked_indexes_out.iter_mut()).map(
                        |(inds_in, inds_out)| IndexSIMD {
                            indexes_in: inds_in,
                            indexes_out: inds_out,
                        },
                    ),
                ),
                Some(IndexSIMD {
                    indexes_in: rest_indexes,
                    indexes_out: rest_indexes_out,
                }),
            )
        },
    );

    let (chunked_elevs, rest_elevs) = elevations.as_chunks::<UNROLL>();

    let (chunked_distances, rest_distances) = unrolled_angles.distances;
    let (chunked_adjustments, rest_adjustments) = unrolled_angles.adjustments;

    let (local_sums, local_longest, prefix) = izip!(
        chunked_elevs,
        chunked_distances,
        chunked_adjustments,
        OptionIter::new(iter),
    )
        .fold(
            (
                [Simd::splat(0.0); UNROLL],
                [Simd::splat(0.0); UNROLL],
                -2000.0f32,
            ),
            |(sum, longest, prefix), (elevs, dists, adjusts, inds)| {
                let (next_sum, next_longest, acc) = line_of_sight::<WIDTH, UNROLL, VS>(
                    vs,      // elevs: &[[i16; N]],
                    elevs,   // distances: &[Simd<f32, N>],
                    dists,   // adjustments: &[Simd<f32, N>],
                    adjusts, // prefix_in: Simd<f32, N>,
                    prefix, inds,
                    pov_height, // pov_height: f32,
                );

                let mut copied_sum = sum;
                zip(copied_sum.iter_mut(), next_sum).for_each(|(old, new)| {
                    *old += new;
                });

                let mut copied_longest_line = longest;
                zip(copied_longest_line.iter_mut(), next_longest).for_each(|(old, new)| {
                    *old = old.simd_max(new);
                });

                (copied_sum, copied_longest_line, acc)
            },
        );

    let mut sum = local_sums
        .iter()
        .fold(0.0f32, |acc, partial| acc + partial.reduce_sum());

    let mut longest = local_longest
        .iter()
        .fold(0.0f32, |acc, new| acc.max(new.reduce_max()));

    let (sum_buf, longest_buf, _) = line_of_sight::<WIDTH, UNROLL, VS>(
        vs,
        rest_elevs,
        rest_distances,
        rest_adjustments,
        prefix,
        rest,
        pov_height,
    );

    sum += sum_buf
        .iter()
        .fold(0.0f32, |acc, partial| acc + partial.reduce_sum());

    longest = longest_buf
        .iter()
        .fold(longest, |acc, new| acc.max(new.reduce_max()));

    #[expect(
        clippy::as_conversions,
        clippy::cast_sign_loss,
        reason = "result_idx should be in [0, 2^31]"
    )]
    // safety: it is guaranteed by the rotation kernel that if the index is
    // greater than zero that it is in-bounds. This saves ~10% of bounds checks
    unsafe {
        *heatmap.get_unchecked_mut(result_tvs_id as usize) += sum;

        // let old_longest: *mut f32 = longest_line.get_unchecked_mut(result_tvs_id as usize);
        // *old_longest = (*old_longest).max(longest);
    }
}





// chunk: [-2000, .1, .2, .3, .4, .5, .6, .7]
//        [-2000, .1, .2, .3][.4, .5, .6, .7]
//        [.1,    .2, .3, .4][.5, .6, .7, .]
//


/// `precalculate_distances` precalculates earth curvature adjustments and
/// the distance from a particular point (which is just linear)
fn precalculate_distances<const WIDTH: usize>(
    max_los: usize,
) -> (Vec<Simd<f32, WIDTH>>, Vec<Simd<f32, WIDTH>>)
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    (0..max_los)
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
        .unzip()
}

/// `total_viewshed` computes a total viewshed heatmap for a given elevation map,
/// and corresponding indexes to store the rotated data
#[inline]
fn total_viewshed<const WIDTH: usize, const UNROLL: usize, V: Viewshed<WIDTH>>(
    vs: &V,
    elevation_map: &[i16],
    indexes: &[i32],
    max_los: usize,
    output_sector_data: bool,
) -> ViewshedAngle
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    assert_eq!(
        elevation_map.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    assert_eq!(
        indexes.len(),
        2 * max_los * max_los,
        "indexes should be 2 * max_los wide, and max_los tall"
    );

    assert_eq!(
        max_los % WIDTH,
        0,
        "to help the vectorizer, max_los must be a multiple of {WIDTH}"
    );

    let mut sector_data_buf = vec![
        0i32;
        if output_sector_data {
            max_los * max_los * max_los
        } else {
            0
        }
    ];
    let mut heatmap = vec![0.0f32; max_los * max_los];
    let mut longest_line = vec![0.0f32; max_los * max_los];
    let mut sector_data: Option<&mut Vec<i32>> = output_sector_data.then_some(&mut sector_data_buf);

    let width = 2 * max_los;

    // precalculate all distances and their spherical earth "adjustments".
    // This saves ~33% of effort inside our hot loop
    let (distances, adjustments) = precalculate_distances::<WIDTH>(max_los);

    let unrolled_angle = UnrolledAngles {
        distances: distances.as_chunks::<UNROLL>(),
        adjustments: adjustments.as_chunks::<UNROLL>(),
    };

    for (line, line_indexes, sector_chunk) in izip!(
        elevation_map.chunks_exact(width),
        indexes.chunks_exact(width),
        OptionIter::new(
            sector_data
                .as_mut()
                .map(|sd| sd.chunks_exact_mut(max_los * max_los))
        ),
    ) {
        for (pov, (&pov_height, &result_dem_id, line_bitmap)) in izip!(
            line.iter().take(max_los),
            line_indexes.iter().take(max_los),
            OptionIter::new(sector_chunk.map(|chunk| chunk.chunks_exact_mut(max_los)))
        )
            .enumerate()
        {
            viewshed(
                vs,
                pov,
                pov_height,
                result_dem_id,
                max_los,
                &mut heatmap,
                &mut longest_line,
                line,
                line_bitmap.map(|bitmap| Indexes {
                    indexes_in: line_indexes,
                    indexes_out: bitmap,
                }),
                &unrolled_angle,
            );
        }
    }

    ViewshedAngle {
        heatmap,
        longest_line,
        sector_data: sector_data.cloned(),
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

    (rotation, elevations)
}

/// `ViewshedAngle` holds the cumulative result of a single angle in the `total_viewshed` algorithm
#[derive(Debug)]
struct ViewshedAngle {
    /// `heatmap` contains the longest line of sight heatmap for rendering
    heatmap: Vec<f32>,
    /// `longest_line` contains the longest distance for a particular point
    longest_line: Vec<f32>,
    /// `sector_data` holds the visibility calculations for each point in row-major order.
    /// elements `[0..max_los)` are for the first point, `[max_los, 2*max_los)` for the second
    /// point, and so on.
    ///
    /// This gets absolutely massive, so we only allocate this if we know for certain we will be using it
    sector_data: Option<Vec<i32>>,
}

impl ViewshedAngle {
    /// `new` creates a new buffer for viewshed results of `heatmap`,
    /// `longest_line`, and `sector_data`
    fn new(max_los: usize, sector_data: bool) -> Self {
        Self {
            heatmap: vec![0.0f32; max_los * max_los],
            longest_line: vec![0.0f32; max_los * max_los],
            sector_data: sector_data.then(|| Vec::with_capacity(max_los * max_los * max_los)),
        }
    }

    /// `acc` accumulates a single `ViewshedAngle` into another
    fn acc(&mut self, other: &Self) {
        zip(&mut self.heatmap, &other.heatmap).for_each(|(to, from)| {
            *to += *from;
        });

        zip(&mut self.longest_line, &other.longest_line).for_each(|(to, from)| {
            *to = (*to).max(*from);
        });

        if let Some(ref mut sector_data) = &mut self.sector_data {
            if let Some(other_sector_data) = &other.sector_data {
                sector_data.extend_from_slice(other_sector_data);
            }
        }
    }
}

/// `kernel` is a CPU-based total viewshed kernel. It makes use of image rotation tof
/// optimize the cache locality of all lookups for a total viewshed calculation
fn kernel(elevations: &[i16], max_los_points: usize, angle: usize) -> ViewshedAngle {
    assert!(angle < 360, "angle must be [0, 360)");
    let mut start = Instant::now();

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "angle is [0,360), not more than 2^54"
    )]
    let (indexes, rotated_elevations) = generate_rotation(elevations, angle as f64, max_los_points);

    tracing::info!(
        "rotated {:?} in {:?}, calculating viewshed",
        angle,
        start.elapsed()
    );

    start = Instant::now();

    let vectorized = Vectorized {};

    // #[cfg(target_feature = "avx512f")]
    // {
    //     let result = total_viewshed::<16, 8, Vectorized>(
    //         &vectorized,
    //         &rotated_elevations,
    //         &indexes,
    //         max_los_points,
    //         false,
    //     );
    //     tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
    //     return result;
    // };
    //
    // #[cfg(all(target_feature = "avx2", target_feature = "avx"))]
    // {
    //     let result = total_viewshed::<8, 8, Vectorized>(
    //         &vectorized,
    //         &rotated_elevations,
    //         &indexes,
    //         max_los_points,
    //         false,
    //     );
    //     tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
    //     return result;
    // };

    #[expect(
        unreachable_code,
        unused_variables,
        reason = "conditionally compiled out"
    )]
    let result = total_viewshed::<4, 8, Vectorized>(
        &vectorized,
        &rotated_elevations,
        &indexes,
        max_los_points,
        false,
    );
    tracing::info!("kernel for {} run in: {:?}", angle, start.elapsed());
    result
}

/// `multithreaded_kernel` parallelizes CPU kernel calculations for a `core_count` and calculates
/// `num_angles` different angles
pub fn multithreaded_kernel(
    elevations_original: &[i16],
    max_los_points_original: usize,
    num_angles: usize,
    core_count: usize,
    output_sector_data: bool,
) -> (Vec<f32>, Vec<f32>, Option<Vec<i32>>) {
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

    #[expect(
        clippy::expect_used,
        reason = "threadpool must be created for program to run"
    )]
    let pool = ThreadPoolBuilder::new()
        .num_threads(core_count)
        .build()
        .expect("couldn't build threadpool");

    let mut final_angle = ViewshedAngle::new(max_los_points, output_sector_data);
    let angle_mu = &Mutex::new(&mut final_angle);

    pool.install(move || {
        (0..num_angles)
            .into_par_iter()
            .map(|angle| kernel(elevations, max_los_points, angle))
            .for_each(|vs| {
                #[expect(clippy::expect_used, reason = "a poisoned mutex should crash")]
                let mut angle_guard = angle_mu.lock().expect("mutex poisoned");

                angle_guard.acc(&vs);
            });
    });

    let (heatmap, longest_line, sector) = (
        mem::take(&mut final_angle.heatmap),
        mem::take(&mut final_angle.longest_line),
        final_angle
            .sector_data
            .map(|mut sector_data| mem::take(&mut sector_data)),
    );

    (heatmap, longest_line, sector)
}

#[cfg(test)]
mod test {
    use std::simd::Simd;
    use itertools::fold;
    use crate::cpu::{line_of_sight, Vectorized};

    #[test]
    fn viewshed() {
        let vs = Vectorized {};
        let (sums, distances, fold_out) = line_of_sight::<4, 1, Vectorized>(
            &vs,
            &[[1, 4, -9, 16]],
            &[Simd::from_array([1.0, 2.0, 3.0, 4.0])],
            &[Simd::splat(0.0)],
            -2000.0,
            None,
            0.0f32,
        );
        println!("{:?}", (sums, distances, fold_out));

        let (sums, distances, fold_out) = line_of_sight::<4, 1, Vectorized>(
            &vs,
            &[[20, 24, 28, 32]],
            &[Simd::from_array([5.0, 6.0, 7.0, 8.0])],
            &[Simd::splat(0.0)],
            fold_out,
            None,
            0.0f32,
        );
        println!("{:?}", (sums, distances, fold_out));
    }
}
