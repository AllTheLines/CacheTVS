use itertools::izip;
use std::iter::zip;
use std::simd::cmp::SimdPartialOrd as _;
use std::simd::num::SimdFloat as _;
use std::simd::prelude::SimdInt as _;
use std::simd::{f32x4, f32x8, LaneCount, Mask, Simd, SupportedLaneCount};

/// `TAN_ONE_RAD` is used in normalizing the
const TAN_ONE_RAD: f32 = 0.017_453_3;

use crate::cpu::los::{Accumulate, Angle, PrefixMax, Unroll};

/// `VectorMax` performs an element-wise SIMD max of floats, allowing for architecture
/// specific implementations
trait VectorMax<const WIDTH: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    /// `max` computes an element-wise maximum from lhs and rhs, assuming neither contain NaNs
    /// or -0.0
    fn max(lhs: Simd<f32, WIDTH>, rhs: Simd<f32, WIDTH>) -> Simd<f32, WIDTH>;
}

/// `VectorGreater` performs a SIMD greater than of floats, allowing for architecture
/// specific implementations
trait VectorGreater<const WIDTH: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    /// gt computes an element-wise maximum from lhs and rhs, assuming neither contain NaNs
    /// or -0.0
    fn gt(lhs: Simd<f32, WIDTH>, rhs: Simd<f32, WIDTH>) -> Mask<i32, WIDTH>;
}

/// `VectorLos` is an implementation of the internals of `PrefixMax`, `Angle`,  and `Accumulate`
/// for Portable SIMD
pub struct VectorLos<const WIDTH: usize>
where
    LaneCount<WIDTH>: SupportedLaneCount;

impl<const WIDTH: usize> VectorMax<WIDTH> for VectorLos<WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    #[inline]
    default fn max(lhs: Simd<f32, WIDTH>, rhs: Simd<f32, WIDTH>) -> Simd<f32, WIDTH> {
        lhs.simd_max(rhs)
    }
}

impl<const WIDTH: usize> VectorGreater<WIDTH> for VectorLos<WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    #[inline]
    default fn gt(lhs: Simd<f32, WIDTH>, rhs: Simd<f32, WIDTH>) -> Mask<i32, WIDTH> {
        lhs.simd_gt(rhs)
    }
}

#[cfg(target_feature = "sse")]
impl VectorMax<4> for VectorLos<4> {
    #[inline]
    fn max(lhs: f32x4, rhs: f32x4) -> f32x4 {
        use std::arch::x86_64::_mm_max_ps;

        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe { _mm_max_ps(lhs.into(), rhs.into()) }.into()
    }
}

#[cfg(target_feature = "avx")]
impl VectorGreater<4> for VectorLos<4> {
    fn gt(lhs: f32x4, rhs: f32x4) -> Mask<i32, 4> {
        use std::arch::x86_64::{_mm_castps_si128, _mm_cmp_ps, _CMP_GT_OS};

        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask = _mm_castps_si128(_mm_cmp_ps::<_CMP_GT_OS>(lhs.into(), rhs.into()));
            Mask::<i32, 4>::from_int_unchecked(mask.into())
        }
    }
}

impl PrefixMax for VectorLos<4> {
    #[inline]
    fn prefix_max(highest: f32, angles_in: &[f32], angles_out: &mut [f32]) {
        let (vector_angles, _) = angles_in.as_chunks::<4>();
        let (vector_prefix, _) = angles_out.as_chunks_mut::<4>();

        for (prefix, &angle) in zip(vector_prefix.iter_mut(), vector_angles.iter()) {
            let start = Simd::from(angle);

            let mut v_prefix_max = {
                let shifted = start.shift_elements_right::<1>(-2000.0f32);
                Self::max(start, shifted)
            };

            v_prefix_max = {
                let shifted = v_prefix_max.shift_elements_right::<2>(-2000.0f32);
                Self::max(v_prefix_max, shifted)
            };

            v_prefix_max.copy_to_slice(prefix);
        }

        let mut local_acc = Simd::splat(highest);

        // accumulate the prefix maxes for blocks, re-computing all prefix maxes
        // to include the accumulated value
        for prefix in vector_prefix {
            let cur_prefix: f32x4 = Simd::from(*prefix);
            let cur_max: f32x4 = Simd::splat(cur_prefix[3]);

            Self::max(local_acc, cur_prefix).copy_to_slice(prefix);
            local_acc = Self::max(local_acc, cur_max);
        }
    }
}

#[cfg(target_feature = "avx")]
impl VectorMax<8> for VectorLos<8> {
    #[inline]
    fn max(lhs: f32x8, rhs: f32x8) -> f32x8 {
        use std::arch::x86_64::_mm256_max_ps;
        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe { _mm256_max_ps(lhs.into(), rhs.into()).into() }
    }
}

#[cfg(all(
    target_feature = "sse",
    target_feature = "avx",
    target_feature = "avx2"
))]
impl PrefixMax for VectorLos<8> {
    #[inline]
    fn prefix_max(highest: f32, angles_in: &[f32], angles_out: &mut [f32]) {
        use std::arch::x86_64::{
            _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_slli_si256,
            _mm_max_ps,
        };

        debug_assert!(
            angles_in.len().is_multiple_of(8) && angles_in.len() == angles_out.len(),
            "inconsistent lengths, buffer must be multiple of vector length"
        );
        {
            let (vector_angles, _) = angles_in.as_chunks::<8>();
            let (vector_prefix, _) = angles_out.as_chunks_mut::<8>();

            izip!(vector_prefix.iter_mut(), vector_angles.iter()).for_each(|(prefix, &angle)| {
                let start = Simd::from_array(angle);
                // safety: PrefixMax for VectorLos<8> is guarded by a cfg block for all SIMD instructions
                let mut v_prefix_max = unsafe {
                    let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(start.into()));
                    let blended = _mm256_blend_ps::<0b0001_0001>(
                        _mm256_castsi256_ps(shifted),
                        Simd::splat(-2000.0f32).into(),
                    );
                    Self::max(start, blended.into())
                };

                // safety: PrefixMax for VectorLos<8> is guarded by a cfg block for all SIMD instructions
                v_prefix_max = unsafe {
                    let shifted = _mm256_slli_si256::<8>(_mm256_castps_si256(v_prefix_max.into()));
                    let blended = _mm256_blend_ps::<0b0011_0011>(
                        _mm256_castsi256_ps(shifted),
                        Simd::splat(-2000.0f32).into(),
                    );

                    Self::max(v_prefix_max, blended.into())
                };

                v_prefix_max.copy_to_slice(prefix);
            });
        };

        {
            let mut acc: f32x4 = Simd::splat(highest);
            let (vector_prefix, _) = angles_out.as_chunks_mut::<4>();
            for prefix in vector_prefix.iter_mut() {
                // safety: PrefixMax for VectorLos<8> is guarded by a cfg block for all SIMD instructions
                let new_max: f32x4 =
                    unsafe { _mm_max_ps(Simd::from_array(*prefix).into(), acc.into()) }.into();

                let cur_max = Simd::splat(prefix[3]);

                // safety: PrefixMax for VectorLos<8> is guarded by a cfg block for all SIMD instructions
                acc = unsafe { _mm_max_ps(acc.into(), cur_max.into()).into() };

                new_max.copy_to_slice(prefix);
            }
        }
    }
}

#[cfg(target_feature = "avx")]
impl VectorGreater<8> for VectorLos<8> {
    #[inline]
    fn gt(lhs: f32x8, rhs: f32x8) -> Mask<i32, 8> {
        use std::arch::x86_64::{_mm256_castps_si256, _mm256_cmp_ps, _CMP_GT_OS};

        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OS>(lhs.into(), rhs.into()));
            Mask::<i32, 8>::from_int_unchecked(mask.into())
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl VectorMax<16> for VectorLos<16> {
    #[inline]
    fn max(lhs: Simd<f32, 16>, rhs: Simd<f32, 16>) -> Simd<f32, 16> {
        use std::arch::x86_64::_mm512_max_ps;
        // safety: the caller of Viewshed<4> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe { _mm512_max_ps(lhs.into(), rhs.into()).into() }
    }
}

#[cfg(target_feature = "avx512f")]
impl VectorGreater<16> for VectorLos<16> {
    #[inline]
    fn gt(lhs: Simd<f32, 16>, rhs: Simd<f32, 16>) -> Mask<i32, 16> {
        use std::arch::x86_64::_mm512_cmple_ps_mask;
        // safety: the caller of Viewshed<8> guarantees that -0.0 or NaN are not in the input
        // thus allowing this to be non IEEE754 compliant
        unsafe {
            let mask = _mm512_cmple_ps_mask(lhs.into(), rhs.into());
            Mask::<i32, 16>::from_bitmask(mask.into())
        }
    }
}

#[cfg(target_feature = "avx512f")]
impl PrefixMax for VectorLos<16> {
    #[inline]
    fn prefix_max(highest: f32, angles_in: &[f32], angles_out: &mut [f32]) {
        use std::arch::x86_64::{
            __m512, _mm512_alignr_epi32, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_max_ps,
        };
        use std::simd::f32x16;

        #[expect(
            clippy::cast_sign_loss,
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "K is always in bounds"
        )]
        fn mm512_slli_si512<const K: usize>(elem: __m512) -> __m512
        where
            [(); { (16 - K) as i32 } as usize]:,
        {
            // safety: all mm512 intrinsics are guarded by avx512f
            unsafe {
                let zero = f32x16::splat(-2000.0f32);
                _mm512_castsi512_ps(_mm512_alignr_epi32::<{ (16 - K) as i32 }>(
                    _mm512_castps_si512(elem),
                    _mm512_castps_si512(zero.into()),
                ))
            }
        }

        let (vector_prefix, _) = angles_out.as_chunks_mut::<16>();
        let (vector_angle, _) = angles_in.as_chunks::<16>();

        for (prefix, &angle) in zip(vector_prefix.iter_mut(), vector_angle.iter()) {
            let simd_angle = Simd::from_array(angle);
            // safety: all the following operations are guarded by the avx512f build falg
            unsafe {
                let mut v_prefix_max =
                    _mm512_max_ps(simd_angle.into(), mm512_slli_si512::<1>(simd_angle.into()));

                v_prefix_max = _mm512_max_ps(v_prefix_max, mm512_slli_si512::<2>(v_prefix_max));

                v_prefix_max = _mm512_max_ps(v_prefix_max, mm512_slli_si512::<4>(v_prefix_max));

                v_prefix_max = _mm512_max_ps(v_prefix_max, mm512_slli_si512::<8>(v_prefix_max));

                let simd_prefix_max: f32x16 = v_prefix_max.into();
                simd_prefix_max.copy_to_slice(prefix);
            }
        }

        let mut local_acc = f32x16::splat(highest);

        // accumulate the prefix maxes for blocks, re-computing all prefix maxes
        // to include the accumulated value
        for prefix in vector_prefix.iter_mut() {
            let cur_prefix: f32x16 = Simd::from_array(*prefix);
            let cur_max: f32x16 = Simd::splat(cur_prefix[15]);

            Self::max(local_acc, cur_prefix).copy_to_slice(prefix);
            local_acc = Self::max(local_acc, cur_max);
        }
    }
}

impl<const WIDTH: usize> Angle for VectorLos<WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
{
    #[inline]
    default fn calculate_angles(
        pov_height: f32,
        elevations: &[i16],
        distances: &[f32],
        adjustments: &[f32],
        angles_out: &mut [f32],
    ) {
        debug_assert!(
            elevations.len().is_multiple_of(WIDTH),
            "expected elevations to be a multiple of {WIDTH}",
        );

        debug_assert!(
            distances.len().is_multiple_of(WIDTH),
            "expected distances to be a multiple of {WIDTH}",
        );

        debug_assert!(
            adjustments.len().is_multiple_of(WIDTH),
            "expected adjustments to be a multiple of {WIDTH}",
        );

        debug_assert!(
            angles_out.len().is_multiple_of(WIDTH),
            "expected angles buf to be a multiple of {WIDTH}",
        );

        let (vector_angles, _) = angles_out.as_chunks_mut::<{ WIDTH }>();
        let (vector_elevations, _) = elevations.as_chunks::<{ WIDTH }>();
        let (vector_adjustments, _) = adjustments.as_chunks::<{ WIDTH }>();
        let (vector_distances, _) = distances.as_chunks::<{ WIDTH }>();

        for (angle, &elevation, &distance, &adjustment) in izip!(
            vector_angles.iter_mut(),
            vector_elevations.iter(),
            vector_distances.iter(),
            vector_adjustments.iter()
        ) {
            let elevation_f32: Simd<f32, { WIDTH }> = Simd::from(elevation).cast();
            let height_delta = elevation_f32 - Simd::splat(pov_height);
            let res = (height_delta + Simd::from_array(adjustment)) / Simd::from_array(distance);
            res.copy_to_slice(angle);
        }
    }
}

/// `GenericExpr` lets a const generic expression be evaluated in its
/// `CONDITION` parameter for traits that need to evaluate constant expressions
/// as part of their trait bounds
struct GenericExpr<const CONDITION: bool>;

/// `IsTrue `is a "marker" trait for trait bounds for when a const generic expr
/// evaluates to true, and is only implemented for `GenericExpr<{true}>`
trait IsTrue {}
impl IsTrue for GenericExpr<true> {}

impl<const SIZE: usize, const WIDTH: usize> Accumulate<Unroll<SIZE>> for VectorLos<WIDTH>
where
    LaneCount<WIDTH>: SupportedLaneCount,
    GenericExpr<{ SIZE.is_multiple_of(WIDTH) }>: IsTrue,
{
    #[inline]
    #[expect(clippy::allow_attributes, reason = "conditional attributes")]
    #[allow(
        unused,
        unused_variables,
        reason = "conditional compilation causes dead parameters"
    )]
    fn accumulate(
        mut init: Unroll<SIZE>,
        angles: &[f32],
        prefix: &[f32],
        distances: &[f32],
        bitmap: &mut Vec<bool>,
    ) -> Unroll<SIZE> {
        debug_assert!(
            angles.len().is_multiple_of(WIDTH),
            "distance unroll should be multiple of width"
        );
        debug_assert!(
            prefix.len().is_multiple_of(WIDTH),
            "distance unroll should be multiple of width"
        );
        debug_assert!(
            distances.len().is_multiple_of(WIDTH),
            "distance unroll should be multiple of width"
        );
        debug_assert!(angles.len() <= SIZE, "angles must be less than unroll size");

        let (vector_sum, _) = init.heatmap.as_chunks_mut::<{ WIDTH }>();
        let (vector_longest, _) = init.longest.as_chunks_mut::<{ WIDTH }>();

        let (vector_angles, _) = angles.as_chunks::<{ WIDTH }>();
        let (vector_prefix, _) = prefix.as_chunks::<{ WIDTH }>();
        let (vector_dists, _) = distances.as_chunks::<{ WIDTH }>();

        izip!(
            vector_sum,
            vector_longest,
            vector_angles,
            vector_prefix,
            vector_dists,
        )
        .for_each(
            |(sum_arr, longest_arr, &angle_arr, &prefix_arr, &distances_arr)| {
                let mask = Self::gt(Simd::from_array(angle_arr), Simd::from_array(prefix_arr));

                if cfg!(any(test, feature = "ring_data")) {
                    bitmap.extend(mask.to_array());
                }

                if !mask.any() {
                    return;
                }

                let dist = mask.select(Simd::from_array(distances_arr), Simd::splat(0.0f32));

                Self::max(Simd::from_array(*longest_arr), dist).copy_to_slice(longest_arr);

                let acc = Simd::from(*sum_arr) + (dist * Simd::splat(TAN_ONE_RAD));

                acc.copy_to_slice(sum_arr);
            },
        );

        init
    }
}

/// `DEFAULT_VECTOR_LENGTH` determines the CPU Kernel's default vector length based off
/// the architecture that the binary is built for
pub const DEFAULT_VECTOR_LENGTH: usize = const {
    if cfg!(any(test, feature = "ring_data")) {
        4
    } else if cfg!(target_feature = "avx512f") {
        16
    } else if cfg!(all(
        target_feature = "sse",
        target_feature = "avx",
        target_feature = "avx2"
    )) {
        8
    } else {
        4
    }
};

impl<const UNROLL: usize> From<Unroll<UNROLL>> for (f32, f32)
where
    GenericExpr<{ UNROLL.is_multiple_of(DEFAULT_VECTOR_LENGTH) }>: IsTrue,
{
    fn from(val: Unroll<UNROLL>) -> Self {
        let (heatmap, _) = val.heatmap.as_chunks::<DEFAULT_VECTOR_LENGTH>();
        let (longest, _) = val.longest.as_chunks::<DEFAULT_VECTOR_LENGTH>();

        let heat = heatmap
            .iter()
            .fold(Simd::splat(0.0f32), |acc, &heat| {
                acc + Simd::from_array(heat)
            })
            .reduce_sum();

        let long = longest
            .iter()
            .fold(Simd::splat(0.0f32), |acc, &long| {
                VectorLos::<DEFAULT_VECTOR_LENGTH>::max(acc, Simd::from_array(long))
            })
            .reduce_max()
            / 100.0;

        (heat, long)
    }
}

#[cfg(test)]
mod test {
    use crate::cpu::los::{LineOfSight as _, UnrolledLOS};
    use crate::cpu::vector::VectorLos;

    #[test]
    #[cfg(all(
        target_feature = "sse",
        target_feature = "avx",
        target_feature = "avx2"
    ))]
    fn line_of_sightsame() {
        let mut vs = UnrolledLOS::<64>::new(16, 0.13);
        let (visibility_four, longest_four, sector_four) = vs.line_of_sight::<VectorLos<4>>(
            0.0f32,
            &[
                100, 0, 300, 400, 500, 0, 300, 0, 100, 0, 300, 0, 100, 0, 300, 0,
            ],
        );

        let (visibility_eight, longest_eight, sector_eight) = vs.line_of_sight::<VectorLos<8>>(
            0.0f32,
            &[
                100, 0, 300, 400, 500, 0, 300, 0, 100, 0, 300, 0, 100, 0, 300, 0,
            ],
        );

        assert_eq!(visibility_four, visibility_eight);
        assert_eq!(longest_four, longest_eight);
        assert_eq!(sector_four, sector_eight);
    }
}
