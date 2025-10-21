//! `cpu` is a CPU version of the total viewshed calculation

use itertools::izip;
use std::arch::x86_64::{
    __m128, __m256, _mm256_blend_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_max_ps,
    _mm256_set1_ps, _mm256_slli_si256, _mm_broadcast_ss, _mm_max_ps, _mm_set1_ps,
};
use std::iter::zip;
use std::mem::transmute;
use std::simd::Simd;
use std::thread;
use std::time::Instant;

/// `EARTH_RADIUS_SQUARED` is the earth's radius squared in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

/// `TAN_ONE_RAD` helps normalize the fact that inner points are sampled more often
/// see the TVS paper for reasoning.
const TAN_ONE_RAD: f32 = 0.017_453_3;

/// `prefix_max` carries out a SIMD prefix max using AVX instructions a la
/// <https://en.algorithmica.org/hpc/algorithms/prefix/> This method is a bit inefficient from an
/// algorithmic perspective, as we are doing n*log(n) amount of work, but in the end, the ability
/// to make use of AVX makes up for this, and doubles the speed on benchmarks.
///
/// First we compute a prefix max for blocks of 4 f32s, but 8 at a time
/// (like `prefix` in algorithmica's algorithm).
/// Each AVX vector register has 8 lanes, but are in groups of 4:
///
/// ```
/// angle_vec = | a_1 | a_2 | a_3 | a_4 | b_1 | b_2 | b_3 | b_4 |
/// ```
///
/// Operations such as `max_ps` operate on 128bit chunks, but two at a time.
///
/// To do start we shift the lanes left by 4 bytes in an intermediary register leaving zeros:
///
/// ```
/// | 0 | a_1 | a_2 | a_3 | 0 | b_1 | b_2 | b_3 |
/// ```
///
/// Having zero shifted in is helpful if you are doing a prefix sum since 0 added to anything is itself,
/// an identity element. However, zero is not a good identity element for `fmax`. Instead, we use
/// -2000.0, which is lower than any angle calculation that we'll ever do making sure that
/// ident(x) = fmax(x, -2000.0) = x.
///
/// We blend in a constant vector full of `-2000.0`s, via `blend_ps`, passing in a mask
/// of `0b1000_1000` which makes sure to only blend the first elements:
/// ```
/// shifted_vec = | -2000.0 | a_1 | a_2 | a_3 | -2000.0 | b_1 | b_2 | b_3 |
/// ```
///
/// And finally:
/// ```
/// v_prefix_max = max_ps(angle_vec, shifted_vec)
/// = max (|   a_1   | a_2 | a_3 | a_4 |   b_1   | b_2 | b_3 | b_4 |,
///        | -2000.0 | a_1 | a_2 | a_3 | -2000.0 | b_2 | b_3 | b_4 |)
/// =      |   a_1   | max(a_1, a_2) | max(a_2, a_3) | ...
///```
///
/// Visually, it should be clear that we have only computed the prefix max for the
/// first two out of four elements. We can repeat this trick a second time but this time
/// shifting our prefix max twice (and blending with a mask of `0b1100_1100`),
/// fully computing the prefix sum for both blocks of 4 elements.
/// For brevity, only the first 4 lanes are pictured:
///
/// ```
/// v_prefix_max = |   a_1   | max(a_1, a_2) | max(a_2, a_3) | max(a_3, a_4) |
/// shifted_vec =  | -2000.0 |    -2000.0    | a_1           | max(a_1, a_2) |
///
/// v_prefix_max = max_ps(v_prefix_max, shifted_vec)
/// = | a_1 | max(a_1, a_2) | max(a_1, max(a_2, a_3)) | max(max(a_1, a_2), max(a_3, a_4))
///```
///
/// MAGICAL
///
/// However, now we have blocks of 4 prefix maxes calculated, not a prefix max calculated
/// for the full array. To do so, we need to make a second pass to accumulate the
/// results across blocks. Doing this is fairly simple. Take the last element of
/// each block and "splat" it across all lanes:
///
/// `highest_from_block = | cur_block[3] | cur_block[3] | cur_block[3] | cur_block[3] |`
///
/// Our global max is 4 lanes of the max of all previous maxes of all other blocks
/// In other words, it is an accumulated max:
///
/// ```
/// global_max = | x | x | x | x |
/// ```
///
/// Our new current block needs to be updated with the computation from the `global_max`,
/// so just re-compute the current block by taking the max of it and `global_max`:
///
/// ```
/// cur_block = max_ps(cur_block, global_max)
/// ```
///
/// and our new `global_max` is a scalar spatted across all lanes of the cumulative
/// maximum of all other blocks:
///
/// ```
/// global_max = max_px(highest_from_block, global_max)
/// ```
///
/// And there you have it!
///
/// To check to see if a point is visible, we can check whether the point is greater
/// than or equal to the current prefix max:
///
/// ```visible(index) = angle[index] >= prefix_max[index]```
#[target_feature(enable = "avx2")]
#[inline]
fn prefix_max_simd(angles: &[__m256], prefix_max: &mut [__m256]) {
    // constant lowest value spatted across all 8 lanes
    let max_angles = _mm256_set1_ps(-2000.0f32);

    // Calculate the 4-wide block prefix max two at a time
    for (prefix, &angle) in zip(prefix_max.iter_mut(), angles.iter()) {
        let mut v_prefix_max: __m256 = {
            let shifted = _mm256_slli_si256::<4>(_mm256_castps_si256(angle));
            let blended = _mm256_blend_ps::<0b1000_1000>(_mm256_castsi256_ps(shifted), max_angles);

            _mm256_max_ps(angle, blended)
        };

        v_prefix_max = {
            let shifted = _mm256_slli_si256::<8>(_mm256_castps_si256(v_prefix_max));
            let blended = _mm256_blend_ps::<0b1100_1100>(_mm256_castsi256_ps(shifted), max_angles);
            _mm256_max_ps(v_prefix_max, blended)
        };

        *prefix = v_prefix_max;
    }

    #[expect(
        clippy::transmute_ptr_to_ptr,
        reason = "transmute handles the slice size conversion, pointers don't"
    )]
    // safety: sizeof(__m256) ==  sizeof(__m128) * 2, meaning it is well-aligned
    let single_wide_angles = unsafe { transmute::<&mut [__m256], &mut [__m128]>(prefix_max) };

    let mut acc = _mm_set1_ps(-2000.0f32);

    // accumulate the prefix maxes for blocks, re-computing all prefix maxes
    // to include the accumulated value
    for prefix in single_wide_angles {
        let cur_prefix = Simd::from(*prefix);
        let cur_max = _mm_broadcast_ss(&cur_prefix[3]);

        *prefix = _mm_max_ps(acc, *prefix);
        acc = _mm_max_ps(acc, cur_max);
    }
}

/// `total_viewshed` calculates a straight-lined total viewshed using AVX2 instructions
#[target_feature(enable = "avx2")]
#[expect(
    clippy::indexing_slicing,
    reason = "line_indexes is max_los long so pov is always in bounds"
)]
fn total_viewshed_vector(
    elevation_map: &[i16],
    indexes: &[i32],
    max_los: usize,
    result: &mut [f32],
) {
    assert_eq!(
        elevation_map.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    assert_eq!(
        max_los % 8,
        0,
        "to help the vectorizer, max_los must be a multiple of 8"
    );

    let width = 2 * max_los;

    // precalculate all distances and their spherical earth "adjustments".
    // This saves ~33% of effort inside our hot loop
    let (distances, adjustments): (Vec<f32>, Vec<f32>) = (0..max_los)
        .map(|x| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_precision_loss,
                reason = "x is in [1..max_los), max_los < 2^23"
            )]
            let distance = (x * 100) as f32;
            (distance, distance / EARTH_RADIUS_SQUARED)
        })
        .unzip();

    // allocate angles and their prefix maxes as mm256s so that the underlying buffer
    // is correctly aligned to 256 bits
    #[expect(clippy::integer_division, reason = "max_los % 8 == 0")]
    let mut angles: Vec<__m256> = vec![_mm256_set1_ps(0.0); max_los / 8];

    #[expect(clippy::integer_division, reason = "max_los % 8 == 0")]
    let mut prefix_max: Vec<__m256> = vec![_mm256_set1_ps(0.0); max_los / 8];

    for line_idx in 0..max_los {
        let elevation_offset = line_idx * width;

        {}
        let line = &elevation_map[elevation_offset..(elevation_offset + width)];

        let indexes_offset = line_idx * max_los;

        let line_indexes = &indexes[indexes_offset..(indexes_offset + max_los)];

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
            let elevations = line[pov..pov + max_los]
                .iter()
                .map(|&x| f32::from(x) - pov_height);

            #[expect(
                clippy::transmute_ptr_to_ptr,
                reason = "transmute handles the slice size conversion, pointers don't"
            )]
            // safety: sizeof(__m256) ==  sizeof(f32) * 8, meaning it is well-aligned
            let flat_angles = unsafe { transmute::<&mut [__m256], &mut [f32]>(angles.as_mut()) };

            izip!(flat_angles.iter_mut(), &distances, &adjustments, elevations).for_each(
                |(angle, distance, adjustment, elevation)| {
                    *angle = (elevation / distance) - adjustment;
                },
            );

            // get rid of NaN in the first angle calculation since there is a division by zero
            flat_angles[0] = -2001.0;

            prefix_max_simd(&angles, &mut prefix_max);

            #[expect(
                clippy::transmute_ptr_to_ptr,
                reason = "transmute handles the slice size conversion, pointers don't"
            )]
            // safety: sizeof(__m256) ==  sizeof(f32) * 8, meaning it is well-aligned
            let flat_prefixes = unsafe { transmute::<&[__m256], &[f32]>(&prefix_max) };

            let surface_bitmap: Vec<bool> = zip(flat_angles, flat_prefixes)
                .map(|(&mut angle, &prefix)| angle >= prefix)
                .collect();

            let sum = zip(&distances, &surface_bitmap).fold(
                0.0f32,
                |surface_area, (distance, &visible)| {
                    if visible {
                        distance.mul_add(TAN_ONE_RAD, surface_area)
                    } else {
                        surface_area
                    }
                },
            );

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

/// `total_viewshed` calculates the total viewshed for a square `elevation_map` and stores the
/// result in `result`
fn total_viewshed(elevation_map: &[i16], indexes: &[i32], max_los: usize, result: &mut [f32]) {
    if cfg!(target_feature = "avx2") {
        // safety: branch is only run when avx2 is enabled
        unsafe { total_viewshed_vector(elevation_map, indexes, max_los, result) }
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
fn kernel(elevations: &[i16], max_los_points: usize, angle: usize, res: &mut [f32]) {
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

    total_viewshed(&rotated_elevations, &indexes, max_los_points, res);
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
            res = zip(res, thread.join().unwrap())
                .map(|(acc, heatmap)| acc + heatmap)
                .collect();
        }
        res
    })
}
