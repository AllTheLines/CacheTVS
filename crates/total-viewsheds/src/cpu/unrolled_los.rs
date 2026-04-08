use crate::cpu::los::{Accumulate, Angle, LineOfSight, PrefixMax};
use crate::cpu::vector_intrinsics::{VectorGreater, VectorLos, VectorMax as _};
use itertools::izip;
use std::simd::prelude::SimdFloat as _;
use std::simd::{Select as _, Simd};

/// `EARTH_RADIUS_SQUARED` is the radius of the earth in meters
const EARTH_DIAMETER: f32 = 12_742_000.0;

/// `generate_distances` generates the distance from
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "max_los is < 2^24"
)]
fn generate_distances(max_los: usize, refraction: f32, scale: f32) -> (Vec<f32>, Vec<f32>) {
    let adjusted_refraction = refraction - 1.0;

    (1..=max_los)
        .map(|step| {
            let distance = (step as f32) * scale;
            let adjustment = (distance * distance * adjusted_refraction) / EARTH_DIAMETER;

            (distance, adjustment)
        })
        .unzip()
}

/// Unroll holds an unrolled heatmap and unrolled longest line of sight calculation
/// Since in Line of Sight-land max/addition are commutative, then Unroll will be materialized
/// into (f32, f32)
pub struct UnrollVector<const UNROLL: usize, const VECTOR_WIDTH: usize>
where
    [(); UNROLL * VECTOR_WIDTH]:,
{
    /// `heatmap` contains the summation of visible surface areas which will be reduced to a single
    /// surface area at the end
    pub heatmap: [f32; UNROLL * VECTOR_WIDTH],
    /// `longest` contains many long lines of sight which will be reduced to a single
    /// line of sight at the end
    pub longest: [f32; UNROLL * VECTOR_WIDTH],
}

/// `UnrolledLOS` implements an Unrolled `LineOfSight` calculation
pub struct UnrolledVectorLos<const UNROLL: usize, const VECTOR_WIDTH: usize> {
    /// `angles` holds a buffer for line of sight angles to be put into
    /// which is exactly `max_los+1` long
    angles: Vec<f32>,
    /// `distances` holds `max_los` distances
    distances: Vec<f32>,
    /// `adjustments` holds `max_los` earth curvature adjustments
    adjustments: Vec<f32>,
}

impl<const UNROLL: usize, const VECTOR_WIDTH: usize> From<UnrollVector<UNROLL, VECTOR_WIDTH>>
    for (f32, f32)
where
    [(); UNROLL * VECTOR_WIDTH]:,
{
    #[inline]
    fn from(val: UnrollVector<UNROLL, VECTOR_WIDTH>) -> Self {
        let (heatmap, _) = val.heatmap.as_chunks::<VECTOR_WIDTH>();
        let (longest, _) = val.longest.as_chunks::<VECTOR_WIDTH>();

        let heat = heatmap
            .iter()
            .fold(Simd::splat(0.0f32), |acc, &heat| {
                acc + Simd::from_array(heat)
            })
            .reduce_sum();

        let long = longest
            .iter()
            .fold(Simd::splat(0.0f32), |acc, &long| {
                VectorLos::<VECTOR_WIDTH>::max(acc, Simd::from_array(long))
            })
            .reduce_max();

        (heat, long)
    }
}

impl<const UNROLL: usize, const VECTOR_WIDTH: usize> UnrolledVectorLos<UNROLL, VECTOR_WIDTH> {
    /// `new` initializes a new `UnrolledLOS`, and precalculates all the distances
    /// and earth curvature adjustments
    pub fn new(max_los: usize, refraction: f32, scale: f32) -> Self {
        assert_eq!(
            max_los % VECTOR_WIDTH,
            0,
            "the maximum line of sight must be divisible by {VECTOR_WIDTH} for vectorization"
        );

        let (distances, adjustments) = generate_distances(max_los, refraction, scale);

        Self {
            distances,
            adjustments,
            angles: vec![-2000.0f32; max_los + 1],
        }
    }
}

impl<const UNROLL: usize, const VECTOR_WIDTH: usize>
    LineOfSight<UnrollVector<UNROLL, VECTOR_WIDTH>, VectorLos<VECTOR_WIDTH>>
    for UnrolledVectorLos<UNROLL, VECTOR_WIDTH>
where
    [(); UNROLL * VECTOR_WIDTH + 1]:,
    VectorLos<VECTOR_WIDTH>: PrefixMax + Angle,
{
    #[expect(
        clippy::indexing_slicing,
        reason = "all indexing and slices are guaranteed by construction of a UnrolledLOS"
    )]
    #[inline]
    fn line_of_sight(&mut self, pov_height: f32, line: &[i16]) -> (f32, f32, Vec<bool>) {
        let mut prefix_max = [0.0f32; UNROLL * VECTOR_WIDTH];

        prefix_max[UNROLL * VECTOR_WIDTH - 1] = -2000.0;

        VectorLos::<VECTOR_WIDTH>::calculate_angles(
            pov_height,
            line,
            &self.distances,
            &self.adjustments,
            &mut self.angles[1..],
        );

        let mut output: Vec<bool> = if cfg!(any(test, feature = "ring_data")) {
            Vec::with_capacity(line.len())
        } else {
            vec![]
        };


        let (chunked_prefix_angles, rest_prefix_angles) =
            self.angles[..self.angles.len() - 1].as_chunks::<{ UNROLL * VECTOR_WIDTH }>();
        let (chunked_angles, rest_angles) =
            self.angles[1..].as_chunks::<{ UNROLL * VECTOR_WIDTH }>();

        let (chunked_distances, rest_distances) =
            self.distances.as_chunks::<{ UNROLL * VECTOR_WIDTH }>();

        let los = izip!(chunked_prefix_angles, chunked_angles, chunked_distances).fold(
            UnrollVector::<UNROLL, VECTOR_WIDTH> {
                longest: [0.0; UNROLL * VECTOR_WIDTH],
                heatmap: [0.0; UNROLL * VECTOR_WIDTH],
            },
            |acc, (prefix_angles, angles, distances)| {
                VectorLos::<VECTOR_WIDTH>::prefix_max(
                    prefix_max[UNROLL * VECTOR_WIDTH - 1],
                    prefix_angles,
                    &mut prefix_max,
                );

                VectorLos::<VECTOR_WIDTH>::accumulate(
                    acc,
                    angles,
                    &prefix_max,
                    distances,
                    &mut output,
                )
            },
        );

        VectorLos::<VECTOR_WIDTH>::prefix_max(
            prefix_max[UNROLL * VECTOR_WIDTH - 1],
            rest_prefix_angles,
            &mut prefix_max[..rest_angles.len()],
        );

        let new_acc = VectorLos::<VECTOR_WIDTH>::accumulate(
            los,
            rest_angles,
            &prefix_max[..rest_angles.len()],
            rest_distances,
            &mut output,
        );

        let (heatmap, longest) = new_acc.into();
        (heatmap, longest, output)
    }
}

/// `TAN_ONE_RAD` is used in normalizing the surface area heatmap
const TAN_ONE_RADIAN: f32 = 0.017_453_3;

impl<const UNROLL: usize, const VECTOR_WIDTH: usize> Accumulate<UnrollVector<UNROLL, VECTOR_WIDTH>>
    for VectorLos<VECTOR_WIDTH>
where
    [(); UNROLL * VECTOR_WIDTH]:,
    Self: VectorGreater<VECTOR_WIDTH>,
{
    #[inline]
    #[expect(clippy::allow_attributes, reason = "conditional attributes")]
    #[allow(
        unused,
        unused_variables,
        reason = "conditional compilation causes dead parameters"
    )]
    fn accumulate(
        mut init: UnrollVector<UNROLL, VECTOR_WIDTH>,
        angles: &[f32],
        prefix: &[f32],
        distances: &[f32],
        bitmap: &mut Vec<bool>,
    ) -> UnrollVector<UNROLL, VECTOR_WIDTH> {
        debug_assert!(
            angles.len().is_multiple_of(VECTOR_WIDTH),
            "angles with len {} should be multiple of {}",
            angles.len(),
            VECTOR_WIDTH,
        );
        debug_assert!(
            prefix.len().is_multiple_of(VECTOR_WIDTH),
            "prefix unroll should be multiple of width"
        );
        debug_assert!(
            distances.len().is_multiple_of(VECTOR_WIDTH),
            "distance unroll should be multiple of width"
        );
        debug_assert!(
            angles.len() <= UNROLL * VECTOR_WIDTH,
            "angles must be less than unroll size"
        );

        let (vector_sum, _) = init.heatmap.as_chunks_mut::<{ VECTOR_WIDTH }>();
        let (vector_longest, _) = init.longest.as_chunks_mut::<{ VECTOR_WIDTH }>();

        let (vector_angles, _) = angles.as_chunks::<{ VECTOR_WIDTH }>();
        let (vector_prefix, _) = prefix.as_chunks::<{ VECTOR_WIDTH }>();
        let (vector_dists, _) = distances.as_chunks::<{ VECTOR_WIDTH }>();

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

                let acc = Simd::from(*sum_arr) + (dist * Simd::splat(TAN_ONE_RADIAN));

                acc.copy_to_slice(sum_arr);
            },
        );

        init
    }
}
