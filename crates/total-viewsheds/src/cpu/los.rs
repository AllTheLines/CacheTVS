use itertools::izip;

/// `LineOfSight` abstracts the implementation of line of sight calculations to
/// any "carry through" that can be materialized into a (f32, f32).
pub trait LineOfSight<Output: Into<(f32, f32)>> {
    /// `line_of_sight` calculates a line of sight for the given `pov_height`
    /// and outputs a triple of the surface area, longest line of sight in meters
    /// and a vector of bools of which
    fn line_of_sight<LOS>(&mut self, pov_height: f32, line: &[i16]) -> (f32, f32, Vec<bool>)
    where
        LOS: Angle + PrefixMax + Accumulate<Output>;
}

/// `Angle` abstracts the angle calculation between a pov and all the elevation data within
/// its band of sight
pub trait Angle {
    /// `calculate_angles` calculates the angle from the `pov_height` to a given elevation
    fn calculate_angles(
        pov_height: f32,
        elevations: &[i16],
        distances: &[f32],
        adjustments: &[f32],
        angles_out: &mut [f32],
    );
}

/// `Accumulate` accumulates the surface area visible and longest line of sight
/// in a pair of (f32, f32). `Accumulate` doesn't care about the implementation details
/// of accumulation so long as the `Output` type can be materialized to (f32, f32)
pub trait Accumulate<Output: Into<(f32, f32)>> {
    /// `accumulate` accumulates the surface area using the distances by comparing
    /// whether a point at a distance is visible (angle > prefix)
    /// If `output_sector` is true, it should output a bitmap of which distances are visible
    /// at their respective locations
    fn accumulate(
        init: Output,
        angles: &[f32],
        prefix: &[f32],
        distances: &[f32],
        bitmap: &mut Vec<bool>,
    ) -> Output;
}

/// `PrefixMax` calculates the prefix maximum of the given angles
pub trait PrefixMax {
    /// `prefix_max` calculates the prefix max of the
    fn prefix_max(highest: f32, angles_in: &[f32], angles_out: &mut [f32]);
}

/// `EARTH_RADIUS_SQUARED` is the radius of the earth in meters
const EARTH_RADIUS_SQUARED: f32 = 12_742_000.0;

/// `generate_distances` generates the distance from
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "max_los is < 2^24"
)]
fn generate_distances(max_los: usize, refraction: f32) -> (Vec<f32>, Vec<f32>) {
    (1..=max_los)
        .map(|step| {
            let distance = (step * 100) as f32;
            let adjustment = (distance * distance * refraction) / EARTH_RADIUS_SQUARED;

            (distance, adjustment)
        })
        .unzip()
}

/// Unroll holds an unrolled heatmap and unrolled longest line of sight calculation
/// Since in Line of Sight-land max/addition are commutative, then Unroll will be materialized
/// into (f32, f32)
pub struct Unroll<const UNROLL: usize> {
    /// `heatmap` contains the summation of visible surface areas which will be reduced to a single
    /// surface area at the end
    pub heatmap: [f32; UNROLL],
    /// `longest` contains many long lines of sight which will be reduced to a single
    /// line of sight at the end
    pub longest: [f32; UNROLL],
}

/// `UnrolledLOS` implements an Unrolled `LineOfSight` calculation
pub struct UnrolledLOS<const UNROLL: usize>
where
    [(); UNROLL + 1]:,
{
    /// `distances` holds `max_los` distances
    distances: Vec<f32>,
    /// `adjustments` holds `max_los` earth curvature adjustments
    adjustments: Vec<f32>,
}

impl<const SIZE: usize> From<Unroll<SIZE>> for (f32, f32) {
    default fn from(val: Unroll<SIZE>) -> Self {
        let heatmap = val.heatmap.iter().sum();
        let longest = val.longest.iter().fold(0.0f32, |acc, &elem| acc.max(elem));
        (heatmap, longest)
    }
}

impl<const UNROLL: usize> UnrolledLOS<UNROLL>
where
    [(); UNROLL + 1]:,
{
    /// `new` initializes a new `UnrolledLOS`, and precalculates all the distances
    /// and earth curvature adjustments
    pub fn new(max_los: usize, refraction: f32) -> Self {
        let (distances, adjustments) = generate_distances(max_los, refraction);

        Self {
            distances,
            adjustments,
        }
    }
}

impl<const UNROLL: usize> LineOfSight<Unroll<UNROLL>> for UnrolledLOS<UNROLL>
where
    [(); UNROLL + 1]:,
{
    #[expect(
        clippy::indexing_slicing,
        reason = "all indexing and slices are guaranteed by construction of a UnrolledLOS"
    )]
    fn line_of_sight<LOS>(&mut self, pov_height: f32, line: &[i16]) -> (f32, f32, Vec<bool>)
    where
        LOS: Angle + PrefixMax + Accumulate<Unroll<UNROLL>>,
    {
        let mut angles = [0.0f32; UNROLL + 1];
        let mut prefix_max = [0.0f32; UNROLL];

        prefix_max[UNROLL - 1] = -2000.0;
        angles[0] = -2000.0;

        let mut output: Vec<bool> = vec![];

        let (chunked_line, rest_line) = line.as_chunks::<{ UNROLL }>();

        let (chunked_distances, rest_distances) = self.distances.as_chunks::<{ UNROLL }>();

        let (chunked_adjustments, rest_adjustments) = self.adjustments.as_chunks::<{ UNROLL }>();

        let los = izip!(chunked_line, chunked_distances, chunked_adjustments).fold(
            Unroll::<UNROLL> {
                longest: [0.0; UNROLL],
                heatmap: [0.0; UNROLL],
            },
            |acc, (unroll_line, distances, adjusts)| {
                LOS::calculate_angles(
                    pov_height,
                    unroll_line,
                    distances,
                    adjusts,
                    &mut angles[1..],
                );

                LOS::prefix_max(prefix_max[UNROLL - 1], &angles[..UNROLL], &mut prefix_max);

                let new_acc =
                    LOS::accumulate(acc, &angles[1..], &prefix_max, distances, &mut output);

                angles[0] = prefix_max[UNROLL - 1];
                new_acc
            },
        );

        LOS::calculate_angles(
            pov_height,
            rest_line,
            rest_distances,
            rest_adjustments,
            &mut angles[1..],
        );

        LOS::prefix_max(prefix_max[UNROLL - 1], &angles[..UNROLL], &mut prefix_max);

        let new_acc = LOS::accumulate(los, &angles[1..], &prefix_max, rest_distances, &mut output);

        let (heatmap, longest) = new_acc.into();
        (heatmap, longest, output)
    }
}
