/// `LineOfSight` abstracts the implementation of line of sight calculations to
/// any "carry through" that can be materialized into a (f32, f32).
pub trait LineOfSight<Output: Into<(f32, f32)>, LOS: Angle + PrefixMax + Accumulate<Output>> {
    /// `line_of_sight` calculates a line of sight for the given `pov_height`
    /// and outputs a triple of the surface area, longest line of sight in meters
    /// and a vector of bools of which
    fn line_of_sight(&mut self, pov_height: f32, line: &[i16]) -> (f32, f32, Vec<bool>);
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
