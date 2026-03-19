use crate::cpu::los::LineOfSight as _;
use crate::cpu::storage::Storage;
use crate::cpu::unrolled_los::UnrolledVectorLos;
use crate::cpu::vector_intrinsics::DEFAULT_VECTOR_LENGTH;
use itertools::izip;

/// The data output by a single angle.
pub struct OutputData {
    /// The visibile surface area.
    pub surfaces: Vec<f32>,
    /// The longest lines of sight.
    pub longest: Vec<f32>,
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "so long as max_los < 2^24, the following as conversions are entirely safe"
)]
#[expect(
    clippy::integer_division,
    reason = "i32 is constructed from (i32, i32) converting back should succeed"
)]
/// `dem_to_pov` turns the `dem_id` to the `pov_id` so that the result can be stored in a heatmap
fn dem_to_pov(dem_id: i32, width: usize, max_los: usize) -> i32 {
    let dem_x = (dem_id / width as i32) - max_los as i32;
    let dem_y = (dem_id % width as i32) - max_los as i32;

    let radius = (max_los - 1) as f32 / 2.0;
    let circ_x = dem_x as f32 - radius;
    let circ_y = dem_y as f32 - radius;

    let dist = circ_x.hypot(circ_y);
    if dist < radius {
        dem_x * (max_los as i32) + dem_y
    } else {
        -1
    }
}

/// `DEFAULT_UNROLL` is the default loop unrolling constant, which is based
/// off of the default vector length. 8-way unrolling for both the 4 and 8 wide
/// vectors, and 10-way unrolling for the 16-wide vector as it is optimal for Turins
const DEFAULT_UNROLL: usize = const {
    match DEFAULT_VECTOR_LENGTH {
        4 | 8 | 16 => 10,
        #[expect(
            clippy::unreachable,
            reason = "no one should be setting any other constants"
        )]
        _ => unreachable!(),
    }
};

/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight is `max_los`
pub fn kernel(
    storage: &Storage,
    elevation_map: &[i16],
    max_los: usize,
    angle: f32,
    refraction: f32,
    scale: f32,
    observer_height: f32,
) -> OutputData {
    let mut surfaces = vec![0.0f32; max_los * max_los];
    let mut longest = vec![0.0f32; max_los * max_los];

    let (indexes, rotated_elevations) =
        super::rotation::generate_rotation(elevation_map, angle, max_los);

    assert_eq!(
        rotated_elevations.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    let width = 2 * max_los;

    let mut vs =
        UnrolledVectorLos::<DEFAULT_UNROLL, DEFAULT_VECTOR_LENGTH>::new(max_los, refraction, scale);
    for (line, line_indexes) in izip!(
        rotated_elevations.chunks_exact(width),
        indexes.chunks_exact(width),
    ) {
        for (pov, (&pov_height, &result_dem_id)) in
            izip!(line.iter().take(max_los), line_indexes.iter().take(max_los)).enumerate()
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

            let neighbor = pov + 1;

            #[expect(
                clippy::indexing_slicing,
                reason = "if slicing is out of bounds, it should panic"
            )]
            let (point_surface, point_longest, point_visibility) = vs.line_of_sight(
                f32::from(pov_height) + observer_height,
                &line[neighbor..neighbor + max_los],
            );

            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "max_los^2 < 2^31"
            )]
            {
                // safety: result_tvs_id is guaranteed to be within [0..max_los^2)
                unsafe {
                    *surfaces.get_unchecked_mut(result_tvs_id as usize) = point_surface;
                };
                // safety: result_tvs_id is guaranteed to be within [0..max_los^2)
                unsafe {
                    *longest.get_unchecked_mut(result_tvs_id as usize) = point_longest;
                };
                if cfg!(any(test, feature = "ring_data")) {
                    storage.store_bitmap(result_tvs_id as u32, angle as u16, &point_visibility);
                }
            }
        }
    }

    OutputData { surfaces, longest }
}

