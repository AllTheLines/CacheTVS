use crate::cpu::los::{LineOfSight as _, UnrolledLOS};
use crate::cpu::vector::{VectorLos, DEFAULT_VECTOR_LENGTH};
use itertools::izip;

/// The data output by a single angle.
pub struct OutputData {
    /// The visibile surface area.
    pub surfaces: Vec<f32>,
    /// The longest lines of sight.
    pub longest: Vec<f32>,
    /// The raw ring data used to reconstruct viewsheds.
    pub visibility: Vec<Vec<bool>>,
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

    let radius = max_los as i32 / 2i32;
    let circ_x = dem_x - radius;
    let circ_y = dem_y - radius;

    let dist = (circ_x.pow(2) + circ_y.pow(2)).isqrt();
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
        4 => 32,
        8 => 64,
        16 => 160,
        #[expect(
            clippy::unreachable,
            reason = "no one should be setting any other constants"
        )]
        _ => unreachable!(),
    }
};

/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight is `max_los`
#[expect(
    clippy::inline_always,
    reason = "I am become Death, destroyer of compilers"
)] // the real reason is that I need output_sector_data to be constant propagated
#[inline(always)]
pub fn kernel(
    elevation_map: &[i16],
    max_los: usize,
    angle: f32,
    is_output_sector_data: bool,
) -> OutputData {
    let mut surfaces = vec![0.0f32; max_los * max_los];
    let mut longest = vec![0.0f32; max_los * max_los];

    let mut sector_data: Vec<Vec<bool>> = vec![
        vec![];
        if is_output_sector_data {
            max_los * max_los
        } else {
            0
        }
    ];

    let (indexes, rotated_elevations) =
        super::rotation::generate_rotation(elevation_map, angle, max_los);

    assert_eq!(
        rotated_elevations.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    let width = 2 * max_los;

    let mut vs = UnrolledLOS::<DEFAULT_UNROLL>::new(max_los);
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
            let (point_surface, point_longest, point_visibility) =
                vs.line_of_sight::<VectorLos<{ DEFAULT_VECTOR_LENGTH }>>(
                    f32::from(pov_height) + 1.65,
                    &line[neighbor..neighbor + max_los],
                    is_output_sector_data,
                );

            #[expect(
                clippy::as_conversions,
                clippy::cast_sign_loss,
                clippy::indexing_slicing,
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

                if is_output_sector_data {
                    // TODO@ryan:
                    //   This rotation of the `result_tvs_id` is just a hack to get the ring data
                    //   into the right format for rendering. Ideally we would just fill up the ring data
                    //   in the order that each point is processed. Though without skipping any points. The
                    //   sector data is just a snapshot of the already rotated TVS grid. The reason for this
                    //   is mainly fidelity. We don't want to have to both rotate the DEM and then rotate the
                    //   sector data. Just the DEM rotation already has all the data we need to reconstruct
                    //   viewsheds.
                    //
                    //   In short: either keep this hack or better, just fill the sector data as you process
                    //   it, but make sure that any skipped points are also filled with empty bitmaps.
                    {
                        let sector = angle.rem_euclid(f32::from(crate::run::compute::SECTOR_STEPS));
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "max_los is always within u32"
                        )]
                        let rotated_tvs_id = kernel::rotation::Rotator::rotate_index(
                            result_tvs_id as u32,
                            max_los as u32,
                            sector,
                        );

                        if rotated_tvs_id != kernel::rotation::NOOP_DEM_ID {
                            sector_data[rotated_tvs_id] = point_visibility;
                        }
                    }
                }
            }
        }
    }

    OutputData {
        surfaces,
        longest,
        visibility: sector_data,
    }
}
