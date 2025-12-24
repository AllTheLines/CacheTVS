use crate::cpu::los::{LineOfSight as _, UnrolledLOS};
use crate::cpu::vector::{VectorLos, DEFAULT_VECTOR_LENGTH};
use itertools::izip;

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

    dem_x * (max_los as i32) + dem_y
}

/// `kernel` will calculate the longest line of sight heatmap for a given angle and elevation map
/// assuming that the maximum line of sight mis `max_los`
#[expect(
    clippy::inline_always,
    reason = "I am become Death, destroyer of compilers"
)] // the real reason is that I need output_sector_data to be constant propagated
#[inline(always)]
pub fn kernel(
    elevation_map: &[i16],
    max_los: usize,
    angle: f32,
    output_sector_data: bool,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<bool>>) {
    let mut heatmap = vec![0.0f32; max_los * max_los];
    let mut longest = vec![0.0f32; max_los * max_los];

    let mut sector_data: Vec<Vec<bool>> = vec![
        vec![];
        if output_sector_data {
            max_los * max_los
        } else {
            0
        }
    ];

    let (indexes, rotated_elevations) = generate_rotation(elevation_map, f64::from(angle), max_los);

    assert_eq!(
        rotated_elevations.len(),
        2 * max_los * max_los,
        "elevations should be 2 * max_los wide, and max_los tall"
    );

    let width = 2 * max_los;

    let mut vs = UnrolledLOS::<64>::new(max_los);
    for (line, line_indexes) in izip!(
        rotated_elevations.chunks_exact(width),
        indexes.chunks_exact(width),
    ) {
        for (pov, (&pov_height, &result_dem_id)) in
            izip!(line.iter().take(max_los), line_indexes.iter().take(max_los),).enumerate()
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
            let (pixel, long, sector) = vs.line_of_sight::<VectorLos<{ DEFAULT_VECTOR_LENGTH }>>(
                f32::from(pov_height),
                &line[neighbor..neighbor + max_los],
                output_sector_data,
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
                    *heatmap.get_unchecked_mut(result_tvs_id as usize) = pixel;
                };
                // safety: result_tvs_id is guaranteed to be within [0..max_los^2)
                unsafe {
                    *longest.get_unchecked_mut(result_tvs_id as usize) = long;
                };

                if output_sector_data {
                    sector_data[result_tvs_id as usize] = sector;
                }
            }
        }
    }

    (heatmap, longest, sector_data)
}
