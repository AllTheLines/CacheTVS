/// `generate_rotation` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "so long as max_los^2 < 2^24, the following `as` conversions are entirely safe"
)]
pub fn generate_rotation(elevs: &[i16], angle: f64, max_los: usize) -> (Vec<i32>, Vec<i16>) {
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

#[cfg(test)]
mod test {
    use super::*;
    use googletest::prelude::*;

    #[rustfmt::skip]
    const DEM: [i16; 36] = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,11,
        12,13,14,15,16,17,
        18,19,20,21,22,23,
        24,25,26,27,28,29,
        30,31,32,33,34,35,
    ];

    #[gtest]
    fn rotate_by_0() {
        #[rustfmt::skip]
        let expected = [
            14, 15, 16, 17,
            20, 21, 22, 23
        ];
        let (rotations, _) = generate_rotation(&DEM, 0.0, 2);
        expect_eq!(&rotations, &expected);
    }

    #[gtest]
    fn rotate_by_45() {
        #[rustfmt::skip]
        let expected = [
            20, 14, 15, 10,
            // TODO@ryan: `26` is outside the TVS
            26, 21, 16, 16
        ];
        let (rotations, _) = generate_rotation(&DEM, 45.0, 2);
        expect_eq!(&rotations, &expected);
    }

    #[gtest]
    fn rotate_by_90() {
        #[rustfmt::skip]
        let expected = [
            // TODO@ryan: `26` is outside the TVS
            26, 20, 14, 8,
            // TODO@ryan: `27` is outside the TVS
            27, 21, 15, 9
        ];
        let (rotations, _) = generate_rotation(&DEM, 90.0, 2);
        expect_eq!(&rotations, &expected);
    }
}

