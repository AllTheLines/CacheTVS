//! Rotate the DEM, but just the so-called "chocolate bar" region.

use itertools::izip;
use kernel::rotation::ANGLE_SHIFT;
use std::rc::Rc;

/// `lines` generates a rotation "map" for a given elevation list
/// Adapted from [this stack overflow answer](https://stackoverflow.com/a/71901621)
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "so long as max_los^2 < 2^24, the following `as` conversions are entirely safe"
)]
#[expect(clippy::indexing_slicing, reason="rotation should not be out of bounds")]
pub fn lines(
    elevs: &[i16],
    max_los: usize,
    angle: f64,
) -> impl Iterator<Item = (Rc<Vec<i16>>, Rc<Vec<i64>>)> {
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

    let (sin, cos) = (
        f64::sin((angle + f64::from(ANGLE_SHIFT)).to_radians()),
        f64::cos((angle + f64::from(ANGLE_SHIFT)).to_radians()),
    );

    let (x_center, y_center) = ((width - 1) as f64 / 2.0, (width - 1) as f64 / 2.0f64);

    let mut indexes = Rc::new(vec![0i64; 2 * max_los]);
    let mut elevations = Rc::new(vec![0i16; 2 * max_los]);

    ((max_los as isize)..(max_los as isize) * 2).map(move |x| {
        let x_sin = (x as f64 - x_center) * sin;
        let x_cos = (x as f64 - x_center) * cos;

        let mut_indexes = Rc::get_mut(&mut indexes).unwrap();
        let mut_elevations = Rc::get_mut(&mut elevations).unwrap();

        izip!(
            (max_los as isize)..width,
            mut_indexes.iter_mut(),
            mut_elevations.iter_mut()
        )
        .for_each(|(y, index, elevation)| {
            let y_sin = (y as f64 - y_center) * sin;
            let y_cos = (y as f64 - y_center) * cos;

            let x_rot = (x_cos - y_sin + y_center).round() as isize;
            let y_rot = (y_cos + x_sin + x_center).round() as isize;

            let new_idx = x_rot.clamp(0, width - 1) * width + y_rot.clamp(0, width - 1);
            *index = new_idx as i64;
            *elevation = elevs[new_idx.cast_unsigned()];
        });

        fill_line_elevations(mut_elevations);

        (Rc::clone(&elevations), Rc::clone(&indexes))
    })
}

/// `fill_in_elevations` will fill in "blank" elevations from NASA data with the last seen elevation
/// in the line of sight
fn fill_line_elevations(line: &mut [i16]) {
    let mut last_seen: i16 = 0;
    for elevation in line.iter_mut() {
        match *elevation {
            i16::MIN => {
                *elevation = last_seen;
            }
            _ => {
                last_seen = *elevation;
            }
        }
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use googletest::prelude::*;
//
//     #[rustfmt::skip]
//     const DEM: [i16; 36] = [
//         0, 1, 2, 3, 4, 5,
//         6, 7, 8, 9, 10,11,
//         12,13,14,15,16,17,
//         18,19,20,21,22,23,
//         24,25,26,27,28,29,
//         30,31,32,33,34,35,
//     ];
//
//     #[gtest]
//     fn rotate_by_0() {
//         #[rustfmt::skip]
//         let expected = [
//             14, 15, 16, 17,
//             20, 21, 22, 23
//         ];
//         let (rotations, _) = lines(&DEM, 0.0, 2).collect();
//         expect_eq!(&rotations, &expected);
//     }
//
//     #[gtest]
//     fn rotate_by_45() {
//         #[rustfmt::skip]
//         let expected = [
//             14, 15, 9, 4,
//             20, 21, 16, 11
//         ];
//         let (rotations, _) = generate_rotation(&DEM, 45.0, 2);
//         expect_eq!(&rotations, &expected);
//     }
//
//     #[gtest]
//     fn rotate_by_90() {
//         #[rustfmt::skip]
//         let expected = [
//             20, 14, 8, 2,
//             21, 15, 9, 3
//         ];
//         let (rotations, _) = generate_rotation(&DEM, 90.0, 2);
//         expect_eq!(&rotations, &expected);
//     }
// }
