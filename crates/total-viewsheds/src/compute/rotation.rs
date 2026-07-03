//! Rotate the DEM, but just the so-called "chocolate bar" region.

use itertools::izip;
use std::rc::Rc;

/// Rotate lines of elevation data.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    reason = "so long as max_los^2 < 2^24, the following `as` conversions are entirely safe"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "rotation should not be out of bounds"
)]
#[expect(clippy::expect_used, reason = "invriants broken if options not none")]
pub(crate) fn lines(
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

    let rotator = super::rotator::Rotator::new(angle, width);

    let mut indexes = Rc::new(vec![0i64; 2 * max_los]);
    let mut elevations = Rc::new(vec![0i16; 2 * max_los]);

    ((max_los as isize)..(max_los as isize) * 2).map(move |y| {
        let mut_indexes = Rc::get_mut(&mut indexes)
            .expect("invariant broken: the caller hasn't droped the previous index buffer cannot start next iteration of rotation");
        let mut_elevations = Rc::get_mut(&mut elevations)
            .expect("invariant broken: the caller hasn't droped the previous elevation buffer cannot start next iteration of rotation");

        izip!(
            (max_los as isize)..width,
            mut_indexes.iter_mut(),
            mut_elevations.iter_mut()
        )
        .for_each(|(x, index, elevation)| {
            let (x_rotated, y_rotated) = rotator.rotate(x, y);
            let new_idx = y_rotated * width + x_rotated;
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

    // skew rotation works off of 0 being straight down rather than
    // straight right as previoius rotation methods
    #[gtest]
    fn rotate_by_0() {
        #[rustfmt::skip]
        let expected = [
            14, 15, 16, 17,
            20, 21, 22, 23
        ];
        let rotations = lines(&DEM, 2, 0.0)
            .flat_map(|(elevs, _)| (*elevs).clone())
            .collect::<Vec<_>>();

        expect_eq!(&rotations, &expected);
    }

    #[gtest]
    fn rotate_by_45() {
        #[rustfmt::skip]
        let expected = [
            20, 14, 10, 4,
            21, 15, 16, 11
        ];
        let rotations = lines(&DEM, 2, 45.0)
            .flat_map(|(elevs, _)| (*elevs).clone())
            .collect::<Vec<_>>();

        expect_eq!(&rotations, &expected);
    }

    #[gtest]
    fn rotate_by_90() {
        #[rustfmt::skip]
        let expected = [
            20, 14, 8, 2,
            21, 15, 9, 3
        ];
        let rotations = lines(&DEM, 2, 90.0)
            .flat_map(|(elevs, _)| (*elevs).clone())
            .collect::<Vec<_>>();

        expect_eq!(&rotations, &expected);
    }
}
