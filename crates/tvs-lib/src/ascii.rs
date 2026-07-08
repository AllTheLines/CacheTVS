//! Output viewsheds as ASCII. Most likely useful for tests.

#![expect(
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "This code is mostly for tests"
)]

use geo::CoordsIter as _;

/// # Panics
#[inline]
#[must_use]
pub fn rasterise_multi_polygon_geo(multi_polygon: &geo::MultiPolygon, width: u32) -> Vec<String> {
    let scale = std::env::var("ASCII_TEST_SIZE")
        .unwrap_or_else(|_| "2".to_owned())
        .parse::<u32>()
        .expect("ASCII_TEST_SIZE env not a valid integer");
    let scale_f64 = f64::from(scale);
    let raster_width = usize::try_from(width * scale).unwrap();

    let mut raster: Vec<Vec<bool>> = Vec::new();
    for _ in 0..width * scale {
        let raster_line = vec![false; raster_width];
        raster.push(raster_line);
    }

    for polygon in multi_polygon.iter() {
        let mut maybe_exterior_start = None;
        for coordinate in polygon.exterior_coords_iter() {
            if maybe_exterior_start.is_none() {
                maybe_exterior_start = Some(coordinate);
                continue;
            }

            let from = maybe_exterior_start.unwrap();
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                reason = "Gotta rasterise"
            )]
            let rasteriser = crate::bresenham::Bresenham::new(
                super::bresenham::RasterCoord {
                    x: (from.x * scale_f64).round() as i32,
                    y: (from.y * scale_f64).round() as i32,
                },
                super::bresenham::RasterCoord {
                    x: (coordinate.x * scale_f64).round() as i32,
                    y: (coordinate.y * scale_f64).round() as i32,
                },
            );
            for coord in rasteriser {
                if coord.x >= 0i32 && coord.y >= 0i32 {
                    let x = usize::try_from(coord.x).unwrap();
                    let y = usize::try_from(coord.y).unwrap();
                    raster[y][x] = true;
                }
            }
            maybe_exterior_start = Some(coordinate);
        }

        for interiors in polygon.interiors() {
            let mut maybe_interior_start = None;
            for coordinate in interiors {
                if maybe_interior_start.is_none() {
                    maybe_interior_start = Some(*coordinate);
                    continue;
                }

                let from = maybe_interior_start.unwrap();
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Gotta rasterise"
                )]
                let rasteriser = crate::bresenham::Bresenham::new(
                    super::bresenham::RasterCoord {
                        x: (from.x * scale_f64).round() as i32,
                        y: (from.y * scale_f64).round() as i32,
                    },
                    super::bresenham::RasterCoord {
                        x: (coordinate.x * scale_f64).round() as i32,
                        y: (coordinate.y * scale_f64).round() as i32,
                    },
                );
                for coord in rasteriser {
                    if coord.x >= 0i32 && coord.y >= 0i32 {
                        let x = usize::try_from(coord.x).unwrap();
                        let y = usize::try_from(coord.y).unwrap();
                        raster[y][x] = true;
                    }
                }
                maybe_interior_start = Some(*coordinate);
            }
        }
    }

    let mut ascii = Vec::new();
    let row_count = raster_width.div_euclid(2);
    #[expect(clippy::needless_range_loop, reason = "clippy is wrong")]
    for row in 0..row_count {
        let mut viewshed_line = String::new();
        for x in 0..raster_width {
            let y = row * 2;
            let upper = raster[y][x];
            let lower = raster[y + 1][x];
            let character = match (upper, lower) {
                (true, true) => ' ',
                (false, true) => '▀',
                (true, false) => '▄',
                (false, false) => '█',
            };
            viewshed_line.push(character);
        }
        ascii.push(viewshed_line);
    }

    ascii
}

/// # Panics
#[inline]
#[expect(clippy::print_stderr, reason = "This is for tests")]
pub fn assert_rasterised(actual: &[String], expected: &[&str]) {
    if actual != expected {
        eprintln!("Actual:");
        eprint!("{}", actual.join("\n"));
        eprintln!();
        eprintln!();
        eprintln!("Expected:");
        eprint!("{}", expected.join("\n"));
        eprintln!();
        panic!("Viewsheds do not match");
    }
}
