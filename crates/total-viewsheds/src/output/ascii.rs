//! Output viewsheds as ASCII. Most likely useful for tests.

#![cfg(test)]
#![expect(clippy::indexing_slicing, reason = "This code is mostly for tests")]

pub fn make_viewshed(
    elevations: &[i16],
    viewshed_pov: geo::Coord,
    config: crate::run::compute::Config,
) -> Vec<String> {
    let mut dem = crate::run::compute::test::make_dem(elevations);
    let dem_half_width = f64::from(dem.width - 1) / 2.0f64;
    let viewshed_pov_metric = geo::Coord {
        x: viewshed_pov.x - dem_half_width,
        y: -(viewshed_pov.y - dem_half_width),
    };
    let coord_lonlat = crate::projection::Converter { base: dem.centre }
        .to_degrees(viewshed_pov_metric)
        .unwrap();

    let compute = crate::run::compute::test::compute(&mut dem, config);

    let mut viewshed = crate::output::viewshed::Viewshed::reconstruct(
        &super::ring_data::Source::RAM(crate::output::ring_data::AllData {
            metadata: compute.metadata().unwrap(),
            ring_data: crate::output::ring_data::SectorData::AllSectors(compute.ring_data),
        }),
        coord_lonlat,
    )
    .unwrap();

    let viewsheder = crate::output::viewshed::Viewshed {
        dem: &dem,
        pov_coord: crate::dem::Coordinate(viewshed_pov),
    };

    let scale = 2;
    let scale_f64 = f64::from(scale);
    let raster_width = usize::try_from(dem.width * scale).unwrap();

    let mut raster: Vec<Vec<bool>> = Vec::new();
    for _ in 0..dem.width * scale {
        let raster_line = vec![false; raster_width];
        raster.push(raster_line);
    }

    for point in viewshed.iter_mut() {
        point.exterior_mut(|line| {
            let mut maybe_from = None;
            for coordinate in line.coords_mut() {
                let projected = viewsheder
                    .convert_viewshed_coord_to_dem_coord(crate::output::viewshed::Coordinate(
                        *coordinate,
                    ))
                    .unwrap();
                if maybe_from.is_none() {
                    maybe_from = Some(projected);
                    continue;
                }

                let from = maybe_from.unwrap();
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_possible_truncation,
                    reason = "Gotta rasterise"
                )]
                let rasteriser = crate::output::bresenham::Bresenham::new(
                    super::bresenham::RasterCoord {
                        x: (from.x * scale_f64).round() as i32,
                        y: (from.y * scale_f64).round() as i32,
                    },
                    super::bresenham::RasterCoord {
                        x: (projected.x * scale_f64).round() as i32,
                        y: (projected.y * scale_f64).round() as i32,
                    },
                );
                for coord in rasteriser {
                    if coord.x >= 0i32 && coord.y >= 0i32 {
                        let x = usize::try_from(coord.x).unwrap();
                        let y = usize::try_from(coord.y).unwrap();
                        raster[y][x] = true;
                    }
                }
                maybe_from = Some(projected);
            }
        });
    }

    let mut ascii = Vec::new();
    let row_count = raster_width.div_euclid(2);
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

#[expect(clippy::print_stderr, reason = "This is for tests")]
pub fn assert_viewshed(actual: &[String], expected: &[&str]) {
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
