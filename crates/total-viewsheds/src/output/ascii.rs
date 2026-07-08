//! Convert a viewshed to pure text. For testing.

#[cfg(test)]
pub(crate) fn make_viewshed(
    elevations: &[i16],
    viewshed_pov: geo::Coord,
    config: crate::run::Config,
) -> Vec<String> {
    let mut dem = crate::run::test::make_dem(elevations);
    let dem_half_width = f64::from(dem.width - 1) / 2.0f64;
    let viewshed_pov_metric = geo::Coord {
        x: viewshed_pov.x - dem_half_width,
        y: -(viewshed_pov.y - dem_half_width),
    };
    let coord_lonlat = tvs_lib::projector::Convert::new(dem.centre)
        .to_degrees(viewshed_pov_metric)
        .unwrap();

    crate::run::test::compute(&mut dem, config.clone());
    let (pov_coord, mut viewshed) =
        super::viewshed::Viewshed::reconstruct(config.viewsheds_db_path, coord_lonlat).unwrap();
    let viewsheder = crate::output::viewshed::Viewshed {
        dem: &dem,
        pov_coord,
    };

    for polygons in viewshed.iter_mut() {
        polygons.exterior_mut(|line| {
            for coordinate in line.coords_mut() {
                *coordinate = viewsheder
                    .convert_viewshed_coord_to_dem_coord(crate::output::viewshed::Coordinate(
                        *coordinate,
                    ))
                    .unwrap();
            }
        });

        polygons.interiors_mut(|interior| {
            for interiror_line in interior {
                for coordinate in interiror_line.coords_mut() {
                    *coordinate = viewsheder
                        .convert_viewshed_coord_to_dem_coord(crate::output::viewshed::Coordinate(
                            *coordinate,
                        ))
                        .unwrap();
                }
            }
        });
    }

    tvs_lib::ascii::rasterise_multi_polygon_geo(&viewshed, dem.width)
}
