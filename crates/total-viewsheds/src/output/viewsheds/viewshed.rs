//! A viewshed is a (multi)polygon representing all the visibile terrain visible from a given point.

use color_eyre::eyre::Result;

/// A viewshed-based coordinate is projected to a metric system where the anchor is the viewshed's
/// point of view. The other option would be a metric projection with an anchor in the DEM centre,
/// but metric projections are not globally correct. So reprojecting to the _viewshed's_ centre
/// just gives us that little bit more accuracy, especially for larger DEMs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Coordinate(pub geo::Coord);

/// `Viewshed`
pub(crate) struct Viewshed<'viewshed> {
    /// The DEM used to compute the final data.
    pub dem: &'viewshed tvs_lib::dem::DEM,
    /// Coordinate of the observer for the viewshed we want to reconstruct.
    pub pov_coord: tvs_lib::dem::Coordinate,
}

impl Viewshed<'_> {
    /// Reconstruct a viewshed.
    pub(crate) fn reconstruct(
        db_path: std::path::PathBuf,
        requested_lonlat: tvs_lib::projector::LonLatCoord,
    ) -> Result<(tvs_lib::dem::Coordinate, geo::MultiPolygon)> {
        let db = crate::storage::db::DB::new(db_path)?;
        let metadata = db.load_metadata()?;
        tracing::debug!("Using metadata: {:?}", metadata);

        let dem = tvs_lib::dem::DEM::new(
            metadata.centre,
            metadata.width,
            metadata.scale,
            metadata.max_line_of_sight,
        )?;

        let dem_coord =
            tvs_lib::projector::Convert::lonlat_to_dem_coord(&metadata, requested_lonlat)?;
        let dem_id = dem.dem_coord_to_id(dem_coord);

        let (segments, pov_dem_coord) = match metadata.neighbourhood_size {
            0 => {
                let id = crate::storage::db::ID::DEM(dem_id.into());
                let (segments, _) = db.load_segments(&id)?;
                (segments, dem_coord)
            }
            _ => {
                let neighbourhood_id = crate::pre_process::get_neighbourhood_id(
                    dem_id.into(),
                    dem.width.into(),
                    metadata.neighbourhood_size.into(),
                );
                let id = crate::storage::db::ID::Neighbourhood(neighbourhood_id);
                let (segments, dem_id_with_biggest_viewshed) = db.load_segments(&id)?;
                let x = dem_id_with_biggest_viewshed.rem_euclid(i64::from(dem.width));
                let y = dem_id_with_biggest_viewshed.div_euclid(i64::from(dem.width));
                #[expect(
                    clippy::as_conversions,
                    clippy::cast_precision_loss,
                    reason = "I assume that we never hit the 52 bit mantissa limit"
                )]
                let coordinate = tvs_lib::dem::Coordinate(
                    geo::coord! {
                        x: x as f64,
                        y: y as f64,
                    } * f64::from(dem.scale),
                );
                (segments, coordinate)
            }
        };

        tracing::info!(
            "Reconstructing viewshed for DEM-relative coord: {:?}.",
            pov_dem_coord
        );

        let start = std::time::Instant::now();
        let viewshed = Viewshed {
            dem: &dem,
            pov_coord: pov_dem_coord,
        };

        let pov_id = viewshed.dem.dem_coord_to_id(viewshed.pov_coord);
        if !viewshed.dem.is_point_computable(pov_id) {
            color_eyre::eyre::bail!("Point of view ({:?}) is not calculable", viewshed.pov_coord);
        }

        let multi_polygon = crate::output::viewsheds::joiner::build_viewshed_polygon(
            &segments,
            viewshed.dem.scale,
        )?;
        tracing::info!("Viewshed reconstructed in {:?}.", start.elapsed());

        Ok((pov_dem_coord, multi_polygon))
    }

    #[expect(
        clippy::panic,
        reason = "The closures expect () so I don't think there's any other way?"
    )]
    /// Convert the local metric coordinates of the viewshed to WGS84 lon/lat coordinates.
    fn convert_viewshed_coords_to_lonlat(
        mut viewshed: geo::MultiPolygon,
        viewshed_latlon: tvs_lib::projector::LonLatCoord,
    ) -> geo::MultiPolygon {
        let projector = tvs_lib::projector::Convert::new(viewshed_latlon);

        for point in viewshed.iter_mut() {
            point.exterior_mut(|line| {
                for coordinate in line.coords_mut() {
                    let projected = projector
                        .to_degrees(geo::Coord {
                            x: coordinate.x,
                            y: coordinate.y,
                        })
                        .unwrap_or_else(|_| {
                            panic!(
                                "Couldn't project viewshed coordinate to degrees: {coordinate:?}",
                            )
                        });
                    *coordinate = projected.0;
                }
            });

            point.interiors_mut(|lines| {
                for line in lines {
                    for coordinate in line.coords_mut() {
                        let projected = projector
                            .to_degrees(geo::Coord {
                                x: coordinate.x,
                                y: coordinate.y,
                            })
                            .unwrap_or_else(|_| {
                                panic!(
                                "Couldn't project viewshed coordinate to degrees: {coordinate:?}",
                            )
                            });
                        *coordinate = projected.0;
                    }
                }
            });
        }

        viewshed
    }

    /// Save the viewshed to disk.
    pub(crate) fn save(
        viewshed: geo::MultiPolygon,
        output_directory: &std::path::Path,
        viewshed_latlon: tvs_lib::projector::LonLatCoord,
    ) -> Result<()> {
        let filename = format!("{}-{}.json", viewshed_latlon.0.x, viewshed_latlon.0.y);
        let directory = output_directory.join("viewsheds");
        std::fs::create_dir_all(&directory)?;
        let path = directory.join(filename);

        let viewshed_lonlat = Self::convert_viewshed_coords_to_lonlat(viewshed, viewshed_latlon);
        let json = geojson::GeoJson::from(&viewshed_lonlat).to_string();
        std::fs::write(path, json)?;

        Ok(())
    }
}

#[cfg(test)]
impl Viewshed<'_> {
    /// Convert from the viewshed projection to DEM coordinates.
    pub(crate) fn convert_viewshed_coord_to_dem_coord(
        &self,
        viewshed_coord: Coordinate,
    ) -> Result<geo::Coord> {
        let scale = f64::from(self.dem.scale);
        let origin = tvs_lib::projector::Convert::new(self.dem.centre)
            .to_degrees((self.pov_coord.0.x, self.pov_coord.0.y).into())?;
        let flipped = Coordinate(geo::Coord {
            x: viewshed_coord.0.x,
            y: -viewshed_coord.0.y,
        });
        let projected_coord = tvs_lib::projector::Convert::change_metric_origin(
            origin,
            // The path back to (0,0) is exactly the opposite of the viewshed's point of view.
            -self.pov_coord.0 * scale,
            flipped.0 * scale,
        )?;
        Ok(projected_coord / scale)
    }
}

#[cfg(test)]
mod test {
    use crate::output::ascii::assert_rasterised;

    const SUMMIT_VIEWSHED: [&str; 12] = [
        "████████████████████████",
        "████████████████████████",
        "███████▀▀ ▄▄▄▄▄ ▀▀██████",
        "█████▀ ▄█████████▄ ▀████",
        "████▀ █████████████ ▀███",
        "████ ███████████████ ███",
        "████ ███████████████ ███",
        "████ ▀█████████████▀ ███",
        "█████ ▀███████████▀ ████",
        "██████▄ ▀▀█████▀▀ ▄█████",
        "█████████▄▄▄▄▄▄▄████████",
        "████████████████████████",
    ];

    const PACMAN_VIEWSHED: [&str; 12] = [
        "████████████████████████",
        "████████████████████████",
        "█████▀▀ ▄▄▄▄▄ ▀▀████████",
        "███▀ ▄█████████▄ ▀██████",
        "██▀ █████████████ ▀█████",
        "██ ███████████████ █████",
        "██ ████████▀ ▄▄▄▄▄▄█████",
        "██ ▀█▀▄ ██ ▄████████████",
        "███ ▀▄▄███ █████████████",
        "████▄ ▀▀██ █████████████",
        "███████▄▄▄▄█████████████",
        "████████████████████████",
    ];

    #[test]
    fn viewshed_in_hole() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            crate::run::Config {
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );

        let expected = &[
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████▀ ▄ ▀███████████",
            "████████ ▀█▀ ███████████",
            "█████████▄▄▄████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&viewshed, expected);
    }

    #[test]
    fn viewshed_on_summit_by_coord() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 6.0, y: 6.0 },
            crate::run::Config {
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );
        assert_rasterised(&viewshed, &SUMMIT_VIEWSHED);
    }

    #[test]
    fn viewshed_on_summit_with_high_angle_count() {
        crate::setup_logging().unwrap();
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 6.0 },
            crate::run::Config {
                dem_metadata: crate::run::test::big_dem_metadata(),
                angle_subdivisions: 4,
                ..crate::run::test::default_config(&temp_db)
            },
        );
        assert_rasterised(&viewshed, &PACMAN_VIEWSHED);
    }

    #[test]
    fn viewshed_on_summit_by_neighbourhood() {
        let temp_db_for_surfaces = tempfile::NamedTempFile::new().unwrap();
        let neighbourhood_size = 16;
        let elevations = crate::tests::fixtures::bigger_dem();

        let viewsheds_to_save = {
            let config_for_heatmap_only = crate::run::Config {
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db_for_surfaces)
            };
            let mut dem = crate::run::test::make_dem(&elevations);
            let compute = crate::run::test::compute(&mut dem, config_for_heatmap_only);

            let tiff = crate::tests::fixtures::create_tvs_tiff(compute.total_surfaces).unwrap();
            crate::pre_process::find_biggest_total_surfaces(&tiff, neighbourhood_size).unwrap()
        };

        assert_eq!(viewsheds_to_save[&4], 78);

        let temp_db_for_viewsheds = tempfile::NamedTempFile::new().unwrap();
        let config = crate::run::Config {
            dem_metadata: tvs_lib::metadata::MetaData {
                neighbourhood_size: neighbourhood_size.try_into().unwrap(),
                ..crate::run::test::big_dem_metadata()
            },
            viewsheds_to_save: Some(viewsheds_to_save),
            ..crate::run::test::default_config(&temp_db_for_viewsheds)
        };

        assert_rasterised(
            &crate::output::ascii::make_viewshed(
                &elevations,
                geo::Coord { x: 5.0, y: 5.0 },
                config.clone(),
            ),
            &SUMMIT_VIEWSHED,
        );

        assert_rasterised(
            &crate::output::ascii::make_viewshed(
                &elevations,
                geo::Coord { x: 6.0, y: 6.0 },
                config.clone(),
            ),
            &SUMMIT_VIEWSHED,
        );

        assert_rasterised(
            &crate::output::ascii::make_viewshed(
                &elevations,
                geo::Coord { x: 7.0, y: 7.0 },
                config,
            ),
            &SUMMIT_VIEWSHED,
        );
    }

    #[test]
    fn viewshed_near_summit() {
        crate::setup_logging().unwrap();
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 6.0 },
            crate::run::Config {
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );

        assert_rasterised(&viewshed, &PACMAN_VIEWSHED);
    }

    #[test]
    fn viewshed_in_hole_tall_observer() {
        let temp_db = tempfile::NamedTempFile::new().unwrap();
        let viewshed = crate::output::ascii::make_viewshed(
            &crate::tests::fixtures::bigger_dem(),
            geo::Coord { x: 5.0, y: 5.0 },
            crate::run::Config {
                observer_height: 20.0,
                dem_metadata: crate::run::test::big_dem_metadata(),
                ..crate::run::test::default_config(&temp_db)
            },
        );

        let expected = &[
            "████████████████████████",
            "█████▀▀ ▄▄▄▄▄ ▀▀████████",
            "███▀ ▄█████████▄ ▀██████",
            "██▀ █████████████ ▀█████",
            "██ ███████████████ █████",
            "██ ███████████████ █████",
            "██ ▀█████████████▀ █████",
            "███ ▀███████████▀ ██████",
            "████▄ ▀▀█████▀▀ ▄███████",
            "███████▄▄▄▄▄▄▄██████████",
            "████████████████████████",
            "████████████████████████",
        ];
        assert_rasterised(&viewshed, expected);
    }
}
