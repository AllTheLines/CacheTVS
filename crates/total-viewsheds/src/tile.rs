//! A `Tile` is the raw data that a user needs to provide. It needs to:
//!   * be square
//!   * have a square resolution for the pixel sizes
//!   * have a single raster band of elevation data
//!   * have a projection that can derive a lat/lon centre

use color_eyre::eyre::{ContextCompat as _, Result};

/// The basic raw data needed to compute total viewsheds.
pub struct Tile {
    /// The width of the tile.
    pub width: u32,
    /// The size of single elevation sample in metres.
    pub scale: f32,
    /// All the elevation data.
    pub data: Vec<i16>,
    /// The lat/lon coordinates of the centre of the tile.
    pub centre: crate::projection::LonLatCoord,
}

impl Tile {
    /// Load a tile.
    pub fn load(config: &crate::config::Compute) -> Result<Self> {
        if !config.input.exists() {
            color_eyre::eyre::bail!("Input file not found: {}", config.input.display());
        }

        let dataset = gdal::Dataset::open(&config.input)?;
        let (width_usize, height_usize) = dataset.raster_size();
        if width_usize != height_usize {
            color_eyre::eyre::bail!("Tile is not square: {width_usize}x{height_usize}");
        }

        let geo = dataset.geo_transform()?;
        let pixel_width = geo[1].abs();
        let pixel_height = geo[5].abs();

        if format!("{pixel_width:.10}") != format!("{pixel_height:.10}") {
            color_eyre::eyre::bail!(
                "Tile pixel resolution is not square: {pixel_width}x{pixel_height}"
            );
        }

        let centre = Self::get_centre_by_projection(&dataset)?;

        let lon = centre.0.x;
        let lat = centre.0.y;
        tracing::info!("Using tile centre: {lon},{lat}");

        Ok(Self {
            width: u32::try_from(width_usize)?,
            #[expect(
                clippy::cast_possible_truncation,
                clippy::as_conversions,
                reason = "It's extremely unlikely the scale will ever hit the limits of floats"
            )]
            scale: pixel_width as f32,
            data: Self::get_elevations(&dataset)?,
            centre,
        })
    }

    /// Convert the raw data elevations to our own internal datatype.
    fn get_elevations(dataset: &gdal::Dataset) -> Result<Vec<i16>> {
        let band = dataset.rasterband(1)?;
        let size = dataset.raster_size();

        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "We only handle the types we know we can process"
        )]
        match band.band_type() {
            gdal::raster::GdalDataType::Int16 => Ok(band
                .read_as::<i16>((0, 0), size, size, None)?
                .data()
                .to_vec()),

            gdal::raster::GdalDataType::UInt16 => {
                let raw_data = band.read_as::<u16>((0, 0), size, size, None)?;

                tracing::warn!("Offsetting `u16` tile data into `i16`...");

                Ok(raw_data
                    .data()
                    .iter()
                    .map(|elevation| {
                        let offset = i32::from(*elevation) - i32::from(i16::MAX);
                        i16::try_from(offset)
                    })
                    .collect::<std::result::Result<Vec<i16>, _>>()?)
            }

            _ => {
                let data_type = band.band_type();
                color_eyre::eyre::bail!("{data_type} not supported (should be easy to add)");
            }
        }
    }

    /// Get the metric centre of the tile simply by dividing the extent in half.
    fn get_metric_centre(dataset: &gdal::Dataset) -> Result<geo::Coord> {
        let (width, height) = dataset.raster_size();

        #[expect(
            clippy::cast_precision_loss,
            clippy::as_conversions,
            reason = "We're not targetting machines with less than 64 bits"
        )]
        let (centre_x, centre_y) = (width as f64 / 2.0f64, height as f64 / 2.0f64);

        let raster = dataset.geo_transform()?;
        let x_top_left = raster[0];
        let pixel_width = raster[1].abs();
        let y_top_left = raster[3];
        let pixel_height = raster[5].abs();
        #[expect(clippy::suboptimal_flops, reason = "Not relevant")]
        let (x_world, y_world) = (
            x_top_left + (centre_x * pixel_width),
            y_top_left - (centre_y * pixel_height),
        );

        Ok(geo::coord! {
            x: x_world,
            y: y_world
        })
    }

    /// Get the lat/lon centre of the tile by querying the projection's definition.
    fn get_centre_by_projection(dataset: &gdal::Dataset) -> Result<crate::projection::LonLatCoord> {
        const ALLOWED_DEVIATION: f64 = 0.001; // 1 milimetre
        let geometric_centre = Self::get_metric_centre(dataset)?;
        if geometric_centre.x.abs() > ALLOWED_DEVIATION
            || geometric_centre.y.abs() > ALLOWED_DEVIATION
        {
            color_eyre::eyre::bail!(
                "Tile centre ({},{}) must be within {ALLOWED_DEVIATION} of 0,0.",
                geometric_centre.x,
                geometric_centre.y
            );
        }

        let projection = &dataset.spatial_ref()?.to_proj4()?;

        let lat_0: f64 = projection
            .split_whitespace()
            .find(|part| part.starts_with("+lat_0="))
            .and_then(|part| part.split('=').nth(1))
            .and_then(|value| value.parse::<f64>().ok())
            .context("Couldn't find `lat_0` in projection defintion")?;

        let lon_0: f64 = projection
            .split_whitespace()
            .find(|part| part.starts_with("+lon_0="))
            .and_then(|part| part.split('=').nth(1))
            .and_then(|value| value.parse::<f64>().ok())
            .context("Couldn't find `lon_0` in projection defintion")?;

        Ok(crate::projection::LonLatCoord(
            geo::coord! { x: lon_0, y: lat_0 },
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn get_centre_by_projection() {
        let config = crate::config::Compute {
            input: "../../benchmarks/samples/aeqd_10x10.tiff".into(),
            ..Default::default()
        };
        let tile = Tile::load(&config).unwrap();
        assert_eq!(
            tile.centre,
            crate::projection::LonLatCoord(geo::Coord {
                x: -0.1278,
                y: 51.5074
            })
        );
    }

    #[test]
    fn get_metric_centre() {
        let dataset = gdal::Dataset::open("../../benchmarks/samples/aeqd_10x10.tiff").unwrap();
        let centre = Tile::get_metric_centre(&dataset).unwrap();
        assert_eq!(centre, geo::Coord { x: 0.0, y: 0.0 });
    }

    #[test]
    fn bad_centre() {
        let dataset = gdal::Dataset::open("../../benchmarks/samples/aeqd_bad_centre.tiff").unwrap();
        let error = Tile::get_centre_by_projection(&dataset)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Tile centre"), "Wrong error: '{error}'");
    }
}
