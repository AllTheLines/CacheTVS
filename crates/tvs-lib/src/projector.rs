//! Project coordinates between different systems.

use color_eyre::{Result, eyre::ContextCompat as _};

/// The radius of the planet in kilometers.
pub const EARTH_RADIUS: f32 = 6378.137;

/// The diameter of the planet in meters.
pub const EARTH_DIAMETER: f32 = 12_756_274.0;

/// A longtitude/latitude coordinate.
#[expect(clippy::exhaustive_structs, reason = "This should never change")]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Default)]
pub struct LonLatCoord(pub geo::Coord);

impl LonLatCoord {
    /// Parse coordinates from a string.
    ///
    /// # Errors
    /// On parsing errors.
    #[inline]
    pub fn parse(coordinates: &str) -> Result<Self> {
        let mut parts = coordinates.split(',');
        let lon_str = parts.next().context("missing longtitude")?.trim();
        let lat_str = parts.next().context("missing latitude")?.trim();

        Ok(Self(geo::Coord {
            x: lon_str.parse()?,
            y: lat_str.parse()?,
        }))
    }
}

impl rstar::Point for LonLatCoord {
    type Scalar = f64;
    const DIMENSIONS: usize = 2;

    #[inline]
    fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
        Self(geo::coord! {
            x: generator(0),
            y: generator(1),
        })
    }

    #[inline]
    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.0.x,
            1 => self.0.y,
            #[expect(clippy::unreachable, reason = "This is just a 2D coordinate.")]
            _ => unreachable!(),
        }
    }

    #[inline]
    fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
        match index {
            0 => &mut self.0.x,
            1 => &mut self.0.y,
            #[expect(clippy::unreachable, reason = "This is just a 2D coordinate.")]
            _ => unreachable!(),
        }
    }
}

/// Convert between different coordinate system.
#[non_exhaustive]
pub struct Convert {
    /// The lon/lat base coordinates for the AEQD mercator projected coordinates.
    pub base: LonLatCoord,
}

impl Convert {
    /// Instantiate.
    #[inline]
    #[must_use]
    pub const fn new(base: LonLatCoord) -> Self {
        Self { base }
    }

    /// The projection description for lat/lon.
    fn degrees_projection() -> Result<proj4rs::Proj> {
        let string = "+proj=latlong +datum=WGS84 +over";
        Ok(proj4rs::Proj::from_proj_string(string)?)
    }

    /// The projection description for the AEQD metric projection.
    fn meters_projection(&self) -> Result<proj4rs::Proj> {
        let string = format!(
            "+proj=aeqd +lat_0={} +lon_0={} +datum=WGS84 +over",
            self.base.0.y, self.base.0.x
        );
        Ok(proj4rs::Proj::from_proj_string(&string)?)
    }

    #[inline]
    /// Convert from degrees to the AEQD metric projection.
    ///
    /// # Errors
    /// When projection conversion fails.
    pub fn to_meters(&self, source: LonLatCoord) -> Result<geo::Coord> {
        let mut converted = (source.0.x.to_radians(), source.0.y.to_radians(), 0.0f64);
        proj4rs::transform::transform(
            &Self::degrees_projection()?,
            &self.meters_projection()?,
            &mut converted,
        )?;

        Ok(geo::coord! { x: converted.0, y: converted.1 })
    }

    #[inline]
    /// Convert from the AEQD metric projection to degrees.
    ///
    /// # Errors
    /// When projection conversion fails.
    pub fn to_degrees(&self, source: geo::Coord) -> Result<LonLatCoord> {
        let mut converted = (source.x, source.y, 0.0f64);
        proj4rs::transform::transform(
            &self.meters_projection()?,
            &Self::degrees_projection()?,
            &mut converted,
        )?;

        Ok(LonLatCoord(
            geo::coord! { x: converted.0.to_degrees(), y: converted.1.to_degrees() },
        ))
    }

    #[inline]
    #[must_use]
    /// Calculate the width of a degree in meters at the given latitude.
    pub fn meters_per_degree(latitude: f64) -> f32 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::as_conversions,
            reason = "It's the only way"
        )]
        (EARTH_RADIUS
            * 1000.0
            * latitude.to_radians().cos() as f32
            * (std::f32::consts::PI / 180.0))
            .abs()
    }

    /// Convert a lat/lon to a DEM coordinate.
    ///
    /// # Errors
    /// If projection fails.
    #[inline]
    pub fn lonlat_to_dem_coord(
        metadata: &crate::metadata::MetaData,
        latlon: LonLatCoord,
    ) -> Result<crate::dem::Coordinate> {
        let width = f64::from(metadata.width - 1);
        let scale = f64::from(metadata.scale);
        let coord_metric = Self {
            base: metadata.centre,
        }
        .to_meters(latlon)?;
        let offset = (width * scale) / 2.0f64;
        let dem_coord = crate::dem::Coordinate(
            geo::coord! {
                x: coord_metric.x + offset,
                // Invert the y coordinate because geographic coordinates are anchored to the bottom left
                // and DEM coordinates are anchored to the top right.
                y: -coord_metric.y + offset
            } / scale,
        );
        Ok(dem_coord)
    }

    /// Change the anchor of the AEQD projection. This just gives slightly more accuracy when
    /// reconstructing larger viewsheds on larger DEMs.
    ///
    /// # Errors
    /// If projection fails.
    #[inline]
    pub fn change_metric_origin(
        // The lat/lon of the DEM's top-left corner
        source_degrees_anchor: LonLatCoord,
        // The AEQD coordinates of a viewshed's centre.
        target_metric_anchor: geo::Coord,
        // The point in the viewshed, in metric coordinates, to be converted.
        point: geo::Coord,
    ) -> Result<geo::Coord> {
        // The viewshed's origin in lat/lon.
        let target_degrees_origin = Self {
            base: source_degrees_anchor,
        }
        .to_degrees(target_metric_anchor)?;

        // The point in the viewshed in lat/lon
        let point_in_degrees = Self {
            base: source_degrees_anchor,
        }
        .to_degrees(point)?;

        // The same `point` but anchored to the metric coordinates of the viewshed's centre.
        let target_metric_point = Self {
            base: target_degrees_origin,
        }
        .to_meters(point_in_degrees)?;

        Ok(target_metric_point)
    }
}

#[expect(
    clippy::default_numeric_fallback,
    clippy::unreadable_literal,
    reason = "These are just tests"
)]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bristol_to_meters() {
        let base = LonLatCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Convert { base };
        assert_eq!(
            converter.to_meters(base).unwrap(),
            geo::Coord { x: 0.0, y: 0.0 }
        );
    }

    #[test]
    fn bristolish_to_meters() {
        let base = LonLatCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Convert { base };
        assert_eq!(
            converter
                .to_meters(LonLatCoord(geo::Coord {
                    x: -2.573510680530247,
                    y: 51.463487311585936
                }))
                .unwrap(),
            geo::Coord {
                x: 1000.0000000004705,
                y: 1000.0000000008044
            }
        );
    }

    #[test]
    fn bristol_to_degrees() {
        let base = LonLatCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Convert { base };
        assert_eq!(
            converter.to_degrees(geo::Coord { x: 0.0, y: 0.0 }).unwrap(),
            LonLatCoord(geo::Coord {
                x: -2.5879,
                y: 51.45450000000001
            })
        );
    }

    #[test]
    fn bristolish_to_degrees() {
        let base = LonLatCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Convert { base };
        assert_eq!(
            converter
                .to_degrees(geo::Coord {
                    x: 1000.0,
                    y: 1000.0
                })
                .unwrap(),
            LonLatCoord(geo::Coord {
                x: -2.573510680530247,
                y: 51.463487311585936
            })
        );
    }
}
