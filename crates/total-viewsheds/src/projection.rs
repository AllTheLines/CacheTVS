//! Project coordinates between different systems.

use color_eyre::Result;

/// A latitude/longtitude coordinate.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Default)]
pub struct LatLonCoord(pub geo::Coord);

/// Convert between different coordinate system.
pub struct Converter {
    /// The lat/lon base coordinates for the AEQD mercator projected coordinates.
    pub base: LatLonCoord,
}

impl Converter {
    /// The projection description for lat/lon.
    fn degrees_projection() -> Result<proj4rs::Proj> {
        let string = "+proj=latlong +datum=WGS84";
        Ok(proj4rs::Proj::from_proj_string(string)?)
    }

    /// The projection description for the AEQD metric projection.
    fn meters_projection(&self) -> Result<proj4rs::Proj> {
        let string = format!(
            "+proj=aeqd +lat_0={} +lon_0={} +datum=WGS84",
            self.base.0.y, self.base.0.x
        );
        Ok(proj4rs::Proj::from_proj_string(&string)?)
    }

    /// Convert from degrees to the AEQD metric projection.
    pub fn to_meters(&self, source: LatLonCoord) -> Result<geo::Coord> {
        let mut converted = (source.0.x.to_radians(), source.0.y.to_radians(), 0.0f64);
        proj4rs::transform::transform(
            &Self::degrees_projection()?,
            &self.meters_projection()?,
            &mut converted,
        )?;

        Ok(geo::coord! { x: converted.0, y: converted.1 })
    }

    /// Convert from the AEQD metric projection to degrees.
    pub fn to_degrees(&self, source: geo::Coord) -> Result<LatLonCoord> {
        let mut converted = (source.x, source.y, 0.0f64);
        proj4rs::transform::transform(
            &self.meters_projection()?,
            &Self::degrees_projection()?,
            &mut converted,
        )?;

        Ok(LatLonCoord(
            geo::coord! { x: converted.0.to_degrees(), y: converted.1.to_degrees() },
        ))
    }

    /// Chante the anchor of the AEQD projection. This just gives slightly more accuracy when
    /// reconstructing larger viewsheds on larger DEMs.
    pub fn change_metric_origin(
        // The lat/lon of the DEM's top-left corner
        source_degrees_anchor: LatLonCoord,
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
        let base = LatLonCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Converter { base };
        assert_eq!(
            converter.to_meters(base).unwrap(),
            geo::Coord { x: 0.0, y: 0.0 }
        );
    }

    #[test]
    fn bristolish_to_meters() {
        let base = LatLonCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Converter { base };
        assert_eq!(
            converter
                .to_meters(LatLonCoord(geo::Coord {
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
        let base = LatLonCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Converter { base };
        assert_eq!(
            converter.to_degrees(geo::Coord { x: 0.0, y: 0.0 }).unwrap(),
            LatLonCoord(geo::Coord {
                x: -2.5879,
                y: 51.45450000000001
            })
        );
    }

    #[test]
    fn bristolish_to_degrees() {
        let base = LatLonCoord(geo::Coord {
            x: -2.5879,
            y: 51.4545,
        });
        let converter = Converter { base };
        assert_eq!(
            converter
                .to_degrees(geo::Coord {
                    x: 1000.0,
                    y: 1000.0
                })
                .unwrap(),
            LatLonCoord(geo::Coord {
                x: -2.573510680530247,
                y: 51.463487311585936
            })
        );
    }
}
