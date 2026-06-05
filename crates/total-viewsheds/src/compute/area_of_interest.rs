//! Determine whether a given point in a DEM needs to have viewsheds computed for it.

use color_eyre::Result;

use geo::Contains as _;

/// `Pruner`
pub struct Pruner {
    /// Width of the DEM in points
    width: u32,
    /// The DEM coordinates for the Area of Interest.
    polygon: geo::Polygon,
}

impl Pruner {
    /// Instantiate.
    pub const fn new(width: u32, polygon: geo::Polygon) -> Self {
        Self { width, polygon }
    }

    /// Convert user-provided lon/lat coordinates to a polygon.
    pub fn lonlat_coords_to_polygon(
        points: Vec<(f32, f32)>,
        metadata: &tvs_lib::metadata::MetaData,
    ) -> Result<geo::Polygon> {
        let mut vertices = Vec::new();

        for intrest_point in points {
            let lonlat = tvs_lib::projector::LonLatCoord(
                geo::coord!(x: intrest_point.0.into(), y: intrest_point.1.into()),
            );
            let dem_coord = tvs_lib::projector::Convert::lonlat_to_dem_coord(metadata, lonlat)?;
            let dem_point = geo::point!(x: dem_coord.0.x, y: dem_coord.0.y);
            vertices.push(dem_point);
        }

        let exterior = geo::LineString::from(vertices);
        let polygon = geo::Polygon::new(exterior, vec![]);

        Ok(polygon)
    }

    /// Can the given DEM ID be ignored in computations.
    pub fn is_prunable(&self, dem_id: i64) -> bool {
        let dem_coord = self.convert_dem_id_to_coord(dem_id);
        !self.polygon.contains(&dem_coord.0)
    }

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "We're only dealing with a max of the DEM's width"
    )]
    /// Convert a DEM 1D index to a 2D coordinate.
    pub fn convert_dem_id_to_coord(&self, dem_id: i64) -> tvs_lib::dem::Coordinate {
        let x = dem_id.rem_euclid(self.width.into()) as f64;
        let y = dem_id.div_euclid(self.width.into()) as f64;
        tvs_lib::dem::Coordinate(geo::coord! {x: x, y: y})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pruner() -> Pruner {
        let width = 300;
        let metadata = tvs_lib::metadata::MetaData {
            width,
            scale: 100.0,
            centre: tvs_lib::projector::LonLatCoord((-3.1791, 51.4816).into()),
            ..Default::default()
        };
        let cardiff_10km = vec![
            (-3.2511, 51.4366),
            (-3.1071, 51.4366),
            (-3.1071, 51.5266),
            (-3.2511, 51.5266),
            (-3.2511, 51.4366),
        ];
        let polygon = Pruner::lonlat_coords_to_polygon(cardiff_10km, &metadata).unwrap();
        Pruner::new(width, polygon)
    }

    #[test]
    fn is_pruned_outside_polygon() {
        let pruner = make_pruner();
        assert!(pruner.is_prunable(0));
    }

    #[test]
    fn is_not_pruned_inside_polygon() {
        let pruner = make_pruner();
        // Why the extra 150 to get to the centre??
        let dem_id = (300i64.pow(2) / 2) + 150;
        assert!(!pruner.is_prunable(dem_id));
    }
}
