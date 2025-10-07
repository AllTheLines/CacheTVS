//! DEM: Digital Elevation Model
//! <https://en.wikipedia.org/wiki/Digital_elevation_model>
//!
//! A 1D array of 2D data. Each item represents the height of a point above sea level. It doesn't
//! contain coordinates itself. But they can be derived from the position of the item in the array
//! and the known location of the DEM origins itself.

use color_eyre::Result;

/// The coordinate space of the DEM:
///   * 0,0 is the top-left.
///   * Each integer maps exactly to a point in the DEM data.
///   
/// These coordinates are used by the kernel and tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate(pub geo::Coord);

/// `DEM`
pub struct DEM {
    /// Trigonomic data about the points when you rotate the DEM through certain angles.
    pub axes: crate::axes::Axes,
    /// The band deltas for every sector.
    pub band_deltas: Vec<i32>,
    /// All the distances for the band.
    pub band_distances: Vec<f32>,
    /// All the elevation data.
    pub elevations: Vec<f32>,
    /// The width of the DEM.
    pub width: u32,
    /// The width of the computable sub-grid within the DEM. Consider a point on the edge of the
    /// DEM, whilst we cannot calculate its viewshed, it is required to calculate a truly coputable
    /// point further inside the DEM.
    pub tvs_width: u32,
    /// The total number of points in the DEM.
    pub size: u32,
    /// The size of each point in meters.
    pub scale: f32,
    /// The geographic location of the centre of the DEM tile.
    pub centre: crate::projection::LatLonCoord,
    /// The maximum distance in metres to search for visible points.
    pub max_line_of_sight: u32,
    /// The maximum distance in terms of points to search.
    pub max_los_as_points: u32,
    /// The total number of points that can have full viewsheds calculated for them.
    pub computable_points_count: u32,
    /// The size of a "band of sight". This is generally the number of points that fit into the max
    /// line of sight. But it could be more, not to increase the distance, but to improve
    /// interpolation.
    pub band_size: u32,
}

impl DEM {
    /// `Instantiate`
    pub fn new(
        centre_latlon: crate::projection::LatLonCoord,
        width: u32,
        scale: f32,
        max_line_of_sight: u32,
    ) -> Result<Self> {
        let size = width * width;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::as_conversions,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "This shouldn't be a problem in most sane cases"
        )]
        let max_los_as_points = (max_line_of_sight as f32 / scale) as u32;
        let max_possible_los_as_points = width.div_euclid(2);

        if max_los_as_points > max_possible_los_as_points {
            color_eyre::eyre::bail!(
                "The maximum line of sight ({max_line_of_sight}m) is longer than the maximum \
                distance that any point can completely see ({}m).",
                f64::from(max_possible_los_as_points) * f64::from(scale)
            );
        }

        let mut dem = Self {
            axes: crate::axes::Axes::default(),
            band_deltas: Vec::default(),
            band_distances: Vec::default(),
            elevations: Vec::default(),
            width,
            tvs_width: 0,
            size,
            scale,
            centre: centre_latlon,
            max_line_of_sight,
            max_los_as_points,
            computable_points_count: 0,
            // Add 1 just to be sure that we always compute points within the line of sight, and no
            // less.
            band_size: max_los_as_points + 1,
        };
        dem.count_computable_points();
        dem.tvs_width = dem.computable_points_count.isqrt();
        Ok(dem)
    }

    /// Count the number of points in the DEM that can have their viewsheds fully calculated.
    fn count_computable_points(&mut self) {
        self.computable_points_count = 0;
        for point in 0..self.size {
            if self.is_point_computable(point) {
                self.computable_points_count += 1;
            }
        }
    }

    /// Depending on the requested max line of sight, only certain points in the middle of the DEM
    /// can truly have their total visible surfaces calculated. This is because points on the edge
    /// do not have access to further elevation data.
    pub fn is_point_computable(&self, dem_id: u32) -> bool {
        let scale = f64::from(self.scale);
        let max_line_of_sight = f64::from(self.max_line_of_sight);
        let coord = self.convert_dem_id_to_coord(dem_id).0 * scale;
        let lower = max_line_of_sight;
        let upper = (f64::from(self.width - 1) * scale) - max_line_of_sight;
        coord.x >= lower && coord.x <= upper && coord.y >= lower && coord.y <= upper
    }

    /// Convert an original DEM ID to the coordinate system of the computable points sub-DEM.
    #[expect(dead_code, reason = "We'll use it in the next commit")]
    pub fn pov_id_to_tvs_id(&self, pov_id: u64) -> u64 {
        let max_los_as_points_u64 = u64::from(self.max_los_as_points);
        let width_u64 = u64::from(self.width);
        let x = pov_id.rem_euclid(width_u64) - max_los_as_points_u64;
        let y = pov_id.div_euclid(width_u64) - max_los_as_points_u64;
        (y * u64::from(self.tvs_width)) + x
    }

    /// Convert a DEM 1D index to a 2D coordinate.
    pub fn convert_dem_id_to_coord(&self, dem_id: u32) -> Coordinate {
        let x = f64::from(dem_id.rem_euclid(self.width));
        let y = f64::from(dem_id.div_euclid(self.width));
        Coordinate(geo::coord! {x: x, y: y})
    }

    /// Convert a DEM coordinate to a DEM ID.
    pub fn dem_coord_to_id(&self, coord: Coordinate) -> u32 {
        let x = coord.0.x.round();
        let yish = coord.0.y.round() * f64::from(self.width);
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "I don't think there's any other way?"
        )]
        {
            (yish + x) as u32
        }
    }

    /// Convert a lat/lon to a DEM coordinate.
    pub fn latlon_to_dem_coord(
        &self,
        latlon: crate::projection::LatLonCoord,
    ) -> Result<crate::dem::Coordinate> {
        let width = f64::from(self.width - 1);
        let scale = f64::from(self.scale);
        let coord_metric = crate::projection::Converter { base: self.centre }.to_meters(latlon)?;
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

    /// Convert a computable sub-DEM ID to its original DEM ID.
    #[cfg(test)]
    pub const fn tvs_id_to_pov_id(&self, tvs_id: u32) -> u32 {
        let x = tvs_id.rem_euclid(self.tvs_width) + self.max_los_as_points;
        let y = tvs_id.div_euclid(self.tvs_width) + self.max_los_as_points;
        (y * self.width) + x
    }

    /// Do the calculations needed to create bands for a new angle.
    pub fn calculate_axes(&mut self, angle: f32) -> Result<()> {
        self.axes = crate::axes::Axes::new(self.width, angle)?;
        self.axes.compute();
        Ok(())
    }
}

/// Serialise for Debugging.
#[expect(
    clippy::missing_fields_in_debug,
    reason = "We don't want to output GBs of data!"
)]
impl std::fmt::Debug for DEM {
    #[expect(clippy::min_ident_chars, reason = "This is from `std`")]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DEM")
            .field("width", &self.width)
            .field("tvs_width", &self.tvs_width)
            .field("size", &self.size)
            .field("scale", &self.scale)
            .field("centre", &self.centre)
            .field("max_line_of_sight", &self.max_line_of_sight)
            .field("max_los_as_points", &self.max_los_as_points)
            .field("computable_points_count", &self.computable_points_count)
            .field("band_size", &self.band_size)
            .finish()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn latlon_to_dem_coord() {
        let centre = crate::projection::LatLonCoord((-33.33f64, 12.34f64).into());
        let dem = DEM::new(centre, 101, 5.0, 250).unwrap();

        assert_eq!(
            dem.latlon_to_dem_coord(centre).unwrap(),
            Coordinate((50.0f64, 50.0f64).into())
        );
    }
}
