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
    /// All the elevation data.
    pub elevations: Vec<i16>,
    /// The width of the DEM.
    pub width: u32,
    /// The width of the computable sub-grid within the DEM. Consider a point on the edge of the
    /// DEM, whilst we cannot calculate its viewshed, it is required to calculate a truly coputable
    /// point further inside the DEM.
    pub tvs_width: u32,
    /// The total number of points in the DEM.
    pub size: u64,
    /// The size of each point in meters.
    pub scale: f32,
    /// The geographic location of the centre of the DEM tile.
    pub centre: crate::projection::LonLatCoord,
    /// The maximum distance in terms of points to search.
    pub max_los_as_points: u32,
    /// The total number of points that can have full viewsheds calculated for them.
    pub computable_points_count: u32,
}

impl DEM {
    /// `Instantiate`
    pub fn new(
        centre_latlon: crate::projection::LonLatCoord,
        width: u32,
        scale: f32,
        max_line_of_sight_as_points: u32,
    ) -> Result<Self> {
        let size: u64 = u64::from(width) * u64::from(width);
        let max_possible_los_as_points = width.div_euclid(2);

        if max_line_of_sight_as_points > max_possible_los_as_points {
            color_eyre::eyre::bail!(
                "The maximum line of sight ({max_line_of_sight_as_points}) is longer than the maximum \
                distance that any point can completely see ({}m).",
                f64::from(max_possible_los_as_points) * f64::from(scale)
            );
        }

        if width.rem_euclid(3) != 0 {
            color_eyre::eyre::bail!("Point width ({width} is not divisible by 3)");
        }

        let tvs_width = width.div_euclid(3);
        let computable_points_count = tvs_width.pow(2);

        let dem = Self {
            elevations: Vec::default(),
            width,
            tvs_width,
            size,
            scale,
            centre: centre_latlon,
            max_los_as_points: max_line_of_sight_as_points,
            computable_points_count,
        };
        Ok(dem)
    }

    /// Is the DEM ID within the DEM such that a valid viewshed can be made for it?
    pub fn is_point_computable(&self, dem_id: u32) -> bool {
        let coord = self.convert_dem_id_to_coord(dem_id).0;
        let lower = f64::from(self.max_los_as_points);
        let upper = f64::from((self.width - 1) - self.max_los_as_points);
        coord.x >= lower && coord.x <= upper && coord.y >= lower && coord.y <= upper
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
}

/// Serialise for Debugging.
#[expect(
    clippy::missing_fields_in_debug,
    reason = "We don't want to output GBs of data!"
)]
impl std::fmt::Debug for DEM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DEM")
            .field("width", &self.width)
            .field("tvs_width", &self.tvs_width)
            .field("size", &self.size)
            .field("scale", &self.scale)
            .field("centre", &self.centre)
            .field("max_los_as_points", &self.max_los_as_points)
            .field("computable_points_count", &self.computable_points_count)
            .finish()
    }
}
