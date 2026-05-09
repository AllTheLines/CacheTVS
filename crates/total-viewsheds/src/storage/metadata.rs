//! Struct for storing essential data about the underlying DEM for which viewsheds are created.

/// Metadata about the viewshed data.
#[derive(serde::Serialize, serde::Deserialize, Default, Debug, Clone)]
pub struct MetaData {
    /// The width of the 2D grid of elevation data. The algorithm requires that the grid be square,
    /// so there is no need for a height field.
    pub width: u32,
    /// The diameter in meters each point of the data covers.
    pub scale: f32,
    /// The maximum line of sight, in points).
    pub max_line_of_sight: u32,
    /// The lat/lon coordinates for the centre of the 2D DEM grid. Used for accurately converting
    /// between degree and metric coordinate systems.
    pub centre: crate::projection::LonLatCoord,
    /// The size of the region (in raster points) within which we will find the viewsheds with the
    /// largest surface area. Used for reducing the final size of viewshed data saved to disk.
    pub neighbourhood_size: u32,
}
