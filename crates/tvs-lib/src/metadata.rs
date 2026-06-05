//! Struct for storing essential data about the underlying DEM for which viewsheds are created.

/// Metadata about the viewshed data.
#[expect(
    clippy::exhaustive_structs,
    reason = "It's just nice to contsruct using struct syntax"
)]
#[derive(serde::Serialize, serde::Deserialize, Default, Debug, Clone)]
pub struct MetaData {
    /// The width of the 2D grid of elevation data. The algorithm requires that the grid be square,
    /// so there is no need for a height field.
    pub width: u32,
    /// The diameter in meters each point of the data covers.
    pub scale: f32,
    /// The maximum line of sight (in meters) that was used to calculate the ring data. It is needed
    /// to instantiate the `DEM` struct and therefore reconstruct the bands of sight used to create
    /// the ring data.
    pub max_line_of_sight: u32,
    /// The lat/lon coordinates for the centre of the 2D DEM grid. Used for accurately converting
    /// between degree and metric coordinate systems.
    pub centre: crate::projector::LonLatCoord,
    /// The size of the region (in raster points) within which we will find the viewsheds with the
    /// largest surface area. Used for reducing the final size of viewshed data saved to disk.
    pub neighbourhood_size: u32,
    /// The number of angle subdivisions used.
    pub angle_subdivisions: u32,
}
