//! Save data to a `.bt` file.

use color_eyre::Result;

/// Save an array of `f32`s (total surfaces, longest lines of sight) to a `.bt` file.
pub fn save(dem: &crate::dem::DEM, data: &[f32], path: &std::path::PathBuf) -> Result<()> {
    let projector = crate::projection::Converter { base: dem.centre };
    let offset = (f64::from(dem.width) * f64::from(dem.scale)) / 2.0f64;
    let bottom_left = projector.to_degrees(geo::coord! { x: -offset, y: -offset })?;
    let top_right = projector.to_degrees(geo::coord! { x: offset, y: offset })?;
    let bt = crate::bt::BinaryTerrain {
        header: crate::bt::header::Header {
            width: dem.tvs_width,
            height: dem.tvs_width,
            left: bottom_left.0.x,
            right: top_right.0.x,
            bottom: bottom_left.0.y,
            top: top_right.0.y,
            ..Default::default()
        },
        data: crate::bt::header::Data::Float32(data.to_vec()),
    };

    bt.write(path)?;

    Ok(())
}
