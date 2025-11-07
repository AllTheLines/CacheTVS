//! Save data to a `.bt` file.

use color_eyre::Result;

/// Save an array of `f32`s (total surfaces, longest lines of sight) to a `.bt` file.
pub fn save(dem: &crate::dem::DEM, data: &[f32], path: &std::path::PathBuf) -> Result<()> {
    let bt = crate::bt::BinaryTerrain {
        header: crate::bt::header::Header {
            width: dem.tvs_width,
            height: dem.tvs_width,
            left: dem.centre.0.x,
            top: dem.centre.0.y,
            right: dem.centre.0.x,
            bottom: dem.centre.0.y,
            ..Default::default()
        },
        data: crate::bt::header::Data::Float32(data.to_vec()),
    };

    bt.write(path)?;

    Ok(())
}
