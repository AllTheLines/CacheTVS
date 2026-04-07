//! Save data to a `.tiff` file.

use color_eyre::Result;

/// Save an array of `f32`s (total surfaces, longest lines of sight) to a `.tiff` file.
pub fn save(dem: &crate::dem::DEM, data: &[f32], path: &std::path::PathBuf) -> Result<()> {
    let driver = gdal::DriverManager::get_driver_by_name("GTiff")?;

    let mut dataset = driver.create_with_band_type::<f32, _>(
        path,
        usize::try_from(dem.tvs_width)?,
        usize::try_from(dem.tvs_width)?,
        1,
    )?;

    // Set the origin and resolution
    let scale = f64::from(dem.scale);
    let top_left = (f64::from(dem.tvs_width) * scale) / 2.0f64;
    // [top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height]
    let geotransform = [-top_left, scale, 0.0f64, top_left, 0.0f64, -scale];
    dataset.set_geo_transform(&geotransform)?;

    let lat = dem.centre.0.y;
    let lon = dem.centre.0.x;
    let projection_string = format!("+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m");
    dataset.set_projection(&projection_string)?;

    let mut band = dataset.rasterband(1)?;
    let width_usize = usize::try_from(dem.tvs_width)?;
    let mut buffer = gdal::raster::Buffer::new((width_usize, width_usize), data.to_vec());
    band.write((0, 0), (width_usize, width_usize), &mut buffer)?;

    Ok(())
}
