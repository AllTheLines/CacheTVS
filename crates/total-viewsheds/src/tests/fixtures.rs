//! Test fixtures

use color_eyre::eyre::Result;

/// The nodata value from NASA's SRTM data.
pub(crate) const NODATA: f32 = -32768.0;

// This is a little map to help orient yourself when looking at the results:
//
// 0,  1,  2,  3,   4,  5,  6,  7,   8,  9,  10, 11
// 12, 13, 14, 15,  16, 17, 18, 19,  20, 21, 22, 23
// 24, 25, 26, 27,  28, 29, 30, 31,  32, 33, 34, 35
// 36, 37, 38, 39,  40, 41, 42, 43,  44, 45, 46, 47
//
// 48, 49, 50, 51,  52, 53, 54, 55,  56, 57, 58, 59
// 60, 61, 62, 63,  64, 65, 66, 67,  68, 69, 70, 71
// 72, 73, 74, 75,  76, 77, 78, 79,  80, 81, 82, 83
// 84, 85, 86, 87,  88, 89, 90, 91,  92, 93, 94, 95
//
// 96, 97, 98, 99,  100,101,102,103, 104,105,106,107
// 108,109,110,111, 112,113,114,115, 116,117,118,119
// 120,121,122,123, 124,125,126,127, 128,129,130,131
// 132,133,134,135, 136,137,138,139, 140,141,142,143
#[inline]
#[must_use]
#[rustfmt::skip]
/// A bigger DEM for making viewsheds.
pub(crate) fn bigger_dem() -> Vec<i16> {
    let x = 15;
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "NODATA is actually i16::MIN"
    )]
    let n = NODATA as i16;
    vec![
        0,0,0,0, 0,0,0,0, 0,0,0,1,
        0,1,1,1, 1,1,1,1, 0,1,0,0,
        0,1,3,3, 2,2,2,3, 4,3,0,0,
        0,1,3,4, 3,3,3,3, 3,2,1,0,

        0,1,3,4, 5,5,5,4, 3,2,1,0,
        0,1,3,4, 4,0,6,5, 4,2,1,0,
        0,1,1,2, 5,9,x,7, 5,2,1,0,
        0,1,1,4, 4,4,4,6, 3,2,1,0,

        0,0,3,4, 4,4,5,4, n,n,n,n,
        0,3,3,4, 3,2,3,3, n,n,n,n,
        0,0,3,4, 2,2,1,2, n,n,n,n,
        3,0,2,3, 1,0,1,0, n,n,n,n,
    ]
}

/// Create the TVS heatmap that a compute run outputs.
pub(crate) fn create_tvs_tiff(data: Vec<f32>) -> Result<gdal::Dataset> {
    let width = data.len().isqrt();
    let driver = gdal::DriverManager::get_driver_by_name("MEM")?;
    let dataset = driver.create_with_band_type::<f32, _>("", width, width, 1)?;
    let mut buffer = gdal::raster::Buffer::new((width, width), data);
    let mut band = dataset.rasterband(1)?;
    band.write((0, 0), (width, width), &mut buffer)?;

    Ok(dataset)
}
