//! Helper code.

use color_eyre::Result;

#[inline]
/// Convert a lon/lat to a DEM ID.
///
/// # Errors
///
/// If projection errors.
pub fn lonlat_to_dem_id(
    metadata: &crate::metadata::MetaData,
    latlon: crate::projector::LonLatCoord,
) -> Result<i64> {
    let width_f64 = f64::from(metadata.width + 1);
    let scale = f64::from(metadata.scale);
    let coord_metric = crate::projector::Convert {
        base: metadata.centre,
    }
    .to_meters(latlon)?;
    let offset = (width_f64 * scale) / 2.0f64;
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The coordinates should always fit in `u64`"
    )]
    let (x, y) = {
        (
            ((coord_metric.x + offset) / scale) as i64,
            ((-coord_metric.y + offset) / scale) as i64,
        )
    };
    let dem_id = (y * i64::from(metadata.width)) + x;
    Ok(dem_id)
}

#[inline]
/// Convert a DEM ID to a lon/lat.
///
/// # Errors
///
/// If projection errors.
pub fn dem_id_to_lonlat(
    metadata: &crate::metadata::MetaData,
    dem_id: i64,
) -> Result<crate::projector::LonLatCoord> {
    let width_f64 = f64::from(metadata.width + 1);
    let scale = f64::from(metadata.scale);
    let offset = (width_f64 * scale) / 2.0f64;

    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "These are just lon/lat coordinates"
    )]
    let (x_raster, y_raster) = {
        (
            dem_id.rem_euclid(metadata.width.into()) as f64,
            dem_id.div_euclid(metadata.width.into()) as f64,
        )
    };
    #[expect(
        clippy::suboptimal_flops,
        reason = "We don't need the perfomance and this reads better"
    )]
    let coord_metric = geo::coord! {
        x: (x_raster * scale) - offset,
        y: (-y_raster * scale) + offset
    };

    let coord_lonlat = crate::projector::Convert {
        base: metadata.centre,
    }
    .to_degrees(coord_metric)?;

    Ok(coord_lonlat)
}
