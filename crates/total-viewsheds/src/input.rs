//! Load elevation data from a `.bt` file.

// TODO: Move over to GeoTiff. The current blessed crate fails for larger DEMs.

#![expect(
    clippy::little_endian_bytes,
    reason = "The `.bt` file format is little endian"
)]

use std::io::{Read as _, Seek as _};

use color_eyre::Result;
use geo::Distance as _;

/// DEM data is allowed to be in integers or floats.
pub enum DataType {
    /// Integer data.
    Int16(std::vec::Vec<i16>),
    /// Float data.
    Float32(std::vec::Vec<f32>),
}

/// The `.bt` file header.
#[expect(
    dead_code,
    reason = "I'd like to make this into its own crate at some point."
)]
#[derive(Default, Debug)]
pub struct Header {
    /// Width of DEM raster.
    pub width: u32,
    /// Height of DEM raster.
    pub height: u32,
    /// Bytes per elevation grid point, either 2 or 4.
    pub data_size: u16,
    /// Is the data floating point or not?
    pub is_float: bool,
    /// Units of the distance between points:
    ///   0: Degrees
    ///   1: Meters
    ///   2: Feet (international foot = .3048 meters)
    ///   3: Feet (U.S. survey foot = 1200/3937 meters)
    pub horizontal_units: u16,
    /// Indicates the UTM zone (1-60) if the file is in UTM.
    /// Negative zone numbers are used for the southern hemisphere.
    pub utm_zone: u16,
    /// Indicates the Datum,
    ///
    /// The Datum field should be an EPSG Geodetic Datum Code, which are in the range of 6001 to 6904.
    /// If you are unfamiliar with these and do not care about Datum, you can simply use the value "6326"
    /// which is the WGS84 Datum.
    ///
    /// The simpler USGS Datum Codes are also supported for as a backward-compatibility with older files,
    /// but all new files should use the more complete EPSG Codes.
    pub datum: u16,
    /// The extents are specified in the coordinate space (projection) of the file. For example, if the
    /// file is using UTM, then the extents are in UTM coordinates.
    pub left: f64,
    /// The right-most extent.
    pub right: f64,
    /// The bottom-most extent.
    pub bottom: f64,
    /// The top-most extent.
    pub top: f64,
    /// Where to find the projection informaation.
    ///   0: Projection is fully described by this header
    ///   1: Projection is specified in a external .prj file
    pub projection_source: u16,
    /// Vertical units in meters, usually 1.0. The value 0.0 should be interpreted as 1.0 to allow
    /// for backward compatibility.
    pub vertical_scale: f32,
}

/// The `.bt` format layout.
pub struct BinaryTerrain {
    /// The header meta data.
    pub header: Header,
    /// The DEM data itself.
    pub data: DataType,
}

impl BinaryTerrain {
    /// Read and parse a `.bt` file.
    #[expect(
        clippy::panic_in_result_fn,
        reason = "The assertions are just to make clippy happy"
    )]
    pub fn read(path: &std::path::PathBuf) -> Result<Self> {
        tracing::info!("Loading DEM data from: {}", path.display());
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 10];
        file.read_exact(&mut magic)?;
        let magic_str = std::string::String::from_utf8_lossy(&magic);
        if !magic_str.starts_with("binterr1.3") {
            color_eyre::eyre::bail!("Not a Binary Terrain v1.3 file: {}", magic_str);
        }

        let width = read_u32_le(&mut file)?;
        let height = read_u32_le(&mut file)?;
        let data_size = read_u16_le(&mut file)?;
        let is_float = read_u16_le(&mut file)? != 0;
        let horizontal_units = read_u16_le(&mut file)?;
        let utm_zone = read_u16_le(&mut file)?;
        let datum = read_u16_le(&mut file)?;

        let left = read_f64_le(&mut file)?;
        let right = read_f64_le(&mut file)?;
        let bottom = read_f64_le(&mut file)?;
        let top = read_f64_le(&mut file)?;

        let projection_source = read_u16_le(&mut file)?;

        let vertical_scale = read_f32_le(&mut file)?;

        let header = Header {
            width,
            height,
            data_size,
            is_float,
            horizontal_units,
            utm_zone,
            datum,
            left,
            right,
            bottom,
            top,
            projection_source,
            vertical_scale,
        };
        tracing::info!("DEM header parsed: {:?}", header);

        // Skip rest of header (256 total)
        file.seek(std::io::SeekFrom::Start(256))?;

        let points_count = usize::try_from(width * height)?;
        let data_bytes = if is_float {
            points_count * 4
        } else {
            points_count * 2
        };
        let mut buffer = vec![0; data_bytes];
        file.read_exact(&mut buffer)?;

        // Elevation data
        tracing::info!("Loading {points_count} DEM points...");
        let data = if is_float {
            let mut values = vec![0.0; points_count];
            for (index, chunk) in buffer.chunks_exact(4).enumerate() {
                assert!(
                    chunk.len() == 4,
                    "to prove to clippy that array indexing is okay"
                );
                #[expect(
                    clippy::indexing_slicing,
                    reason = "We've already proven the array sizes"
                )]
                {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    values[Self::flip(index, width, height)?] = value;
                }
            }
            DataType::Float32(values)
        } else {
            if data_size != 2 {
                color_eyre::eyre::bail!("Unsupported `.bt` field value for data size: {data_size}");
            }
            let mut values = vec![0; points_count];
            for (index, chunk) in buffer.chunks_exact(2).enumerate() {
                assert!(
                    chunk.len() == 2,
                    "to prove to clippy that array indexing is okay"
                );
                #[expect(
                    clippy::indexing_slicing,
                    reason = "We've already proven the array size"
                )]
                {
                    let value = i16::from_le_bytes([chunk[0], chunk[1]]);
                    values[Self::flip(index, width, height)?] = value;
                }
            }
            DataType::Int16(values)
        };

        tracing::info!("Dem file loaded.");
        Ok(Self { header, data })
    }

    /// Flip data so that it starts in the top-left and ends at the bottom-right.
    fn flip(index: usize, width_u32: u32, height_u32: u32) -> Result<usize> {
        let width = usize::try_from(width_u32)?;
        let height = usize::try_from(height_u32)?;
        let x = index.div_euclid(height);
        let y = index.rem_euclid(height);
        let y_flipped = (width - 1) - y;

        Ok((y_flipped * height) + x)
    }

    /// Derive the scale of the DEM. Units are in meters.
    pub fn scale(&self) -> f64 {
        let top_right = geo::Point::new(self.header.top, self.header.right);
        let top_left = geo::Point::new(self.header.top, self.header.left);
        let distance = geo::Haversine.distance(top_right, top_left);
        let scale = distance / f64::from(self.header.width);
        tracing::debug!("DEM scale calculated to {scale}m.");
        scale
    }

    /// Get the centre of the tile. We repurpose the extent fields for this.
    pub fn centre(&self) -> crate::projection::LatLonCoord {
        crate::projection::LatLonCoord((self.header.left, self.header.top).into())
    }
}

/// Convert raw bytes to `u16`.
fn read_u16_le(file: &mut std::fs::File) -> std::io::Result<u16> {
    let mut buffer = [0u8; 2];
    file.read_exact(&mut buffer)?;
    Ok(u16::from_le_bytes(buffer))
}

/// Convert raw bytes to `u32`.
fn read_u32_le(file: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

/// Convert raw bytes to `f32`.
fn read_f32_le(file: &mut std::fs::File) -> std::io::Result<f32> {
    let mut buffer = [0u8; 4];
    file.read_exact(&mut buffer)?;
    Ok(f32::from_le_bytes(buffer))
}

/// Convert raw bytes to `f64`.
fn read_f64_le(file: &mut std::fs::File) -> std::io::Result<f64> {
    let mut buffer = [0u8; 8];
    file.read_exact(&mut buffer)?;
    Ok(f64::from_le_bytes(buffer))
}

#[expect(
    clippy::default_numeric_fallback,
    clippy::needless_range_loop,
    clippy::indexing_slicing,
    reason = "These are just tests"
)]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn flip() {
        #[rustfmt::skip]
        let original = [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ];

        let mut flipped = vec![0; original.len()];
        for index in 0..original.len() {
            let flipped_index = BinaryTerrain::flip(index, 3, 3).unwrap();
            flipped[flipped_index] = original[index];
        }

        #[rustfmt::skip]
        assert_eq!(flipped, [
            2, 5, 8,
            1, 4, 7,
            0, 3, 6
        ]);
    }
}
