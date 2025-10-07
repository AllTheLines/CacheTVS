//! Load elevation data from a `.bt` file.

#![expect(
    clippy::little_endian_bytes,
    reason = "The `.bt` file format is little endian"
)]

use std::io::{Read as _, Seek as _};

use color_eyre::Result;
use geo::Distance as _;

impl super::BinaryTerrain {
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
        if !magic_str.starts_with(super::header::MAGIC_BT_STRING) {
            color_eyre::eyre::bail!("Not a Binary Terrain v1.3 file: {}", magic_str);
        }

        let header = super::header::Header {
            width: read_u32_le(&mut file)?,
            height: read_u32_le(&mut file)?,
            data_size: read_u16_le(&mut file)?,
            data_type: if read_u16_le(&mut file)? == 1 {
                super::header::DataType::Float32
            } else {
                super::header::DataType::Int16
            },
            horizontal_units: read_u16_le(&mut file)?,
            utm_zone: read_u16_le(&mut file)?,
            datum: read_u16_le(&mut file)?,
            left: read_f64_le(&mut file)?,
            right: read_f64_le(&mut file)?,
            bottom: read_f64_le(&mut file)?,
            top: read_f64_le(&mut file)?,
            projection_source: read_u16_le(&mut file)?,
            vertical_scale: read_f32_le(&mut file)?,
        };
        tracing::info!("DEM header parsed: {:?}", header);

        // Skip rest of header (256 total)
        file.seek(std::io::SeekFrom::Start(crate::bt::header::HEADER_SIZE))?;

        let points_count = usize::try_from(header.width * header.height)?;
        let data_bytes = match header.data_type {
            super::header::DataType::Int16 => points_count * 2,
            super::header::DataType::Float32 => points_count * 4,
        };
        let mut buffer = vec![0; data_bytes];
        file.read_exact(&mut buffer)?;

        // Elevation data
        tracing::info!("Loading {points_count} DEM points...");
        let data = match header.data_type {
            super::header::DataType::Float32 => {
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
                        values[Self::flip_from(index, header.width, header.height)?] = value;
                    }
                }
                super::header::Data::Float32(values)
            }
            super::header::DataType::Int16 => {
                if header.data_size != 2 {
                    color_eyre::eyre::bail!(
                        "Unsupported `.bt` field value for data size: {}",
                        header.data_size
                    );
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
                        values[Self::flip_from(index, header.width, header.height)?] = value;
                    }
                }
                super::header::Data::Int16(values)
            }
        };

        tracing::info!("Dem file loaded.");
        Ok(Self { header, data })
    }

    /// Flip data so that it starts in the top-left and ends at the bottom-right.
    fn flip_from(index: usize, width_u32: u32, height_u32: u32) -> Result<usize> {
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

    #[test]
    fn flip_from() {
        #[rustfmt::skip]
        let original = [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ];

        let mut flipped = vec![0; original.len()];
        for index in 0..original.len() {
            let flipped_index = crate::bt::header::BinaryTerrain::flip_from(index, 3, 3).unwrap();
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
