//! Load elevation data from a `.bt` file.

#![expect(
    clippy::little_endian_bytes,
    reason = "The `.bt` file format is little endian"
)]
#![expect(clippy::indexing_slicing, reason = "All the offsets are known")]

use std::io::Write as _;

use color_eyre::Result;

impl super::BinaryTerrain {
    /// Write a `.bt` file.
    pub fn write(&self, path: &std::path::PathBuf) -> Result<()> {
        tracing::info!("Writing DEM data to: {}", path.display());
        let mut file = std::fs::File::create(path)?;

        let mut header = vec![0u8; crate::bt::header::HEADER_SIZE.try_into()?];

        header[0..crate::bt::header::MAGIC_BT_STRING.len()]
            .copy_from_slice(crate::bt::header::MAGIC_BT_STRING.as_bytes());
        write_le_u32(&mut header, 10, self.header.width);
        write_le_u32(&mut header, 14, self.header.height);
        write_le_i16(&mut header, 18, 4); // We're only supporting `f32` here.
        write_le_i16(&mut header, 20, 1); // Mark as "is_float".
        write_le_i16(&mut header, 22, 0); // Units are "degrees".
        write_le_i16(&mut header, 24, 0); // No UTM zone.
        write_le_i16(&mut header, 26, 0); // No datum.
        write_le_f64(&mut header, 28, self.header.left);
        write_le_f64(&mut header, 36, self.header.right);
        write_le_f64(&mut header, 44, self.header.top);
        write_le_f64(&mut header, 52, self.header.bottom);
        file.write_all(&header)?;

        let crate::bt::header::Data::Float32(data) = &self.data else {
            color_eyre::eyre::bail!("Only writing float data is currently supported.");
        };

        let mut flipped = vec![0f32; data.len()];
        for (index, value) in data.iter().enumerate() {
            let flipped_index = Self::flip_into(index, self.header.width)?;
            flipped[flipped_index] = *value;
        }

        let mut bytes = Vec::with_capacity(flipped.len() * 4);
        for value in flipped {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)?;

        Ok(())
    }

    /// Flip data so that it is column-major order, each column ordering from south to north.
    fn flip_into(index: usize, width_u32: u32) -> Result<usize> {
        let width = usize::try_from(width_u32)?;
        let x = index.div_euclid(width);
        let y = index.rem_euclid(width);
        let x_flipped = (width - 1) - x;

        Ok((y * width) + x_flipped)
    }
}

/// Write a `u32` to the `.bt` file.
fn write_le_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Write a `i16` to the `.bt` file.
fn write_le_i16(buffer: &mut [u8], offset: usize, value: i16) {
    buffer[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Write a `f64` to the `.bt` file.
fn write_le_f64(buffer: &mut [u8], offset: usize, value: f64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
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
    fn flip_into() {
        #[rustfmt::skip]
        let original = [
            2, 5, 8,
            1, 4, 7,
            0, 3, 6
        ];

        let mut flipped = vec![0; original.len()];
        for index in 0..original.len() {
            let flipped_index = crate::bt::header::BinaryTerrain::flip_into(index, 3).unwrap();
            flipped[flipped_index] = original[index];
        }

        #[rustfmt::skip]
        assert_eq!(flipped, [
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        ]);
    }
}
