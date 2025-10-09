//! Converting final output to PNG images.
use color_eyre::{eyre::ContextCompat as _, Result};

/// Convert an array of floats to a grayscale heatmap.
pub fn save(data: &[f32], width: u32, height: u32, path: std::path::PathBuf) -> Result<()> {
    tracing::info!("Writing PNG data to: {}", path.display());
    let (min, max) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
            (min.min(value), max.max(value))
        });
    let range = if max - min == 0.0 { 1.0 } else { max - min };

    let pixels: Vec<u8> = data
        .iter()
        .map(|&value| {
            let normalised = ((value - min) / range * 255.0).clamp(0.0, 255.0);
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "We've already guaranteed all the values are within the correct range"
            )]
            {
                normalised as u8
            }
        })
        .collect();

    let count = pixels.len();
    if count != usize::try_from(width * height)? {
        color_eyre::eyre::bail!(
            "Pixel count ({count}) doesn't fit into dimensions: {width}x{height}"
        );
    }
    let png: image::GrayImage = image::GrayImage::from_vec(width, height, pixels).context(
        format!("Dimensions ({width}x{height}) don't match the amount of data ({count})."),
    )?;

    png.save(path)?;

    Ok(())
}
