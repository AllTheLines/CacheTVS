//! Converting final output to PNG images.
use color_eyre::{Result, eyre::ContextCompat as _};

/// Convert an array of floats to a grayscale heatmap.
pub(crate) fn save(
    data: &[f32],
    width: u32,
    height: u32,
    path: std::path::PathBuf,
    normalisation: crate::config::HeatmapNormalisation,
) -> Result<()> {
    tracing::info!("Writing PNG data to: {}", path.display());

    let data_normalised = match normalisation {
        crate::config::HeatmapNormalisation::UnitScale => unit_normalise(data),
        crate::config::HeatmapNormalisation::Exponential => exponential_normalise(data),
        crate::config::HeatmapNormalisation::Welford => welford_normalise(data),
    };

    let pixels: Vec<u8> = data_normalised
        .iter()
        .map(|&value| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "We've already guaranteed all the values are within the correct range"
            )]
            {
                (value * 255.0) as u8
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

/// Scale values from 0 to 1.
fn unit_normalise(data: &[f32]) -> Vec<f32> {
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    data.iter().map(|&x| (x - min) / (max - min)).collect()
}

/// Scale values from 0 to 1 with an exponential factor. Can be useful to counteract heatmaps that
/// have too much of one colour in them.
fn exponential_normalise(data: &[f32]) -> Vec<f32> {
    let factor = 0.5;
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    data.iter().map(|&x| (x / max).powf(factor)).collect()
}

/// Redistribute values from 0 to 1 such that the distribution is centered.
/// This might help with heatmaps that are overly dark or bright.
///
/// GPT-5 helped with this.
fn welford_normalise(data: &[f32]) -> Vec<f32> {
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "It's just a vector length"
    )]
    let total = data.len() as f32;

    // Compute mean and population variance using Welford's algorithm.
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (index, &value) in data.iter().enumerate() {
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "Just incrementing by 1"
        )]
        let count = index as f32 + 1.0;
        let delta = value - mean;
        mean += delta / count;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }
    let variance = m2 / total;
    let standard_deviation = variance.sqrt().max(f32::EPSILON);

    let z_score: Vec<f32> = data
        .iter()
        .map(|&x| (x - mean) / standard_deviation)
        .collect();

    let (mut min_zscore, mut max_zscore) = (f32::INFINITY, f32::NEG_INFINITY);
    for &candidate in &z_score {
        if candidate < min_zscore {
            min_zscore = candidate;
        }
        if candidate > max_zscore {
            max_zscore = candidate;
        }
    }

    // Scale from 0 to 1.
    let mut scaled: Vec<f32> = z_score
        .iter()
        .map(|&z| (z - min_zscore) / (max_zscore - min_zscore))
        .collect();

    // Enforce mean to be 0.5
    let mean_scaled = scaled.iter().sum::<f32>() / total;
    let shift = 0.5 - mean_scaled;
    for value in &mut scaled {
        *value = (*value + shift).clamp(0.0, 1.0);
    }

    // If the previous clampings overly changed the mean, then adjust proportionally.
    let final_mean = scaled.iter().sum::<f32>() / total;
    if (final_mean - 0.5).abs() > f32::EPSILON {
        let diff = 0.5 - final_mean;
        for value in &mut scaled {
            *value = (*value + diff).clamp(0.0, 1.0);
        }
    }

    scaled
}

#[expect(clippy::unreadable_literal, reason = "These are just tests")]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn welford_normalisation() {
        let values = vec![1.0, 120.2, 40.9, 91.0];
        let normalised = welford_normalise(&values);
        assert_eq!(normalised, vec![0.0, 0.9719484, 0.30667996, 0.726982]);
    }
}
