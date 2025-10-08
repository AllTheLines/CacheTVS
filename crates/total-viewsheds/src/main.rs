//! Total Viewshed Calculator

use clap::Parser as _;
use color_eyre::eyre::{ContextCompat as _, Result};
use tracing_subscriber::{layer::SubscriberExt as _, util::SubscriberInitExt as _, Layer as _};

mod axes;
mod band_of_sight;
mod cache;
mod compute;
mod config;
mod dem;
mod gpu;
mod input;

/// Various ways to output data.
mod output {
    pub mod ascii;
    pub mod png;
}

fn main() -> Result<()> {
    color_eyre::install()?;
    setup_logging()?;

    let mut config = crate::config::Config::parse();

    let tile = input::BinaryTerrain::read(&config.input)?;
    let scale = config.scale.unwrap_or_else(|| tile.scale());

    #[expect(
        clippy::as_conversions,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Sign loss and truncation aren't relevant"
    )]
    let max_line_of_sight = config
        .max_line_of_sight
        .unwrap_or_else(|| (f64::from(tile.header.width.div_euclid(3)) * scale) as u32);

    config.max_line_of_sight = Some(max_line_of_sight);

    tracing::info!("Initialising with config: {config:?}");

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "This is the only `std` way to cast `f64` to `f32`"
    )]
    let mut dem = crate::dem::DEM::new(tile.header.width, scale as f32, max_line_of_sight)?;
    tracing::info!("Converting DEM data to `f32`");
    match &tile.data {
        input::DataType::Int16(points) => {
            dem.elevations = points.iter().map(|point| f32::from(*point)).collect();
        }
        input::DataType::Float32(points) => dem.elevations.clone_from(points),
    }

    drop(tile);

    tracing::info!("Starting computations");
    let mut compute = crate::compute::Compute::new(
        config.compute,
        Some(dirs::state_dir().context("Couldn't get the OS's state directory")?),
        Some(config.output_dir),
        &mut dem,
        config.rings_per_km,
    )?;
    compute.run()?;

    Ok(())
}

/// Setup logging.
fn setup_logging() -> Result<()> {
    let filters = tracing_subscriber::EnvFilter::builder()
        .with_default_directive("info".parse()?)
        .from_env_lossy();
    let filter_layer = tracing_subscriber::fmt::layer().with_filter(filters);
    let tracing_setup = tracing_subscriber::registry().with(filter_layer);
    tracing_setup.init();

    Ok(())
}
