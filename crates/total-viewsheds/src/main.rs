//! Total Viewshed Calculator

#![expect(clippy::pub_use, reason = "I admit I don't understand the other way.")]

use clap::Parser as _;
use color_eyre::eyre::{ContextCompat as _, Result};
use tracing_subscriber::{layer::SubscriberExt as _, util::SubscriberInitExt as _, Layer as _};

mod axes;
mod band_of_sight;
/// The `.bt` file type for reading and writing the data we consume and output.
mod bt {
    pub mod header;
    pub use header::BinaryTerrain;
    pub mod read;
    pub mod write;
}
mod cache;
mod compute;
mod config;
mod dem;
mod dump_usage;
mod vulkan;
/// Various ways to output data.
mod output {
    pub mod ascii;
    pub mod bt;
    pub mod png;
    pub mod ring_data;
    pub mod viewshed;
}
mod projection;

fn main() -> Result<()> {
    color_eyre::install()?;
    setup_logging()?;
    let config = crate::config::Config::parse();
    tracing::info!("Initialising with config: {config:?}",);

    match &config.command {
        config::Commands::Compute(compute_config) => compute(compute_config)?,
        config::Commands::Viewshed(viewshed_config) => {
            for coordinate in &viewshed_config.coordinates {
                let geo_coord = projection::LatLonCoord(
                    geo::coord! {x: f64::from(coordinate.0), y: f64::from(coordinate.1)},
                );
                let viewshed = crate::output::viewshed::Viewshed::reconstruct(
                    &output::ring_data::Source::Directory(viewshed_config.output_dir.clone()),
                    geo_coord,
                )?;
                crate::output::viewshed::Reconstructor::save(
                    viewshed,
                    &viewshed_config.output_dir,
                    geo_coord,
                )?;
            }
        }
        config::Commands::DumpUsage => dump_usage::dump_full_usage_for_readme()?,
    }

    Ok(())
}

/// Setup logging.
fn setup_logging() -> Result<()> {
    let filters = tracing_subscriber::EnvFilter::builder()
        .with_default_directive("total_viewsheds=info".parse()?)
        .from_env_lossy();
    let filter_layer = tracing_subscriber::fmt::layer().with_filter(filters);
    let tracing_setup = tracing_subscriber::registry().with(filter_layer);
    tracing_setup.init();

    Ok(())
}

/// Run computations
fn compute(config: &config::Compute) -> Result<()> {
    let tile = bt::BinaryTerrain::read(&config.input)?;
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

    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "I don't think there's any other way"
    )]
    let mut dem = crate::dem::DEM::new(
        tile.centre(),
        tile.header.width,
        scale as f32,
        max_line_of_sight,
    )?;

    tracing::info!("Converting DEM data to `f32`");
    match &tile.data {
        bt::header::Data::Int16(points) => {
            dem.elevations = points.iter().map(|point| f32::from(*point)).collect();
        }
        bt::header::Data::Float32(points) => dem.elevations.clone_from(points),
    }

    // Free up RAM
    drop(tile);

    tracing::debug!("Created DEM: {dem:?}");

    tracing::info!("Starting computations");
    let mut compute = crate::compute::Compute::new(
        config.backend.clone(),
        config.process.clone(),
        Some(dirs::state_dir().context("Couldn't get the OS's state directory")?),
        Some(config.output_dir.clone()),
        &mut dem,
        config.rings_per_km,
        config.observer_height,
    )?;
    compute.run()?;
    Ok(())
}
