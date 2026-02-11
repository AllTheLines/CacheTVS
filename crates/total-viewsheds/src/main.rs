//! Total Viewshed Calculator
#![feature(portable_simd)]
#![feature(specialization)]
#![expect(
    incomplete_features,
    reason = "our usage isn't crazy and unlikely to break"
)]
#![feature(generic_const_exprs)]
#![expect(clippy::pub_use, reason = "I admit I don't understand the other way.")]
#![cfg_attr(
    test,
    expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::unreadable_literal,
        clippy::default_numeric_fallback,
        clippy::integer_division,
        clippy::integer_division_remainder_used,
        reason = "It's just for the tests"
    )
)]

extern crate core;

use clap::Parser as _;
use color_eyre::eyre::Result;
use tracing_subscriber::{layer::SubscriberExt as _, util::SubscriberInitExt as _, Layer as _};

/// The `.bt` file type for reading and writing the data we consume and output.
mod bt {
    pub mod header;
    pub use header::BinaryTerrain;
    pub mod read;
    pub mod write;
}
/// Handling the running of computations.
mod run {
    pub mod compute;
    pub mod parallel;
    pub mod serial;
}
mod config;
mod dem;
mod dump_usage;
mod los_pack;
mod vulkan;
/// Various ways to output data.
mod output {
    pub mod ascii;
    pub mod bresenham;
    pub mod bt;
    pub mod png;
    pub mod ring_data;
    pub mod viewshed;
}

/// cpu implements a CPU kernel for the longest line of sight
mod cpu;

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
    // for now, only the CPU backend knows how to interpret more than one angle division
    if !matches!(config.backend, config::Backend::CPU) && config.angle_subdivisions > 1 {
        color_eyre::eyre::bail!(
            "An angle division higher than 1 is only supported on the `cpu` backend"
        )
    }

    // we can't subdivide rotation into more than 182 rotations or else we overflow the u16
    // we use for angles, so 100 is an artificial limit
    if config.angle_subdivisions > 100 {
        color_eyre::eyre::bail!("The maximum angle divisions is 100")
    }

    let tile = bt::BinaryTerrain::read(&config.input)?;
    let scale = config.scale.unwrap_or_else(|| tile.scale());

    #[expect(
        clippy::as_conversions,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Sign loss and truncation aren't relevant"
    )]
    let max_line_of_sight = (tile.header.width.div_euclid(3) as f32 * scale) as u32;

    let mut dem = crate::dem::DEM::new(tile.centre(), tile.header.width, scale, max_line_of_sight)?;

    tracing::info!("Converting DEM data to `f32`");
    match &tile.data {
        bt::header::Data::Int16(points) => dem.elevations.clone_from(points),
        bt::header::Data::Float32(_) => {
            color_eyre::eyre::bail!("Float `.bt` files aren't supported yet.")
        }
    }

    // Free up RAM
    drop(tile);

    tracing::debug!("Created DEM: {dem:?}");

    tracing::info!("Starting computations");
    let compute_config = run::compute::Config {
        observer_height: config.observer_height,
        scale: config.scale.unwrap_or(1.0),
        backend: config.backend.clone(),
        process: config.process.clone(),
        output_directory: Some(config.output_dir.clone()),
        rings_per_km: config.rings_per_km,
        heatmap: config.heatmap,
        refraction: config.refraction,
        thread_count: config.thread_count,
        disable_render_image: config.disable_image_render,
        angle_subdivisions: config.angle_subdivisions,
    };
    let mut compute = run::compute::Compute::new(compute_config, &mut dem)?;
    compute.run()?;
    Ok(())
}
