//! Total Viewshed Calculator
#![feature(portable_simd)]
#![feature(specialization)]
#![feature(mpmc_channel)]
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
use std::mem;
use tracing_subscriber::{Layer as _, layer::SubscriberExt as _, util::SubscriberInitExt as _};

/// Handling the running of computations.
mod run {
    pub mod compute;
    pub mod parallel;
    pub mod serial;
}
mod config;
/// cpu implements a CPU kernel for the longest line of sight
mod cpu {
    pub mod area_of_interest;

    /// los contains all the traits necessary for implementing a line of sight algorithm
    mod los;

    mod rotation;
    mod rotator;

    /// Database for viewsheds
    pub mod storage {
        pub mod db;
        pub mod engine;
        pub mod metadata;
        pub mod segments;
        pub mod worker;
    }

    /// kernel is the exported kernel module
    pub mod kernel;

    /// `unrolled_los` holds a fully implemented los trait for unrolled vectorization
    mod unrolled_los;

    /// `vector_intrinsics` holds all the vector-related LOS intrinsics
    mod vector_intrinsics;

    pub use kernel::kernel;
}
mod dem;
mod dump_usage;
mod los_pack;
mod post_process;
mod tile;
mod vulkan;
/// Various ways to output data.
mod output {
    pub mod ascii;
    pub mod bresenham;
    pub mod png;
    pub mod ring_data;
    pub mod tiff;
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
                let geo_coord = projection::LonLatCoord(
                    geo::coord! {x: f64::from(coordinate.0), y: f64::from(coordinate.1)},
                );

                let is_sqlite = rusqlite::Connection::open(viewshed_config.db_path.clone()).is_ok();
                let source = if is_sqlite {
                    output::ring_data::Source::SQLite(viewshed_config.db_path.clone())
                } else {
                    output::ring_data::Source::Directory(viewshed_config.db_path.clone())
                };

                let viewshed = crate::output::viewshed::Viewshed::reconstruct(&source, geo_coord)?;
                crate::output::viewshed::Reconstructor::save(
                    viewshed,
                    &viewshed_config.output_dir,
                    geo_coord,
                )?;
            }
        }
        config::Commands::PostProcess(post_process_config) => {
            post_process::run(post_process_config)?;
        }
        config::Commands::DumpUsage => dump_usage::dump_full_usage_for_readme()?,
    }

    Ok(())
}

/// Setup logging.
fn setup_logging() -> Result<()> {
    let filters = tracing_subscriber::EnvFilter::builder()
        .with_default_directive("total_viewsheds=debug".parse()?)
        .from_env_lossy();
    let filter_layer = tracing_subscriber::fmt::layer().with_filter(filters);
    let tracing_setup = tracing_subscriber::registry().with(filter_layer);
    tracing_setup.init();

    Ok(())
}

/// Run computations
fn compute(config: &config::Compute) -> Result<()> {
    if !config.output_dir.exists() {
        std::fs::create_dir_all(&config.output_dir)?;
    }

    let mut tile = tile::Tile::load(config)?;

    let max_line_of_sight_as_points = tile.width.div_euclid(3);

    let mut dem = crate::dem::DEM::new(
        tile.centre,
        tile.width,
        tile.scale,
        max_line_of_sight_as_points,
    )?;

    dem.elevations = mem::take(&mut tile.data);

    tracing::debug!("Created DEM: {dem:?}");

    #[expect(
        clippy::as_conversions,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "Sign loss and truncation aren't relevant"
    )]
    let max_line_of_sight_as_metres = (max_line_of_sight_as_points as f32 * tile.scale) as u32;

    // Free up RAM
    drop(tile);

    let dem_metadata = crate::cpu::storage::metadata::MetaData {
        width: dem.width,
        scale: dem.scale,
        max_line_of_sight: max_line_of_sight_as_points,
        reserved_ring_size: run::compute::Compute::ring_count_per_band(
            config.rings_per_km,
            max_line_of_sight_as_metres,
        ),
        centre: dem.centre,
    };

    tracing::info!("Starting computations");
    let compute_config = run::compute::Config {
        observer_height: config.observer_height,
        backend: config.backend.clone(),
        process: config.process.clone(),
        output_directory: Some(config.output_dir.clone()),
        heatmap: config.heatmap,
        refraction: config.refraction,
        thread_count: config.thread_count,
        disable_render_image: config.disable_image_render,
        viewsheds_db_path: config.viewsheds_db_path.clone(),
        area_of_interest: crate::cpu::area_of_interest::Pruner::lonlat_coords_to_polygon(
            config.aoi_point.clone(),
            &dem_metadata,
        )?,
        dem_metadata,
        database_per_thread: config.database_per_thread,
    };

    let mut compute = run::compute::Compute::new(compute_config, &mut dem)?;
    compute.run()?;
    Ok(())
}
