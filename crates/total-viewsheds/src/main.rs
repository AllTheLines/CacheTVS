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
        clippy::indexing_slicing,
        reason = "It's just for the tests"
    )
)]

extern crate core;

use clap::Parser as _;
use color_eyre::eyre::{Result, bail};
use std::mem;
use tracing_subscriber::{Layer as _, layer::SubscriberExt as _, util::SubscriberInitExt as _};

mod config;
mod dump_usage;
mod los_pack;
mod post_process;
mod pre_process;
mod run;
mod tile;
mod workers;

/// cpu implements a CPU kernel for the longest line of sight
mod compute {
    pub mod area_of_interest;

    /// los contains all the traits necessary for implementing a line of sight algorithm
    mod los;

    mod rotation;
    mod rotator;

    /// kernel is the exported kernel module
    pub mod kernel;

    /// `unrolled_los` holds a fully implemented los trait for unrolled vectorization
    mod unrolled_los;

    /// `vector_intrinsics` holds all the vector-related LOS intrinsics
    mod vector_intrinsics;

    pub use kernel::kernel;
}

/// Database for viewsheds
mod storage {
    pub mod db;
    pub mod engine;
    pub mod segments;
    pub mod worker;
}

/// Various ways to output data.
mod output {
    pub mod ascii;
    pub mod bresenham;
    pub mod png;
    pub mod tiff;

    /// Load, parse and reconstruct euclidean polygon viewsheds from their raw polar segments.
    pub mod viewsheds {
        pub mod growable_polygon;
        pub mod joiner_common;
        pub mod joiner_final;
        pub mod segment_polygon;
        pub mod viewshed;
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    setup_logging()?;
    let config = crate::config::Config::parse();
    tracing::info!("Initialising with config: {config:?}",);

    match &config.command {
        config::Commands::Compute(compute_config) => compute(compute_config)?,
        config::Commands::Viewshed(viewshed_config) => {
            for coordinate in &viewshed_config.coordinates {
                let geo_coord = tvs_lib::projector::LonLatCoord(
                    geo::coord! {x: f64::from(coordinate.0), y: f64::from(coordinate.1)},
                );

                let (_, viewshed) = crate::output::viewsheds::viewshed::Viewshed::reconstruct(
                    viewshed_config.db_path.clone(),
                    geo_coord,
                )?;
                crate::output::viewsheds::viewshed::Viewshed::save(
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

    // Technically angle_subdivisions must be within 3 subdivisions for the
    // longest line of sight packing to work, but if you aren't intrerested
    // in the longest line this works just fine. TODO: @ryan-berger rethink
    // the type design
    if config.angle_subdivisions > 100 || config.angle_subdivisions == 0 {
        color_eyre::eyre::bail!("Angle subdivisions must be between (0, 100]")
    }

    let mut tile = tile::Tile::load(config)?;

    let max_line_of_sight_as_points = tile.width.div_euclid(3);

    let mut dem = tvs_lib::dem::DEM::new(
        tile.centre,
        tile.width,
        tile.scale,
        max_line_of_sight_as_points,
    )?;

    dem.elevations = mem::take(&mut tile.data);

    tracing::debug!("Created DEM: {dem:?}");

    // Free up RAM
    drop(tile);

    let dem_metadata = tvs_lib::metadata::MetaData {
        width: dem.width,
        scale: dem.scale,
        max_line_of_sight: max_line_of_sight_as_points,
        centre: dem.centre,
        neighbourhood_size: config.only_save_biggest_viewsheds.unwrap_or(0).into(),
        angle_subdivisions: config.angle_subdivisions.into(),
    };

    let save_viewshed_dem_ids = if config.only_save_biggest_viewsheds.is_some() {
        if config.tvs_source_path.is_none() {
            bail!("Must provide --tvs_source_path argument");
        }

        Some(pre_process::create_biggest_tvs_subgrid(config)?)
    } else {
        None
    };

    tracing::info!("Starting computations");
    let compute_config = run::Config {
        observer_height: config.observer_height,
        process: config.process.clone(),
        output_directory: Some(config.output_dir.clone()),
        heatmap: config.heatmap,
        refraction: config.refraction,
        thread_count: config.thread_count,
        disable_render_image: config.disable_image_render,
        viewsheds_db_path: config.viewsheds_db_path.clone(),
        area_of_interest: crate::compute::area_of_interest::Pruner::lonlat_coords_to_polygon(
            config.aoi_point.clone(),
            &dem_metadata,
        )?,
        dem_metadata,
        database_per_thread: config.database_per_thread,
        viewsheds_to_save: save_viewshed_dem_ids,
        angle_subdivisions: config.angle_subdivisions,
    };

    let mut compute = run::Compute::new(compute_config, &mut dem)?;
    compute.run()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    pub mod fixtures;
}
