//! Reconstruct viewsheds from polar segments.
//!
//! You could just use

#[cfg(test)]
use tracing_subscriber::{Layer as _, layer::SubscriberExt as _, util::SubscriberInitExt as _};

mod growable_polygon;
pub mod joiner;
pub mod polygon;
pub mod segment;
mod segment_polygon;
mod vertices;

/// Setup logging.
#[cfg(test)]
pub(crate) fn setup_logging() {
    let filters = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(
            "total_viewsheds=debug"
                .parse()
                .expect("Couldn't parse log ENV filter"),
        )
        .from_env_lossy();
    let filter_layer = tracing_subscriber::fmt::layer().with_filter(filters);
    let tracing_setup = tracing_subscriber::registry().with(filter_layer);
    tracing_setup.init();
}
