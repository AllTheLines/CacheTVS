//! Post-processing work after doing a compute run to generate viewshed databases.

use color_eyre::Result;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator as _};

/// Run post-processing tasks.
pub fn run(config: &crate::config::PostProcess) -> Result<()> {
    #[expect(
        clippy::redundant_closure_for_method_calls,
        reason = "It's too verbose"
    )]
    let paths: Vec<_> = std::fs::read_dir(&config.db_dir)?
        .filter_map(|result| result.ok())
        .map(|file| file.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "db"))
        .collect();

    tracing::info!(
        "Found {} shards. Starting parallel indexing...",
        paths.len()
    );

    if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    paths.par_iter().for_each(|path| {
        if let Err(error) = index_one_shard(path) {
            tracing::error!("Error indexing {:?}: {}", path.file_name(), error);
        } else {
            tracing::info!("Finished: {:?}", path.file_name());
        }
    });

    tracing::info!("Indexing complete.");
    Ok(())
}

/// Carry out indexing and cleanup for a single database.
fn index_one_shard(path: &std::path::Path) -> rusqlite::Result<()> {
    let conn = rusqlite::Connection::open(path)?;

    conn.execute_batch(
        "
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        ",
    )?;

    tracing::debug!("Creating indexes for {path:?}...");
    conn.execute(
        "CREATE INDEX IF NOT EXISTS dem_id__index ON polar_segments(dem_id)",
        [],
    )?;

    tracing::debug!("Analyzing {path:?}...");
    conn.execute("ANALYZE", [])?;

    // Why we don't VACCUM:
    // It will attempt to copy the newly indexed database over to a new file, rebuilding the indexes
    // and re-packing the data in the process. This is overzelous for our needs, so we do not need it.
    // This also allows us to need less special PRAGMAs used for copying, because we aren't copying.

    Ok(())
}
