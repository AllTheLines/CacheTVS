//! Conventional API for the DB. See `worker.rs` for how to write to the DB from the kernel.

use color_eyre::Result;

/// Sqlite DB connection details.
pub struct DB {
    /// A connection to Sqlite
    connection: rusqlite::Connection,
}

impl DB {
    /// Instantitate.
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        let connection = rusqlite::Connection::open(db_path)?;
        Ok(Self { connection })
    }

    /// Save the metadata.
    pub fn save_metadata(&self, metadata: &super::metadata::MetaData) -> Result<()> {
        tracing::debug!("Saving metadata to {:?}...", self.connection.path());

        self.connection.execute(
            "
            CREATE TABLE IF NOT EXISTS metadata (
                json TEXT NOT NULL
            )",
            (),
        )?;

        let json = serde_json::to_string_pretty(&metadata)?;
        self.connection.execute(
            "INSERT INTO metadata (json) VALUES (?1)",
            rusqlite::params![json],
        )?;
        tracing::info!("Saved metadata: {metadata:?}");

        Ok(())
    }

    /// Load the metadata.
    pub fn load_metadata(&self) -> Result<super::metadata::MetaData> {
        tracing::debug!("Loading metadata from {:?}...", self.connection.path());

        let metadata_string: String =
            self.connection
                .query_row("SELECT json FROM metadata", [], |row| row.get(0))?;
        let metadata: super::metadata::MetaData = serde_json::from_str(&metadata_string)?;
        tracing::info!("Loaded metadata: {metadata:?}");

        Ok(metadata)
    }

    /// Load all the polar segments for a given DEM ID.
    pub fn load_segments_for_tvs_id(
        &self,
        tvs_id: u32,
    ) -> Result<Vec<Vec<super::segments::Segment>>> {
        tracing::debug!(
            "Loading polar segments for {tvs_id} from {:?}...",
            self.connection.path()
        );

        // We use `MIN(...)` because it's possible that for any given DEM ID and angle, there can
        // be mulitple records. This is because of how DEM rotation is quantised to the resolution
        // of the grid. It is assumed that each of these duplicates have approximately the same
        // viewshed segements.
        let mut statement = self.connection.prepare(
            "
            SELECT MIN(visible_segments) AS visible_segments
            FROM polar_segments
            WHERE dem_id = ?1
            GROUP BY angle_id
            ORDER BY angle_id ASC;
            ",
        )?;
        let rows = statement.query_map([tvs_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        })?;

        let mut segments = Vec::new();
        for row in rows {
            segments.push(Self::bytes_to_segments(&row?)?);
        }

        Ok(segments)
    }

    /// Convert blob to `Segment`s.
    fn bytes_to_segments(bytes: &[u8]) -> Result<Vec<super::segments::Segment>> {
        let mut out = Vec::with_capacity(bytes.len().div_euclid(4));
        for chunk in bytes.chunks(4) {
            let array: [u8; 4] = chunk.try_into()?;
            #[expect(clippy::big_endian_bytes, reason = "That's how we save them in the DB")]
            out.push(super::segments::Segment(u32::from_be_bytes(array)));
        }
        Ok(out)
    }
}
