//! Conventional API for the DB. See `worker.rs` for how to write to the DB from the kernel.

use color_eyre::Result;
use std::path::Path;

/// How we look up polar segments in the database.
#[derive(Debug)]
pub enum ID {
    /// The precise DEM ID of the polar segment.
    DEM(i64),
    /// The neighbourhood ID within which a single DEM ID has been isolated.
    Neighbourhood(i64),
}

/// Sqlite DB connection details.
pub struct DB {
    /// A connection to Sqlite
    connection: rusqlite::Connection,
}

impl DB {
    /// Instantitate.
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
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
            )
            ",
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

    /// Load all the polar segments for a given ID.
    pub fn load_segments(&self, id: &ID) -> Result<(Vec<Vec<super::segments::Segment>>, i64)> {
        tracing::debug!(
            "Loading polar segments for {id:?} from {:?}...",
            self.connection.path()
        );

        let (field, id_value) = match id {
            ID::DEM(value) => ("dem_id", value),
            ID::Neighbourhood(value) => ("neighbourhood_id", value),
        };

        let statement = format!(
            "
            SELECT visible_segments
            FROM polar_segments
            WHERE {field} = ?1
            GROUP BY angle_id
            ORDER BY angle_id ASC;
            "
        );

        let mut prepared = self.connection.prepare(&statement)?;
        let rows = prepared.query_map([id_value], |row| {
            let blob: Vec<u8> = row.get(0)?;
            Ok(blob)
        })?;

        let mut segments = Vec::new();
        for row in rows {
            segments.push(Self::bytes_to_segments(&row?)?);
        }

        let dem_id = match id {
            ID::DEM(dem_id) => *dem_id,
            ID::Neighbourhood(neighbourhood_id) => self
                .connection
                .prepare(
                    "
                    SELECT dem_id FROM polar_segments
                    WHERE neighbourhood_id = ?1
                    LIMIT 1
                    ",
                )?
                .query_row([neighbourhood_id], |row| {
                    let dem_id: i64 = row.get(0)?;
                    Ok(dem_id)
                })?,
        };

        Ok((segments, dem_id))
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
