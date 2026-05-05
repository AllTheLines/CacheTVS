//! A worker that receives messages over a channel from the running kernel, which then writes
//! viewshed data as binary blobs to a database.

/// A wrapper for a storage `Engine`. It converts bitmaps from the kernel into a `PolarSegments`
/// to communicate. Keeping it a concrete struct lets us hide the underlying engine from the user
/// but gives us flexibility to NOOP storage for testing purposes
pub struct Worker {
    /// `engine` is the underlying storage engine for our `Storage` struct.
    /// We keep it `Box<dyn>` so that Storage doesn't have a generic type parameter
    /// because:
    /// a) The performance penalty of a pointer is negligible if we're already gong to disk
    /// b) Making Storage generic in terms of an Engine causes trait resolution issues (todo: @ryan-berger)
    engine: Box<dyn super::engine::Engine>,
}

impl Worker {
    /// `new_noop` initializes a Worker with a dummy engine for testing
    pub fn new_noop() -> Self {
        Self {
            engine: Box::new(super::engine::Noop),
        }
    }

    /// `new` creates a new Worker with Sqlite as its backing when
    /// the `test` or `ring_data` feature is enabled, otherwise returning a noop
    pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
        if !cfg!(any(test, feature = "ring_data")) {
            return Self {
                engine: Box::new(crate::storage::engine::Noop),
            };
        }

        Self {
            engine: Box::new(crate::storage::engine::Sqlite::new(path)),
        }
    }

    /// `store_bitmap` converts a bitmap into `PolarSegments` and uses its `Engine` to store it
    pub fn store_bitmap(&self, dem_id: u64, angle: u16, bitmap: &[bool]) {
        self.engine.store_segments(
            dem_id,
            super::segments::PolarSegments::from_bools(angle, bitmap),
        );
    }
}

/// Creates a new sqlite database with the `engine::Sqlite`'s schema.
/// It then turns off synchronous writes and puts the journal mode to in-memory
/// to allow for quick writes.
///
/// All segments sent to `recv` will be run in the same transaction to save on transaction
/// overhead. This means that any panic or error will end in a corrupted database
pub fn writer<P: AsRef<std::path::Path>>(
    path: P,
    recv: std::sync::mpsc::Receiver<(u64, super::segments::PolarSegments)>,
) -> Result<(), rusqlite::Error> {
    let mut conn = rusqlite::Connection::open(path)?;

    conn.execute(
        "
        CREATE TABLE IF NOT EXISTS polar_segments (
            dem_id INTEGER,
            angle_id INTEGER,
            visible_segments BLOB
        )",
        (),
    )?;
    conn.pragma_update(None, "synchronous", "OFF")?;
    conn.pragma_update(None, "journal_mode", "OFF")?;

    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT INTO polar_segments(dem_id, angle_id, visible_segments) VALUES (?1, ?2, ?3)",
        )?;

        for (tvs_id, segments) in recv {
            #[expect(clippy::big_endian_bytes, reason = "it is documented in the format")]
            let vec_bytes = segments
                .visible_segments
                .iter()
                .flat_map(|vector| vector.0.to_be_bytes())
                .collect::<Vec<_>>();

            let params = (&tvs_id.cast_signed(), &segments.degree, &vec_bytes);
            stmt.execute(params)?;
        }
    }

    tx.commit()?;
    Ok(())
}
