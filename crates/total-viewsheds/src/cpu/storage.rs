use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

/// `Segment` is the rho portion of a line segment in polar coordinates
/// as (`rho`: u16, `delta_rho`: u16) which are packed into a single u32 for storage
#[derive(Clone)]
pub struct Segment(u32);

impl Segment {
    /// `new` creates a `Segment` the segment's start point and the distance
    fn new(start: u16, distance: u16) -> Self {
        // pack start/distsance into a u32 in the format of (start|distance)
        let wide_start: u32 = start.into();
        let wide_distance: u32 = distance.into();
        Self((wide_start << 16) | wide_distance)
    }

    /// `start` returns the starting point of the `Segment`
    #[expect(
        clippy::as_conversions,
        reason = "the top 16 bits are guaranteed to be 0"
    )]
    pub const fn start(&self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// `distance` returns the distance the `Segment` takes
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "the top 16 bits are guaranteed to be 0"
    )]
    pub const fn distance(&self) -> u16 {
        self.0 as u16
    }
}

/// `PolarSegments` holds the degree of a line of sight and the list
/// of visible `Segments` which is constucted through a Run Length
/// Encoding algorithm.
///
/// Each `tvs_id` will have at most 360 degrees.
#[derive(Clone)]
pub struct PolarSegments {
    /// `degree` is a whole degree in the range [0, 359]
    degree: u16,
    /// `visible_segments` is a list of segments visible for a given
    /// angle and tvs id
    visible_segments: Vec<Segment>,
}

impl PolarSegments {
    /// `from_bools` constructs an `PolarSegments` from a visibility bitmap.
    /// It does so by implementing a binary Run Lenght Encoding.
    #[expect(
        clippy::expect_used,
        reason = "We assume everything fits in a u16, we want a panic if it doesn't"
    )]
    pub fn from_bools(degree: u16, bitmap: &[bool]) -> Self {
        let mut visible_segments: Vec<Segment> = vec![];

        let mut cur_index = 0;
        while let Some(visible) = bitmap.get(cur_index) {
            if !visible {
                cur_index += 1;
                continue;
            }

            // it must be the case it is visible, scan the list
            // until we find the last visible
            let mut to_index = cur_index + 1;
            while let Some(next_visible) = bitmap.get(to_index) {
                if !next_visible {
                    break;
                }

                to_index += 1;
            }

            visible_segments.push(Segment::new(
                u16::try_from(cur_index).expect("bitmap too long!"),
                u16::try_from(to_index - cur_index).expect("distance in bitmap too long!"),
            ));
            cur_index = to_index;
        }

        Self {
            degree,
            visible_segments,
        }
    }
}

/// `Engine` is a thread-safe trait to store `PolarSegments` for a given `tvs_id`
pub trait Engine
where
    Self: Send + Sync,
{
    /// The
    fn store_segments(&self, tvs_id: u32, segments: PolarSegments);
}

/// `NoopEngine` stores no `PolarSegments`, it exists for testing purposes
pub struct NoopEngine;
impl Engine for NoopEngine {
    fn store_segments(&self, _tvs_id: u32, _segments: PolarSegments) {}
}

/// `SqliteEngine` stores all `PolarSegments` in a sqlite database. It uses
/// a channel to communicate with a worker thread to make sure sqlite writers are not
/// overwhelmed.
///
/// To save quite a bit in space, it encodes `PolarSegments` into their big-endian
/// representation of bytes and stores them as a sqlite BLOB. This lets it pack as
/// many segments together as possible and eliminates the need for database normalization.
///
/// The full schema looks like `CREATE TABLE segments(tvs_id INTEGER, angle_id INTEGER, visible_segments BLOB)`
///
/// Where `tvs_id` is a point's unique id, the `angle_id` is generally an int between [0, 360).
/// It is "generally" because the schema is intentionally left open so a user could interpret
/// a larger range of angle ids such as[0, 720) as subdivisions of the usual 360 degree structure.
///
/// Once it is dropped, it will clean up its worker thread and insert all segments in a single commit
///
/// Because of the large amount of data and need to optimize for quick writes t is not crash safe,
/// meaning if your program crashes the underlying sql database will likely be corrupted
pub struct SqliteEngine {
    /// `worker_handle` holds the `JoinHandle` for a worker thread which is reading
    /// from the Receiver end of `sender`'s channel.
    /// It uses an `Option` so that it can `take` the inner `JoinHandle` inside drop,
    /// meaning so long as `SqliteEngine` isn't being dropped it is `Some`
    worker_handle: Option<thread::JoinHandle<Result<(), rusqlite::Error>>>,
    /// `sender` is a mpsc channel that communicates segments to store to the `worker_handle`
    sender: mpsc::Sender<(u32, PolarSegments)>,
}

impl SqliteEngine {
    /// Create a new `SqliteEngine` storing the database at `path`
    fn new<P: AsRef<Path>>(path: P) -> Self {
        let (tx, rx) = mpsc::channel();

        // make an owned copy of Path so that it can be moved into the worker
        let db_path = PathBuf::from(path.as_ref());

        // move the receiver into the thread, leaving it as the only reference
        let handle = thread::spawn(move || storage_worker(db_path, rx));

        Self {
            worker_handle: Some(handle),
            sender: tx,
        }
    }
}

impl Engine for SqliteEngine {
    fn store_segments(&self, tvs_id: u32, segments: PolarSegments) {
        #[expect(
            clippy::expect_used,
            reason = "we should crash to make sure not to deadlock"
        )]
        self.sender
            .send((tvs_id, segments))
            .expect("channel sending error in sqlite engine");
    }
}

impl Drop for SqliteEngine {
    fn drop(&mut self) {
        // This is hacky, I am sorry.

        // Currently the thread inside `worker_handle` is waiting for messages
        // from its `Receiver`, but we still have our `Sender` inside `SqliteEngine`.
        // It hasn't been dropped yet  which prevents the worker's `Receiver` from being closed.
        //
        // We cannot drop the self.sender, even inside `drop`, because it violates Rust's memory
        // safety guarantees.
        //
        // To solve this we initialize an empty channel, use `std::mem::replace` to replace
        // `self.sender` with a local Sender which moves the "live" Sender out of the struct,
        // to give us a `&mut Sender` which we can then drop.
        //
        // Once the sender is dropped the thread will be unblocked and complete, and we
        // `join` it to make sure the thread runs its cleanup code (mainly committing open transactions)
        let (tx, _) = mpsc::channel();
        let orig = std::mem::replace(&mut self.sender, tx);
        drop(orig);

        // deadlock: joining in a drop could deadlock, but does not in our case
        //
        // Suppose the parallel threads writing to the channel encounter a non-storage error
        // and panic. The worker thread depends on the parallel threads to `recv()`. However, because
        // they panic and this `drop` is called the `Sender` is dropped unblocking the worker.
        //
        // Suppose the worker thread encounters an error. All parallel threads use a single `Sender`
        // which will unblock and close when the worker thread drops its `Receiver`. Since the worker
        // thread will `return`, the `Receiver` is dropped when the error is encountered. All parallel
        // threads will then get a `SendErr` when trying to send and will be unblocked
        #[expect(
            clippy::expect_used,
            reason = "we really should crash under any of these conditions"
        )]
        self.worker_handle
            .take()
            .expect("no one should have taken this option")
            .join()
            .expect("error joining thread")
            .expect("sqlite error");

        // our newly constructed sender and our join handle will be dropped here, as part of Rust's
        // usual drop process
    }
}

/// Storage is an wrapper for a storage `Engine`. It converts bitmaps
/// from the kernel into a `PolarSegments` to communicate
/// Keeping it a concrete struct lets us hide the underlying engine from the user
/// but gives us flexibility to NOOP storage for testing purposes
pub struct Storage {
    /// `engine` is the underlying storage engine for our `Storage` struct.
    /// We keep it `Box<dyn>` so that Storage doesn't have a generic type parameter
    /// because:
    /// a) The performance penalty of a pointer is negligible if we're already gong to disk
    /// b) Making Storage generic in terms of an Engine causes trait resolution issues (todo: @ryan-berger)
    engine: Box<dyn Engine>,
}

impl Storage {
    /// `new_noop` initializes a Storage with a dummy engine for testing
    pub fn new_noop() -> Self {
        Self {
            engine: Box::new(NoopEngine),
        }
    }

    /// `new` creates a new Storage with Sqlite as its backing when
    /// the `test` or `ring_data` feature is enabled, otherwise returning a noop
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        if !cfg!(any(test, feature = "ring_data")) {
            return Self {
                engine: Box::new(NoopEngine),
            };
        }

        Self {
            engine: Box::new(SqliteEngine::new(path)),
        }
    }

    /// `store_bitmap` converts a bitmap into `PolarSegments` and uses its `Engine` to store it
    pub fn store_bitmap(&self, tvs_id: u32, angle: u16, bitmap: &[bool]) {
        self.engine
            .store_segments(tvs_id, PolarSegments::from_bools(angle, bitmap));
    }
}

/// `storage_worker` creates a new sqlite database with the `SqliteEngine`'s schema.
/// It then turns off synchronous writes and puts the journal mode to in-memory
/// to allow for quick writes.
///
/// All segments sent to `recv` will be run in the same transaction to save on transaction
/// overhead. This means that any panic or error will end in a corrupted database
fn storage_worker<P: AsRef<Path>>(
    path: P,
    recv: mpsc::Receiver<(u32, PolarSegments)>,
) -> Result<(), rusqlite::Error> {
    let mut conn = Connection::open(path)?;

    conn.execute(
        "
        CREATE TABLE IF NOT EXISTS polar_segments (
            tvs_id INTEGER,
            angle_id INTEGER,
            visible_segments BLOB
        )",
        (),
    )?;
    conn.pragma_update(None, "synchronous", "OFF")?;
    conn.pragma_update(None, "journal_mode", "MEMORY")?;

    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT INTO polar_segments(tvs_id, angle_id, visible_segments) VALUES (?1, ?2, ?3)",
        )?;

        for (tvs_id, angle) in recv {
            #[expect(clippy::big_endian_bytes, reason = "it is documented in the format")]
            let vec_bytes = angle
                .visible_segments
                .iter()
                .flat_map(|vector| vector.0.to_be_bytes())
                .collect::<Vec<_>>();

            stmt.execute((&tvs_id, &angle.degree, &vec_bytes))?;
        }
    }

    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::cpu::storage::PolarSegments;

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "testing, slicing/panicking is okay"
    )]
    fn bitmap_to_angle() {
        {
            let test_visibility = vec![true, true, true, false, true];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 2);
            assert_eq!(angles.visible_segments[0].distance(), 3);
            assert_eq!(angles.visible_segments[1].distance(), 1);
        };

        {
            let test_visibility = vec![true, false, false];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 1);
            assert_eq!(angles.visible_segments[0].distance(), 1);
            assert_eq!(angles.visible_segments[0].start(), 0);
        };

        {
            let test_visibility = vec![false, true, true, true, true, false];
            let angles = PolarSegments::from_bools(0, &test_visibility);
            assert_eq!(angles.visible_segments.len(), 1);
            assert_eq!(angles.visible_segments[0].distance(), 4);
        }
    }
}
