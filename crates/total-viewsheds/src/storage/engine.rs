//! Threaded, channel-communicated DB connection for safely saving viewshed data from the hot loop
//! of the kernel.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

/// `Engine` is a thread-safe trait to store `PolarSegments` for a given `tvs_id`
pub(crate) trait Engine
where
    Self: Send + Sync,
{
    /// The
    fn store_segments(
        &self,
        dem_id: u64,
        neighbourhood_id: Option<i64>,
        segments: super::segments::PolarSegments,
    );
}

/// `NoopEngine` stores no `PolarSegments`, it exists for testing purposes
pub(crate) struct Noop;
impl Engine for Noop {
    fn store_segments(
        &self,
        _dem_id: u64,
        _neighbourhood_id: Option<i64>,
        _segments: super::segments::PolarSegments,
    ) {
    }
}

/// Stores all `PolarSegments` in a sqlite database. It uses a channel to communicate with a worker thread
/// to make sure sqlite writers are not overwhelmed.
///
/// To save quite a bit in space, it encodes `PolarSegments` into their big-endian
/// representation of bytes and stores them as a sqlite BLOB. This lets it pack as
/// many segments together as possible and eliminates the need for database normalization.
///
/// The full schema looks like:
///   `CREATE TABLE segments(dem_id INTEGER, angle_id INTEGER, visible_segments BLOB)`
/// Where `dem_id` is a point's unique id, the `angle_id` is generally an int between [0, 360).
/// It is "generally" because the schema is intentionally left open so a user could interpret
/// a larger range of angle ids such as[0, 720) as subdivisions of the usual 360 degree structure.
///
/// Once it is dropped, it will clean up its worker thread and insert all segments in a single commit
///
/// Because of the large amount of data and need to optimize for quick writes t is not crash safe,
/// meaning if your program crashes the underlying sql database will likely be corrupted
pub(crate) struct Sqlite {
    /// `worker_handle` holds the `JoinHandle` for a worker thread which is reading
    /// from the Receiver end of `sender`'s channel.
    /// It uses an `Option` so that it can `take` the inner `JoinHandle` inside drop,
    /// meaning so long as `SqliteEngine` isn't being dropped it is `Some`
    worker_handle: Option<thread::JoinHandle<Result<(), rusqlite::Error>>>,
    /// `sender` is a mpsc channel that communicates segments to store to the `worker_handle`
    sender: mpsc::SyncSender<(u64, Option<i64>, super::segments::PolarSegments)>,
}

impl Sqlite {
    /// Create a new `SqliteEngine` storing the database at `path`
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> Self {
        let (tx, rx) = mpsc::sync_channel(1024);

        // make an owned copy of Path so that it can be moved into the worker
        let db_path = PathBuf::from(path.as_ref());

        // move the receiver into the thread, leaving it as the only reference
        let handle = thread::spawn(move || super::worker::writer(db_path, rx));

        Self {
            worker_handle: Some(handle),
            sender: tx,
        }
    }
}

impl Engine for Sqlite {
    fn store_segments(
        &self,
        dem_id: u64,
        neighbourhood_id: Option<i64>,
        segments: super::segments::PolarSegments,
    ) {
        #[expect(
            clippy::expect_used,
            reason = "we should crash to make sure not to deadlock"
        )]
        self.sender
            .send((dem_id, neighbourhood_id, segments))
            .expect("channel sending error in sqlite engine");
    }
}

impl Drop for Sqlite {
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
        let (tx, _) = mpsc::sync_channel(1);
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
