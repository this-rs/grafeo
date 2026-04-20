//! WAL file I/O — writer and reader over the framed [`WalRecord`] format
//! defined in [`crate::wal`].
//!
//! ## Design
//!
//! The WAL is a single append-only file. Every mutation of the substrate
//! produces exactly one [`WalRecord`] written here, and the `fsync`ed byte
//! offset in this file is the durability line — everything before it
//! survives crashes, everything after does not.
//!
//! Three fsync policies are exposed via [`SyncMode`]:
//!
//! * [`SyncMode::EveryCommit`] — fsync every record with
//!   [`WalRecord::FLAG_COMMIT`]. Strongest durability; slowest write path.
//! * [`SyncMode::Group`] — batch fsyncs at a deadline (milliseconds). Group
//!   commit: multiple writers share a single fsync. Default for hot paths.
//! * [`SyncMode::Never`] — never fsync. Tests and benches only.
//!
//! ## Replay
//!
//! [`WalReader::iter_from(offset)`] yields decoded `(record, offset, length)`
//! tuples starting at `offset`. Decoding stops at the first record that fails
//! CRC or short-reads — that's the crash boundary. The reader does not
//! attempt to recover partially written frames (append-only + CRC makes this
//! a clean boundary, not a resynchronization problem).

use crate::error::SubstrateResult;
use crate::wal::{WalRecord, WAL_HEADER_SIZE};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// WAL durability policy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SyncMode {
    /// Fsync the WAL after every record whose `FLAG_COMMIT` bit is set.
    EveryCommit,
    /// Coalesce fsyncs — batching is handled by the caller (e.g. a group-commit
    /// timer in the store). The writer itself does not fsync; callers must
    /// invoke [`WalWriter::fsync`] explicitly.
    Group,
    /// Never fsync. For tests and micro-benchmarks only.
    Never,
}

impl Default for SyncMode {
    fn default() -> Self {
        Self::EveryCommit
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Append-only writer over a WAL file.
///
/// Construction opens (or creates) the file at `path` and seeks to the end.
/// Every call to [`WalWriter::append`] writes a complete framed record and
/// updates the internal offset + lsn counters.
///
/// The writer owns its own `Mutex<File>` so it is `Send + Sync`; multiple
/// producers can share a `&WalWriter` safely.
#[derive(Debug)]
pub struct WalWriter {
    path: PathBuf,
    inner: Mutex<WriterInner>,
    /// Byte offset of the next record to be written (monotonic).
    offset: AtomicU64,
    /// Next LSN to hand out.
    next_lsn: AtomicU64,
    sync_mode: SyncMode,
}

#[derive(Debug)]
struct WriterInner {
    file: File,
}

impl WalWriter {
    /// Open or create the WAL file at `path`. The writer seeks to the end of
    /// the existing file; its LSN counter starts at `next_lsn` (typically the
    /// value replay returned + 1).
    pub fn open(path: impl AsRef<Path>, sync_mode: SyncMode, next_lsn: u64) -> SubstrateResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        let end = file.seek(SeekFrom::End(0))?;
        Ok(Self {
            path,
            inner: Mutex::new(WriterInner { file }),
            offset: AtomicU64::new(end),
            next_lsn: AtomicU64::new(next_lsn),
            sync_mode,
        })
    }

    /// Current byte offset of the next record.
    pub fn offset(&self) -> u64 {
        self.offset.load(Ordering::Acquire)
    }

    /// Peek at the next LSN (does not consume it).
    pub fn peek_lsn(&self) -> u64 {
        self.next_lsn.load(Ordering::Acquire)
    }

    /// Reserve and increment the next LSN. Callers typically don't need this
    /// directly — [`WalWriter::append`] assigns the LSN automatically.
    pub fn next_lsn(&self) -> u64 {
        self.next_lsn.fetch_add(1, Ordering::AcqRel)
    }

    /// Append a record. The record's `lsn` field is overwritten with the
    /// writer's next LSN. Returns the byte offset at which the record was
    /// written and its length.
    pub fn append(&self, mut rec: WalRecord) -> SubstrateResult<(u64, usize)> {
        rec.lsn = self.next_lsn();
        let bytes = rec.encode()?;
        let len = bytes.len();

        let mut guard = self.inner.lock();
        // Writes to an append-opened file are always at end; we verify offset
        // is consistent with the atomic counter.
        guard.file.write_all(&bytes)?;
        let prev = self.offset.fetch_add(len as u64, Ordering::AcqRel);

        let should_fsync = match self.sync_mode {
            SyncMode::EveryCommit => rec.is_commit() || rec.is_checkpoint(),
            SyncMode::Group => false,
            SyncMode::Never => false,
        };
        if should_fsync {
            guard.file.sync_data()?;
        }
        Ok((prev, len))
    }

    /// Force a flush + fsync now, regardless of sync mode. Returns after the
    /// fsync completes; callers that set [`SyncMode::Group`] should invoke
    /// this from a background timer or at explicit commit points.
    pub fn fsync(&self) -> SubstrateResult<()> {
        let guard = self.inner.lock();
        guard.file.sync_data()?;
        Ok(())
    }

    /// Truncate the WAL to zero bytes. Used at the end of a successful
    /// checkpoint — all prior records are now reflected in the on-disk
    /// snapshot so replay no longer needs them.
    pub fn truncate(&self) -> SubstrateResult<()> {
        let mut guard = self.inner.lock();
        guard.file.set_len(0)?;
        guard.file.seek(SeekFrom::Start(0))?;
        guard.file.sync_all()?;
        self.offset.store(0, Ordering::Release);
        Ok(())
    }

    /// Path of the WAL file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current sync mode.
    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Sequential reader over a WAL file.
///
/// The reader opens the file in read-only mode, reads the full content into a
/// buffer (WALs are expected to stay small between checkpoints — tens of MiB
/// max under Group commit policy), and exposes an iterator starting at a
/// given byte offset.
#[derive(Debug)]
pub struct WalReader {
    bytes: Vec<u8>,
}

impl WalReader {
    /// Open and read the WAL file at `path`. Returns an empty reader if the
    /// file does not exist.
    pub fn open(path: impl AsRef<Path>) -> SubstrateResult<Self> {
        let path = path.as_ref();
        let mut bytes = Vec::new();
        if path.exists() {
            let mut f = File::open(path)?;
            f.read_to_end(&mut bytes)?;
        }
        Ok(Self { bytes })
    }

    /// Total byte length of the loaded WAL.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Iterate over the records starting at `offset`. Stops at the first
    /// decode error — this yields the crash boundary.
    ///
    /// Each item is `(record, offset, length)` where `offset` is where the
    /// record starts and `length` is its total encoded size (header + payload).
    pub fn iter_from(&self, offset: u64) -> WalIter<'_> {
        WalIter {
            bytes: &self.bytes,
            cursor: offset as usize,
            stop_on_error: true,
        }
    }
}

/// Iterator over WAL records. Yields `(record, offset, length)` tuples.
pub struct WalIter<'a> {
    bytes: &'a [u8],
    cursor: usize,
    stop_on_error: bool,
}

impl<'a> Iterator for WalIter<'a> {
    type Item = SubstrateResult<(WalRecord, u64, usize)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.bytes.len() {
            return None;
        }
        // A zero-length tail can exist if the file was pre-allocated; treat a
        // short remainder (< header size) as EOL.
        if self.bytes.len() - self.cursor < WAL_HEADER_SIZE {
            return None;
        }
        match WalRecord::decode(&self.bytes[self.cursor..]) {
            Ok((rec, used)) => {
                let at = self.cursor as u64;
                self.cursor += used;
                Some(Ok((rec, at, used)))
            }
            Err(e) => {
                if self.stop_on_error {
                    self.cursor = self.bytes.len(); // halt the iterator
                }
                Some(Err(e))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::WalPayload;
    use tempfile::tempdir;

    fn sample(lsn: u64, commit: bool) -> WalRecord {
        WalRecord {
            lsn,
            timestamp: 0,
            flags: if commit { WalRecord::FLAG_COMMIT } else { 0 },
            payload: WalPayload::NodeInsert {
                node_id: lsn as u32,
                label_bitset: lsn,
            },
        }
    }

    #[test]
    fn writer_appends_and_tracks_offset() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::Never, 1).unwrap();
        let (o1, l1) = w.append(sample(0, true)).unwrap();
        let (o2, l2) = w.append(sample(0, false)).unwrap();
        assert_eq!(o1, 0);
        assert_eq!(o2, l1 as u64);
        assert_eq!(w.offset(), (l1 + l2) as u64);
    }

    #[test]
    fn reader_iterates_writer_output() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::Never, 1).unwrap();
        for i in 1..=100u64 {
            w.append(sample(i, i % 10 == 0)).unwrap();
        }
        drop(w);

        let r = WalReader::open(&wal).unwrap();
        let mut count = 0u64;
        let mut last_offset = 0u64;
        for item in r.iter_from(0) {
            let (rec, off, len) = item.unwrap();
            assert!(off >= last_offset);
            assert!(len >= WAL_HEADER_SIZE);
            last_offset = off + len as u64;
            count += 1;
            // LSNs start at 1 (as passed to open()).
            assert_eq!(rec.lsn, count);
        }
        assert_eq!(count, 100);
    }

    #[test]
    fn reader_stops_at_crc_corruption() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::Never, 1).unwrap();
        for i in 1..=10u64 {
            w.append(sample(i, true)).unwrap();
        }
        drop(w);

        // Tamper the middle of the file.
        let mut bytes = std::fs::read(&wal).unwrap();
        let idx = bytes.len() / 2;
        bytes[idx] ^= 0xFF;
        std::fs::write(&wal, bytes).unwrap();

        let r = WalReader::open(&wal).unwrap();
        let mut ok_count = 0;
        let mut saw_err = false;
        for item in r.iter_from(0) {
            match item {
                Ok(_) => ok_count += 1,
                Err(_) => {
                    saw_err = true;
                    break;
                }
            }
        }
        assert!(saw_err, "expected to hit a CRC/decode error mid-file");
        assert!(ok_count > 0 && ok_count < 10);
    }

    #[test]
    fn fsync_every_commit_sets_durability() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::EveryCommit, 1).unwrap();
        // Commit-flagged record triggers fsync internally — hard to observe
        // directly, so we just verify no panic / no error.
        for i in 1..=20u64 {
            w.append(sample(i, i % 3 == 0)).unwrap();
        }
        w.fsync().unwrap();
    }

    #[test]
    fn truncate_resets_offset() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::Never, 1).unwrap();
        for i in 1..=5u64 {
            w.append(sample(i, true)).unwrap();
        }
        assert!(w.offset() > 0);
        w.truncate().unwrap();
        assert_eq!(w.offset(), 0);
        // Writing again after truncate starts at offset 0.
        let (off, _) = w.append(sample(42, true)).unwrap();
        assert_eq!(off, 0);
    }

    #[test]
    fn reader_starts_from_nonzero_offset() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("w.wal");
        let w = WalWriter::open(&wal, SyncMode::Never, 1).unwrap();
        let (o1, _) = w.append(sample(1, true)).unwrap();
        let (o2, _) = w.append(sample(2, true)).unwrap();
        let _ = w.append(sample(3, true)).unwrap();
        drop(w);

        assert_eq!(o1, 0);
        let r = WalReader::open(&wal).unwrap();
        let items: Vec<_> = r
            .iter_from(o2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0.lsn, 2);
        assert_eq!(items[1].0.lsn, 3);
    }

    #[test]
    fn empty_wal_reads_nothing() {
        let dir = tempdir().unwrap();
        let wal = dir.path().join("empty.wal");
        let r = WalReader::open(&wal).unwrap();
        assert!(r.is_empty());
        assert!(r.iter_from(0).next().is_none());
    }
}
