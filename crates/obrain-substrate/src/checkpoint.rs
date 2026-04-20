//! Atomic checkpoint — durably snapshot the substrate state and reset the WAL.
//!
//! A checkpoint is the mechanism that keeps the WAL bounded: every time we
//! checkpoint, all mutations up to that point are reflected on disk in the
//! zone mmaps and the WAL can safely start over from offset 0.
//!
//! ## Protocol
//!
//! 1. **Pause writes** — the caller holds an exclusive lock on the writer so
//!    no new WAL records can be appended during the checkpoint.
//! 2. **Flush zones** — `msync` every live zone mmap, then `fsync` each
//!    backing file. Zone bytes are now durable.
//! 3. **Meta rewrite (atomic)** — build a new `MetaHeader` with
//!    `last_checkpoint = now`, `last_wal_offset = 0`, and write it to
//!    `substrate.meta.tmp`, fsync, rename → `substrate.meta`.
//! 4. **Checkpoint marker** — append a `WalPayload::Checkpoint` record with
//!    `FLAG_CHECKPOINT` to the new WAL. This marker is the first frame of
//!    the post-checkpoint WAL; any replay will see it at offset 0.
//! 5. **WAL truncate** — the new WAL starts with only the checkpoint marker.
//!
//! ## Crash safety
//!
//! The rename in step 3 is atomic on POSIX: the new meta either exists
//! (checkpoint succeeded) or doesn't (the old meta + old WAL are intact and
//! will be replayed). Either way the on-disk state is consistent.

use crate::error::SubstrateResult;
use crate::file::{SubstrateFile, Zone};
use crate::meta::MetaHeader;
use crate::wal::{WalPayload, WalRecord};
use crate::wal_io::WalWriter;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use tracing::info;

/// Outcome of a checkpoint operation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct CheckpointStats {
    /// WAL byte offset at the start of the checkpoint (will be 0 after).
    pub wal_offset_before: u64,
    /// The LSN stamped into the Checkpoint marker in the new WAL.
    pub at_lsn: u64,
    /// `last_checkpoint` timestamp written into the new meta.
    pub checkpoint_timestamp: i64,
}

/// Perform a full checkpoint.
///
/// The caller owns `substrate` (typically wrapped in an `Arc<Mutex<..>>`) and
/// `wal` and MUST ensure no concurrent writes are in flight for the duration
/// of this call. The WAL is truncated and reseeded with a Checkpoint marker.
pub fn checkpoint(substrate: &mut SubstrateFile, wal: &WalWriter) -> SubstrateResult<CheckpointStats> {
    let before = wal.offset();

    // ---- (1) Flush zones. Open each zone briefly, msync+fsync. -------------
    for zone in [
        Zone::Nodes,
        Zone::Edges,
        Zone::Props,
        Zone::Strings,
        Zone::Tier0,
        Zone::Tier1,
        Zone::Tier2,
        Zone::Hilbert,
        Zone::Community,
    ] {
        let zf = substrate.open_zone(zone)?;
        // An empty zone has no mmap; skip silently.
        if !zf.is_empty() {
            zf.msync()?;
            zf.fsync()?;
        }
    }

    // ---- (2) Build the new meta header and atomically write it. ------------
    let mut header = substrate.meta_header();
    let now = unix_seconds();
    header.last_checkpoint = now;
    header.last_wal_offset = 0;
    atomically_write_meta(substrate.path(), &header)?;
    // Reopen the mmap-backed meta with the new bytes.
    substrate.write_meta_header(&header)?;

    // ---- (3) Truncate the WAL and write the checkpoint marker. ------------
    wal.truncate()?;
    let at_lsn = wal.peek_lsn();
    let marker = WalRecord {
        lsn: 0, // assigned by append
        timestamp: unix_micros(),
        flags: WalRecord::FLAG_CHECKPOINT | WalRecord::FLAG_COMMIT,
        payload: WalPayload::Checkpoint { at_lsn },
    };
    wal.append(marker)?;
    wal.fsync()?;

    // ---- (4) Drop a checkpoint marker file next to the WAL. ---------------
    // Useful for external tooling (obrain-migrate, audit) to see the last
    // checkpoint without reading the WAL.
    let marker_path = substrate.checkpoint_path();
    let body = format!("{now}\n{at_lsn}\n");
    fs::write(&marker_path, body)?;

    info!(
        wal_before = before,
        at_lsn,
        timestamp = now,
        "checkpoint completed"
    );
    Ok(CheckpointStats {
        wal_offset_before: before,
        at_lsn,
        checkpoint_timestamp: now,
    })
}

/// Write the meta header to `<dir>/substrate.meta.tmp`, fsync it, then rename
/// to `<dir>/substrate.meta`. The rename is atomic on POSIX.
fn atomically_write_meta(dir: &Path, header: &MetaHeader) -> SubstrateResult<()> {
    let final_path = dir.join(crate::file::zone::META);
    let tmp_path = dir.join(format!("{}.tmp", crate::file::zone::META));

    let current_len = fs::metadata(&final_path)?.len() as usize;
    let mut buf = vec![0u8; current_len.max(crate::meta::META_FILE_SIZE)];
    buf[..core::mem::size_of::<MetaHeader>()].copy_from_slice(bytemuck::bytes_of(header));

    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, &final_path)?;
    // Fsync the directory so the rename is durable.
    #[cfg(unix)]
    {
        let d = fs::File::open(dir)?;
        d.sync_all()?;
    }
    Ok(())
}

fn unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn unix_micros() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::SubstrateFile;
    use crate::record::{f32_to_q1_15, NodeRecord, PackedScarUtilAff, U48};
    use crate::wal_io::{SyncMode, WalReader};
    use crate::writer::Writer;
    use tempfile::tempdir;

    fn n(i: u32) -> NodeRecord {
        NodeRecord {
            label_bitset: i as u64 + 1,
            first_edge_off: U48::default(),
            first_prop_off: U48::default(),
            community_id: 0,
            energy: f32_to_q1_15(0.5),
            scar_util_affinity: PackedScarUtilAff::new(0, 0, 0, false).pack(),
            centrality_cached: 0,
            flags: 0,
        }
    }

    #[test]
    fn checkpoint_resets_wal() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..5u32 {
            w.write_node(i, n(i)).unwrap();
        }
        w.commit().unwrap();
        let before = w.wal().offset();
        assert!(before > 0);

        // Checkpoint needs an exclusive &mut SubstrateFile — drop the writer's
        // ownership and re-open the substrate directly.
        let wal_writer = w.wal();
        drop(w);
        let mut sub2 = SubstrateFile::open(&sub_path).unwrap();
        let stats = checkpoint(&mut sub2, &wal_writer).unwrap();
        assert_eq!(stats.wal_offset_before, before);

        // The new WAL should contain exactly one record: the checkpoint marker.
        let r = WalReader::open(&wal_path).unwrap();
        let items: Vec<_> = r.iter_from(0).collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].0.is_checkpoint());
        matches!(items[0].0.payload, WalPayload::Checkpoint { .. });

        // Meta reflects the checkpoint.
        let h = sub2.meta_header();
        assert!(h.last_checkpoint > 0);
        assert_eq!(h.last_wal_offset, 0);
    }

    #[test]
    fn checkpoint_persists_zone_data() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..20u32 {
            w.write_node(i, n(i)).unwrap();
        }
        w.commit().unwrap();
        let wal_writer = w.wal();
        drop(w);
        let mut sub2 = SubstrateFile::open(&sub_path).unwrap();
        checkpoint(&mut sub2, &wal_writer).unwrap();

        // Reopen, no replay needed — checkpoint flushed everything.
        drop(sub2);
        let sub3 = SubstrateFile::open(&sub_path).unwrap();
        let nz = sub3.open_zone(Zone::Nodes).unwrap();
        let slice: &[NodeRecord] = bytemuck::cast_slice(
            &nz.as_slice()[..20 * NodeRecord::SIZE],
        );
        for i in 0..20u32 {
            assert_eq!(slice[i as usize].label_bitset, i as u64 + 1);
        }
    }

    #[test]
    fn checkpoint_marker_file_written() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        w.commit().unwrap();
        let wal_writer = w.wal();
        drop(w);
        let mut sub2 = SubstrateFile::open(&sub_path).unwrap();
        let marker = sub2.checkpoint_path();
        checkpoint(&mut sub2, &wal_writer).unwrap();
        assert!(marker.exists());
        let body = std::fs::read_to_string(&marker).unwrap();
        assert!(body.lines().count() >= 2);
    }

    #[test]
    fn replay_after_checkpoint_sees_only_marker() {
        let dir = tempdir().unwrap();
        let sub_path = dir.path().join("kb");
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 0..10u32 {
                w.write_node(i, n(i)).unwrap();
            }
            w.commit().unwrap();
            let wal_writer = w.wal();
            drop(w);
            let mut sub2 = SubstrateFile::open(&sub_path).unwrap();
            checkpoint(&mut sub2, &wal_writer).unwrap();
        }

        // Reopen + replay: should find exactly 1 applied record (the marker).
        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats = crate::replay::replay_from(&sub, 0).unwrap();
        assert_eq!(stats.applied, 1);
        assert_eq!(stats.decode_errors, 0);
    }
}
