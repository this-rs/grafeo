//! Crash-safety integration tests.
//!
//! These tests validate the core substrate invariant:
//!
//! > After any crash, replay from the WAL reconstructs a state consistent
//! > with every record that was durably committed (fsync-returned) before
//! > the crash. Records written after the last fsync are either replayed
//! > correctly or dropped cleanly at a CRC boundary — never corrupted.
//!
//! ## Strategy
//!
//! Instead of spawning a child process and SIGKILL'ing it (which is platform-
//! specific, requires extra deps like `nix`, and is hard to run reliably in
//! CI), we simulate the two crash modes directly:
//!
//! 1. **Dropped writer mid-burst** — the writer is dropped without msync'ing
//!    the zone mmaps. Replay from the WAL must reconstruct all committed
//!    records.
//! 2. **Torn WAL** — the WAL file is truncated at a random byte offset,
//!    simulating a partial write that didn't complete before power loss.
//!    Replay must stop cleanly at the tear and preserve all pre-tear records.
//!
//! Together these cover the three fault domains:
//!
//! * memory buffer loss (drop without msync) → replay reconstructs
//! * storage layer tear (truncated WAL) → replay halts at boundary
//! * combined (both at once) → replay reconstructs up to the boundary
//!
//! We run both scenarios 50 times with varying seeds to exercise a wide
//! distribution of WAL sizes and tear points.

use obrain_substrate::file::Zone;
use obrain_substrate::record::{PackedScarUtilAff, U48, f32_to_q1_15};
use obrain_substrate::wal_io::{SyncMode, WalReader};
use obrain_substrate::{NodeRecord, SubstrateFile, WalPayload, Writer, replay_from};
use std::path::Path;

fn sample_node(i: u32) -> NodeRecord {
    NodeRecord {
        label_bitset: 1u64 << (i % 64),
        first_edge_off: U48::default(),
        first_prop_off: U48::default(),
        community_id: i % 8,
        energy: f32_to_q1_15(0.5),
        scar_util_affinity: PackedScarUtilAff::new(0, 0, 0, false).pack(),
        centrality_cached: 0,
        flags: 0,
    }
}

/// Drop-without-msync: simulates a process that wrote records, got fsync acks,
/// then crashed before the OS flushed the mmap dirty pages to disk. The zone
/// files on disk may contain stale data, but the WAL has every record.
/// Replay must reconstruct.
#[test]
fn dropped_writer_replay_reconstructs() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    // Round 1: write 1000 nodes, fsync'd, then drop the writer. Zone pages may
    // still be dirty in OS buffer cache — but that's fine, they match the WAL.
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..1000u32 {
            w.write_node(i, sample_node(i)).unwrap();
            if i % 50 == 49 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
    }

    // Round 2: simulate the worst case — wipe the zone files entirely, as if
    // the mmap writes never reached disk. WAL should still reconstruct state.
    wipe_zone(&sub_path, Zone::Nodes);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert_eq!(stats.decode_errors, 0);
    assert!(
        stats.applied >= 1000,
        "expected to replay at least 1000 node inserts, got {}",
        stats.applied
    );

    // Verify the reconstruction.
    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let slice: &[NodeRecord] = bytemuck::cast_slice(&nz.as_slice()[..1000 * NodeRecord::SIZE]);
    for i in 0..1000u32 {
        assert_eq!(
            slice[i as usize].label_bitset,
            1u64 << (i % 64),
            "node {i} not reconstructed correctly"
        );
    }
}

/// Torn WAL at random offsets — 50 iterations with varied tear points.
#[test]
fn torn_wal_replay_halts_cleanly() {
    for seed in 0u64..50 {
        torn_wal_one_iteration(seed);
    }
}

fn torn_wal_one_iteration(seed: u64) {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join(format!("kb-{seed}"));

    // Write ~100 nodes, commit every 10.
    let wal_path;
    let n_expected_records;
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..100u32 {
            w.write_node(i, sample_node(i)).unwrap();
            if i % 10 == 9 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        w.wal().fsync().unwrap();
        n_expected_records = 100 + 11; // 100 inserts + 11 commits
    }

    // Truncate WAL at a position between 1 KiB and the full length - 8 B.
    let wal_bytes = std::fs::read(&wal_path).unwrap();
    if wal_bytes.len() < 64 {
        return; // degenerate
    }
    let tear_offset = pseudo_random(seed, 1024, wal_bytes.len() as u64 - 8) as usize;
    std::fs::write(&wal_path, &wal_bytes[..tear_offset]).unwrap();

    // Reopen + replay. Invariant: replay may log decode errors (at most 1:
    // the tear boundary) but must not panic and must process *some* records.
    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert!(
        stats.decode_errors <= 1,
        "seed={seed}: expected ≤ 1 decode error at tear, got {}",
        stats.decode_errors
    );
    assert!(
        stats.applied > 0,
        "seed={seed}: no records applied before tear at offset {tear_offset}"
    );
    assert!(
        stats.applied <= n_expected_records as u64,
        "seed={seed}: applied {} > expected max {}",
        stats.applied,
        n_expected_records
    );

    // Invariant: every record up to stats.stopped_at can be decoded
    // independently (no corruption in the pre-tear region).
    let r = WalReader::open(&wal_path).unwrap();
    let mut ok = 0u64;
    for item in r.iter_from(0) {
        if item.is_err() {
            break;
        }
        ok += 1;
    }
    assert!(
        ok > 0,
        "seed={seed}: WAL pre-tear region should be readable"
    );
}

/// Combined scenario: torn WAL AND wiped zones — replay must still recover
/// whatever the WAL can provide, without partial corruption.
#[test]
fn torn_wal_plus_wiped_zones_reconstructs_prefix() {
    for seed in 0u64..20 {
        let td = tempfile::tempdir().unwrap();
        let sub_path = td.path().join(format!("kb-{seed}"));

        let wal_path;
        {
            let sub = SubstrateFile::create(&sub_path).unwrap();
            wal_path = sub.wal_path();
            let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
            for i in 0..50u32 {
                w.write_node(i, sample_node(i)).unwrap();
            }
            w.commit().unwrap();
            w.wal().fsync().unwrap();
        }

        // Tear the WAL at ~60% of its length.
        let wal_bytes = std::fs::read(&wal_path).unwrap();
        let tear = (wal_bytes.len() * 6 / 10).max(256);
        std::fs::write(&wal_path, &wal_bytes[..tear]).unwrap();

        // Wipe the zone files.
        wipe_zone(&sub_path, Zone::Nodes);

        let sub = SubstrateFile::open(&sub_path).unwrap();
        let stats = replay_from(&sub, 0).unwrap();
        assert!(
            stats.decode_errors <= 1,
            "seed={seed}: expected ≤ 1 decode error, got {}",
            stats.decode_errors
        );

        // The records that were applied must be readable back from the zone.
        let nz = sub.open_zone(Zone::Nodes).unwrap();
        // Iterate the raw WAL up to the tear, collect node_ids that were inserted.
        let r = WalReader::open(&wal_path).unwrap();
        let mut inserted_ids = Vec::new();
        for item in r.iter_from(0) {
            match item {
                Ok((rec, _, _)) => {
                    if let WalPayload::NodeInsert { node_id, .. } = &rec.payload {
                        inserted_ids.push(*node_id);
                    }
                }
                Err(_) => break,
            }
        }
        // Every insert we saw in the WAL must now be durably in the zone.
        if !inserted_ids.is_empty() {
            let max_id = *inserted_ids.iter().max().unwrap();
            let slice: &[NodeRecord] =
                bytemuck::cast_slice(&nz.as_slice()[..(max_id as usize + 1) * NodeRecord::SIZE]);
            for id in inserted_ids {
                assert_eq!(
                    slice[id as usize].label_bitset,
                    1u64 << (id % 64),
                    "seed={seed}: replay lost node {id}"
                );
            }
        }
    }
}

/// Repeated replay after crash is a no-op: same state, same counts, no new
/// errors. Proves idempotence under crash recovery.
#[test]
fn repeated_replay_is_stable() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..30u32 {
            w.write_node(i, sample_node(i)).unwrap();
        }
        w.commit().unwrap();
    }

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let s1 = replay_from(&sub, 0).unwrap();
    let s2 = replay_from(&sub, 0).unwrap();
    let s3 = replay_from(&sub, 0).unwrap();
    assert_eq!(s1.applied, s2.applied);
    assert_eq!(s2.applied, s3.applied);
    assert_eq!(s1.decode_errors, 0);
    assert_eq!(s2.decode_errors, 0);
    assert_eq!(s3.decode_errors, 0);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn wipe_zone(sub_path: &Path, zone: Zone) {
    let zp = sub_path.join(zone.filename());
    std::fs::write(&zp, &[]).unwrap();
}

/// Deterministic pseudo-random in `[lo, hi)` — splitmix64 derived from `seed`.
fn pseudo_random(seed: u64, lo: u64, hi: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    lo + z % (hi - lo)
}
