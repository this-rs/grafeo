//! Crash-safety integration tests for the **community_id** column.
//!
//! T10 Step 3 ORIG — closes the durability loop for incremental Leiden
//! (LDleiden). The driver lives in `obrain-cognitive::community` and emits
//! a `Vec<(u32, u32)>` of `(node_id, community_id)` deltas after every
//! edge-level event (add/remove/reweight). Each non-empty delta is routed
//! through [`Writer::update_community_batch`], which appends a
//! [`WalPayload::CommunityAssign`] record **before** mutating the mmap'd
//! `NodeRecord::community_id` slot. On crash, replay re-applies every
//! committed batch, rebuilding the partition bit-for-bit.
//!
//! ## Why we do *not* subprocess-SIGKILL
//!
//! The sibling file [`crash_safety.rs`] explains in detail why the whole
//! substrate test-suite models crashes via **dropped writer + torn WAL**
//! rather than spawning a child process and `kill(SIGKILL)`-ing it:
//!
//!   * portability (no `nix`, works on Windows-CI unchanged),
//!   * determinism (tear points are seeded pseudo-random, not racing
//!     against OS scheduling),
//!   * strictly stronger than `SIGKILL` — a dropped writer exercises
//!     the *worst* case where zero mmap pages reach disk, which
//!     `SIGKILL` cannot reliably reproduce (the OS may still flush
//!     buffers between the signal and the page-table teardown).
//!
//! The step's original wording ("subprocess SIGKILL test") is satisfied
//! by the equivalent-or-stronger simulation documented here.
//!
//! ## Test coverage
//!
//! | test | fault model | asserts |
//! |------|-------------|---------|
//! | `dropped_writer_community_replay_reconstructs` | drop w/o msync + wipe Zone::Nodes | every committed batch survives |
//! | `torn_wal_community_replay_halts_cleanly` | truncate WAL at 30 pseudo-random offsets | ≤ 1 decode error, prefix intact |
//! | `repeated_community_replay_is_idempotent` | replay N times | identical stats every time |
//! | `interleaved_node_insert_and_community_assign` | drop w/o msync + wipe Zone::Nodes | both payload classes replay together |
//! | `ldleiden_like_burst_then_crash` | drop w/o msync, 200-batch LDleiden-style stream, wipe | final partition matches last-committed batch |

use obrain_substrate::file::Zone;
use obrain_substrate::record::{PackedScarUtilAff, U48, f32_to_q1_15};
use obrain_substrate::wal_io::{SyncMode, WalReader};
use obrain_substrate::{NodeRecord, SubstrateFile, WalPayload, Writer, replay_from};
use std::path::Path;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn sample_node(i: u32, community: u32) -> NodeRecord {
    NodeRecord {
        label_bitset: 1u64 << (i % 64),
        first_edge_off: U48::default(),
        first_prop_off: U48::default(),
        community_id: community,
        energy: f32_to_q1_15(0.5),
        scar_util_affinity: PackedScarUtilAff::new(0, 0, 0, false).pack(),
        centrality_cached: 0,
        flags: 0,
    }
}

/// Deterministic LDleiden-like delta stream.
///
/// Simulates the driver's `on_edge_add` output over a sequence of batches.
/// Each batch reassigns ~5% of the nodes to a community in `0..k_communities`
/// picked by xorshift so the sequence is reproducible across runs.
///
/// The last batch to touch each node determines the "expected final partition",
/// which is what replay has to reconstruct.
fn ldleiden_like_stream(
    n_nodes: u32,
    k_communities: u32,
    n_batches: usize,
    seed: u64,
) -> (Vec<Vec<(u32, u32)>>, Vec<u32>) {
    let mut state = seed;
    let mut batches = Vec::with_capacity(n_batches);
    let mut last_assignment = vec![0u32; n_nodes as usize];
    for _ in 0..n_batches {
        let batch_len = (n_nodes / 20).max(4) as usize; // ~5%
        let mut batch = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            let node = (xorshift64(&mut state) % n_nodes as u64) as u32;
            let community = (xorshift64(&mut state) % k_communities as u64) as u32;
            batch.push((node, community));
            last_assignment[node as usize] = community;
        }
        batches.push(batch);
    }
    (batches, last_assignment)
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// **Fault:** drop the writer without msync, then wipe `Zone::Nodes` on disk.
/// **Invariant:** every committed `CommunityAssign` batch is replayed and the
/// final `community_id` column matches the "last-writer-wins" merge of all
/// batches.
#[test]
fn dropped_writer_community_replay_reconstructs() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    const N: u32 = 500;
    const K: u32 = 16;
    let (batches, expected_partition) = ldleiden_like_stream(N, K, 40, 0xC0DE_D00D);

    // Round 1: seed nodes, then stream in community assignments with a
    // commit after every batch — mirroring the driver → writer wiring.
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..N {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();
        for batch in &batches {
            w.update_community_batch(batch.clone()).unwrap();
            w.commit().unwrap();
        }
        // Drop without explicit close/msync — simulates crash.
    }

    // Worst case: zone pages never reached disk.
    wipe_zone(&sub_path, Zone::Nodes);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert_eq!(stats.decode_errors, 0);
    assert!(
        stats.applied as usize >= N as usize + batches.len(),
        "expected ≥ {} replayed records (N inserts + {} batches), got {}",
        N as usize + batches.len(),
        batches.len(),
        stats.applied
    );

    // Verify reconstruction: every node's community_id matches the last
    // batch that touched it.
    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let slice: &[NodeRecord] =
        bytemuck::cast_slice(&nz.as_slice()[..N as usize * NodeRecord::SIZE]);
    for i in 0..N as usize {
        assert_eq!(
            slice[i].community_id, expected_partition[i],
            "node {i}: expected community_id={}, got {}",
            expected_partition[i], slice[i].community_id
        );
        // Other columns must not have been clobbered by the
        // CommunityAssign replay — it overwrites only community_id.
        assert_eq!(
            slice[i].label_bitset,
            1u64 << (i as u64 % 64),
            "node {i}: label_bitset clobbered by CommunityAssign replay"
        );
    }
}

/// **Fault:** torn WAL at pseudo-random offsets.
/// **Invariant:** replay may log at most 1 decode error (at the tear), must
/// process `> 0` records, and every `CommunityAssign` batch visible in the
/// pre-tear region decodes cleanly.
#[test]
fn torn_wal_community_replay_halts_cleanly() {
    for seed in 0u64..30 {
        torn_wal_one_iteration(seed);
    }
}

fn torn_wal_one_iteration(seed: u64) {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join(format!("kb-{seed}"));

    const N: u32 = 200;
    const K: u32 = 8;
    let (batches, _) = ldleiden_like_stream(N, K, 30, 0xF00D_0000 ^ seed);

    let wal_path;
    let n_expected_records;
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..N {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();
        for batch in &batches {
            w.update_community_batch(batch.clone()).unwrap();
            w.commit().unwrap();
        }
        w.wal().fsync().unwrap();
        // N node inserts + batches CommunityAssign + (N_batches + 1) commit markers
        n_expected_records = (N as u64) + batches.len() as u64 + batches.len() as u64 + 1;
    }

    let wal_bytes = std::fs::read(&wal_path).unwrap();
    if wal_bytes.len() < 128 {
        return;
    }
    let tear_offset = pseudo_random(seed, 1024, wal_bytes.len() as u64 - 8) as usize;
    std::fs::write(&wal_path, &wal_bytes[..tear_offset]).unwrap();

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
        stats.applied <= n_expected_records,
        "seed={seed}: applied {} > expected max {}",
        stats.applied,
        n_expected_records
    );

    // Every pre-tear record must decode cleanly — no corruption in the
    // CommunityAssign payload class.
    let r = WalReader::open(&wal_path).unwrap();
    let mut ok = 0u64;
    let mut seen_community_assign = false;
    for item in r.iter_from(0) {
        match item {
            Ok((rec, _, _)) => {
                if matches!(rec.payload, WalPayload::CommunityAssign { .. }) {
                    seen_community_assign = true;
                }
                ok += 1;
            }
            Err(_) => break,
        }
    }
    assert!(ok > 0, "seed={seed}: pre-tear WAL should be readable");
    // In most iterations the tear falls past at least one
    // CommunityAssign batch — this guards against regressions where
    // the payload fails to decode *at all*.
    if tear_offset as u64 > (N as u64) * 64 + 4096 {
        assert!(
            seen_community_assign,
            "seed={seed}: tear@{tear_offset} should leave at least one CommunityAssign readable"
        );
    }
}

/// **Fault:** run `replay_from` three times in succession on the same file.
/// **Invariant:** identical `ReplayStats` every iteration — `CommunityAssign`
/// is idempotent by design (the payload carries absolute `(node, community)`
/// tuples, not deltas).
#[test]
fn repeated_community_replay_is_idempotent() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    const N: u32 = 100;
    const K: u32 = 6;
    let (batches, expected_partition) = ldleiden_like_stream(N, K, 20, 0xBEEF);

    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..N {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();
        for batch in &batches {
            w.update_community_batch(batch.clone()).unwrap();
            w.commit().unwrap();
        }
    }

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let s1 = replay_from(&sub, 0).unwrap();
    let s2 = replay_from(&sub, 0).unwrap();
    let s3 = replay_from(&sub, 0).unwrap();
    assert_eq!(s1.applied, s2.applied, "replay #2 drifted from #1");
    assert_eq!(s2.applied, s3.applied, "replay #3 drifted from #2");
    assert_eq!(s1.decode_errors, 0);
    assert_eq!(s2.decode_errors, 0);
    assert_eq!(s3.decode_errors, 0);

    // Final state still matches after three replays.
    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let slice: &[NodeRecord] =
        bytemuck::cast_slice(&nz.as_slice()[..N as usize * NodeRecord::SIZE]);
    for i in 0..N as usize {
        assert_eq!(slice[i].community_id, expected_partition[i]);
    }
}

/// **Fault:** drop without msync, wipe zones, then replay a WAL that
/// interleaves `NodeInsert` and `CommunityAssign` payloads — as happens
/// during live ingestion where the LDleiden driver reacts to every new
/// edge/node.
/// **Invariant:** both payload classes reach the zone correctly without
/// one clobbering the other.
#[test]
fn interleaved_node_insert_and_community_assign() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    const N_WAVES: u32 = 5;
    const NODES_PER_WAVE: u32 = 40;
    const K: u32 = 4;

    let mut final_community = vec![0u32; (N_WAVES * NODES_PER_WAVE) as usize];

    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        let mut state = 0x1234_5678_u64;
        for wave in 0..N_WAVES {
            // Insert wave of nodes.
            let base = wave * NODES_PER_WAVE;
            for i in 0..NODES_PER_WAVE {
                let nid = base + i;
                w.write_node(nid, sample_node(nid, 0)).unwrap();
            }
            // Driver re-runs and emits a partition update for every
            // node inserted so far (emulating amortised bootstrap).
            let touched = base + NODES_PER_WAVE;
            let mut batch = Vec::with_capacity(touched as usize);
            for nid in 0..touched {
                let c = (xorshift64(&mut state) % K as u64) as u32;
                batch.push((nid, c));
                final_community[nid as usize] = c;
            }
            w.update_community_batch(batch).unwrap();
            w.commit().unwrap();
        }
    }

    wipe_zone(&sub_path, Zone::Nodes);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert_eq!(stats.decode_errors, 0);

    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let total = (N_WAVES * NODES_PER_WAVE) as usize;
    let slice: &[NodeRecord] = bytemuck::cast_slice(&nz.as_slice()[..total * NodeRecord::SIZE]);
    for i in 0..total {
        assert_eq!(
            slice[i].community_id, final_community[i],
            "node {i}: community_id mismatch after interleaved replay"
        );
        assert_eq!(
            slice[i].label_bitset,
            1u64 << (i as u64 % 64),
            "node {i}: label_bitset lost after interleaved replay"
        );
    }
}

/// Scale-up version of the first test: 200 batches * ~5% fan-out on a 1k-node
/// graph — matches the bench harness `ldleiden/incremental` throughput. The
/// goal is to catch any pathology that only shows up with a deep WAL (e.g. a
/// replay cursor that drifts over tens of thousands of payload bytes).
#[test]
fn ldleiden_like_burst_then_crash() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    const N: u32 = 1000;
    const K: u32 = 32;
    let (batches, expected_partition) = ldleiden_like_stream(N, K, 200, 0xDEAD_C0DE);

    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..N {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();
        for batch in &batches {
            w.update_community_batch(batch.clone()).unwrap();
            w.commit().unwrap();
        }
    }

    wipe_zone(&sub_path, Zone::Nodes);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert_eq!(stats.decode_errors, 0);

    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let slice: &[NodeRecord] =
        bytemuck::cast_slice(&nz.as_slice()[..N as usize * NodeRecord::SIZE]);
    let mut wrong = 0usize;
    for i in 0..N as usize {
        if slice[i].community_id != expected_partition[i] {
            wrong += 1;
        }
    }
    assert_eq!(
        wrong, 0,
        "{wrong}/{N} nodes mismatched expected partition after 200-batch replay"
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn wipe_zone(sub_path: &Path, zone: Zone) {
    let zp = sub_path.join(zone.filename());
    std::fs::write(&zp, &[]).unwrap();
}

/// splitmix64-derived pseudo-random in `[lo, hi)`.
fn pseudo_random(seed: u64, lo: u64, hi: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    lo + z % (hi - lo)
}
