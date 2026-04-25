//! T10 Step 3 ORIG — end-to-end wiring test for the LDleiden driver onto
//! `obrain-substrate::Writer::update_community_batch`.
//!
//! The sibling test
//! [`crates/obrain-substrate/tests/community_crash_safety.rs`] covers the
//! crash-safety + replay invariants on the *substrate side* by streaming
//! synthetic `CommunityAssign` batches. This file closes the loop on the
//! *driver side*: we drive a real [`LDleiden`] instance, feed its
//! `(node_id, community_id)` deltas into the Writer after every event,
//! drop the Writer without msync, wipe `Zone::Nodes`, reopen + replay,
//! and reconstruct the partition via [`LDleiden::from_partition`]. The
//! reconstructed driver must reach the same modularity as the pre-crash
//! one, to the bit.
//!
//! ## Why this test exists
//!
//! Without it, the contract "every driver delta reaches the WAL before
//! `on_edge_add` returns" is unverified. Tests that only exercise one
//! side of the wire can pass even if a batch is dropped silently between
//! `(Vec<(u32, u32)>, LDleidenStats)` and `Writer::update_community_batch`.

#![cfg(all(feature = "community", feature = "substrate"))]

use obrain_cognitive::community::{Graph, LDleiden, LeidenConfig};
use obrain_substrate::file::Zone;
use obrain_substrate::record::{PackedScarUtilAff, U48, f32_to_q1_15};
use obrain_substrate::wal_io::SyncMode;
use obrain_substrate::{NodeRecord, SubstrateFile, Writer, replay_from};

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

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn build_sbm(
    n_blocks: u32,
    block_size: u32,
    p_in: f64,
    p_out: f64,
    seed: u64,
) -> (u32, Vec<(u32, u32, f64)>) {
    let n = n_blocks * block_size;
    let mut state = seed;
    let mut edges = Vec::new();
    for u in 0..n {
        let bu = u / block_size;
        for v in (u + 1)..n {
            let bv = v / block_size;
            let p = if bu == bv { p_in } else { p_out };
            let r = (xorshift64(&mut state) as f64) / (u64::MAX as f64);
            if r < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    (n, edges)
}

/// Drive a real LDleiden against a bootstrapped graph, route every delta
/// through the substrate Writer, simulate a crash (drop without msync +
/// wipe Zone::Nodes), replay, and verify the substrate column matches
/// the pre-crash driver partition exactly.
#[test]
fn ldleiden_driver_persists_across_crash() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    // -- build a small SBM (4 blocks × 20 nodes, dense intra, sparse inter) -
    let (n, edges) = build_sbm(4, 20, 0.3, 0.01, 0xC0DE_D00D);
    let graph = Graph::from_edges(n, edges.clone());
    let mut driver = LDleiden::bootstrap(graph, LeidenConfig::default());

    // Build a deterministic stream of intra-block delta edges so the
    // driver emits non-trivial partition updates we can route to the WAL.
    let mut delta_state = 0xF00D_F00D_u64;
    let n_deltas = 60usize;
    let mut deltas: Vec<(u32, u32)> = Vec::with_capacity(n_deltas);
    for _ in 0..n_deltas {
        let block = (xorshift64(&mut delta_state) % 4) as u32;
        let base = block * 20;
        let u = base + (xorshift64(&mut delta_state) % 20) as u32;
        let v = base + (xorshift64(&mut delta_state) % 20) as u32;
        if u != v {
            deltas.push((u, v));
        }
    }

    // -- round 1: seed nodes, persist initial partition, stream deltas -------
    let expected_partition: Vec<u32>;
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..n {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();

        // Initial partition — persisted as one batch.
        let initial: Vec<(u32, u32)> = driver
            .partition()
            .iter()
            .enumerate()
            .map(|(u, &c)| (u as u32, c))
            .collect();
        w.update_community_batch(initial).unwrap();
        w.commit().unwrap();

        // Stream of per-edge deltas — one CommunityAssign per event.
        for &(u, v) in &deltas {
            let (partition_delta, _stats) = driver.on_edge_add(u, v, 1.0);
            if !partition_delta.is_empty() {
                w.update_community_batch(partition_delta).unwrap();
                w.commit().unwrap();
            }
        }

        expected_partition = driver.partition().to_vec();
        // Drop without explicit close/msync — simulates crash.
    }

    // -- simulate zone loss: wipe Zone::Nodes on disk ------------------------
    {
        let zp = sub_path.join(Zone::Nodes.filename());
        std::fs::write(&zp, &[]).unwrap();
    }

    // -- round 2: reopen, replay, verify substrate column matches -----------
    let sub = SubstrateFile::open(&sub_path).unwrap();
    let stats = replay_from(&sub, 0).unwrap();
    assert_eq!(stats.decode_errors, 0);

    let nz = sub.open_zone(Zone::Nodes).unwrap();
    let slice: &[NodeRecord] =
        bytemuck::cast_slice(&nz.as_slice()[..n as usize * NodeRecord::SIZE]);
    for i in 0..n as usize {
        assert_eq!(
            slice[i].community_id, expected_partition[i],
            "node {i}: substrate community_id {} != driver partition {}",
            slice[i].community_id, expected_partition[i]
        );
    }

    // -- round 3: rebuild the driver from the replayed partition -----------
    let final_graph = {
        let mut all_edges = edges.clone();
        for &(u, v) in &deltas {
            all_edges.push((u, v, 1.0));
        }
        Graph::from_edges(n, all_edges)
    };
    let reloaded_partition: Vec<u32> = (0..n).map(|i| slice[i as usize].community_id).collect();
    let reloaded =
        LDleiden::from_partition(final_graph, reloaded_partition, LeidenConfig::default());

    // Reloaded modularity must match the pre-crash driver bit-for-bit (same
    // graph, same partition ⇒ modularity is a deterministic function of both).
    let q_expected = driver.modularity();
    let q_reloaded = reloaded.modularity();
    assert!(
        (q_expected - q_reloaded).abs() < 1e-12,
        "modularity drifted after crash + replay: pre={q_expected}, post={q_reloaded}"
    );
}

/// Regression: the WAL contract guarantees that a batch returned by
/// `on_edge_add` is durable *before the method returns*. This test
/// inspects the WAL byte-for-byte to confirm each non-empty driver
/// delta produces a `CommunityAssign` record.
#[test]
fn each_driver_delta_produces_one_wal_record() {
    use obrain_substrate::WalPayload;
    use obrain_substrate::wal_io::WalReader;

    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");

    let (n, edges) = build_sbm(4, 15, 0.3, 0.01, 0xABCD_1234);
    let graph = Graph::from_edges(n, edges);
    let mut driver = LDleiden::bootstrap(graph, LeidenConfig::default());

    let wal_path;
    let mut expected_batches = 0usize;
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        wal_path = sub.wal_path();
        let w = Writer::new(sub, SyncMode::EveryCommit).unwrap();
        for i in 0..n {
            w.write_node(i, sample_node(i, 0)).unwrap();
        }
        w.commit().unwrap();

        // Seed partition — counts as 1 batch.
        let initial: Vec<(u32, u32)> = driver
            .partition()
            .iter()
            .enumerate()
            .map(|(u, &c)| (u as u32, c))
            .collect();
        w.update_community_batch(initial).unwrap();
        expected_batches += 1;

        // Stream deltas.
        let mut state = 0x1111_2222u64;
        for _ in 0..40 {
            let block = (xorshift64(&mut state) % 4) as u32;
            let base = block * 15;
            let u = base + (xorshift64(&mut state) % 15) as u32;
            let v = base + (xorshift64(&mut state) % 15) as u32;
            if u == v {
                continue;
            }
            let (delta, _) = driver.on_edge_add(u, v, 1.0);
            if !delta.is_empty() {
                w.update_community_batch(delta).unwrap();
                expected_batches += 1;
            }
        }
        w.commit().unwrap();
        w.wal().fsync().unwrap();
    }

    // Count CommunityAssign records in the WAL.
    let r = WalReader::open(&wal_path).unwrap();
    let mut seen_batches = 0usize;
    for item in r.iter_from(0) {
        if let Ok((rec, _, _)) = item {
            if matches!(rec.payload, WalPayload::CommunityAssign { .. }) {
                seen_batches += 1;
            }
        } else {
            break;
        }
    }
    assert_eq!(
        seen_batches, expected_batches,
        "WAL has {seen_batches} CommunityAssign records, driver produced {expected_batches}"
    );
}
