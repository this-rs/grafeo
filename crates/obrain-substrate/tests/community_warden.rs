//! T11 Step 4 — CommunityWarden end-to-end.
//!
//! Verifies that:
//! - the warden's read-only `scan()` correctly reports per-community
//!   fragmentation;
//! - `tick()` fires compaction on communities exceeding the trigger
//!   threshold;
//! - post-compaction, the overall fragmentation ratio drops at or below
//!   `1.15` (≤ 15% extra pages vs perfect packing), satisfying Step 4's
//!   verification criterion.

use std::sync::Arc;

use obrain_substrate::{CommunityWarden, SubstrateStore};

/// Setup: alternate inserts across 4 communities to force worst-case
/// fragmentation (each insert's community differs from the previous
/// one, so every insert takes the slow path and opens a new page).
#[test]
fn warden_compacts_fragmented_communities() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let store = Arc::new(SubstrateStore::create(&path).unwrap());

    // Round-robin 4 communities × 256 inserts each = 1024 inserts. With
    // strict alternation, community_placements sees each slot in a
    // different page than its prior allocation, forcing slow path
    // (page-boundary alignment) on every call — every insert opens a
    // fresh page, so pre-compaction fragmentation is ~1024/8 = 128×
    // the ideal.
    //
    // We use 256 per community (not 32) so the 1-slot slot-0 reservation
    // overhead amortizes below the 15% budget post-compaction: 1024
    // live nodes need ⌈1024/128⌉ = 8 ideal pages; slot 0 costs at most
    // one extra page → actual 9 pages → ratio ≤ 1.125.
    //
    // We tag every node with a non-zero label ("x") so the warden's
    // padding filter (zero-bitset) doesn't discard them.
    let labels = &["x"];
    for _ in 0..256 {
        for cid in 0..4u32 {
            let _ = store.create_node_in_community(labels, cid);
        }
    }
    store.flush().unwrap();

    let warden = CommunityWarden::new(store.clone());

    // Pre-scan: fragmentation must be well above the trigger threshold.
    let pre = warden.scan().unwrap();
    println!(
        "pre: live={}, distinct_pages={}, ideal_pages={}, ratio={:.2}",
        pre.total_live_nodes,
        pre.total_distinct_pages,
        pre.total_ideal_pages,
        pre.overall_ratio()
    );
    for c in &pre.communities {
        println!(
            "  pre-c{}: live={} distinct={} ideal={} ratio={:.2}",
            c.community_id, c.live_count, c.distinct_pages, c.ideal_pages, c.fragmentation
        );
    }
    assert!(
        pre.overall_ratio() > warden.trigger(),
        "pre-condition: expected fragmentation above trigger ({}), got {}",
        warden.trigger(),
        pre.overall_ratio()
    );

    // Tick: warden should fire compaction on at least one community.
    let fired = warden.tick().unwrap();
    assert!(
        !fired.is_empty(),
        "warden.tick() did not fire any compaction despite high fragmentation"
    );

    // Post-scan: overall ratio must drop to ≤ 1.15.
    let post = warden.scan().unwrap();
    println!(
        "post: live={}, distinct_pages={}, ideal_pages={}, ratio={:.2}",
        post.total_live_nodes,
        post.total_distinct_pages,
        post.total_ideal_pages,
        post.overall_ratio()
    );
    for c in &post.communities {
        println!(
            "  post-c{}: live={} distinct={} ideal={} ratio={:.2}",
            c.community_id, c.live_count, c.distinct_pages, c.ideal_pages, c.fragmentation
        );
    }
    assert!(
        post.overall_ratio() <= 1.15,
        "post-compaction: expected overall ratio ≤ 1.15, got {}",
        post.overall_ratio()
    );

    // Live node counts must be preserved exactly across compaction.
    assert_eq!(
        pre.total_live_nodes, post.total_live_nodes,
        "compaction lost or duplicated live nodes ({} → {})",
        pre.total_live_nodes, post.total_live_nodes
    );
}

/// A non-fragmented store doesn't trigger compaction.
#[test]
fn warden_is_noop_on_well_packed_store() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let store = Arc::new(SubstrateStore::create(&path).unwrap());

    // Sequential inserts into a single community: each page is fully
    // packed, so fragmentation ratio = 1.0 (below any sane trigger).
    let labels = &["y"];
    for _ in 0..500 {
        let _ = store.create_node_in_community(labels, 9);
    }
    store.flush().unwrap();

    let warden = CommunityWarden::new(store.clone());
    let report = warden.scan().unwrap();
    assert!(
        report.overall_ratio() <= 1.10,
        "sequential single-community insert is somehow fragmented: ratio={}",
        report.overall_ratio()
    );

    let fired = warden.tick().unwrap();
    assert!(
        fired.is_empty(),
        "warden fired compaction on a well-packed store: {:?}",
        fired
    );
}

/// Custom trigger threshold is respected.
#[test]
fn warden_respects_custom_trigger() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let store = Arc::new(SubstrateStore::create(&path).unwrap());

    // Modest fragmentation: 2 communities × 20 inserts alternated.
    // Each community has 20 nodes in 20 separate pages → ratio 20.
    let labels = &["z"];
    for _ in 0..20 {
        let _ = store.create_node_in_community(labels, 0);
        let _ = store.create_node_in_community(labels, 1);
    }
    store.flush().unwrap();

    // Loose warden (trigger = 100) — must NOT fire.
    let loose = CommunityWarden::with_trigger(store.clone(), 100.0);
    assert!(
        loose.tick().unwrap().is_empty(),
        "loose warden (trigger=100) fired on ratio ~20"
    );

    // Tight warden (trigger = 1.10) — MUST fire.
    let tight = CommunityWarden::with_trigger(store.clone(), 1.10);
    let fired = tight.tick().unwrap();
    assert!(
        !fired.is_empty(),
        "tight warden (trigger=1.10) failed to fire"
    );
}
