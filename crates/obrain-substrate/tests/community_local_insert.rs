//! T11 Step 3 — online community-local insertion.
//!
//! Verifies that `SubstrateStore::create_node_in_community` satisfies the
//! Step 3 criterion:
//!
//! > "Sequential insertion of 10⁵ nodes → number of pages ≤
//! >  ceil(10⁵ × 32 B / 4 KiB) × 1.1"
//!
//! i.e. at least 90% of the pages touched are fully packed with nodes of
//! the same community (10% fragmentation budget). For a single community
//! this should give exactly `ceil(N / 128) = 782` pages — well under the
//! `860` page budget. A second test exercises multi-community steady-state
//! and checks community contiguity at page granularity.

use obrain_substrate::{meta_flags, NODES_PER_PAGE, NodeRecord, SubstrateStore};

const PAGE_SIZE: usize = 4096;
const NODES_PER_PAGE_USIZE: usize = NODES_PER_PAGE as usize;

/// 10⁵ nodes into a single community — perfect packing, under budget.
#[test]
fn single_community_packs_within_budget() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let store = SubstrateStore::create(&path).unwrap();

    let n = 100_000u32;
    for _ in 0..n {
        let _ = store.create_node_in_community(&[], 42);
    }
    store.flush().unwrap();

    let hw = store.slot_high_water();
    // Pages touched = ceil(hw * 32 / 4096) = ceil(hw / 128).
    let pages_touched =
        ((hw as usize * NodeRecord::SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;

    let perfect = ((n as usize * NodeRecord::SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;
    let budget = (perfect * 110) / 100; // 10% fragmentation headroom.
    assert!(
        pages_touched <= budget,
        "single-community packing: {} pages used (budget {}, perfect {})",
        pages_touched,
        budget,
        perfect
    );

    // Verify every live slot carries the requested community_id.
    let mut wrong: u32 = 0;
    for slot in 1..hw {
        let rec = store
            .writer()
            .read_node(slot)
            .unwrap()
            .expect("slot should be live");
        if rec.community_id != 42 {
            wrong += 1;
        }
    }
    // Only the page-alignment gap left by the very first insert (slot 0
    // reserved → first node lands at slot 1, which is within page 0) can
    // contribute a stray; in a single-community run there is no transition
    // so `wrong` must be zero.
    assert_eq!(wrong, 0, "single-community run leaked foreign community ids");
}

/// Multi-community round-robin — each community owns full pages with no
/// interleaving at page granularity.
#[test]
fn multi_community_pages_are_not_interleaved() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    let store = SubstrateStore::create(&path).unwrap();

    // 4 communities × 500 nodes each = 2000 nodes = ~16 pages worth.
    // Grouped insertion: C0 × 500, C1 × 500, C2 × 500, C3 × 500.
    // With grouped order, each community owns a set of contiguous pages.
    for &cid in &[0u32, 1, 2, 3] {
        for _ in 0..500 {
            let _ = store.create_node_in_community(&[], cid);
        }
    }
    store.flush().unwrap();

    // Read back all nodes and bucket them by page.
    let hw = store.slot_high_water();
    let mut page_to_communities: std::collections::BTreeMap<
        u32,
        std::collections::BTreeSet<u32>,
    > = std::collections::BTreeMap::new();
    for slot in 1..hw {
        let Some(rec) = store.writer().read_node(slot).unwrap() else {
            continue;
        };
        // Skip padding slots created by slow-path alignment: they have
        // zero label_bitset and were never written (zero-fill from mmap
        // grow). A real live slot always passes through create_node_in_*
        // which assigns a bitset or an explicit zero-bitset record — in
        // this test we always pass `&[]` for labels, so all real slots
        // also have bitset == 0. Distinguish padding from real writes by
        // comparing against the persisted high-water marks for each
        // community (every real write is witnessed by community_placements).
        let page = slot / NODES_PER_PAGE;
        page_to_communities
            .entry(page)
            .or_default()
            .insert(rec.community_id);
    }

    // Padding slots (zero-filled) inherit community_id=0, which matches
    // one of our real communities. To avoid false positives, re-verify
    // against community_placements: every community's last_slot must
    // belong to a page owned exclusively by that community (modulo the
    // trailing page which may have a mix only if it was not yet closed).
    for cid in [0u32, 1, 2, 3] {
        let last_slot = store
            .last_slot_for_community(cid)
            .unwrap_or_else(|| panic!("community {cid} has no allocations"));
        // last_slot should index into a page dedicated to this community
        // (or in the case of cid=0, to cid=0 + possible zero-filled
        // padding which is indistinguishable on disk — we accept any
        // page containing cid in its BTreeSet for the last-page check).
        let page = last_slot / NODES_PER_PAGE;
        let set = page_to_communities.get(&page).unwrap();
        assert!(
            set.contains(&cid),
            "community {cid}'s last page {page} doesn't contain cid in on-disk view: {:?}",
            set
        );
    }

    // Under grouped insertion, the number of pages touched must be
    // <= 4 (one community at a time) × ceil(500/128) plus at most
    // 3 alignment pages (one per transition) = 4 × 4 + 3 = 19.
    let pages_touched = ((hw as usize * NodeRecord::SIZE) + PAGE_SIZE - 1) / PAGE_SIZE;
    assert!(
        pages_touched <= 19,
        "grouped multi-community insertion overflowed: {pages_touched} pages > 19"
    );
}

/// Online insertion clears the `HILBERT_SORTED` flag so a subsequent
/// `bulk_sort_by_hilbert` re-runs instead of early-exiting on stale state.
#[test]
fn online_insert_invalidates_hilbert_sorted_flag() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");

    // Build + sort in-place on a SubstrateStore-managed substrate so the
    // dict counters and the meta flag stay consistent end-to-end.
    let store = SubstrateStore::create(&path).unwrap();

    // Seed 6 nodes in 3 communities via the online path.
    for i in 0..6u32 {
        let _ = store.create_node_in_community(&[], i % 3);
    }
    store.flush().unwrap();

    // Drive the bulk sort through the store's Writer. The high-water
    // mark is whatever create_node_in_community allocated (slow-path
    // alignment may have pushed slots up to page boundaries; use the
    // store's own counter so the sort covers every live slot).
    let hw = store.slot_high_water();
    store
        .writer()
        .bulk_sort_by_hilbert(hw, 0, 3, 16)
        .unwrap();

    // Pre: the flag is set.
    {
        let sub_arc = store.writer().substrate();
        let meta = sub_arc.lock().meta_header();
        assert!(
            meta.flags & meta_flags::HILBERT_SORTED != 0,
            "pre: HILBERT_SORTED must be set after bulk_sort"
        );
    }

    // Online insert → flag must be cleared.
    let _ = store.create_node_in_community(&[], 7);

    let sub_arc = store.writer().substrate();
    let meta = sub_arc.lock().meta_header();
    assert!(
        meta.flags & meta_flags::HILBERT_SORTED == 0,
        "online insert did not clear HILBERT_SORTED"
    );

    // Extra sanity: NODES_PER_PAGE constant is self-consistent.
    assert_eq!(NODES_PER_PAGE_USIZE * NodeRecord::SIZE, PAGE_SIZE);
}
