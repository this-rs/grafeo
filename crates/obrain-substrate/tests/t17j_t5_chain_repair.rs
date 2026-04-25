//! T17j T5 — tests for `SubstrateStore::repair_outgoing_chains`.
//!
//! Verifies :
//! - Idempotence on a healthy store (run twice = same final state)
//! - Fix on a corrupted fixture (manually break a chain, repair
//!   reconciles rec.src authoritative vs chain)

use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;

/// Tiny 5-node 6-edge graph, then repair → chain walks should match rec.src.
#[test]
fn repair_is_idempotent_on_healthy_store() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let a = store.create_node(&["A"]);
    let b = store.create_node(&["B"]);
    let c = store.create_node(&["C"]);
    let d = store.create_node(&["D"]);
    store.create_edge(a, b, "T1");
    store.create_edge(a, c, "T1");
    store.create_edge(b, c, "T2");
    store.create_edge(c, d, "T3");
    store.create_edge(d, a, "T4");

    // 1st repair — already healthy, should be no-op in effect.
    let (_reset, _rewired) = store.repair_outgoing_chains().unwrap();

    // Chain walks should match rec.src for every chain-owner.
    for nid in [a, b, c, d] {
        let (walked, matches, mismatches, _) = store.diagnose_chain_vs_rec_src(nid);
        assert_eq!(
            mismatches, 0,
            "chain of {nid:?} has {mismatches} mismatches after repair (walked={walked}, matches={matches})"
        );
    }

    // 2nd repair — idempotent, same final state.
    let _ = store.repair_outgoing_chains().unwrap();
    for nid in [a, b, c, d] {
        let (_, _, mismatches, _) = store.diagnose_chain_vs_rec_src(nid);
        assert_eq!(mismatches, 0, "second repair broke something");
    }
}

/// Verify that repair rebuilds the out-chain correctly : every edge
/// created with src=N must appear in N's outgoing chain after repair.
#[test]
fn repair_reconciles_chain_to_rec_src() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let files: Vec<_> = (0..5).map(|_| store.create_node(&["File"])).collect();
    let funcs: Vec<_> = (0..3).map(|_| store.create_node(&["Function"])).collect();

    // Create 3 IMPORTS from File[0] to File[1..4] + 2 IMPORTS from Function
    store.create_edge(files[0], files[1], "IMPORTS");
    store.create_edge(files[0], files[2], "IMPORTS");
    store.create_edge(files[0], files[3], "IMPORTS");
    store.create_edge(funcs[0], files[4], "IMPORTS");
    store.create_edge(funcs[1], files[4], "IMPORTS");

    let _ = store.repair_outgoing_chains().unwrap();

    // files[0] has 3 IMPORTS outgoing, funcs[0] has 1, funcs[1] has 1.
    let (w, m, mm, _) = store.diagnose_chain_vs_rec_src(files[0]);
    assert_eq!(mismatches_should_be_zero(mm), 0);
    assert_eq!(w, 3, "files[0] must walk 3 edges");
    assert_eq!(m, 3);
    let (w, _, mm, _) = store.diagnose_chain_vs_rec_src(funcs[0]);
    assert_eq!(mismatches_should_be_zero(mm), 0);
    assert_eq!(w, 1);
    let (w, _, mm, _) = store.diagnose_chain_vs_rec_src(funcs[1]);
    assert_eq!(mismatches_should_be_zero(mm), 0);
    assert_eq!(w, 1);
    // File[1] is purely a target, no outgoing IMPORTS
    let (w, _, _, _) = store.diagnose_chain_vs_rec_src(files[1]);
    assert_eq!(w, 0, "files[1] has no outgoing edges");
}

fn mismatches_should_be_zero(x: u64) -> u64 {
    assert_eq!(x, 0, "unexpected chain mismatch after repair");
    x
}
