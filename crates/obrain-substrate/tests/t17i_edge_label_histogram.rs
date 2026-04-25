//! T17i T2 — integration tests for per-edge-type × label-bit
//! histograms (`edge_target_labels` / `edge_source_labels` trait
//! methods, backed by DashMap counters on SubstrateStore).
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-substrate --test t17i_edge_label_histogram
//! ```

use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::collections::HashSet;

/// Helper — string set from a sequence of &str.
fn set(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// T17i T2 — a fresh edge populates the target & source histograms for
/// every label bit on each endpoint. Queries via the trait return
/// exactly the set of label names observed.
#[test]
fn edge_target_labels_on_mixed_graph() {
    let store = SubstrateStore::open_tempfile().unwrap();
    // Mixed label graph : every IMPORTS edge targets a different label.
    let f0 = store.create_node(&["File"]);
    let m0 = store.create_node(&["Module"]);
    let fn0 = store.create_node(&["Function"]);
    let src = store.create_node(&["File"]);

    store.create_edge(src, f0, "IMPORTS");
    store.create_edge(src, m0, "IMPORTS");
    store.create_edge(src, fn0, "IMPORTS");

    let targets = store.edge_target_labels("IMPORTS");
    assert_eq!(targets, set(&["File", "Module", "Function"]));

    let sources = store.edge_source_labels("IMPORTS");
    assert_eq!(sources, set(&["File"]));

    assert!(store.supports_edge_label_histogram());
}

/// T17i T2 — when every edge targets the same label, the histogram is
/// a singleton. This is the condition the T17i T3 Cypher rewrite
/// checks before accepting a peer-label constraint.
#[test]
fn edge_target_labels_on_homogeneous_graph() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let files: Vec<_> = (0..5).map(|_| store.create_node(&["File"])).collect();
    for i in 0..4 {
        store.create_edge(files[i], files[i + 1], "IMPORTS");
    }

    let targets = store.edge_target_labels("IMPORTS");
    assert_eq!(targets, set(&["File"]));
    let sources = store.edge_source_labels("IMPORTS");
    assert_eq!(sources, set(&["File"]));
}

/// T17i T2 — delete_edge decrements the histogram ; a label with no
/// remaining edges disappears from the set returned by the accessor.
#[test]
fn edge_target_labels_decrement_on_delete() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let src = store.create_node(&["File"]);
    let dst_f = store.create_node(&["File"]);
    let dst_m = store.create_node(&["Module"]);

    let e_to_file = store.create_edge(src, dst_f, "IMPORTS");
    let _e_to_mod = store.create_edge(src, dst_m, "IMPORTS");

    let targets = store.edge_target_labels("IMPORTS");
    assert_eq!(targets, set(&["File", "Module"]));

    // Delete the IMPORTS-to-File edge — Module remains.
    assert!(store.delete_edge(e_to_file));
    let targets = store.edge_target_labels("IMPORTS");
    assert_eq!(
        targets,
        set(&["Module"]),
        "File target label must disappear once its last edge is gone"
    );
}

/// T17i T2 — the histograms persist across close/reopen via
/// DictSnapshot v6. A reopened store must return the same label
/// sets without re-scanning the edge zone.
#[test]
fn edge_target_labels_survive_reopen() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("kb");
    {
        let s = SubstrateStore::create(&path).unwrap();
        let src = s.create_node(&["File"]);
        let dst_f = s.create_node(&["File"]);
        let dst_m = s.create_node(&["Module"]);
        s.create_edge(src, dst_f, "IMPORTS");
        s.create_edge(src, dst_m, "IMPORTS");
        s.flush().unwrap();
    }
    let s = SubstrateStore::open(&path).unwrap();
    assert_eq!(s.edge_target_labels("IMPORTS"), set(&["File", "Module"]));
    assert_eq!(s.edge_source_labels("IMPORTS"), set(&["File"]));
}

/// T17i T2 — unknown edge type returns an empty set. Used as a safety
/// guard in the planner rewrite.
#[test]
fn edge_target_labels_unknown_type_is_empty() {
    let store = SubstrateStore::open_tempfile().unwrap();
    let a = store.create_node(&["File"]);
    let b = store.create_node(&["File"]);
    store.create_edge(a, b, "IMPORTS");

    assert!(store.edge_target_labels("NOSUCHTYPE").is_empty());
    assert!(store.edge_source_labels("NOSUCHTYPE").is_empty());
}
