// T6 Step 2b smoke test — direct cross-restart edge persistence via ObrainDB
// opened in substrate mode.

#![cfg(feature = "substrate-backend")]

use obrain_core::graph::{Direction, GraphStoreMut};
use obrain_engine::ObrainDB;

#[test]
fn substrate_cross_restart_edge_survives() {
    // SAFETY: tests in this file are single-threaded and the env var is set
    // before ObrainDB::open reads it.
    unsafe {
        std::env::set_var("OBRAIN_BACKEND", "substrate");
    }
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("brain.db");

    let (a, b) = {
        let db = ObrainDB::open(&db_path).unwrap();
        let gs = db.graph_store_mut();
        let a = gs.create_node(&["P"]);
        let b = gs.create_node(&["P"]);
        let _eid = gs.create_edge(a, b, "SYNAPSE");
        let out = gs.edges_from(a, Direction::Outgoing);
        eprintln!(
            "phase1: a={:?} b={:?} outgoing.len={} edges={:?}",
            a,
            b,
            out.len(),
            out
        );
        assert_eq!(out.len(), 1, "edge should be visible before drop");
        (a, b)
    };

    let db2 = ObrainDB::open(&db_path).unwrap();
    let gs2 = db2.graph_store_mut();
    let outgoing = gs2.edges_from(a, Direction::Outgoing);
    eprintln!("phase2: a={:?} outgoing.len={} edges={:?}", a, outgoing.len(), outgoing);
    assert_eq!(
        outgoing.len(),
        1,
        "edge should survive ObrainDB close+reopen"
    );
    assert_eq!(outgoing[0].0, b);
}
