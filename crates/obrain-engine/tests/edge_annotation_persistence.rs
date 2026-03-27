//! Persistence tests for EdgeAnnotator — annotations must survive close/reopen.
//!
//! Uses the single-file `.obrain` format to avoid the checkpoint.meta WAL
//! recovery gotcha (cf. knowledge note on WAL recovery bug).

#![cfg(all(feature = "cognitive", feature = "obrain-file"))]

use obrain_cognitive::EdgeAnnotator;
use obrain_engine::{Config, ObrainDB};

#[test]
fn edge_annotation_persistence_across_close_reopen() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("annotations.obrain");

    // --- Phase 1: create graph, annotate, close ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();
        let a = db.create_node(&["Concept"]);
        let b = db.create_node(&["Concept"]);
        let edge = db.create_edge(a, b, "RELATED");

        db.annotate(edge, "pheromone_query", 0.85);
        db.annotate(edge, "pheromone_error", 0.12);

        // Sanity check before close
        assert_eq!(db.get_annotation(edge, "pheromone_query"), Some(0.85));
        assert_eq!(db.get_annotation(edge, "pheromone_error"), Some(0.12));

        db.close().unwrap();
    }

    // --- Phase 2: reopen and verify annotations survived ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();

        // The edge ID should be deterministic (first edge = EdgeId(0))
        let edge = obrain_common::types::EdgeId::new(0);

        let pq = db.get_annotation(edge, "pheromone_query");
        let pe = db.get_annotation(edge, "pheromone_error");

        assert_eq!(pq, Some(0.85), "pheromone_query should survive reopen");
        assert_eq!(pe, Some(0.12), "pheromone_error should survive reopen");

        // Verify non-existent annotation still returns None
        assert_eq!(db.get_annotation(edge, "nonexistent"), None);

        db.close().unwrap();
    }
}

#[test]
fn edge_annotation_remove_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("ann_remove.obrain");

    // --- Phase 1: annotate then remove one key ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();
        let a = db.create_node(&["X"]);
        let b = db.create_node(&["Y"]);
        let edge = db.create_edge(a, b, "LINK");

        db.annotate(edge, "keep", 1.0);
        db.annotate(edge, "drop", 2.0);
        db.remove_annotation(edge, "drop");

        assert_eq!(db.get_annotation(edge, "keep"), Some(1.0));
        assert_eq!(db.get_annotation(edge, "drop"), None);

        db.close().unwrap();
    }

    // --- Phase 2: verify removal persisted ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();
        let edge = obrain_common::types::EdgeId::new(0);

        assert_eq!(db.get_annotation(edge, "keep"), Some(1.0));
        assert_eq!(db.get_annotation(edge, "drop"), None);

        db.close().unwrap();
    }
}

#[test]
fn edge_annotation_overwrite_persists() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("ann_overwrite.obrain");

    // --- Phase 1: annotate, then overwrite ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let edge = db.create_edge(a, b, "REL");

        db.annotate(edge, "strength", 0.5);
        db.annotate(edge, "strength", 0.99);

        assert_eq!(db.get_annotation(edge, "strength"), Some(0.99));
        db.close().unwrap();
    }

    // --- Phase 2: verify overwrite persisted ---
    {
        let db = ObrainDB::with_config(Config::persistent(&path)).unwrap();
        let edge = obrain_common::types::EdgeId::new(0);

        assert_eq!(
            db.get_annotation(edge, "strength"),
            Some(0.99),
            "overwritten value should persist"
        );
        db.close().unwrap();
    }
}
