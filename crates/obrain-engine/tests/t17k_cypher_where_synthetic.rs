//! T17k T4 — end-to-end Cypher WHERE / RETURN on bulk-hydrated
//! properties. Synthetic 100-Article corpus in a tempfile, no
//! dependency on any local base. CI-friendly.

#![cfg(feature = "cypher")]

use obrain_common::types::Value;
use obrain_engine::ObrainDB;

fn fresh_db() -> (tempfile::TempDir, ObrainDB) {
    // SAFETY: setting the env before opening; tests run single-threaded.
    unsafe {
        std::env::set_var("OBRAIN_PROPS_V2", "1");
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let db = ObrainDB::open(tmp.path()).expect("open substrate tempdir");
    (tmp, db)
}

// ── WHERE =  (scalar equality via PropsZone v2) ──

#[test]
fn cypher_where_eq_on_v2_scalar() {
    let (_tmp, db) = fresh_db();
    let store = db.store();
    // 10 Files with distinct small scalar `kind`.
    for i in 0..10 {
        let nid = store.create_node(&["File"]);
        let kind = if i < 5 { "rust" } else { "toml" };
        store.set_node_property(nid, "kind", Value::from(kind));
    }
    let session = db.session();
    let r = session
        .execute_cypher("MATCH (f:File) WHERE f.kind = 'rust' RETURN f.kind")
        .expect("cypher");
    assert_eq!(r.rows.len(), 5, "expected 5 rust files, got {}", r.rows.len());
    for row in &r.rows {
        assert_eq!(row.first(), Some(&Value::from("rust")));
    }
}

// ── WHERE CONTAINS on large String (blob_columns) ──

#[test]
fn cypher_where_contains_on_blob_string() {
    let (_tmp, db) = fresh_db();
    let store = db.store();
    // 100 Articles, each with a long (> 256 B) title prefix guaranteed
    // to route to blob_columns. 11 titles contain "_5".
    for i in 0..100 {
        let nid = store.create_node(&["Article"]);
        let padding = "x".repeat(300);
        let title = format!("article_{}_{}", i, padding);
        store.set_node_property(nid, "title", Value::from(title.as_str()));
    }
    let session = db.session();
    let r = session
        .execute_cypher("MATCH (a:Article) WHERE a.title CONTAINS '_5' RETURN a.title")
        .expect("cypher");
    // 11 rows : article_5, article_50..article_59
    assert_eq!(
        r.rows.len(),
        11,
        "expected 11 matches for '_5' pattern, got {}",
        r.rows.len()
    );
}

// ── RETURN on Vector-typed property (vec_columns hydration via planner) ──

#[test]
fn cypher_return_vector_property() {
    let (_tmp, db) = fresh_db();
    let store = db.store();
    // 3 Decisions with 80-dim non-zero embeddings.
    for i in 0..3 {
        let nid = store.create_node(&["Decision"]);
        let vec: Vec<f32> = (0..80).map(|j| 0.01 + (i as f32) + (j as f32) * 0.001).collect();
        store.set_node_property(
            nid,
            "embedding",
            Value::Vector(std::sync::Arc::from(vec.as_slice())),
        );
    }
    let session = db.session();
    let r = session
        .execute_cypher("MATCH (d:Decision) RETURN d.embedding")
        .expect("cypher");
    assert_eq!(r.rows.len(), 3);
    for row in &r.rows {
        match row.first() {
            Some(Value::Vector(v)) => assert_eq!(v.len(), 80),
            other => panic!("expected Vector(80), got {other:?}"),
        }
    }
}

// ── WHERE on edge property ──

#[test]
fn cypher_where_on_edge_property() {
    let (_tmp, db) = fresh_db();
    let store = db.store();
    let a = store.create_node(&["Function"]);
    let b = store.create_node(&["Function"]);
    let c = store.create_node(&["Function"]);
    let _e1 = store.create_edge(a, b, "CALLS");
    let e2 = store.create_edge(a, c, "CALLS");
    // Set property only on the second edge.
    store.set_edge_property(e2, "line", Value::Int64(42));

    let session = db.session();
    let r = session
        .execute_cypher("MATCH ()-[r:CALLS]->() WHERE r.line = 42 RETURN r.line")
        .expect("cypher");
    assert_eq!(r.rows.len(), 1, "expected 1 edge with line=42, got {}", r.rows.len());
    assert_eq!(r.rows[0].first(), Some(&Value::Int64(42)));
}
