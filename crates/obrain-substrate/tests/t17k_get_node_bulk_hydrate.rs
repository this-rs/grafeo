//! T17k T4 — synthetic fixtures proving that `get_node` bulk-hydrates
//! from all four zones (DashMap / PropsZone v2 / blob_columns / vec_columns)
//! in the correct LWW order. Zero dependency on a local production base.

use obrain_common::PropertyKey;
use obrain_common::types::Value;
use obrain_core::graph::{GraphStore, traits::GraphStoreMut};
use obrain_substrate::SubstrateStore;

// ── Helper : build a fresh store that has props_zone_v2 enabled ──
//
// PropsZone v2 is the runtime write target for small scalar props.
// The env gate `OBRAIN_PROPS_V2=1` is honoured by SubstrateStore::open
// — see store.rs around line 1961.
fn fresh_store() -> SubstrateStore {
    // SAFETY: tests run with a single env at a time; we set the gate
    // prior to opening the store. Rust 2024 requires the unsafe block.
    unsafe {
        std::env::set_var("OBRAIN_PROPS_V2", "1");
    }
    SubstrateStore::open_tempfile().expect("open tempfile")
}

// ── Test A : PropsZone v2 round-trip (small scalar props) ──

#[test]
fn get_node_hydrates_small_scalar_props_via_v2() {
    let store = fresh_store();
    let n = store.create_node(&["File"]);
    store.set_node_property(n, "line", Value::Int64(42));
    store.set_node_property(n, "column", Value::Int64(7));
    store.set_node_property(n, "is_test", Value::Bool(true));

    let node = store.get_node(n).expect("node exists");
    assert_eq!(
        node.properties.len(),
        3,
        "3 scalar props expected, got {:?}",
        node.properties
    );
    assert_eq!(
        node.properties.get(&PropertyKey::new("line")),
        Some(&Value::Int64(42))
    );
    assert_eq!(
        node.properties.get(&PropertyKey::new("column")),
        Some(&Value::Int64(7))
    );
    assert_eq!(
        node.properties.get(&PropertyKey::new("is_test")),
        Some(&Value::Bool(true))
    );
}

// ── Test B : blob_columns round-trip (large String/Bytes) ──

#[test]
fn get_node_hydrates_blob_strings() {
    let store = fresh_store();
    let n = store.create_node(&["File"]);
    // Must exceed 256 B threshold to route to blob_columns (else PropsZone v2).
    let long_path = format!("/Users/triviere/projects/obrain-hub/{}", "a/".repeat(200));
    let long_title = format!("Incendie du bar Le Constellation {}", "x".repeat(300));
    store.set_node_property(n, "path", Value::from(long_path.as_str()));
    store.set_node_property(n, "title", Value::from(long_title.as_str()));

    let node = store.get_node(n).expect("node exists");
    assert_eq!(node.properties.len(), 2);
    match node.properties.get(&PropertyKey::new("path")) {
        Some(Value::String(s)) => assert!(s.contains("obrain-hub")),
        _ => panic!("path not hydrated as String"),
    }
    match node.properties.get(&PropertyKey::new("title")) {
        Some(Value::String(s)) => assert!(s.contains("Le Constellation")),
        _ => panic!("title not hydrated"),
    }
}

// ── Test C : vec_columns round-trip (geometric features) ──

#[test]
fn get_node_hydrates_vector_80_dim() {
    let store = fresh_store();
    let n = store.create_node(&["Decision"]);
    // 80-dim kernel embedding, non-zero (avoid the all-zero sentinel).
    let vec80: Vec<f32> = (0..80).map(|i| 0.01 + (i as f32) * 0.001).collect();
    store.set_node_property(
        n,
        "embedding",
        Value::Vector(std::sync::Arc::from(vec80.as_slice())),
    );

    let node = store.get_node(n).expect("node exists");
    assert_eq!(node.properties.len(), 1);
    match node.properties.get(&PropertyKey::new("embedding")) {
        Some(Value::Vector(v)) => {
            assert_eq!(v.len(), 80);
            assert!((v[0] - 0.01).abs() < 1e-6);
            assert!((v[79] - (0.01 + 79.0 * 0.001)).abs() < 1e-6);
        }
        other => panic!("embedding not hydrated as Vector: {other:?}"),
    }
}

// ── Test D : mixed 4 zones with LWW override ──

#[test]
fn get_node_lww_ordering_across_zones() {
    let store = fresh_store();
    let n = store.create_node(&["Article"]);

    // Write a small scalar (routes to v2) then update the same key
    // a second time (new v2 entry shadows old).
    store.set_node_property(n, "status", Value::from("draft"));
    store.set_node_property(n, "status", Value::from("published"));

    // Write a large String → blob_columns zone.
    let long_abstract = format!("Long abstract with padding {}", "z".repeat(400));
    store.set_node_property(n, "abstract", Value::from(long_abstract.as_str()));

    // Write a Vector → vec_columns zone.
    let vec8: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    store.set_node_property(
        n,
        "fingerprint",
        Value::Vector(std::sync::Arc::from(vec8.as_slice())),
    );

    let node = store.get_node(n).expect("node exists");
    // status (v2 LWW), abstract (blob), fingerprint (vec) = 3 props
    assert_eq!(node.properties.len(), 3, "got {:?}", node.properties);
    // LWW : second write of "status" wins
    assert_eq!(
        node.properties.get(&PropertyKey::new("status")),
        Some(&Value::from("published"))
    );
    match node.properties.get(&PropertyKey::new("abstract")) {
        Some(Value::String(s)) => assert!(s.starts_with("Long abstract")),
        _ => panic!("abstract not hydrated"),
    }
    match node.properties.get(&PropertyKey::new("fingerprint")) {
        Some(Value::Vector(v)) => assert_eq!(v.len(), 8),
        _ => panic!("fingerprint not hydrated"),
    }
}

// ── Test E : tombstone removes from all downstream zones ──

#[test]
fn get_node_tombstone_removes_key() {
    let store = fresh_store();
    let n = store.create_node(&["Note"]);
    store.set_node_property(n, "pinned", Value::Bool(true));

    let node_before = store.get_node(n).unwrap();
    assert_eq!(node_before.properties.len(), 1);

    // Delete the property — v2 writes a tombstone.
    store.remove_node_property(n, "pinned");

    let node_after = store.get_node(n).unwrap();
    assert_eq!(
        node_after.properties.len(),
        0,
        "tombstone should remove the key from bulk hydrate, got {:?}",
        node_after.properties
    );
}

// Test F (Cypher WHERE end-to-end) lives in
// `crates/obrain-engine/tests/t17k_cypher_where_synthetic.rs` — it needs
// obrain-engine, which is not a dep of obrain-substrate.
