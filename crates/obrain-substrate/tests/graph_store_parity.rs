//! Parity suite — each scenario runs against both `LpgStore` and
//! `SubstrateStore` to prove the substrate-backed implementation is a
//! drop-in replacement for the in-memory reference.
//!
//! ## How it works
//!
//! Each test is a free-standing helper `fn parity_<name>(store:
//! &dyn GraphStoreMut)` that exercises a specific behavior via the
//! `GraphStore` + `GraphStoreMut` trait surface only — no concrete
//! type required. The `parity_suite!` macro then emits two `#[test]`
//! wrappers per helper, one fed an `LpgStore::new()`, the other a
//! freshly-created `SubstrateStore`.
//!
//! Tests that exercise LpgStore-only features (custom config, epoch
//! management, property indexes, savepoints / freeze / clear,
//! versioning) are intentionally **not** ported here — those are
//! covered by LpgStore's in-crate tests and are not part of the
//! GraphStore parity surface.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p obrain-substrate --test graph_store_parity
//! ```

use obrain_common::types::{EdgeId, NodeId, PropertyKey, Value};
use obrain_core::LpgStore;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;

// Shorthand for building `&[(PropertyKey, Value)]` from literal key/value pairs.
// `PropertyKey: From<&str>`, so `PropertyKey::from(k)` handles the str → key
// conversion without requiring callers to sprinkle `PropertyKey::new(...)`
// at every call site.
macro_rules! props {
    ( $( $k:expr => $v:expr ),* $(,)? ) => {
        &[
            $( (::obrain_common::types::PropertyKey::from($k), $v) ),*
        ] as &[(::obrain_common::types::PropertyKey, ::obrain_common::types::Value)]
    };
}

// ---------------------------------------------------------------------------
// Factory helpers — hand-rolled so the macro below stays readable.
// ---------------------------------------------------------------------------

fn lpg() -> LpgStore {
    LpgStore::new().unwrap()
}

struct SubstrateHandle {
    _td: tempfile::TempDir,
    store: SubstrateStore,
}

impl std::ops::Deref for SubstrateHandle {
    type Target = SubstrateStore;
    fn deref(&self) -> &SubstrateStore {
        &self.store
    }
}

fn substrate() -> SubstrateHandle {
    let td = tempfile::tempdir().unwrap();
    let store = SubstrateStore::create(td.path().join("kb")).unwrap();
    SubstrateHandle { _td: td, store }
}

// ---------------------------------------------------------------------------
// The macro
// ---------------------------------------------------------------------------

/// Emit two `#[test]` wrappers per helper, one per backend.
macro_rules! parity_suite {
    ( $( $name:ident ),* $(,)? ) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn lpg() {
                    let s = super::lpg();
                    super::$name(&s);
                }

                #[test]
                fn substrate() {
                    let s = super::substrate();
                    super::$name(&*s);
                }
            }
        )*
    };
}

// ---------------------------------------------------------------------------
// Parity helpers — each consumes &dyn GraphStoreMut and asserts behavior.
// ---------------------------------------------------------------------------

fn create_node_single(s: &dyn GraphStoreMut) {
    // NB: backends differ on whether NodeId 0 is a valid allocation (LpgStore
    // starts at 0, SubstrateStore reserves 0 as null sentinel). The parity
    // we assert is behavioural, not numeric.
    let id = s.create_node(&["Person"]);
    let n = s.get_node(id).unwrap();
    assert_eq!(n.id, id);
    assert!(n.labels.iter().any(|l| l.as_str() == "Person"));
    assert!(!n.labels.iter().any(|l| l.as_str() == "Animal"));
}

fn create_node_multiple_labels(s: &dyn GraphStoreMut) {
    let id = s.create_node(&["Person", "Employee", "Manager"]);
    let n = s.get_node(id).unwrap();
    assert_eq!(n.labels.len(), 3);
    let names: Vec<&str> = n.labels.iter().map(|l| l.as_str()).collect();
    assert!(names.contains(&"Person"));
    assert!(names.contains(&"Employee"));
    assert!(names.contains(&"Manager"));
}

fn create_node_no_labels(s: &dyn GraphStoreMut) {
    let id = s.create_node(&[]);
    let n = s.get_node(id).unwrap();
    assert!(n.labels.is_empty());
}

fn create_node_with_props(s: &dyn GraphStoreMut) {
    let id = s.create_node_with_props(
        &["Person"],
        props!("name" => Value::from("Alix"), "age" => Value::Int64(30)),
    );
    let name = s.get_node_property(id, &PropertyKey::new("name")).unwrap();
    let age = s.get_node_property(id, &PropertyKey::new("age")).unwrap();
    assert_eq!(name, Value::from("Alix"));
    assert_eq!(age, Value::Int64(30));
}

fn delete_node_basic(s: &dyn GraphStoreMut) {
    let id = s.create_node(&["Person"]);
    assert_eq!(s.node_count(), 1);
    assert!(s.delete_node(id));
    assert_eq!(s.node_count(), 0);
    assert!(s.get_node(id).is_none());
    // Double delete returns false
    assert!(!s.delete_node(id));
}

fn delete_nonexistent_node(s: &dyn GraphStoreMut) {
    let fake = NodeId(999);
    assert!(!s.delete_node(fake));
}

fn create_edge_basic(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["Person"]);
    let b = s.create_node(&["Person"]);
    let e = s.create_edge(a, b, "KNOWS");
    let edge = s.get_edge(e).unwrap();
    assert_eq!(edge.src, a);
    assert_eq!(edge.dst, b);
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
}

fn create_edge_with_props(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["Person"]);
    let b = s.create_node(&["Person"]);
    let e = s.create_edge_with_props(
        a,
        b,
        "KNOWS",
        props!(
            "since" => Value::Int64(2020),
            "weight" => Value::Float64(1.0),
        ),
    );
    let since = s.get_edge_property(e, &PropertyKey::new("since")).unwrap();
    let weight = s.get_edge_property(e, &PropertyKey::new("weight")).unwrap();
    assert_eq!(since, Value::Int64(2020));
    assert_eq!(weight, Value::Float64(1.0));
}

fn delete_edge_basic(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["Person"]);
    let b = s.create_node(&["Person"]);
    let e = s.create_edge(a, b, "KNOWS");
    assert_eq!(s.edge_count(), 1);
    assert!(s.delete_edge(e));
    assert_eq!(s.edge_count(), 0);
    assert!(s.get_edge(e).is_none());
    // Double delete
    assert!(!s.delete_edge(e));
}

fn delete_nonexistent_edge(s: &dyn GraphStoreMut) {
    let fake = EdgeId(999);
    assert!(!s.delete_edge(fake));
}

fn neighbors_outgoing_incoming(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    s.create_edge(a, b, "R");
    s.create_edge(a, c, "R");
    let out = s.neighbors(a, Direction::Outgoing);
    assert_eq!(out.len(), 2);
    assert!(out.contains(&b));
    assert!(out.contains(&c));
    let in_b = s.neighbors(b, Direction::Incoming);
    assert_eq!(in_b.len(), 1);
    assert!(in_b.contains(&a));
}

fn neighbors_both_directions(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    s.create_edge(a, b, "R");
    s.create_edge(c, a, "R");
    let both = s.neighbors(a, Direction::Both);
    assert_eq!(both.len(), 2);
    assert!(both.contains(&b));
    assert!(both.contains(&c));
}

fn edges_from_returns_all_directions(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    let e_ab = s.create_edge(a, b, "R");
    let e_ac = s.create_edge(a, c, "R");
    let e_ba = s.create_edge(b, a, "R");

    let out_a = s.edges_from(a, Direction::Outgoing);
    assert_eq!(out_a.len(), 2);
    let out_set: std::collections::BTreeSet<_> = out_a
        .iter()
        .map(|(peer, id)| (*peer, *id))
        .collect();
    assert!(out_set.contains(&(b, e_ab)));
    assert!(out_set.contains(&(c, e_ac)));

    let in_a = s.edges_from(a, Direction::Incoming);
    assert_eq!(in_a.len(), 1);
    assert_eq!(in_a[0], (b, e_ba));

    let both_a = s.edges_from(a, Direction::Both);
    assert_eq!(both_a.len(), 3);
}

fn out_and_in_degree(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    s.create_edge(a, b, "R");
    s.create_edge(a, c, "R");
    s.create_edge(b, a, "R");
    assert_eq!(s.out_degree(a), 2);
    assert_eq!(s.in_degree(a), 1);
    assert_eq!(s.out_degree(c), 0);
    assert_eq!(s.in_degree(c), 1);
}

fn edge_type_lookup(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let e = s.create_edge(a, b, "WORKS_AT");
    let t = s.edge_type(e).unwrap();
    assert_eq!(t.as_str(), "WORKS_AT");
    // unknown id
    assert!(s.edge_type(EdgeId(9999)).is_none());
}

fn node_count_tracks_lifecycle(s: &dyn GraphStoreMut) {
    assert_eq!(s.node_count(), 0);
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let _c = s.create_node(&["P"]);
    assert_eq!(s.node_count(), 3);
    assert!(s.delete_node(a));
    assert_eq!(s.node_count(), 2);
    assert!(s.delete_node(b));
    assert_eq!(s.node_count(), 1);
}

fn edge_count_tracks_lifecycle(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    assert_eq!(s.edge_count(), 0);
    let e1 = s.create_edge(a, b, "R");
    let e2 = s.create_edge(a, b, "R");
    assert_eq!(s.edge_count(), 2);
    assert!(s.delete_edge(e1));
    assert_eq!(s.edge_count(), 1);
    assert!(s.delete_edge(e2));
    assert_eq!(s.edge_count(), 0);
}

fn set_get_node_property(s: &dyn GraphStoreMut) {
    let id = s.create_node(&["P"]);
    s.set_node_property(id, "name", Value::from("Alix"));
    let got = s.get_node_property(id, &PropertyKey::new("name")).unwrap();
    assert_eq!(got, Value::from("Alix"));
    // Update
    s.set_node_property(id, "name", Value::from("Gus"));
    let got = s.get_node_property(id, &PropertyKey::new("name")).unwrap();
    assert_eq!(got, Value::from("Gus"));
    // Remove
    let old = s.remove_node_property(id, "name").unwrap();
    assert_eq!(old, Value::from("Gus"));
    assert!(s.get_node_property(id, &PropertyKey::new("name")).is_none());
    // Remove non-existent
    assert!(s.remove_node_property(id, "nonexistent").is_none());
}

fn set_get_edge_property(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let e = s.create_edge(a, b, "R");
    s.set_edge_property(e, "since", Value::Int64(2020));
    let got = s.get_edge_property(e, &PropertyKey::new("since")).unwrap();
    assert_eq!(got, Value::Int64(2020));
    let old = s.remove_edge_property(e, "since").unwrap();
    assert_eq!(old, Value::Int64(2020));
    assert!(s.get_edge_property(e, &PropertyKey::new("since")).is_none());
}

fn add_remove_label(s: &dyn GraphStoreMut) {
    let id = s.create_node(&["Person"]);
    // Add a new label
    assert!(s.add_label(id, "Employee"));
    let n = s.get_node(id).unwrap();
    assert!(n.labels.iter().any(|l| l.as_str() == "Person"));
    assert!(n.labels.iter().any(|l| l.as_str() == "Employee"));
    // Duplicate add
    assert!(!s.add_label(id, "Employee"));
    // Remove
    assert!(s.remove_label(id, "Employee"));
    let n = s.get_node(id).unwrap();
    assert!(!n.labels.iter().any(|l| l.as_str() == "Employee"));
    // Remove non-existent
    assert!(!s.remove_label(id, "Employee"));
    assert!(!s.remove_label(id, "NonExistent"));
}

fn add_label_on_missing_node(s: &dyn GraphStoreMut) {
    let fake = NodeId(999);
    assert!(!s.add_label(fake, "L"));
    assert!(!s.remove_label(fake, "L"));
}

fn node_ids_excludes_tombstones(s: &dyn GraphStoreMut) {
    let n1 = s.create_node(&["P"]);
    let n2 = s.create_node(&["P"]);
    let n3 = s.create_node(&["P"]);
    let ids = s.node_ids();
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&n1) && ids.contains(&n2) && ids.contains(&n3));
    s.delete_node(n2);
    let ids = s.node_ids();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&n2));
}

fn nodes_by_label_returns_matches_only(s: &dyn GraphStoreMut) {
    let p1 = s.create_node(&["Person"]);
    let p2 = s.create_node(&["Person"]);
    let _a = s.create_node(&["Animal"]);
    let persons = s.nodes_by_label("Person");
    assert_eq!(persons.len(), 2);
    assert!(persons.contains(&p1));
    assert!(persons.contains(&p2));
    let animals = s.nodes_by_label("Animal");
    assert_eq!(animals.len(), 1);
    let none = s.nodes_by_label("Nonexistent");
    assert_eq!(none.len(), 0);
}

fn multiple_labels_on_same_node(s: &dyn GraphStoreMut) {
    let id = s.create_node(&["Person", "Employee", "Manager"]);
    assert!(s.nodes_by_label("Person").contains(&id));
    assert!(s.nodes_by_label("Employee").contains(&id));
    assert!(s.nodes_by_label("Manager").contains(&id));
}

fn delete_node_edges_drops_both_directions(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    s.create_edge(a, b, "R"); // a -> b
    s.create_edge(c, a, "R"); // c -> a
    assert_eq!(s.edge_count(), 2);
    s.delete_node_edges(a);
    assert_eq!(s.edge_count(), 0);
}

fn delete_node_does_not_cascade_edges(s: &dyn GraphStoreMut) {
    // LpgStore semantics: delete_node does NOT detach edges. Callers must
    // explicitly call delete_node_edges first.
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let _e = s.create_edge(a, b, "R");
    assert_eq!(s.edge_count(), 1);
    assert!(s.delete_node(a));
    // Edge is still in the count — but the source node is gone.
    assert_eq!(s.edge_count(), 1);
}

fn get_missing_node_returns_none(s: &dyn GraphStoreMut) {
    let fake = NodeId(999);
    assert!(s.get_node(fake).is_none());
}

fn get_missing_edge_returns_none(s: &dyn GraphStoreMut) {
    let fake = EdgeId(999);
    assert!(s.get_edge(fake).is_none());
}

fn batch_get_node_property(s: &dyn GraphStoreMut) {
    let ids: Vec<NodeId> = (0..5)
        .map(|i| {
            s.create_node_with_props(
                &["P"],
                props!("score" => Value::Int64(i as i64 * 10)),
            )
        })
        .collect();
    let results = s.get_node_property_batch(&ids, &PropertyKey::new("score"));
    assert_eq!(results.len(), 5);
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.as_ref(), Some(&Value::Int64(i as i64 * 10)));
    }
}

fn batch_get_node_property_empty(s: &dyn GraphStoreMut) {
    let res = s.get_node_property_batch(&[], &PropertyKey::new("x"));
    assert!(res.is_empty());
}

fn batch_get_nodes_properties(s: &dyn GraphStoreMut) {
    let a = s.create_node_with_props(
        &["P"],
        props!("name" => Value::from("Alix"), "age" => Value::Int64(30)),
    );
    let b = s.create_node_with_props(
        &["P"],
        props!("name" => Value::from("Gus"), "city" => Value::from("Paris")),
    );
    let res = s.get_nodes_properties_batch(&[a, b]);
    assert_eq!(res.len(), 2);
    assert_eq!(res[0].get(&PropertyKey::new("name")), Some(&Value::from("Alix")));
    assert_eq!(res[0].get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
    assert_eq!(res[1].get(&PropertyKey::new("name")), Some(&Value::from("Gus")));
    assert_eq!(res[1].get(&PropertyKey::new("city")), Some(&Value::from("Paris")));
}

fn selective_batch_returns_only_requested_keys(s: &dyn GraphStoreMut) {
    let a = s.create_node_with_props(
        &["P"],
        props!(
            "name" => Value::from("Alix"),
            "age" => Value::Int64(30),
            "city" => Value::from("Paris"),
        ),
    );
    let res = s.get_nodes_properties_selective_batch(
        &[a],
        &[PropertyKey::new("name"), PropertyKey::new("age")],
    );
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].len(), 2);
    assert!(res[0].contains_key(&PropertyKey::new("name")));
    assert!(res[0].contains_key(&PropertyKey::new("age")));
    assert!(!res[0].contains_key(&PropertyKey::new("city")));
}

fn selective_batch_missing_keys_absent(s: &dyn GraphStoreMut) {
    let a = s.create_node_with_props(
        &["P"],
        props!("name" => Value::from("Alix")),
    );
    let res = s.get_nodes_properties_selective_batch(
        &[a],
        &[PropertyKey::new("name"), PropertyKey::new("missing")],
    );
    assert_eq!(res[0].len(), 1);
    assert!(res[0].contains_key(&PropertyKey::new("name")));
    assert!(!res[0].contains_key(&PropertyKey::new("missing")));
}

fn selective_batch_edges(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let e0 = s.create_edge(a, b, "R");
    let e1 = s.create_edge(a, b, "R");
    let e2 = s.create_edge(a, b, "R");
    s.set_edge_property(e0, "w", Value::Int64(10));
    s.set_edge_property(e2, "w", Value::Int64(30));
    let res = s.get_edges_properties_selective_batch(
        &[e0, e1, e2],
        &[PropertyKey::new("w")],
    );
    assert_eq!(res.len(), 3);
    assert_eq!(res[0].get(&PropertyKey::new("w")), Some(&Value::Int64(10)));
    assert!(res[1].is_empty());
    assert_eq!(res[2].get(&PropertyKey::new("w")), Some(&Value::Int64(30)));
}

fn find_nodes_in_range_inclusive(s: &dyn GraphStoreMut) {
    for age in [20, 25, 30, 35, 40].iter() {
        s.create_node_with_props(
            &["P"],
            props!("age" => Value::Int64(*age)),
        );
    }
    let min = Value::Int64(25);
    let max = Value::Int64(35);
    let got = s.find_nodes_in_range("age", Some(&min), Some(&max), true, true);
    assert_eq!(got.len(), 3, "25, 30, 35 are in [25, 35]");
}

fn find_nodes_in_range_exclusive(s: &dyn GraphStoreMut) {
    for age in [20, 25, 30, 35, 40].iter() {
        s.create_node_with_props(
            &["P"],
            props!("age" => Value::Int64(*age)),
        );
    }
    let min = Value::Int64(25);
    let max = Value::Int64(35);
    let got = s.find_nodes_in_range("age", Some(&min), Some(&max), false, false);
    assert_eq!(got.len(), 1, "only 30 is strictly between 25 and 35");
}

fn find_nodes_in_range_open_ended(s: &dyn GraphStoreMut) {
    for age in [20, 25, 30, 35, 40].iter() {
        s.create_node_with_props(
            &["P"],
            props!("age" => Value::Int64(*age)),
        );
    }
    let min = Value::Int64(30);
    let got = s.find_nodes_in_range("age", Some(&min), None, true, true);
    assert_eq!(got.len(), 3, "30, 35, 40");
}

fn find_nodes_in_range_nonexistent_property(s: &dyn GraphStoreMut) {
    s.create_node_with_props(&["P"], props!("age" => Value::Int64(25)));
    let min = Value::Int64(0);
    let max = Value::Int64(100);
    let got = s.find_nodes_in_range("nonexistent", Some(&min), Some(&max), true, true);
    assert!(got.is_empty());
}

fn find_nodes_by_properties_multi(s: &dyn GraphStoreMut) {
    s.create_node_with_props(
        &["P"],
        props!("role" => Value::from("engineer"), "level" => Value::Int64(3)),
    );
    s.create_node_with_props(
        &["P"],
        props!("role" => Value::from("engineer"), "level" => Value::Int64(5)),
    );
    s.create_node_with_props(
        &["P"],
        props!("role" => Value::from("manager"), "level" => Value::Int64(5)),
    );
    let got = s.find_nodes_by_properties(&[
        ("role", Value::from("engineer")),
        ("level", Value::Int64(3)),
    ]);
    assert_eq!(got.len(), 1, "only the first node matches both conditions");
}

fn find_nodes_by_properties_empty_is_all(s: &dyn GraphStoreMut) {
    s.create_node(&["P"]);
    s.create_node(&["P"]);
    // By trait contract: empty conditions match all nodes.
    let got = s.find_nodes_by_properties(&[]);
    assert_eq!(got.len(), 2);
}

fn find_nodes_by_properties_no_match(s: &dyn GraphStoreMut) {
    s.create_node_with_props(
        &["P"],
        props!("role" => Value::from("engineer")),
    );
    let got = s.find_nodes_by_properties(&[("role", Value::from("sales"))]);
    assert!(got.is_empty());
}

fn multiple_parallel_edges(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    // Two edges between the same pair, same type — independent edge ids.
    let e1 = s.create_edge(a, b, "R");
    let e2 = s.create_edge(a, b, "R");
    assert_ne!(e1, e2);
    s.set_edge_property(e1, "w", Value::Int64(1));
    s.set_edge_property(e2, "w", Value::Int64(2));
    let w1 = s.get_edge_property(e1, &PropertyKey::new("w")).unwrap();
    let w2 = s.get_edge_property(e2, &PropertyKey::new("w")).unwrap();
    assert_eq!(w1, Value::Int64(1));
    assert_eq!(w2, Value::Int64(2));
    assert_eq!(s.edge_count(), 2);
}

fn empty_store_invariants(s: &dyn GraphStoreMut) {
    assert_eq!(s.node_count(), 0);
    assert_eq!(s.edge_count(), 0);
    assert!(s.node_ids().is_empty());
    assert!(s.nodes_by_label("anything").is_empty());
    assert!(s.get_node(NodeId(1)).is_none());
    assert!(s.get_edge(EdgeId(1)).is_none());
}

fn set_property_on_deleted_node_does_not_panic(s: &dyn GraphStoreMut) {
    // LpgStore retains property buckets after `delete_node` (tombstone
    // semantics — get_node_property can still return values), while
    // SubstrateStore short-circuits writes to deleted slots. We intentionally
    // do NOT assert on the read result — only that the write path is safe.
    let id = s.create_node(&["P"]);
    assert!(s.delete_node(id));
    s.set_node_property(id, "name", Value::from("x"));
    // Node is gone from get_node either way.
    assert!(s.get_node(id).is_none());
}

fn set_property_on_deleted_edge_does_not_panic(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let e = s.create_edge(a, b, "R");
    assert!(s.delete_edge(e));
    s.set_edge_property(e, "w", Value::Int64(1));
    assert!(s.get_edge(e).is_none());
}

fn batch_create_edges_basic(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    let ids = s.batch_create_edges(&[(a, b, "R"), (a, c, "R"), (b, c, "R")]);
    assert_eq!(ids.len(), 3);
    for id in &ids {
        assert!(s.get_edge(*id).is_some());
    }
    assert_eq!(s.edge_count(), 3);
}

fn self_loop_edge(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let e = s.create_edge(a, a, "R");
    assert_eq!(s.out_degree(a), 1);
    assert_eq!(s.in_degree(a), 1);
    let out = s.edges_from(a, Direction::Outgoing);
    let in_ = s.edges_from(a, Direction::Incoming);
    assert_eq!(out.len(), 1);
    assert_eq!(in_.len(), 1);
    assert!(s.delete_edge(e));
    assert_eq!(s.out_degree(a), 0);
    assert_eq!(s.in_degree(a), 0);
}

fn delete_one_of_many_edges_keeps_rest(s: &dyn GraphStoreMut) {
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let e1 = s.create_edge(a, b, "R");
    let e2 = s.create_edge(a, b, "R");
    let e3 = s.create_edge(a, b, "R");
    assert!(s.delete_edge(e2));
    let out = s.edges_from(a, Direction::Outgoing);
    assert_eq!(out.len(), 2);
    let ids: std::collections::BTreeSet<_> = out.iter().map(|(_, id)| *id).collect();
    assert!(ids.contains(&e1));
    assert!(ids.contains(&e3));
    assert!(!ids.contains(&e2));
}

fn count_methods_consistency(s: &dyn GraphStoreMut) {
    // Ensure node_count / edge_count / node_ids length all agree through
    // a lifecycle.
    let a = s.create_node(&["P"]);
    let b = s.create_node(&["P"]);
    let c = s.create_node(&["P"]);
    let _e = s.create_edge(a, b, "R");
    let _e = s.create_edge(b, c, "R");
    assert_eq!(s.node_count(), 3);
    assert_eq!(s.node_ids().len(), 3);
    assert_eq!(s.edge_count(), 2);
    s.delete_node(c);
    assert_eq!(s.node_count(), 2);
    assert_eq!(s.node_ids().len(), 2);
}

fn statistics_object_is_live(s: &dyn GraphStoreMut) {
    // Just smoke-test that the Statistics snapshot is returned.
    let _ = s.create_node(&["P"]);
    let stats = s.statistics();
    // The reference is valid — we don't assert contents because the
    // stats population policy differs between backends.
    let _ = std::sync::Arc::strong_count(&stats);
}

fn create_node_many(s: &dyn GraphStoreMut) {
    // Bulk smoke test — exercise grow policy under slight pressure.
    let mut ids = Vec::new();
    for _ in 0..200 {
        ids.push(s.create_node(&["P"]));
    }
    assert_eq!(s.node_count(), 200);
    // All ids must be distinct.
    let set: std::collections::BTreeSet<_> = ids.iter().copied().collect();
    assert_eq!(set.len(), 200);
}

fn create_edge_many(s: &dyn GraphStoreMut) {
    let nodes: Vec<NodeId> = (0..50).map(|_| s.create_node(&["P"])).collect();
    let mut edges = Vec::new();
    for i in 0..nodes.len() - 1 {
        edges.push(s.create_edge(nodes[i], nodes[i + 1], "R"));
    }
    assert_eq!(s.edge_count(), 49);
    // Each middle node should have in=out=1.
    for i in 1..nodes.len() - 1 {
        assert_eq!(s.out_degree(nodes[i]), 1);
        assert_eq!(s.in_degree(nodes[i]), 1);
    }
    // Silence unused bindings warning.
    let _ = edges;
}

// ---------------------------------------------------------------------------
// Wire up all helpers.
// ---------------------------------------------------------------------------

parity_suite! {
    create_node_single,
    create_node_multiple_labels,
    create_node_no_labels,
    create_node_with_props,
    delete_node_basic,
    delete_nonexistent_node,
    create_edge_basic,
    create_edge_with_props,
    delete_edge_basic,
    delete_nonexistent_edge,
    neighbors_outgoing_incoming,
    neighbors_both_directions,
    edges_from_returns_all_directions,
    out_and_in_degree,
    edge_type_lookup,
    node_count_tracks_lifecycle,
    edge_count_tracks_lifecycle,
    set_get_node_property,
    set_get_edge_property,
    add_remove_label,
    add_label_on_missing_node,
    node_ids_excludes_tombstones,
    nodes_by_label_returns_matches_only,
    multiple_labels_on_same_node,
    delete_node_edges_drops_both_directions,
    delete_node_does_not_cascade_edges,
    get_missing_node_returns_none,
    get_missing_edge_returns_none,
    batch_get_node_property,
    batch_get_node_property_empty,
    batch_get_nodes_properties,
    selective_batch_returns_only_requested_keys,
    selective_batch_missing_keys_absent,
    selective_batch_edges,
    find_nodes_in_range_inclusive,
    find_nodes_in_range_exclusive,
    find_nodes_in_range_open_ended,
    find_nodes_in_range_nonexistent_property,
    find_nodes_by_properties_multi,
    find_nodes_by_properties_empty_is_all,
    find_nodes_by_properties_no_match,
    multiple_parallel_edges,
    empty_store_invariants,
    set_property_on_deleted_node_does_not_panic,
    set_property_on_deleted_edge_does_not_panic,
    batch_create_edges_basic,
    self_loop_edge,
    delete_one_of_many_edges_keeps_rest,
    count_methods_consistency,
    statistics_object_is_live,
    create_node_many,
    create_edge_many,
}
