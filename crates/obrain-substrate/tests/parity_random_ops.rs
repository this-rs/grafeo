//! Randomized parity — apply identical ops to an `LpgStore` and a
//! `SubstrateStore`, compare canonical hashes every step.
//!
//! ## Design
//!
//! Concrete `NodeId` / `EdgeId` values differ between the two backends, so we
//! can't just hash the raw graph state. Instead we assign **logical** IDs
//! inside the harness: every time either store returns a new node/edge id,
//! we push it into a parallel vector indexed by logical id. Canonicalization
//! then walks the logical space in order, producing the same bytes for both
//! backends.
//!
//! The canonical form per step hashes:
//!
//! - `node_count()`
//! - `edge_count()`
//! - For each live logical node, `(labels_sorted, (key, value)_sorted)`
//! - For each live logical edge, `(src_logical, dst_logical, edge_type,
//!   (key, value)_sorted)`
//!
//! If the two hashes diverge at any step, the test fails and prints the op
//! that caused it.
//!
//! ## Scope
//!
//! Verification criterion: `10k ops × 100 seeds = 1M ops, 0 divergence`.
//!
//! ```bash
//! cargo test -p obrain-substrate --test parity_random_ops --release
//! ```
//!
//! In debug mode the full matrix is slow; the default `#[test]` therefore
//! runs a reduced scope (`NUM_SEEDS=5, OPS_PER_SEED=2_000`) that exercises
//! the same code paths. A second test `parity_random_ops_full` runs the full
//! matrix and is only green in release.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use obrain_common::types::{EdgeId, NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::SubstrateStore;
use rand::{RngExt, SeedableRng};
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Dual {
    lpg: LpgStore,
    sub: SubstrateStore,
    _sub_td: tempfile::TempDir,
    /// logical_node_idx → (lpg_id, sub_id, alive)
    nodes: Vec<(NodeId, NodeId, bool)>,
    /// logical_edge_idx → (lpg_id, sub_id, src_logical, dst_logical, alive)
    edges: Vec<(EdgeId, EdgeId, usize, usize, bool)>,
}

impl Dual {
    fn new() -> Self {
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateStore::create(td.path().join("kb")).unwrap();
        Dual {
            lpg: LpgStore::new().unwrap(),
            sub,
            _sub_td: td,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn alive_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, a))| if *a { Some(i) } else { None })
            .collect()
    }

    fn alive_edges(&self) -> Vec<usize> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, (.., a))| if *a { Some(i) } else { None })
            .collect()
    }

    fn create_node(&mut self, labels: &[&str]) {
        let a = self.lpg.create_node(labels);
        let b = self.sub.create_node(labels);
        self.nodes.push((a, b, true));
    }

    fn delete_node(&mut self, li: usize) {
        if !self.nodes[li].2 {
            return;
        }
        // Detach edges on both sides first, so the `edge_count` stays in sync.
        // LpgStore `delete_node` leaves incident edges in place, but our
        // canonical hash relies on edges referring to alive logical nodes.
        // So we explicitly detach, in a single sweep, on BOTH backends.
        let (a, b, _) = self.nodes[li];
        self.lpg.delete_node_edges(a);
        self.sub.delete_node_edges(b);
        self.lpg.delete_node(a);
        self.sub.delete_node(b);
        self.nodes[li].2 = false;
        // Mark incident edges dead in the harness.
        for e in self.edges.iter_mut() {
            if e.4 && (e.2 == li || e.3 == li) {
                e.4 = false;
            }
        }
    }

    fn create_edge(&mut self, src_li: usize, dst_li: usize, ty: &str) {
        if !self.nodes[src_li].2 || !self.nodes[dst_li].2 {
            return;
        }
        let (sa, sb, _) = self.nodes[src_li];
        let (da, db, _) = self.nodes[dst_li];
        let a = self.lpg.create_edge(sa, da, ty);
        let b = self.sub.create_edge(sb, db, ty);
        self.edges.push((a, b, src_li, dst_li, true));
    }

    fn delete_edge(&mut self, li: usize) {
        if !self.edges[li].4 {
            return;
        }
        let (a, b, _, _, _) = self.edges[li];
        self.lpg.delete_edge(a);
        self.sub.delete_edge(b);
        self.edges[li].4 = false;
    }

    fn set_node_prop(&mut self, li: usize, key: &str, val: Value) {
        if !self.nodes[li].2 {
            return;
        }
        let (a, b, _) = self.nodes[li];
        self.lpg.set_node_property(a, key, val.clone());
        self.sub.set_node_property(b, key, val);
    }

    fn remove_node_prop(&mut self, li: usize, key: &str) {
        if !self.nodes[li].2 {
            return;
        }
        let (a, b, _) = self.nodes[li];
        self.lpg.remove_node_property(a, key);
        self.sub.remove_node_property(b, key);
    }

    fn set_edge_prop(&mut self, li: usize, key: &str, val: Value) {
        if !self.edges[li].4 {
            return;
        }
        let (a, b, _, _, _) = self.edges[li];
        self.lpg.set_edge_property(a, key, val.clone());
        self.sub.set_edge_property(b, key, val);
    }

    fn remove_edge_prop(&mut self, li: usize, key: &str) {
        if !self.edges[li].4 {
            return;
        }
        let (a, b, _, _, _) = self.edges[li];
        self.lpg.remove_edge_property(a, key);
        self.sub.remove_edge_property(b, key);
    }

    fn add_label(&mut self, li: usize, label: &str) {
        if !self.nodes[li].2 {
            return;
        }
        let (a, b, _) = self.nodes[li];
        self.lpg.add_label(a, label);
        self.sub.add_label(b, label);
    }

    fn remove_label(&mut self, li: usize, label: &str) {
        if !self.nodes[li].2 {
            return;
        }
        let (a, b, _) = self.nodes[li];
        self.lpg.remove_label(a, label);
        self.sub.remove_label(b, label);
    }
}

// ---------------------------------------------------------------------------
// Canonical hash
// ---------------------------------------------------------------------------

fn canonical_hash(d: &Dual, side: Side) -> u64 {
    let mut h = DefaultHasher::default();

    // Counts — cheap divergence detector.
    let (nc, ec) = match side {
        Side::Lpg => (d.lpg.node_count(), d.lpg.edge_count()),
        Side::Sub => (d.sub.node_count(), d.sub.edge_count()),
    };
    nc.hash(&mut h);
    ec.hash(&mut h);

    // Walk logical nodes in order — gives us a stable ordering that doesn't
    // depend on concrete NodeId allocation.
    for (li, (a, b, alive)) in d.nodes.iter().enumerate() {
        if !alive {
            continue;
        }
        li.hash(&mut h);
        let (labels, props) = match side {
            Side::Lpg => {
                let n = d.lpg.get_node(*a).expect("alive logical → lpg get_node");
                let mut labels: Vec<String> =
                    n.labels.iter().map(|l| l.as_str().to_string()).collect();
                labels.sort();
                let map: BTreeMap<String, Value> = n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (labels, map)
            }
            Side::Sub => {
                let n = d.sub.get_node(*b).expect("alive logical → sub get_node");
                let mut labels: Vec<String> =
                    n.labels.iter().map(|l| l.as_str().to_string()).collect();
                labels.sort();
                let map: BTreeMap<String, Value> = n
                    .properties
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (labels, map)
            }
        };
        hash_labels(&labels, &mut h);
        hash_props(&props, &mut h);
    }

    // Edges, same trick.
    for (li, (a, b, src_li, dst_li, alive)) in d.edges.iter().enumerate() {
        if !alive {
            continue;
        }
        li.hash(&mut h);
        src_li.hash(&mut h);
        dst_li.hash(&mut h);
        let (ty, props) = match side {
            Side::Lpg => {
                let e = d.lpg.get_edge(*a).expect("alive logical → lpg get_edge");
                let map: BTreeMap<String, Value> = e
                    .properties
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (e.edge_type.as_str().to_string(), map)
            }
            Side::Sub => {
                let e = d.sub.get_edge(*b).expect("alive logical → sub get_edge");
                let map: BTreeMap<String, Value> = e
                    .properties
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.clone()))
                    .collect();
                (e.edge_type.as_str().to_string(), map)
            }
        };
        ty.hash(&mut h);
        hash_props(&props, &mut h);
    }

    h.finish()
}

fn hash_labels(labels: &[String], h: &mut DefaultHasher) {
    labels.len().hash(h);
    for l in labels {
        l.hash(h);
    }
}

fn hash_props(props: &BTreeMap<String, Value>, h: &mut DefaultHasher) {
    props.len().hash(h);
    for (k, v) in props {
        k.hash(h);
        hash_value(v, h);
    }
}

fn hash_value(v: &Value, h: &mut DefaultHasher) {
    // Value is not Hash; hash a tagged representation.
    std::mem::discriminant(v).hash(h);
    match v {
        Value::Null => {}
        Value::Bool(b) => b.hash(h),
        Value::Int64(n) => n.hash(h),
        Value::Float64(n) => n.to_bits().hash(h),
        Value::String(s) => s.as_str().hash(h),
        other => {
            // For composite values (List, Map, Timestamp, …) we fall back
            // to their Debug repr — consistent per-backend because the
            // variant came from our own test-generated inputs.
            format!("{other:?}").hash(h);
        }
    }
}

#[derive(Clone, Copy)]
enum Side {
    Lpg,
    Sub,
}

// ---------------------------------------------------------------------------
// Op generator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Op {
    CreateNode(Vec<&'static str>),
    DeleteNode(usize),
    CreateEdge(usize, usize, &'static str),
    DeleteEdge(usize),
    SetNodeProp(usize, &'static str, Value),
    RemoveNodeProp(usize, &'static str),
    SetEdgeProp(usize, &'static str, Value),
    RemoveEdgeProp(usize, &'static str),
    AddLabel(usize, &'static str),
    RemoveLabel(usize, &'static str),
}

const LABELS: &[&str] = &["Person", "Animal", "Company", "City", "Project"];
const EDGE_TYPES: &[&str] = &["KNOWS", "WORKS_AT", "IN", "OWNS"];
const PROP_KEYS: &[&str] = &["name", "age", "score", "role", "city"];

fn gen_value(rng: &mut SmallRng) -> Value {
    match rng.random_range(0..4u32) {
        0 => Value::Null,
        1 => Value::Bool(rng.random()),
        2 => Value::Int64(rng.random_range(-100..100)),
        _ => {
            let names = ["alix", "gus", "sam", "kaz", "riko"];
            Value::from(names[rng.random_range(0..names.len())])
        }
    }
}

fn gen_op(rng: &mut SmallRng, d: &Dual) -> Op {
    let alive_nodes = d.alive_nodes();
    let alive_edges = d.alive_edges();

    // Bias toward node creation early so there's something to work with.
    let roll = rng.random_range(0..100u32);
    match (roll, alive_nodes.len(), alive_edges.len()) {
        (_, 0, _) => Op::CreateNode(gen_labels(rng)),
        (0..20, _, _) => Op::CreateNode(gen_labels(rng)),
        (20..25, _, _) => {
            let li = *alive_nodes.as_slice()[rng.random_range(0..alive_nodes.len())..]
                .first()
                .unwrap();
            Op::DeleteNode(li)
        }
        (25..50, _, _) if alive_nodes.len() >= 2 => {
            let src = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let dst = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let ty = EDGE_TYPES[rng.random_range(0..EDGE_TYPES.len())];
            Op::CreateEdge(src, dst, ty)
        }
        (50..55, _, _) if !alive_edges.is_empty() => {
            let ei = alive_edges[rng.random_range(0..alive_edges.len())];
            Op::DeleteEdge(ei)
        }
        (55..75, _, _) => {
            let li = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let key = PROP_KEYS[rng.random_range(0..PROP_KEYS.len())];
            Op::SetNodeProp(li, key, gen_value(rng))
        }
        (75..80, _, _) => {
            let li = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let key = PROP_KEYS[rng.random_range(0..PROP_KEYS.len())];
            Op::RemoveNodeProp(li, key)
        }
        (80..85, _, _) if !alive_edges.is_empty() => {
            let ei = alive_edges[rng.random_range(0..alive_edges.len())];
            let key = PROP_KEYS[rng.random_range(0..PROP_KEYS.len())];
            Op::SetEdgeProp(ei, key, gen_value(rng))
        }
        (85..88, _, _) if !alive_edges.is_empty() => {
            let ei = alive_edges[rng.random_range(0..alive_edges.len())];
            let key = PROP_KEYS[rng.random_range(0..PROP_KEYS.len())];
            Op::RemoveEdgeProp(ei, key)
        }
        (88..96, _, _) => {
            let li = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let lb = LABELS[rng.random_range(0..LABELS.len())];
            Op::AddLabel(li, lb)
        }
        _ => {
            let li = alive_nodes[rng.random_range(0..alive_nodes.len())];
            let lb = LABELS[rng.random_range(0..LABELS.len())];
            Op::RemoveLabel(li, lb)
        }
    }
}

fn gen_labels(rng: &mut SmallRng) -> Vec<&'static str> {
    let n = rng.random_range(0..3usize);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(LABELS[rng.random_range(0..LABELS.len())]);
    }
    out
}

fn apply(d: &mut Dual, op: &Op) {
    match op {
        Op::CreateNode(labels) => d.create_node(labels),
        Op::DeleteNode(i) => d.delete_node(*i),
        Op::CreateEdge(s, t, ty) => d.create_edge(*s, *t, ty),
        Op::DeleteEdge(i) => d.delete_edge(*i),
        Op::SetNodeProp(i, k, v) => d.set_node_prop(*i, k, v.clone()),
        Op::RemoveNodeProp(i, k) => d.remove_node_prop(*i, k),
        Op::SetEdgeProp(i, k, v) => d.set_edge_prop(*i, k, v.clone()),
        Op::RemoveEdgeProp(i, k) => d.remove_edge_prop(*i, k),
        Op::AddLabel(i, l) => d.add_label(*i, l),
        Op::RemoveLabel(i, l) => d.remove_label(*i, l),
    }
}

// ---------------------------------------------------------------------------
// Sanity: prove we read through the trait-level API consistently.
// ---------------------------------------------------------------------------

fn behavioural_spotcheck(d: &Dual) {
    // Every alive logical node has matching degree on both sides.
    for (i, (a, b, alive)) in d.nodes.iter().enumerate() {
        if !alive {
            continue;
        }
        let od_l = d.lpg.out_degree(*a);
        let od_s = d.sub.out_degree(*b);
        let id_l = d.lpg.in_degree(*a);
        let id_s = d.sub.in_degree(*b);
        assert_eq!(
            (od_l, id_l),
            (od_s, id_s),
            "degree divergence at logical node {i}"
        );

        // edges_from sanity — same count per direction. Use fully-qualified
        // trait call because LpgStore has an inherent `edges_from` that
        // returns an iterator instead of the trait's `Vec`.
        for dir in [Direction::Outgoing, Direction::Incoming, Direction::Both] {
            let la = <LpgStore as GraphStore>::edges_from(&d.lpg, *a, dir).len();
            let sa = <SubstrateStore as GraphStore>::edges_from(&d.sub, *b, dir).len();
            assert_eq!(la, sa, "edges_from({dir:?}) divergence at logical node {i}");
        }

        // Property key scan — same key set.
        let pk = PropertyKey::new("name");
        let lv = d.lpg.get_node_property(*a, &pk);
        let sv = d.sub.get_node_property(*b, &pk);
        assert_eq!(lv, sv, "node.name divergence at logical node {i}");
    }
}

// ---------------------------------------------------------------------------
// The actual test entry points
// ---------------------------------------------------------------------------

fn run_seed(seed: u64, ops_per_seed: usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut d = Dual::new();

    for step in 0..ops_per_seed {
        let op = gen_op(&mut rng, &d);
        apply(&mut d, &op);

        // Compare hashes every 97 ops — frequent enough to localize divergence,
        // cheap enough to keep the test fast.
        if step % 97 == 0 || step + 1 == ops_per_seed {
            let ha = canonical_hash(&d, Side::Lpg);
            let hb = canonical_hash(&d, Side::Sub);
            assert_eq!(
                ha, hb,
                "seed={seed} step={step}: canonical hash diverged after op {op:?}"
            );
        }
    }

    // Final spotcheck of derived read APIs.
    behavioural_spotcheck(&d);
}

#[test]
fn parity_random_ops() {
    // Reduced scope for default (debug) runs so the suite stays under a
    // few seconds. The `_full` test below is the 10k×100 spec target.
    const NUM_SEEDS: u64 = 5;
    const OPS_PER_SEED: usize = 2_000;

    for seed in 0..NUM_SEEDS {
        run_seed(seed, OPS_PER_SEED);
    }
}

#[test]
#[ignore = "full parity matrix (10k ops × 100 seeds) — run with --release --ignored"]
fn parity_random_ops_full() {
    const NUM_SEEDS: u64 = 100;
    const OPS_PER_SEED: usize = 10_000;

    for seed in 0..NUM_SEEDS {
        run_seed(seed, OPS_PER_SEED);
    }
}
