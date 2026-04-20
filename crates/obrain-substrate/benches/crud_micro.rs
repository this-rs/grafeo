//! CRUD micro-benchmarks — SubstrateStore vs LpgStore.
//!
//! # What we measure
//!
//! Each operation is benchmarked on both backends, exposing a per-op delta.
//! The scope is deliberately narrow — singleton CRUD on a warm store — so
//! reviewers can see where mmap/WAL infra buys back the cost at the margin.
//!
//! - `create_node`       — allocate + label-set write
//! - `create_edge`       — allocate + chain splice
//! - `set_node_property` — heap append + property page slot write
//! - `get_node`          — read-path: deref node slot + labels + properties
//! - `edges_from_out`    — intrusive chain walk (outgoing direction)
//!
//! # Guardrail
//!
//! Acceptance criterion from T4 Step 6:
//! `delta ≤ 2× LpgStore` on each individual op. The gain from the substrate
//! family comes later — startup time, recall, persistence, mmap sharing —
//! not from CRUD singletons themselves.
//!
//! # Running
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench crud_micro
//! ```
//!
//! Filter to a single op:
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench crud_micro -- create_node
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::LpgStore;
use obrain_core::graph::Direction;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::SubstrateStore;

// ---------------------------------------------------------------------------
// Scaffolding — one store + its backing tempdir per backend.
// ---------------------------------------------------------------------------

struct SubHarness {
    _td: tempfile::TempDir,
    store: SubstrateStore,
}

fn fresh_sub() -> SubHarness {
    let td = tempfile::tempdir().unwrap();
    let store = SubstrateStore::create(td.path().join("kb")).unwrap();
    SubHarness { _td: td, store }
}

fn warm_sub(n_nodes: usize, n_edges_per_node: usize) -> SubHarness {
    let h = fresh_sub();
    let nodes: Vec<NodeId> = (0..n_nodes).map(|_| h.store.create_node(&["P"])).collect();
    for (i, src) in nodes.iter().enumerate() {
        for j in 0..n_edges_per_node {
            let dst = nodes[(i + j + 1) % nodes.len()];
            h.store.create_edge(*src, dst, "R");
        }
    }
    h
}

fn fresh_lpg() -> LpgStore {
    LpgStore::new().unwrap()
}

fn warm_lpg(n_nodes: usize, n_edges_per_node: usize) -> LpgStore {
    let s = fresh_lpg();
    let nodes: Vec<NodeId> = (0..n_nodes).map(|_| s.create_node(&["P"])).collect();
    for (i, src) in nodes.iter().enumerate() {
        for j in 0..n_edges_per_node {
            let dst = nodes[(i + j + 1) % nodes.len()];
            s.create_edge(*src, dst, "R");
        }
    }
    s
}

// ---------------------------------------------------------------------------
// create_node
// ---------------------------------------------------------------------------

fn bench_create_node(c: &mut Criterion) {
    let mut g = c.benchmark_group("create_node");

    g.bench_function("lpg", |b| {
        let s = fresh_lpg();
        b.iter(|| {
            black_box(s.create_node(black_box(&["Person"])));
        });
    });

    g.bench_function("substrate", |b| {
        let h = fresh_sub();
        b.iter(|| {
            black_box(h.store.create_node(black_box(&["Person"])));
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// create_edge — pre-populate 1k nodes so allocation isn't the dominant cost.
// ---------------------------------------------------------------------------

fn bench_create_edge(c: &mut Criterion) {
    let mut g = c.benchmark_group("create_edge");

    g.bench_function("lpg", |b| {
        let s = warm_lpg(1024, 0);
        let a = NodeId(0);
        let dst_ids: Vec<NodeId> = (0..1024).map(NodeId).collect();
        let mut i = 0usize;
        b.iter(|| {
            let d = dst_ids[i % dst_ids.len()];
            i = i.wrapping_add(1);
            black_box(s.create_edge(a, d, black_box("R")));
        });
    });

    g.bench_function("substrate", |b| {
        let h = warm_sub(1024, 0);
        // We can't assume NodeId(0) is alive in substrate (0 is null). Pull
        // real ids from `node_ids()`.
        let ids: Vec<NodeId> = h.store.node_ids();
        let a = ids[0];
        let mut i = 0usize;
        b.iter(|| {
            let d = ids[i % ids.len()];
            i = i.wrapping_add(1);
            black_box(h.store.create_edge(a, d, black_box("R")));
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// set_node_property
// ---------------------------------------------------------------------------

fn bench_set_node_property(c: &mut Criterion) {
    let mut g = c.benchmark_group("set_node_property");

    g.bench_function("lpg", |b| {
        let s = warm_lpg(1024, 0);
        let target = NodeId(0);
        let mut i = 0i64;
        b.iter(|| {
            i = i.wrapping_add(1);
            s.set_node_property(target, black_box("score"), Value::Int64(i));
        });
    });

    g.bench_function("substrate", |b| {
        let h = warm_sub(1024, 0);
        let ids = h.store.node_ids();
        let target = ids[0];
        let mut i = 0i64;
        b.iter(|| {
            i = i.wrapping_add(1);
            h.store
                .set_node_property(target, black_box("score"), Value::Int64(i));
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// get_node — read-path hot loop.
// ---------------------------------------------------------------------------

fn bench_get_node(c: &mut Criterion) {
    let mut g = c.benchmark_group("get_node");

    g.bench_function("lpg", |b| {
        let s = warm_lpg(4096, 0);
        let ids = s.node_ids();
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            black_box(s.get_node(id));
        });
    });

    g.bench_function("substrate", |b| {
        let h = warm_sub(4096, 0);
        let ids = h.store.node_ids();
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            black_box(h.store.get_node(id));
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// edges_from (outgoing) — intrusive chain walk.
// ---------------------------------------------------------------------------

fn bench_edges_from(c: &mut Criterion) {
    let mut g = c.benchmark_group("edges_from_out");

    const N_NODES: usize = 1024;
    const FANOUT: usize = 8;

    g.bench_function("lpg", |b| {
        let s = warm_lpg(N_NODES, FANOUT);
        let ids = s.node_ids();
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            // Use trait call explicitly: LpgStore has an inherent method
            // `edges_from` returning an iterator that would shadow the trait's
            // `Vec<(NodeId, EdgeId)>` return type.
            let v = <LpgStore as GraphStore>::edges_from(&s, id, Direction::Outgoing);
            black_box(v.len());
        });
    });

    g.bench_function("substrate", |b| {
        let h = warm_sub(N_NODES, FANOUT);
        let ids = h.store.node_ids();
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            let v = h.store.edges_from(id, Direction::Outgoing);
            black_box(v.len());
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// get_node_property — single-value point read after write.
// ---------------------------------------------------------------------------

fn bench_get_node_property(c: &mut Criterion) {
    let mut g = c.benchmark_group("get_node_property");

    g.bench_function("lpg", |b| {
        let s = warm_lpg(4096, 0);
        let ids = s.node_ids();
        for (i, id) in ids.iter().enumerate() {
            s.set_node_property(*id, "score", Value::Int64(i as i64));
        }
        let key = PropertyKey::new("score");
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            black_box(s.get_node_property(id, &key));
        });
    });

    g.bench_function("substrate", |b| {
        let h = warm_sub(4096, 0);
        let ids = h.store.node_ids();
        for (i, id) in ids.iter().enumerate() {
            h.store.set_node_property(*id, "score", Value::Int64(i as i64));
        }
        let key = PropertyKey::new("score");
        let mut i = 0usize;
        b.iter(|| {
            let id = ids[i % ids.len()];
            i = i.wrapping_add(1);
            black_box(h.store.get_node_property(id, &key));
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_create_node,
    bench_create_edge,
    bench_set_node_property,
    bench_get_node,
    bench_get_node_property,
    bench_edges_from,
);
criterion_main!(benches);
