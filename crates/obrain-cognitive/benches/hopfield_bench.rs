//! Benchmarks for Modern Hopfield retrieval.
//!
//! Measures retrieve latency for 100, 500, and 1000 engrams.
//! Target: < 50ms for 1000 engrams.
//!
//! Run with: cargo bench -p grafeo-cognitive --features hopfield -- hopfield

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grafeo_cognitive::engram::{
    Engram, EngramStore, PatternMatrix, SpectralEncoder, hopfield_retrieve,
};
use grafeo_common::types::NodeId;

const DIM: usize = 64;

/// Build a store + pattern matrix with `n` engrams, each having a random-ish
/// spectral signature of dimension `DIM`.
fn build_store_and_matrix(n: usize) -> (EngramStore, PatternMatrix, Vec<f64>) {
    let store = EngramStore::new(None);
    let spectral = SpectralEncoder::new();
    let mut matrix = PatternMatrix::new(DIM);

    for i in 1..=n {
        let id = store.next_id();
        // Create diverse ensembles: each engram has 3 nodes offset by i
        let base = i as u64;
        let nodes: Vec<(NodeId, f64)> = (0..3)
            .map(|j| (NodeId(base * 7 + j), 1.0 / (j as f64 + 1.0)))
            .collect();

        let mut engram = Engram::new(id, nodes.clone());
        let sig = spectral.encode(&nodes);
        engram.spectral_signature.clone_from(&sig);
        // Vary precision: β ∈ [0.5, 5.0]
        engram.precision = 0.5 + (i as f64 % 10.0) * 0.5;

        matrix.add_pattern(id, &sig, engram.precision);
        store.insert(engram);
    }

    // Build a query vector from some arbitrary nodes
    let query_nodes: Vec<(NodeId, f64)> =
        vec![(NodeId(42), 1.0), (NodeId(99), 0.8), (NodeId(7), 0.5)];
    let query_vec = spectral.encode(&query_nodes);

    (store, matrix, query_vec)
}

fn bench_hopfield_retrieve(c: &mut Criterion) {
    let mut group = c.benchmark_group("hopfield_retrieve");
    group.significance_level(0.05);

    for &n in &[100, 500, 1000] {
        let (store, matrix, query) = build_store_and_matrix(n);

        group.bench_with_input(BenchmarkId::new("retrieve_top10", n), &n, |b, _| {
            b.iter(|| {
                let results = hopfield_retrieve(&matrix, &query, &store, 10);
                std::hint::black_box(results);
            });
        });
    }

    group.finish();
}

fn bench_pattern_matrix_from_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("hopfield_matrix_build");

    for &n in &[100, 500, 1000] {
        let (store, _, _) = build_store_and_matrix(n);

        group.bench_with_input(BenchmarkId::new("from_store", n), &n, |b, _| {
            b.iter(|| {
                let matrix = PatternMatrix::from_store(&store, DIM);
                std::hint::black_box(matrix);
            });
        });
    }

    group.finish();
}

criterion_group!(
    hopfield_benches,
    bench_hopfield_retrieve,
    bench_pattern_matrix_from_store
);
criterion_main!(hopfield_benches);
