//! GDS (Graph Data Science) algorithm benchmarks.
//!
//! Measures the performance of PageRank, Louvain, Leiden, and Betweenness Centrality
//! on synthetic graphs of increasing size (Barabási-Albert scale-free model).
//!
//! Run with: cargo bench -p obrain-adapters --features "algos,parallel"

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use obrain_core::LpgStore;

use obrain_adapters::plugins::algorithms::{
    betweenness_centrality, contract_subgraph, leiden, louvain, pagerank, personalized_pagerank,
    stabilize_communities, ContractionConfig, PprConfig,
};

/// Creates a Barabási-Albert scale-free graph with `n` nodes and `m` edges per new node.
///
/// The BA model produces power-law degree distributions typical of real-world networks
/// (social graphs, citation networks, the web).
fn barabasi_albert(n: usize, m: usize) -> LpgStore {
    let store = LpgStore::new().expect("LpgStore::new failed");

    // Create initial complete graph of m+1 nodes
    let mut node_ids = Vec::with_capacity(n);
    for _ in 0..=m {
        node_ids.push(store.create_node(&["Node"]));
    }
    // Connect initial nodes fully
    for i in 0..=m {
        for j in (i + 1)..=m {
            store.create_edge(node_ids[i], node_ids[j], "CONNECTED");
        }
    }

    // Degree array for preferential attachment (repeating IDs proportional to degree)
    let mut degree_bag: Vec<usize> = Vec::new();
    for i in 0..=m {
        for _ in 0..m {
            degree_bag.push(i);
        }
    }

    // Add remaining nodes with preferential attachment
    let mut rng_state: u64 = 42; // Simple LCG for deterministic benchmarks
    for _ in (m + 1)..n {
        let new_node = store.create_node(&["Node"]);
        let new_idx = node_ids.len();
        node_ids.push(new_node);

        // Pick m distinct targets via preferential attachment
        let mut targets = Vec::with_capacity(m);
        let mut attempts = 0;
        while targets.len() < m && attempts < m * 10 {
            // LCG random
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state >> 33) as usize % degree_bag.len();
            let target = degree_bag[idx];
            if target != new_idx && !targets.contains(&target) {
                targets.push(target);
            }
            attempts += 1;
        }

        for &target in &targets {
            store.create_edge(new_node, node_ids[target], "CONNECTED");
            degree_bag.push(new_idx);
            degree_bag.push(target);
        }
    }

    store
}

// ============================================================================
// PageRank
// ============================================================================

fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/pagerank");
    group.sample_size(10);

    for &size in &[1_000, 5_000, 10_000] {
        let store = barabasi_albert(size, 3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| pagerank(store, 0.85, 20, 1e-6));
        });
    }
    group.finish();
}

// ============================================================================
// Louvain community detection
// ============================================================================

fn bench_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/louvain");
    group.sample_size(10);

    for &size in &[1_000, 5_000, 10_000] {
        let store = barabasi_albert(size, 3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| louvain(store, 1.0));
        });
    }
    group.finish();
}

// ============================================================================
// Leiden community detection
// ============================================================================

fn bench_leiden(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/leiden");
    group.sample_size(10);

    for &size in &[1_000, 5_000, 10_000] {
        let store = barabasi_albert(size, 3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| leiden(store, 1.0, 0.01));
        });
    }
    group.finish();
}

// ============================================================================
// Betweenness Centrality
// ============================================================================

fn bench_betweenness(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/betweenness");
    group.sample_size(10);

    // Betweenness is O(V*E), so use smaller graphs
    for &size in &[500, 1_000, 2_000] {
        let store = barabasi_albert(size, 3);
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| betweenness_centrality(store, true));
        });
    }
    group.finish();
}

// ============================================================================
// Stable Communities
// ============================================================================

fn bench_stable_communities(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/stable_communities");
    group.sample_size(10);

    for &size in &[500, 2_000, 10_000] {
        let store = barabasi_albert(size, 3);
        // Pre-compute a "previous" Louvain result
        let prev = louvain(&store, 1.0);
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| {
                let curr = louvain(store, 1.0);
                stabilize_communities(&prev, &curr, store, 0.3)
            });
        });
    }
    group.finish();
}

// ============================================================================
// Personalized PageRank
// ============================================================================

fn bench_ppr(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/ppr");
    group.sample_size(10);

    for &size in &[500, 2_000, 10_000] {
        let store = barabasi_albert(size, 3);
        let node_ids = store.node_ids();
        // Pick 3 deterministic seeds
        let seeds = vec![node_ids[0], node_ids[size / 3], node_ids[2 * size / 3]];
        let config = PprConfig {
            budget: 50,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::from_parameter(size), &store, |b, store| {
            b.iter(|| personalized_pagerank(store, &seeds, &config));
        });
    }
    group.finish();
}

// ============================================================================
// Subgraph Contraction
// ============================================================================

fn bench_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("gds/contraction");
    group.sample_size(10);

    for &size in &[500, 2_000, 10_000] {
        // Build graph, pick a Louvain community to contract
        let store = barabasi_albert(size, 3);
        let result = louvain(&store, 1.0);
        // Find the largest community
        let mut comm_nodes: std::collections::HashMap<u64, Vec<obrain_common::types::NodeId>> =
            std::collections::HashMap::new();
        for (&nid, &comm) in &result.communities {
            comm_nodes.entry(comm).or_default().push(nid);
        }
        let largest: Vec<obrain_common::types::NodeId> = comm_nodes
            .into_values()
            .max_by_key(|v| v.len())
            .unwrap_or_default();

        let config = ContractionConfig::default();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                // Rebuild graph each iteration since contraction is destructive
                let s = barabasi_albert(size, 3);
                // Re-use the same node IDs (deterministic BA graph)
                let ids = s.node_ids();
                let contract_ids: Vec<obrain_common::types::NodeId> = largest
                    .iter()
                    .filter_map(|nid| ids.get(nid.0 as usize).copied())
                    .collect();
                if !contract_ids.is_empty() {
                    let _ = contract_subgraph(&s, &contract_ids, &config);
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pagerank,
    bench_louvain,
    bench_leiden,
    bench_betweenness,
    bench_stable_communities,
    bench_ppr,
    bench_contraction
);
criterion_main!(benches);
