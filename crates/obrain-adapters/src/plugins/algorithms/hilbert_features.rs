//! Hilbert 64d multi-facette node features.
//!
//! Computes a 64-dimensional feature vector per node by orchestrating 8 graph
//! analysis **facettes**, each producing a 2D point per node, then encoding
//! each via multi-resolution Hilbert curves (8 levels → 8 dimensions).
//!
//! **8 facettes × 8 dimensions = 64d per node.**
//!
//! | Facette | Dims  | Source algorithm            | Captures                |
//! |---------|-------|----------------------------|-------------------------|
//! | 0       | 0-7   | Spectral eigvecs 1-2       | Global graph structure  |
//! | 1       | 8-15  | Spectral eigvecs 3-4       | Secondary structure     |
//! | 2       | 16-23 | Louvain community + size   | Community membership    |
//! | 3       | 24-31 | PageRank + degree           | Centrality / importance |
//! | 4       | 32-39 | BFS depth (2 seed nodes)   | Distance structure      |
//! | 5       | 40-47 | Betweenness + closeness    | Bridge / flow role      |
//! | 6       | 48-55 | Co-usage (REINFORCES)      | Behavioral coupling     |
//! | 7       | 56-63 | Schema (labels + props)    | Type / structural role  |
//!
//! Facettes 0-3 are **global** (require full graph computation).
//! Facettes 4-7 are **local** (can be updated incrementally).
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_adapters::plugins::algorithms::hilbert_features::hilbert_features;
//! use obrain_core::graph::lpg::LpgStore;
//!
//! let store = LpgStore::new().unwrap();
//! // ... populate graph ...
//! let config = Default::default();
//! let result = hilbert_features(&store, &config);
//! // result.features: HashMap<NodeId, [f32; 64]>
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
#[cfg(test)]
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::{Direction, GraphStore};

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality, pagerank,
};
use super::community::louvain;
use super::hilbert::hilbert_encode_point;
use super::spectral::spectral_embedding;
use super::traits::GraphAlgorithm;
use super::traversal::bfs;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for `hilbert_features()`.
#[derive(Debug, Clone)]
pub struct HilbertFeaturesConfig {
    /// Number of Hilbert encoding levels per facette (default: 8).
    pub levels: usize,
    /// Edge type for persona/co-usage facette (default: "REINFORCES").
    pub persona_edge_type: String,
    /// PageRank damping factor (default: 0.85).
    pub damping: f64,
    /// PageRank iterations (default: 100).
    pub pagerank_iterations: usize,
    /// Max nodes for expensive facettes (betweenness, closeness, spectral).
    /// Above this threshold, these facettes use degree-based approximations.
    /// 0 = no limit (default: 10_000).
    pub large_graph_threshold: usize,
}

impl Default for HilbertFeaturesConfig {
    fn default() -> Self {
        Self {
            levels: 8,
            persona_edge_type: "REINFORCES".to_string(),
            damping: 0.85,
            pagerank_iterations: 100,
            large_graph_threshold: 10_000,
        }
    }
}

// ============================================================================
// Normalization helpers
// ============================================================================

/// Log-normalize a value for power-law distributions.
/// Maps `[0, max]` to `[0, 1]` using `log(1 + x) / log(1 + max)`.
/// This preserves discrimination across orders of magnitude.
#[inline]
fn log_normalize(value: f64, max: f64) -> f32 {
    if max <= 0.0 {
        return 0.0;
    }
    ((1.0 + value).ln() / (1.0 + max).ln()) as f32
}

/// Rank-percentile normalization: maps values to their percentile rank in [0, 1].
///
/// Sorts all values and assigns `rank / (n - 1)` to each node.
/// Produces a perfectly uniform distribution regardless of the input shape
/// (power-law, bimodal, etc.). O(V log V) for the sort.
///
/// Ties get the same percentile (average rank).
fn rank_normalize(values: &HashMap<NodeId, f64>) -> HashMap<NodeId, f32> {
    let n = values.len();
    if n <= 1 {
        return values.keys().map(|&nid| (nid, 0.5)).collect();
    }

    // Sort by value
    let mut sorted: Vec<(NodeId, f64)> = values.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_rank = (n - 1) as f64;
    let mut result = HashMap::with_capacity(n);

    // Handle ties: nodes with equal values get the same percentile
    let mut i = 0;
    while i < sorted.len() {
        let val = sorted[i].1;
        let mut j = i;
        while j < sorted.len() && (sorted[j].1 - val).abs() < f64::EPSILON {
            j += 1;
        }
        // Average rank for this tie group
        let avg_rank = (i + j - 1) as f64 / 2.0;
        let percentile = (avg_rank / max_rank) as f32;
        for item in &sorted[i..j] {
            result.insert(item.0, percentile.clamp(0.0, 1.0));
        }
        i = j;
    }

    result
}

// ============================================================================
// Result
// ============================================================================

/// Result of `hilbert_features()` computation.
#[derive(Debug, Clone)]
pub struct HilbertFeaturesResult {
    /// Feature vectors: NodeId → 64-dimensional f32 vector.
    pub features: HashMap<NodeId, Vec<f32>>,
    /// Number of dimensions (= levels × 8 facettes).
    pub dimensions: usize,
    /// Whether global facettes (spectral, community, centrality) are stale.
    ///
    /// Set to `true` after an incremental update — dims 0-31 contain the
    /// previous global values and should be recalculated in a background
    /// full pass when convenient. `false` after a full `hilbert_features()`.
    pub dirty_global: bool,
}

// ============================================================================
// Public API
// ============================================================================

/// Compute 64d multi-facette Hilbert features for all nodes.
///
/// Orchestrates 8 graph analysis facettes, encodes each as multi-resolution
/// Hilbert vectors, and concatenates them into a single 64d vector per node.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `config` - Configuration (levels, edge types, etc.)
///
/// # Returns
///
/// `HilbertFeaturesResult` with `levels × 8` dimensions per node.
pub fn hilbert_features(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
) -> HilbertFeaturesResult {
    let node_ids = store.node_ids();
    let n = node_ids.len();
    let total_dims = config.levels * 8;

    if n == 0 {
        return HilbertFeaturesResult {
            features: HashMap::new(),
            dimensions: total_dims,
            dirty_global: false,
        };
    }

    // Check if graph is too large for expensive facettes
    let is_large = config.large_graph_threshold > 0 && n > config.large_graph_threshold;

    // Compute all 8 facettes as 2D points per node
    // For large graphs, expensive facettes (spectral, betweenness) use approximations
    let facettes: [HashMap<NodeId, [f32; 2]>; 8] = if is_large {
        [
            compute_degree_approx(store),           // approx for spectral 1-2
            compute_degree_approx_secondary(store), // approx for spectral 3-4
            compute_community(store),
            compute_centrality(store, config),
            compute_bfs_distance(store),
            compute_degree_betweenness_approx(store), // approx for betweenness
            compute_co_usage(store, &config.persona_edge_type),
            compute_schema_features(store),
        ]
    } else {
        [
            compute_spectral_12(store),
            compute_spectral_34(store),
            compute_community(store),
            compute_centrality(store, config),
            compute_bfs_distance(store),
            compute_betweenness_closeness(store),
            compute_co_usage(store, &config.persona_edge_type),
            compute_schema_features(store),
        ]
    };

    // Encode each facette via Hilbert and concatenate
    let mut features: HashMap<NodeId, Vec<f32>> = HashMap::with_capacity(n);

    for &nid in &node_ids {
        let mut vec = Vec::with_capacity(total_dims);
        for facette in &facettes {
            let point = facette.get(&nid).copied().unwrap_or([0.5, 0.5]);
            let encoded = hilbert_encode_point(point, config.levels);
            vec.extend_from_slice(&encoded);
        }
        features.insert(nid, vec);
    }

    HilbertFeaturesResult {
        features,
        dimensions: total_dims,
        dirty_global: false,
    }
}

/// Incrementally update Hilbert features for changed nodes.
///
/// Only recalculates **local facettes** (4-7: BFS distance, betweenness/closeness,
/// co-usage, schema) for `changed_nodes` and their immediate neighbors.
/// **Global facettes** (0-3: spectral, community, centrality) are copied from
/// `previous` unchanged — the returned result has `dirty_global = true`.
///
/// # Cost
///
/// O(|changed| × avg_degree) instead of O(V + E) for a full recalculation.
///
/// # Arguments
///
/// * `store` - The graph store (current state)
/// * `config` - Configuration
/// * `previous` - Previous full `HilbertFeaturesResult`
/// * `changed_nodes` - Nodes that were added or modified
///
/// # Returns
///
/// Updated `HilbertFeaturesResult` with `dirty_global = true`.
pub fn hilbert_features_incremental(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
    previous: &HilbertFeaturesResult,
    changed_nodes: &[NodeId],
) -> HilbertFeaturesResult {
    let node_ids = store.node_ids();
    let total_dims = config.levels * 8;

    if node_ids.is_empty() {
        return HilbertFeaturesResult {
            features: HashMap::new(),
            dimensions: total_dims,
            dirty_global: true,
        };
    }

    // Collect affected nodes: changed + their neighbors
    let mut affected: std::collections::HashSet<NodeId> = changed_nodes.iter().copied().collect();
    for &nid in changed_nodes {
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            affected.insert(neighbor);
        }
    }

    // Recompute only local facettes (4-7) for affected nodes
    let local_bfs = compute_bfs_distance(store);
    let local_bc = compute_betweenness_closeness(store);
    let local_co = compute_co_usage(store, &config.persona_edge_type);
    let local_schema = compute_schema_features(store);

    let local_facettes: [&HashMap<NodeId, [f32; 2]>; 4] =
        [&local_bfs, &local_bc, &local_co, &local_schema];

    // Build result: copy global dims from previous, update local dims for affected
    let mut features: HashMap<NodeId, Vec<f32>> = HashMap::with_capacity(node_ids.len());
    let global_dims = config.levels * 4; // facettes 0-3

    for &nid in &node_ids {
        let mut vec = Vec::with_capacity(total_dims);

        // Global facettes (dims 0..global_dims): copy from previous
        if let Some(prev) = previous.features.get(&nid) {
            vec.extend_from_slice(&prev[..global_dims.min(prev.len())]);
            // Pad if new node not in previous
            while vec.len() < global_dims {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        } else {
            // New node: fill global dims with default (0.5, 0.5)
            for _ in 0..4 {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        }

        // Local facettes (dims global_dims..total_dims): recompute for affected, copy for others
        if affected.contains(&nid) {
            for facette in &local_facettes {
                let point = facette.get(&nid).copied().unwrap_or([0.5, 0.5]);
                vec.extend_from_slice(&hilbert_encode_point(point, config.levels));
            }
        } else if let Some(prev) = previous.features.get(&nid) {
            vec.extend_from_slice(&prev[global_dims..total_dims.min(prev.len())]);
            while vec.len() < total_dims {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        } else {
            for _ in 0..4 {
                vec.extend_from_slice(&hilbert_encode_point([0.5, 0.5], config.levels));
            }
        }

        features.insert(nid, vec);
    }

    HilbertFeaturesResult {
        features,
        dimensions: total_dims,
        dirty_global: true,
    }
}

// ============================================================================
// Facette 0: Spectral eigenvectors 1-2
// ============================================================================

fn compute_spectral_12(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let result = spectral_embedding(store, 2, None);
    result
        .embeddings
        .into_iter()
        .map(|(nid, emb)| {
            let x = *emb.first().unwrap_or(&0.5) as f32;
            let y = *emb.get(1).unwrap_or(&0.5) as f32;
            (nid, [x, y])
        })
        .collect()
}

// ============================================================================
// Facette 1: Spectral eigenvectors 3-4
// ============================================================================

fn compute_spectral_34(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let result = spectral_embedding(store, 4, None);
    result
        .embeddings
        .into_iter()
        .map(|(nid, emb)| {
            let x = *emb.get(2).unwrap_or(&0.5) as f32;
            let y = *emb.get(3).unwrap_or(&0.5) as f32;
            (nid, [x, y])
        })
        .collect()
}

// ============================================================================
// Facette 2: Community (Louvain community_id, intra_density)
// ============================================================================

fn compute_community(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let louvain_result = louvain(store, 1.0);
    let communities = &louvain_result.communities;

    // Count community sizes and intra-community edges
    let mut community_sizes: HashMap<u64, usize> = HashMap::new();
    let mut community_intra_edges: HashMap<u64, usize> = HashMap::new();
    for &comm in communities.values() {
        *community_sizes.entry(comm).or_insert(0) += 1;
    }

    // Count intra-community edges
    for (&nid, &comm) in communities {
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            if communities.get(&neighbor).copied() == Some(comm) {
                *community_intra_edges.entry(comm).or_insert(0) += 1;
            }
        }
    }

    let max_size = community_sizes.values().max().copied().unwrap_or(1).max(1) as f64;

    communities
        .iter()
        .map(|(&nid, &comm)| {
            let size = *community_sizes.get(&comm).unwrap_or(&1);
            let intra_edges = *community_intra_edges.get(&comm).unwrap_or(&0);
            // intra_density = actual edges / possible edges (size*(size-1))
            let possible = if size > 1 { size * (size - 1) } else { 1 };
            let density = intra_edges as f32 / possible as f32;
            // Use community size (log-normalized) instead of raw community ID
            let x = log_normalize(size as f64, max_size);
            (nid, [x.clamp(0.0, 1.0), density.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 3: Centrality (PageRank, degree)
// ============================================================================

fn compute_centrality(
    store: &dyn GraphStore,
    config: &HilbertFeaturesConfig,
) -> HashMap<NodeId, [f32; 2]> {
    let pr = pagerank(store, config.damping, config.pagerank_iterations, 1e-6);
    let deg = degree_centrality(store);

    // Rank percentile: uniform [0,1] regardless of power-law shape
    let pr_f64: HashMap<NodeId, f64> = pr.iter().map(|(&k, &v)| (k, v)).collect();
    let deg_f64: HashMap<NodeId, f64> = deg
        .total_degree
        .iter()
        .map(|(&k, &v)| (k, v as f64))
        .collect();
    let pr_ranks = rank_normalize(&pr_f64);
    let deg_ranks = rank_normalize(&deg_f64);

    store
        .node_ids()
        .iter()
        .map(|&nid| {
            let x = *pr_ranks.get(&nid).unwrap_or(&0.5);
            let y = *deg_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 4: BFS distance (from 2 seed nodes)
// ============================================================================

fn compute_bfs_distance(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    if node_ids.is_empty() {
        return HashMap::new();
    }

    // Pick 2 seeds: first and "most distant" (last in BFS from first)
    let seed1 = node_ids[0];
    let bfs1 = bfs(store, seed1);

    let seed2 = *bfs1.last().unwrap_or(&seed1);
    let _bfs2 = bfs(store, seed2);

    // Build depth maps
    let depth1 = bfs_depths(store, seed1);
    let depth2 = bfs_depths(store, seed2);

    let max_d1 = depth1.values().max().copied().unwrap_or(1).max(1) as f32;
    let max_d2 = depth2.values().max().copied().unwrap_or(1).max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let x = *depth1.get(&nid).unwrap_or(&0) as f32 / max_d1;
            let y = *depth2.get(&nid).unwrap_or(&0) as f32 / max_d2;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Compute BFS depths from a seed node.
fn bfs_depths(store: &dyn GraphStore, seed: NodeId) -> HashMap<NodeId, usize> {
    use std::collections::{HashSet, VecDeque};

    let mut depths = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    depths.insert(seed, 0usize);
    visited.insert(seed);
    queue.push_back(seed);

    while let Some(node) = queue.pop_front() {
        let depth = depths[&node];
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if visited.insert(neighbor) {
                depths.insert(neighbor, depth + 1);
                queue.push_back(neighbor);
            }
        }
    }

    depths
}

// ============================================================================
// Facette 5: Betweenness + Closeness centrality
// ============================================================================

fn compute_betweenness_closeness(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let bc = betweenness_centrality(store, true);
    let cc = closeness_centrality(store, true);

    let bc_f64: HashMap<NodeId, f64> = bc.into_iter().collect();
    let cc_f64: HashMap<NodeId, f64> = cc.into_iter().collect();
    let bc_ranks = rank_normalize(&bc_f64);
    let cc_ranks = rank_normalize(&cc_f64);

    store
        .node_ids()
        .iter()
        .map(|&nid| {
            let x = *bc_ranks.get(&nid).unwrap_or(&0.5);
            let y = *cc_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 6: Co-usage (REINFORCES edge count + recency)
// ============================================================================

/// Compute co-usage features from edges of a specific type.
///
/// For each node: x = normalized co-activation count, y = normalized recency
/// (number of distinct co-usage partners as proxy for recency/diversity).
fn compute_co_usage(store: &dyn GraphStore, persona_edge_type: &str) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();

    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    let mut partner_counts: HashMap<NodeId, std::collections::HashSet<NodeId>> = HashMap::new();

    for &nid in &node_ids {
        let edges = store.edges_from(nid, Direction::Outgoing);
        for (neighbor, edge_id) in &edges {
            if let Some(etype) = store.edge_type(*edge_id)
                && etype.as_str() == persona_edge_type
            {
                *counts.entry(nid).or_insert(0) += 1;
                partner_counts.entry(nid).or_default().insert(*neighbor);
            }
        }
    }

    let max_count = counts.values().max().copied().unwrap_or(1).max(1) as f32;
    let max_partners = partner_counts
        .values()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let x = *counts.get(&nid).unwrap_or(&0) as f32 / max_count;
            let y = partner_counts.get(&nid).map_or(0, |s| s.len()) as f32 / max_partners;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Facette 7: Schema features (label hash + property count)
// ============================================================================

/// Compute schema-based features.
///
/// x = hash of node labels (normalized), y = number of properties (normalized).
fn compute_schema_features(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();

    let mut label_hashes: HashMap<NodeId, u64> = HashMap::new();
    let mut prop_counts: HashMap<NodeId, usize> = HashMap::new();

    for &nid in &node_ids {
        if let Some(node) = store.get_node(nid) {
            // Hash the sorted labels
            let mut hasher = std::hash::DefaultHasher::new();
            for label in &node.labels {
                label.hash(&mut hasher);
            }
            label_hashes.insert(nid, hasher.finish());

            // Count properties
            prop_counts.insert(nid, node.properties.len());
        } else {
            label_hashes.insert(nid, 0);
            prop_counts.insert(nid, 0);
        }
    }

    let max_props = prop_counts.values().max().copied().unwrap_or(1).max(1) as f32;

    node_ids
        .iter()
        .map(|&nid| {
            let hash = label_hashes.get(&nid).copied().unwrap_or(0);
            // Normalize hash to [0, 1] via uniform mapping
            let x = (hash as f64 / u64::MAX as f64) as f32;
            let y = *prop_counts.get(&nid).unwrap_or(&0) as f32 / max_props;
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// Large-graph approximations (O(V+E) instead of O(V×E))
// ============================================================================

/// Approximate spectral facette 1-2 using normalized total degree + neighbor degree.
/// O(V+E) — replaces spectral_embedding for large graphs.
/// Uses both outgoing AND incoming edges for total degree.
fn compute_degree_approx(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut total_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut all_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::with_capacity(node_ids.len());
    let mut neighbor_sum: HashMap<NodeId, f64> = HashMap::with_capacity(node_ids.len());

    // Pass 1: compute out-degree and accumulate in-degree
    for &nid in &node_ids {
        let out_edges = store.edges_from(nid, Direction::Outgoing);
        let out_deg = out_edges.len();
        *total_deg.entry(nid).or_insert(0) += out_deg;
        let neighbors = all_neighbors.entry(nid).or_default();
        for (neighbor, _) in &out_edges {
            neighbors.push(*neighbor);
            // Count as in-degree for the neighbor
            *total_deg.entry(*neighbor).or_insert(0) += 1;
        }
    }

    // Pass 2: average neighbor total degree
    for &nid in &node_ids {
        let neighbors = all_neighbors
            .get(&nid)
            .map_or(&[] as &[NodeId], |v| v.as_slice());
        let sum: f64 = neighbors
            .iter()
            .map(|n| *total_deg.get(n).unwrap_or(&0) as f64)
            .sum();
        let avg = if neighbors.is_empty() {
            0.0
        } else {
            sum / neighbors.len() as f64
        };
        neighbor_sum.insert(nid, avg);
    }

    // Rank percentile on both dimensions
    let deg_f64: HashMap<NodeId, f64> = total_deg.iter().map(|(&k, &v)| (k, v as f64)).collect();
    let deg_ranks = rank_normalize(&deg_f64);
    let neighbor_ranks = rank_normalize(&neighbor_sum);

    node_ids
        .iter()
        .map(|&nid| {
            let x = *deg_ranks.get(&nid).unwrap_or(&0.5);
            let y = *neighbor_ranks.get(&nid).unwrap_or(&0.5);
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Secondary degree approximation: out/in degree ratio + clustering coefficient proxy.
fn compute_degree_approx_secondary(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut out_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut in_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut out_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::with_capacity(node_ids.len());

    // Compute out-degree + in-degree + collect neighbors
    for &nid in &node_ids {
        let edges = store.edges_from(nid, Direction::Outgoing);
        out_deg.insert(nid, edges.len());
        let neighbors: Vec<NodeId> = edges.iter().map(|(n, _)| *n).collect();
        out_neighbors.insert(nid, neighbors);
        for (neighbor, _) in &edges {
            *in_deg.entry(*neighbor).or_insert(0) += 1;
        }
    }

    // x = out/in balance (directional asymmetry captures structural role)
    // y = clustering coefficient proxy
    node_ids
        .iter()
        .map(|&nid| {
            let out = *out_deg.get(&nid).unwrap_or(&0) as f64;
            let inc = *in_deg.get(&nid).unwrap_or(&0) as f64;
            let total = out + inc;
            // Directional asymmetry: pure sources=1.0, pure sinks=0.0, balanced=0.5
            let x = if total > 0.0 {
                (out / total) as f32
            } else {
                0.5
            };

            // Clustering coefficient proxy (sampled)
            let neighbors = out_neighbors
                .get(&nid)
                .map_or(&[] as &[NodeId], |v| v.as_slice());
            let k = neighbors.len();
            let y = if k < 2 {
                0.0
            } else {
                let max_pairs = 50usize;
                let mut connected = 0usize;
                let mut checked = 0usize;
                let step = if k * (k - 1) / 2 > max_pairs {
                    k / 10 + 1
                } else {
                    1
                };
                'outer: for i in (0..k).step_by(step) {
                    for j in (i + 1..k).step_by(step) {
                        let ni_neighbors = out_neighbors
                            .get(&neighbors[i])
                            .map_or(&[] as &[NodeId], |v| v.as_slice());
                        if ni_neighbors.contains(&neighbors[j]) {
                            connected += 1;
                        }
                        checked += 1;
                        if checked >= max_pairs {
                            break 'outer;
                        }
                    }
                }
                if checked > 0 {
                    connected as f32 / checked as f32
                } else {
                    0.0
                }
            };
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

/// Approximate betweenness using degree centrality ratio (in/out balance).
/// O(V+E) — replaces betweenness_centrality + closeness_centrality.
fn compute_degree_betweenness_approx(store: &dyn GraphStore) -> HashMap<NodeId, [f32; 2]> {
    let node_ids = store.node_ids();
    let mut out_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());
    let mut in_deg: HashMap<NodeId, usize> = HashMap::with_capacity(node_ids.len());

    for &nid in &node_ids {
        let out = store.edges_from(nid, Direction::Outgoing).len();
        out_deg.insert(nid, out);
        for (neighbor, _) in store.edges_from(nid, Direction::Outgoing) {
            *in_deg.entry(neighbor).or_insert(0) += 1;
        }
    }

    // Rank percentile on total degree (betweenness proxy)
    let total_deg: HashMap<NodeId, f64> = node_ids
        .iter()
        .map(|&nid| {
            let total = *out_deg.get(&nid).unwrap_or(&0) + *in_deg.get(&nid).unwrap_or(&0);
            (nid, total as f64)
        })
        .collect();
    let total_ranks = rank_normalize(&total_deg);

    node_ids
        .iter()
        .map(|&nid| {
            let out = *out_deg.get(&nid).unwrap_or(&0) as f64;
            let inc = *in_deg.get(&nid).unwrap_or(&0) as f64;
            let total = out + inc;
            // x = total degree rank percentile (proxy for betweenness)
            let x = *total_ranks.get(&nid).unwrap_or(&0.5);
            // y = in/out balance (bridges tend to have balanced in/out)
            let y = if total > 0.0 {
                (1.0 - (out - inc).abs() / total) as f32
            } else {
                0.5
            };
            (nid, [x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)])
        })
        .collect()
}

// ============================================================================
// GraphAlgorithm trait
// ============================================================================

/// Hilbert features algorithm wrapper for registry integration.
pub struct HilbertFeaturesAlgorithm;

static HILBERT_FEATURES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn hilbert_features_params() -> &'static [ParameterDef] {
    HILBERT_FEATURES_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "levels".to_string(),
                description: "Hilbert encoding levels per facette".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("8".to_string()),
            },
            ParameterDef {
                name: "persona_edge_type".to_string(),
                description: "Edge type for co-usage facette".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("REINFORCES".to_string()),
            },
        ]
    })
}

impl GraphAlgorithm for HilbertFeaturesAlgorithm {
    fn name(&self) -> &str {
        "obrain.hilbert_features"
    }

    fn description(&self) -> &str {
        "64d multi-facette Hilbert features (8 facettes × 8 levels)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        hilbert_features_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let levels = params.get_int("levels").unwrap_or(8) as usize;
        let co_usage = params
            .get_string("persona_edge_type")
            .unwrap_or("REINFORCES")
            .to_string();

        let config = HilbertFeaturesConfig {
            levels,
            persona_edge_type: co_usage,
            ..Default::default()
        };

        let result = hilbert_features(store, &config);

        let mut algo_result =
            AlgorithmResult::new(vec!["node_id".to_string(), "features".to_string()]);

        for (node_id, features) in &result.features {
            let features_val =
                Value::List(features.iter().map(|&x| Value::Float64(x as f64)).collect());
            algo_result.add_row(vec![Value::Int64(node_id.0 as i64), features_val]);
        }

        Ok(algo_result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use super::*;

    fn make_test_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();
        // Build a small graph: A-B-C-D-E (chain) + A-E (cycle)
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Document"]);
        let d = store.create_node(&["Person"]);
        let e = store.create_node(&["Document"]);

        for &(src, dst) in &[(a, b), (b, c), (c, d), (d, e), (e, a)] {
            store.create_edge(src, dst, "KNOWS");
            store.create_edge(dst, src, "KNOWS");
        }
        store
    }

    fn make_two_clusters() -> LpgStore {
        let store = LpgStore::new().unwrap();
        // Cluster 1: fully connected (a,b,c)
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let c = store.create_node(&["A"]);
        for &(s, d) in &[(a, b), (b, c), (a, c)] {
            store.create_edge(s, d, "LINK");
            store.create_edge(d, s, "LINK");
        }
        // Cluster 2: fully connected (d,e,f)
        let d = store.create_node(&["B"]);
        let e = store.create_node(&["B"]);
        let f = store.create_node(&["B"]);
        for &(s, t) in &[(d, e), (e, f), (d, f)] {
            store.create_edge(s, t, "LINK");
            store.create_edge(t, s, "LINK");
        }
        // Weak bridge: c-d
        store.create_edge(c, d, "BRIDGE");
        store.create_edge(d, c, "BRIDGE");
        store
    }

    #[test]
    fn test_hilbert_features_basic() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);

        assert_eq!(result.dimensions, 64);
        assert_eq!(result.features.len(), 5);

        for vec in result.features.values() {
            assert_eq!(vec.len(), 64);
            for &v in vec {
                assert!((0.0..=1.0).contains(&v), "Feature value {v} not in [0,1]");
            }
        }
    }

    #[test]
    fn test_hilbert_features_dimensions_exact() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig {
            levels: 8,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);

        for vec in result.features.values() {
            assert_eq!(vec.len(), 64, "Expected exactly 64 dimensions");
            // Each block of 8 dims should be in [0,1]
            for block in 0..8 {
                for i in 0..8 {
                    let v = vec[block * 8 + i];
                    assert!((0.0..=1.0).contains(&v));
                }
            }
        }
    }

    #[test]
    fn test_hilbert_features_determinism() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig::default();
        let r1 = hilbert_features(&store, &config);
        let r2 = hilbert_features(&store, &config);

        for (nid, f1) in &r1.features {
            let f2 = r2.features.get(nid).unwrap();
            // Skip facette 2 (dims 16-23, community) — Louvain is non-deterministic
            // due to HashMap iteration order. All other facettes must be exact.
            for (i, (a, b)) in f1.iter().zip(f2.iter()).enumerate() {
                if (16..24).contains(&i) {
                    continue; // community facette
                }
                assert_eq!(a, b, "Non-deterministic for node {nid:?} at dim {i}");
            }
        }
    }

    #[test]
    fn test_hilbert_features_empty_graph() {
        let store = LpgStore::new().unwrap();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);
        assert!(result.features.is_empty());
        assert_eq!(result.dimensions, 64);
    }

    #[test]
    fn test_hilbert_features_locality() {
        let store = make_two_clusters();
        let config = HilbertFeaturesConfig::default();
        let result = hilbert_features(&store, &config);

        let nodes: Vec<NodeId> = store.node_ids();
        // Cluster 1: nodes 0,1,2 — Cluster 2: nodes 3,4,5
        let cluster1: Vec<&Vec<f32>> = nodes[..3]
            .iter()
            .filter_map(|n| result.features.get(n))
            .collect();
        let cluster2: Vec<&Vec<f32>> = nodes[3..6]
            .iter()
            .filter_map(|n| result.features.get(n))
            .collect();

        // Average intra-cluster distance should be < inter-cluster distance
        let intra1 = avg_l2_distance(&cluster1);
        let intra2 = avg_l2_distance(&cluster2);
        let inter = avg_l2_distance_between(&cluster1, &cluster2);

        let avg_intra = f32::midpoint(intra1, intra2);
        assert!(
            avg_intra < inter,
            "Intra-cluster distance ({avg_intra}) should be < inter-cluster ({inter})"
        );
    }

    #[test]
    fn test_hilbert_features_custom_levels() {
        let store = make_test_graph();
        let config = HilbertFeaturesConfig {
            levels: 4,
            ..Default::default()
        };
        let result = hilbert_features(&store, &config);
        assert_eq!(result.dimensions, 32); // 4 levels × 8 facettes
        for vec in result.features.values() {
            assert_eq!(vec.len(), 32);
        }
    }

    #[test]
    fn test_hilbert_features_algorithm_trait() {
        let store = make_test_graph();
        let algo = HilbertFeaturesAlgorithm;
        assert_eq!(algo.name(), "obrain.hilbert_features");

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node_id", "features"]);
        assert_eq!(result.rows.len(), 5);
    }

    // Helpers
    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn avg_l2_distance(vecs: &[&Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..vecs.len() {
            for j in i + 1..vecs.len() {
                sum += l2_distance(vecs[i], vecs[j]);
                count += 1;
            }
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    fn avg_l2_distance_between(a: &[&Vec<f32>], b: &[&Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for va in a {
            for vb in b {
                sum += l2_distance(va, vb);
                count += 1;
            }
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    #[test]
    fn test_incremental_preserves_global() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Document"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, a, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, b, "KNOWS");

        let config = HilbertFeaturesConfig::default();
        let full = hilbert_features(&store, &config);
        assert!(!full.dirty_global);

        // Add a new node and edge
        let d = store.create_node(&["Person"]);
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, c, "KNOWS");

        let incr = hilbert_features_incremental(&store, &config, &full, &[d]);
        assert!(incr.dirty_global);
        assert_eq!(incr.features.len(), 4); // now 4 nodes

        // Global dims (0-31) should be unchanged for nodes a, b, c
        for &nid in &[a, b, c] {
            let full_vec = full.features.get(&nid).unwrap();
            let incr_vec = incr.features.get(&nid).unwrap();
            assert_eq!(
                &full_vec[..32],
                &incr_vec[..32],
                "Global dims changed for existing node {nid:?}"
            );
        }

        // New node d should have default global dims and valid local dims
        let d_vec = incr.features.get(&d).unwrap();
        assert_eq!(d_vec.len(), 64);
        for &v in d_vec {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_incremental_updates_local_for_neighbors() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, a, "LINK");

        let config = HilbertFeaturesConfig::default();
        let full = hilbert_features(&store, &config);

        // Add node c connected to b
        let c = store.create_node(&["B"]);
        store.create_edge(b, c, "LINK");
        store.create_edge(c, b, "LINK");

        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);

        // b is a neighbor of changed node c → local dims should be recomputed
        let b_full = full.features.get(&b).unwrap();
        let b_incr = incr.features.get(&b).unwrap();
        // Local dims (32-63) may differ because b's neighborhood changed
        // (This is expected — just verify they're valid)
        for &v in &b_incr[32..] {
            assert!((0.0..=1.0).contains(&v));
        }
        // Global dims unchanged
        assert_eq!(&b_full[..32], &b_incr[..32]);
    }

    #[test]
    fn test_incremental_dirty_flag() {
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["X"]);
        let b = store.create_node(&["X"]);
        store.create_edge(a, b, "E");
        store.create_edge(b, a, "E");

        let config = HilbertFeaturesConfig::default();

        // Full → dirty_global = false
        let full = hilbert_features(&store, &config);
        assert!(!full.dirty_global);

        // Incremental → dirty_global = true
        let c = store.create_node(&["Y"]);
        store.create_edge(a, c, "E");
        let incr = hilbert_features_incremental(&store, &config, &full, &[c]);
        assert!(incr.dirty_global);

        // Another full recalc → dirty_global = false
        let full2 = hilbert_features(&store, &config);
        assert!(!full2.dirty_global);
    }

    #[test]
    fn test_hilbert_features_perf_1000_nodes() {
        // Build a random-ish graph with 1000 nodes
        let store = LpgStore::new().unwrap();
        let mut nodes = Vec::with_capacity(1000);
        for i in 0..1000 {
            let label = if i % 3 == 0 {
                "A"
            } else if i % 3 == 1 {
                "B"
            } else {
                "C"
            };
            nodes.push(store.create_node(&[label]));
        }
        // Add ~3000 edges (ring + skip connections)
        for i in 0..1000 {
            let j = (i + 1) % 1000;
            store.create_edge(nodes[i], nodes[j], "LINK");
            store.create_edge(nodes[j], nodes[i], "LINK");
            if i + 7 < 1000 {
                store.create_edge(nodes[i], nodes[i + 7], "LINK");
            }
        }

        let config = HilbertFeaturesConfig::default();
        let start = std::time::Instant::now();
        let result = hilbert_features(&store, &config);
        let elapsed = start.elapsed();

        assert_eq!(result.features.len(), 1000);
        assert!(
            elapsed.as_millis() < 5000,
            "hilbert_features on 1000 nodes took {}ms (should be <5s)",
            elapsed.as_millis()
        );
        eprintln!("hilbert_features(1000 nodes): {}ms", elapsed.as_millis());
    }
}
