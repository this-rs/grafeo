//! Clustering coefficient algorithms and Hilbert bank allocation.
//!
//! These algorithms measure how tightly connected the neighbors of each node are.
//! A high clustering coefficient indicates that neighbors tend to be connected to each other.
//!
//! Also provides [`hilbert_bank_allocation()`] — k-means clustering on Hilbert 64d/72d
//! feature vectors for semantically homogeneous bank allocation in attention masking.

#[cfg(feature = "parallel")]
use std::sync::Arc;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
use obrain_common::utils::hash::{FxHashMap, FxHashSet};
use obrain_core::graph::Direction;
use obrain_core::graph::GraphStore;
#[cfg(test)]
use obrain_core::graph::lpg::LpgStore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;
#[cfg(feature = "parallel")]
use super::traits::ParallelGraphAlgorithm;

// ============================================================================
// Result Types
// ============================================================================

/// Result of clustering coefficient computation.
#[derive(Debug, Clone)]
pub struct ClusteringCoefficientResult {
    /// Local clustering coefficient for each node (0.0 to 1.0).
    pub coefficients: FxHashMap<NodeId, f64>,
    /// Number of triangles containing each node.
    pub triangle_counts: FxHashMap<NodeId, u64>,
    /// Total number of unique triangles in the graph.
    pub total_triangles: u64,
    /// Global (average) clustering coefficient.
    pub global_coefficient: f64,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Builds undirected neighbor sets for all nodes.
///
/// Treats the graph as undirected by combining both outgoing and incoming edges.
fn build_undirected_neighbors(store: &dyn GraphStore) -> FxHashMap<NodeId, FxHashSet<NodeId>> {
    let nodes = store.node_ids();
    let mut neighbors: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();

    // Initialize all nodes with empty sets
    for &node in &nodes {
        neighbors.insert(node, FxHashSet::default());
    }

    // Add edges in both directions (undirected treatment)
    for &node in &nodes {
        // Outgoing edges: node -> neighbor
        for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            // Add reverse direction for undirected
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }

        // Incoming edges: neighbor -> node (ensures we capture all connections)
        for (neighbor, _) in store.edges_from(node, Direction::Incoming) {
            if let Some(set) = neighbors.get_mut(&node) {
                set.insert(neighbor);
            }
            if let Some(set) = neighbors.get_mut(&neighbor) {
                set.insert(node);
            }
        }
    }

    neighbors
}

/// Counts triangles for a single node given its neighbors and the full neighbor map.
fn count_node_triangles(
    node_neighbors: &FxHashSet<NodeId>,
    all_neighbors: &FxHashMap<NodeId, FxHashSet<NodeId>>,
) -> u64 {
    let neighbor_list: Vec<NodeId> = node_neighbors.iter().copied().collect();
    let k = neighbor_list.len();
    let mut triangles = 0u64;

    // For each pair of neighbors, check if they're connected
    for i in 0..k {
        for j in (i + 1)..k {
            let u = neighbor_list[i];
            let w = neighbor_list[j];

            // Check if u and w are neighbors (completing a triangle)
            if let Some(u_neighbors) = all_neighbors.get(&u)
                && u_neighbors.contains(&w)
            {
                triangles += 1;
            }
        }
    }

    triangles
}

// ============================================================================
// Core Algorithm Functions
// ============================================================================

/// Counts the number of triangles containing each node.
///
/// A triangle is a set of three nodes where each pair is connected.
/// Each triangle is counted once for each of its three vertices.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to the number of triangles containing that node.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn triangle_count(store: &dyn GraphStore) -> FxHashMap<NodeId, u64> {
    let neighbors = build_undirected_neighbors(store);
    let mut counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let triangles = count_node_triangles(node_neighbors, &neighbors);
        counts.insert(node, triangles);
    }

    counts
}

/// Computes the local clustering coefficient for each node.
///
/// The local clustering coefficient measures how close a node's neighbors are
/// to being a complete graph (clique). For a node v with degree k and T triangles:
///
/// C(v) = 2T / (k * (k-1)) for undirected graphs
///
/// Nodes with degree < 2 have coefficient 0.0 (cannot form triangles).
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Map from NodeId to local clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn local_clustering_coefficient(store: &dyn GraphStore) -> FxHashMap<NodeId, f64> {
    let neighbors = build_undirected_neighbors(store);
    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();

        if k < 2 {
            // Cannot form triangles with fewer than 2 neighbors
            coefficients.insert(node, 0.0);
        } else {
            let triangles = count_node_triangles(node_neighbors, &neighbors);
            let max_triangles = (k * (k - 1)) / 2;
            let coefficient = triangles as f64 / max_triangles as f64;
            coefficients.insert(node, coefficient);
        }
    }

    coefficients
}

/// Computes the global (average) clustering coefficient.
///
/// The global clustering coefficient is the average of all local coefficients
/// across all nodes in the graph.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Average clustering coefficient (0.0 to 1.0).
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn global_clustering_coefficient(store: &dyn GraphStore) -> f64 {
    let local = local_clustering_coefficient(store);

    if local.is_empty() {
        return 0.0;
    }

    let sum: f64 = local.values().sum();
    sum / local.len() as f64
}

/// Counts the total number of unique triangles in the graph.
///
/// Each triangle is counted exactly once (not three times).
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Total number of unique triangles.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn total_triangles(store: &dyn GraphStore) -> u64 {
    let per_node = triangle_count(store);
    // Each triangle is counted 3 times (once per vertex), so divide by 3
    per_node.values().sum::<u64>() / 3
}

/// Computes all clustering metrics in a single pass.
///
/// More efficient than calling each function separately since it builds
/// the neighbor structure only once.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
///
/// # Returns
///
/// Complete clustering coefficient result including local coefficients,
/// triangle counts, total triangles, and global coefficient.
///
/// # Complexity
///
/// O(V * d^2) where d is the average degree
pub fn clustering_coefficient(store: &dyn GraphStore) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    let mut coefficients: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut triangle_counts: FxHashMap<NodeId, u64> = FxHashMap::default();

    for (&node, node_neighbors) in &neighbors {
        let k = node_neighbors.len();
        let triangles = count_node_triangles(node_neighbors, &neighbors);

        triangle_counts.insert(node, triangles);

        let coefficient = if k < 2 {
            0.0
        } else {
            let max_triangles = (k * (k - 1)) / 2;
            triangles as f64 / max_triangles as f64
        };
        coefficients.insert(node, coefficient);
    }

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

// ============================================================================
// Parallel Implementation
// ============================================================================

/// Computes clustering coefficients in parallel using rayon.
///
/// Automatically falls back to sequential execution for small graphs
/// to avoid parallelization overhead.
///
/// # Arguments
///
/// * `store` - The graph store (treated as undirected)
/// * `parallel_threshold` - Minimum node count to enable parallelism
///
/// # Returns
///
/// Complete clustering coefficient result.
///
/// # Complexity
///
/// O(V * d^2 / threads) where d is the average degree
#[cfg(feature = "parallel")]
pub fn clustering_coefficient_parallel(
    store: &dyn GraphStore,
    parallel_threshold: usize,
) -> ClusteringCoefficientResult {
    let neighbors = build_undirected_neighbors(store);
    let n = neighbors.len();

    if n < parallel_threshold {
        // Fall back to sequential for small graphs
        return clustering_coefficient(store);
    }

    // Use Arc for shared neighbor data across threads
    let neighbors = Arc::new(neighbors);
    let nodes: Vec<NodeId> = neighbors.keys().copied().collect();

    // Parallel computation using fold-reduce pattern
    let (coefficients, triangle_counts): (FxHashMap<NodeId, f64>, FxHashMap<NodeId, u64>) = nodes
        .par_iter()
        .fold(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut coeffs, mut triangles), &node| {
                let node_neighbors = neighbors.get(&node).expect("node in neighbor map");
                let k = node_neighbors.len();

                let t = count_node_triangles(node_neighbors, &neighbors);

                triangles.insert(node, t);

                let coefficient = if k < 2 {
                    0.0
                } else {
                    let max_triangles = (k * (k - 1)) / 2;
                    t as f64 / max_triangles as f64
                };
                coeffs.insert(node, coefficient);

                (coeffs, triangles)
            },
        )
        .reduce(
            || (FxHashMap::default(), FxHashMap::default()),
            |(mut c1, mut t1), (c2, t2)| {
                c1.extend(c2);
                t1.extend(t2);
                (c1, t1)
            },
        );

    let total_triangles = triangle_counts.values().sum::<u64>() / 3;
    let global_coefficient = if n == 0 {
        0.0
    } else {
        coefficients.values().sum::<f64>() / n as f64
    };

    ClusteringCoefficientResult {
        coefficients,
        triangle_counts,
        total_triangles,
        global_coefficient,
    }
}

// ============================================================================
// Algorithm Wrapper for Plugin Registry
// ============================================================================

/// Static parameter definitions for Clustering Coefficient algorithm.
static CLUSTERING_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn clustering_params() -> &'static [ParameterDef] {
    CLUSTERING_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "parallel".to_string(),
                description: "Enable parallel computation (default: true)".to_string(),
                param_type: ParameterType::Boolean,
                required: false,
                default: Some("true".to_string()),
            },
            ParameterDef {
                name: "parallel_threshold".to_string(),
                description: "Minimum nodes for parallel execution (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
        ]
    })
}

/// Clustering Coefficient algorithm wrapper for the plugin registry.
pub struct ClusteringCoefficientAlgorithm;

impl GraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn name(&self) -> &str {
        "clustering_coefficient"
    }

    fn description(&self) -> &str {
        "Local and global clustering coefficients with triangle counts"
    }

    fn parameters(&self) -> &[ParameterDef] {
        clustering_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        #[cfg(feature = "parallel")]
        let result = {
            let parallel = params.get_bool("parallel").unwrap_or(true);
            let threshold = params.get_int("parallel_threshold").unwrap_or(50) as usize;

            if parallel {
                clustering_coefficient_parallel(store, threshold)
            } else {
                clustering_coefficient(store)
            }
        };

        #[cfg(not(feature = "parallel"))]
        let result = {
            let _ = params; // suppress unused warning
            clustering_coefficient(store)
        };

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
            ]);
        }

        Ok(output)
    }
}

#[cfg(feature = "parallel")]
impl ParallelGraphAlgorithm for ClusteringCoefficientAlgorithm {
    fn parallel_threshold(&self) -> usize {
        50
    }

    fn execute_parallel(
        &self,
        store: &dyn GraphStore,
        _params: &Parameters,
        _num_threads: usize,
    ) -> Result<AlgorithmResult> {
        let result = clustering_coefficient_parallel(store, self.parallel_threshold());

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "clustering_coefficient".to_string(),
            "triangle_count".to_string(),
        ]);

        for (node, coefficient) in result.coefficients {
            let triangles = *result.triangle_counts.get(&node).unwrap_or(&0);
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Float64(coefficient),
                Value::Int64(triangles as i64),
            ]);
        }

        Ok(output)
    }
}

// ============================================================================
// Hilbert bank allocation (k-means on Hilbert features)
// ============================================================================

use super::hilbert_features::HilbertFeaturesResult;

/// Allocate graph nodes into banks using k-means on Hilbert feature vectors.
///
/// Produces `n_banks` clusters of nodes that are semantically homogeneous
/// (similar across all 8/9 Hilbert facettes). Intended as a replacement for
/// BFS-based bank allocation in attention masking.
///
/// Uses **k-means++** initialization for stable, reproducible cluster seeds,
/// followed by Lloyd's algorithm with early termination when centroid delta
/// falls below `1e-6`.
///
/// # Arguments
///
/// * `features` - Hilbert features result (64d or 72d per node)
/// * `n_banks` - Number of banks (clusters) to produce
/// * `max_iter` - Maximum Lloyd's iterations (typically 50)
///
/// # Returns
///
/// A `Vec<Vec<NodeId>>` of length `n_banks`, sorted by cluster size
/// (largest first). All input nodes are assigned to exactly one cluster.
/// Empty clusters are removed (so the result may have fewer than `n_banks` entries).
///
/// # Example
///
/// ```no_run
/// use obrain_adapters::plugins::algorithms::clustering::hilbert_bank_allocation;
/// use obrain_adapters::plugins::algorithms::hilbert_features::{hilbert_features, HilbertFeaturesConfig};
/// use obrain_core::graph::lpg::LpgStore;
///
/// let store = LpgStore::new().unwrap();
/// // ... populate graph ...
/// let features = hilbert_features(&store, &HilbertFeaturesConfig::default());
/// let banks = hilbert_bank_allocation(&features, 8, 50);
/// ```
pub fn hilbert_bank_allocation(
    features: &HilbertFeaturesResult,
    n_banks: usize,
    max_iter: usize,
) -> Vec<Vec<NodeId>> {
    if features.features.is_empty() || n_banks == 0 {
        return Vec::new();
    }

    let dims = features.dimensions;
    let nodes: Vec<NodeId> = features.features.keys().copied().collect();
    let vectors: Vec<&Vec<f32>> = nodes.iter().map(|n| &features.features[n]).collect();
    let n = nodes.len();

    if n_banks >= n {
        // More banks than nodes → one node per bank
        return nodes.into_iter().map(|n| vec![n]).collect();
    }

    // k-means++ initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_banks);

    // First centroid: pick the first node (deterministic)
    centroids.push(vectors[0].to_vec());

    for _ in 1..n_banks {
        // For each point, compute min distance to existing centroids
        let mut dists: Vec<f32> = Vec::with_capacity(n);

        for v in &vectors {
            let min_d = centroids
                .iter()
                .map(|c| sq_dist(v, c))
                .fold(f32::INFINITY, f32::min);
            dists.push(min_d);
        }

        // Pick the point with maximum min-distance (deterministic variant of k-means++)
        // This avoids RNG dependency while still spreading centroids well.
        let best_idx = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.push(vectors[best_idx].to_vec());
    }

    // Lloyd's iterations
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        // Assignment step: assign each point to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let best_k = centroids
                .iter()
                .enumerate()
                .map(|(k, c)| (k, sq_dist(vectors[i], c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k)
                .unwrap_or(0);

            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }

        if !changed {
            break; // Converged
        }

        // Update step: recompute centroids
        let mut new_centroids = vec![vec![0.0_f32; dims]; n_banks];
        let mut counts = vec![0usize; n_banks];

        for i in 0..n {
            let k = assignments[i];
            counts[k] += 1;
            for d in 0..dims {
                new_centroids[k][d] += vectors[i][d];
            }
        }

        // Check convergence (max centroid delta)
        let mut max_delta: f32 = 0.0;
        for k in 0..n_banks {
            if counts[k] > 0 {
                let c = counts[k] as f32;
                for d in 0..dims {
                    new_centroids[k][d] /= c;
                }
                let delta = sq_dist(&centroids[k], &new_centroids[k]).sqrt();
                if delta > max_delta {
                    max_delta = delta;
                }
            }
            // Keep old centroid for empty clusters
        }

        centroids = new_centroids;

        if max_delta < 1e-6 {
            break;
        }
    }

    // Build result: group nodes by cluster
    let mut banks: Vec<Vec<NodeId>> = vec![Vec::new(); n_banks];
    for i in 0..n {
        banks[assignments[i]].push(nodes[i]);
    }

    // Remove empty clusters and sort by size (largest first)
    banks.retain(|b| !b.is_empty());
    banks.sort_by(|a, b| b.len().cmp(&a.len()));

    banks
}

/// Squared Euclidean distance (avoids sqrt for comparison).
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triangle_graph() -> LpgStore {
        // Simple triangle: 0 - 1 - 2 - 0
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        // Bidirectional edges for undirected treatment
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        store
    }

    fn create_star_graph() -> LpgStore {
        // Star: center (0) connected to leaves (1, 2, 3, 4)
        // No triangles because leaves don't connect to each other
        let store = LpgStore::new().unwrap();
        let center = store.create_node(&["Center"]);

        for _ in 0..4 {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center, leaf, "EDGE");
            store.create_edge(leaf, center, "EDGE");
        }

        store
    }

    fn create_complete_graph(n: usize) -> LpgStore {
        // K_n: complete graph with n nodes (all pairs connected)
        let store = LpgStore::new().unwrap();
        let nodes: Vec<NodeId> = (0..n).map(|_| store.create_node(&["Node"])).collect();

        for i in 0..n {
            for j in (i + 1)..n {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        store
    }

    fn create_path_graph() -> LpgStore {
        // Path: 0 - 1 - 2 - 3
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        store
    }

    #[test]
    fn test_triangle_graph_clustering() {
        let store = create_triangle_graph();
        let result = clustering_coefficient(&store);

        // All nodes in a triangle have coefficient 1.0
        for (_, coeff) in &result.coefficients {
            assert!(
                (*coeff - 1.0).abs() < 1e-10,
                "Expected coefficient 1.0, got {}",
                coeff
            );
        }

        // One unique triangle
        assert_eq!(result.total_triangles, 1);

        // Global coefficient should be 1.0
        assert!(
            (result.global_coefficient - 1.0).abs() < 1e-10,
            "Expected global 1.0, got {}",
            result.global_coefficient
        );
    }

    #[test]
    fn test_star_graph_clustering() {
        let store = create_star_graph();
        let result = clustering_coefficient(&store);

        // All coefficients should be 0 (no triangles in a star)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }

        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_complete_graph_clustering() {
        let store = create_complete_graph(5);
        let result = clustering_coefficient(&store);

        // In a complete graph, all coefficients are 1.0
        for (_, coeff) in &result.coefficients {
            assert!((*coeff - 1.0).abs() < 1e-10, "Expected 1.0, got {}", coeff);
        }

        // K_5 has C(5,3) = 10 triangles
        assert_eq!(result.total_triangles, 10);
    }

    #[test]
    fn test_path_graph_clustering() {
        let store = create_path_graph();
        let result = clustering_coefficient(&store);

        // Path has no triangles
        assert_eq!(result.total_triangles, 0);

        // All coefficients should be 0 (endpoints have degree 1, middle have no triangles)
        for (_, coeff) in &result.coefficients {
            assert_eq!(*coeff, 0.0);
        }
    }

    #[test]
    fn test_empty_graph() {
        let store = LpgStore::new().unwrap();
        let result = clustering_coefficient(&store);

        assert!(result.coefficients.is_empty());
        assert!(result.triangle_counts.is_empty());
        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.global_coefficient, 0.0);
    }

    #[test]
    fn test_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);

        let result = clustering_coefficient(&store);

        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_two_connected_nodes() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");

        let result = clustering_coefficient(&store);

        // Both nodes have degree 1, so coefficient is 0
        assert_eq!(*result.coefficients.get(&n0).unwrap(), 0.0);
        assert_eq!(*result.coefficients.get(&n1).unwrap(), 0.0);
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_triangle_count_function() {
        let store = create_triangle_graph();
        let counts = triangle_count(&store);

        // Each node in a triangle has 1 triangle
        for (_, count) in counts {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_local_clustering_coefficient_function() {
        let store = create_complete_graph(4);
        let coefficients = local_clustering_coefficient(&store);

        // K_4: all nodes have coefficient 1.0
        for (_, coeff) in coefficients {
            assert!((coeff - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_global_clustering_coefficient_function() {
        let store = create_triangle_graph();
        let global = global_clustering_coefficient(&store);
        assert!((global - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_triangles_function() {
        let store = create_complete_graph(4);
        let total = total_triangles(&store);
        // K_4 has C(4,3) = 4 triangles
        assert_eq!(total, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_matches_sequential() {
        let store = create_complete_graph(20);

        let sequential = clustering_coefficient(&store);
        let parallel = clustering_coefficient_parallel(&store, 1); // Force parallel

        // Results should match
        for (node, seq_coeff) in &sequential.coefficients {
            let par_coeff = parallel.coefficients.get(node).unwrap();
            assert!(
                (seq_coeff - par_coeff).abs() < 1e-10,
                "Mismatch for node {:?}: seq={}, par={}",
                node,
                seq_coeff,
                par_coeff
            );
        }

        assert_eq!(sequential.total_triangles, parallel.total_triangles);
        assert!((sequential.global_coefficient - parallel.global_coefficient).abs() < 1e-10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_threshold_fallback() {
        let store = create_triangle_graph();

        // With threshold higher than node count, should use sequential
        let result = clustering_coefficient_parallel(&store, 100);

        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.total_triangles, 1);
    }

    #[test]
    fn test_algorithm_wrapper() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.name(), "clustering_coefficient");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 2);

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();

        assert_eq!(result.columns.len(), 3);
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_algorithm_wrapper_sequential() {
        let store = create_triangle_graph();
        let algo = ClusteringCoefficientAlgorithm;

        let mut params = Parameters::new();
        params.set_bool("parallel", false);

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_algorithm_trait() {
        let store = create_complete_graph(10);
        let algo = ClusteringCoefficientAlgorithm;

        assert_eq!(algo.parallel_threshold(), 50);

        let params = Parameters::new();
        let result = algo.execute_parallel(&store, &params, 4).unwrap();

        assert_eq!(result.row_count(), 10);
    }

    #[test]
    fn test_two_triangles_sharing_edge() {
        // Two triangles sharing edge 0-1:
        //     2
        //    / \
        //   0---1
        //    \ /
        //     3
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        let n3 = store.create_node(&["Node"]);

        // Triangle 0-1-2
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n2, n0, "EDGE");
        store.create_edge(n0, n2, "EDGE");

        // Triangle 0-1-3
        store.create_edge(n1, n3, "EDGE");
        store.create_edge(n3, n1, "EDGE");
        store.create_edge(n3, n0, "EDGE");
        store.create_edge(n0, n3, "EDGE");

        let result = clustering_coefficient(&store);

        // 2 unique triangles
        assert_eq!(result.total_triangles, 2);

        // Nodes 0 and 1 are in both triangles
        assert_eq!(*result.triangle_counts.get(&n0).unwrap(), 2);
        assert_eq!(*result.triangle_counts.get(&n1).unwrap(), 2);

        // Nodes 2 and 3 are in one triangle each
        assert_eq!(*result.triangle_counts.get(&n2).unwrap(), 1);
        assert_eq!(*result.triangle_counts.get(&n3).unwrap(), 1);

        // Node 0 has 3 neighbors (1, 2, 3), 2 triangles
        // max_triangles = 3*2/2 = 3, coefficient = 2/3
        let coeff_0 = *result.coefficients.get(&n0).unwrap();
        assert!(
            (coeff_0 - 2.0 / 3.0).abs() < 1e-10,
            "Expected 2/3, got {}",
            coeff_0
        );
    }

    // ====================================================================
    // Hilbert bank allocation tests
    // ====================================================================

    use crate::plugins::algorithms::hilbert_features::{
        HilbertFeaturesConfig, HilbertFeaturesResult, hilbert_features,
    };

    fn create_two_cluster_graph() -> (LpgStore, Vec<NodeId>, Vec<NodeId>) {
        let store = LpgStore::new().unwrap();
        // Cluster A: tightly connected
        let a: Vec<NodeId> = (0..5).map(|_| store.create_node(&["A"])).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                store.create_edge(a[i], a[j], "LINK");
                store.create_edge(a[j], a[i], "LINK");
            }
        }

        // Cluster B: tightly connected
        let b: Vec<NodeId> = (0..5).map(|_| store.create_node(&["B"])).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                store.create_edge(b[i], b[j], "LINK");
                store.create_edge(b[j], b[i], "LINK");
            }
        }

        // Single bridge edge between clusters
        store.create_edge(a[0], b[0], "BRIDGE");

        (store, a, b)
    }

    #[test]
    fn test_kmeans_convergence() {
        let (store, _, _) = create_two_cluster_graph();
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(&store, &config);

        let banks = hilbert_bank_allocation(&features, 2, 50);

        // Should produce exactly 2 non-empty banks
        assert_eq!(banks.len(), 2);

        // All 10 nodes should be assigned
        let total: usize = banks.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_hilbert_banks_nonempty() {
        let (store, _, _) = create_two_cluster_graph();
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(&store, &config);

        let banks = hilbert_bank_allocation(&features, 4, 50);

        // All banks should be non-empty (empty ones are removed)
        for bank in &banks {
            assert!(!bank.is_empty());
        }

        // Total nodes = 10
        let total: usize = banks.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_hilbert_banks_sorted_by_size() {
        let (store, _, _) = create_two_cluster_graph();
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(&store, &config);

        let banks = hilbert_bank_allocation(&features, 3, 50);

        // Banks should be sorted by size (largest first)
        for i in 1..banks.len() {
            assert!(
                banks[i - 1].len() >= banks[i].len(),
                "Banks not sorted: bank[{}].len()={} < bank[{}].len()={}",
                i - 1,
                banks[i - 1].len(),
                i,
                banks[i].len()
            );
        }
    }

    #[test]
    fn test_hilbert_banks_empty_features() {
        let features = HilbertFeaturesResult {
            features: std::collections::HashMap::new(),
            dimensions: 64,
            dirty_global: false,
        };

        let banks = hilbert_bank_allocation(&features, 4, 50);
        assert!(banks.is_empty());
    }

    #[test]
    fn test_hilbert_banks_more_banks_than_nodes() {
        let (store, _, _) = create_two_cluster_graph();
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(&store, &config);

        // 20 banks for 10 nodes → one node per bank
        let banks = hilbert_bank_allocation(&features, 20, 50);
        assert_eq!(banks.len(), 10);
        for bank in &banks {
            assert_eq!(bank.len(), 1);
        }
    }

    #[test]
    fn test_hilbert_banks_intra_cluster_distance() {
        let (store, _, _) = create_two_cluster_graph();
        let config = HilbertFeaturesConfig::default();
        let features = hilbert_features(&store, &config);

        let banks = hilbert_bank_allocation(&features, 2, 50);

        // Compute average intra-cluster distance
        let mut total_intra: f64 = 0.0;
        let mut count = 0;
        for bank in &banks {
            for i in 0..bank.len() {
                for j in (i + 1)..bank.len() {
                    let a = &features.features[&bank[i]];
                    let b = &features.features[&bank[j]];
                    total_intra += sq_dist(a, b) as f64;
                    count += 1;
                }
            }
        }
        let avg_intra = if count > 0 {
            total_intra / count as f64
        } else {
            0.0
        };

        // Compute average inter-cluster distance
        let mut total_inter: f64 = 0.0;
        let mut inter_count = 0;
        if banks.len() >= 2 {
            for i in 0..banks[0].len() {
                for j in 0..banks[1].len() {
                    let a = &features.features[&banks[0][i]];
                    let b = &features.features[&banks[1][j]];
                    total_inter += sq_dist(a, b) as f64;
                    inter_count += 1;
                }
            }
        }
        let avg_inter = if inter_count > 0 {
            total_inter / inter_count as f64
        } else {
            0.0
        };

        // Intra-cluster should be <= inter-cluster (clusters are internally homogeneous)
        assert!(
            avg_intra <= avg_inter + 1e-6,
            "Intra-cluster distance ({avg_intra}) should be <= inter-cluster ({avg_inter})"
        );
    }
}
