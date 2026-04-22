//! Node similarity algorithms based on neighborhood overlap.
//!
//! Implements standard GDS similarity metrics: Jaccard, Overlap, Cosine,
//! Adamic-Adar, and Resource Allocation. All are purely set-based, computed
//! from neighbor sets N(u) and N(v).
//!
//! Two modes are supported:
//! - **Pairwise**: similarity between two specific nodes
//! - **Top-K**: for a given node, find the K most similar nodes
//!
//! Used by PO in `predict_missing_links` and `find_structural_twins`.

use std::collections::BinaryHeap;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::{Error, Result};
use obrain_common::utils::hash::FxHashSet;
use obrain_core::graph::Direction;
use obrain_core::graph::GraphStore;
#[cfg(test)]
use obrain_core::graph::GraphStoreMut;
#[cfg(test)]
use obrain_substrate::SubstrateStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Neighbor set helper
// ============================================================================

/// Collects the undirected neighbor set of a node (both directions, deduplicated).
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `node` - The node whose neighbors to collect
///
/// # Returns
///
/// A `FxHashSet<NodeId>` of all unique neighbors (union of outgoing and incoming).
///
/// # Complexity
///
/// O(deg(node))
///
/// # Example
///
/// ```ignore
/// let neighbors = neighbor_set(&store, node_id);
/// assert!(neighbors.contains(&other_node));
/// ```
fn neighbor_set(store: &dyn GraphStore, node: NodeId) -> FxHashSet<NodeId> {
    let mut set = FxHashSet::default();
    for n in store.neighbors(node, Direction::Both) {
        set.insert(n);
    }
    set
}

// ============================================================================
// Jaccard Similarity
// ============================================================================

/// Computes the Jaccard similarity between two nodes based on their neighborhoods.
///
/// Jaccard similarity is defined as `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`, where
/// `N(x)` is the set of neighbors of node `x`. Returns 0.0 if both nodes are
/// isolated (no neighbors).
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `u` - First node
/// * `v` - Second node
///
/// # Returns
///
/// A similarity score in `[0.0, 1.0]`. Returns 1.0 if both nodes have the exact
/// same neighborhood, 0.0 if they share no neighbors.
///
/// # Complexity
///
/// O(deg(u) + deg(v))
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::jaccard;
/// let score = jaccard(&store, node_a, node_b);
/// assert!(score >= 0.0 && score <= 1.0);
/// ```
pub fn jaccard(store: &dyn GraphStore, u: NodeId, v: NodeId) -> f64 {
    let nu = neighbor_set(store, u);
    let nv = neighbor_set(store, v);

    let intersection = nu.intersection(&nv).count();
    let union = nu.len() + nv.len() - intersection;

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ============================================================================
// Overlap Coefficient
// ============================================================================

/// Computes the Overlap coefficient between two nodes based on their neighborhoods.
///
/// The Overlap coefficient is defined as `|N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)`.
/// It measures the fraction of the smaller neighborhood that overlaps with the larger.
/// Returns 0.0 if either node is isolated.
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `u` - First node
/// * `v` - Second node
///
/// # Returns
///
/// A similarity score in `[0.0, 1.0]`. Returns 1.0 if one node's neighborhood
/// is a subset of the other's. Returns 0.0 if they share no neighbors.
///
/// # Complexity
///
/// O(deg(u) + deg(v))
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::overlap_coefficient;
/// let score = overlap_coefficient(&store, node_a, node_b);
/// assert!(score >= 0.0 && score <= 1.0);
/// ```
pub fn overlap_coefficient(store: &dyn GraphStore, u: NodeId, v: NodeId) -> f64 {
    let nu = neighbor_set(store, u);
    let nv = neighbor_set(store, v);

    let intersection = nu.intersection(&nv).count();
    let min_size = nu.len().min(nv.len());

    if min_size == 0 {
        0.0
    } else {
        intersection as f64 / min_size as f64
    }
}

// ============================================================================
// Cosine Similarity
// ============================================================================

/// Computes the Cosine similarity between two nodes based on their neighborhoods.
///
/// Cosine similarity is defined as `|N(u) ∩ N(v)| / (√|N(u)| × √|N(v)|)`.
/// It normalizes by the geometric mean of the neighborhood sizes.
/// Returns 0.0 if either node is isolated.
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `u` - First node
/// * `v` - Second node
///
/// # Returns
///
/// A similarity score in `[0.0, 1.0]`. Returns 0.0 if either node has no neighbors.
///
/// # Complexity
///
/// O(deg(u) + deg(v))
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::cosine_similarity;
/// let score = cosine_similarity(&store, node_a, node_b);
/// assert!(score >= 0.0 && score <= 1.0);
/// ```
pub fn cosine_similarity(store: &dyn GraphStore, u: NodeId, v: NodeId) -> f64 {
    let nu = neighbor_set(store, u);
    let nv = neighbor_set(store, v);

    let intersection = nu.intersection(&nv).count();
    let denom = (nu.len() as f64).sqrt() * (nv.len() as f64).sqrt();

    if denom == 0.0 {
        0.0
    } else {
        intersection as f64 / denom
    }
}

// ============================================================================
// Adamic-Adar Index
// ============================================================================

/// Computes the Adamic-Adar index between two nodes.
///
/// The Adamic-Adar index sums `1 / log(|N(w)|)` for each common neighbor `w`
/// of `u` and `v`. Neighbors with lower degree contribute more, capturing the
/// intuition that a shared rare neighbor is more significant than a shared hub.
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `u` - First node
/// * `v` - Second node
///
/// # Returns
///
/// A non-negative score. Higher values indicate stronger similarity.
/// Returns 0.0 if the nodes share no common neighbors.
///
/// # Complexity
///
/// O(deg(u) + deg(v) + Σ deg(w) for w ∈ N(u) ∩ N(v))
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::adamic_adar;
/// let score = adamic_adar(&store, node_a, node_b);
/// assert!(score >= 0.0);
/// ```
pub fn adamic_adar(store: &dyn GraphStore, u: NodeId, v: NodeId) -> f64 {
    let nu = neighbor_set(store, u);
    let nv = neighbor_set(store, v);

    let mut score = 0.0;
    for &w in nu.intersection(&nv) {
        let degree = neighbor_set(store, w).len();
        if degree > 1 {
            score += 1.0 / (degree as f64).ln();
        }
        // degree <= 1: log(1)=0, skip to avoid division by zero
    }
    score
}

// ============================================================================
// Resource Allocation Index
// ============================================================================

/// Computes the Resource Allocation index between two nodes.
///
/// The Resource Allocation index sums `1 / |N(w)|` for each common neighbor `w`.
/// It models a resource-spreading process where each common neighbor distributes
/// a unit resource equally among its neighbors.
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `u` - First node
/// * `v` - Second node
///
/// # Returns
///
/// A non-negative score. Higher values indicate stronger similarity.
/// Returns 0.0 if the nodes share no common neighbors.
///
/// # Complexity
///
/// O(deg(u) + deg(v) + Σ deg(w) for w ∈ N(u) ∩ N(v))
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::resource_allocation;
/// let score = resource_allocation(&store, node_a, node_b);
/// assert!(score >= 0.0);
/// ```
pub fn resource_allocation(store: &dyn GraphStore, u: NodeId, v: NodeId) -> f64 {
    let nu = neighbor_set(store, u);
    let nv = neighbor_set(store, v);

    let mut score = 0.0;
    for &w in nu.intersection(&nv) {
        let degree = neighbor_set(store, w).len();
        if degree > 0 {
            score += 1.0 / degree as f64;
        }
    }
    score
}

// ============================================================================
// Top-K Similar
// ============================================================================

/// A similarity result entry: a neighbor node and its similarity score.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityScore {
    /// The neighbor node ID.
    pub node: NodeId,
    /// The similarity score.
    pub score: f64,
}

/// Similarity metric to use for top-k computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Jaccard similarity: `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`
    Jaccard,
    /// Overlap coefficient: `|N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)`
    Overlap,
    /// Cosine similarity: `|N(u) ∩ N(v)| / (√|N(u)| × √|N(v)|)`
    Cosine,
    /// Adamic-Adar index: `Σ 1/log(|N(w)|)` for common neighbors `w`
    AdamicAdar,
    /// Resource Allocation: `Σ 1/|N(w)|` for common neighbors `w`
    ResourceAllocation,
}

impl SimilarityMetric {
    /// Parses a metric name from a string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "jaccard" => Some(SimilarityMetric::Jaccard),
            "overlap" => Some(SimilarityMetric::Overlap),
            "cosine" => Some(SimilarityMetric::Cosine),
            "adamic_adar" | "adamicadar" | "adamic-adar" => Some(SimilarityMetric::AdamicAdar),
            "resource_allocation" | "resourceallocation" | "resource-allocation" => {
                Some(SimilarityMetric::ResourceAllocation)
            }
            _ => None,
        }
    }
}

/// Finds the K most similar nodes to a given node, using the specified metric.
///
/// Scans all nodes in the graph, computes the similarity to `node`, and returns
/// the top K results sorted by descending score. Nodes with zero similarity are
/// excluded from the results.
///
/// # Arguments
///
/// * `store` - The graph store to query
/// * `node` - The reference node
/// * `k` - Maximum number of results to return
/// * `metric` - The similarity metric to use
///
/// # Returns
///
/// A `Vec<SimilarityScore>` of at most `k` entries, sorted by descending score.
///
/// # Complexity
///
/// O(V × (deg(node) + deg(v))) where V is the total node count, since every
/// node is compared against `node`. Uses a min-heap of size K for efficient
/// top-k selection: O(V × log(K)) for the heap operations.
///
/// # Example
///
/// ```ignore
/// use obrain_adapters::plugins::algorithms::{top_k_similar, SimilarityMetric};
/// let results = top_k_similar(&store, node_id, 5, SimilarityMetric::Jaccard);
/// for r in &results {
///     println!("Node {:?} — score: {:.4}", r.node, r.score);
/// }
/// ```
pub fn top_k_similar(
    store: &dyn GraphStore,
    node: NodeId,
    k: usize,
    metric: SimilarityMetric,
) -> Vec<SimilarityScore> {
    if k == 0 {
        return Vec::new();
    }

    let all_nodes = store.node_ids();

    // Use a min-heap of size k for efficient top-k selection.
    // We store (score_bits, node_id) where score_bits is the IEEE 754 bits
    // of the f64 score (works for non-negative values).
    let mut heap: BinaryHeap<std::cmp::Reverse<(u64, u64)>> = BinaryHeap::new();

    let similarity_fn = |store: &dyn GraphStore, u: NodeId, v: NodeId| -> f64 {
        match metric {
            SimilarityMetric::Jaccard => jaccard(store, u, v),
            SimilarityMetric::Overlap => overlap_coefficient(store, u, v),
            SimilarityMetric::Cosine => cosine_similarity(store, u, v),
            SimilarityMetric::AdamicAdar => adamic_adar(store, u, v),
            SimilarityMetric::ResourceAllocation => resource_allocation(store, u, v),
        }
    };

    for &candidate in &all_nodes {
        if candidate == node {
            continue;
        }

        let score = similarity_fn(store, node, candidate);
        if score <= 0.0 {
            continue;
        }

        let bits = score.to_bits();
        if heap.len() < k {
            heap.push(std::cmp::Reverse((bits, candidate.0)));
        } else if let Some(&std::cmp::Reverse((min_bits, _))) = heap.peek()
            && bits > min_bits
        {
            heap.pop();
            heap.push(std::cmp::Reverse((bits, candidate.0)));
        }
    }

    // Extract results. into_sorted_vec() on BinaryHeap<Reverse<T>> gives
    // ascending Reverse order = descending original order. No need to reverse.
    let results: Vec<SimilarityScore> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|std::cmp::Reverse((bits, nid))| SimilarityScore {
            node: NodeId(nid),
            score: f64::from_bits(bits),
        })
        .collect();

    results
}

// ============================================================================
// GraphAlgorithm implementations
// ============================================================================

/// Pairwise node similarity algorithm (Jaccard, Overlap, Cosine, Adamic-Adar, Resource Allocation).
///
/// Implements the [`GraphAlgorithm`] trait for use via `CALL obrain.similarity(node1, node2, metric)`.
///
/// # Parameters
///
/// | Name | Type | Required | Description |
/// | ---- | ---- | -------- | ----------- |
/// | `node1` | NodeId | yes | First node ID |
/// | `node2` | NodeId | yes | Second node ID |
/// | `metric` | String | no | Metric name (default: `"jaccard"`) |
///
/// # Output columns
///
/// `node1`, `node2`, `metric`, `score`
pub struct NodeSimilarityAlgorithm;

fn similarity_params() -> &'static [ParameterDef] {
    static PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();
    PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "node1".to_string(),
                description: "First node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "node2".to_string(),
                description: "Second node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "metric".to_string(),
                description:
                    "Similarity metric: jaccard, overlap, cosine, adamic_adar, resource_allocation"
                        .to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("jaccard".to_string()),
            },
        ]
    })
}

impl GraphAlgorithm for NodeSimilarityAlgorithm {
    fn name(&self) -> &str {
        "similarity"
    }

    fn description(&self) -> &str {
        "Pairwise node similarity (Jaccard, Overlap, Cosine, Adamic-Adar, Resource Allocation)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        similarity_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let node1_id = params.get_int("node1").ok_or_else(|| {
            Error::Internal("similarity: missing required parameter 'node1'".into())
        })?;
        let node2_id = params.get_int("node2").ok_or_else(|| {
            Error::Internal("similarity: missing required parameter 'node2'".into())
        })?;

        let metric_name = params.get_string("metric").unwrap_or("jaccard");
        let metric = SimilarityMetric::from_str(metric_name).ok_or_else(|| {
            Error::Internal(format!(
                "similarity: unknown metric '{}'. Valid: jaccard, overlap, cosine, adamic_adar, resource_allocation",
                metric_name
            ))
        })?;

        let u = NodeId(node1_id as u64);
        let v = NodeId(node2_id as u64);

        let score = match metric {
            SimilarityMetric::Jaccard => jaccard(store, u, v),
            SimilarityMetric::Overlap => overlap_coefficient(store, u, v),
            SimilarityMetric::Cosine => cosine_similarity(store, u, v),
            SimilarityMetric::AdamicAdar => adamic_adar(store, u, v),
            SimilarityMetric::ResourceAllocation => resource_allocation(store, u, v),
        };

        let mut result = AlgorithmResult::new(vec![
            "node1".to_string(),
            "node2".to_string(),
            "metric".to_string(),
            "score".to_string(),
        ]);
        result.add_row(vec![
            Value::Int64(node1_id),
            Value::Int64(node2_id),
            Value::from(metric_name),
            Value::Float64(score),
        ]);
        Ok(result)
    }
}

/// Top-K similar nodes algorithm.
///
/// Implements the [`GraphAlgorithm`] trait for use via
/// `CALL obrain.similarity.topk(node, k, metric)`.
///
/// # Parameters
///
/// | Name | Type | Required | Description |
/// | ---- | ---- | -------- | ----------- |
/// | `node` | NodeId | yes | Reference node ID |
/// | `k` | Integer | no | Max results (default: 10) |
/// | `metric` | String | no | Metric name (default: `"jaccard"`) |
///
/// # Output columns
///
/// `neighbor`, `score`
pub struct TopKSimilarAlgorithm;

fn topk_params() -> &'static [ParameterDef] {
    static PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();
    PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "node".to_string(),
                description: "Reference node ID".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "k".to_string(),
                description: "Maximum number of similar nodes to return".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("10".to_string()),
            },
            ParameterDef {
                name: "metric".to_string(),
                description:
                    "Similarity metric: jaccard, overlap, cosine, adamic_adar, resource_allocation"
                        .to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("jaccard".to_string()),
            },
        ]
    })
}

impl GraphAlgorithm for TopKSimilarAlgorithm {
    fn name(&self) -> &str {
        "similarity.topk"
    }

    fn description(&self) -> &str {
        "Find top-K most similar nodes by neighborhood overlap"
    }

    fn parameters(&self) -> &[ParameterDef] {
        topk_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let node_id = params.get_int("node").ok_or_else(|| {
            Error::Internal("similarity.topk: missing required parameter 'node'".into())
        })?;

        let k = params.get_int("k").unwrap_or(10) as usize;

        let metric_name = params.get_string("metric").unwrap_or("jaccard");
        let metric = SimilarityMetric::from_str(metric_name).ok_or_else(|| {
            Error::Internal(format!(
                "similarity.topk: unknown metric '{}'. Valid: jaccard, overlap, cosine, adamic_adar, resource_allocation",
                metric_name
            ))
        })?;

        let node = NodeId(node_id as u64);
        let results = top_k_similar(store, node, k, metric);

        let mut output = AlgorithmResult::new(vec!["neighbor".to_string(), "score".to_string()]);
        for entry in results {
            output.add_row(vec![
                Value::Int64(entry.node.0 as i64),
                Value::Float64(entry.score),
            ]);
        }
        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a test graph:
    ///
    /// ```text
    ///   A --- B --- C
    ///   |     |
    ///   D --- E
    /// ```
    ///
    /// Undirected (edges in both directions).
    #[allow(clippy::many_single_char_names)]
    fn create_test_graph() -> (SubstrateStore, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let store = SubstrateStore::open_tempfile().unwrap();

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let e = store.create_node(&["Node"]);

        // A -- B (bidirectional for undirected)
        store.create_edge(a, b, "CONNECTS");
        store.create_edge(b, a, "CONNECTS");

        // B -- C
        store.create_edge(b, c, "CONNECTS");
        store.create_edge(c, b, "CONNECTS");

        // A -- D
        store.create_edge(a, d, "CONNECTS");
        store.create_edge(d, a, "CONNECTS");

        // B -- E
        store.create_edge(b, e, "CONNECTS");
        store.create_edge(e, b, "CONNECTS");

        // D -- E
        store.create_edge(d, e, "CONNECTS");
        store.create_edge(e, d, "CONNECTS");

        (store, a, b, c, d, e)
    }

    // ========================================================================
    // Jaccard tests
    // ========================================================================

    #[test]
    fn test_jaccard_identical_neighborhoods() {
        // Two nodes with identical neighborhoods should have Jaccard = 1.0
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);

        // Both a and b connect to c (and each other)
        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, c, "E");
        store.create_edge(c, b, "E");
        store.create_edge(a, b, "E");
        store.create_edge(b, a, "E");

        // N(a) = {b, c}, N(b) = {a, c}
        // intersection(considering u,v excluded from own neighbor sets) = {c}
        // Actually N(a)={b,c}, N(b)={a,c}, intersection={c}, union={a,b,c}
        let score = jaccard(&store, a, b);
        // |{c}| / |{a, b, c}| = 1/3
        assert!((score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_no_common_neighbors() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);

        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, d, "E");
        store.create_edge(d, b, "E");

        // N(a) = {c}, N(b) = {d}, intersection = {}, union = {c, d}
        let score = jaccard(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_isolated_nodes() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);

        let score = jaccard(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_hand_computed() {
        let (store, a, b, _c, _d, _e) = create_test_graph();
        // N(A) = {B, D}
        // N(B) = {A, C, E}
        // intersection = {} (no common neighbors)
        // union = {A, B, C, D, E} — wait, union is of the neighbor sets
        // N(A) = {B, D}, N(B) = {A, C, E}
        // intersection(N(A), N(B)) = {} (B is in N(A) but not in N(B); A is in N(B) but not in N(A))
        // Actually: B ∈ N(A) but B ∉ N(B) (a node is not its own neighbor); A ∈ N(B) but A ∉ N(A)
        // So intersection = {} and union = {B, D, A, C, E} = 5 elements
        // Jaccard = 0/5 = 0.0
        let score = jaccard(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_with_common_neighbor() {
        let (store, a, _b, _c, d, e) = create_test_graph();
        // N(A) = {B, D}
        // N(E) = {B, D}
        // intersection = {B, D} = 2
        // union = {B, D} = 2
        // Jaccard = 2/2 = 1.0
        let score = jaccard(&store, a, e);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "A and E have same neighbors: score={score}"
        );

        // N(D) = {A, E}
        // N(A) = {B, D}
        // intersection(N(D), N(A)) = {} — A ∈ N(D) but A ∉ N(A)? No: D ∈ N(A) but D ∉ N(D)? D is not its own neighbor
        // N(D) = {A, E}, N(A) = {B, D}
        // intersection = {} (A not in {B,D}; wait: A IS in N(D)? and A is in N(A)? No, A is NOT in N(A))
        // Hmm: D ∈ N(A)? yes. D ∈ N(D)? no. A ∈ N(D)? yes. A ∈ N(A)? no.
        // So intersection = {} and union = {A, E, B, D} = 4
        // Jaccard(D, A) = 0/4 = 0.0
        let score_da = jaccard(&store, d, a);
        assert!((score_da - 0.0).abs() < 1e-10);
    }

    // ========================================================================
    // Overlap coefficient tests
    // ========================================================================

    #[test]
    fn test_overlap_same_neighbors() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        // N(A) = {B, D}, N(E) = {B, D}
        // intersection = {B, D} = 2, min(2,2) = 2
        // overlap = 2/2 = 1.0
        let score = overlap_coefficient(&store, a, e);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_isolated() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let score = overlap_coefficient(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_subset() {
        // If N(u) ⊂ N(v), overlap should be 1.0
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);

        // N(a) = {c}
        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        // N(b) = {c, d}
        store.create_edge(b, c, "E");
        store.create_edge(c, b, "E");
        store.create_edge(b, d, "E");
        store.create_edge(d, b, "E");

        // intersection = {c} = 1, min(1, 2) = 1, overlap = 1/1 = 1.0
        let score = overlap_coefficient(&store, a, b);
        assert!((score - 1.0).abs() < 1e-10);
    }

    // ========================================================================
    // Cosine similarity tests
    // ========================================================================

    #[test]
    fn test_cosine_same_neighbors() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        // N(A) = {B, D}, N(E) = {B, D}
        // intersection = 2, sqrt(2)*sqrt(2) = 2
        // cosine = 2/2 = 1.0
        let score = cosine_similarity(&store, a, e);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_isolated() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let score = cosine_similarity(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_cosine_hand_computed() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);
        let e = store.create_node(&["N"]);

        // N(a) = {c, d, e} (3 neighbors)
        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(a, d, "E");
        store.create_edge(d, a, "E");
        store.create_edge(a, e, "E");
        store.create_edge(e, a, "E");

        // N(b) = {c, d} (2 neighbors)
        store.create_edge(b, c, "E");
        store.create_edge(c, b, "E");
        store.create_edge(b, d, "E");
        store.create_edge(d, b, "E");

        // intersection = {c, d} = 2
        // cosine = 2 / (sqrt(3) * sqrt(2)) = 2 / sqrt(6) ≈ 0.8165
        let expected = 2.0 / (3.0_f64.sqrt() * 2.0_f64.sqrt());
        let score = cosine_similarity(&store, a, b);
        assert!(
            (score - expected).abs() < 1e-10,
            "Expected {expected}, got {score}"
        );
    }

    // ========================================================================
    // Adamic-Adar tests
    // ========================================================================

    #[test]
    fn test_adamic_adar_no_common() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);

        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, d, "E");
        store.create_edge(d, b, "E");

        let score = adamic_adar(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_adamic_adar_hand_computed() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let w = store.create_node(&["N"]); // common neighbor
        let x = store.create_node(&["N"]); // extra neighbor of w

        // N(a) = {w}, N(b) = {w}
        store.create_edge(a, w, "E");
        store.create_edge(w, a, "E");
        store.create_edge(b, w, "E");
        store.create_edge(w, b, "E");
        // Give w an extra neighbor so deg(w)=3 (a, b, x)
        store.create_edge(w, x, "E");
        store.create_edge(x, w, "E");

        // common neighbor: w with deg=3
        // AA = 1/ln(3)
        let expected = 1.0 / 3.0_f64.ln();
        let score = adamic_adar(&store, a, b);
        assert!(
            (score - expected).abs() < 1e-10,
            "Expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_adamic_adar_degree_one_common() {
        // A common neighbor with degree 1 is skipped (log(1) = 0)
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let w = store.create_node(&["N"]);

        // w only connects to a (deg=1)
        store.create_edge(a, w, "E");
        store.create_edge(w, a, "E");
        // w also connects to b (deg=2 now)
        store.create_edge(b, w, "E");
        store.create_edge(w, b, "E");

        // deg(w) = 2 (a, b), so AA = 1/ln(2)
        let expected = 1.0 / 2.0_f64.ln();
        let score = adamic_adar(&store, a, b);
        assert!(
            (score - expected).abs() < 1e-10,
            "Expected {expected}, got {score}"
        );
    }

    // ========================================================================
    // Resource Allocation tests
    // ========================================================================

    #[test]
    fn test_resource_allocation_no_common() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        let d = store.create_node(&["N"]);

        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, d, "E");
        store.create_edge(d, b, "E");

        let score = resource_allocation(&store, a, b);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_resource_allocation_hand_computed() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let w = store.create_node(&["N"]);
        let x = store.create_node(&["N"]);

        store.create_edge(a, w, "E");
        store.create_edge(w, a, "E");
        store.create_edge(b, w, "E");
        store.create_edge(w, b, "E");
        store.create_edge(w, x, "E");
        store.create_edge(x, w, "E");

        // common neighbor: w with deg=3 (a, b, x)
        // RA = 1/3
        let expected = 1.0 / 3.0;
        let score = resource_allocation(&store, a, b);
        assert!(
            (score - expected).abs() < 1e-10,
            "Expected {expected}, got {score}"
        );
    }

    // ========================================================================
    // Top-K tests
    // ========================================================================

    #[test]
    fn test_top_k_basic() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        // A and E have identical neighborhoods {B, D}
        let results = top_k_similar(&store, a, 3, SimilarityMetric::Jaccard);
        assert!(!results.is_empty());
        // E should be the most similar to A
        assert_eq!(results[0].node, e, "E should be most similar to A");
        assert!((results[0].score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_respects_limit() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();
        let results = top_k_similar(&store, a, 1, SimilarityMetric::Jaccard);
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_top_k_zero() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();
        let results = top_k_similar(&store, a, 0, SimilarityMetric::Jaccard);
        assert!(results.is_empty());
    }

    #[test]
    fn test_top_k_all_metrics() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();
        for metric in [
            SimilarityMetric::Jaccard,
            SimilarityMetric::Overlap,
            SimilarityMetric::Cosine,
            SimilarityMetric::AdamicAdar,
            SimilarityMetric::ResourceAllocation,
        ] {
            let results = top_k_similar(&store, a, 5, metric);
            // Should not panic
            for r in &results {
                assert!(r.score > 0.0, "All returned scores should be positive");
            }
        }
    }

    #[test]
    fn test_top_k_descending_order() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();
        let results = top_k_similar(&store, a, 10, SimilarityMetric::Jaccard);
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results should be in descending order"
            );
        }
    }

    // ========================================================================
    // SimilarityMetric parsing tests
    // ========================================================================

    #[test]
    fn test_metric_from_str() {
        assert_eq!(
            SimilarityMetric::from_str("jaccard"),
            Some(SimilarityMetric::Jaccard)
        );
        assert_eq!(
            SimilarityMetric::from_str("JACCARD"),
            Some(SimilarityMetric::Jaccard)
        );
        assert_eq!(
            SimilarityMetric::from_str("overlap"),
            Some(SimilarityMetric::Overlap)
        );
        assert_eq!(
            SimilarityMetric::from_str("cosine"),
            Some(SimilarityMetric::Cosine)
        );
        assert_eq!(
            SimilarityMetric::from_str("adamic_adar"),
            Some(SimilarityMetric::AdamicAdar)
        );
        assert_eq!(
            SimilarityMetric::from_str("adamic-adar"),
            Some(SimilarityMetric::AdamicAdar)
        );
        assert_eq!(
            SimilarityMetric::from_str("resource_allocation"),
            Some(SimilarityMetric::ResourceAllocation)
        );
        assert_eq!(SimilarityMetric::from_str("unknown"), None);
    }

    // ========================================================================
    // GraphAlgorithm trait tests
    // ========================================================================

    #[test]
    fn test_similarity_algorithm_execute() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        let algo = NodeSimilarityAlgorithm;

        let mut params = Parameters::new();
        params.set_int("node1", a.0 as i64);
        params.set_int("node2", e.0 as i64);
        params.set_string("metric", "jaccard");

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node1", "node2", "metric", "score"]);
        assert_eq!(result.rows.len(), 1);

        // A and E should have jaccard = 1.0
        if let Value::Float64(score) = &result.rows[0][3] {
            assert!((*score - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected Float64 score");
        }
    }

    #[test]
    fn test_similarity_algorithm_missing_params() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let algo = NodeSimilarityAlgorithm;
        let params = Parameters::new();
        assert!(algo.execute(&store, &params).is_err());
    }

    #[test]
    fn test_similarity_algorithm_invalid_metric() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let algo = NodeSimilarityAlgorithm;

        let mut params = Parameters::new();
        params.set_int("node1", a.0 as i64);
        params.set_int("node2", b.0 as i64);
        params.set_string("metric", "invalid_metric");

        assert!(algo.execute(&store, &params).is_err());
    }

    #[test]
    fn test_topk_algorithm_execute() {
        let (store, a, _b, _c, _d, e) = create_test_graph();
        let algo = TopKSimilarAlgorithm;

        let mut params = Parameters::new();
        params.set_int("node", a.0 as i64);
        params.set_int("k", 2);
        params.set_string("metric", "jaccard");

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["neighbor", "score"]);
        assert!(result.rows.len() <= 2);

        // First result should be E with score 1.0
        if let Value::Int64(neighbor_id) = &result.rows[0][0] {
            assert_eq!(*neighbor_id as u64, e.0);
        }
        if let Value::Float64(score) = &result.rows[0][1] {
            assert!((*score - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_topk_algorithm_missing_node() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let algo = TopKSimilarAlgorithm;
        let params = Parameters::new();
        assert!(algo.execute(&store, &params).is_err());
    }

    #[test]
    fn test_topk_default_params() {
        let (store, a, _b, _c, _d, _e) = create_test_graph();
        let algo = TopKSimilarAlgorithm;

        let mut params = Parameters::new();
        params.set_int("node", a.0 as i64);
        // k and metric should use defaults (10 and jaccard)

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["neighbor", "score"]);
    }

    #[test]
    fn test_algorithm_names() {
        assert_eq!(NodeSimilarityAlgorithm.name(), "similarity");
        assert_eq!(TopKSimilarAlgorithm.name(), "similarity.topk");
    }

    #[test]
    fn test_algorithm_descriptions() {
        assert!(!NodeSimilarityAlgorithm.description().is_empty());
        assert!(!TopKSimilarAlgorithm.description().is_empty());
    }

    #[test]
    fn test_algorithm_parameters_defined() {
        assert_eq!(NodeSimilarityAlgorithm.parameters().len(), 3);
        assert_eq!(TopKSimilarAlgorithm.parameters().len(), 3);
    }

    // ========================================================================
    // Edge-case tests for coverage
    // ========================================================================

    #[test]
    fn test_similarity_all_metrics_via_algorithm() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, c, "E");
        store.create_edge(c, b, "E");

        let algo = NodeSimilarityAlgorithm;
        for metric in [
            "jaccard",
            "overlap",
            "cosine",
            "adamic_adar",
            "resource_allocation",
        ] {
            let mut params = Parameters::new();
            params.set_int("node1", a.0 as i64);
            params.set_int("node2", b.0 as i64);
            params.set_string("metric", metric);
            let result = algo.execute(&store, &params).unwrap();
            assert_eq!(result.rows.len(), 1, "metric={metric}");
        }
    }

    #[test]
    fn test_topk_all_metrics_via_algorithm() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let b = store.create_node(&["N"]);
        let c = store.create_node(&["N"]);
        store.create_edge(a, c, "E");
        store.create_edge(c, a, "E");
        store.create_edge(b, c, "E");
        store.create_edge(c, b, "E");

        let algo = TopKSimilarAlgorithm;
        for metric in [
            "jaccard",
            "overlap",
            "cosine",
            "adamic_adar",
            "resource_allocation",
        ] {
            let mut params = Parameters::new();
            params.set_int("node", a.0 as i64);
            params.set_int("k", 5);
            params.set_string("metric", metric);
            let result = algo.execute(&store, &params).unwrap();
            assert!(result.rows.len() <= 5, "metric={metric}");
        }
    }

    #[test]
    fn test_topk_invalid_metric_via_algorithm() {
        let store = SubstrateStore::open_tempfile().unwrap();
        let a = store.create_node(&["N"]);
        let algo = TopKSimilarAlgorithm;

        let mut params = Parameters::new();
        params.set_int("node", a.0 as i64);
        params.set_string("metric", "nonexistent");

        assert!(algo.execute(&store, &params).is_err());
    }

    #[test]
    fn test_metric_aliases() {
        assert_eq!(
            SimilarityMetric::from_str("adamicadar"),
            Some(SimilarityMetric::AdamicAdar)
        );
        assert_eq!(
            SimilarityMetric::from_str("resourceallocation"),
            Some(SimilarityMetric::ResourceAllocation)
        );
        assert_eq!(
            SimilarityMetric::from_str("resource-allocation"),
            Some(SimilarityMetric::ResourceAllocation)
        );
    }
}
