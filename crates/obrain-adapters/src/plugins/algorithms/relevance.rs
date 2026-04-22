//! Personalized PageRank (PPR) and relevance subgraph extraction.
//!
//! Standard PageRank computes global importance. **Personalized PageRank**
//! concentrates the random-walk restart on a set of **seed nodes**, producing
//! scores that reflect proximity/relevance to those seeds rather than global
//! centrality.
//!
//! This is the core primitive for "given these nodes, what else in the graph
//! is most relevant?" — used by obrain-chat to extract local context from a
//! large knowledge graph.
//!
//! # Algorithm
//!
//! Iterative power-iteration with teleport:
//!
//! ```text
//! r(t+1) = (1 - d) * teleport + d * M^T * r(t)
//! ```
//!
//! where `teleport` is uniform over seeds, `d` is the damping factor (0.85),
//! and `M` is the column-stochastic transition matrix.
//!
//! Convergence is checked via L1 norm of the delta vector; early exit when
//! `delta < epsilon` (default 1e-8).
//!
//! # Example
//!
//! ```no_run
//! use obrain_core::graph::GraphStoreMut;
//! use obrain_substrate::SubstrateStore;
//! use obrain_adapters::plugins::algorithms::relevance::{personalized_pagerank, PprConfig};
//!
//! let store = SubstrateStore::open_tempfile().unwrap();
//! // ... populate graph ...
//! let seeds = vec![store.create_node(&["Seed"])];
//! let config = PprConfig::default();
//! let result = personalized_pagerank(&store, &seeds, &config);
//! // result.scores: top nodes ranked by relevance to seeds
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
use obrain_common::utils::hash::FxHashMap;
use obrain_core::graph::{Direction, GraphStore, GraphStoreMut};
use obrain_substrate::SubstrateStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Personalized PageRank.
#[derive(Debug, Clone)]
pub struct PprConfig {
    /// Damping factor — probability of following an edge vs teleporting back
    /// to a seed node (default: 0.85).
    pub damping: f64,
    /// Maximum number of iterations (default: 50).
    pub iterations: usize,
    /// Maximum number of nodes in the extracted subgraph (default: 50).
    pub budget: usize,
    /// Minimum score to include a node in results (default: 1e-6).
    pub min_score: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            iterations: 50,
            budget: 50,
            min_score: 1e-6,
        }
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of Personalized PageRank computation.
#[derive(Debug, Clone)]
pub struct PprResult {
    /// Score per node — higher means more relevant to the seeds.
    pub scores: HashMap<NodeId, f64>,
    /// Number of iterations actually used (may be less than max if converged).
    pub iterations_used: usize,
}

// ============================================================================
// Core algorithm
// ============================================================================

/// Compute Personalized PageRank from a set of seed nodes.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `seeds` - Seed node IDs (teleport targets)
/// * `config` - PPR configuration (damping, iterations, budget, min_score)
///
/// # Returns
///
/// `PprResult` with scores for all reachable nodes and the number of iterations used.
///
/// # Complexity
///
/// O(iterations × E) where E is the number of edges.
pub fn personalized_pagerank(
    store: &dyn GraphStore,
    seeds: &[NodeId],
    config: &PprConfig,
) -> PprResult {
    if seeds.is_empty() {
        return PprResult {
            scores: HashMap::new(),
            iterations_used: 0,
        };
    }

    let node_ids = store.node_ids();
    let n = node_ids.len();

    if n == 0 {
        return PprResult {
            scores: HashMap::new(),
            iterations_used: 0,
        };
    }

    // Build node index
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &nid) in node_ids.iter().enumerate() {
        node_to_idx.insert(nid, idx);
    }

    // Teleport vector: uniform over seeds
    let mut teleport = vec![0.0_f64; n];
    let seed_weight = 1.0 / seeds.len() as f64;
    for &seed in seeds {
        if let Some(&idx) = node_to_idx.get(&seed) {
            teleport[idx] = seed_weight;
        }
    }

    // Build adjacency: for each node, list of (neighbor_idx, weight)
    // We follow outgoing edges for the random walk
    let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, &nid) in node_ids.iter().enumerate() {
        for neighbor in store.neighbors(nid, Direction::Outgoing) {
            if let Some(&nidx) = node_to_idx.get(&neighbor) {
                out_neighbors[idx].push(nidx);
            }
        }
    }

    // Initialize scores: teleport distribution
    let mut scores = teleport.clone();
    let epsilon = 1e-8;
    let d = config.damping;
    let mut iterations_used = 0;

    for iter in 0..config.iterations {
        let mut new_scores = vec![0.0_f64; n];

        // Distribute score along outgoing edges
        for i in 0..n {
            let out_deg = out_neighbors[i].len();
            if out_deg > 0 {
                let share = scores[i] / out_deg as f64;
                for &j in &out_neighbors[i] {
                    new_scores[j] += d * share;
                }
            } else {
                // Dangling node: redistribute uniformly to seeds (not all nodes)
                let share = scores[i] / seeds.len() as f64;
                for &seed in seeds {
                    if let Some(&sidx) = node_to_idx.get(&seed) {
                        new_scores[sidx] += d * share;
                    }
                }
            }
        }

        // Add teleport
        for i in 0..n {
            new_scores[i] += (1.0 - d) * teleport[i];
        }

        // Check convergence (L1 norm)
        let delta: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        scores = new_scores;
        iterations_used = iter + 1;

        if delta < epsilon {
            break;
        }
    }

    // Build result: filter by min_score
    let result_scores: HashMap<NodeId, f64> = node_ids
        .iter()
        .enumerate()
        .filter_map(|(idx, &nid)| {
            let s = scores[idx];
            if s >= config.min_score {
                Some((nid, s))
            } else {
                None
            }
        })
        .collect();

    PprResult {
        scores: result_scores,
        iterations_used,
    }
}

/// Extract a relevance subgraph: top-K nodes by PPR score + induced edges.
///
/// # Arguments
///
/// * `store` - The source graph store
/// * `ppr` - PPR result (scores)
/// * `budget` - Maximum number of nodes to include
///
/// # Returns
///
/// A new `SubstrateStore` containing the top-K nodes and all edges between them.
///
/// # Complexity
///
/// O(K log K + K × avg_degree) for sorting + edge induction.
pub fn extract_subgraph(
    store: &dyn GraphStore,
    ppr: &PprResult,
    budget: usize,
) -> SubstrateStore {
    let sub = SubstrateStore::open_tempfile().expect("SubstrateStore::open_tempfile");

    if ppr.scores.is_empty() || budget == 0 {
        return sub;
    }

    // Sort nodes by score descending, take top budget
    let mut sorted: Vec<(NodeId, f64)> = ppr.scores.iter().map(|(&k, &v)| (k, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(budget);

    let selected: FxHashMap<NodeId, NodeId> = {
        let mut map = FxHashMap::default();
        for &(nid, _) in &sorted {
            // Copy node with labels
            if let Some(node) = store.get_node(nid) {
                let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
                let new_id = sub.create_node(&label_refs);
                // Copy properties
                if let Some(original) = store.get_node(nid) {
                    for (key, value) in &original.properties {
                        sub.set_node_property(new_id, key.as_str(), value.clone());
                    }
                }
                // Store PPR score as property
                sub.set_node_property(new_id, "_ppr_score", Value::Float64(ppr.scores[&nid]));
                map.insert(nid, new_id);
            }
        }
        map
    };

    // Induce edges: add all edges between selected nodes
    for &(src_orig, _) in &sorted {
        if let Some(&src_new) = selected.get(&src_orig) {
            for (dst_orig, edge_id) in store.edges_from(src_orig, Direction::Outgoing) {
                if let Some(&dst_new) = selected.get(&dst_orig)
                    && let Some(edge_type) = store.edge_type(edge_id)
                {
                    sub.create_edge(src_new, dst_new, &edge_type);
                }
            }
        }
    }

    sub
}

// ============================================================================
// GraphAlgorithm trait impl
// ============================================================================

/// Static parameter definitions for PPR.
static PPR_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn ppr_params() -> &'static [ParameterDef] {
    PPR_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "damping".to_string(),
                description: "Damping factor / teleport probability (default: 0.85)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("0.85".to_string()),
            },
            ParameterDef {
                name: "iterations".to_string(),
                description: "Maximum iterations (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
            ParameterDef {
                name: "budget".to_string(),
                description: "Max nodes in extracted subgraph (default: 50)".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("50".to_string()),
            },
            ParameterDef {
                name: "seed_nodes".to_string(),
                description: "Comma-separated seed node IDs".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            },
        ]
    })
}

/// Personalized PageRank algorithm wrapper for the registry.
pub struct PersonalizedPageRankAlgorithm;

impl GraphAlgorithm for PersonalizedPageRankAlgorithm {
    fn name(&self) -> &str {
        "personalized_pagerank"
    }

    fn description(&self) -> &str {
        "Personalized PageRank — relevance scores from seed nodes"
    }

    fn parameters(&self) -> &[ParameterDef] {
        ppr_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let damping = params.get_float("damping").unwrap_or(0.85);
        let iterations = params.get_int("iterations").unwrap_or(50) as usize;
        let budget = params.get_int("budget").unwrap_or(50) as usize;

        let seeds: Vec<NodeId> = params
            .get_string("seed_nodes")
            .unwrap_or("")
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok().map(NodeId))
            .collect();

        let config = PprConfig {
            damping,
            iterations,
            budget,
            min_score: 1e-6,
        };

        let ppr = personalized_pagerank(store, &seeds, &config);

        let mut output = AlgorithmResult::new(vec!["node_id".to_string(), "score".to_string()]);

        // Sort by score descending for consistent output
        let mut entries: Vec<_> = ppr.scores.iter().collect();
        entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (&node, &score) in entries {
            output.add_row(vec![Value::Int64(node.0 as i64), Value::Float64(score)]);
        }

        Ok(output)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "algos"))]
mod tests {
    use super::*;

    /// Star graph: center connected to n leaves.
    fn create_star(n: usize) -> (SubstrateStore, NodeId, Vec<NodeId>) {
        let store = SubstrateStore::open_tempfile().unwrap();
        let center = store.create_node(&["Center"]);
        let leaves: Vec<NodeId> = (0..n)
            .map(|_| {
                let leaf = store.create_node(&["Leaf"]);
                store.create_edge(center, leaf, "POINTS_TO");
                leaf
            })
            .collect();
        (store, center, leaves)
    }

    #[test]
    fn test_ppr_single_seed() {
        let (store, center, _leaves) = create_star(5);
        let config = PprConfig::default();
        let result = personalized_pagerank(&store, &[center], &config);

        // Center (seed) should have the highest score
        let center_score = result.scores.get(&center).copied().unwrap_or(0.0);
        for (&nid, &score) in &result.scores {
            if nid != center {
                assert!(
                    center_score >= score,
                    "seed score {} should be >= leaf score {}",
                    center_score,
                    score
                );
            }
        }
        assert!(result.iterations_used > 0);
    }

    #[test]
    fn test_ppr_multi_seed() {
        // Two clusters, seed in each
        let store = SubstrateStore::open_tempfile().unwrap();
        // Cluster A
        let a: Vec<_> = (0..5).map(|_| store.create_node(&["A"])).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                store.create_edge(a[i], a[j], "LINK");
            }
        }
        // Cluster B
        let b: Vec<_> = (0..5).map(|_| store.create_node(&["B"])).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                store.create_edge(b[i], b[j], "LINK");
            }
        }
        // Weak bridge
        store.create_edge(a[4], b[0], "BRIDGE");

        let result = personalized_pagerank(&store, &[a[0], b[0]], &PprConfig::default());

        // Both seeds should have scores
        assert!(result.scores.contains_key(&a[0]));
        assert!(result.scores.contains_key(&b[0]));
    }

    #[test]
    fn test_ppr_budget() {
        let (store, center, _leaves) = create_star(20);
        let config = PprConfig {
            budget: 5,
            ..Default::default()
        };
        let result = personalized_pagerank(&store, &[center], &config);
        let subgraph = extract_subgraph(&store, &result, 5);

        assert_eq!(subgraph.node_count(), 5);
    }

    #[test]
    fn test_ppr_disconnected() {
        // Two disconnected components, seed in one
        let store = SubstrateStore::open_tempfile().unwrap();
        let a0 = store.create_node(&["A"]);
        let a1 = store.create_node(&["A"]);
        store.create_edge(a0, a1, "LINK");

        let b0 = store.create_node(&["B"]);
        let b1 = store.create_node(&["B"]);
        store.create_edge(b0, b1, "LINK");

        let result = personalized_pagerank(&store, &[a0], &PprConfig::default());

        // Seed component should have scores, isolated component should not
        assert!(result.scores.get(&a0).copied().unwrap_or(0.0) > 0.0);
        // b0/b1 may have near-zero or zero score (no path from a0)
        let b0_score = result.scores.get(&b0).copied().unwrap_or(0.0);
        let a0_score = result.scores.get(&a0).copied().unwrap_or(0.0);
        assert!(a0_score > b0_score * 10.0, "seed component should dominate");
    }

    #[test]
    fn test_ppr_convergence() {
        let (store, center, _leaves) = create_star(10);
        let config = PprConfig {
            iterations: 1000,
            ..Default::default()
        };
        let result = personalized_pagerank(&store, &[center], &config);

        // Should converge before max iterations on a tiny graph
        assert!(
            result.iterations_used < 1000,
            "should converge early, used {} iterations",
            result.iterations_used
        );
    }
}
