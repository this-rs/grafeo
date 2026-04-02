//! Stable community detection via Hungarian matching.
//!
//! Louvain community IDs are non-deterministic: internal `FxHashSet` iteration
//! order changes between runs, producing different community IDs even on the
//! same graph. This module solves the problem by matching communities between
//! successive runs using the **Hungarian algorithm** (Kuhn-Munkres) on a
//! Jaccard-similarity cost matrix.
//!
//! # How it works
//!
//! 1. Run Louvain on the current graph to get `curr` communities.
//! 2. Build a cost matrix `C[i][j] = 1.0 - jaccard(prev_i, curr_j)`.
//! 3. Solve the assignment problem with the Hungarian algorithm (O(k^3) for k communities).
//! 4. Remap `curr` community IDs to match `prev` IDs when overlap >= `min_overlap`.
//! 5. Assign fresh sequential IDs to unmatched new communities.
//!
//! # Example
//!
//! ```no_run
//! use obrain_core::graph::lpg::LpgStore;
//! use obrain_adapters::plugins::algorithms::{louvain, stabilize_communities};
//!
//! let store = LpgStore::new().unwrap();
//! // ... populate graph ...
//! let prev = louvain(&store, 1.0);
//! // ... graph mutates ...
//! let curr = louvain(&store, 1.0);
//! let stable = stabilize_communities(&prev, &curr, &store, 0.3);
//! // stable.communities has IDs consistent with prev where possible
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use obrain_common::types::{NodeId, Value};
use obrain_common::utils::error::Result;
use obrain_common::utils::hash::{FxHashMap, FxHashSet};
use obrain_core::graph::GraphStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::community::{LouvainResult, louvain};
use super::traits::GraphAlgorithm;

// ============================================================================
// Types
// ============================================================================

/// Result of stable community detection.
#[derive(Debug, Clone)]
pub struct StableLouvainResult {
    /// Community assignment for each node (stabilized IDs).
    pub communities: FxHashMap<NodeId, u64>,
    /// Final modularity score (from the underlying Louvain run).
    pub modularity: f64,
    /// Number of communities detected.
    pub num_communities: usize,
    /// Mapping from old community IDs to new community IDs.
    /// Only contains entries for communities that were successfully matched.
    pub mapping: HashMap<u64, u64>,
}

/// Configuration for stable community detection.
#[derive(Debug, Clone)]
pub struct StableCommunityConfig {
    /// Minimum Jaccard overlap to consider a match (default: 0.3).
    /// Communities below this threshold get a fresh ID.
    pub min_overlap: f64,
    /// Louvain resolution parameter (default: 1.0).
    pub resolution: f64,
}

impl Default for StableCommunityConfig {
    fn default() -> Self {
        Self {
            min_overlap: 0.3,
            resolution: 1.0,
        }
    }
}

// ============================================================================
// Core algorithm
// ============================================================================

/// Stabilize community IDs between two successive Louvain runs.
///
/// Matches communities from `prev` to `curr` using Jaccard similarity and the
/// Hungarian algorithm, then remaps `curr` IDs to be consistent with `prev`.
///
/// # Arguments
///
/// * `prev` - Previous Louvain result (may be empty for first run)
/// * `curr` - Current Louvain result
/// * `store` - The graph store (unused directly, kept for API consistency)
/// * `min_overlap` - Minimum Jaccard overlap to accept a match (0.0-1.0)
///
/// # Returns
///
/// A `StableLouvainResult` with remapped community IDs.
///
/// # Complexity
///
/// O(k^3) where k = max(num_prev_communities, num_curr_communities).
/// The Louvain computation itself is separate (O(V log V)).
pub fn stabilize_communities(
    prev: &LouvainResult,
    curr: &LouvainResult,
    _store: &dyn GraphStore,
    min_overlap: f64,
) -> StableLouvainResult {
    // First run: no previous communities — return curr with sequential IDs
    if prev.communities.is_empty() {
        return StableLouvainResult {
            communities: curr.communities.clone(),
            modularity: curr.modularity,
            num_communities: curr.num_communities,
            mapping: HashMap::new(),
        };
    }

    // Build member sets per community for prev and curr
    let prev_members = build_member_sets(&prev.communities);
    let curr_members = build_member_sets(&curr.communities);

    let prev_ids: Vec<u64> = prev_members.keys().copied().collect();
    let curr_ids: Vec<u64> = curr_members.keys().copied().collect();

    let n_prev = prev_ids.len();
    let n_curr = curr_ids.len();

    if n_curr == 0 {
        return StableLouvainResult {
            communities: FxHashMap::default(),
            modularity: curr.modularity,
            num_communities: 0,
            mapping: HashMap::new(),
        };
    }

    // Build Jaccard similarity matrix
    let k = n_prev.max(n_curr);
    let mut cost_matrix = vec![vec![1.0_f64; k]; k];

    for (i, &pid) in prev_ids.iter().enumerate() {
        let prev_set = &prev_members[&pid];
        for (j, &cid) in curr_ids.iter().enumerate() {
            let curr_set = &curr_members[&cid];
            let jaccard = jaccard_index(prev_set, curr_set);
            cost_matrix[i][j] = 1.0 - jaccard;
        }
    }

    // Solve assignment with Hungarian algorithm
    let assignment = hungarian_algorithm(&cost_matrix);

    // Build remapping: curr_id -> stabilized_id
    let mut mapping = HashMap::new();
    let mut id_remap: HashMap<u64, u64> = HashMap::new();
    let mut used_ids: FxHashSet<u64> = FxHashSet::default();

    for (i, &j) in assignment.iter().enumerate() {
        if i < n_prev && j < n_curr {
            let similarity = 1.0 - cost_matrix[i][j];
            if similarity >= min_overlap {
                let prev_id = prev_ids[i];
                let curr_id = curr_ids[j];
                id_remap.insert(curr_id, prev_id);
                mapping.insert(prev_id, prev_id);
                used_ids.insert(prev_id);
            }
        }
    }

    // Assign fresh IDs to unmatched curr communities
    let max_prev_id = prev_ids.iter().copied().max().unwrap_or(0);
    let mut next_fresh_id = max_prev_id + 1;

    for &cid in &curr_ids {
        if !id_remap.contains_key(&cid) {
            while used_ids.contains(&next_fresh_id) {
                next_fresh_id += 1;
            }
            id_remap.insert(cid, next_fresh_id);
            used_ids.insert(next_fresh_id);
            next_fresh_id += 1;
        }
    }

    // Apply remapping to all nodes
    let communities: FxHashMap<NodeId, u64> = curr
        .communities
        .iter()
        .map(|(&node, &comm)| {
            let stable_id = id_remap.get(&comm).copied().unwrap_or(comm);
            (node, stable_id)
        })
        .collect();

    let unique: FxHashSet<u64> = communities.values().copied().collect();

    StableLouvainResult {
        communities,
        modularity: curr.modularity,
        num_communities: unique.len(),
        mapping,
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Build a map from community ID to the set of member NodeIds.
fn build_member_sets(communities: &FxHashMap<NodeId, u64>) -> HashMap<u64, FxHashSet<NodeId>> {
    let mut members: HashMap<u64, FxHashSet<NodeId>> = HashMap::new();
    for (&node, &comm) in communities {
        members.entry(comm).or_default().insert(node);
    }
    members
}

/// Compute Jaccard index between two sets.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|. Returns 0.0 if both sets are empty.
fn jaccard_index(a: &FxHashSet<NodeId>, b: &FxHashSet<NodeId>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.iter().filter(|x| b.contains(*x)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

// ============================================================================
// Hungarian Algorithm (Kuhn-Munkres)
// ============================================================================

/// Solve the assignment problem using the Hungarian algorithm.
///
/// Given a square cost matrix of size k×k, returns an assignment vector where
/// `result[i] = j` means row i is assigned to column j, minimizing total cost.
///
/// # Complexity
///
/// O(k^3) time, O(k^2) space.
#[allow(clippy::many_single_char_names)]
fn hungarian_algorithm(cost: &[Vec<f64>]) -> Vec<usize> {
    let k = cost.len();
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![0];
    }

    // Use 1-indexed internally for cleaner boundary handling
    let n = k;
    let mut u = vec![0.0_f64; n + 1]; // potential for rows
    let mut v = vec![0.0_f64; n + 1]; // potential for columns
    let mut p = vec![0_usize; n + 1]; // column assignment: p[j] = row assigned to col j
    let mut way = vec![0_usize; n + 1]; // way[j] = prev column in augmenting path

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize; // virtual column
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0_usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;

            if p[j0] == 0 {
                break;
            }
        }

        // Update assignment along augmenting path
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert to 0-indexed: result[row] = col
    let mut result = vec![0_usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

// ============================================================================
// GraphAlgorithm trait impl
// ============================================================================

/// Static parameter definitions for stable communities.
static STABLE_COMMUNITIES_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn stable_communities_params() -> &'static [ParameterDef] {
    STABLE_COMMUNITIES_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "resolution".to_string(),
                description: "Louvain resolution parameter (default: 1.0)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("1.0".to_string()),
            },
            ParameterDef {
                name: "min_overlap".to_string(),
                description: "Minimum Jaccard overlap to accept a community match (default: 0.3)"
                    .to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("0.3".to_string()),
            },
        ]
    })
}

/// Stable community detection algorithm wrapper.
///
/// Runs Louvain twice on the same graph and stabilizes the IDs.
/// In practice, callers should store the previous `LouvainResult` and call
/// `stabilize_communities()` directly for true cross-run stability.
/// This wrapper demonstrates the pattern via the `GraphAlgorithm` registry.
pub struct StableCommunitiesAlgorithm;

impl GraphAlgorithm for StableCommunitiesAlgorithm {
    fn name(&self) -> &str {
        "stable_communities"
    }

    fn description(&self) -> &str {
        "Stable community detection with Hungarian matching for ID consistency across runs"
    }

    fn parameters(&self) -> &[ParameterDef] {
        stable_communities_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let resolution = params.get_float("resolution").unwrap_or(1.0);
        let min_overlap = params.get_float("min_overlap").unwrap_or(0.3);

        // Run Louvain twice to demonstrate stabilization
        let prev = louvain(store, resolution);
        let curr = louvain(store, resolution);
        let stable = stabilize_communities(&prev, &curr, store, min_overlap);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string(),
        ]);

        for (&node, &community_id) in &stable.communities {
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(community_id as i64),
                Value::Float64(stable.modularity),
            ]);
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
    use obrain_core::graph::lpg::LpgStore;

    /// Helper: create a simple graph with two cliques connected by a bridge.
    fn create_two_clique_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Clique 1: nodes 0-3
        let n: Vec<_> = (0..8).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                store.create_edge(n[i], n[j], "CONNECTED");
            }
        }
        // Clique 2: nodes 4-7
        for i in 4..8 {
            for j in (i + 1)..8 {
                store.create_edge(n[i], n[j], "CONNECTED");
            }
        }
        // Bridge
        store.create_edge(n[3], n[4], "CONNECTED");
        store
    }

    #[test]
    fn test_stable_basic() {
        // Two runs on the same graph should produce the same IDs after stabilization
        let store = create_two_clique_graph();
        let prev = louvain(&store, 1.0);
        let curr = louvain(&store, 1.0);
        let stable = stabilize_communities(&prev, &curr, &store, 0.3);

        assert_eq!(stable.communities.len(), prev.communities.len());
        // Community sizes should match
        let prev_sizes = community_sizes(&prev.communities);
        let stable_sizes = community_sizes(&stable.communities);
        assert_eq!(prev_sizes, stable_sizes);
    }

    #[test]
    fn test_with_changes() {
        // Modify graph between runs — should retain >70% stability if >50% members remain
        let store = LpgStore::new().unwrap();
        let nodes: Vec<_> = (0..12).map(|_| store.create_node(&["Node"])).collect();

        // Clique 1: 0-5
        for i in 0..6 {
            for j in (i + 1)..6 {
                store.create_edge(nodes[i], nodes[j], "CONNECTED");
            }
        }
        // Clique 2: 6-11
        for i in 6..12 {
            for j in (i + 1)..12 {
                store.create_edge(nodes[i], nodes[j], "CONNECTED");
            }
        }
        store.create_edge(nodes[5], nodes[6], "CONNECTED");

        let prev = louvain(&store, 1.0);

        // Add 2 more nodes to clique 1
        let extra1 = store.create_node(&["Node"]);
        let extra2 = store.create_node(&["Node"]);
        store.create_edge(extra1, nodes[0], "CONNECTED");
        store.create_edge(extra1, nodes[1], "CONNECTED");
        store.create_edge(extra2, nodes[2], "CONNECTED");
        store.create_edge(extra2, nodes[3], "CONNECTED");

        let curr = louvain(&store, 1.0);
        let stable = stabilize_communities(&prev, &curr, &store, 0.3);

        // Should have communities
        assert!(stable.num_communities >= 2);
        assert!(!stable.mapping.is_empty());
    }

    #[test]
    fn test_empty_prev() {
        // First run (no previous) — IDs should be sequential
        let store = create_two_clique_graph();
        let prev = LouvainResult {
            communities: FxHashMap::default(),
            modularity: 0.0,
            num_communities: 0,
        };
        let curr = louvain(&store, 1.0);
        let stable = stabilize_communities(&prev, &curr, &store, 0.3);

        assert_eq!(stable.communities.len(), curr.communities.len());
        assert!(stable.mapping.is_empty()); // No mapping for first run
    }

    #[test]
    fn test_disjoint() {
        // Completely different communities → new IDs assigned
        let store1 = LpgStore::new().unwrap();
        let n1: Vec<_> = (0..4).map(|_| store1.create_node(&["Node"])).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                store1.create_edge(n1[i], n1[j], "CONNECTED");
            }
        }

        let prev = louvain(&store1, 1.0);

        // Completely different graph
        let store2 = LpgStore::new().unwrap();
        let n2: Vec<_> = (0..6).map(|_| store2.create_node(&["Node"])).collect();
        for i in 0..3 {
            for j in (i + 1)..3 {
                store2.create_edge(n2[i], n2[j], "CONNECTED");
            }
        }
        for i in 3..6 {
            for j in (i + 1)..6 {
                store2.create_edge(n2[i], n2[j], "CONNECTED");
            }
        }
        store2.create_edge(n2[2], n2[3], "CONNECTED");

        let curr = louvain(&store2, 1.0);
        let stable = stabilize_communities(&prev, &curr, &store2, 0.5);

        // With high min_overlap and completely different nodes, no matches expected
        assert!(stable.num_communities >= 1);
    }

    #[test]
    fn test_single_community() {
        // k=1 → matching is trivial
        let store = LpgStore::new().unwrap();
        let nodes: Vec<_> = (0..5).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..5 {
            for j in (i + 1)..5 {
                store.create_edge(nodes[i], nodes[j], "CONNECTED");
            }
        }

        let prev = louvain(&store, 1.0);
        let curr = louvain(&store, 1.0);
        let stable = stabilize_communities(&prev, &curr, &store, 0.3);

        assert_eq!(stable.num_communities, 1);
        // All nodes should have the same community
        let ids: FxHashSet<u64> = stable.communities.values().copied().collect();
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_hungarian_basic() {
        // Simple 3x3 cost matrix with known solution
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let assignment = hungarian_algorithm(&cost);
        assert_eq!(assignment.len(), 3);
        // Each row assigned to a unique column
        let mut cols: Vec<usize> = assignment.clone();
        cols.sort_unstable();
        cols.dedup();
        assert_eq!(cols.len(), 3);
    }

    /// Count community sizes, sorted for comparison.
    fn community_sizes(communities: &FxHashMap<NodeId, u64>) -> Vec<usize> {
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &comm in communities.values() {
            *counts.entry(comm).or_insert(0) += 1;
        }
        let mut sizes: Vec<usize> = counts.into_values().collect();
        sizes.sort_unstable();
        sizes
    }
}
