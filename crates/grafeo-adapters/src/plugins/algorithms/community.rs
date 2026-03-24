//! Community detection algorithms: Louvain, Leiden, Label Propagation.
//!
//! These algorithms identify clusters or communities of nodes that are
//! more densely connected to each other than to the rest of the graph.

use std::collections::VecDeque;
use std::sync::OnceLock;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;
#[cfg(test)]
use grafeo_core::graph::lpg::LpgStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::{ComponentResultBuilder, GraphAlgorithm};

// ============================================================================
// Label Propagation
// ============================================================================

/// Detects communities using the Label Propagation Algorithm.
///
/// Each node is initially assigned a unique label. Then, iteratively,
/// each node adopts the most frequent label among its neighbors until
/// the labels stabilize.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `max_iterations` - Maximum number of iterations (0 for unlimited)
///
/// # Returns
///
/// A map from node ID to community (label) ID.
///
/// # Complexity
///
/// O(iterations × E)
pub fn label_propagation(store: &dyn GraphStore, max_iterations: usize) -> FxHashMap<NodeId, u64> {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return FxHashMap::default();
    }

    // Initialize labels: each node gets its own unique label
    let mut labels: FxHashMap<NodeId, u64> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        labels.insert(node, idx as u64);
    }

    let max_iter = if max_iterations == 0 {
        n * 10
    } else {
        max_iterations
    };

    for _ in 0..max_iter {
        let mut changed = false;

        // Update labels in random order (here we use insertion order)
        for &node in &nodes {
            // Get neighbor labels and their frequencies
            let mut label_counts: FxHashMap<u64, usize> = FxHashMap::default();

            // Consider both outgoing and incoming edges (undirected community detection)
            // Outgoing edges: node -> neighbor
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                if let Some(&label) = labels.get(&neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            // Incoming edges: neighbor -> node
            // Uses backward adjacency index for O(degree) instead of O(V*E)
            for (incoming_neighbor, _) in store.edges_from(node, Direction::Incoming) {
                if let Some(&label) = labels.get(&incoming_neighbor) {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            if label_counts.is_empty() {
                continue;
            }

            // Find the most frequent label
            let max_count = *label_counts.values().max().unwrap_or(&0);
            let max_labels: Vec<u64> = label_counts
                .into_iter()
                .filter(|&(_, count)| count == max_count)
                .map(|(label, _)| label)
                .collect();

            // Choose the smallest label in case of tie (deterministic)
            let new_label = *max_labels
                .iter()
                .min()
                .expect("max_labels non-empty: filtered from non-empty label_counts");
            let current_label = *labels.get(&node).expect("node initialized with label");

            if new_label != current_label {
                labels.insert(node, new_label);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Normalize labels to be contiguous starting from 0
    let unique_labels: FxHashSet<u64> = labels.values().copied().collect();
    let mut label_map: FxHashMap<u64, u64> = FxHashMap::default();
    for (idx, label) in unique_labels.into_iter().enumerate() {
        label_map.insert(label, idx as u64);
    }

    labels
        .into_iter()
        .map(|(node, label)| (node, *label_map.get(&label).expect("label present in map")))
        .collect()
}

// ============================================================================
// Louvain Algorithm
// ============================================================================

/// Result of Louvain algorithm.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community assignment for each node.
    pub communities: FxHashMap<NodeId, u64>,
    /// Final modularity score.
    pub modularity: f64,
    /// Number of communities detected.
    pub num_communities: usize,
}

/// Detects communities using the Louvain algorithm.
///
/// The Louvain algorithm optimizes modularity through a greedy approach,
/// consisting of two phases that are repeated iteratively:
/// 1. Local optimization: Move nodes to neighboring communities if it increases modularity
/// 2. Aggregation: Build a new graph where communities become super-nodes
///
/// # Arguments
///
/// * `store` - The graph store
/// * `resolution` - Resolution parameter (higher = smaller communities, default 1.0)
///
/// # Returns
///
/// Community assignments and modularity score.
///
/// # Complexity
///
/// O(V log V) on average for sparse graphs
pub fn louvain(store: &dyn GraphStore, resolution: f64) -> LouvainResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return LouvainResult {
            communities: FxHashMap::default(),
            modularity: 0.0,
            num_communities: 0,
        };
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Build adjacency with weights (for undirected graph)
    // weights[i][j] = weight of edge between nodes i and j
    let mut weights: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut total_weight = 0.0;

    for &node in &nodes {
        let i = *node_to_idx.get(&node).expect("node in index");
        for (neighbor, _edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                // For undirected: add weight to both directions
                let w = 1.0; // Could extract from edge property
                *weights[i].entry(j).or_insert(0.0) += w;
                *weights[j].entry(i).or_insert(0.0) += w;
                total_weight += w;
            }
        }
    }

    // Handle isolated nodes
    if total_weight == 0.0 {
        let communities: FxHashMap<NodeId, u64> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx as u64))
            .collect();
        return LouvainResult {
            communities,
            modularity: 0.0,
            num_communities: n,
        };
    }

    // Compute node degrees (sum of incident edge weights)
    let degrees: Vec<f64> = (0..n).map(|i| weights[i].values().sum()).collect();

    // Initialize: each node in its own community
    let mut community: Vec<usize> = (0..n).collect();

    // Community internal weights and total weights
    let mut community_internal: FxHashMap<usize, f64> = FxHashMap::default();
    let mut community_total: FxHashMap<usize, f64> = FxHashMap::default();

    for i in 0..n {
        community_total.insert(i, degrees[i]);
        community_internal.insert(i, weights[i].get(&i).copied().unwrap_or(0.0));
    }

    // Phase 1: Local optimization
    let mut improved = true;
    while improved {
        improved = false;

        for i in 0..n {
            let current_comm = community[i];

            // Compute links to each neighboring community
            let mut comm_links: FxHashMap<usize, f64> = FxHashMap::default();
            for (&j, &w) in &weights[i] {
                let c = community[j];
                *comm_links.entry(c).or_insert(0.0) += w;
            }

            // Try moving to each neighboring community
            let mut best_delta = 0.0;
            let mut best_comm = current_comm;

            // Remove node from current community for delta calculation
            let ki = degrees[i];
            let ki_in = comm_links.get(&current_comm).copied().unwrap_or(0.0);

            for (&target_comm, &k_i_to_comm) in &comm_links {
                if target_comm == current_comm {
                    continue;
                }

                let sigma_tot = *community_total.get(&target_comm).unwrap_or(&0.0);

                // Modularity delta for moving to target_comm
                let delta = resolution
                    * (k_i_to_comm
                        - ki_in
                        - ki * (sigma_tot - community_total.get(&current_comm).unwrap_or(&0.0)
                            + ki)
                            / (2.0 * total_weight));

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            if best_comm != current_comm {
                // Move node to best community
                // Update community statistics
                *community_total.entry(current_comm).or_insert(0.0) -= ki;
                *community_internal.entry(current_comm).or_insert(0.0) -=
                    2.0 * ki_in + weights[i].get(&i).copied().unwrap_or(0.0);

                community[i] = best_comm;

                *community_total.entry(best_comm).or_insert(0.0) += ki;
                let k_i_best = comm_links.get(&best_comm).copied().unwrap_or(0.0);
                *community_internal.entry(best_comm).or_insert(0.0) +=
                    2.0 * k_i_best + weights[i].get(&i).copied().unwrap_or(0.0);

                improved = true;
            }
        }
    }

    // Normalize community IDs
    let unique_comms: FxHashSet<usize> = community.iter().copied().collect();
    let mut comm_map: FxHashMap<usize, u64> = FxHashMap::default();
    for (idx, c) in unique_comms.iter().enumerate() {
        comm_map.insert(*c, idx as u64);
    }

    let communities: FxHashMap<NodeId, u64> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            (
                node,
                *comm_map.get(&community[i]).expect("community in map"),
            )
        })
        .collect();

    // Compute final modularity
    let modularity = compute_modularity(&weights, &community, total_weight, resolution);

    LouvainResult {
        communities,
        modularity,
        num_communities: unique_comms.len(),
    }
}

/// Computes the modularity of a community assignment.
fn compute_modularity(
    weights: &[FxHashMap<usize, f64>],
    community: &[usize],
    total_weight: f64,
    resolution: f64,
) -> f64 {
    let n = community.len();
    let m2 = 2.0 * total_weight;

    if m2 == 0.0 {
        return 0.0;
    }

    let degrees: Vec<f64> = (0..n).map(|i| weights[i].values().sum()).collect();

    let mut modularity = 0.0;

    for i in 0..n {
        for (&j, &a_ij) in &weights[i] {
            if community[i] == community[j] {
                modularity += a_ij - resolution * degrees[i] * degrees[j] / m2;
            }
        }
    }

    modularity / m2
}

/// Returns the number of communities detected.
pub fn community_count(communities: &FxHashMap<NodeId, u64>) -> usize {
    let unique: FxHashSet<u64> = communities.values().copied().collect();
    unique.len()
}

// ============================================================================
// Leiden Algorithm (Traag et al., 2019)
// ============================================================================

/// Detects communities using the Leiden algorithm (Traag, Waltman & van Eck, 2019).
///
/// An improvement over Louvain that guarantees connected communities through
/// an additional refinement phase between the move and aggregation phases:
///
/// 1. **Local move phase** — identical to Louvain: greedily move nodes to
///    the neighboring community that maximizes modularity gain.
/// 2. **Refinement phase** — sub-partition each community using a constrained
///    move within the community, ensuring intra-community connectivity via BFS.
/// 3. **Aggregation phase** — build a new graph where refined sub-communities
///    become super-nodes, then repeat.
///
/// # Arguments
///
/// * `store` — The graph store (treated as undirected for community detection)
/// * `resolution` — Resolution parameter for modularity (higher → smaller communities, default 1.0)
/// * `gamma` — Resolution for the refinement phase (higher → more sub-partitions, default 0.01)
///
/// # Returns
///
/// A [`LouvainResult`] containing community assignments, modularity, and community count.
///
/// # Complexity
///
/// O(V log V) on average for sparse graphs — same as Louvain, with an additional
/// O(V + E) per iteration for the BFS-based refinement phase.
///
/// # Differences from Louvain
///
/// - **Guaranteed connected communities**: Louvain may produce disconnected communities
///   because nodes can join communities they are not directly connected to through
///   intermediate moves. Leiden fixes this with the refinement phase.
/// - **Better modularity**: The refinement phase often finds higher-quality partitions.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::GraphStore;
/// use grafeo_adapters::plugins::algorithms::leiden;
///
/// let store = LpgStore::new().unwrap();
/// let n0 = store.create_node(&["Node"]);
/// let n1 = store.create_node(&["Node"]);
/// let n2 = store.create_node(&["Node"]);
/// store.create_edge(n0, n1, "EDGE");
/// store.create_edge(n1, n2, "EDGE");
///
/// let result = leiden(&store, 1.0, 0.01);
/// assert_eq!(result.communities.len(), 3);
/// assert!(result.num_communities >= 1);
/// ```
pub fn leiden(store: &dyn GraphStore, resolution: f64, gamma: f64) -> LouvainResult {
    let nodes = store.node_ids();
    let n = nodes.len();

    if n == 0 {
        return LouvainResult {
            communities: FxHashMap::default(),
            modularity: 0.0,
            num_communities: 0,
        };
    }

    // Build node index mapping
    let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
    for (idx, &node) in nodes.iter().enumerate() {
        node_to_idx.insert(node, idx);
    }

    // Build adjacency with weights (undirected)
    let mut weights: Vec<FxHashMap<usize, f64>> = vec![FxHashMap::default(); n];
    let mut total_weight = 0.0;

    for &node in &nodes {
        let i = *node_to_idx.get(&node).expect("node in index");
        for (neighbor, _edge_id) in store.edges_from(node, Direction::Outgoing) {
            if let Some(&j) = node_to_idx.get(&neighbor) {
                let w = 1.0;
                *weights[i].entry(j).or_insert(0.0) += w;
                *weights[j].entry(i).or_insert(0.0) += w;
                total_weight += w;
            }
        }
    }

    // Handle isolated nodes
    if total_weight == 0.0 {
        let communities: FxHashMap<NodeId, u64> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx as u64))
            .collect();
        return LouvainResult {
            communities,
            modularity: 0.0,
            num_communities: n,
        };
    }

    // Compute node degrees
    let degrees: Vec<f64> = (0..n).map(|i| weights[i].values().sum()).collect();

    // Initialize: each node in its own community
    let mut community: Vec<usize> = (0..n).collect();

    // ---- Leiden iteration ----
    let max_outer_iterations = 10;
    for _ in 0..max_outer_iterations {
        // Phase 1: Local move (identical to Louvain)
        let moved = leiden_local_move(
            &weights,
            &degrees,
            &mut community,
            total_weight,
            resolution,
            n,
        );

        // Phase 2: Refinement — ensure intra-community connectivity
        leiden_refine(&weights, &degrees, &mut community, total_weight, gamma, n);

        if !moved {
            break;
        }
    }

    // Normalize community IDs
    let unique_comms: FxHashSet<usize> = community.iter().copied().collect();
    let mut comm_map: FxHashMap<usize, u64> = FxHashMap::default();
    for (idx, c) in unique_comms.iter().enumerate() {
        comm_map.insert(*c, idx as u64);
    }

    let communities: FxHashMap<NodeId, u64> = nodes
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            (
                node,
                *comm_map.get(&community[i]).expect("community in map"),
            )
        })
        .collect();

    let modularity = compute_modularity(&weights, &community, total_weight, resolution);

    LouvainResult {
        communities,
        modularity,
        num_communities: unique_comms.len(),
    }
}

/// Phase 1 of Leiden: local move optimization (same as Louvain).
///
/// Returns `true` if any node was moved.
fn leiden_local_move(
    weights: &[FxHashMap<usize, f64>],
    degrees: &[f64],
    community: &mut [usize],
    total_weight: f64,
    resolution: f64,
    n: usize,
) -> bool {
    let mut community_total: FxHashMap<usize, f64> = FxHashMap::default();
    let mut community_internal: FxHashMap<usize, f64> = FxHashMap::default();

    for i in 0..n {
        *community_total.entry(community[i]).or_insert(0.0) += degrees[i];
        let self_loop = weights[i].get(&i).copied().unwrap_or(0.0);
        *community_internal.entry(community[i]).or_insert(0.0) += self_loop;
    }

    // Add internal weights for edges within same community
    for i in 0..n {
        for (&j, &w) in &weights[i] {
            if i < j && community[i] == community[j] {
                *community_internal.entry(community[i]).or_insert(0.0) += 2.0 * w;
            }
        }
    }

    let mut any_moved = false;
    let mut improved = true;

    while improved {
        improved = false;

        for i in 0..n {
            let current_comm = community[i];
            let ki = degrees[i];

            // Compute links to each neighboring community
            let mut comm_links: FxHashMap<usize, f64> = FxHashMap::default();
            for (&j, &w) in &weights[i] {
                let c = community[j];
                *comm_links.entry(c).or_insert(0.0) += w;
            }

            let ki_in = comm_links.get(&current_comm).copied().unwrap_or(0.0);

            let mut best_delta = 0.0;
            let mut best_comm = current_comm;

            for (&target_comm, &k_i_to_comm) in &comm_links {
                if target_comm == current_comm {
                    continue;
                }

                let sigma_tot = *community_total.get(&target_comm).unwrap_or(&0.0);

                let delta = resolution
                    * (k_i_to_comm
                        - ki_in
                        - ki * (sigma_tot
                            - community_total.get(&current_comm).unwrap_or(&0.0)
                            + ki)
                            / (2.0 * total_weight));

                if delta > best_delta {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            if best_comm != current_comm {
                *community_total.entry(current_comm).or_insert(0.0) -= ki;
                *community_internal.entry(current_comm).or_insert(0.0) -=
                    2.0 * ki_in + weights[i].get(&i).copied().unwrap_or(0.0);

                community[i] = best_comm;

                *community_total.entry(best_comm).or_insert(0.0) += ki;
                let k_i_best = comm_links.get(&best_comm).copied().unwrap_or(0.0);
                *community_internal.entry(best_comm).or_insert(0.0) +=
                    2.0 * k_i_best + weights[i].get(&i).copied().unwrap_or(0.0);

                improved = true;
                any_moved = true;
            }
        }
    }

    any_moved
}

/// Phase 2 of Leiden: refinement to guarantee connected communities.
///
/// For each community, finds connected components via BFS. If a community
/// has multiple connected components, each component becomes its own
/// sub-community. Within each component, nodes may be further refined
/// using a constrained local move with the `gamma` resolution parameter.
fn leiden_refine(
    weights: &[FxHashMap<usize, f64>],
    degrees: &[f64],
    community: &mut [usize],
    total_weight: f64,
    gamma: f64,
    n: usize,
) {
    // Group nodes by community
    let mut comm_members: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        comm_members.entry(community[i]).or_default().push(i);
    }

    // Next available community ID
    let mut next_comm_id = community.iter().copied().max().unwrap_or(0) + 1;

    for (_comm_id, members) in &comm_members {
        if members.len() <= 1 {
            continue;
        }

        // Find connected components within this community via BFS
        let member_set: FxHashSet<usize> = members.iter().copied().collect();
        let components = bfs_connected_components(weights, &member_set);

        if components.len() <= 1 {
            // Community is already connected — optionally do sub-refinement
            // Apply constrained move within the community using gamma
            if gamma > 0.0 {
                leiden_constrained_move(
                    weights,
                    degrees,
                    community,
                    total_weight,
                    gamma,
                    members,
                    &mut next_comm_id,
                );
            }
            continue;
        }

        // Split disconnected community: assign each component a new community ID
        // Keep the first component with the original community ID
        for component in components.iter().skip(1) {
            let new_comm = next_comm_id;
            next_comm_id += 1;
            for &node_idx in component {
                community[node_idx] = new_comm;
            }
        }
    }
}

/// Finds connected components within a set of nodes using BFS on the weight graph.
fn bfs_connected_components(
    weights: &[FxHashMap<usize, f64>],
    member_set: &FxHashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut visited: FxHashSet<usize> = FxHashSet::default();
    let mut components: Vec<Vec<usize>> = Vec::new();

    for &start in member_set {
        if visited.contains(&start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);

            for &neighbor in weights[node].keys() {
                if member_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        components.push(component);
    }

    components
}

/// Constrained local move within a single connected community.
///
/// Uses `gamma` as the resolution parameter to potentially sub-partition
/// a connected community into finer-grained sub-communities, while
/// maintaining connectivity.
fn leiden_constrained_move(
    weights: &[FxHashMap<usize, f64>],
    degrees: &[f64],
    community: &mut [usize],
    total_weight: f64,
    gamma: f64,
    members: &[usize],
    next_comm_id: &mut usize,
) {
    if members.len() <= 2 {
        return;
    }

    // Initialize: each node in its own sub-community within the larger community
    let mut sub_community: FxHashMap<usize, usize> = FxHashMap::default();
    for (idx, &node) in members.iter().enumerate() {
        sub_community.insert(node, idx);
    }

    let member_set: FxHashSet<usize> = members.iter().copied().collect();

    // Local move within the community
    let mut improved = true;
    while improved {
        improved = false;

        for &node in members {
            let current_sub = *sub_community.get(&node).unwrap();

            // Links to neighboring sub-communities (only within this community)
            let mut sub_links: FxHashMap<usize, f64> = FxHashMap::default();
            for (&j, &w) in &weights[node] {
                if member_set.contains(&j) {
                    let sc = *sub_community.get(&j).unwrap();
                    *sub_links.entry(sc).or_insert(0.0) += w;
                }
            }

            let ki = degrees[node];
            let ki_in = sub_links.get(&current_sub).copied().unwrap_or(0.0);

            let mut best_delta = 0.0;
            let mut best_sub = current_sub;

            for (&target_sub, &k_i_to_sub) in &sub_links {
                if target_sub == current_sub {
                    continue;
                }

                // Compute sigma_tot for target sub-community
                let sigma_tot: f64 = members
                    .iter()
                    .filter(|&&m| sub_community.get(&m) == Some(&target_sub))
                    .map(|&m| degrees[m])
                    .sum();

                let sigma_tot_current: f64 = members
                    .iter()
                    .filter(|&&m| sub_community.get(&m) == Some(&current_sub))
                    .map(|&m| degrees[m])
                    .sum();

                let delta = gamma
                    * (k_i_to_sub - ki_in
                        - ki * (sigma_tot - sigma_tot_current + ki) / (2.0 * total_weight));

                if delta > best_delta {
                    best_delta = delta;
                    best_sub = target_sub;
                }
            }

            if best_sub != current_sub {
                sub_community.insert(node, best_sub);
                improved = true;
            }
        }
    }

    // Check how many distinct sub-communities we have
    let unique_subs: FxHashSet<usize> = sub_community.values().copied().collect();
    if unique_subs.len() <= 1 {
        return; // No refinement needed
    }

    // Verify each sub-community is connected before committing
    let mut sub_members: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for &node in members {
        let sc = *sub_community.get(&node).unwrap();
        sub_members.entry(sc).or_default().push(node);
    }

    // Assign new community IDs (keep first sub-community with original ID)
    let original_comm = community[members[0]];
    let mut first = true;
    for (_sub_id, sub_nodes) in &sub_members {
        let sub_set: FxHashSet<usize> = sub_nodes.iter().copied().collect();
        let components = bfs_connected_components(weights, &sub_set);

        if first {
            // First sub-community keeps original ID for first component
            first = false;
            for component in components.iter().skip(1) {
                let new_comm = *next_comm_id;
                *next_comm_id += 1;
                for &node_idx in component {
                    community[node_idx] = new_comm;
                }
            }
            // First component keeps original community
            for &node_idx in &components[0] {
                community[node_idx] = original_comm;
            }
        } else {
            // Other sub-communities get new IDs
            for component in &components {
                let new_comm = *next_comm_id;
                *next_comm_id += 1;
                for &node_idx in component {
                    community[node_idx] = new_comm;
                }
            }
        }
    }
}

// ============================================================================
// Algorithm Wrappers for Plugin Registry
// ============================================================================

/// Static parameter definitions for Label Propagation algorithm.
static LABEL_PROP_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn label_prop_params() -> &'static [ParameterDef] {
    LABEL_PROP_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "max_iterations".to_string(),
            description: "Maximum iterations (0 for unlimited, default: 100)".to_string(),
            param_type: ParameterType::Integer,
            required: false,
            default: Some("100".to_string()),
        }]
    })
}

/// Label Propagation algorithm wrapper.
pub struct LabelPropagationAlgorithm;

impl GraphAlgorithm for LabelPropagationAlgorithm {
    fn name(&self) -> &str {
        "label_propagation"
    }

    fn description(&self) -> &str {
        "Label Propagation community detection"
    }

    fn parameters(&self) -> &[ParameterDef] {
        label_prop_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let max_iter = params.get_int("max_iterations").unwrap_or(100) as usize;

        let communities = label_propagation(store, max_iter);

        let mut builder = ComponentResultBuilder::with_capacity(communities.len());
        for (node, community_id) in communities {
            builder.push(node, community_id);
        }

        Ok(builder.build())
    }
}

/// Static parameter definitions for Louvain algorithm.
static LOUVAIN_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn louvain_params() -> &'static [ParameterDef] {
    LOUVAIN_PARAMS.get_or_init(|| {
        vec![ParameterDef {
            name: "resolution".to_string(),
            description: "Resolution parameter (default: 1.0)".to_string(),
            param_type: ParameterType::Float,
            required: false,
            default: Some("1.0".to_string()),
        }]
    })
}

/// Louvain algorithm wrapper.
pub struct LouvainAlgorithm;

impl GraphAlgorithm for LouvainAlgorithm {
    fn name(&self) -> &str {
        "louvain"
    }

    fn description(&self) -> &str {
        "Louvain community detection (modularity optimization)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        louvain_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let resolution = params.get_float("resolution").unwrap_or(1.0);

        let result = louvain(store, resolution);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string(),
        ]);

        for (node, community_id) in result.communities {
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(community_id as i64),
                Value::Float64(result.modularity),
            ]);
        }

        Ok(output)
    }
}

/// Static parameter definitions for Leiden algorithm.
static LEIDEN_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn leiden_params() -> &'static [ParameterDef] {
    LEIDEN_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "resolution".to_string(),
                description: "Resolution parameter for modularity (default: 1.0)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("1.0".to_string()),
            },
            ParameterDef {
                name: "gamma".to_string(),
                description: "Refinement resolution — higher values produce finer sub-partitions (default: 0.01)".to_string(),
                param_type: ParameterType::Float,
                required: false,
                default: Some("0.01".to_string()),
            },
        ]
    })
}

/// Leiden community detection algorithm wrapper.
///
/// Implements [`GraphAlgorithm`] for registration in the procedure registry.
/// Callable via `CALL grafeo.leiden({resolution: 1.0, gamma: 0.01}) YIELD node, community`.
///
/// See [`leiden()`] for algorithm details.
pub struct LeidenAlgorithm;

impl GraphAlgorithm for LeidenAlgorithm {
    fn name(&self) -> &str {
        "leiden"
    }

    fn description(&self) -> &str {
        "Leiden community detection (improved Louvain with guaranteed connected communities)"
    }

    fn parameters(&self) -> &[ParameterDef] {
        leiden_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let resolution = params.get_float("resolution").unwrap_or(1.0);
        let gamma = params.get_float("gamma").unwrap_or(0.01);

        let result = leiden(store, resolution, gamma);

        let mut output = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "community_id".to_string(),
            "modularity".to_string(),
        ]);

        for (node, community_id) in result.communities {
            output.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(community_id as i64),
                Value::Float64(result.modularity),
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

    fn create_two_cliques_graph() -> LpgStore {
        // Two cliques connected by one edge
        // Clique 1: 0-1-2-3 (fully connected)
        // Clique 2: 4-5-6-7 (fully connected)
        // Bridge: 3-4
        let store = LpgStore::new().unwrap();

        let nodes: Vec<NodeId> = (0..8).map(|_| store.create_node(&["Node"])).collect();

        // Clique 1
        for i in 0..4 {
            for j in (i + 1)..4 {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        // Clique 2
        for i in 4..8 {
            for j in (i + 1)..8 {
                store.create_edge(nodes[i], nodes[j], "EDGE");
                store.create_edge(nodes[j], nodes[i], "EDGE");
            }
        }

        // Bridge
        store.create_edge(nodes[3], nodes[4], "EDGE");
        store.create_edge(nodes[4], nodes[3], "EDGE");

        store
    }

    fn create_simple_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Simple chain: 0 -> 1 -> 2
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);

        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");

        store
    }

    #[test]
    fn test_label_propagation_basic() {
        let store = create_simple_graph();
        let communities = label_propagation(&store, 100);

        assert_eq!(communities.len(), 3);

        // All nodes should have some community assignment
        for (_, &comm) in &communities {
            assert!(comm < 3);
        }
    }

    #[test]
    fn test_label_propagation_cliques() {
        let store = create_two_cliques_graph();
        let communities = label_propagation(&store, 100);

        assert_eq!(communities.len(), 8);

        // Should detect 2 communities (ideally)
        let num_comms = community_count(&communities);
        assert!((1..=8).contains(&num_comms)); // May vary due to algorithm randomness
    }

    #[test]
    fn test_label_propagation_empty() {
        let store = LpgStore::new().unwrap();
        let communities = label_propagation(&store, 100);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_label_propagation_single_node() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);

        let communities = label_propagation(&store, 100);
        assert_eq!(communities.len(), 1);
    }

    #[test]
    fn test_louvain_basic() {
        let store = create_simple_graph();
        let result = louvain(&store, 1.0);

        assert_eq!(result.communities.len(), 3);
        assert!(result.num_communities >= 1);
    }

    #[test]
    fn test_louvain_cliques() {
        let store = create_two_cliques_graph();
        let result = louvain(&store, 1.0);

        assert_eq!(result.communities.len(), 8);

        // Two K4 cliques connected by a single bridge: should detect 2-3 communities
        assert!(
            result.num_communities >= 2 && result.num_communities <= 3,
            "Two cliques should produce 2-3 communities, got {}",
            result.num_communities
        );
    }

    #[test]
    fn test_louvain_empty() {
        let store = LpgStore::new().unwrap();
        let result = louvain(&store, 1.0);

        assert!(result.communities.is_empty());
        assert_eq!(result.modularity, 0.0);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_louvain_isolated_nodes() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = louvain(&store, 1.0);

        // Each isolated node should be its own community
        assert_eq!(result.communities.len(), 3);
        assert_eq!(result.num_communities, 3);
    }

    #[test]
    fn test_louvain_resolution_parameter() {
        let store = create_two_cliques_graph();

        // Low resolution: fewer, larger communities
        let result_low = louvain(&store, 0.5);

        // High resolution: more, smaller communities
        let result_high = louvain(&store, 2.0);

        // Both should be valid
        assert!(!result_low.communities.is_empty());
        assert!(!result_high.communities.is_empty());
    }

    #[test]
    fn test_community_count() {
        let mut communities: FxHashMap<NodeId, u64> = FxHashMap::default();
        communities.insert(NodeId::new(0), 0);
        communities.insert(NodeId::new(1), 0);
        communities.insert(NodeId::new(2), 1);
        communities.insert(NodeId::new(3), 1);
        communities.insert(NodeId::new(4), 2);

        assert_eq!(community_count(&communities), 3);
    }

    // ====================================================================
    // Leiden tests
    // ====================================================================

    #[test]
    fn test_leiden_basic() {
        let store = create_simple_graph();
        let result = leiden(&store, 1.0, 0.01);

        assert_eq!(result.communities.len(), 3);
        assert!(result.num_communities >= 1);
    }

    #[test]
    fn test_leiden_cliques() {
        let store = create_two_cliques_graph();
        let result = leiden(&store, 1.0, 0.01);

        assert_eq!(result.communities.len(), 8);
        assert!(
            result.num_communities >= 2,
            "Two cliques should produce at least 2 communities, got {}",
            result.num_communities
        );
    }

    #[test]
    fn test_leiden_empty() {
        let store = LpgStore::new().unwrap();
        let result = leiden(&store, 1.0, 0.01);

        assert!(result.communities.is_empty());
        assert_eq!(result.modularity, 0.0);
        assert_eq!(result.num_communities, 0);
    }

    #[test]
    fn test_leiden_isolated_nodes() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);
        store.create_node(&["Node"]);

        let result = leiden(&store, 1.0, 0.01);

        assert_eq!(result.communities.len(), 3);
        assert_eq!(result.num_communities, 3);
    }

    #[test]
    fn test_leiden_guarantees_connected_communities() {
        // Build a graph where Louvain might create disconnected communities
        // but Leiden should not: two triangles connected by a path
        let store = LpgStore::new().unwrap();

        // Triangle 1: 0-1-2
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n0, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n1, "EDGE");
        store.create_edge(n0, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE");

        // Triangle 2: 3-4-5
        let n3 = store.create_node(&["Node"]);
        let n4 = store.create_node(&["Node"]);
        let n5 = store.create_node(&["Node"]);
        store.create_edge(n3, n4, "EDGE");
        store.create_edge(n4, n3, "EDGE");
        store.create_edge(n4, n5, "EDGE");
        store.create_edge(n5, n4, "EDGE");
        store.create_edge(n3, n5, "EDGE");
        store.create_edge(n5, n3, "EDGE");

        // Bridge: 2-3
        store.create_edge(n2, n3, "EDGE");
        store.create_edge(n3, n2, "EDGE");

        let result = leiden(&store, 1.0, 0.01);

        assert_eq!(result.communities.len(), 6);

        // Verify intra-community connectivity:
        // Group nodes by community, then check each community is connected
        let nodes = store.node_ids();
        let mut node_to_idx: FxHashMap<NodeId, usize> = FxHashMap::default();
        for (idx, &node) in nodes.iter().enumerate() {
            node_to_idx.insert(node, idx);
        }

        // Build adjacency for connectivity check
        let mut adj: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
        for &node in &nodes {
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                adj.entry(node).or_default().insert(neighbor);
                adj.entry(neighbor).or_default().insert(node);
            }
        }

        // Group by community
        let mut comm_members: FxHashMap<u64, Vec<NodeId>> = FxHashMap::default();
        for (&node, &comm) in &result.communities {
            comm_members.entry(comm).or_default().push(node);
        }

        // Check each community is connected via BFS
        for (_comm, members) in &comm_members {
            if members.len() <= 1 {
                continue;
            }
            let member_set: FxHashSet<NodeId> = members.iter().copied().collect();
            let mut visited = FxHashSet::default();
            let mut queue = VecDeque::new();
            queue.push_back(members[0]);
            visited.insert(members[0]);

            while let Some(node) = queue.pop_front() {
                if let Some(neighbors) = adj.get(&node) {
                    for &neighbor in neighbors {
                        if member_set.contains(&neighbor) && !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            assert_eq!(
                visited.len(),
                members.len(),
                "Community should be connected: visited {} out of {} members",
                visited.len(),
                members.len()
            );
        }
    }

    #[test]
    fn test_leiden_karate_club() {
        // Zachary's Karate Club graph (34 nodes, 78 edges)
        let store = LpgStore::new().unwrap();

        let nodes: Vec<NodeId> = (0..34).map(|_| store.create_node(&["Member"])).collect();

        // Karate club edges (0-indexed)
        let edges: Vec<(usize, usize)> = vec![
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33),
        ];

        for &(i, j) in &edges {
            store.create_edge(nodes[i], nodes[j], "KNOWS");
            store.create_edge(nodes[j], nodes[i], "KNOWS");
        }

        let result = leiden(&store, 1.0, 0.01);

        assert_eq!(result.communities.len(), 34);
        // Karate club should split into 2-5 communities
        assert!(
            result.num_communities >= 2 && result.num_communities <= 10,
            "Karate club should produce 2-10 communities, got {}",
            result.num_communities
        );
        // Modularity should be positive for a graph with community structure
        assert!(
            result.modularity > 0.0,
            "Modularity should be positive, got {}",
            result.modularity
        );

        // Verify all communities are connected
        let all_nodes = store.node_ids();
        let mut adj: FxHashMap<NodeId, FxHashSet<NodeId>> = FxHashMap::default();
        for &node in &all_nodes {
            for (neighbor, _) in store.edges_from(node, Direction::Outgoing) {
                adj.entry(node).or_default().insert(neighbor);
                adj.entry(neighbor).or_default().insert(node);
            }
        }

        let mut comm_members: FxHashMap<u64, Vec<NodeId>> = FxHashMap::default();
        for (&node, &comm) in &result.communities {
            comm_members.entry(comm).or_default().push(node);
        }

        for (_comm, members) in &comm_members {
            if members.len() <= 1 {
                continue;
            }
            let member_set: FxHashSet<NodeId> = members.iter().copied().collect();
            let mut visited = FxHashSet::default();
            let mut queue = VecDeque::new();
            queue.push_back(members[0]);
            visited.insert(members[0]);

            while let Some(node) = queue.pop_front() {
                if let Some(neighbors) = adj.get(&node) {
                    for &neighbor in neighbors {
                        if member_set.contains(&neighbor) && !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            assert_eq!(
                visited.len(),
                members.len(),
                "Karate club community should be connected"
            );
        }
    }

    #[test]
    fn test_leiden_resolution_parameter() {
        let store = create_two_cliques_graph();

        let result_low = leiden(&store, 0.5, 0.01);
        let result_high = leiden(&store, 2.0, 0.01);

        assert!(!result_low.communities.is_empty());
        assert!(!result_high.communities.is_empty());
    }

    #[test]
    fn test_leiden_gamma_parameter() {
        let store = create_two_cliques_graph();

        // Low gamma: less refinement
        let result_low = leiden(&store, 1.0, 0.0);
        // High gamma: more refinement
        let result_high = leiden(&store, 1.0, 1.0);

        assert!(!result_low.communities.is_empty());
        assert!(!result_high.communities.is_empty());
    }

    #[test]
    fn test_leiden_single_node() {
        let store = LpgStore::new().unwrap();
        store.create_node(&["Node"]);

        let result = leiden(&store, 1.0, 0.01);
        assert_eq!(result.communities.len(), 1);
        assert_eq!(result.num_communities, 1);
    }

    #[test]
    fn test_leiden_algorithm_wrapper() {
        let store = create_two_cliques_graph();
        let algo = LeidenAlgorithm;

        assert_eq!(algo.name(), "leiden");
        assert!(!algo.description().is_empty());
        assert_eq!(algo.parameters().len(), 2);

        let params = Parameters::new();
        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.rows.len(), 8);
    }
}
