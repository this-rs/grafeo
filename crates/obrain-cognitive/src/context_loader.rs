//! # HMR Context Loader
//!
//! Community-aware spreading activation with accept boundaries.
//! Modeled after Hot Module Replacement (HMR) from frontend bundlers:
//!
//! 1. **Entry point** -- seed node(s)
//! 2. **First circle** -- direct imports/neighbors
//! 3. **Accept boundary** -- stop at community borders (Louvain cohesion > threshold)
//! 4. **Cross-community** -- conditional propagation via high-energy synapses
//! 5. **Budget truncation** -- respect `context_budget_tokens`

use std::collections::{HashMap, HashSet, VecDeque};

use obrain_common::types::NodeId;

use crate::kernel_params::CognitiveKernelConfig;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of context loading: activated nodes with scores and metadata.
#[derive(Debug, Clone)]
pub struct ContextBundle {
    /// Activated nodes sorted by score (highest first).
    pub nodes: Vec<ActivatedNode>,
    /// Total token cost estimate.
    pub total_tokens: u32,
    /// Communities touched.
    pub communities_touched: HashSet<u64>,
    /// Max depth reached.
    pub max_depth: u32,
    /// Number of boundary stops.
    pub boundary_stops: u32,
}

/// A node activated during context loading.
#[derive(Debug, Clone)]
pub struct ActivatedNode {
    /// The activated node's ID.
    pub node_id: NodeId,
    /// Activation score (higher = more relevant).
    pub score: f64,
    /// BFS depth at which this node was reached.
    pub depth: u32,
    /// Community membership, if known.
    pub community_id: Option<u64>,
    /// How this node was activated.
    pub source: ActivationSource,
}

/// How a node was activated during context loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationSource {
    /// Seed node (initial entry point).
    Seed,
    /// Direct neighbor within the same community.
    DirectNeighbor,
    /// Propagated via synapse within a community.
    SynapsePropagate,
    /// Crossed a community boundary (weak cohesion).
    CrossCommunity,
}

// ---------------------------------------------------------------------------
// Provider traits
// ---------------------------------------------------------------------------

/// Provides community membership and cohesion scores for nodes.
pub trait CommunityProvider: Send + Sync {
    /// Returns the community ID for a node.
    fn community_of(&self, node_id: NodeId) -> Option<u64>;
    /// Returns the cohesion score for a community (0.0 .. 1.0).
    fn community_cohesion(&self, community_id: u64) -> f64;
}

/// Provides weighted neighbors for spreading activation.
pub trait NeighborProvider: Send + Sync {
    /// Returns neighbors of a node with their synapse weights.
    fn neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f64)>;
}

/// Estimates the token cost of including a node in context.
pub trait TokenEstimator: Send + Sync {
    /// Returns the estimated token count for the given node.
    fn estimate_tokens(&self, node_id: NodeId) -> u32;
}

/// Default estimator: fixed cost per node.
pub struct FixedTokenEstimator(pub u32);

impl TokenEstimator for FixedTokenEstimator {
    fn estimate_tokens(&self, _: NodeId) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Loads context using HMR-style accept boundaries.
///
/// Starting from seed nodes, performs BFS spreading activation that respects
/// community boundaries: propagation stops at community borders with high
/// cohesion (accept boundary) but continues through weak boundaries.
///
/// The result is budget-truncated to `config.context_budget_tokens`.
pub fn load_context(
    seeds: &[NodeId],
    config: &CognitiveKernelConfig,
    neighbors: &dyn NeighborProvider,
    communities: &dyn CommunityProvider,
    tokens: &dyn TokenEstimator,
) -> ContextBundle {
    let mut result_map: HashMap<NodeId, ActivatedNode> = HashMap::new();
    let mut total_tokens: u32 = 0;
    let mut communities_touched: HashSet<u64> = HashSet::new();
    let mut max_depth: u32 = 0;
    let mut boundary_stops: u32 = 0;

    // BFS queue: (node_id, score, depth, source)
    let mut queue: VecDeque<(NodeId, f64, u32, ActivationSource)> = VecDeque::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    // Seed sources
    for &seed in seeds {
        if visited.insert(seed) {
            queue.push_back((seed, 1.0, 0, ActivationSource::Seed));
        }
    }

    let budget = config.context_budget_tokens;

    while let Some((node_id, score, depth, source)) = queue.pop_front() {
        // Budget check
        if total_tokens >= budget {
            break;
        }

        // Depth check
        if depth > config.max_hops {
            continue;
        }

        // Energy cutoff (seeds always pass)
        if source != ActivationSource::Seed && score < config.min_propagated_energy {
            continue;
        }

        // Add node to results
        let node_community = communities.community_of(node_id);
        if let Some(cid) = node_community {
            communities_touched.insert(cid);
        }

        let token_cost = tokens.estimate_tokens(node_id);
        if total_tokens.saturating_add(token_cost) > budget && source != ActivationSource::Seed {
            // Would exceed budget; skip this node but continue draining queue
            // in case cheaper nodes remain.
            continue;
        }

        total_tokens = total_tokens.saturating_add(token_cost);
        if depth > max_depth {
            max_depth = depth;
        }

        result_map.insert(
            node_id,
            ActivatedNode {
                node_id,
                score,
                depth,
                community_id: node_community,
                source,
            },
        );

        // Propagate to neighbors
        for (neighbor, weight) in neighbors.neighbors(node_id) {
            if visited.contains(&neighbor) {
                continue;
            }

            let propagated_score = score * weight * (1.0 - config.propagation_decay);

            if propagated_score < config.min_propagated_energy {
                continue;
            }

            let neighbor_community = communities.community_of(neighbor);
            let same_community = match (node_community, neighbor_community) {
                (Some(a), Some(b)) => a == b,
                _ => true, // If either has no community, treat as same
            };

            if same_community {
                // Same community: propagate normally
                let neighbor_source = if depth == 0 {
                    ActivationSource::DirectNeighbor
                } else {
                    ActivationSource::SynapsePropagate
                };
                visited.insert(neighbor);
                queue.push_back((neighbor, propagated_score, depth + 1, neighbor_source));
            } else {
                // Different community: check accept boundary
                if let Some(src_cid) = node_community {
                    let cohesion = communities.community_cohesion(src_cid);
                    if cohesion > config.community_cohesion_threshold {
                        // High cohesion = strong boundary: STOP
                        boundary_stops += 1;
                        continue;
                    }
                }
                // Weak boundary: allow cross-community propagation
                visited.insert(neighbor);
                queue.push_back((
                    neighbor,
                    propagated_score,
                    depth + 1,
                    ActivationSource::CrossCommunity,
                ));
            }
        }
    }

    // Collect and sort by score descending
    let mut nodes: Vec<ActivatedNode> = result_map.into_values().collect();
    nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    ContextBundle {
        nodes,
        total_tokens,
        communities_touched,
        max_depth,
        boundary_stops,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Mock providers
    // -----------------------------------------------------------------------

    /// Simple graph with configurable neighbors.
    struct MockNeighbors {
        edges: HashMap<NodeId, Vec<(NodeId, f64)>>,
    }

    impl MockNeighbors {
        fn new() -> Self {
            Self {
                edges: HashMap::new(),
            }
        }

        fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f64) {
            self.edges
                .entry(from)
                .or_default()
                .push((to, weight));
            self.edges
                .entry(to)
                .or_default()
                .push((from, weight));
        }
    }

    impl NeighborProvider for MockNeighbors {
        fn neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f64)> {
            self.edges.get(&node_id).cloned().unwrap_or_default()
        }
    }

    /// Community provider with configurable membership and cohesion.
    struct MockCommunities {
        membership: HashMap<NodeId, u64>,
        cohesion: HashMap<u64, f64>,
    }

    impl MockCommunities {
        fn new() -> Self {
            Self {
                membership: HashMap::new(),
                cohesion: HashMap::new(),
            }
        }

        fn set_community(&mut self, node_id: NodeId, community_id: u64) {
            self.membership.insert(node_id, community_id);
        }

        fn set_cohesion(&mut self, community_id: u64, score: f64) {
            self.cohesion.insert(community_id, score);
        }
    }

    impl CommunityProvider for MockCommunities {
        fn community_of(&self, node_id: NodeId) -> Option<u64> {
            self.membership.get(&node_id).copied()
        }

        fn community_cohesion(&self, community_id: u64) -> f64 {
            self.cohesion.get(&community_id).copied().unwrap_or(0.5)
        }
    }

    fn nid(n: u64) -> NodeId {
        NodeId::from(n)
    }

    fn default_config() -> CognitiveKernelConfig {
        CognitiveKernelConfig {
            propagation_decay: 0.3,
            community_cohesion_threshold: 0.7,
            max_hops: 3,
            min_propagated_energy: 0.01,
            context_budget_tokens: 10000,
            ..CognitiveKernelConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn single_community_all_activated() {
        // All nodes in community 1 → no boundary stops
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);
        neighbors.add_edge(nid(2), nid(3), 1.0);
        neighbors.add_edge(nid(3), nid(4), 1.0);

        let mut communities = MockCommunities::new();
        for i in 1..=4 {
            communities.set_community(nid(i), 1);
        }
        communities.set_cohesion(1, 0.9);

        let tokens = FixedTokenEstimator(10);
        let config = default_config();

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        assert_eq!(bundle.nodes.len(), 4);
        assert_eq!(bundle.boundary_stops, 0);
        assert_eq!(bundle.communities_touched.len(), 1);
        assert!(bundle.communities_touched.contains(&1));
    }

    #[test]
    fn two_communities_high_cohesion_stops_at_boundary() {
        // Community 1 (nodes 1,2) with high cohesion → should NOT cross to community 2
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);
        neighbors.add_edge(nid(2), nid(3), 1.0); // boundary edge

        let mut communities = MockCommunities::new();
        communities.set_community(nid(1), 1);
        communities.set_community(nid(2), 1);
        communities.set_community(nid(3), 2);
        communities.set_cohesion(1, 0.9); // high cohesion → accept boundary
        communities.set_cohesion(2, 0.5);

        let tokens = FixedTokenEstimator(10);
        let config = default_config(); // threshold = 0.7

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        let activated_ids: HashSet<NodeId> =
            bundle.nodes.iter().map(|n| n.node_id).collect();
        assert!(activated_ids.contains(&nid(1)));
        assert!(activated_ids.contains(&nid(2)));
        assert!(!activated_ids.contains(&nid(3)), "should not cross high-cohesion boundary");
        assert!(bundle.boundary_stops > 0);
    }

    #[test]
    fn two_communities_low_cohesion_crosses_boundary() {
        // Community 1 (nodes 1,2) with low cohesion → should cross to community 2
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);
        neighbors.add_edge(nid(2), nid(3), 1.0); // boundary edge

        let mut communities = MockCommunities::new();
        communities.set_community(nid(1), 1);
        communities.set_community(nid(2), 1);
        communities.set_community(nid(3), 2);
        communities.set_cohesion(1, 0.5); // low cohesion → weak boundary
        communities.set_cohesion(2, 0.5);

        let tokens = FixedTokenEstimator(10);
        let config = default_config(); // threshold = 0.7

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        let activated_ids: HashSet<NodeId> =
            bundle.nodes.iter().map(|n| n.node_id).collect();
        assert!(activated_ids.contains(&nid(1)));
        assert!(activated_ids.contains(&nid(2)));
        assert!(activated_ids.contains(&nid(3)), "should cross low-cohesion boundary");
        assert_eq!(bundle.boundary_stops, 0);
        assert_eq!(bundle.communities_touched.len(), 2);
    }

    #[test]
    fn budget_truncation() {
        // Budget only allows 2 nodes (20 tokens), but graph has 4
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);
        neighbors.add_edge(nid(2), nid(3), 1.0);
        neighbors.add_edge(nid(3), nid(4), 1.0);

        let mut communities = MockCommunities::new();
        for i in 1..=4 {
            communities.set_community(nid(i), 1);
        }

        let tokens = FixedTokenEstimator(10);
        let mut config = default_config();
        config.context_budget_tokens = 20; // Only 2 nodes fit

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        assert_eq!(bundle.total_tokens, 20);
        assert!(bundle.nodes.len() <= 2);
    }

    #[test]
    fn max_hops_limit() {
        // Chain of 6 nodes, max_hops = 2 → should reach at most 3 nodes (seed + 2 hops)
        let mut neighbors = MockNeighbors::new();
        for i in 1..=5 {
            neighbors.add_edge(nid(i), nid(i + 1), 1.0);
        }

        let mut communities = MockCommunities::new();
        for i in 1..=6 {
            communities.set_community(nid(i), 1);
        }

        let tokens = FixedTokenEstimator(10);
        let mut config = default_config();
        config.max_hops = 2;

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        assert!(bundle.max_depth <= 2);
        assert!(bundle.nodes.len() <= 3);
    }

    #[test]
    fn min_energy_cutoff() {
        // High decay + low weights → energy should die off quickly
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 0.5);
        neighbors.add_edge(nid(2), nid(3), 0.5);
        neighbors.add_edge(nid(3), nid(4), 0.5);

        let mut communities = MockCommunities::new();
        for i in 1..=4 {
            communities.set_community(nid(i), 1);
        }

        let tokens = FixedTokenEstimator(10);
        let mut config = default_config();
        config.propagation_decay = 0.5; // high decay
        config.min_propagated_energy = 0.2; // high cutoff

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        // Score at depth 1: 1.0 * 0.5 * (1.0 - 0.5) = 0.25 (passes 0.2)
        // Score at depth 2: 0.25 * 0.5 * 0.5 = 0.0625 (fails 0.2)
        assert!(bundle.nodes.len() <= 2, "energy should die off, got {} nodes", bundle.nodes.len());
    }

    #[test]
    fn results_sorted_by_score_descending() {
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);
        neighbors.add_edge(nid(1), nid(3), 0.5);

        let mut communities = MockCommunities::new();
        for i in 1..=3 {
            communities.set_community(nid(i), 1);
        }

        let tokens = FixedTokenEstimator(10);
        let config = default_config();

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        for w in bundle.nodes.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results should be sorted descending: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn cross_community_source_tagged() {
        let mut neighbors = MockNeighbors::new();
        neighbors.add_edge(nid(1), nid(2), 1.0);

        let mut communities = MockCommunities::new();
        communities.set_community(nid(1), 1);
        communities.set_community(nid(2), 2);
        communities.set_cohesion(1, 0.3); // low cohesion → crosses

        let tokens = FixedTokenEstimator(10);
        let config = default_config();

        let bundle = load_context(&[nid(1)], &config, &neighbors, &communities, &tokens);

        let node2 = bundle.nodes.iter().find(|n| n.node_id == nid(2));
        assert!(node2.is_some());
        assert_eq!(node2.unwrap().source, ActivationSource::CrossCommunity);
    }
}
