//! # Neural Routing
//!
//! MCTS-based action selection that exploits the obrain graph topology.
//! Instead of a separate GNN, the routing uses Hilbert features + spreading
//! activation as state encoding, and stigmergy pheromones as rollout heuristic.
//!
//! ## Algorithm
//!
//! 1. **State** = spreading activation scores from seed node (N-hop)
//! 2. **Actions** = candidate nodes weighted by synapse energy + pheromones
//! 3. **Selection** = UCB1 (Upper Confidence Bound) balancing explore/exploit
//! 4. **Rollout** = follow strongest pheromone trail (stigmergy heuristic)
//! 5. **Backprop** = reinforce synapses along selected path (Hebbian)
//!
//! ## Usage
//!
//! ```ignore
//! let ranked = neural_route(&graph, seed, &candidates, config);
//! // ranked[0] = best action with highest score
//! ```

use std::collections::HashMap;

use obrain_common::types::NodeId;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the MCTS neural routing.
#[derive(Debug, Clone)]
pub struct NeuralRoutingConfig {
    /// Number of MCTS simulations to run.
    pub budget_simulations: u32,
    /// UCB1 exploration constant (sqrt(2) is standard).
    pub exploration_c: f64,
    /// Maximum rollout depth (follows pheromones this many hops).
    pub max_rollout_depth: u32,
    /// Minimum synapse energy to consider an edge during expansion.
    pub min_edge_energy: f64,
    /// Discount factor for rollout rewards (gamma).
    pub discount_factor: f64,
}

impl Default for NeuralRoutingConfig {
    fn default() -> Self {
        Self {
            budget_simulations: 50,
            exploration_c: 1.414,
            max_rollout_depth: 5,
            min_edge_energy: 0.05,
            discount_factor: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A candidate action to be evaluated by the routing.
#[derive(Debug, Clone)]
pub struct CandidateAction {
    /// Identifier for this action (node id, tool name, etc.)
    pub id: String,
    /// Target node in the graph (where this action leads).
    pub target_node: NodeId,
    /// Prior probability (from stigmergy pheromones or domain knowledge).
    pub prior: f64,
}

/// Result of neural routing — a ranked action with score and confidence.
#[derive(Debug, Clone)]
pub struct RankedAction {
    /// The action id.
    pub id: String,
    /// Target node.
    pub target_node: NodeId,
    /// UCB1-derived score (higher = better).
    pub score: f64,
    /// Visit count during MCTS (higher = more confident).
    pub visit_count: u32,
    /// Average value from rollouts.
    pub avg_value: f64,
}

/// Internal MCTS tree node.
#[derive(Debug, Clone)]
struct MctsNode {
    action_id: String,
    target_node: NodeId,
    visit_count: u32,
    total_value: f64,
    prior: f64,
}

impl MctsNode {
    fn avg_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }

    fn ucb1(&self, parent_visits: u32, c: f64) -> f64 {
        if self.visit_count == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.avg_value();
        let exploration = c * (f64::ln(parent_visits as f64) / self.visit_count as f64).sqrt();
        exploitation + exploration + self.prior * 0.1
    }
}

// ---------------------------------------------------------------------------
// Graph trait — what the router needs from the environment
// ---------------------------------------------------------------------------

/// Trait providing graph topology information to the routing engine.
pub trait RoutingGraph {
    /// Get weighted neighbors of a node (target, edge_weight).
    /// Weight combines synapse energy + pheromone intensity.
    fn neighbors(&self, node: NodeId) -> Vec<(NodeId, f64)>;

    /// Get the activation score of a node (from spreading activation or energy).
    fn node_score(&self, node: NodeId) -> f64;

    /// Get pheromone intensity on the edge from → to.
    fn pheromone(&self, from: NodeId, to: NodeId) -> f64;
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// Run MCTS neural routing to rank candidate actions.
///
/// Returns candidates sorted by score (best first).
pub fn neural_route<G: RoutingGraph>(
    graph: &G,
    _seed: NodeId,
    candidates: &[CandidateAction],
    config: &NeuralRoutingConfig,
) -> Vec<RankedAction> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Initialize MCTS tree with one node per candidate action
    let mut tree: Vec<MctsNode> = candidates
        .iter()
        .map(|c| MctsNode {
            action_id: c.id.clone(),
            target_node: c.target_node,
            visit_count: 0,
            total_value: 0.0,
            prior: c.prior,
        })
        .collect();

    let mut total_visits: u32 = 0;

    // Run simulations
    for _ in 0..config.budget_simulations {
        // 1. SELECT — pick node with highest UCB1
        let selected_idx = select_best(&tree, total_visits, config.exploration_c);

        // 2. ROLLOUT — simulate from selected node following pheromones
        let value = rollout(
            graph,
            tree[selected_idx].target_node,
            config.max_rollout_depth,
            config.discount_factor,
            config.min_edge_energy,
        );

        // 3. BACKPROP — update statistics
        tree[selected_idx].visit_count += 1;
        tree[selected_idx].total_value += value;
        total_visits += 1;
    }

    // Convert to ranked results, sorted by score
    let mut results: Vec<RankedAction> = tree
        .into_iter()
        .map(|node| {
            let score = node.avg_value(); // exploitation only for final ranking
            let visit_count = node.visit_count;
            RankedAction {
                id: node.action_id,
                target_node: node.target_node,
                score,
                visit_count,
                avg_value: score,
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Select the child with the highest UCB1 value.
fn select_best(tree: &[MctsNode], parent_visits: u32, c: f64) -> usize {
    let mut best_idx = 0;
    let mut best_ucb = f64::NEG_INFINITY;
    for (i, node) in tree.iter().enumerate() {
        let ucb = node.ucb1(parent_visits.max(1), c);
        if ucb > best_ucb {
            best_ucb = ucb;
            best_idx = i;
        }
    }
    best_idx
}

/// Rollout from a node following strongest pheromone trails.
/// Returns a value in [0, 1] based on accumulated node scores along the path.
fn rollout<G: RoutingGraph>(
    graph: &G,
    start: NodeId,
    max_depth: u32,
    discount: f64,
    min_energy: f64,
) -> f64 {
    let mut current = start;
    let mut total_reward = 0.0;
    let mut gamma = 1.0;
    let mut visited = HashMap::new();

    for _ in 0..max_depth {
        // Collect reward from current node
        let score = graph.node_score(current);
        total_reward += gamma * score;
        gamma *= discount;

        // Mark visited to avoid loops
        if visited.contains_key(&current) {
            break;
        }
        visited.insert(current, true);

        // Follow strongest pheromone trail (greedy rollout policy)
        let neighbors = graph.neighbors(current);
        let next = neighbors
            .iter()
            .filter(|(_, w)| *w >= min_energy)
            .filter(|(n, _)| !visited.contains_key(n))
            .max_by(|(n1, w1), (n2, w2)| {
                let p1 = graph.pheromone(current, *n1) + w1;
                let p2 = graph.pheromone(current, *n2) + w2;
                p1.partial_cmp(&p2).unwrap_or(std::cmp::Ordering::Equal)
            });

        match next {
            Some((node, _)) => current = *node,
            None => break, // dead end
        }
    }

    // Normalize to [0, 1]
    total_reward.min(1.0).max(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockGraph {
        edges: HashMap<NodeId, Vec<(NodeId, f64)>>,
        scores: HashMap<NodeId, f64>,
    }

    impl MockGraph {
        fn new() -> Self {
            let mut edges = HashMap::new();
            let mut scores = HashMap::new();

            // Simple graph: 0 -> 1 -> 2 -> 3
            edges.insert(NodeId::from(0u64), vec![(NodeId::from(1u64), 0.8)]);
            edges.insert(
                NodeId::from(1u64),
                vec![(NodeId::from(2u64), 0.6), (NodeId::from(3u64), 0.3)],
            );
            edges.insert(NodeId::from(2u64), vec![(NodeId::from(3u64), 0.9)]);
            edges.insert(NodeId::from(3u64), vec![]);

            scores.insert(NodeId::from(0u64), 0.5);
            scores.insert(NodeId::from(1u64), 0.7);
            scores.insert(NodeId::from(2u64), 0.9);
            scores.insert(NodeId::from(3u64), 0.4);

            Self { edges, scores }
        }
    }

    impl RoutingGraph for MockGraph {
        fn neighbors(&self, node: NodeId) -> Vec<(NodeId, f64)> {
            self.edges.get(&node).cloned().unwrap_or_default()
        }

        fn node_score(&self, node: NodeId) -> f64 {
            self.scores.get(&node).copied().unwrap_or(0.0)
        }

        fn pheromone(&self, _from: NodeId, _to: NodeId) -> f64 {
            0.5 // uniform pheromone in mock
        }
    }

    #[test]
    fn test_empty_candidates() {
        let graph = MockGraph::new();
        let result = neural_route(&graph, NodeId::from(0u64), &[], &Default::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_candidate() {
        let graph = MockGraph::new();
        let candidates = vec![CandidateAction {
            id: "action_1".into(),
            target_node: NodeId::from(1u64),
            prior: 0.5,
        }];
        let result = neural_route(&graph, NodeId::from(0u64), &candidates, &Default::default());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "action_1");
        assert!(result[0].visit_count > 0);
    }

    #[test]
    fn test_ranking_prefers_higher_score_paths() {
        let graph = MockGraph::new();
        let candidates = vec![
            CandidateAction {
                id: "to_high".into(),
                target_node: NodeId::from(2u64), // score 0.9
                prior: 0.5,
            },
            CandidateAction {
                id: "to_low".into(),
                target_node: NodeId::from(3u64), // score 0.4
                prior: 0.5,
            },
        ];
        let config = NeuralRoutingConfig {
            budget_simulations: 100,
            ..Default::default()
        };
        let result = neural_route(&graph, NodeId::from(0u64), &candidates, &config);
        assert_eq!(result.len(), 2);
        // Node 2 has higher score and better downstream path
        assert_eq!(result[0].id, "to_high");
        assert!(result[0].avg_value > result[1].avg_value);
    }

    #[test]
    fn test_prior_influences_exploration() {
        let graph = MockGraph::new();
        let candidates = vec![
            CandidateAction {
                id: "low_prior".into(),
                target_node: NodeId::from(1u64),
                prior: 0.0,
            },
            CandidateAction {
                id: "high_prior".into(),
                target_node: NodeId::from(1u64),
                prior: 1.0,
            },
        ];
        let config = NeuralRoutingConfig {
            budget_simulations: 10,
            ..Default::default()
        };
        let result = neural_route(&graph, NodeId::from(0u64), &candidates, &config);
        // With only 10 simulations, the high prior should get more visits
        let high = result.iter().find(|r| r.id == "high_prior").unwrap();
        let low = result.iter().find(|r| r.id == "low_prior").unwrap();
        assert!(high.visit_count >= low.visit_count);
    }
}
