//! Spreading activation engine — BFS-based energy propagation through synapses.
//!
//! Starting from one or more source nodes, energy propagates to neighbors
//! weighted by synapse strength and attenuated by a `decay_factor` per hop.
//! The result is an [`ActivationMap`] that can be used to rank query results
//! or identify contextually relevant subgraphs.

use crate::synapse::SynapseStore;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Type alias for the activation result: node -> accumulated energy.
pub type ActivationMap = HashMap<NodeId, f64>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for spreading activation.
#[derive(Debug, Clone)]
pub struct SpreadConfig {
    /// Maximum number of hops from any source.
    pub max_hops: u32,
    /// Minimum propagated energy to continue spreading (cutoff).
    pub min_propagated_energy: f64,
    /// Decay factor applied per hop: `propagated = parent * weight * decay_factor`.
    pub decay_factor: f64,
    /// Only include nodes with activation above this threshold in the result.
    pub activation_threshold: f64,
    /// Maximum number of nodes that can be activated (circuit breaker).
    /// Once this limit is reached, no further nodes are enqueued.
    /// Default: 1000. Set to 0 for unlimited (not recommended for dense graphs).
    pub max_activated_nodes: usize,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            min_propagated_energy: 0.01,
            decay_factor: 0.5,
            activation_threshold: 0.0,
            max_activated_nodes: 1000,
        }
    }
}

impl SpreadConfig {
    /// Builder: set max hops.
    pub fn with_max_hops(mut self, hops: u32) -> Self {
        self.max_hops = hops;
        self
    }

    /// Builder: set decay factor.
    pub fn with_decay_factor(mut self, factor: f64) -> Self {
        self.decay_factor = factor;
        self
    }

    /// Builder: set minimum propagated energy cutoff.
    pub fn with_min_energy(mut self, min_energy: f64) -> Self {
        self.min_propagated_energy = min_energy;
        self
    }

    /// Builder: set activation threshold.
    pub fn with_activation_threshold(mut self, threshold: f64) -> Self {
        self.activation_threshold = threshold;
        self
    }

    /// Builder: set maximum activated nodes (circuit breaker).
    pub fn with_max_activated_nodes(mut self, max: usize) -> Self {
        self.max_activated_nodes = max;
        self
    }
}

// ---------------------------------------------------------------------------
// ActivationSource trait
// ---------------------------------------------------------------------------

/// Provides weighted neighbors for the spreading activation algorithm.
///
/// The default implementation reads from a [`SynapseStore`].
pub trait ActivationSource: Send + Sync {
    /// Returns neighbors of `node_id` with their synapse weights.
    fn neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f64)>;
}

/// [`ActivationSource`] backed by a [`SynapseStore`].
pub struct SynapseActivationSource {
    store: Arc<SynapseStore>,
}

impl SynapseActivationSource {
    /// Creates a new source from a synapse store.
    pub fn new(store: Arc<SynapseStore>) -> Self {
        Self { store }
    }
}

impl ActivationSource for SynapseActivationSource {
    fn neighbors(&self, node_id: NodeId) -> Vec<(NodeId, f64)> {
        self.store
            .list_synapses(node_id)
            .into_iter()
            .map(|s| {
                let neighbor = if s.source == node_id {
                    s.target
                } else {
                    s.source
                };
                (neighbor, s.current_weight())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Spread algorithm
// ---------------------------------------------------------------------------

/// Performs spreading activation from multiple source nodes.
///
/// Energy propagates via BFS: at each hop, the energy reaching a neighbor is
/// `parent_energy * synapse_weight * decay_factor`. If a node is reached
/// by multiple paths, the energies are summed (superposition).
///
/// # Arguments
/// * `sources` - Initial activation: `(node_id, initial_energy)` pairs.
/// * `graph` - Provides weighted neighbors (typically backed by synapses).
/// * `config` - Spread parameters (max hops, decay, thresholds).
///
/// # Returns
/// An [`ActivationMap`] with accumulated energy for each reached node.
pub fn spread(
    sources: &[(NodeId, f64)],
    graph: &dyn ActivationSource,
    config: &SpreadConfig,
) -> ActivationMap {
    let mut activation: ActivationMap = HashMap::new();

    // BFS queue: (node_id, energy_at_node, current_hop)
    let mut queue: VecDeque<(NodeId, f64, u32)> = VecDeque::new();

    // Track which (node, hop) pairs have been enqueued to prevent re-enqueue
    // at the same depth (avoids exponential queue growth on dense graphs).
    let mut visited: HashSet<(NodeId, u32)> = HashSet::new();

    // Seed sources
    for &(node_id, energy) in sources {
        *activation.entry(node_id).or_insert(0.0) += energy;
        queue.push_back((node_id, energy, 0));
        visited.insert((node_id, 0));
    }

    let max_nodes = config.max_activated_nodes;

    // BFS propagation with circuit breaker
    while let Some((node_id, node_energy, hop)) = queue.pop_front() {
        if hop >= config.max_hops {
            continue;
        }

        // Circuit breaker: stop enqueuing new nodes once we hit the limit
        if max_nodes > 0 && activation.len() >= max_nodes {
            break;
        }

        for (neighbor, weight) in graph.neighbors(node_id) {
            let propagated = node_energy * weight * config.decay_factor;

            if propagated < config.min_propagated_energy {
                continue;
            }

            *activation.entry(neighbor).or_insert(0.0) += propagated;

            // Only enqueue if not yet visited at this hop depth
            let next_hop = hop + 1;
            if next_hop < config.max_hops && visited.insert((neighbor, next_hop)) {
                queue.push_back((neighbor, propagated, next_hop));
            }
        }
    }

    // Filter by activation threshold
    if config.activation_threshold > 0.0 {
        activation.retain(|_, v| *v >= config.activation_threshold);
    }

    activation
}

// ---------------------------------------------------------------------------
// Convenience: spread from a single source
// ---------------------------------------------------------------------------

/// Spreads activation from a single source node.
pub fn spread_single(
    source: NodeId,
    energy: f64,
    graph: &dyn ActivationSource,
    config: &SpreadConfig,
) -> ActivationMap {
    spread(&[(source, energy)], graph, config)
}
