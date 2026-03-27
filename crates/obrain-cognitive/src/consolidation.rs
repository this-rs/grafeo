//! Structural Consolidation Engine — generic node merging via community detection.
//!
//! Uses Louvain/Leiden (from `grafeo-adapters`) to detect clusters of strongly
//! connected nodes. Low-energy nodes within the same cluster are merged into a
//! condensed node that inherits aggregated properties (max energy, union of
//! labels, rewired synapses). `DERIVED_FROM` edges link condensed nodes back
//! to their originals for full traceability.
//!
//! **Topological protection**: nodes with PageRank above a configurable
//! threshold are excluded from consolidation (they are structurally important).
//!
//! Applicable to any node type — not just Memory nodes — and configurable via
//! an optional label filter.

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::{Direction, GraphStore, GraphStoreMut};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the consolidation engine.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Optional label filter: only nodes with at least one of these labels
    /// are candidates for consolidation. If empty, all nodes are considered.
    pub label_filter: Vec<String>,

    /// Energy threshold: nodes with energy below this value are candidates
    /// for merging (within the same community cluster).
    pub energy_threshold: f64,

    /// PageRank protection threshold: nodes with PageRank above this value
    /// are excluded from consolidation (they are topologically important).
    pub pagerank_protection: f64,

    /// Louvain resolution parameter (higher = more smaller communities).
    pub louvain_resolution: f64,

    /// Minimum cluster size to trigger consolidation (at least 2 low-energy
    /// nodes must exist in a cluster for merging to occur).
    pub min_cluster_size: usize,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            label_filter: Vec::new(),
            energy_threshold: 0.3,
            pagerank_protection: 0.1,
            louvain_resolution: 1.0,
            min_cluster_size: 2,
        }
    }
}

impl ConsolidationConfig {
    /// Creates a config with a label filter.
    pub fn with_label_filter(mut self, labels: Vec<String>) -> Self {
        self.label_filter = labels;
        self
    }

    /// Creates a config with a custom energy threshold.
    pub fn with_energy_threshold(mut self, threshold: f64) -> Self {
        self.energy_threshold = threshold;
        self
    }

    /// Creates a config with a custom PageRank protection threshold.
    pub fn with_pagerank_protection(mut self, threshold: f64) -> Self {
        self.pagerank_protection = threshold;
        self
    }
}

// ---------------------------------------------------------------------------
// Edge type constants
// ---------------------------------------------------------------------------

/// Edge type for traceability links from condensed nodes to original nodes.
pub const EDGE_DERIVED_FROM: &str = "DERIVED_FROM";

// ---------------------------------------------------------------------------
// ConsolidationResult
// ---------------------------------------------------------------------------

/// Result of a consolidation run.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Mapping from condensed node ID → list of original node IDs it replaced.
    pub condensed_nodes: FxHashMap<NodeId, Vec<NodeId>>,

    /// DERIVED_FROM edge IDs created (condensed → original).
    pub derived_from_edges: Vec<EdgeId>,

    /// Nodes that were protected from consolidation (high PageRank).
    pub protected_nodes: Vec<NodeId>,

    /// Total number of nodes removed (merged into condensed nodes).
    pub nodes_removed: usize,

    /// Number of communities detected by Louvain.
    pub communities_detected: usize,
}

// ---------------------------------------------------------------------------
// ConsolidationEngine
// ---------------------------------------------------------------------------

/// Generic structural consolidation engine.
///
/// Detects communities via Louvain, identifies low-energy nodes within each
/// cluster, merges them into condensed nodes, and creates DERIVED_FROM edges.
pub struct ConsolidationEngine {
    /// Configuration.
    config: ConsolidationConfig,
}

impl ConsolidationEngine {
    /// Creates a new consolidation engine with the given configuration.
    pub fn new(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// Runs consolidation on the given graph store.
    ///
    /// # Arguments
    ///
    /// * `store` - Mutable graph store to consolidate
    /// * `energy_map` - Current energy values for nodes (NodeId → energy)
    /// * `pagerank_map` - Current PageRank values for nodes (NodeId → pagerank)
    ///
    /// # Returns
    ///
    /// A [`ConsolidationResult`] describing what was consolidated.
    pub fn consolidate(
        &self,
        store: &dyn GraphStoreMut,
        energy_map: &FxHashMap<NodeId, f64>,
        pagerank_map: &FxHashMap<NodeId, f64>,
    ) -> ConsolidationResult {
        use grafeo_adapters::plugins::algorithms::louvain;

        // 1. Run Louvain community detection
        let louvain_result = louvain(store, self.config.louvain_resolution);

        // 2. Group nodes by community
        let mut communities: HashMap<u64, Vec<NodeId>> = HashMap::new();
        for (&node_id, &community_id) in &louvain_result.communities {
            communities.entry(community_id).or_default().push(node_id);
        }

        let mut result = ConsolidationResult {
            condensed_nodes: FxHashMap::default(),
            derived_from_edges: Vec::new(),
            protected_nodes: Vec::new(),
            nodes_removed: 0,
            communities_detected: louvain_result.num_communities,
        };

        // 3. For each community, identify merge candidates
        for members in communities.values() {
            let mut candidates = Vec::new();

            for &node_id in members {
                // Check label filter
                if !self.passes_label_filter(store, node_id) {
                    continue;
                }

                // Check PageRank protection
                let pr = pagerank_map.get(&node_id).copied().unwrap_or(0.0);
                if pr >= self.config.pagerank_protection {
                    result.protected_nodes.push(node_id);
                    continue;
                }

                // Check energy threshold
                let energy = energy_map.get(&node_id).copied().unwrap_or(0.0);
                if energy < self.config.energy_threshold {
                    candidates.push(node_id);
                }
            }

            // Need at least min_cluster_size candidates to merge
            if candidates.len() < self.config.min_cluster_size {
                continue;
            }

            // Sort candidates for deterministic behavior
            candidates.sort();

            // 4. Merge candidates into a condensed node
            let (condensed_id, edges) = self.merge_nodes(store, &candidates, energy_map);
            result.nodes_removed += candidates.len();
            result.condensed_nodes.insert(condensed_id, candidates);
            result.derived_from_edges.extend(edges);
        }

        result
    }

    /// Checks whether a node passes the label filter.
    fn passes_label_filter(&self, store: &dyn GraphStore, node_id: NodeId) -> bool {
        if self.config.label_filter.is_empty() {
            return true;
        }
        if let Some(node) = store.get_node(node_id) {
            for label in &node.labels {
                if self.config.label_filter.iter().any(|f| f == label.as_str()) {
                    return true;
                }
            }
        }
        false
    }

    /// Merges a set of candidate nodes into a single condensed node.
    ///
    /// The condensed node:
    /// - Gets the union of all labels from original nodes
    /// - Gets max energy among originals
    /// - Gets all properties merged (last-write-wins for conflicts)
    /// - All edges to/from original nodes are rewired to the condensed node
    /// - DERIVED_FROM edges are created from condensed to each original
    /// - Original nodes are deleted
    fn merge_nodes(
        &self,
        store: &dyn GraphStoreMut,
        candidates: &[NodeId],
        energy_map: &FxHashMap<NodeId, f64>,
    ) -> (NodeId, Vec<EdgeId>) {
        // Collect union of labels
        let mut all_labels: BTreeSet<String> = BTreeSet::new();
        let mut all_properties: FxHashMap<String, Value> = FxHashMap::default();
        let mut max_energy: f64 = 0.0;

        for &node_id in candidates {
            if let Some(node) = store.get_node(node_id) {
                for label in &node.labels {
                    all_labels.insert(label.to_string());
                }
                for (key, value) in &node.properties {
                    all_properties.insert(key.to_string(), value.clone());
                }
            }
            let energy = energy_map.get(&node_id).copied().unwrap_or(0.0);
            if energy > max_energy {
                max_energy = energy;
            }
        }

        // Add a "Condensed" label
        all_labels.insert("Condensed".to_string());

        // Create the condensed node with all labels
        let label_refs: Vec<&str> = all_labels.iter().map(|s| s.as_str()).collect();
        let condensed_id = store.create_node(&label_refs);

        // Set aggregated properties on the condensed node
        for (key, value) in &all_properties {
            store.set_node_property(condensed_id, key, value.clone());
        }

        // Store the max energy
        store.set_node_property(condensed_id, "_cog_energy", Value::Float64(max_energy));

        // Store count of merged nodes
        store.set_node_property(
            condensed_id,
            "_cog_merged_count",
            Value::Int64(candidates.len() as i64),
        );

        // Collect edges to rewire (before deleting originals)
        let candidate_set: HashSet<NodeId> = candidates.iter().copied().collect();
        let mut edges_to_rewire: Vec<(NodeId, NodeId, String)> = Vec::new();

        for &node_id in candidates {
            // Outgoing edges
            for (target, edge_id) in store.edges_from(node_id, Direction::Outgoing) {
                if candidate_set.contains(&target) {
                    continue; // Skip intra-cluster edges
                }
                if let Some(edge) = store.get_edge(edge_id) {
                    edges_to_rewire.push((condensed_id, target, edge.edge_type.to_string()));
                }
            }
            // Incoming edges
            for (source, edge_id) in store.edges_from(node_id, Direction::Incoming) {
                if candidate_set.contains(&source) {
                    continue; // Skip intra-cluster edges
                }
                if let Some(edge) = store.get_edge(edge_id) {
                    edges_to_rewire.push((source, condensed_id, edge.edge_type.to_string()));
                }
            }
        }

        // Deduplicate rewired edges (same src, dst, type)
        let mut seen_edges: HashSet<(NodeId, NodeId, String)> = HashSet::new();
        let mut unique_rewired: Vec<(NodeId, NodeId, String)> = Vec::new();
        for edge in edges_to_rewire {
            if seen_edges.insert(edge.clone()) {
                unique_rewired.push(edge);
            }
        }

        // Create rewired edges
        for (src, dst, edge_type) in &unique_rewired {
            store.create_edge(*src, *dst, edge_type);
        }

        // Delete original nodes' non-DERIVED_FROM edges first
        // We collect edge IDs to delete to avoid mutating while iterating
        for &node_id in candidates {
            let edges_to_delete: Vec<grafeo_common::types::EdgeId> = store
                .edges_from(node_id, Direction::Outgoing)
                .into_iter()
                .map(|(_, eid)| eid)
                .chain(
                    store
                        .edges_from(node_id, Direction::Incoming)
                        .into_iter()
                        .map(|(_, eid)| eid),
                )
                .collect();
            for eid in edges_to_delete {
                store.delete_edge(eid);
            }
        }

        // Create DERIVED_FROM edges (condensed → original) — after clearing old edges
        let mut derived_edges = Vec::new();
        for &original_id in candidates {
            let edge_id = store.create_edge(condensed_id, original_id, EDGE_DERIVED_FROM);
            derived_edges.push(edge_id);
        }

        // Mark original nodes as consolidated (don't delete — keep for traceability)
        for &node_id in candidates {
            store.set_node_property(node_id, "_cog_consolidated", Value::Bool(true));
        }

        (condensed_id, derived_edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ConsolidationConfig::default();
        assert!(config.label_filter.is_empty());
        assert!((config.energy_threshold - 0.3).abs() < f64::EPSILON);
        assert!((config.pagerank_protection - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.min_cluster_size, 2);
    }

    #[test]
    fn test_config_builders() {
        let config = ConsolidationConfig::default()
            .with_label_filter(vec!["Memory".into()])
            .with_energy_threshold(0.5)
            .with_pagerank_protection(0.2);

        assert_eq!(config.label_filter, vec!["Memory"]);
        assert!((config.energy_threshold - 0.5).abs() < f64::EPSILON);
        assert!((config.pagerank_protection - 0.2).abs() < f64::EPSILON);
    }
}
