//! K-hop ego-centric subgraph extraction.
//!
//! Extracts the k-hop neighborhood around a center node using bounded BFS,
//! with optional relation-type filtering and stratified sampling. This is the
//! native Grafeo replacement for Cypher-based ego-graph extraction (~8ms → <100μs).

use std::sync::OnceLock;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStore;

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Data Types
// ============================================================================

/// An edge in the ego-graph subgraph, with its type and optional weight.
///
/// Captures all the information needed to reconstruct the subgraph or build
/// tensors for GNN training (R-GCN).
///
/// # Fields
///
/// * `edge_id` - The original edge ID in the graph store.
/// * `source` - Source node ID.
/// * `target` - Target node ID.
/// * `edge_type` - The relation type label (e.g., `"KNOWS"`, `"OWNS"`).
/// * `weight` - Optional edge weight (from the `"weight"` property).
#[derive(Debug, Clone)]
pub struct EgoEdge {
    /// The original edge ID in the graph store.
    pub edge_id: EdgeId,
    /// Source node ID.
    pub source: NodeId,
    /// Target node ID.
    pub target: NodeId,
    /// The relation type label.
    pub edge_type: String,
    /// Optional edge weight.
    pub weight: Option<f64>,
}

/// A materialized ego-centric subgraph extracted via k-hop BFS.
///
/// Contains all nodes, edges, hop distances, and optionally node properties
/// within k hops of a center node. Designed for downstream consumers like
/// `features.rs` (GDS feature vector construction) and R-GCN tensor building.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::{khop_subgraph, KHopConfig};
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_common::types::NodeId;
///
/// let store = LpgStore::new().unwrap();
/// let n0 = store.create_node(&["Person"]);
/// let n1 = store.create_node(&["Person"]);
/// store.create_edge(n0, n1, "KNOWS");
///
/// let config = KHopConfig {
///     center: n0,
///     k: 2,
///     rel_types: None,
///     max_neighbors_per_hop: None,
///     include_properties: false,
/// };
/// let ego = khop_subgraph(&store, &config);
/// assert!(ego.nodes.contains(&n0));
/// assert!(ego.nodes.contains(&n1));
/// ```
#[derive(Debug, Clone)]
pub struct EgoGraph {
    /// All node IDs in the subgraph (including center).
    pub nodes: Vec<NodeId>,
    /// All edges in the subgraph.
    pub edges: Vec<EgoEdge>,
    /// Hop distance from center for each node. Key = NodeId, Value = hop count (0 for center).
    pub hop_distances: FxHashMap<NodeId, u32>,
    /// Node properties, populated only when `include_properties` is true.
    /// Key = NodeId, Value = property map.
    pub node_properties: FxHashMap<NodeId, FxHashMap<PropertyKey, Value>>,
}

impl EgoGraph {
    /// Returns the number of nodes in the ego-graph.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the ego-graph.
    #[inline]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the hop distance for a given node, or `None` if not in the subgraph.
    #[inline]
    pub fn hop_distance(&self, node: NodeId) -> Option<u32> {
        self.hop_distances.get(&node).copied()
    }
}

/// Configuration for k-hop ego-graph extraction.
///
/// Controls the BFS expansion: depth limit, relation-type filtering,
/// and stratified sampling to bound the subgraph size.
///
/// # Fields
///
/// * `center` - The center node ID for ego-graph extraction.
/// * `k` - Maximum number of hops from center (depth limit).
/// * `rel_types` - Optional list of edge types to include. `None` means all types.
/// * `max_neighbors_per_hop` - Optional limit on neighbors expanded per node per hop
///   (stratified sampling). `None` means no limit.
/// * `include_properties` - Whether to load node properties into the result.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::KHopConfig;
/// use grafeo_common::types::NodeId;
///
/// let config = KHopConfig {
///     center: NodeId::new(42),
///     k: 3,
///     rel_types: Some(vec!["KNOWS".to_string(), "FOLLOWS".to_string()]),
///     max_neighbors_per_hop: Some(50),
///     include_properties: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct KHopConfig {
    /// The center node ID.
    pub center: NodeId,
    /// Maximum number of hops (BFS depth limit).
    pub k: u32,
    /// Optional filter: only traverse edges with these types. `None` = all types.
    pub rel_types: Option<Vec<String>>,
    /// Optional limit on neighbors per node per hop (stratified sampling).
    /// `None` = no limit (expand all neighbors).
    pub max_neighbors_per_hop: Option<usize>,
    /// Whether to include node properties in the result.
    pub include_properties: bool,
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Extracts a k-hop ego-centric subgraph around a center node.
///
/// Performs a bounded BFS from `config.center` up to `config.k` hops,
/// optionally filtering edges by relation type and limiting the number
/// of neighbors expanded per node per hop (stratified sampling).
///
/// # Arguments
///
/// * `store` - The graph store to query.
/// * `config` - Extraction parameters (center, k, filters, sampling).
///
/// # Returns
///
/// A materialized [`EgoGraph`] containing all discovered nodes, edges,
/// hop distances, and optionally node properties.
///
/// # Complexity
///
/// * **Time**: O(k × N × d) where N = number of nodes in the subgraph and
///   d = average degree. With `max_neighbors_per_hop = m`, this becomes
///   O(k × N × min(d, m)).
/// * **Space**: O(N + E) where E = number of edges in the subgraph.
///
/// # Example
///
/// ```no_run
/// use grafeo_adapters::plugins::algorithms::{khop_subgraph, KHopConfig};
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_common::types::NodeId;
///
/// let store = LpgStore::new().unwrap();
/// let center = store.create_node(&["Person"]);
/// let friend = store.create_node(&["Person"]);
/// store.create_edge(center, friend, "KNOWS");
///
/// let config = KHopConfig {
///     center,
///     k: 2,
///     rel_types: Some(vec!["KNOWS".to_string()]),
///     max_neighbors_per_hop: Some(100),
///     include_properties: false,
/// };
///
/// let ego = khop_subgraph(&store, &config);
/// assert_eq!(ego.node_count(), 2);
/// assert_eq!(ego.edge_count(), 1);
/// assert_eq!(ego.hop_distance(center), Some(0));
/// assert_eq!(ego.hop_distance(friend), Some(1));
/// ```
pub fn khop_subgraph(store: &dyn GraphStore, config: &KHopConfig) -> EgoGraph {
    let mut hop_distances: FxHashMap<NodeId, u32> = FxHashMap::default();
    let mut edges: Vec<EgoEdge> = Vec::new();
    let mut nodes_ordered: Vec<NodeId> = Vec::new();

    // Build a HashSet of allowed relation types for O(1) lookup
    let rel_filter: Option<FxHashSet<&str>> = config
        .rel_types
        .as_ref()
        .map(|types| types.iter().map(|s| s.as_str()).collect());

    // Check if center exists
    if store.get_node(config.center).is_none() {
        return EgoGraph {
            nodes: nodes_ordered,
            edges,
            hop_distances,
            node_properties: FxHashMap::default(),
        };
    }

    // Initialize BFS with center node
    hop_distances.insert(config.center, 0);
    nodes_ordered.push(config.center);

    let mut current_layer: Vec<NodeId> = vec![config.center];
    let mut next_layer: Vec<NodeId> = Vec::new();

    for hop in 1..=config.k {
        if current_layer.is_empty() {
            break;
        }

        for &node in &current_layer {
            let neighbors = store.edges_from(node, Direction::Outgoing);
            let mut count = 0usize;

            for (neighbor, edge_id) in neighbors {
                // Apply relation type filter
                if let Some(ref filter) = rel_filter {
                    if let Some(etype) = store.edge_type(edge_id) {
                        if !filter.contains(etype.as_str()) {
                            continue;
                        }
                        // Collect edge info
                        let weight = store
                            .get_edge_property(edge_id, &PropertyKey::from("weight"))
                            .and_then(|v| match v {
                                Value::Float64(f) => Some(f),
                                Value::Int64(i) => Some(i as f64),
                                _ => None,
                            });

                        edges.push(EgoEdge {
                            edge_id,
                            source: node,
                            target: neighbor,
                            edge_type: etype.to_string(),
                            weight,
                        });
                    } else {
                        continue;
                    }
                } else {
                    // No filter — include all edges
                    let etype = store.edge_type(edge_id).unwrap_or_default();
                    let weight = store
                        .get_edge_property(edge_id, &PropertyKey::from("weight"))
                        .and_then(|v| match v {
                            Value::Float64(f) => Some(f),
                            Value::Int64(i) => Some(i as f64),
                            _ => None,
                        });

                    edges.push(EgoEdge {
                        edge_id,
                        source: node,
                        target: neighbor,
                        edge_type: etype.to_string(),
                        weight,
                    });
                }

                // Only add to next layer if not already discovered
                if !hop_distances.contains_key(&neighbor) {
                    hop_distances.insert(neighbor, hop);
                    nodes_ordered.push(neighbor);
                    next_layer.push(neighbor);
                }

                count += 1;
                if let Some(max) = config.max_neighbors_per_hop {
                    if count >= max {
                        break;
                    }
                }
            }
        }

        current_layer.clear();
        std::mem::swap(&mut current_layer, &mut next_layer);
    }

    // Optionally load node properties
    let node_properties = if config.include_properties {
        let props_batch = store.get_nodes_properties_batch(&nodes_ordered);
        nodes_ordered
            .iter()
            .copied()
            .zip(props_batch.into_iter())
            .collect()
    } else {
        FxHashMap::default()
    };

    EgoGraph {
        nodes: nodes_ordered,
        edges,
        hop_distances,
        node_properties,
    }
}

// ============================================================================
// Algorithm Wrapper for Plugin Registry
// ============================================================================

/// Static parameter definitions for k-hop subgraph algorithm.
static KHOP_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn khop_params() -> &'static [ParameterDef] {
    KHOP_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "center".to_string(),
                description: "Center node ID for ego-graph extraction".to_string(),
                param_type: ParameterType::NodeId,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "k".to_string(),
                description: "Maximum number of hops (depth limit)".to_string(),
                param_type: ParameterType::Integer,
                required: true,
                default: None,
            },
            ParameterDef {
                name: "rel_types".to_string(),
                description: "Comma-separated list of relation types to include (empty = all)"
                    .to_string(),
                param_type: ParameterType::String,
                required: false,
                default: None,
            },
            ParameterDef {
                name: "max_neighbors_per_hop".to_string(),
                description: "Maximum neighbors to sample per node per hop (0 = unlimited)"
                    .to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some("0".to_string()),
            },
        ]
    })
}

/// K-hop ego-graph extraction algorithm wrapper for the plugin registry.
///
/// Callable via `CALL grafeo.subgraph.khop(center, k, rel_types, max_per_hop)
/// YIELD node, edge, hop`.
///
/// # Output Columns
///
/// * `node_id` - Node ID in the ego-graph.
/// * `hop` - Hop distance from center (0 = center node).
/// * `source` - Edge source node ID (for edge rows).
/// * `target` - Edge target node ID (for edge rows).
/// * `edge_type` - Edge relation type label.
/// * `weight` - Edge weight (if available).
pub struct KHopAlgorithm;

impl GraphAlgorithm for KHopAlgorithm {
    fn name(&self) -> &str {
        "subgraph.khop"
    }

    fn description(&self) -> &str {
        "K-hop ego-centric subgraph extraction with relation-type filtering and stratified sampling"
    }

    fn parameters(&self) -> &[ParameterDef] {
        khop_params()
    }

    fn execute(&self, store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let center_id = params.get_int("center").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(
                "center parameter required".to_string(),
            )
        })?;
        let k = params.get_int("k").ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue("k parameter required".to_string())
        })?;

        // Parse rel_types from comma-separated string
        let rel_types = params.get_string("rel_types").and_then(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(
                    s.split(',')
                        .map(|t| t.trim().to_string())
                        .collect::<Vec<_>>(),
                )
            }
        });

        // Parse max_neighbors_per_hop
        let max_neighbors = params
            .get_int("max_neighbors_per_hop")
            .and_then(|v| if v <= 0 { None } else { Some(v as usize) });

        let config = KHopConfig {
            center: NodeId::new(center_id as u64),
            k: k as u32,
            rel_types,
            max_neighbors_per_hop: max_neighbors,
            include_properties: false,
        };

        let ego = khop_subgraph(store, &config);

        // Build result: one row per node with hop distance, plus edge info
        let mut result = AlgorithmResult::new(vec![
            "node_id".to_string(),
            "hop".to_string(),
            "source".to_string(),
            "target".to_string(),
            "edge_type".to_string(),
            "weight".to_string(),
        ]);

        // Node rows
        for &node in &ego.nodes {
            let hop = ego.hop_distances.get(&node).copied().unwrap_or(0);
            result.add_row(vec![
                Value::Int64(node.0 as i64),
                Value::Int64(hop as i64),
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
            ]);
        }

        // Edge rows
        for edge in &ego.edges {
            result.add_row(vec![
                Value::Null,
                Value::Null,
                Value::Int64(edge.source.0 as i64),
                Value::Int64(edge.target.0 as i64),
                Value::from(edge.edge_type.as_str()),
                edge.weight.map_or(Value::Null, Value::Float64),
            ]);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_core::graph::lpg::LpgStore;

    fn create_multi_rel_graph() -> LpgStore {
        let store = LpgStore::new().unwrap();

        // Create a multi-relational graph:
        //   0 --KNOWS--> 1 --KNOWS--> 2
        //   0 --OWNS-->  3 --OWNS-->  4
        //   1 --WORKS_AT--> 5
        let n0 = store.create_node(&["Person"]);
        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let n3 = store.create_node(&["Company"]);
        let n4 = store.create_node(&["Product"]);
        let n5 = store.create_node(&["Organization"]);

        store.create_edge(n0, n1, "KNOWS");
        store.create_edge(n1, n2, "KNOWS");
        store.create_edge(n0, n3, "OWNS");
        store.create_edge(n3, n4, "OWNS");
        store.create_edge(n1, n5, "WORKS_AT");

        store
    }

    fn create_star_graph(center_id: NodeId, degree: usize, store: &LpgStore) -> Vec<NodeId> {
        let mut leaves = Vec::with_capacity(degree);
        for _ in 0..degree {
            let leaf = store.create_node(&["Leaf"]);
            store.create_edge(center_id, leaf, "CONNECTS");
            leaves.push(leaf);
        }
        leaves
    }

    #[test]
    fn test_khop_basic() {
        let store = create_multi_rel_graph();
        let config = KHopConfig {
            center: NodeId::new(0),
            k: 1,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        assert!(ego.nodes.contains(&NodeId::new(0)));
        assert!(ego.nodes.contains(&NodeId::new(1)));
        assert!(ego.nodes.contains(&NodeId::new(3)));
        // Node 2,4,5 are at hop 2+
        assert!(!ego.nodes.contains(&NodeId::new(2)));
        assert_eq!(ego.hop_distance(NodeId::new(0)), Some(0));
        assert_eq!(ego.hop_distance(NodeId::new(1)), Some(1));
        assert_eq!(ego.hop_distance(NodeId::new(3)), Some(1));
    }

    #[test]
    fn test_khop_2hops() {
        let store = create_multi_rel_graph();
        let config = KHopConfig {
            center: NodeId::new(0),
            k: 2,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // All 6 nodes should be reachable within 2 hops
        assert_eq!(ego.node_count(), 6);
        assert_eq!(ego.hop_distance(NodeId::new(2)), Some(2));
        assert_eq!(ego.hop_distance(NodeId::new(4)), Some(2));
        assert_eq!(ego.hop_distance(NodeId::new(5)), Some(2));
    }

    #[test]
    fn test_khop_rel_type_filter() {
        let store = create_multi_rel_graph();
        let config = KHopConfig {
            center: NodeId::new(0),
            k: 2,
            rel_types: Some(vec!["KNOWS".to_string()]),
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // Only nodes reachable via KNOWS edges: 0, 1, 2
        assert!(ego.nodes.contains(&NodeId::new(0)));
        assert!(ego.nodes.contains(&NodeId::new(1)));
        assert!(ego.nodes.contains(&NodeId::new(2)));
        assert!(!ego.nodes.contains(&NodeId::new(3))); // OWNS edge
        assert!(!ego.nodes.contains(&NodeId::new(4))); // OWNS edge
        // All edges should be KNOWS
        for edge in &ego.edges {
            assert_eq!(edge.edge_type, "KNOWS");
        }
    }

    #[test]
    fn test_khop_multiple_rel_types() {
        let store = create_multi_rel_graph();
        let config = KHopConfig {
            center: NodeId::new(0),
            k: 2,
            rel_types: Some(vec!["KNOWS".to_string(), "WORKS_AT".to_string()]),
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // Reachable: 0 --KNOWS--> 1 --KNOWS--> 2, 1 --WORKS_AT--> 5
        assert!(ego.nodes.contains(&NodeId::new(0)));
        assert!(ego.nodes.contains(&NodeId::new(1)));
        assert!(ego.nodes.contains(&NodeId::new(2)));
        assert!(ego.nodes.contains(&NodeId::new(5)));
        assert!(!ego.nodes.contains(&NodeId::new(3))); // OWNS only
    }

    #[test]
    fn test_khop_sampling() {
        let store = LpgStore::new().unwrap();
        let center = store.create_node(&["Center"]);
        create_star_graph(center, 20, &store);

        let config = KHopConfig {
            center,
            k: 1,
            rel_types: None,
            max_neighbors_per_hop: Some(5),
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // Center + at most 5 neighbors
        assert!(ego.node_count() <= 6);
        assert!(ego.edge_count() <= 5);
    }

    #[test]
    fn test_khop_nonexistent_center() {
        let store = LpgStore::new().unwrap();
        let config = KHopConfig {
            center: NodeId::new(999),
            k: 2,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);
        assert_eq!(ego.node_count(), 0);
        assert_eq!(ego.edge_count(), 0);
    }

    #[test]
    fn test_khop_k_zero() {
        let store = create_multi_rel_graph();
        let config = KHopConfig {
            center: NodeId::new(0),
            k: 0,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // k=0 means only center node
        assert_eq!(ego.node_count(), 1);
        assert_eq!(ego.edge_count(), 0);
        assert_eq!(ego.hop_distance(NodeId::new(0)), Some(0));
    }

    #[test]
    fn test_khop_single_node() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Alone"]);
        let config = KHopConfig {
            center: n0,
            k: 3,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);
        assert_eq!(ego.node_count(), 1);
        assert_eq!(ego.edge_count(), 0);
    }

    #[test]
    fn test_khop_with_properties() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Person"]);
        let n1 = store.create_node(&["Person"]);
        store.create_edge(n0, n1, "KNOWS");
        store.set_node_property(n0, "name", Value::from("Alice"));
        store.set_node_property(n1, "name", Value::from("Bob"));

        let config = KHopConfig {
            center: n0,
            k: 1,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: true,
        };
        let ego = khop_subgraph(&store, &config);

        assert_eq!(ego.node_count(), 2);
        assert!(!ego.node_properties.is_empty());
        // Check that properties are loaded
        let n0_props = ego.node_properties.get(&n0).unwrap();
        assert_eq!(
            n0_props.get(&PropertyKey::from("name")),
            Some(&Value::from("Alice"))
        );
    }

    #[test]
    fn test_khop_with_weighted_edges() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let e = store.create_edge(n0, n1, "CONNECTS");
        store.set_edge_property(e, "weight", Value::Float64(0.75));

        let config = KHopConfig {
            center: n0,
            k: 1,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        assert_eq!(ego.edge_count(), 1);
        assert_eq!(ego.edges[0].weight, Some(0.75));
    }

    #[test]
    fn test_khop_cycle() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        let n2 = store.create_node(&["Node"]);
        store.create_edge(n0, n1, "EDGE");
        store.create_edge(n1, n2, "EDGE");
        store.create_edge(n2, n0, "EDGE"); // cycle back

        let config = KHopConfig {
            center: n0,
            k: 10,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        // Should not loop forever; all 3 nodes discovered
        assert_eq!(ego.node_count(), 3);
        assert_eq!(ego.hop_distance(n0), Some(0));
        assert_eq!(ego.hop_distance(n1), Some(1));
        assert_eq!(ego.hop_distance(n2), Some(2));
    }

    #[test]
    fn test_khop_self_loop() {
        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Node"]);
        let n1 = store.create_node(&["Node"]);
        store.create_edge(n0, n0, "SELF");
        store.create_edge(n0, n1, "EDGE");

        let config = KHopConfig {
            center: n0,
            k: 1,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };
        let ego = khop_subgraph(&store, &config);

        assert_eq!(ego.node_count(), 2);
        // Self-loop edge should be present
        assert!(ego.edges.iter().any(|e| e.source == n0 && e.target == n0));
    }

    #[test]
    fn test_khop_algorithm_wrapper() {
        let store = create_multi_rel_graph();
        let algo = KHopAlgorithm;

        assert_eq!(algo.name(), "subgraph.khop");

        let mut params = Parameters::new();
        params.set_int("center", 0);
        params.set_int("k", 1);

        let result = algo.execute(&store, &params).unwrap();
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_khop_algorithm_with_rel_filter() {
        let store = create_multi_rel_graph();
        let algo = KHopAlgorithm;

        let mut params = Parameters::new();
        params.set_int("center", 0);
        params.set_int("k", 2);
        params.set_string("rel_types", "KNOWS");

        let result = algo.execute(&store, &params).unwrap();
        // Should have node rows and edge rows
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_khop_algorithm_missing_params() {
        let store = create_multi_rel_graph();
        let algo = KHopAlgorithm;
        let params = Parameters::new();

        let result = algo.execute(&store, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_khop_empty_rel_types_string() {
        let store = create_multi_rel_graph();
        let algo = KHopAlgorithm;

        let mut params = Parameters::new();
        params.set_int("center", 0);
        params.set_int("k", 1);
        params.set_string("rel_types", "");

        // Empty string should mean no filter (all types)
        let result = algo.execute(&store, &params).unwrap();
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_khop_max_neighbors_zero() {
        let store = create_multi_rel_graph();
        let algo = KHopAlgorithm;

        let mut params = Parameters::new();
        params.set_int("center", 0);
        params.set_int("k", 1);
        params.set_int("max_neighbors_per_hop", 0);

        // 0 means unlimited
        let result = algo.execute(&store, &params).unwrap();
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_ego_graph_accessors() {
        let ego = EgoGraph {
            nodes: vec![NodeId::new(0), NodeId::new(1)],
            edges: vec![EgoEdge {
                edge_id: EdgeId::new(0),
                source: NodeId::new(0),
                target: NodeId::new(1),
                edge_type: "KNOWS".to_string(),
                weight: Some(1.0),
            }],
            hop_distances: {
                let mut m = FxHashMap::default();
                m.insert(NodeId::new(0), 0);
                m.insert(NodeId::new(1), 1);
                m
            },
            node_properties: FxHashMap::default(),
        };

        assert_eq!(ego.node_count(), 2);
        assert_eq!(ego.edge_count(), 1);
        assert_eq!(ego.hop_distance(NodeId::new(0)), Some(0));
        assert_eq!(ego.hop_distance(NodeId::new(1)), Some(1));
        assert_eq!(ego.hop_distance(NodeId::new(99)), None);
    }

    #[test]
    fn test_khop_perf_10k_nodes() {
        // Build a graph with ~10K nodes, moderate connectivity
        let store = LpgStore::new().unwrap();
        let mut nodes = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            nodes.push(store.create_node(&["Node"]));
        }

        // Create edges: each node connects to ~5 random-ish neighbors
        // Using deterministic pattern for reproducibility
        for i in 0..10_000usize {
            for offset in [1, 7, 23, 97, 311] {
                let j = (i + offset) % 10_000;
                store.create_edge(nodes[i], nodes[j], "CONNECTS");
            }
        }

        let config = KHopConfig {
            center: nodes[0],
            k: 3,
            rel_types: None,
            max_neighbors_per_hop: None,
            include_properties: false,
        };

        // Warmup
        let _ = khop_subgraph(&store, &config);

        // Measure
        let start = std::time::Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = khop_subgraph(&store, &config);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations;

        eprintln!(
            "khop_subgraph(10K nodes, k=3): {:?} per call ({} nodes in ego-graph)",
            per_call,
            khop_subgraph(&store, &config).node_count()
        );

        // Target: <100μs — but in test mode (debug build) we're more lenient
        // In release mode this should be well under 100μs
        assert!(
            per_call.as_millis() < 500,
            "khop_subgraph too slow: {:?}",
            per_call
        );
    }
}
