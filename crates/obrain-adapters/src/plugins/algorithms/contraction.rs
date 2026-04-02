//! Subgraph contraction (graph coarsening).
//!
//! Contracts a set of nodes into a single **super-node**, redirecting external
//! edges and optionally aggregating numeric properties. The operation is
//! reversible via [`expand_supernode`] using the saved [`ContractionSnapshot`].
//!
//! ## Use cases
//!
//! - **Visualization**: collapse a community into one node to reduce clutter
//! - **Algorithmic speedup**: run expensive algorithms on a coarsened graph
//! - **Hierarchical analysis**: contract by Louvain communities for multi-scale view
//!
//! ## Complexity
//!
//! - `contract_subgraph`: O(V_sub + E_sub + E_external) where V_sub/E_sub are
//!   the contracted subgraph size and E_external are edges crossing the boundary
//! - `expand_supernode`: O(V_sub + E_sub + E_external) — symmetric to contraction
//! - `contract_by_communities`: O(V + E) — one pass over the full graph

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use obrain_common::types::{EdgeId, NodeId, PropertyKey, Value};
use obrain_common::utils::error::Result;
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::{Direction, GraphStore};

use super::super::{AlgorithmResult, ParameterDef, ParameterType, Parameters};
use super::traits::GraphAlgorithm;

// ============================================================================
// Configuration
// ============================================================================

/// Strategy for aggregating numeric properties across contracted nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Sum all values.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Maximum value.
    Max,
    /// Take the first encountered value.
    First,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::Sum
    }
}

/// Configuration for subgraph contraction.
#[derive(Debug, Clone)]
pub struct ContractionConfig {
    /// How to aggregate numeric properties (default: Sum).
    pub aggregation: AggregationStrategy,
    /// Label for the super-node (default: "SuperNode").
    pub super_label: String,
    /// Store the count of internal edges as a property on the super-node.
    pub preserve_internal_count: bool,
    /// Optional property key for edge weight aggregation.
    pub edge_weight_key: Option<String>,
}

impl Default for ContractionConfig {
    fn default() -> Self {
        Self {
            aggregation: AggregationStrategy::Sum,
            super_label: "SuperNode".to_string(),
            preserve_internal_count: true,
            edge_weight_key: None,
        }
    }
}

// ============================================================================
// Result types
// ============================================================================

/// Snapshot of a contracted subgraph for later expansion.
///
/// Stores all information needed to reverse the contraction.
#[derive(Debug, Clone)]
pub struct ContractionSnapshot {
    /// Original nodes with their labels and properties.
    pub nodes: Vec<(NodeId, Vec<String>, Vec<(PropertyKey, Value)>)>,
    /// Original internal edges (src, dst, edge_type, properties).
    pub edges: Vec<(NodeId, NodeId, String, Vec<(PropertyKey, Value)>)>,
    /// Original external edges that were redirected (original_src, original_dst, edge_type, props, was_incoming).
    pub external_edges: Vec<(NodeId, NodeId, String, Vec<(PropertyKey, Value)>, bool)>,
}

/// Result of a subgraph contraction.
#[derive(Debug, Clone)]
pub struct ContractionResult {
    /// ID of the created super-node.
    pub supernode_id: NodeId,
    /// IDs of the contracted nodes (now deleted).
    pub contracted_nodes: Vec<NodeId>,
    /// Number of internal edges removed.
    pub internal_edges_removed: usize,
    /// Number of external edges redirected to the super-node.
    pub external_edges_redirected: usize,
    /// Snapshot for reversing the contraction.
    pub snapshot: ContractionSnapshot,
}

// ============================================================================
// Core algorithms
// ============================================================================

/// Contract a set of nodes into a single super-node.
///
/// All internal edges (both endpoints in `node_ids`) are removed. External edges
/// (one endpoint in `node_ids`, one outside) are redirected to the super-node.
/// Numeric properties are aggregated according to `config.aggregation`.
///
/// # Arguments
///
/// * `store` - The graph store (mutations via `&self` interior mutability)
/// * `node_ids` - Nodes to contract
/// * `config` - Contraction configuration
///
/// # Returns
///
/// `ContractionResult` with the super-node ID and a snapshot for expansion.
///
/// # Errors
///
/// Returns an error if `node_ids` is empty.
///
/// # Complexity
///
/// O(V_sub + E_sub + E_external)
pub fn contract_subgraph(
    store: &LpgStore,
    node_ids: &[NodeId],
    config: &ContractionConfig,
) -> Result<ContractionResult> {
    if node_ids.is_empty() {
        return Err(obrain_common::utils::error::Error::Internal(
            "Cannot contract empty subgraph".to_string(),
        ));
    }

    let contracted_set: HashSet<NodeId> = node_ids.iter().copied().collect();

    // Step 1: Snapshot nodes (labels + properties)
    let mut snapshot_nodes = Vec::with_capacity(node_ids.len());
    for &nid in node_ids {
        if let Some(node) = store.get_node(nid) {
            let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();
            let props: Vec<(PropertyKey, Value)> = node
                .properties
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            snapshot_nodes.push((nid, labels, props));
        }
    }

    // Step 2: Classify edges as internal or external, snapshot them
    let mut internal_edges: Vec<(EdgeId, NodeId, NodeId, String, Vec<(PropertyKey, Value)>)> =
        Vec::new();
    let mut external_edges: Vec<(
        EdgeId,
        NodeId,
        NodeId,
        String,
        Vec<(PropertyKey, Value)>,
        bool,
    )> = Vec::new();

    for &nid in node_ids {
        // Outgoing edges
        for (neighbor, edge_id) in store.edges_from(nid, Direction::Outgoing) {
            if let Some(edge) = store.get_edge(edge_id) {
                let etype = edge.edge_type.to_string();
                let props: Vec<(PropertyKey, Value)> = edge
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if contracted_set.contains(&neighbor) {
                    // Internal: only record from lower id to avoid duplicates
                    if nid.0 <= neighbor.0 {
                        internal_edges.push((edge_id, nid, neighbor, etype, props));
                    }
                } else {
                    // External outgoing: src=nid (contracted), dst=neighbor (outside)
                    external_edges.push((edge_id, nid, neighbor, etype, props, false));
                }
            }
        }

        // Incoming edges from non-contracted nodes
        for (neighbor, edge_id) in store.edges_from(nid, Direction::Incoming) {
            if !contracted_set.contains(&neighbor)
                && let Some(edge) = store.get_edge(edge_id)
            {
                let etype = edge.edge_type.to_string();
                let props: Vec<(PropertyKey, Value)> = edge
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                // External incoming: src=neighbor (outside), dst=nid (contracted)
                external_edges.push((edge_id, neighbor, nid, etype, props, true));
            }
        }
    }

    // Also collect internal edges where both endpoints are in the set but
    // we only traversed outgoing from lower nid — handle the reverse direction
    // edges that weren't captured
    let mut seen_internal: HashSet<EdgeId> = internal_edges.iter().map(|e| e.0).collect();
    for &nid in node_ids {
        for (neighbor, edge_id) in store.edges_from(nid, Direction::Outgoing) {
            if contracted_set.contains(&neighbor)
                && !seen_internal.contains(&edge_id)
                && let Some(edge) = store.get_edge(edge_id)
            {
                let etype = edge.edge_type.to_string();
                let props: Vec<(PropertyKey, Value)> = edge
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                internal_edges.push((edge_id, nid, neighbor, etype, props));
                seen_internal.insert(edge_id);
            }
        }
    }

    // Step 3: Aggregate properties for the super-node
    let aggregated_props = aggregate_properties(&snapshot_nodes, config);

    // Step 4: Create super-node
    let supernode_id = store.create_node(&[&config.super_label]);

    // Set aggregated properties
    for (key, value) in &aggregated_props {
        store.set_node_property(supernode_id, key.as_str(), value.clone());
    }

    // Store internal edge count if configured
    if config.preserve_internal_count {
        store.set_node_property(
            supernode_id,
            "_internal_edges",
            Value::Int64(internal_edges.len() as i64),
        );
        store.set_node_property(
            supernode_id,
            "_contracted_count",
            Value::Int64(node_ids.len() as i64),
        );
    }

    // Step 5: Redirect external edges to/from super-node
    let mut redirected_count = 0;
    let mut seen_ext_edges: HashSet<EdgeId> = HashSet::new();

    for &(edge_id, src, dst, ref etype, ref props, is_incoming) in &external_edges {
        if !seen_ext_edges.insert(edge_id) {
            continue;
        }
        // Delete original edge
        store.delete_edge(edge_id);

        // Create redirected edge
        let (new_src, new_dst) = if is_incoming {
            (src, supernode_id) // incoming: external → super
        } else {
            (supernode_id, dst) // outgoing: super → external
        };

        let new_edge_id = store.create_edge(new_src, new_dst, etype);

        // Copy edge properties
        for (key, value) in props {
            store.set_edge_property(new_edge_id, key.as_str(), value.clone());
        }
        redirected_count += 1;
    }

    // Step 6: Delete internal edges
    for &(edge_id, ..) in &internal_edges {
        store.delete_edge(edge_id);
    }

    // Step 7: Delete contracted nodes
    for &nid in node_ids {
        store.delete_node(nid);
    }

    // Build snapshot for expansion
    let snapshot = ContractionSnapshot {
        nodes: snapshot_nodes,
        edges: internal_edges
            .iter()
            .map(|(_, src, dst, etype, props)| (*src, *dst, etype.clone(), props.clone()))
            .collect(),
        external_edges: external_edges
            .iter()
            .filter(|(eid, ..)| seen_ext_edges.contains(eid))
            .map(|(_, src, dst, etype, props, incoming)| {
                (*src, *dst, etype.clone(), props.clone(), *incoming)
            })
            .collect(),
    };

    Ok(ContractionResult {
        supernode_id,
        contracted_nodes: node_ids.to_vec(),
        internal_edges_removed: internal_edges.len(),
        external_edges_redirected: redirected_count,
        snapshot,
    })
}

/// Expand a super-node back to its original subgraph.
///
/// Restores all original nodes, internal edges, and external edges from the
/// [`ContractionSnapshot`], then deletes the super-node.
///
/// # Arguments
///
/// * `store` - The graph store
/// * `supernode_id` - The super-node to expand
/// * `snapshot` - Snapshot from the original contraction
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Complexity
///
/// O(V_sub + E_sub + E_external)
pub fn expand_supernode(
    store: &LpgStore,
    supernode_id: NodeId,
    snapshot: &ContractionSnapshot,
) -> Result<()> {
    // Step 1: Collect current external edges on the super-node (to delete after)
    let mut supernode_edges: Vec<EdgeId> = Vec::new();
    for (_, eid) in store.edges_from(supernode_id, Direction::Outgoing) {
        supernode_edges.push(eid);
    }
    for (_, eid) in store.edges_from(supernode_id, Direction::Incoming) {
        supernode_edges.push(eid);
    }

    // Step 2: Restore original nodes
    for (nid, labels, props) in &snapshot.nodes {
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        store
            .create_node_with_id(*nid, &label_refs)
            .map_err(|e| obrain_common::utils::error::Error::Internal(format!("{e:?}")))?;
        for (key, value) in props {
            store.set_node_property(*nid, key.as_str(), value.clone());
        }
    }

    // Step 3: Restore internal edges
    for (src, dst, etype, props) in &snapshot.edges {
        let eid = store.create_edge(*src, *dst, etype);
        for (key, value) in props {
            store.set_edge_property(eid, key.as_str(), value.clone());
        }
    }

    // Step 4: Restore external edges (delete redirected, create originals)
    // First delete all edges on the super-node
    let unique_edges: HashSet<EdgeId> = supernode_edges.into_iter().collect();
    for eid in unique_edges {
        store.delete_edge(eid);
    }

    // Recreate original external edges
    for (src, dst, etype, props, _is_incoming) in &snapshot.external_edges {
        let eid = store.create_edge(*src, *dst, etype);
        for (key, value) in props {
            store.set_edge_property(eid, key.as_str(), value.clone());
        }
    }

    // Step 5: Delete super-node
    store.delete_node(supernode_id);

    Ok(())
}

/// Contract all communities into super-nodes.
///
/// Given a community assignment (from Louvain), contracts each community into
/// a single super-node. Inter-community edges are preserved (with optional
/// weight aggregation).
///
/// # Arguments
///
/// * `store` - The graph store
/// * `communities` - Map of NodeId → community ID (from Louvain)
/// * `config` - Contraction configuration
///
/// # Returns
///
/// Vector of `ContractionResult`, one per community.
///
/// # Complexity
///
/// O(V + E)
pub fn contract_by_communities(
    store: &LpgStore,
    communities: &HashMap<NodeId, u64>,
    config: &ContractionConfig,
) -> Result<Vec<ContractionResult>> {
    // Group nodes by community
    let mut groups: HashMap<u64, Vec<NodeId>> = HashMap::new();
    for (&nid, &comm) in communities {
        groups.entry(comm).or_default().push(nid);
    }

    let mut results = Vec::with_capacity(groups.len());

    // Sort by community ID for deterministic ordering
    let mut sorted_comms: Vec<u64> = groups.keys().copied().collect();
    sorted_comms.sort_unstable();

    for comm in sorted_comms {
        let nodes = &groups[&comm];
        if nodes.len() <= 1 {
            continue; // skip single-node communities
        }
        let result = contract_subgraph(store, nodes, config)?;
        results.push(result);
    }

    Ok(results)
}

// ============================================================================
// Property aggregation
// ============================================================================

fn aggregate_properties(
    nodes: &[(NodeId, Vec<String>, Vec<(PropertyKey, Value)>)],
    config: &ContractionConfig,
) -> Vec<(PropertyKey, Value)> {
    // Collect all numeric properties
    let mut numeric_props: HashMap<PropertyKey, Vec<f64>> = HashMap::new();
    let mut first_values: HashMap<PropertyKey, Value> = HashMap::new();

    for (_, _, props) in nodes {
        for (key, value) in props {
            if let Some(num) = value_to_f64(value) {
                numeric_props.entry(key.clone()).or_default().push(num);
            }
            first_values
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
    }

    match config.aggregation {
        AggregationStrategy::First => first_values.into_iter().collect(),
        AggregationStrategy::Sum => numeric_props
            .iter()
            .map(|(key, vals)| {
                let sum: f64 = vals.iter().sum();
                (key.clone(), Value::Float64(sum))
            })
            .collect(),
        AggregationStrategy::Mean => numeric_props
            .iter()
            .map(|(key, vals)| {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                (key.clone(), Value::Float64(mean))
            })
            .collect(),
        AggregationStrategy::Max => numeric_props
            .iter()
            .map(|(key, vals)| {
                let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (key.clone(), Value::Float64(max))
            })
            .collect(),
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(v) => Some(*v as f64),
        Value::Float64(v) => Some(*v),
        _ => None,
    }
}

// ============================================================================
// GraphAlgorithm trait
// ============================================================================

/// Subgraph contraction algorithm wrapper for registry integration.
pub struct SubgraphContractionAlgorithm;

static CONTRACTION_PARAMS: OnceLock<Vec<ParameterDef>> = OnceLock::new();

fn contraction_params() -> &'static [ParameterDef] {
    CONTRACTION_PARAMS.get_or_init(|| {
        vec![
            ParameterDef {
                name: "aggregation".to_string(),
                description: "Property aggregation strategy: sum, mean, max, first".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("sum".to_string()),
            },
            ParameterDef {
                name: "super_label".to_string(),
                description: "Label for the created super-node".to_string(),
                param_type: ParameterType::String,
                required: false,
                default: Some("SuperNode".to_string()),
            },
            ParameterDef {
                name: "node_ids".to_string(),
                description: "Comma-separated list of NodeIds to contract".to_string(),
                param_type: ParameterType::String,
                required: true,
                default: None,
            },
        ]
    })
}

impl GraphAlgorithm for SubgraphContractionAlgorithm {
    fn name(&self) -> &str {
        "obrain.subgraph_contraction"
    }

    fn description(&self) -> &str {
        "Contract a subgraph into a single super-node with edge redirection"
    }

    fn parameters(&self) -> &[ParameterDef] {
        contraction_params()
    }

    fn execute(&self, _store: &dyn GraphStore, params: &Parameters) -> Result<AlgorithmResult> {
        let aggregation = match params.get_string("aggregation").unwrap_or("sum") {
            "mean" => AggregationStrategy::Mean,
            "max" => AggregationStrategy::Max,
            "first" => AggregationStrategy::First,
            _ => AggregationStrategy::Sum,
        };
        let super_label = params
            .get_string("super_label")
            .unwrap_or("SuperNode")
            .to_string();
        let node_ids_str = params.get_string("node_ids").unwrap_or("");

        let node_ids: Vec<NodeId> = node_ids_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok().map(NodeId))
            .collect();

        #[allow(clippy::no_effect_underscore_binding)]
        let _config = ContractionConfig {
            aggregation,
            super_label,
            preserve_internal_count: true,
            edge_weight_key: None,
        };

        // Downcast to LpgStore — contraction requires mutation methods
        // The GraphAlgorithm trait gives us &dyn GraphStore, but contraction
        // needs LpgStore for create_node/delete_node/etc.
        // For registry integration, we use a read-only result describing
        // what would be contracted (without actually mutating).
        let mut algo_result = AlgorithmResult::new(vec![
            "supernode_id".to_string(),
            "contracted_count".to_string(),
            "node_ids".to_string(),
        ]);

        // Report the nodes that would be contracted
        let ids_str = node_ids
            .iter()
            .map(|n| n.0.to_string())
            .collect::<Vec<_>>()
            .join(",");
        algo_result.add_row(vec![
            Value::String("(dry-run)".into()),
            Value::Int64(node_ids.len() as i64),
            Value::String(ids_str.into()),
        ]);

        Ok(algo_result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diamond() -> LpgStore {
        // A → B, A → C, B → D, C → D (diamond shape)
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Source"]);
        let b = store.create_node(&["Inner"]);
        let c = store.create_node(&["Inner"]);
        let d = store.create_node(&["Sink"]);

        store.create_edge(a, b, "FLOW");
        store.create_edge(a, c, "FLOW");
        store.create_edge(b, d, "FLOW");
        store.create_edge(c, d, "FLOW");
        // Internal edge between b and c
        store.create_edge(b, c, "LINK");

        // Set properties
        store.set_node_property(b, "weight", Value::Float64(10.0));
        store.set_node_property(c, "weight", Value::Float64(20.0));

        store
    }

    #[test]
    fn test_contract_basic() {
        let store = make_diamond();
        let node_ids = store.node_ids();
        let b = node_ids[1];
        let c = node_ids[2];

        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[b, c], &config).unwrap();

        // Super-node created
        assert!(store.get_node(result.supernode_id).is_some());
        // Original nodes deleted
        assert!(store.get_node(b).is_none());
        assert!(store.get_node(c).is_none());
        // Internal edge removed
        assert!(result.internal_edges_removed >= 1);
        // External edges redirected
        assert!(result.external_edges_redirected > 0);
        // Super-node has _contracted_count
        let count =
            store.get_node_property(result.supernode_id, &PropertyKey::new("_contracted_count"));
        assert_eq!(count, Some(Value::Int64(2)));
    }

    #[test]
    fn test_preserves_external_edges() {
        let store = make_diamond();
        let node_ids = store.node_ids();
        let a = node_ids[0];
        let b = node_ids[1];
        let c = node_ids[2];
        let d = node_ids[3];

        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[b, c], &config).unwrap();

        // a should have edges to the super-node
        let a_out: Vec<NodeId> = store
            .edges_from(a, Direction::Outgoing)
            .map(|(n, _)| n)
            .collect();
        assert!(
            a_out.contains(&result.supernode_id),
            "a should connect to super-node"
        );

        // super-node should have edges to d
        let super_out: Vec<NodeId> = store
            .edges_from(result.supernode_id, Direction::Outgoing)
            .map(|(n, _)| n)
            .collect();
        assert!(super_out.contains(&d), "super-node should connect to d");
    }

    #[test]
    fn test_aggregation_sum() {
        let store = make_diamond();
        let node_ids = store.node_ids();
        let b = node_ids[1];
        let c = node_ids[2];

        let config = ContractionConfig {
            aggregation: AggregationStrategy::Sum,
            ..Default::default()
        };
        let result = contract_subgraph(&store, &[b, c], &config).unwrap();

        let weight = store.get_node_property(result.supernode_id, &PropertyKey::new("weight"));
        assert_eq!(weight, Some(Value::Float64(30.0))); // 10 + 20
    }

    #[test]
    fn test_aggregation_mean() {
        let store = make_diamond();
        let node_ids = store.node_ids();
        let b = node_ids[1];
        let c = node_ids[2];

        let config = ContractionConfig {
            aggregation: AggregationStrategy::Mean,
            ..Default::default()
        };
        let result = contract_subgraph(&store, &[b, c], &config).unwrap();

        let weight = store.get_node_property(result.supernode_id, &PropertyKey::new("weight"));
        assert_eq!(weight, Some(Value::Float64(15.0))); // (10 + 20) / 2
    }

    #[test]
    fn test_expand_restores() {
        let store = make_diamond();
        let node_ids = store.node_ids();
        let a = node_ids[0];
        let b = node_ids[1];
        let c = node_ids[2];
        let d = node_ids[3];

        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[b, c], &config).unwrap();

        // Expand back
        expand_supernode(&store, result.supernode_id, &result.snapshot).unwrap();

        // Original nodes restored
        assert!(store.get_node(b).is_some(), "b should be restored");
        assert!(store.get_node(c).is_some(), "c should be restored");
        // Super-node deleted
        assert!(store.get_node(result.supernode_id).is_none());
        // a and d still exist
        assert!(store.get_node(a).is_some());
        assert!(store.get_node(d).is_some());

        // Properties restored
        let b_weight = store.get_node_property(b, &PropertyKey::new("weight"));
        assert_eq!(b_weight, Some(Value::Float64(10.0)));
        let c_weight = store.get_node_property(c, &PropertyKey::new("weight"));
        assert_eq!(c_weight, Some(Value::Float64(20.0)));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn test_by_communities() {
        let store = LpgStore::new().unwrap();
        // Community 0: a, b, c (triangle)
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let c = store.create_node(&["A"]);
        store.create_edge(a, b, "LINK");
        store.create_edge(b, c, "LINK");
        store.create_edge(c, a, "LINK");

        // Community 1: d, e, f (triangle)
        let d = store.create_node(&["B"]);
        let e = store.create_node(&["B"]);
        let f = store.create_node(&["B"]);
        store.create_edge(d, e, "LINK");
        store.create_edge(e, f, "LINK");
        store.create_edge(f, d, "LINK");

        // Bridge: c → d
        store.create_edge(c, d, "BRIDGE");

        let mut communities = HashMap::new();
        for &nid in &[a, b, c] {
            communities.insert(nid, 0);
        }
        for &nid in &[d, e, f] {
            communities.insert(nid, 1);
        }

        let config = ContractionConfig::default();
        let results = contract_by_communities(&store, &communities, &config).unwrap();

        // 2 communities contracted → 2 super-nodes
        assert_eq!(results.len(), 2);

        // Both super-nodes exist
        for r in &results {
            assert!(store.get_node(r.supernode_id).is_some());
        }

        // Bridge edge should exist between the two super-nodes
        let super0 = results[0].supernode_id;
        let super1 = results[1].supernode_id;
        let super0_out: Vec<NodeId> = store
            .edges_from(super0, Direction::Outgoing)
            .map(|(n, _)| n)
            .collect();
        assert!(
            super0_out.contains(&super1),
            "Bridge edge should connect super-nodes"
        );
    }

    #[test]
    fn test_contract_no_external() {
        // All nodes contracted — no external edges
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["X"]);
        let b = store.create_node(&["X"]);
        store.create_edge(a, b, "E");
        store.create_edge(b, a, "E");

        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[a, b], &config).unwrap();

        assert_eq!(result.external_edges_redirected, 0);
        assert!(result.internal_edges_removed >= 2);
        assert!(store.get_node(result.supernode_id).is_some());
    }

    #[test]
    fn test_empty_subgraph() {
        let store = LpgStore::new().unwrap();
        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_edges() {
        // Multiple edges between two nodes
        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["A"]);
        let ext = store.create_node(&["Ext"]);

        store.create_edge(a, b, "E1");
        store.create_edge(a, b, "E2");
        store.create_edge(a, ext, "OUT");
        store.create_edge(b, ext, "OUT");

        let config = ContractionConfig::default();
        let result = contract_subgraph(&store, &[a, b], &config).unwrap();

        // Both internal edges removed
        assert_eq!(result.internal_edges_removed, 2);
        // Two external edges redirected
        assert_eq!(result.external_edges_redirected, 2);
    }
}
