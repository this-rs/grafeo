//! Integration tests for the structural consolidation engine.

#![cfg(feature = "consolidation")]

use grafeo_cognitive::consolidation::{
    ConsolidationConfig, ConsolidationEngine, EDGE_DERIVED_FROM,
};
use grafeo_common::types::NodeId;
use grafeo_common::utils::hash::FxHashMap;
use grafeo_core::graph::Direction;
use grafeo_core::LpgStore;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Creates a densely connected cluster of `n` nodes with the given label.
/// Each node connects to every other node in the cluster → strong community.
fn create_cluster(
    store: &LpgStore,
    label: &str,
    n: usize,
    edge_type: &str,
) -> Vec<NodeId> {
    let mut nodes = Vec::with_capacity(n);
    for _ in 0..n {
        nodes.push(store.create_node(&[label]));
    }
    // Fully connect nodes within the cluster
    for i in 0..n {
        for j in 0..n {
            if i != j {
                store.create_edge(nodes[i], nodes[j], edge_type);
            }
        }
    }
    nodes
}

/// Creates a sparse bridge between two clusters (1-2 edges).
fn bridge_clusters(store: &LpgStore, cluster_a: &[NodeId], cluster_b: &[NodeId]) {
    // Single weak link between the last node of A and first node of B
    store.create_edge(cluster_a[cluster_a.len() - 1], cluster_b[0], "BRIDGE");
}

// ---------------------------------------------------------------------------
// Test 1: 20 nodes in 3 clusters → consolidation → 3 condensed nodes + DERIVED_FROM
// ---------------------------------------------------------------------------

#[test]
fn consolidation_merge_three_clusters() {
    let store = LpgStore::new().expect("LpgStore::new");

    // Create 3 clusters of ~7, ~7, ~6 nodes (= 20 total)
    let cluster_a = create_cluster(&store, "Data", 7, "LINKS");
    let cluster_b = create_cluster(&store, "Data", 7, "LINKS");
    let cluster_c = create_cluster(&store, "Data", 6, "LINKS");

    // Add weak bridges so Louvain sees 3 distinct communities
    bridge_clusters(&store, &cluster_a, &cluster_b);
    bridge_clusters(&store, &cluster_b, &cluster_c);

    // All nodes have low energy → candidates for consolidation
    let mut energy_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut pagerank_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    for nodes in [&cluster_a, &cluster_b, &cluster_c] {
        for &n in nodes {
            energy_map.insert(n, 0.1);
            pagerank_map.insert(n, 0.01);
        }
    }

    let engine = ConsolidationEngine::new(
        ConsolidationConfig::default()
            .with_energy_threshold(0.5)
            .with_pagerank_protection(0.5), // High threshold so nobody is protected
    );

    let result = engine.consolidate(&store, &energy_map, &pagerank_map);

    // Should detect 3 communities
    assert!(
        result.communities_detected >= 3,
        "expected >= 3 communities, got {}",
        result.communities_detected
    );

    // Should have created condensed nodes (at least 3)
    assert!(
        result.condensed_nodes.len() >= 3,
        "expected >= 3 condensed nodes, got {}",
        result.condensed_nodes.len()
    );

    // DERIVED_FROM edges should exist
    assert!(
        !result.derived_from_edges.is_empty(),
        "expected DERIVED_FROM edges"
    );

    // Total merged nodes should be 20
    assert_eq!(
        result.nodes_removed, 20,
        "all 20 low-energy nodes should be merged"
    );

    // Verify DERIVED_FROM edges in the graph
    for &edge_id in &result.derived_from_edges {
        let edge = store.get_edge(edge_id);
        assert!(
            edge.is_some(),
            "DERIVED_FROM edge {edge_id:?} should exist in graph"
        );
        if let Some(e) = edge {
            assert_eq!(
                e.edge_type.as_str(),
                EDGE_DERIVED_FROM,
                "edge type should be DERIVED_FROM"
            );
        }
    }

    // Each condensed node should have the "Condensed" label
    for (&condensed_id, originals) in &result.condensed_nodes {
        let node = store.get_node(condensed_id);
        assert!(node.is_some(), "condensed node should exist");
        let node = node.unwrap();
        assert!(
            node.labels.iter().any(|l| l.as_str() == "Condensed"),
            "condensed node should have 'Condensed' label, got: {:?}",
            node.labels
        );
        // Should have DERIVED_FROM edges to all originals
        let derived_targets: Vec<NodeId> = store
            .edges_from(condensed_id, Direction::Outgoing)
            .filter_map(|(target, eid)| {
                store
                    .get_edge(eid)
                    .filter(|e| e.edge_type.as_str() == EDGE_DERIVED_FROM)
                    .map(|_| target)
            })
            .collect();
        assert_eq!(
            derived_targets.len(),
            originals.len(),
            "condensed node should have DERIVED_FROM edge to each original"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: high-PageRank node excluded from merge
// ---------------------------------------------------------------------------

#[test]
fn consolidation_high_pagerank_excluded() {
    let store = LpgStore::new().expect("LpgStore::new");

    // Create a small cluster of 5 nodes
    let cluster = create_cluster(&store, "Module", 5, "DEPENDS");

    let mut energy_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut pagerank_map: FxHashMap<NodeId, f64> = FxHashMap::default();

    // All nodes have low energy
    for &n in &cluster {
        energy_map.insert(n, 0.05);
        pagerank_map.insert(n, 0.01);
    }

    // But node 0 has high PageRank → should be protected
    let protected_node = cluster[0];
    pagerank_map.insert(protected_node, 0.5);

    let config = ConsolidationConfig::default()
        .with_energy_threshold(0.5)
        .with_pagerank_protection(0.1); // threshold at 0.1, node0 has 0.5

    let engine = ConsolidationEngine::new(config);
    let result = engine.consolidate(&store, &energy_map, &pagerank_map);

    // Protected node should be in the protected list
    assert!(
        result.protected_nodes.contains(&protected_node),
        "high-PageRank node should be protected"
    );

    // Protected node should NOT be in any condensed group
    for originals in result.condensed_nodes.values() {
        assert!(
            !originals.contains(&protected_node),
            "protected node should not appear in condensed originals"
        );
    }

    // The protected node should still exist in the graph
    assert!(
        store.get_node(protected_node).is_some(),
        "protected node should still exist in graph"
    );
}

// ---------------------------------------------------------------------------
// Test 3: label_filter only consolidates matching nodes
// ---------------------------------------------------------------------------

#[test]
fn consolidation_label_filter() {
    let store = LpgStore::new().expect("LpgStore::new");

    // Create two clusters with different labels
    let memory_nodes = create_cluster(&store, "Memory", 5, "RECALL");
    let code_nodes = create_cluster(&store, "Code", 5, "IMPORTS");

    // Weak bridge between them
    bridge_clusters(&store, &memory_nodes, &code_nodes);

    let mut energy_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut pagerank_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    for nodes in [&memory_nodes, &code_nodes] {
        for &n in nodes {
            energy_map.insert(n, 0.05);
            pagerank_map.insert(n, 0.01);
        }
    }

    // Only consolidate "Memory" nodes
    let config = ConsolidationConfig::default()
        .with_label_filter(vec!["Memory".into()])
        .with_energy_threshold(0.5)
        .with_pagerank_protection(0.5);

    let engine = ConsolidationEngine::new(config);
    let result = engine.consolidate(&store, &energy_map, &pagerank_map);

    // Only Memory nodes should have been consolidated
    for originals in result.condensed_nodes.values() {
        for &original_id in originals {
            // All original IDs should be from memory_nodes
            assert!(
                memory_nodes.contains(&original_id),
                "only Memory nodes should be consolidated, but found {original_id:?}"
            );
        }
    }

    // All Code nodes should still exist untouched
    for &code_node in &code_nodes {
        assert!(
            store.get_node(code_node).is_some(),
            "Code node {code_node:?} should still exist (not consolidated)"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: synapse edges rewired to condensed node
// ---------------------------------------------------------------------------

#[test]
fn consolidation_synapse_rewire() {
    let store = LpgStore::new().expect("LpgStore::new");

    // Create a cluster of 4 nodes that will be consolidated
    let cluster = create_cluster(&store, "Concept", 4, "RELATED");

    // Create an external node that connects to/from the cluster
    let external = store.create_node(&["External"]);
    store.create_edge(external, cluster[0], "SYNAPSE");
    store.create_edge(cluster[3], external, "SYNAPSE");

    let mut energy_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    let mut pagerank_map: FxHashMap<NodeId, f64> = FxHashMap::default();
    for &n in &cluster {
        energy_map.insert(n, 0.05);
        pagerank_map.insert(n, 0.01);
    }
    // External node has high energy (won't be consolidated)
    energy_map.insert(external, 5.0);
    pagerank_map.insert(external, 0.01);

    let config = ConsolidationConfig::default()
        .with_energy_threshold(0.5)
        .with_pagerank_protection(0.5);

    let engine = ConsolidationEngine::new(config);
    let result = engine.consolidate(&store, &energy_map, &pagerank_map);

    // Cluster should be consolidated
    assert!(
        !result.condensed_nodes.is_empty(),
        "cluster should produce a condensed node"
    );

    // External node should still exist
    assert!(
        store.get_node(external).is_some(),
        "external node should still exist"
    );

    // The condensed node should have edges to/from external
    let condensed_id = *result.condensed_nodes.keys().next().unwrap();

    // Check that external → condensed SYNAPSE edge exists (rewired from external → cluster[0])
    let outgoing_from_external: Vec<(NodeId, _)> = store
        .edges_from(external, Direction::Outgoing)
        .collect();
    let has_synapse_to_condensed = outgoing_from_external.iter().any(|(target, eid)| {
        *target == condensed_id
            && store
                .get_edge(*eid)
                .map_or(false, |e| e.edge_type.as_str() == "SYNAPSE")
    });
    assert!(
        has_synapse_to_condensed,
        "external → condensed SYNAPSE edge should exist (rewired)"
    );

    // Check that condensed → external SYNAPSE edge exists (rewired from cluster[3] → external)
    let outgoing_from_condensed: Vec<(NodeId, _)> = store
        .edges_from(condensed_id, Direction::Outgoing)
        .collect();
    let has_synapse_from_condensed = outgoing_from_condensed.iter().any(|(target, eid)| {
        *target == external
            && store
                .get_edge(*eid)
                .map_or(false, |e| e.edge_type.as_str() == "SYNAPSE")
    });
    assert!(
        has_synapse_from_condensed,
        "condensed → external SYNAPSE edge should exist (rewired)"
    );
}
