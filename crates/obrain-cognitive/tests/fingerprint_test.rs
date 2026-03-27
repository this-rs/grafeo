//! Integration tests for the structural fingerprinting subsystem.

#![cfg(feature = "fingerprint")]

use obrain_cognitive::fingerprint::{MotifType, compare, detect_twins, fingerprint};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a complete graph K_n (undirected adjacency list).
fn complete_graph(n: u64) -> HashMap<u64, Vec<u64>> {
    let mut adj = HashMap::new();
    for i in 0..n {
        let neighbors: Vec<u64> = (0..n).filter(|&j| j != i).collect();
        adj.insert(i, neighbors);
    }
    adj
}

/// Build a star graph with one hub (node 0) and `leaves` leaf nodes.
fn star_graph(leaves: u64) -> HashMap<u64, Vec<u64>> {
    let mut adj = HashMap::new();
    // Hub connects to all leaves
    adj.insert(0, (1..=leaves).collect());
    // Each leaf connects only to hub
    for i in 1..=leaves {
        adj.insert(i, vec![0]);
    }
    adj
}

/// Build a path graph: 0-1-2-..-(n-1).
fn path_graph(n: u64) -> HashMap<u64, Vec<u64>> {
    let mut adj = HashMap::new();
    for i in 0..n {
        let mut neighbors = Vec::new();
        if i > 0 {
            neighbors.push(i - 1);
        }
        if i < n - 1 {
            neighbors.push(i + 1);
        }
        adj.insert(i, neighbors);
    }
    adj
}

// ---------------------------------------------------------------------------
// Fingerprint of complete graph K5
// ---------------------------------------------------------------------------

#[test]
fn complete_graph_k5_clustering_coeff() {
    let adj = complete_graph(5);
    let fp = fingerprint(&adj);

    assert_eq!(fp.node_count, 5);
    assert_eq!(fp.edge_count, 10);
    assert!(
        (fp.clustering_coeff - 1.0).abs() < 1e-10,
        "K5 clustering coefficient should be 1.0, got {}",
        fp.clustering_coeff
    );
}

#[test]
fn complete_graph_k5_triangles() {
    let adj = complete_graph(5);
    let fp = fingerprint(&adj);

    // K5 has C(5,3) = 10 triangles
    assert_eq!(
        fp.motif_counts.get(&MotifType::Triangle),
        Some(&10),
        "K5 should have 10 triangles"
    );
}

// ---------------------------------------------------------------------------
// Fingerprint of star graph
// ---------------------------------------------------------------------------

#[test]
fn star_graph_clustering_coeff() {
    let adj = star_graph(5);
    let fp = fingerprint(&adj);

    assert_eq!(fp.node_count, 6); // hub + 5 leaves
    assert_eq!(fp.edge_count, 5);
    assert!(
        fp.clustering_coeff.abs() < 1e-10,
        "Star graph clustering coefficient should be 0.0, got {}",
        fp.clustering_coeff
    );
}

#[test]
fn star_graph_no_triangles() {
    let adj = star_graph(5);
    let fp = fingerprint(&adj);

    assert_eq!(
        fp.motif_counts.get(&MotifType::Triangle),
        Some(&0),
        "Star graph should have 0 triangles"
    );
}

#[test]
fn star_graph_star3_motifs() {
    let adj = star_graph(5);
    let fp = fingerprint(&adj);

    // Hub has degree 5 → C(5,3) = 10 star-3 motifs
    assert_eq!(
        fp.motif_counts.get(&MotifType::Star3),
        Some(&10),
        "Star-5 hub should produce C(5,3) = 10 star-3 motifs"
    );
}

// ---------------------------------------------------------------------------
// Degree distribution correctness
// ---------------------------------------------------------------------------

#[test]
fn degree_distribution_complete_graph() {
    let adj = complete_graph(5);
    let fp = fingerprint(&adj);

    // All 5 nodes have degree 4
    assert_eq!(fp.degree_distribution.len(), 5); // indices 0..4
    assert_eq!(fp.degree_distribution[4], 5);
    // No nodes with degree 0, 1, 2, 3
    for d in 0..4 {
        assert_eq!(fp.degree_distribution[d], 0);
    }
}

#[test]
fn degree_distribution_star_graph() {
    let adj = star_graph(4);
    let fp = fingerprint(&adj);

    // Hub has degree 4, 4 leaves have degree 1
    assert_eq!(fp.degree_distribution.len(), 5); // indices 0..4
    assert_eq!(fp.degree_distribution[1], 4, "4 leaves with degree 1");
    assert_eq!(fp.degree_distribution[4], 1, "1 hub with degree 4");
    assert_eq!(fp.degree_distribution[0], 0);
    assert_eq!(fp.degree_distribution[2], 0);
    assert_eq!(fp.degree_distribution[3], 0);
}

#[test]
fn degree_distribution_path_graph() {
    let adj = path_graph(5);
    let fp = fingerprint(&adj);

    // 0-1-2-3-4: nodes 0,4 have degree 1; nodes 1,2,3 have degree 2
    assert_eq!(fp.degree_distribution.len(), 3); // indices 0..2
    assert_eq!(fp.degree_distribution[0], 0);
    assert_eq!(fp.degree_distribution[1], 2, "2 endpoints with degree 1");
    assert_eq!(
        fp.degree_distribution[2], 3,
        "3 interior nodes with degree 2"
    );
}

// ---------------------------------------------------------------------------
// Self-similarity
// ---------------------------------------------------------------------------

#[test]
fn compare_self_similarity_is_one() {
    let adj = complete_graph(5);
    let fp = fingerprint(&adj);
    let sim = compare(&fp, &fp);
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "compare(fp, fp) should be exactly 1.0, got {sim}"
    );
}

#[test]
fn compare_self_similarity_star() {
    let adj = star_graph(6);
    let fp = fingerprint(&adj);
    let sim = compare(&fp, &fp);
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "compare(fp, fp) should be exactly 1.0 for star, got {sim}"
    );
}

#[test]
fn compare_self_similarity_path() {
    let adj = path_graph(10);
    let fp = fingerprint(&adj);
    let sim = compare(&fp, &fp);
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "compare(fp, fp) should be exactly 1.0 for path, got {sim}"
    );
}

// ---------------------------------------------------------------------------
// Cross-graph comparison
// ---------------------------------------------------------------------------

#[test]
fn compare_complete_vs_star_is_low() {
    let fp_complete = fingerprint(&complete_graph(5));
    let fp_star = fingerprint(&star_graph(5));
    let sim = compare(&fp_complete, &fp_star);
    assert!(
        sim < 0.5,
        "Complete graph vs star graph similarity should be < 0.5, got {sim}"
    );
}

// ---------------------------------------------------------------------------
// Twin detection
// ---------------------------------------------------------------------------

#[test]
fn detect_twins_isomorphic_subgraphs() {
    // Two complete-K4 graphs should be detected as twins
    let fp1 = fingerprint(&complete_graph(4));
    let fp2 = fingerprint(&complete_graph(4));

    let fingerprints = vec![(1_u64, fp1), (2_u64, fp2)];
    let twins = detect_twins(&fingerprints, 0.99);

    assert_eq!(twins.len(), 1, "Should detect exactly one twin pair");
    assert_eq!(twins[0].0, 1);
    assert_eq!(twins[0].1, 2);
    assert!(
        (twins[0].2 - 1.0).abs() < 1e-10,
        "Isomorphic graphs should have similarity 1.0"
    );
}

#[test]
fn detect_twins_mixed_graphs() {
    let fp_k4_a = fingerprint(&complete_graph(4));
    let fp_k4_b = fingerprint(&complete_graph(4));
    let fp_star = fingerprint(&star_graph(5));

    let fingerprints = vec![(10_u64, fp_k4_a), (20_u64, fp_star), (30_u64, fp_k4_b)];

    // High threshold: only isomorphic pairs
    let twins = detect_twins(&fingerprints, 0.99);
    assert_eq!(
        twins.len(),
        1,
        "Only the two K4 graphs should be twins at threshold 0.99"
    );
    assert_eq!(twins[0].0, 10);
    assert_eq!(twins[0].1, 30);
}

#[test]
fn detect_twins_below_threshold_returns_empty() {
    let fp_complete = fingerprint(&complete_graph(5));
    let fp_star = fingerprint(&star_graph(5));

    let fingerprints = vec![(1_u64, fp_complete), (2_u64, fp_star)];
    let twins = detect_twins(&fingerprints, 0.99);

    assert!(
        twins.is_empty(),
        "Dissimilar graphs should not be detected as twins"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_graph_fingerprint() {
    let adj: HashMap<u64, Vec<u64>> = HashMap::new();
    let fp = fingerprint(&adj);

    assert_eq!(fp.node_count, 0);
    assert_eq!(fp.edge_count, 0);
    assert!(fp.degree_distribution.is_empty());
    assert!((fp.clustering_coeff - 0.0).abs() < 1e-10);
}

#[test]
fn single_node_fingerprint() {
    let mut adj = HashMap::new();
    adj.insert(0_u64, vec![]);
    let fp = fingerprint(&adj);

    assert_eq!(fp.node_count, 1);
    assert_eq!(fp.edge_count, 0);
    assert_eq!(fp.degree_distribution, vec![1]); // one node with degree 0
    assert!((fp.clustering_coeff - 0.0).abs() < 1e-10);
}

#[test]
fn compare_empty_graphs_is_one() {
    let adj: HashMap<u64, Vec<u64>> = HashMap::new();
    let fp = fingerprint(&adj);
    let sim = compare(&fp, &fp);
    assert!((sim - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// P2P use case: before/after injection diff
// ---------------------------------------------------------------------------

#[test]
fn p2p_injection_quality_evaluation() {
    // Simulate P2P distillation: source graph is K5, target starts as star-5
    let source_fp = fingerprint(&complete_graph(5));
    let target_before_fp = fingerprint(&star_graph(4));

    // After injection, target becomes more like K5 (add cross-edges between leaves)
    let mut target_after = star_graph(4);
    // Add edges 1-2, 2-3, 3-4, 1-3, 1-4, 2-4 (making it a complete graph K5)
    for i in 1..=4u64 {
        for j in (i + 1)..=4u64 {
            target_after.get_mut(&i).unwrap().push(j);
            target_after.get_mut(&j).unwrap().push(i);
        }
    }
    let target_after_fp = fingerprint(&target_after);

    let sim_before = compare(&source_fp, &target_before_fp);
    let sim_after = compare(&source_fp, &target_after_fp);

    assert!(
        sim_after > sim_before,
        "After injection, target should be more similar to source: before={sim_before}, after={sim_after}"
    );
    assert!(
        (sim_after - 1.0).abs() < 1e-10,
        "After full injection (both K5), similarity should be 1.0, got {sim_after}"
    );
}

#[test]
fn p2p_partial_injection_improvement() {
    // Source: K6 (complete graph with 6 nodes)
    let source_fp = fingerprint(&complete_graph(6));

    // Target before: path graph (very different topology)
    let target_before_fp = fingerprint(&path_graph(6));

    // Target after partial injection: add some triangles to path graph
    let mut partial = path_graph(6);
    // Add edges 0-2 and 3-5 to create two triangles
    partial.get_mut(&0).unwrap().push(2);
    partial.get_mut(&2).unwrap().push(0);
    partial.get_mut(&3).unwrap().push(5);
    partial.get_mut(&5).unwrap().push(3);
    let target_after_fp = fingerprint(&partial);

    let sim_before = compare(&source_fp, &target_before_fp);
    let sim_after = compare(&source_fp, &target_after_fp);

    // Adding structure should move us closer to the fully-connected source
    assert!(
        sim_after > sim_before,
        "Partial injection should improve similarity: before={sim_before}, after={sim_after}"
    );
    // But still far from perfect
    assert!(
        sim_after < 0.8,
        "Partial injection should not make it fully similar: got {sim_after}"
    );
}

#[test]
fn detect_twins_with_relabeled_graphs() {
    // Two K4 graphs with different node IDs should be twins
    let k4_a = complete_graph(4); // nodes 0,1,2,3
    let mut k4_b = HashMap::new(); // nodes 100,101,102,103
    for i in 100..104u64 {
        let neighbors: Vec<u64> = (100..104).filter(|&j| j != i).collect();
        k4_b.insert(i, neighbors);
    }

    let fp_a = fingerprint(&k4_a);
    let fp_b = fingerprint(&k4_b);
    let fp_star = fingerprint(&star_graph(10));

    let all = vec![(1u64, fp_a), (2u64, fp_b), (3u64, fp_star)];
    let twins = detect_twins(&all, 0.95);

    assert_eq!(twins.len(), 1, "Only the two K4 should be twins");
    assert_eq!(twins[0].0, 1);
    assert_eq!(twins[0].1, 2);
}
