//! Structural Fingerprinting — local topology signatures for twin detection.
//!
//! Computes a compact fingerprint of a graph's structure from its adjacency
//! list, enabling cross-graph comparison and structural twin detection.
//!
//! The fingerprint captures:
//! - **Degree distribution** — histogram of node degrees
//! - **Clustering coefficient** — global transitivity measure
//! - **Motif counts** — triangles, 3-stars, and 3-paths

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Motif types
// ---------------------------------------------------------------------------

/// Types of structural motifs counted in a fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotifType {
    /// A triangle (3-clique): three mutually connected nodes.
    Triangle,
    /// A 3-star: one hub node connected to at least 3 neighbors.
    Star3,
    /// A path of length 3 (3 edges, 4 nodes). Approximated.
    Path3,
}

// ---------------------------------------------------------------------------
// Structural fingerprint
// ---------------------------------------------------------------------------

/// A compact structural fingerprint of a graph.
///
/// Captures degree distribution, clustering coefficient, and motif counts
/// to enable fast cross-graph similarity comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralFingerprint {
    /// Degree histogram: index = degree, value = number of nodes with that degree.
    pub degree_distribution: Vec<u32>,
    /// Global clustering coefficient (transitivity).
    pub clustering_coeff: f64,
    /// Counts of each motif type.
    pub motif_counts: HashMap<MotifType, u32>,
    /// Total number of nodes.
    pub node_count: usize,
    /// Total number of edges (undirected).
    pub edge_count: usize,
}

// ---------------------------------------------------------------------------
// Fingerprint computation
// ---------------------------------------------------------------------------

/// Compute a [`StructuralFingerprint`] from an adjacency list.
///
/// The adjacency list maps each node ID to the list of its neighbor IDs.
/// The graph is treated as undirected — each edge should appear in both
/// directions in the adjacency map.
pub fn fingerprint(adjacency: &HashMap<u64, Vec<u64>>) -> StructuralFingerprint {
    let node_count = adjacency.len();

    // -- Degree distribution --
    let max_degree: usize = adjacency.values().map(|n| n.len()).max().unwrap_or(0);
    let mut degree_distribution = if node_count == 0 {
        Vec::new()
    } else {
        vec![0u32; max_degree + 1]
    };
    for neighbors in adjacency.values() {
        degree_distribution[neighbors.len()] += 1;
    }

    // -- Edge count (undirected: each edge counted twice) --
    let total_half_edges: usize = adjacency.values().map(|v| v.len()).sum();
    let edge_count = total_half_edges / 2;

    // Build neighbor sets for fast lookup
    let neighbor_sets: HashMap<u64, std::collections::HashSet<u64>> = adjacency
        .iter()
        .map(|(&node, neighbors)| (node, neighbors.iter().copied().collect()))
        .collect();

    // -- Triangles & clustering coefficient --
    // Count each triangle once by requiring u < v < w.
    let mut triangle_count: u32 = 0;
    let mut total_possible: u64 = 0;

    for (&node, neighbors) in adjacency {
        let deg = neighbors.len();
        if deg >= 2 {
            // Possible triangles centered at this node: C(deg, 2)
            total_possible += (deg as u64) * (deg as u64 - 1) / 2;

            // Count actual triangles: pairs of neighbors that are connected
            let neigh = &neighbor_sets[&node];
            let neigh_vec: Vec<u64> = neighbors.iter().copied().filter(|&n| n > node).collect();
            for &v in &neigh_vec {
                if let Some(v_neigh) = neighbor_sets.get(&v) {
                    for &w in &neigh_vec {
                        if w > v && v_neigh.contains(&w) && neigh.contains(&w) {
                            triangle_count += 1;
                        }
                    }
                }
            }
        }
    }

    // Global clustering coefficient: ratio of closed triplets to all triplets
    // Each triangle contributes 1 closed triplet per vertex (3 total),
    // but we counted each triangle once. Triplets centered at each node = C(deg,2).
    // clustering = 3 * triangles / sum_of_C(deg,2)
    let clustering_coeff = if total_possible > 0 {
        (3.0 * f64::from(triangle_count)) / total_possible as f64
    } else {
        0.0
    };

    // -- Motif counts --
    let mut motif_counts = HashMap::new();
    motif_counts.insert(MotifType::Triangle, triangle_count);

    // Star3: count nodes with degree >= 3, each contributes C(deg, 3) star-3 instances
    let star3_count: u32 = adjacency
        .values()
        .filter(|n| n.len() >= 3)
        .map(|n| {
            let d = n.len() as u64;
            // C(d, 3) = d*(d-1)*(d-2)/6, capped to u32
            (d * (d - 1) * (d - 2) / 6) as u32
        })
        .sum();
    motif_counts.insert(MotifType::Star3, star3_count);

    // Path3: approximate 3-length paths (3 edges).
    // A path of length 3 is u-v-w-x where all 4 nodes are distinct.
    // Approximation: for each edge (v, w), count (deg(v)-1) * (deg(w)-1),
    // then subtract triangle-based over-counting (each triangle contributes
    // paths that are not simple length-3 paths). Divide by 2 for symmetry.
    let mut path3_approx: u64 = 0;
    for (&v, neighbors) in adjacency {
        let dv = neighbors.len() as u64;
        for &w in neighbors {
            if w > v {
                let dw = adjacency.get(&w).map_or(0, |n| n.len()) as u64;
                if dv > 0 && dw > 0 {
                    path3_approx += (dv - 1) * (dw - 1);
                }
            }
        }
    }
    // Subtract overcounting from triangles: each triangle creates 3 false path-3 counts
    path3_approx = path3_approx.saturating_sub(3 * u64::from(triangle_count));
    motif_counts.insert(MotifType::Path3, path3_approx as u32);

    StructuralFingerprint {
        degree_distribution,
        clustering_coeff,
        motif_counts,
        node_count,
        edge_count,
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Compare two structural fingerprints, returning a similarity score in `[0, 1]`.
///
/// The score is a weighted average of:
/// - **Degree distribution similarity** (0.4): Wasserstein-like L1 distance, normalized
/// - **Clustering coefficient similarity** (0.3): `1.0 - |c1 - c2|`
/// - **Motif cosine similarity** (0.3)
pub fn compare(fp1: &StructuralFingerprint, fp2: &StructuralFingerprint) -> f64 {
    let degree_sim =
        degree_distribution_similarity(&fp1.degree_distribution, &fp2.degree_distribution);
    let clustering_sim = 1.0 - (fp1.clustering_coeff - fp2.clustering_coeff).abs();
    let motif_sim = motif_cosine_similarity(&fp1.motif_counts, &fp2.motif_counts);

    0.4 * degree_sim + 0.3 * clustering_sim + 0.3 * motif_sim
}

/// Wasserstein-like (L1) similarity on normalized degree distributions.
fn degree_distribution_similarity(d1: &[u32], d2: &[u32]) -> f64 {
    let len = d1.len().max(d2.len());
    if len == 0 {
        return 1.0;
    }

    let sum1: f64 = d1.iter().map(|&x| f64::from(x)).sum();
    let sum2: f64 = d2.iter().map(|&x| f64::from(x)).sum();

    if sum1 == 0.0 && sum2 == 0.0 {
        return 1.0;
    }

    let mut l1_distance = 0.0;
    for i in 0..len {
        let v1 = if i < d1.len() {
            f64::from(d1[i]) / sum1.max(1.0)
        } else {
            0.0
        };
        let v2 = if i < d2.len() {
            f64::from(d2[i]) / sum2.max(1.0)
        } else {
            0.0
        };
        l1_distance += (v1 - v2).abs();
    }

    // L1 distance of two probability distributions is in [0, 2]
    1.0 - (l1_distance / 2.0)
}

/// Cosine similarity of motif count vectors.
fn motif_cosine_similarity(m1: &HashMap<MotifType, u32>, m2: &HashMap<MotifType, u32>) -> f64 {
    let all_types = [MotifType::Triangle, MotifType::Star3, MotifType::Path3];

    let mut dot = 0.0_f64;
    let mut mag1 = 0.0_f64;
    let mut mag2 = 0.0_f64;

    for mt in &all_types {
        let v1 = f64::from(*m1.get(mt).unwrap_or(&0));
        let v2 = f64::from(*m2.get(mt).unwrap_or(&0));
        dot += v1 * v2;
        mag1 += v1 * v1;
        mag2 += v2 * v2;
    }

    let denom = mag1.sqrt() * mag2.sqrt();
    if denom == 0.0 {
        // Both zero vectors — structurally identical (both empty)
        return 1.0;
    }

    dot / denom
}

// ---------------------------------------------------------------------------
// Twin detection
// ---------------------------------------------------------------------------

/// Detect structural twins: pairs of fingerprints with similarity >= `threshold`.
///
/// Returns a list of `(id_a, id_b, similarity)` tuples where `id_a < id_b`.
pub fn detect_twins(
    fingerprints: &[(u64, StructuralFingerprint)],
    threshold: f64,
) -> Vec<(u64, u64, f64)> {
    let mut twins = Vec::new();

    for i in 0..fingerprints.len() {
        for j in (i + 1)..fingerprints.len() {
            let sim = compare(&fingerprints[i].1, &fingerprints[j].1);
            if sim >= threshold {
                let (a, b) = if fingerprints[i].0 < fingerprints[j].0 {
                    (fingerprints[i].0, fingerprints[j].0)
                } else {
                    (fingerprints[j].0, fingerprints[i].0)
                };
                twins.push((a, b, sim));
            }
        }
    }

    twins
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a complete graph K_n (undirected adjacency list).
    fn complete_graph(n: u64) -> HashMap<u64, Vec<u64>> {
        let mut adj = HashMap::new();
        for i in 0..n {
            let neighbors: Vec<u64> = (0..n).filter(|&j| j != i).collect();
            adj.insert(i, neighbors);
        }
        adj
    }

    #[test]
    fn self_similarity_is_one() {
        let adj = complete_graph(5);
        let fp = fingerprint(&adj);
        let sim = compare(&fp, &fp);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "compare(fp, fp) should be 1.0, got {sim}"
        );
    }

    #[test]
    fn empty_graph_self_similarity() {
        let adj: HashMap<u64, Vec<u64>> = HashMap::new();
        let fp = fingerprint(&adj);
        assert_eq!(fp.node_count, 0);
        assert_eq!(fp.edge_count, 0);
        let sim = compare(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-10);
    }
}
