//! C6 — Topological contrastive loss for ProjectionNet training.
//!
//! Core idea: nodes connected in the graph should have more similar projected
//! embeddings than unconnected nodes. Uses InfoNCE loss with hard negatives.
//!
//! Loss components:
//!   L = L_reconstruction (MSE + cosine) + λ_topo · L_contrastive (InfoNCE)
//!
//! InfoNCE for a (anchor, positive) pair with K negatives:
//!   L_InfoNCE = -log( exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + Σ exp(sim(a,nk)/τ)) )

use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

/// A contrastive training sample: anchor node + positive (connected) + negatives (unconnected).
#[derive(Debug, Clone)]
pub struct ContrastiveSample {
    /// Anchor node ID
    pub anchor: NodeId,
    /// Positive node (1-hop neighbor of anchor)
    pub positive: NodeId,
    /// Negative nodes (not connected to anchor within 2 hops)
    pub negatives: Vec<NodeId>,
}

/// Adjacency information extracted from a graph for contrastive pair generation.
pub struct GraphTopology {
    /// node → set of 1-hop neighbors
    pub neighbors_1hop: HashMap<NodeId, HashSet<NodeId>>,
    /// node → set of 2-hop neighbors (includes 1-hop)
    pub neighbors_2hop: HashMap<NodeId, HashSet<NodeId>>,
    /// All node IDs in the graph
    pub all_nodes: Vec<NodeId>,
}

impl GraphTopology {
    /// Build topology from an adjacency list.
    ///
    /// `edges`: list of (src, dst) pairs (will be treated as bidirectional)
    pub fn from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> Self {
        let mut neighbors_1hop: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        // Init empty sets for all nodes
        for &nid in nodes {
            neighbors_1hop.entry(nid).or_default();
        }

        // Build 1-hop (bidirectional)
        for &(a, b) in edges {
            neighbors_1hop.entry(a).or_default().insert(b);
            neighbors_1hop.entry(b).or_default().insert(a);
        }

        // Build 2-hop
        let mut neighbors_2hop: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for (&nid, hop1) in &neighbors_1hop {
            let mut hop2 = hop1.clone();
            for &n1 in hop1 {
                if let Some(n1_neighbors) = neighbors_1hop.get(&n1) {
                    hop2.extend(n1_neighbors);
                }
            }
            hop2.remove(&nid); // don't include self
            neighbors_2hop.insert(nid, hop2);
        }

        Self {
            neighbors_1hop,
            neighbors_2hop,
            all_nodes: nodes.to_vec(),
        }
    }

    /// Generate contrastive samples for training.
    ///
    /// For each node with ≥1 neighbor, creates a sample with:
    /// - anchor = the node
    /// - positive = a random neighbor (deterministic via hash)
    /// - negatives = up to `n_negatives` nodes NOT in the 2-hop neighborhood
    ///
    /// `available_nodes`: nodes that have embeddings (filter for nodes with both text + GNN embeddings)
    pub fn generate_samples(
        &self,
        available_nodes: &HashSet<NodeId>,
        n_negatives: usize,
        seed: u64,
    ) -> Vec<ContrastiveSample> {
        let mut samples = Vec::new();
        let available_vec: Vec<NodeId> = available_nodes.iter().copied().collect();

        if available_vec.len() < 3 {
            return samples; // Need at least anchor + positive + 1 negative
        }

        for &anchor in &available_vec {
            // Get neighbors that also have embeddings
            let neighbors: Vec<NodeId> = self
                .neighbors_1hop
                .get(&anchor)
                .map(|ns| {
                    ns.iter()
                        .filter(|n| available_nodes.contains(n))
                        .copied()
                        .collect()
                })
                .unwrap_or_default();

            if neighbors.is_empty() {
                continue; // Skip isolated nodes
            }

            // Deterministic "random" positive selection
            let pos_idx = (anchor.as_u64().wrapping_mul(seed) as usize) % neighbors.len();
            let positive = neighbors[pos_idx];

            // Negatives: nodes NOT in 2-hop neighborhood
            let hop2 = self.neighbors_2hop.get(&anchor);
            let mut negatives = Vec::new();
            let mut neg_seed = anchor.as_u64().wrapping_mul(2654435761);

            // Shuffle available nodes deterministically and pick negatives
            let mut candidates: Vec<NodeId> = available_vec
                .iter()
                .filter(|&&n| {
                    n != anchor
                        && n != positive
                        && !hop2.map_or(false, |h| h.contains(&n))
                        && !self
                            .neighbors_1hop
                            .get(&anchor)
                            .map_or(false, |h| h.contains(&n))
                })
                .copied()
                .collect();

            // Deterministic shuffle via xorshift on indices
            for i in (1..candidates.len()).rev() {
                neg_seed ^= neg_seed << 13;
                neg_seed ^= neg_seed >> 7;
                neg_seed ^= neg_seed << 17;
                let j = (neg_seed as usize) % (i + 1);
                candidates.swap(i, j);
            }

            negatives.extend(candidates.iter().take(n_negatives));

            if !negatives.is_empty() {
                samples.push(ContrastiveSample {
                    anchor,
                    positive,
                    negatives,
                });
            }
        }

        samples
    }
}

/// InfoNCE loss for one sample given pre-computed cosine similarities.
///
/// Returns (loss, d_sim_pos, d_sim_negatives) — gradients w.r.t. similarities.
///
/// L = -log( exp(sim_pos/τ) / (exp(sim_pos/τ) + Σ exp(sim_neg_k/τ)) )
pub fn info_nce_loss(
    sim_pos: f32,
    sim_negatives: &[f32],
    temperature: f32,
) -> (f32, f32, Vec<f32>) {
    let tau = temperature.max(0.01); // safety clamp

    let exp_pos = (sim_pos / tau).exp();
    let exp_negs: Vec<f32> = sim_negatives.iter().map(|&s| (s / tau).exp()).collect();
    let denom = exp_pos + exp_negs.iter().sum::<f32>();

    // Loss = -log(exp_pos / denom) = -sim_pos/τ + log(denom)
    let loss = -(sim_pos / tau) + denom.ln();

    // Gradient w.r.t. sim_pos: -1/τ + (exp_pos/denom)/τ = (1/τ)(exp_pos/denom - 1)
    let softmax_pos = exp_pos / denom;
    let d_sim_pos = (softmax_pos - 1.0) / tau;

    // Gradient w.r.t. sim_neg_k: (exp_neg_k/denom)/τ
    let d_sim_negs: Vec<f32> = exp_negs.iter().map(|&e| (e / denom) / tau).collect();

    (loss, d_sim_pos, d_sim_negs)
}

/// Compute cosine similarity between two vectors.
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    dot / (na * nb)
}

/// Gradient of cosine similarity w.r.t. vector `a`: d cos(a,b) / d a
///
/// d cos / d a_i = b_i/(|a|·|b|) - a_i · dot(a,b) / (|a|³·|b|)
pub fn cosine_sim_grad_a(a: &[f32], b: &[f32]) -> Vec<f32> {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let na3 = na * na * na;

    a.iter()
        .enumerate()
        .map(|(i, &ai)| b[i] / (na * nb) - ai * dot / (na3 * nb))
        .collect()
}

/// Contrastive training configuration.
#[derive(Debug, Clone)]
pub struct ContrastiveConfig {
    /// Weight for topological contrastive loss (λ_topo). Default: 0.5
    pub lambda_topo: f32,
    /// InfoNCE temperature τ. Default: 0.07
    pub temperature: f32,
    /// Number of negatives per sample. Default: 5
    pub n_negatives: usize,
    /// Cosine margin for the margin component. Default: 0.2
    pub cosine_margin: f32,
    /// Weight for cosine margin loss. Default: 0.1
    pub lambda_margin: f32,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            lambda_topo: 0.5,
            temperature: 0.07,
            n_negatives: 5,
            cosine_margin: 0.2,
            lambda_margin: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: u64) -> NodeId {
        NodeId::from(id)
    }

    #[test]
    fn test_info_nce_perfect_separation() {
        // Positive very similar, negatives very different
        let (loss, d_pos, d_negs) = info_nce_loss(0.95, &[-0.1, -0.2, -0.3], 0.07);
        // Loss should be small (good separation)
        assert!(loss < 1.0, "loss should be small: {}", loss);
        // Gradient on positive should be small (already good)
        assert!(d_pos.abs() < 0.5, "d_pos should be small: {}", d_pos);
        // Gradients on negatives should be near zero (already repelled)
        for (i, &d) in d_negs.iter().enumerate() {
            assert!(d.abs() < 0.5, "d_neg[{}] should be small: {}", i, d);
        }
    }

    #[test]
    fn test_info_nce_no_separation() {
        // All similarities equal — model confused
        let (loss, d_pos, d_negs) = info_nce_loss(0.5, &[0.5, 0.5, 0.5], 0.07);
        // Loss should be -log(1/4) = log(4) ≈ 1.386
        assert!(
            (loss - (4.0f32).ln()).abs() < 0.1,
            "loss ≈ log(4): {}",
            loss
        );
        // Gradient on positive should push it up
        assert!(
            d_pos < 0.0,
            "d_pos should be negative (push sim up): {}",
            d_pos
        );
    }

    #[test]
    fn test_cosine_sim_identity() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_sim(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_topology_generation() {
        let nodes: Vec<NodeId> = (0..6).map(make_node).collect();
        // Graph: 0-1, 0-2, 1-3, 4-5 (4,5 disconnected from 0,1,2,3)
        let edges = vec![
            (nodes[0], nodes[1]),
            (nodes[0], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[4], nodes[5]),
        ];

        let topo = GraphTopology::from_edges(&nodes, &edges);

        // Node 0: 1-hop = {1, 2}, 2-hop = {1, 2, 3}
        assert!(topo.neighbors_1hop[&nodes[0]].contains(&nodes[1]));
        assert!(topo.neighbors_1hop[&nodes[0]].contains(&nodes[2]));
        assert!(topo.neighbors_2hop[&nodes[0]].contains(&nodes[3]));

        // Node 4: 1-hop = {5}, not connected to 0,1,2,3
        assert!(!topo.neighbors_2hop[&nodes[4]].contains(&nodes[0]));

        // Generate samples
        let available: HashSet<NodeId> = nodes.iter().copied().collect();
        let samples = topo.generate_samples(&available, 3, 42);

        assert!(!samples.is_empty(), "should generate samples");
        for s in &samples {
            // Positive must be a 1-hop neighbor
            assert!(
                topo.neighbors_1hop[&s.anchor].contains(&s.positive),
                "positive must be neighbor of anchor"
            );
            // Negatives must NOT be 1-hop neighbors
            for neg in &s.negatives {
                assert!(
                    !topo.neighbors_1hop[&s.anchor].contains(neg),
                    "negative must not be 1-hop neighbor"
                );
                assert_ne!(*neg, s.anchor, "negative must not be anchor");
            }
        }
    }

    #[test]
    fn test_cosine_grad_direction() {
        let a = vec![1.0, 0.5, -0.3];
        let b = vec![0.8, 0.6, -0.1];
        let grad = cosine_sim_grad_a(&a, &b);

        // Verify gradient direction: small perturbation along grad should increase cosine
        let eps = 0.001;
        let a_perturbed: Vec<f32> = a
            .iter()
            .zip(&grad)
            .map(|(&ai, &gi)| ai + eps * gi)
            .collect();
        let cos_orig = cosine_sim(&a, &b);
        let cos_perturbed = cosine_sim(&a_perturbed, &b);
        assert!(
            cos_perturbed > cos_orig - 1e-4,
            "gradient should increase cosine: {} → {}",
            cos_orig,
            cos_perturbed
        );
    }
}
