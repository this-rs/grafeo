//! Ranking — composite scoring with diversity for retrieved nodes.
//!
//! Combines multiple signals (retrieval score, text richness, label diversity)
//! into a composite ranking. Uses a simplified MMR (Maximal Marginal Relevance)
//! to penalize redundant nodes that share the same labels as already-selected ones.

use std::collections::HashSet;

use crate::traits::RetrievedNode;

/// Weight for the raw retrieval score in the composite ranking.
const W_SCORE: f64 = 0.6;
/// Weight for text richness (more content = more useful for RAG).
const W_RICHNESS: f64 = 0.25;
/// Weight for label diversity bonus (unique labels get a boost).
const W_DIVERSITY: f64 = 0.15;
/// MMR redundancy penalty: nodes sharing labels with already-selected nodes
/// have their composite score multiplied by (1 - REDUNDANCY_PENALTY).
const REDUNDANCY_PENALTY: f64 = 0.30;

/// Rank retrieved nodes by composite score (descending).
///
/// The composite score is the retrieval score itself, which already
/// combines engram recall confidence × ensemble weight × activation energy.
/// This function is kept for backward compatibility and simple use cases.
pub fn rank_nodes(nodes: &mut [RetrievedNode]) {
    nodes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Rank nodes using composite scoring with label diversity (MMR-inspired).
///
/// Composite score per node:
///   `W_SCORE × normalized_score + W_RICHNESS × normalized_richness + W_DIVERSITY × diversity_bonus`
///
/// Then applies MMR: iteratively selects the best node, and penalizes remaining
/// candidates that share labels with already-selected nodes. This ensures
/// the final ranking prefers diverse content over redundant same-type nodes.
///
/// Returns a new Vec with nodes reordered by diversity-aware composite score.
pub fn rank_with_diversity(nodes: &mut Vec<RetrievedNode>) {
    if nodes.len() <= 1 {
        return;
    }

    // Pre-compute raw signals
    let max_score = nodes
        .iter()
        .map(|n| n.score)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-9);

    let richness_values: Vec<usize> = nodes.iter().map(text_richness).collect();
    let max_richness = (*richness_values.iter().max().unwrap_or(&1)).max(1) as f64;

    // Compute initial composite scores
    let mut composite: Vec<f64> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let norm_score = n.score / max_score;
            let norm_richness = richness_values[i] as f64 / max_richness;
            W_SCORE * norm_score + W_RICHNESS * norm_richness + W_DIVERSITY
        })
        .collect();

    // MMR-style greedy selection with label diversity penalty
    let n = nodes.len();
    let mut selected_order: Vec<usize> = Vec::with_capacity(n);
    let mut selected_labels: HashSet<String> = HashSet::new();
    let mut used = vec![false; n];

    for _ in 0..n {
        // Find best unselected candidate
        let mut best_idx = None;
        let mut best_score = f64::NEG_INFINITY;

        for j in 0..n {
            if used[j] {
                continue;
            }
            let mut score = composite[j];

            // Apply redundancy penalty if this node's labels overlap with selected
            let has_overlap = nodes[j].labels.iter().any(|l| selected_labels.contains(l));
            if has_overlap {
                score *= 1.0 - REDUNDANCY_PENALTY;
            }

            if score > best_score {
                best_score = score;
                best_idx = Some(j);
            }
        }

        if let Some(idx) = best_idx {
            used[idx] = true;
            selected_order.push(idx);
            // Track this node's labels for future diversity penalty
            for label in &nodes[idx].labels {
                selected_labels.insert(label.clone());
            }
            // Update composite score with the penalized value for consistent ordering
            composite[idx] = best_score;
        }
    }

    // Reorder nodes according to MMR selection order, update scores
    let old_nodes: Vec<RetrievedNode> = std::mem::take(nodes);
    let old_nodes_vec: Vec<RetrievedNode> = old_nodes;

    // Build ordered result
    let mut indexed: Vec<(usize, RetrievedNode)> = old_nodes_vec.into_iter().enumerate().collect();
    let mut result = Vec::with_capacity(n);

    for &sel_idx in &selected_order {
        // Find the node with this original index
        if let Some(pos) = indexed.iter().position(|(i, _)| *i == sel_idx) {
            let (_, mut node) = indexed.remove(pos);
            node.score = composite[sel_idx]; // Update score to composite
            result.push(node);
        }
    }

    *nodes = result;
}

/// Compute a text-richness score for a node.
///
/// Nodes with more text content are more valuable for RAG context.
/// Used as a signal in composite ranking.
pub fn text_richness(node: &RetrievedNode) -> usize {
    node.properties.values().map(|v| v.len()).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::RetrievalSource;
    use obrain_common::types::NodeId;
    use std::collections::HashMap;

    fn make_node(id: u64, score: f64) -> RetrievedNode {
        RetrievedNode {
            node_id: NodeId(id),
            labels: vec!["Test".into()],
            properties: HashMap::new(),
            score,
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: score,
            },
            outgoing_relations: vec![],
            incoming_relations: vec![],
        }
    }

    fn make_labeled_node(id: u64, score: f64, labels: Vec<&str>) -> RetrievedNode {
        let mut props = HashMap::new();
        props.insert("name".into(), format!("Node {id}"));
        RetrievedNode {
            node_id: NodeId(id),
            labels: labels.into_iter().map(String::from).collect(),
            properties: props,
            score,
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: score,
            },
            outgoing_relations: vec![],
            incoming_relations: vec![],
        }
    }

    #[test]
    fn rank_sorts_descending() {
        let mut nodes = vec![
            make_node(1, 0.3),
            make_node(2, 0.9),
            make_node(3, 0.1),
            make_node(4, 0.7),
        ];

        rank_nodes(&mut nodes);

        assert_eq!(nodes[0].node_id, NodeId(2));
        assert_eq!(nodes[1].node_id, NodeId(4));
        assert_eq!(nodes[2].node_id, NodeId(1));
        assert_eq!(nodes[3].node_id, NodeId(3));
    }

    #[test]
    fn text_richness_sums_property_lengths() {
        let mut props = HashMap::new();
        props.insert("name".into(), "Hello World".into());
        props.insert("desc".into(), "A test".into());

        let node = RetrievedNode {
            node_id: NodeId(1),
            labels: vec![],
            properties: props,
            score: 1.0,
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: 1.0,
            },
            outgoing_relations: vec![],
            incoming_relations: vec![],
        };

        assert_eq!(text_richness(&node), 17); // 11 + 6
    }

    #[test]
    fn diversity_prefers_varied_labels() {
        // Three Project nodes at equal score, one Note, one Task
        let mut nodes = vec![
            make_labeled_node(1, 0.9, vec!["Project"]),
            make_labeled_node(2, 0.9, vec!["Project"]),
            make_labeled_node(3, 0.9, vec!["Project"]),
            make_labeled_node(4, 0.8, vec!["Note"]),
            make_labeled_node(5, 0.7, vec!["Task"]),
        ];

        rank_with_diversity(&mut nodes);

        // First should be a Project (highest score), but Note and Task
        // should appear before the 2nd and 3rd Project due to diversity bonus
        let labels: Vec<&str> = nodes.iter().map(|n| n.labels[0].as_str()).collect();

        // The first node is Project (highest raw score)
        assert_eq!(labels[0], "Project");
        // Note and Task should be promoted above duplicate Projects
        let note_pos = labels.iter().position(|l| *l == "Note").unwrap();
        let task_pos = labels.iter().position(|l| *l == "Task").unwrap();
        // Both Note and Task should appear before the 3rd Project
        assert!(note_pos < 4, "Note should be in top 4");
        assert!(task_pos < 4, "Task should be in top 4");
    }

    #[test]
    fn diversity_handles_single_node() {
        let mut nodes = vec![make_node(1, 0.5)];
        rank_with_diversity(&mut nodes);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, NodeId(1));
    }

    #[test]
    fn diversity_handles_empty() {
        let mut nodes: Vec<RetrievedNode> = vec![];
        rank_with_diversity(&mut nodes);
        assert!(nodes.is_empty());
    }
}
