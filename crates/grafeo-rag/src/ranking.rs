//! Ranking — composite scoring for retrieved nodes.
//!
//! Combines multiple signals (retrieval score, energy, synapse weight)
//! into a single ranking used to select which nodes to include in the
//! LLM context.

use crate::traits::RetrievedNode;

/// Rank retrieved nodes by composite score (descending).
///
/// The composite score is the retrieval score itself, which already
/// combines engram recall confidence × ensemble weight × activation energy.
/// This function is the extension point for adding more signals
/// (energy decay, fabric scores, etc.) in the future.
pub fn rank_nodes(nodes: &mut [RetrievedNode]) {
    nodes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Compute a text-richness score for a node.
///
/// Nodes with more text content are more valuable for RAG context.
/// This is used as a tiebreaker when retrieval scores are equal.
pub fn text_richness(node: &RetrievedNode) -> usize {
    node.properties
        .values()
        .map(|v| v.len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::RetrievalSource;
    use grafeo_common::types::NodeId;
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
            source: RetrievalSource::SpreadingActivation { depth: 0, activation: 1.0 },
            outgoing_relations: vec![],
            incoming_relations: vec![],
        };

        assert_eq!(text_richness(&node), 17); // 11 + 6
    }
}
