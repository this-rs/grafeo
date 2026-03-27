//! Token budget management — selects the optimal subset of nodes
//! that fits within the token budget.
//!
//! Uses a greedy approach: nodes are already ranked by score, so we
//! take as many as we can from the top until the budget is exhausted.

use crate::config::RagConfig;
use crate::traits::RetrievedNode;

/// Estimate the token count for a node's text representation.
///
/// Uses a simple chars/token heuristic. More accurate estimation
/// would require a tokenizer, but this is sufficient for budgeting.
pub fn estimate_tokens(node: &RetrievedNode, config: &RagConfig) -> usize {
    let mut chars = 0usize;

    // Labels line
    if config.include_labels && !node.labels.is_empty() {
        chars += node.labels.iter().map(|l| l.len() + 2).sum::<usize>() + 10;
    }

    // Properties
    for (key, value) in &node.properties {
        chars += key.len() + value.len() + 4; // "key: value\n"
    }

    // Relations summary
    if config.include_relations {
        chars += node.outgoing_relations.len() * 30; // approximate
        chars += node.incoming_relations.len() * 30;
    }

    // Section overhead (header, separators)
    chars += 20;

    (chars as f64 / config.chars_per_token).ceil() as usize
}

/// Select nodes that fit within the token budget.
///
/// Assumes nodes are already sorted by score (highest first).
/// Returns the selected nodes and their estimated total tokens.
pub fn select_within_budget<'a>(
    nodes: &'a [RetrievedNode],
    config: &RagConfig,
) -> (Vec<&'a RetrievedNode>, usize) {
    let mut selected = Vec::new();
    let mut total_tokens = 0usize;

    for node in nodes {
        let node_tokens = estimate_tokens(node, config);

        if total_tokens + node_tokens > config.token_budget {
            // If we haven't selected anything yet, take at least the first node
            // even if it exceeds the budget slightly
            if selected.is_empty() {
                selected.push(node);
                total_tokens += node_tokens;
            }
            break;
        }

        selected.push(node);
        total_tokens += node_tokens;
    }

    (selected, total_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::RetrievalSource;
    use grafeo_common::types::NodeId;
    use std::collections::HashMap;

    fn make_node_with_content(id: u64, score: f64, content: &str) -> RetrievedNode {
        let mut props = HashMap::new();
        props.insert("content".into(), content.into());

        RetrievedNode {
            node_id: NodeId(id),
            labels: vec!["Note".into()],
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
    fn budget_selects_top_nodes() {
        let nodes: Vec<RetrievedNode> = (0..10)
            .map(|i| make_node_with_content(i, 1.0 - i as f64 * 0.1, &"x".repeat(100)))
            .collect();

        let config = RagConfig {
            token_budget: 200,
            ..RagConfig::default()
        };

        let (selected, total) = select_within_budget(&nodes, &config);
        assert!(!selected.is_empty());
        assert!(total <= 200 || selected.len() == 1); // At least 1 node even if over budget
    }

    #[test]
    fn budget_always_includes_at_least_one() {
        let nodes = vec![make_node_with_content(1, 1.0, &"x".repeat(10000))];

        let config = RagConfig {
            token_budget: 10, // Very small budget
            ..RagConfig::default()
        };

        let (selected, _) = select_within_budget(&nodes, &config);
        assert_eq!(selected.len(), 1); // Still includes the one node
    }

    #[test]
    fn budget_respects_limit() {
        let nodes: Vec<RetrievedNode> = (0..5)
            .map(|i| make_node_with_content(i, 1.0, "short"))
            .collect();

        let config = RagConfig {
            token_budget: 10000, // Huge budget
            ..RagConfig::default()
        };

        let (selected, total) = select_within_budget(&nodes, &config);
        assert_eq!(selected.len(), 5);
        assert!(total <= 10000);
    }
}
