//! Token budget management — selects the optimal subset of nodes
//! that fits within the token budget.
//!
//! Uses a greedy approach: nodes are already ranked by score, so we
//! take as many as we can from the top until the budget is exhausted.

use crate::config::RagConfig;
use crate::traits::RetrievedNode;

/// Estimate the token count for a node's text representation.
///
/// Accounts for markdown formatting overhead (headers, bullets, separators)
/// on top of raw content characters. Uses chars/token heuristic.
pub fn estimate_tokens(node: &RetrievedNode, config: &RagConfig) -> usize {
    let mut chars = 0usize;

    // Section header: "### NodeName (score: X.XX)\n"
    // or "## [Label] (score: X.XX)\n"
    if config.include_labels && !node.labels.is_empty() {
        // "### " + labels + " (score: X.XX)\n"
        chars += 4 + node.labels.iter().map(|l| l.len() + 2).sum::<usize>() + 16;
    } else {
        chars += 25; // "### Node N (score: X.XX)\n"
    }

    // Properties: "- **key**: value\n" per property
    let noise: Vec<String> = config
        .noise_properties
        .iter()
        .map(|s| s.to_lowercase())
        .collect();
    for (key, value) in &node.properties {
        if noise.iter().any(|n| key.to_lowercase() == *n) {
            continue; // Skip noise properties in estimate too
        }
        let val_len = value.len().min(500);
        chars += 6 + key.len() + val_len + 1; // "- **" + key + "**: " + value + "\n"
    }

    // Relations: "- _outgoing_: -[TYPE]→Name, ...\n"
    if config.include_relations {
        let out_count = node
            .outgoing_relations
            .len()
            .min(config.max_relations_display);
        let in_count = node
            .incoming_relations
            .len()
            .min(config.max_relations_display);
        if out_count > 0 {
            chars += 16 + out_count * 25; // prefix + per-relation estimate
            if node.outgoing_relations.len() > config.max_relations_display {
                chars += 25; // "... and N more outgoing\n"
            }
        }
        if in_count > 0 {
            chars += 16 + in_count * 25;
            if node.incoming_relations.len() > config.max_relations_display {
                chars += 25;
            }
        }
    }

    // Trailing blank line between nodes
    chars += 1;

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
