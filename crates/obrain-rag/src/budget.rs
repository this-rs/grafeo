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
    use obrain_common::types::NodeId;
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
    fn estimate_tokens_without_labels() {
        let node = make_node_with_content(1, 1.0, "test");
        let config = RagConfig {
            include_labels: false,
            ..RagConfig::default()
        };
        let tokens = estimate_tokens(&node, &config);
        assert!(tokens > 0);
    }

    #[test]
    fn estimate_tokens_with_relations_and_overflow() {
        let mut props = HashMap::new();
        props.insert("name".into(), "Test".into());

        let node = RetrievedNode {
            node_id: NodeId(1),
            labels: vec!["Note".into()],
            properties: props,
            score: 1.0,
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: 1.0,
            },
            // More relations than max_relations_display
            outgoing_relations: (0..15).map(|i| ("REL".into(), NodeId(i))).collect(),
            incoming_relations: (0..15).map(|i| ("REL".into(), NodeId(100 + i))).collect(),
        };

        let config = RagConfig {
            include_relations: true,
            max_relations_display: 5,
            ..RagConfig::default()
        };

        let tokens = estimate_tokens(&node, &config);
        // Should account for the "... and N more" truncation indicators
        assert!(tokens > 0);
    }

    #[test]
    fn estimate_tokens_skips_noise_properties() {
        let mut props = HashMap::new();
        props.insert("name".into(), "visible".into());
        props.insert("id".into(), "should-be-skipped".into());
        props.insert("uuid".into(), "also-skipped".into());

        let node = RetrievedNode {
            node_id: NodeId(1),
            labels: vec!["Note".into()],
            properties: props,
            score: 1.0,
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: 1.0,
            },
            outgoing_relations: vec![],
            incoming_relations: vec![],
        };

        let config_with_noise = RagConfig::default();
        let config_no_noise = RagConfig {
            noise_properties: vec![],
            ..RagConfig::default()
        };

        let tokens_filtered = estimate_tokens(&node, &config_with_noise);
        let tokens_all = estimate_tokens(&node, &config_no_noise);
        assert!(tokens_filtered < tokens_all);
    }

    #[test]
    fn estimate_tokens_no_relations() {
        let node = make_node_with_content(1, 1.0, "test");
        let config = RagConfig {
            include_relations: false,
            ..RagConfig::default()
        };
        let tokens = estimate_tokens(&node, &config);
        assert!(tokens > 0);
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
