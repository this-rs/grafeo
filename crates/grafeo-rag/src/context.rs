//! Context builder — formats retrieved nodes into LLM-consumable text.
//!
//! The output is structured markdown with sections per concept, including
//! labels, properties, and relation metadata. The builder handles ranking,
//! token budgeting, and formatting in a single pipeline.

use std::fmt::Write;

use crate::budget::select_within_budget;
use crate::config::RagConfig;
use crate::error::RagResult;
use crate::ranking::rank_nodes;
use crate::traits::{ContextBuilder, RagContext, RetrievalResult};

/// Default context builder that produces structured markdown.
pub struct GraphContextBuilder;

impl GraphContextBuilder {
    /// Create a new context builder.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextBuilder for GraphContextBuilder {
    fn build(&self, result: &RetrievalResult, config: &RagConfig) -> RagResult<RagContext> {
        // Step 1: Rank nodes
        let mut nodes = result.nodes.clone();
        rank_nodes(&mut nodes);

        // Step 2: Select within budget
        let (selected, estimated_tokens) = select_within_budget(&nodes, config);

        if selected.is_empty() {
            return Ok(RagContext {
                text: String::new(),
                estimated_tokens: 0,
                nodes_included: 0,
                node_ids: Vec::new(),
            });
        }

        // Step 3: Format
        let mut text = String::with_capacity(estimated_tokens * 4); // rough char estimate
        let mut node_ids = Vec::with_capacity(selected.len());

        writeln!(text, "# Graph Knowledge Context").unwrap();
        writeln!(
            text,
            "_Retrieved {} relevant nodes from the knowledge graph._\n",
            selected.len()
        )
        .unwrap();

        for (i, node) in selected.iter().enumerate() {
            node_ids.push(node.node_id);

            // Section header with labels
            if config.include_labels && !node.labels.is_empty() {
                let labels = node.labels.join(", ");
                writeln!(text, "## [{labels}] (score: {:.2})", node.score).unwrap();
            } else {
                writeln!(text, "## Node {} (score: {:.2})", i + 1, node.score).unwrap();
            }

            // Properties
            for (key, value) in &node.properties {
                // Truncate very long values for readability
                let val_str: &str = value.as_str();
                let display_value = if val_str.len() > 500 {
                    format!("{}...", &val_str[..500])
                } else {
                    val_str.to_string()
                };
                writeln!(text, "- **{key}**: {display_value}").unwrap();
            }

            // Relations
            if config.include_relations {
                if !node.outgoing_relations.is_empty() {
                    write!(text, "- _outgoing_: ").unwrap();
                    let rels: Vec<String> = node
                        .outgoing_relations
                        .iter()
                        .take(10) // Limit displayed relations
                        .map(|(rel_type, target)| format!("-[{rel_type}]→{}", target.0))
                        .collect();
                    writeln!(text, "{}", rels.join(", ")).unwrap();
                }
                if !node.incoming_relations.is_empty() {
                    write!(text, "- _incoming_: ").unwrap();
                    let rels: Vec<String> = node
                        .incoming_relations
                        .iter()
                        .take(10)
                        .map(|(rel_type, source)| format!("{}←[{rel_type}]-", source.0))
                        .collect();
                    writeln!(text, "{}", rels.join(", ")).unwrap();
                }
            }

            writeln!(text).unwrap();
        }

        // Re-estimate tokens on the final formatted text
        let final_tokens = (text.len() as f64 / config.chars_per_token).ceil() as usize;

        Ok(RagContext {
            text,
            estimated_tokens: final_tokens,
            nodes_included: node_ids.len(),
            node_ids,
        })
    }
}

/// Estimate tokens for a pre-formatted text string.
pub fn estimate_text_tokens(text: &str, chars_per_token: f64) -> usize {
    (text.len() as f64 / chars_per_token).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{RetrievalResult, RetrievalSource, RetrievedNode};
    use grafeo_common::types::NodeId;
    use std::collections::HashMap;

    fn make_result(count: usize) -> RetrievalResult {
        let nodes = (0..count)
            .map(|i| {
                let mut props = HashMap::new();
                props.insert("name".into(), format!("Node {}", i));
                props.insert("description".into(), format!("Description for node {}", i));

                RetrievedNode {
                    node_id: NodeId(i as u64),
                    labels: vec!["Concept".into()],
                    properties: props,
                    score: 1.0 - (i as f64 * 0.1),
                    source: RetrievalSource::EngramRecall {
                        engram_id: i as u64,
                        confidence: 0.9,
                    },
                    outgoing_relations: vec![("RELATES_TO".into(), NodeId(100))],
                    incoming_relations: vec![],
                }
            })
            .collect();

        RetrievalResult {
            nodes,
            engrams_matched: count,
            nodes_activated: count * 3,
        }
    }

    #[test]
    fn builds_non_empty_context() {
        let result = make_result(5);
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();
        assert!(!ctx.text.is_empty());
        assert!(ctx.nodes_included > 0);
        assert!(ctx.estimated_tokens > 0);
    }

    #[test]
    fn context_contains_node_properties() {
        let result = make_result(1);
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();
        assert!(ctx.text.contains("Node 0"));
        assert!(ctx.text.contains("Description for node 0"));
    }

    #[test]
    fn context_includes_relations() {
        let result = make_result(1);
        let config = RagConfig {
            include_relations: true,
            ..RagConfig::default()
        };
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();
        assert!(ctx.text.contains("RELATES_TO"));
    }

    #[test]
    fn empty_result_produces_empty_context() {
        let result = RetrievalResult {
            nodes: vec![],
            engrams_matched: 0,
            nodes_activated: 0,
        };
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();
        assert!(ctx.text.is_empty());
        assert_eq!(ctx.nodes_included, 0);
    }

    #[test]
    fn respects_token_budget() {
        let result = make_result(50);
        let config = RagConfig {
            token_budget: 100,
            ..RagConfig::default()
        };
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();
        // Should have fewer nodes than the 50 available
        assert!(ctx.nodes_included < 50);
    }
}
