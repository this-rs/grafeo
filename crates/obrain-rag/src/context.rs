//! Context builder — formats retrieved nodes into LLM-consumable text.
//!
//! The output is structured markdown grouped by label type, with properties,
//! relation metadata, and noise filtering. The builder handles ranking,
//! token budgeting, and formatting in a single pipeline.

use std::collections::HashMap;
use std::fmt::Write;

use crate::budget::select_within_budget;
use crate::config::RagConfig;
use crate::error::RagResult;
use crate::ranking::rank_with_diversity;
use crate::traits::{ContextBuilder, RagContext, RetrievalResult, RetrievedNode};

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
        // Step 1: Rank nodes with diversity-aware composite scoring
        let mut nodes = result.nodes.clone();
        rank_with_diversity(&mut nodes);

        // Step 2: Select within budget
        let (selected, _estimated_tokens) = select_within_budget(&nodes, config);

        if selected.is_empty() {
            return Ok(RagContext {
                text: String::new(),
                estimated_tokens: 0,
                nodes_included: 0,
                node_ids: Vec::new(),
                node_texts: Vec::new(),
            });
        }

        // Build a name lookup for relation target nodes (best-effort)
        let name_lookup = build_name_lookup(&selected);

        // Collect noise property names (lowercased for comparison)
        let noise: Vec<String> = config
            .noise_properties
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        // Step 3: Group selected nodes by primary label
        let mut label_groups: Vec<(String, Vec<&RetrievedNode>)> = Vec::new();
        let mut label_order: HashMap<String, usize> = HashMap::new();

        for node in &selected {
            let primary_label = if config.include_labels && !node.labels.is_empty() {
                node.labels[0].clone()
            } else {
                "Other".to_string()
            };

            if let Some(&idx) = label_order.get(&primary_label) {
                label_groups[idx].1.push(node);
            } else {
                let idx = label_groups.len();
                label_order.insert(primary_label.clone(), idx);
                label_groups.push((primary_label, vec![node]));
            }
        }

        // Step 4: Format grouped markdown
        let mut text = String::with_capacity(4096);
        let mut node_ids = Vec::with_capacity(selected.len());
        let mut node_texts: Vec<(obrain_common::types::NodeId, Vec<String>)> =
            Vec::with_capacity(selected.len());

        // Count distinct label types for the summary
        let type_count = label_groups.len();

        writeln!(text, "# Graph Knowledge Context").unwrap();
        writeln!(
            text,
            "_Retrieved {} nodes across {} types._\n",
            selected.len(),
            type_count
        )
        .unwrap();

        for (label, group_nodes) in &label_groups {
            // Section header per label group
            writeln!(text, "## {label}\n").unwrap();

            for node in group_nodes {
                node_ids.push(node.node_id);

                // Collect significant text values for feedback mention detection
                let text_values: Vec<String> = node
                    .properties
                    .values()
                    .filter(|v| v.len() > 3)
                    .cloned()
                    .collect();
                node_texts.push((node.node_id, text_values));

                // Node sub-header with all labels and score
                if node.labels.len() > 1 {
                    let all_labels = node.labels.join(", ");
                    writeln!(text, "### [{all_labels}] (score: {:.2})", node.score).unwrap();
                } else {
                    // Try to use a "name" or "title" property as header
                    let display_name = node
                        .properties
                        .get("name")
                        .or_else(|| node.properties.get("title"))
                        .map_or("", |s| s.as_str());
                    if display_name.is_empty() {
                        writeln!(text, "### (score: {:.2})", node.score).unwrap();
                    } else {
                        writeln!(text, "### {display_name} (score: {:.2})", node.score).unwrap();
                    }
                }

                // Properties (filtered for noise)
                for (key, value) in &node.properties {
                    if noise.iter().any(|n| key.to_lowercase() == *n) {
                        continue;
                    }
                    let val_str: &str = value.as_str();
                    if val_str.is_empty() {
                        continue;
                    }
                    let display_value = if val_str.len() > 500 {
                        // Find a char boundary at or before byte 500 to avoid
                        // panicking on multi-byte UTF-8 characters.
                        let truncate_at = val_str
                            .char_indices()
                            .take_while(|&(i, _)| i <= 500)
                            .last()
                            .map_or(0, |(i, _)| i);
                        let omitted = val_str[truncate_at..].chars().count();
                        format!("{}… ({} chars omitted)", &val_str[..truncate_at], omitted)
                    } else {
                        val_str.to_string()
                    };
                    writeln!(text, "- **{key}**: {display_value}").unwrap();
                }

                // Relations with resolved names
                if config.include_relations {
                    let max_display = config.max_relations_display;

                    if !node.outgoing_relations.is_empty() {
                        write!(text, "- _outgoing_: ").unwrap();
                        let rels: Vec<String> = node
                            .outgoing_relations
                            .iter()
                            .take(max_display)
                            .map(|(rel_type, target)| {
                                let target_name = name_lookup.get(target).map_or_else(
                                    || format!("{}", target.0),
                                    |s| format!("\"{}\"", s),
                                );
                                format!("-[{rel_type}]→{target_name}")
                            })
                            .collect();
                        write!(text, "{}", rels.join(", ")).unwrap();
                        let remaining = node.outgoing_relations.len().saturating_sub(max_display);
                        if remaining > 0 {
                            write!(text, " … and {remaining} more").unwrap();
                        }
                        writeln!(text).unwrap();
                    }

                    if !node.incoming_relations.is_empty() {
                        write!(text, "- _incoming_: ").unwrap();
                        let rels: Vec<String> = node
                            .incoming_relations
                            .iter()
                            .take(max_display)
                            .map(|(rel_type, source)| {
                                let source_name = name_lookup.get(source).map_or_else(
                                    || format!("{}", source.0),
                                    |s| format!("\"{}\"", s),
                                );
                                format!("{source_name}←[{rel_type}]-")
                            })
                            .collect();
                        write!(text, "{}", rels.join(", ")).unwrap();
                        let remaining = node.incoming_relations.len().saturating_sub(max_display);
                        if remaining > 0 {
                            write!(text, " … and {remaining} more").unwrap();
                        }
                        writeln!(text).unwrap();
                    }
                }

                writeln!(text).unwrap();
            }
        }

        // Re-estimate tokens on the final formatted text
        let final_tokens = (text.len() as f64 / config.chars_per_token).ceil() as usize;

        Ok(RagContext {
            text,
            estimated_tokens: final_tokens,
            nodes_included: node_ids.len(),
            node_ids,
            node_texts,
        })
    }
}

/// Build a NodeId → display name lookup from the selected nodes.
///
/// Uses the first available "name" or "title" property as the display name.
/// This allows relation targets that are also in the selected set to show
/// human-readable names instead of raw IDs.
fn build_name_lookup(nodes: &[&RetrievedNode]) -> HashMap<obrain_common::types::NodeId, String> {
    let mut lookup = HashMap::new();
    for node in nodes {
        if let Some(name) = node
            .properties
            .get("name")
            .or_else(|| node.properties.get("title"))
            && !name.is_empty()
        {
            lookup.insert(node.node_id, name.clone());
        }
    }
    lookup
}

/// Estimate tokens for a pre-formatted text string.
pub fn estimate_text_tokens(text: &str, chars_per_token: f64) -> usize {
    (text.len() as f64 / chars_per_token).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{RetrievalResult, RetrievalSource, RetrievedNode};
    use obrain_common::types::NodeId;
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

    fn make_mixed_result() -> RetrievalResult {
        let nodes = vec![
            RetrievedNode {
                node_id: NodeId(1),
                labels: vec!["Project".into()],
                properties: [
                    ("name".into(), "Obrain".into()),
                    ("id".into(), "123".into()),
                ]
                .into_iter()
                .collect(),
                score: 0.9,
                source: RetrievalSource::EngramRecall {
                    engram_id: 1,
                    confidence: 0.9,
                },
                outgoing_relations: vec![("HAS_NOTE".into(), NodeId(2))],
                incoming_relations: vec![],
            },
            RetrievedNode {
                node_id: NodeId(2),
                labels: vec!["Note".into()],
                properties: [
                    ("title".into(), "WAL Bug".into()),
                    ("uuid".into(), "abc".into()),
                ]
                .into_iter()
                .collect(),
                score: 0.7,
                source: RetrievalSource::SpreadingActivation {
                    depth: 1,
                    activation: 0.7,
                },
                outgoing_relations: vec![],
                incoming_relations: vec![("HAS_NOTE".into(), NodeId(1))],
            },
        ];

        RetrievalResult {
            nodes,
            engrams_matched: 1,
            nodes_activated: 2,
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
        assert!(ctx.nodes_included < 50);
    }

    #[test]
    fn groups_by_label_and_filters_noise() {
        let result = make_mixed_result();
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();

        // Should have label group headers
        assert!(ctx.text.contains("## Project"));
        assert!(ctx.text.contains("## Note"));

        // Should show summary with type count
        assert!(ctx.text.contains("across 2 types"));

        // Noise properties should be filtered
        assert!(
            !ctx.text.contains("**id**: 123"),
            "id property should be filtered"
        );
        assert!(
            !ctx.text.contains("**uuid**: abc"),
            "uuid property should be filtered"
        );

        // Real properties should be present
        assert!(ctx.text.contains("Obrain"));
        assert!(ctx.text.contains("WAL Bug"));
    }

    #[test]
    fn truncation_handles_multibyte_utf8() {
        // Regression test: truncating at byte 500 can land inside a
        // multi-byte char (e.g. '─' = 3 bytes), causing a panic.
        let mut long_text = "a".repeat(498);
        long_text.push_str("─── suite du texte très long pour dépasser les 500 bytes");
        assert!(long_text.len() > 500);

        let mut props = HashMap::new();
        props.insert("content".into(), long_text);

        let result = RetrievalResult {
            nodes: vec![RetrievedNode {
                node_id: NodeId(1),
                labels: vec!["Test".into()],
                properties: props,
                score: 1.0,
                source: RetrievalSource::EngramRecall {
                    engram_id: 1,
                    confidence: 1.0,
                },
                outgoing_relations: vec![],
                incoming_relations: vec![],
            }],
            engrams_matched: 1,
            nodes_activated: 1,
        };

        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();

        // This should NOT panic
        let ctx = builder.build(&result, &config).unwrap();
        assert!(ctx.text.contains("chars omitted"));
    }

    #[test]
    fn node_without_name_uses_score_header() {
        let result = RetrievalResult {
            nodes: vec![RetrievedNode {
                node_id: NodeId(1),
                labels: vec!["Orphan".into()],
                properties: [("data".into(), "some value".into())].into_iter().collect(),
                score: 0.5,
                source: RetrievalSource::SpreadingActivation {
                    depth: 0,
                    activation: 0.5,
                },
                outgoing_relations: vec![],
                incoming_relations: vec![],
            }],
            engrams_matched: 1,
            nodes_activated: 1,
        };
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();
        let ctx = builder.build(&result, &config).unwrap();
        // No name/title → header should just show score
        assert!(ctx.text.contains("(score: 0.50)"));
    }

    #[test]
    fn multi_label_node_shows_all_labels() {
        let result = RetrievalResult {
            nodes: vec![RetrievedNode {
                node_id: NodeId(1),
                labels: vec!["Note".into(), "Gotcha".into()],
                properties: [("title".into(), "Bug fix".into())].into_iter().collect(),
                score: 0.8,
                source: RetrievalSource::EngramRecall {
                    engram_id: 1,
                    confidence: 0.8,
                },
                outgoing_relations: vec![],
                incoming_relations: vec![],
            }],
            engrams_matched: 1,
            nodes_activated: 1,
        };
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();
        let ctx = builder.build(&result, &config).unwrap();
        assert!(
            ctx.text.contains("Note, Gotcha"),
            "Multi-label nodes should show all labels"
        );
    }

    #[test]
    fn incoming_relations_with_overflow() {
        let incoming: Vec<(String, NodeId)> = (0..15)
            .map(|i| ("USED_BY".into(), NodeId(100 + i)))
            .collect();
        let result = RetrievalResult {
            nodes: vec![RetrievedNode {
                node_id: NodeId(1),
                labels: vec!["Module".into()],
                properties: [("name".into(), "core".into())].into_iter().collect(),
                score: 0.9,
                source: RetrievalSource::EngramRecall {
                    engram_id: 1,
                    confidence: 0.9,
                },
                outgoing_relations: vec![],
                incoming_relations: incoming,
            }],
            engrams_matched: 1,
            nodes_activated: 1,
        };
        let config = RagConfig {
            include_relations: true,
            max_relations_display: 5,
            ..RagConfig::default()
        };
        let builder = GraphContextBuilder::new();
        let ctx = builder.build(&result, &config).unwrap();
        assert!(
            ctx.text.contains("more"),
            "Should show truncation indicator"
        );
    }

    #[test]
    fn labels_disabled_uses_other_group() {
        let result = make_result(2);
        let config = RagConfig {
            include_labels: false,
            ..RagConfig::default()
        };
        let builder = GraphContextBuilder::new();
        let ctx = builder.build(&result, &config).unwrap();
        assert!(ctx.text.contains("## Other"));
    }

    #[test]
    fn node_texts_populated_for_feedback() {
        let result = make_mixed_result();
        let config = RagConfig::default();
        let builder = GraphContextBuilder::new();
        let ctx = builder.build(&result, &config).unwrap();
        assert!(
            !ctx.node_texts.is_empty(),
            "node_texts should be populated for feedback"
        );
        // "Obrain" (6 chars > 3) should be in node_texts
        let obrain_texts = ctx.node_texts.iter().find(|(id, _)| *id == NodeId(1));
        assert!(obrain_texts.is_some());
    }

    #[test]
    fn estimate_text_tokens_basic() {
        let tokens = estimate_text_tokens("hello world", 4.0);
        assert_eq!(tokens, 3); // ceil(11/4.0) = 3
    }

    #[test]
    fn resolves_relation_names() {
        let result = make_mixed_result();
        let config = RagConfig {
            include_relations: true,
            ..RagConfig::default()
        };
        let builder = GraphContextBuilder::new();

        let ctx = builder.build(&result, &config).unwrap();

        // Relations should show resolved names when available
        assert!(
            ctx.text.contains("\"WAL Bug\"") || ctx.text.contains("\"Obrain\""),
            "Relations should show resolved node names"
        );
    }
}
