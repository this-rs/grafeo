//! Feedback loop — reinforces cognitive structures after LLM response.
//!
//! After the LLM generates a response using the RAG context, this module
//! identifies which concepts from the context were mentioned in the response
//! and reinforces the synapses between them. This creates a Hebbian learning
//! loop: concepts that are useful together get stronger connections.

use std::sync::Arc;

use obrain_cognitive::energy::EnergyStore;
use obrain_cognitive::synapse::SynapseStore;
use obrain_common::types::NodeId;

use crate::config::RagConfig;
use crate::error::RagResult;
use crate::traits::{FeedbackSink, FeedbackStats, RagContext};

/// Cognitive feedback sink that reinforces synapses and boosts energy.
pub struct CognitiveFeedback {
    /// Synapse store for Hebbian reinforcement.
    synapse_store: Option<Arc<SynapseStore>>,

    /// Energy store for boosting used nodes.
    energy_store: Option<Arc<EnergyStore>>,
}

impl CognitiveFeedback {
    /// Create a new feedback sink.
    pub fn new(
        synapse_store: Option<Arc<SynapseStore>>,
        energy_store: Option<Arc<EnergyStore>>,
    ) -> Self {
        Self {
            synapse_store,
            energy_store,
        }
    }

    /// Extract concept identifiers from text by checking which node properties
    /// appear in the response text.
    fn find_mentioned_nodes(
        &self,
        _context: &RagContext,
        response: &str,
        all_nodes: &[(NodeId, Vec<String>)], // (node_id, text_values)
    ) -> Vec<NodeId> {
        let response_lower = response.to_lowercase();
        let mut mentioned = Vec::new();

        for (node_id, text_values) in all_nodes {
            let is_mentioned = text_values.iter().any(|text| {
                // Check if any significant text value from the node appears in the response
                let text_lower = text.to_lowercase();
                // Only match on non-trivial text (> 3 chars)
                text_lower.len() > 3 && response_lower.contains(&text_lower)
            });

            if is_mentioned {
                mentioned.push(*node_id);
            }
        }

        mentioned
    }
}

impl FeedbackSink for CognitiveFeedback {
    fn feedback(
        &self,
        context: &RagContext,
        response: &str,
        config: &RagConfig,
    ) -> RagResult<FeedbackStats> {
        let mut stats = FeedbackStats::default();

        if context.node_ids.is_empty() || response.is_empty() {
            return Ok(stats);
        }

        // Find which nodes are actually mentioned in the LLM response
        // using the pre-extracted text values from context building
        let mentioned = self.find_mentioned_nodes(context, response, &context.node_texts);

        // Boost energy: full boost for mentioned nodes, reduced for context-only
        if let Some(ref energy_store) = self.energy_store {
            for node_id in &context.node_ids {
                let boost = if mentioned.contains(node_id) {
                    config.feedback_energy_boost
                } else {
                    config.feedback_energy_boost * 0.3 // Reduced boost for unmentioned
                };
                energy_store.boost(*node_id, boost);
                stats.nodes_boosted += 1;
            }
        }

        // Reinforce synapses only between nodes that were both mentioned
        // in the response (response-aware Hebbian reinforcement).
        // Falls back to all context pairs if no mentions detected.
        if let Some(ref synapse_store) = self.synapse_store {
            let reinforce_set = if mentioned.len() >= 2 {
                &mentioned
            } else {
                // Fallback: reinforce all context pairs if we can't detect mentions
                &context.node_ids
            };

            for i in 0..reinforce_set.len() {
                for j in (i + 1)..reinforce_set.len() {
                    synapse_store.reinforce(
                        reinforce_set[i],
                        reinforce_set[j],
                        config.feedback_reinforce_amount,
                    );
                    stats.synapses_reinforced += 1;
                }
            }
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_with_no_stores_is_noop() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: "some context".into(),
            estimated_tokens: 10,
            nodes_included: 2,
            node_ids: vec![NodeId(1), NodeId(2)],
            node_texts: vec![],
        };
        let config = RagConfig::default();

        let stats = feedback
            .feedback(&context, "some response", &config)
            .unwrap();
        assert_eq!(stats.synapses_reinforced, 0);
        assert_eq!(stats.nodes_boosted, 0);
    }

    #[test]
    fn feedback_empty_response_is_noop() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 5,
            nodes_included: 1,
            node_ids: vec![NodeId(1)],
            node_texts: vec![],
        };
        let config = RagConfig::default();

        let stats = feedback.feedback(&context, "", &config).unwrap();
        assert_eq!(stats.synapses_reinforced, 0);
        assert_eq!(stats.nodes_boosted, 0);
    }

    #[test]
    fn feedback_empty_node_ids_is_noop() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 5,
            nodes_included: 0,
            node_ids: vec![],
            node_texts: vec![],
        };
        let config = RagConfig::default();

        let stats = feedback
            .feedback(&context, "some response", &config)
            .unwrap();
        assert_eq!(stats.synapses_reinforced, 0);
        assert_eq!(stats.nodes_boosted, 0);
    }

    #[test]
    fn find_mentioned_nodes_detects_text_matches() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: String::new(),
            estimated_tokens: 0,
            nodes_included: 3,
            node_ids: vec![NodeId(1), NodeId(2), NodeId(3)],
            node_texts: vec![
                (NodeId(1), vec!["Obrain database".into()]),
                (NodeId(2), vec!["WAL recovery".into()]),
                (NodeId(3), vec!["hello".into()]),
            ],
        };

        let response = "The Obrain database uses WAL recovery for durability.";
        let mentioned = feedback.find_mentioned_nodes(&context, response, &context.node_texts);

        assert!(
            mentioned.contains(&NodeId(1)),
            "Should find 'Obrain database'"
        );
        assert!(mentioned.contains(&NodeId(2)), "Should find 'WAL recovery'");
        assert!(
            !mentioned.contains(&NodeId(3)),
            "Short text 'hello' should not match"
        );
    }

    #[test]
    fn find_mentioned_nodes_ignores_short_text() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: String::new(),
            estimated_tokens: 0,
            nodes_included: 2,
            node_ids: vec![NodeId(1), NodeId(2)],
            node_texts: vec![
                (NodeId(1), vec!["abc".into()]), // 3 chars, at threshold
                (NodeId(2), vec!["ab".into()]),  // 2 chars, below threshold
            ],
        };

        let response = "abc ab test";
        let mentioned = feedback.find_mentioned_nodes(&context, response, &context.node_texts);
        assert!(mentioned.is_empty(), "Short texts should be ignored");
    }

    #[test]
    fn find_mentioned_nodes_case_insensitive() {
        let feedback = CognitiveFeedback::new(None, None);
        let context = RagContext {
            text: String::new(),
            estimated_tokens: 0,
            nodes_included: 1,
            node_ids: vec![NodeId(1)],
            node_texts: vec![(NodeId(1), vec!["OBRAIN".into()])],
        };

        let response = "obrain is great";
        let mentioned = feedback.find_mentioned_nodes(&context, response, &context.node_texts);
        assert_eq!(mentioned.len(), 1);
    }

    #[test]
    fn feedback_with_energy_store() {
        use obrain_cognitive::energy::{EnergyConfig, EnergyStore};

        let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
        let feedback = CognitiveFeedback::new(None, Some(Arc::clone(&energy_store)));

        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 10,
            nodes_included: 2,
            node_ids: vec![NodeId(1), NodeId(2)],
            node_texts: vec![
                (NodeId(1), vec!["Obrain database".into()]),
                (NodeId(2), vec!["WAL recovery".into()]),
            ],
        };
        let config = RagConfig::default();

        let stats = feedback
            .feedback(&context, "Obrain database is great", &config)
            .unwrap();
        assert_eq!(stats.nodes_boosted, 2);
        // Node 1 mentioned → full boost, Node 2 not mentioned → reduced boost
        let e1 = energy_store.get_energy(NodeId(1));
        let e2 = energy_store.get_energy(NodeId(2));
        assert!(e1 > e2, "Mentioned node should get higher energy boost");
    }

    #[test]
    fn feedback_with_synapse_store() {
        use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};

        let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
        let feedback = CognitiveFeedback::new(Some(Arc::clone(&synapse_store)), None);

        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 10,
            nodes_included: 3,
            node_ids: vec![NodeId(1), NodeId(2), NodeId(3)],
            node_texts: vec![
                (NodeId(1), vec!["Obrain database".into()]),
                (NodeId(2), vec!["WAL recovery".into()]),
                (NodeId(3), vec!["some note".into()]),
            ],
        };
        let config = RagConfig::default();

        // Response mentions nodes 1 and 2 → synapse only between them
        let stats = feedback
            .feedback(&context, "Obrain database uses WAL recovery", &config)
            .unwrap();
        assert_eq!(stats.synapses_reinforced, 1); // 1 pair: (1,2)

        // Check synapse exists
        let synapse = synapse_store.get_synapse(NodeId(1), NodeId(2));
        assert!(
            synapse.is_some(),
            "Synapse should exist between mentioned nodes"
        );
    }

    #[test]
    fn feedback_fallback_reinforces_all_when_no_mentions() {
        use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};

        let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
        let feedback = CognitiveFeedback::new(Some(Arc::clone(&synapse_store)), None);

        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 10,
            nodes_included: 3,
            node_ids: vec![NodeId(1), NodeId(2), NodeId(3)],
            node_texts: vec![
                (NodeId(1), vec!["aaaa".into()]), // too short, won't match
                (NodeId(2), vec!["bbbb".into()]),
                (NodeId(3), vec!["cccc".into()]),
            ],
        };
        let config = RagConfig::default();

        // Response doesn't mention any node text → fallback to all pairs
        let stats = feedback
            .feedback(&context, "completely unrelated response text here", &config)
            .unwrap();
        // 3 nodes → C(3,2) = 3 pairs
        assert_eq!(stats.synapses_reinforced, 3);
    }
}
