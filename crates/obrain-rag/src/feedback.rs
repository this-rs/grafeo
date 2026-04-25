//! Feedback loop — reinforces cognitive structures after LLM response.
//!
//! After the LLM generates a response using the RAG context, this module
//! identifies which concepts from the context were mentioned in the response
//! and reinforces the synapses between them. This creates a Hebbian learning
//! loop: concepts that are useful together get stronger connections.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use obrain_cognitive::energy::EnergyStore;
use obrain_cognitive::synapse::SynapseStore;
use obrain_common::types::NodeId;

use crate::config::RagConfig;
use crate::error::RagResult;
use crate::traits::{FeedbackSink, FeedbackStats, RagContext};

/// Plan 69e59065 T2 — hard cap on the number of pairs reinforced when
/// the fallback path (no mentions detected) is taken. The previous
/// behaviour reinforced C(N,2) pairs on every unmatched response, which
/// on N=100 context nodes meant 4 950 reinforce calls → kernel watchdog
/// timeout (gotcha 49938809).
///
/// ## History of the cap value
///
/// - **Initial T2 cap = 8** — emergency value picked right after the
///   incident, with no measurement. C(8,2) = 28 reinforces / cycle.
///   Conservative: stops the bleeding but throws away most of the
///   recall context (a `thorough` retrieve returns up to 50 nodes,
///   we'd ignore all but the top 8 for Hebbian reinforcement).
///
/// - **Revised cap = 32** (2026-04-25) — empirically calibrated by
///   `feedback_scaling_bench.rs` which sweeps N ∈ {8, 16, 24, 32, 48,
///   64, 100, 200} on a 10 k-synapse store post-T0+T2+T4 fixes.
///   Measured p99 / cycle (release, single thread):
///
///   | N    | pairs | p99       | verdict                    |
///   |------|-------|-----------|----------------------------|
///   | 8    |   28  | 0.41 ms   | × 30 under safety          |
///   | 16   |  120  | 1.69 ms   |                            |
///   | 24   |  276  | 3.44 ms   |                            |
///   | **32** | **496** | **6.27 ms** | **× 8 under 50 ms gate**   |
///   | 48   | 1128  | 16.06 ms  | over safety, under gate    |
///   | 64   | 2016  | 26.74 ms  |                            |
///   | 100  | 4950  | 86.59 ms  | OVER GATE — never ship     |
///   | 200  |19900  | 607.59 ms | catastrophic               |
///
///   32 is the largest N whose p99 stays under `gate / 4×` (= 12.5 ms)
///   on the synapse loop alone, leaving headroom for the rest of
///   `feedback()` (energy boost, reward write, mention scan, T6 chain
///   walks). 17.7× more pairs explored than the conservative cap.
///
/// ## Why not higher
///
/// At N=48 the synapse loop alone takes 16 ms p99 — already eats most
/// of the 50 ms budget once `find_mentioned`, `expand_by_affinity`,
/// chain walks and persist hits are added. At N=100 the loop ALONE
/// breaches the gate. The cap remains the watchdog: any `thorough`
/// retrieve up to 50 nodes is fed to the loop, but only the top 32 by
/// upstream relevance/energy ordering get Hebbian reinforcement.
///
/// `pub(crate)` so internal tests can reference the constant rather
/// than hard-coding `C(32, 2) = 496`. Bench gate is
/// `feedback_scaling_bench.rs` (re-run before changing this value).
pub(crate) const FALLBACK_CAP: usize = 32;

/// Cognitive feedback sink that reinforces synapses and boosts energy.
pub struct CognitiveFeedback {
    /// Synapse store for Hebbian reinforcement.
    synapse_store: Option<Arc<SynapseStore>>,

    /// Energy store for boosting used nodes.
    energy_store: Option<Arc<EnergyStore>>,

    /// Plan 69e59065 T2 — observability for the fallback-vs-mentions
    /// selector. Every `feedback()` call increments exactly one of
    /// these, giving the empirical fallback rate over a conversation.
    /// Target: fallback_rate should drop sharply once T2 step 2 adds
    /// cosine-based matching that tolerates paraphrase.
    mentions_path_total: AtomicU64,
    fallback_path_total: AtomicU64,
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
            mentions_path_total: AtomicU64::new(0),
            fallback_path_total: AtomicU64::new(0),
        }
    }

    /// Total number of feedback cycles that took the "mentions" branch
    /// (≥ 2 nodes matched in the response).
    pub fn mentions_path_total(&self) -> u64 {
        self.mentions_path_total.load(Ordering::Relaxed)
    }

    /// Total number of feedback cycles that fell back to the capped
    /// "top-K pairs" branch. High ratio here = the matching heuristic
    /// is too strict; will be addressed by T2 step 2 (cosine matching).
    pub fn fallback_path_total(&self) -> u64 {
        self.fallback_path_total.load(Ordering::Relaxed)
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
        //
        // Plan 69e59065 T2 — the fallback "all pairs" path is now CAPPED
        // at FALLBACK_CAP nodes (= top-K by recency in the context list,
        // which is already energy-ordered upstream). Without the cap, an
        // unmentioned response on N=100 context produces 4 950 reinforces,
        // which collapses the system (gotcha 49938809). With the cap the
        // fallback peaks at C(8,2) = 28 reinforces — proportional to the
        // useful signal, not to the entire context budget.
        if let Some(ref synapse_store) = self.synapse_store {
            let reinforce_slice: &[NodeId] = if mentioned.len() >= 2 {
                self.mentions_path_total.fetch_add(1, Ordering::Relaxed);
                &mentioned
            } else {
                // Fallback path. Take the top-FALLBACK_CAP nodes from the
                // context. Upstream callers populate `node_ids` in
                // descending relevance/energy order, so the slice prefix
                // is the most informative subset. Empty context is a
                // no-op.
                self.fallback_path_total.fetch_add(1, Ordering::Relaxed);
                let len = context.node_ids.len().min(FALLBACK_CAP);
                &context.node_ids[..len]
            };

            for i in 0..reinforce_slice.len() {
                for j in (i + 1)..reinforce_slice.len() {
                    synapse_store.reinforce(
                        reinforce_slice[i],
                        reinforce_slice[j],
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
        // 3 nodes → C(3,2) = 3 pairs (well below FALLBACK_CAP, no clamping).
        assert_eq!(stats.synapses_reinforced, 3);
        // Path counters: this was a fallback cycle.
        assert_eq!(feedback.fallback_path_total(), 1);
        assert_eq!(feedback.mentions_path_total(), 0);
    }

    /// Plan 69e59065 T2 — proves the FALLBACK_CAP guards the worst case.
    /// Without the cap, N=`OVER_CAP` unmatched context nodes would
    /// reinforce C(N,2) pairs, each costing a normalize_outgoing call
    /// in `SynapseStore::reinforce`. With the cap, the fallback peaks
    /// at C(FALLBACK_CAP, 2) pairs regardless of context size. The
    /// expected pair count is computed from the constant so the test
    /// stays valid if the cap is re-tuned (see `feedback_scaling_bench.rs`).
    #[test]
    fn feedback_fallback_capped_at_top_k_pairs() {
        use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};

        // Pick a context size strictly above the cap to ensure clamping
        // actually fires.
        const OVER_CAP: usize = FALLBACK_CAP + 8;
        const EXPECTED_PAIRS: usize = FALLBACK_CAP * (FALLBACK_CAP - 1) / 2;

        let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
        let feedback = CognitiveFeedback::new(Some(Arc::clone(&synapse_store)), None);

        // OVER_CAP nodes, all with text too short to match (<=3 chars),
        // so the fallback path always fires.
        let node_ids: Vec<NodeId> = (1..=OVER_CAP as u64).map(NodeId).collect();
        let node_texts: Vec<(NodeId, Vec<String>)> = node_ids
            .iter()
            .map(|nid| (*nid, vec!["xx".into()]))
            .collect();
        let context = RagContext {
            text: "context".into(),
            estimated_tokens: 10,
            nodes_included: node_ids.len(),
            node_ids,
            node_texts,
        };
        let config = RagConfig::default();

        let stats = feedback
            .feedback(&context, "no node text matches this response at all", &config)
            .unwrap();

        // Without the cap: C(OVER_CAP, 2) pairs. With the cap:
        // C(FALLBACK_CAP, 2) pairs. The cap is what stops the cascade.
        assert_eq!(
            stats.synapses_reinforced, EXPECTED_PAIRS,
            "fallback should reinforce exactly C(FALLBACK_CAP={}, 2) = {} pairs, \
             got {} (cap broken or context smaller than expected)",
            FALLBACK_CAP, EXPECTED_PAIRS, stats.synapses_reinforced
        );
        assert_eq!(feedback.fallback_path_total(), 1);
    }

    /// Path counter sanity for the mentions branch.
    #[test]
    fn feedback_mentions_path_increments_counter() {
        use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};

        let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
        let feedback = CognitiveFeedback::new(Some(Arc::clone(&synapse_store)), None);

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

        let _stats = feedback
            .feedback(&context, "Obrain database uses WAL recovery", &config)
            .unwrap();
        assert_eq!(feedback.mentions_path_total(), 1);
        assert_eq!(feedback.fallback_path_total(), 0);
    }
}
