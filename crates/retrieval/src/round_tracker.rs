//! Round Tracker — tracks node activation history across query rounds
//! for dynamic tier promotion/demotion (Phase D).

use kv_registry::{KvNodeRegistry, KvTier};
use obrain_common::types::NodeId;
use std::collections::HashMap;

/// Record of which nodes were activated in a single round.
#[derive(Debug, Clone)]
pub struct RoundRecord {
    pub round_id: u64,
    pub activated_nodes: HashMap<NodeId, (f64, KvTier)>, // (score, tier_at_activation)
    pub query_embedding: Vec<f32>,
}

/// Describes a tier demotion to apply.
#[derive(Debug, Clone, Copy)]
pub enum DemotionType {
    AlphaToBeta,
    BetaToGamma,
    AlphaToGamma, // accelerated demotion (reward-driven, D3)
}

/// Tracks node activation across rounds for dynamic tier management.
#[derive(Debug)]
pub struct RoundTracker {
    rounds: Vec<RoundRecord>,
    current_round_id: u64,
    /// Cumulative activation counts across all rounds (for D6 fact_scores).
    activation_counts: HashMap<NodeId, u32>,
    /// Node rewards from last ablation (set by D3 reward_feedback).
    pub last_node_rewards: HashMap<NodeId, f32>,
}

impl RoundTracker {
    pub fn new() -> Self {
        Self {
            rounds: Vec::new(),
            current_round_id: 0,
            activation_counts: HashMap::new(),
            last_node_rewards: HashMap::new(),
        }
    }

    /// Record which nodes were activated in this round.
    pub fn record_round(
        &mut self,
        scored_nodes: &[(NodeId, f64)],
        query_embedding: Vec<f32>,
        registry: &KvNodeRegistry,
    ) {
        self.current_round_id += 1;

        let mut activated = HashMap::new();
        for &(nid, score) in scored_nodes {
            let tier = registry.get_tier(nid).unwrap_or(KvTier::Gamma);
            activated.insert(nid, (score, tier));
            *self.activation_counts.entry(nid).or_insert(0) += 1;
        }

        self.rounds.push(RoundRecord {
            round_id: self.current_round_id,
            activated_nodes: activated,
            query_embedding,
        });

        // Keep only last 5 rounds to bound memory
        if self.rounds.len() > 5 {
            self.rounds.remove(0);
        }
    }

    /// Detect domain shift: cosine(query_N, query_N-1) < threshold.
    pub fn detect_domain_shift(&self, threshold: f32) -> bool {
        if self.rounds.len() < 2 {
            return false;
        }
        let current = &self.rounds[self.rounds.len() - 1].query_embedding;
        let previous = &self.rounds[self.rounds.len() - 2].query_embedding;
        let cos = cosine_sim(current, previous);
        cos < threshold
    }

    /// Get nodes that should be demoted based on activation decay.
    /// - Alpha nodes not reactivated in current round → demote to Beta
    /// - Beta nodes not reactivated in last 2 rounds → demote to Gamma
    /// If domain shift detected, demote ALL non-Gamma nodes to Gamma.
    pub fn get_demotions(&self, registry: &KvNodeRegistry) -> Vec<(NodeId, DemotionType)> {
        let mut demotions = Vec::new();

        // Domain shift → reset everything to Gamma
        if self.detect_domain_shift(0.3) {
            for (nid, slot) in &registry.nodes {
                match slot.tier {
                    KvTier::Alpha => demotions.push((*nid, DemotionType::AlphaToGamma)),
                    KvTier::Beta => demotions.push((*nid, DemotionType::BetaToGamma)),
                    KvTier::Gamma => {}
                }
            }
            return demotions;
        }

        if self.rounds.is_empty() {
            return demotions;
        }

        let current = &self.rounds[self.rounds.len() - 1];

        // Alpha nodes not in current round → demote to Beta
        for (nid, slot) in &registry.nodes {
            if slot.tier == KvTier::Alpha && !current.activated_nodes.contains_key(nid) {
                // Skip Token-mode nodes (text nodes can't be demoted)
                if slot.mode == kv_registry::KvSlotMode::Token {
                    continue;
                }
                demotions.push((*nid, DemotionType::AlphaToBeta));
            }
        }

        // Beta nodes not in last 2 rounds → demote to Gamma
        if self.rounds.len() >= 2 {
            let prev = &self.rounds[self.rounds.len() - 2];
            for (nid, slot) in &registry.nodes {
                if slot.tier == KvTier::Beta
                    && !current.activated_nodes.contains_key(nid)
                    && !prev.activated_nodes.contains_key(nid)
                {
                    if slot.mode == kv_registry::KvSlotMode::Token {
                        continue;
                    }
                    demotions.push((*nid, DemotionType::BetaToGamma));
                }
            }
        }

        demotions
    }

    /// Get nodes that should be promoted (candidates not yet at Alpha).
    /// Returns node IDs ordered by score (highest first).
    pub fn get_promotions(
        &self,
        candidates: &[(NodeId, f64)],
        registry: &KvNodeRegistry,
    ) -> Vec<NodeId> {
        let mut to_promote: Vec<(NodeId, f64)> = candidates.iter()
            .filter(|(nid, _)| {
                match registry.get_tier(*nid) {
                    Some(KvTier::Gamma) | Some(KvTier::Beta) => true,
                    _ => false,
                }
            })
            .copied()
            .collect();

        // Sort by score descending (promote highest scoring first)
        to_promote.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        to_promote.into_iter().map(|(nid, _)| nid).collect()
    }

    /// Phase D3: Apply reward feedback from ablation — accelerated demote/promote.
    ///
    /// - Nodes with reward < -0.1 → accelerated demotion (Alpha→Gamma direct)
    /// - Nodes with reward > 0.2 → their 1-hop neighbors in Gamma get preemptive promote to Beta
    ///
    /// Returns (n_accelerated_demotes, n_preemptive_promotes).
    pub fn reward_feedback(
        &mut self,
        node_rewards: &HashMap<NodeId, f32>,
        adjacency: &HashMap<NodeId, std::collections::HashSet<NodeId>>,
        registry: &mut KvNodeRegistry,
        engine: &dyn kv_registry::Tokenizer,
    ) -> (u32, u32) {
        self.last_node_rewards = node_rewards.clone();
        let mut n_demoted = 0u32;
        let mut n_promoted = 0u32;

        // Accelerated demotion: negative reward nodes
        for (&nid, &reward) in node_rewards {
            if reward < -0.1 {
                if let Some(tier) = registry.get_tier(nid) {
                    if tier == KvTier::Alpha || tier == KvTier::Beta {
                        if registry.demote_to_gamma(nid, engine).is_ok() {
                            n_demoted += 1;
                        }
                    }
                }
            }
        }

        // Preemptive promote: positive reward → promote Gamma neighbors to Beta
        for (&nid, &reward) in node_rewards {
            if reward > 0.2 {
                if let Some(neighbors) = adjacency.get(&nid) {
                    for &neighbor in neighbors {
                        if !registry.tier_budget.can_promote() { break; }
                        if let Some(KvTier::Gamma) = registry.get_tier(neighbor) {
                            // Use node ID as a simple label for preemptive promote
                            let label = format!(":{}", neighbor.as_u64());
                            if registry.promote_to_beta(neighbor, &label, engine).is_ok() {
                                n_promoted += 1;
                            }
                        }
                    }
                }
            }
        }

        (n_demoted, n_promoted)
    }

    /// Get reward-adjusted scores for cosine retrieval (D3.4).
    /// Returns the last node_rewards map. Caller applies: score * (1 + 0.3 * reward).
    pub fn get_reward_adjustments(&self) -> &HashMap<NodeId, f32> {
        &self.last_node_rewards
    }

    /// Get fact_scores: normalized activation frequency per node (D6.1).
    /// Returns activation_count / max_count, normalized to [0, 1].
    pub fn get_fact_scores(&self) -> HashMap<NodeId, f32> {
        if self.activation_counts.is_empty() {
            return HashMap::new();
        }
        let max_count = *self.activation_counts.values().max().unwrap_or(&1) as f32;
        self.activation_counts.iter()
            .map(|(nid, &count)| (*nid, count as f32 / max_count))
            .collect()
    }

    /// Current round ID.
    pub fn current_round(&self) -> u64 {
        self.current_round_id
    }

    /// Number of recorded rounds.
    pub fn n_rounds(&self) -> usize {
        self.rounds.len()
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 { 0.0 } else { dot / denom }
}
