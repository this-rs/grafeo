//! Round Tracker — tracks node activation history across query rounds
//! for dynamic tier promotion/demotion (Phase D).

use kv_registry::{KvNodeRegistry, KvTier};
use obrain_common::types::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Entry tracking how often two nodes are co-activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoactivationEntry {
    pub count: u32,
    pub last_round: u32,
    pub decay_score: f32,
}

/// Map of co-activated node pairs. Key = (min(a,b), max(a,b)) to normalize.
pub type CoactivationMap = HashMap<(NodeId, NodeId), CoactivationEntry>;

const MAX_COACTIVATION_ENTRIES: usize = 10_000;
const DECAY_INTERVAL: u64 = 20;
const DECAY_FACTOR: f32 = 0.95;

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
    /// Co-activation tracking (E2).
    coactivation_map: CoactivationMap,
}

impl RoundTracker {
    pub fn new() -> Self {
        Self {
            rounds: Vec::new(),
            current_round_id: 0,
            activation_counts: HashMap::new(),
            last_node_rewards: HashMap::new(),
            coactivation_map: HashMap::new(),
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

        // E2: Co-activation tracking
        {
            let mut node_ids: Vec<NodeId> = scored_nodes.iter().map(|&(nid, _)| nid).collect();
            node_ids.sort();
            node_ids.dedup();
            let round_u32 = self.current_round_id as u32;
            for i in 0..node_ids.len() {
                for j in (i + 1)..node_ids.len() {
                    let a = node_ids[i];
                    let b = node_ids[j];
                    let key = if a < b { (a, b) } else { (b, a) };
                    let entry = self
                        .coactivation_map
                        .entry(key)
                        .or_insert(CoactivationEntry {
                            count: 0,
                            last_round: round_u32,
                            decay_score: 0.0,
                        });
                    entry.count += 1;
                    entry.last_round = round_u32;
                    entry.decay_score += 1.0;
                }
            }

            // Decay every DECAY_INTERVAL rounds
            if self.current_round_id % DECAY_INTERVAL == 0 {
                self.coactivation_map.retain(|_, e| {
                    e.decay_score *= DECAY_FACTOR;
                    e.decay_score >= 0.01
                });
            }

            // Purge if over cap
            if self.coactivation_map.len() > MAX_COACTIVATION_ENTRIES {
                let mut entries: Vec<((NodeId, NodeId), f32)> = self
                    .coactivation_map
                    .iter()
                    .map(|(&k, v)| (k, v.decay_score))
                    .collect();
                entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let to_remove = self.coactivation_map.len() - MAX_COACTIVATION_ENTRIES;
                for (key, _) in entries.into_iter().take(to_remove) {
                    self.coactivation_map.remove(&key);
                }
            }
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
        let mut to_promote: Vec<(NodeId, f64)> = candidates
            .iter()
            .filter(|(nid, _)| match registry.get_tier(*nid) {
                Some(KvTier::Gamma) | Some(KvTier::Beta) => true,
                _ => false,
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
                        if !registry.tier_budget.can_promote() {
                            break;
                        }
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
        self.activation_counts
            .iter()
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

    // ── E2: Co-activation API ──────────────────────────────────────

    /// Get top-k nodes most co-activated with the given node, sorted by decay_score desc.
    pub fn get_top_coactivated(&self, node_id: NodeId, top_k: usize) -> Vec<(NodeId, f32)> {
        let mut results: Vec<(NodeId, f32)> = self
            .coactivation_map
            .iter()
            .filter_map(|(&(a, b), entry)| {
                if a == node_id {
                    Some((b, entry.decay_score))
                } else if b == node_id {
                    Some((a, entry.decay_score))
                } else {
                    None
                }
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Total number of co-activation events (sum of all counts).
    pub fn total_coactivation_events(&self) -> u64 {
        self.coactivation_map.values().map(|e| e.count as u64).sum()
    }

    /// Number of unique pairs with count >= threshold.
    pub fn unique_pairs_above(&self, min_count: u32) -> usize {
        self.coactivation_map
            .values()
            .filter(|e| e.count >= min_count)
            .count()
    }

    /// Get a reference to the coactivation map (for E3 affinity expansion).
    pub fn coactivation(&self) -> &CoactivationMap {
        &self.coactivation_map
    }

    /// Save coactivation map to a file (bincode format).
    pub fn save_coactivations(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let data = bincode::serialize(&self.coactivation_map)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load coactivation map from a file. Returns empty map if file doesn't exist.
    pub fn load_coactivations(&mut self, path: &std::path::Path) -> anyhow::Result<usize> {
        if !path.exists() {
            return Ok(0);
        }
        let data = std::fs::read(path)?;
        self.coactivation_map = bincode::deserialize(&data)?;
        let n = self.coactivation_map.len();
        Ok(n)
    }

    /// Purge entries referencing NodeIds not in the provided set (for --db changes).
    pub fn validate_coactivations(
        &mut self,
        valid_ids: &std::collections::HashSet<NodeId>,
    ) -> usize {
        let before = self.coactivation_map.len();
        self.coactivation_map
            .retain(|&(a, b), _| valid_ids.contains(&a) && valid_ids.contains(&b));
        before - self.coactivation_map.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> KvNodeRegistry {
        KvNodeRegistry::new("test", 1)
    }

    fn n(id: u64) -> NodeId {
        NodeId(id)
    }

    fn record(tracker: &mut RoundTracker, reg: &KvNodeRegistry, ids: &[u64]) {
        let scored: Vec<(NodeId, f64)> = ids.iter().map(|&id| (n(id), 1.0)).collect();
        tracker.record_round(&scored, vec![0.0], reg);
    }

    #[test]
    fn test_coactivation_basic() {
        let reg = make_registry();
        let mut t = RoundTracker::new();

        // 3 rounds with nodes {1,2,3}
        for _ in 0..3 {
            record(&mut t, &reg, &[1, 2, 3]);
        }

        // 3 pairs: (1,2), (1,3), (2,3) — each with count=3
        assert_eq!(t.coactivation_map.len(), 3);
        assert_eq!(t.total_coactivation_events(), 9); // 3 pairs * 3 counts
        assert_eq!(t.unique_pairs_above(3), 3);
        assert_eq!(t.unique_pairs_above(4), 0);

        for entry in t.coactivation_map.values() {
            assert_eq!(entry.count, 3);
            assert!((entry.decay_score - 3.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_coactivation_decay() {
        let reg = make_registry();
        let mut t = RoundTracker::new();

        // Record once with nodes 1,2
        record(&mut t, &reg, &[1, 2]);
        let initial_score = t.coactivation_map[&(n(1), n(2))].decay_score;
        assert!((initial_score - 1.0).abs() < f32::EPSILON);

        // Advance to round 20 (DECAY_INTERVAL) with empty rounds to trigger decay
        // We're at round 1, need to get to round 20
        for _ in 0..19 {
            record(&mut t, &reg, &[]); // no nodes = no pairs
        }

        // Round 20 triggers decay: 1.0 * 0.95 = 0.95
        let decayed = t.coactivation_map[&(n(1), n(2))].decay_score;
        assert!(
            decayed < initial_score,
            "decay_score should decrease: {decayed}"
        );
        assert!((decayed - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_get_top_coactivated() {
        let reg = make_registry();
        let mut t = RoundTracker::new();

        // Node 1 co-activated with 2 three times, with 3 once
        for _ in 0..3 {
            record(&mut t, &reg, &[1, 2]);
        }
        record(&mut t, &reg, &[1, 3]);

        let top = t.get_top_coactivated(n(1), 10);
        assert_eq!(top.len(), 2);
        // (1,2) has decay_score=3.0, (1,3) has decay_score=1.0
        assert_eq!(top[0].0, n(2));
        assert_eq!(top[1].0, n(3));
        assert!(top[0].1 > top[1].1);

        // top_k=1 should limit
        let top1 = t.get_top_coactivated(n(1), 1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0, n(2));
    }

    #[test]
    fn test_coactivation_purge_cap() {
        let reg = make_registry();
        let mut t = RoundTracker::new();

        // Generate more than MAX_COACTIVATION_ENTRIES pairs.
        // C(n,2) > 10_000 when n >= 142 (142*141/2 = 10_011)
        let ids: Vec<u64> = (1..=142).collect();
        record(&mut t, &reg, &ids);

        assert!(
            t.coactivation_map.len() <= MAX_COACTIVATION_ENTRIES,
            "map size {} should be <= {}",
            t.coactivation_map.len(),
            MAX_COACTIVATION_ENTRIES
        );
    }

    #[test]
    fn test_validate_coactivations() {
        let reg = make_registry();
        let mut t = RoundTracker::new();
        record(&mut t, &reg, &[1, 2, 3]);

        let mut valid: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        valid.insert(n(1));
        valid.insert(n(2));
        // Node 3 is NOT valid — entries (1,3) and (2,3) should be purged

        let purged = t.validate_coactivations(&valid);
        assert_eq!(purged, 2); // (1,3) and (2,3) removed
        assert_eq!(t.coactivation_map.len(), 1); // only (1,2) remains
        assert!(t.coactivation_map.contains_key(&(n(1), n(2))));
    }

    #[test]
    fn test_persistence_roundtrip() {
        let reg = make_registry();
        let mut t = RoundTracker::new();
        record(&mut t, &reg, &[1, 2, 3]);
        record(&mut t, &reg, &[2, 3, 4]);

        let dir = std::env::temp_dir().join("obrain_test_coact");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("coact.bin");

        t.save_coactivations(&path).unwrap();

        let mut t2 = RoundTracker::new();
        let loaded = t2.load_coactivations(&path).unwrap();
        assert_eq!(loaded, t.coactivation_map.len());

        // Every entry should match
        for (key, entry) in &t.coactivation_map {
            let e2 = t2.coactivation_map.get(key).expect("key should exist");
            assert_eq!(entry.count, e2.count);
            assert_eq!(entry.last_round, e2.last_round);
            assert!((entry.decay_score - e2.decay_score).abs() < f32::EPSILON);
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
