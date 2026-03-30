use crate::PersonaDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════════
// Ξ(t) T3 — RewardDetector: structural reward signals (zero-seed)
//
// Architecture: 100% language-agnostic. No hardcoded reward tokens.
// All signals are structural (reformulation, engagement, factual success,
// generation entropy). Works in any language from turn 1.
//
// T4 — Token Polarity Auto-Learning: after each turn, observe() accumulates
// an EMA of structural reward per token_id. After ≥20 observations, discovered
// polarities feed back as a 5th signal. This auto-discovers "merci" = positive,
// "faux" = negative in any language.
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum observations before a token's polarity is considered reliable.
const MIN_OBSERVATIONS: u32 = 20;

/// EMA smoothing factor for token polarity updates.
const EMA_ALPHA: f32 = 0.15;

/// Token Polarity Auto-Learner.
///
/// Accumulates an exponential moving average (EMA) of structural reward per
/// token_id. After `MIN_OBSERVATIONS` observations, the polarity is considered
/// reliable and contributes to the composite reward.
///
/// Zero-seed: starts empty. Discovers language-specific reward tokens
/// autonomously from conversation structure (e.g., "merci" → positive,
/// "reformuler" → negative).
pub struct TokenPolarityLearner {
    /// token_id → (ema_polarity, observation_count)
    polarities: HashMap<i32, (f32, u32)>,
}

impl TokenPolarityLearner {
    pub fn new() -> Self {
        TokenPolarityLearner {
            polarities: HashMap::new(),
        }
    }

    /// Observe a set of tokens with their associated structural reward.
    /// Updates the EMA for each unique token in the input.
    pub fn observe(&mut self, tokens: &[i32], reward: f32) {
        // Deduplicate tokens to avoid over-counting repeated tokens in one turn
        let unique: HashSet<i32> = tokens.iter().copied().collect();
        for tid in unique {
            let entry = self.polarities.entry(tid).or_insert((0.0, 0));
            entry.1 += 1; // count
            // EMA update: polarity = α * reward + (1-α) * polarity
            entry.0 = EMA_ALPHA * reward + (1.0 - EMA_ALPHA) * entry.0;
        }
    }

    /// Get the polarity signal for a set of tokens.
    /// Returns the mean polarity of tokens that have ≥ MIN_OBSERVATIONS,
    /// or None if no token has enough observations.
    pub fn get_polarity(&self, tokens: &[i32]) -> Option<f32> {
        let unique: HashSet<i32> = tokens.iter().copied().collect();
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for tid in unique {
            if let Some(&(pol, obs)) = self.polarities.get(&tid) {
                if obs >= MIN_OBSERVATIONS {
                    sum += pol;
                    count += 1;
                }
            }
        }
        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    /// Number of distinct tokens being tracked.
    pub fn tracked_count(&self) -> usize {
        self.polarities.len()
    }

    /// Number of tokens with enough observations to be reliable.
    pub fn reliable_count(&self) -> usize {
        self.polarities.values().filter(|(_, obs)| *obs >= MIN_OBSERVATIONS).count()
    }

    /// Export all polarities (for persistence).
    pub fn export(&self) -> &HashMap<i32, (f32, u32)> {
        &self.polarities
    }

    /// Import polarities (from persistence).
    pub fn import(&mut self, data: HashMap<i32, (f32, u32)>) {
        self.polarities = data;
    }
}

/// Decomposed reward signals — allows callers to inspect individual components.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct RewardSignals {
    /// Composite reward [-1.0, +1.0] (weighted sum of all signals)
    pub reward: f32,
    /// Factual success signal: did the response use injected facts?
    /// This is the "mask quality" indicator — positive means the topology mask
    /// guided attention to relevant facts that the model actually used.
    pub factual_signal: f32,
}

impl std::fmt::Display for RewardSignals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RewardSignals(reward={:.4}, factual={:.4})",
            self.reward, self.factual_signal
        )
    }
}

/// Structural reward detector. Computes reward [-1.0, +1.0] from conversation
/// structure alone — no hardcoded token lists, works in any language.
///
/// Includes an optional token polarity learner (T4) that auto-discovers
/// reward-associated tokens from structural signals.
pub struct RewardDetector {
    /// Previous turn's token IDs for reformulation detection
    pub prev_query_tokens: Vec<i32>,
    /// T4: Auto-learned token polarities from structural reward
    pub token_learner: TokenPolarityLearner,
}

impl RewardDetector {
    /// Create a new RewardDetector. No seeds, no tokenizer needed.
    pub fn new() -> Self {
        RewardDetector {
            prev_query_tokens: Vec::new(),
            token_learner: TokenPolarityLearner::new(),
        }
    }

    /// Compute reward [-1.0, +1.0] for the previous turn based on the current user input.
    ///
    /// Signals (5 total, weighted — all structural/learned, zero hardcoded lexical dependency):
    /// 1. Reformulation (0.30) — cosine > 0.85 on token bags → user is rephrasing = bad
    /// 2. Engagement (0.10) — longer sessions → small positive signal
    /// 3. Factual success (0.20) — did the response use injected facts/memories?
    /// 4. Entropy signal (0.25) — low entropy = confident generation = good context
    /// 5. Token polarity (0.15) — auto-learned EMA per token_id (T4, kicks in after ≥20 obs)
    ///
    /// Optional params (None = backward compatible):
    /// - `injected_facts`: the (key, value) facts that were in the header
    /// - `response_text`: the LLM response text (for fact matching)
    /// - `avg_entropy`: average entropy from GenerationSignals
    pub fn compute_reward(
        &mut self,
        user_tokens: &[i32],
        turn_count: u32,
        injected_facts: Option<&[(String, String)]>,
        response_text: Option<&str>,
        avg_entropy: Option<f32>,
    ) -> RewardSignals {
        if turn_count <= 1 {
            self.prev_query_tokens = user_tokens.to_vec();
            return RewardSignals {
                reward: 0.0,
                factual_signal: 0.0,
            };
        }

        // (1) Reformulation detection
        let reformulation_penalty = if !self.prev_query_tokens.is_empty() && !user_tokens.is_empty()
        {
            let cos = Self::bag_cosine(user_tokens, &self.prev_query_tokens);
            if cos > 0.85 { -0.3 } else { 0.0 }
        } else {
            0.0
        };

        // (2) Engagement bonus
        let engagement = 0.02 * (turn_count.min(20) as f32);

        // (3) Factual success — did the response use injected facts?
        let factual_signal = match (injected_facts, response_text) {
            (Some(facts), Some(resp)) if !facts.is_empty() => {
                let resp_lower = resp.to_lowercase();
                let mut matches = 0u32;
                for (_key, value) in facts {
                    if value.len() >= 2 && resp_lower.contains(&value.to_lowercase()) {
                        matches += 1;
                    }
                }
                if matches > 0 {
                    (0.15 * matches as f32).min(0.3) // cap at +0.3
                } else {
                    -0.1 // facts injected but none used
                }
            }
            _ => 0.0, // no facts or no response → neutral
        };

        // (4) Entropy signal — low entropy = confident = good context
        let entropy_signal = match avg_entropy {
            Some(e) if e < 1.5 => 0.1,  // confident generation
            Some(e) if e > 3.0 => -0.1, // very uncertain
            _ => 0.0,                   // neutral or not provided
        };

        // (5) Token polarity — auto-learned from structural reward (T4)
        let polarity_signal = self.token_learner.get_polarity(user_tokens).unwrap_or(0.0);

        self.prev_query_tokens = user_tokens.to_vec();

        // Weighted combination: 0.30 + 0.10 + 0.20 + 0.25 + 0.15 = 1.0
        // Note: factual went from 0.35 to 0.20 when T4 polarity (0.15) was added
        let reward = (0.30 * reformulation_penalty
            + 0.10 * engagement
            + 0.20 * factual_signal
            + 0.25 * entropy_signal
            + 0.15 * polarity_signal)
            .clamp(-1.0, 1.0);

        RewardSignals {
            reward,
            factual_signal,
        }
    }

    /// Cosine similarity between two bags of token IDs.
    pub fn bag_cosine(a: &[i32], b: &[i32]) -> f32 {
        let mut freq_a: HashMap<i32, u32> = HashMap::new();
        let mut freq_b: HashMap<i32, u32> = HashMap::new();
        for &t in a {
            *freq_a.entry(t).or_insert(0) += 1;
        }
        for &t in b {
            *freq_b.entry(t).or_insert(0) += 1;
        }

        let mut dot: f64 = 0.0;
        let mut norm_a: f64 = 0.0;
        let mut norm_b: f64 = 0.0;

        let all_keys: HashSet<i32> = freq_a.keys().chain(freq_b.keys()).copied().collect();
        for k in all_keys {
            let va = *freq_a.get(&k).unwrap_or(&0) as f64;
            let vb = *freq_b.get(&k).unwrap_or(&0) as f64;
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
        (dot / denom) as f32
    }

    /// Propagate reward into the persona graph:
    /// - Update :ConvTurn.reward
    /// - Adjust energy of :Fact nodes that were USED_IN that turn
    /// - Update avg_reward of :Pattern nodes that EXTRACTS used facts
    pub fn propagate_reward(
        &self,
        pdb: &PersonaDB,
        conv_turn_id: NodeId,
        reward: f32,
        used_fact_ids: &[NodeId],
    ) {
        self.propagate_reward_ex(pdb, conv_turn_id, reward, used_fact_ids, None);
    }

    /// Extended propagation with optional mask_reward (factual_signal from RewardSignals).
    /// Stores mask_reward on ConvTurn for retrospective analysis of mask quality.
    pub fn propagate_reward_ex(
        &self,
        pdb: &PersonaDB,
        conv_turn_id: NodeId,
        reward: f32,
        used_fact_ids: &[NodeId],
        mask_reward: Option<f32>,
    ) {
        // (1) Set reward on the ConvTurn
        pdb.db
            .set_node_property(conv_turn_id, "reward", Value::Float64(reward as f64));
        // (1b) Store mask quality signal separately for A/B analysis
        if let Some(mr) = mask_reward {
            pdb.db
                .set_node_property(conv_turn_id, "mask_reward", Value::Float64(mr as f64));
        }

        let store = pdb.db.store();

        // (2) Adjust energy + utility of facts USED_IN this turn
        for &fid in used_fact_ids {
            if let Some(node) = store.get_node(fid) {
                let energy = node
                    .properties
                    .get(&PropertyKey::from("energy"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0);
                let new_energy = (energy + 0.1 * reward as f64).clamp(0.0, 2.0);
                pdb.db
                    .set_node_property(fid, "energy", Value::Float64(new_energy));

                // Ξ(t) T5: Update utility (incremental mean) and cost_efficiency
                let act_count = node
                    .properties
                    .get(&PropertyKey::from("activation_count"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                let utility = node
                    .properties
                    .get(&PropertyKey::from("utility"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.5);
                let token_cost = node
                    .properties
                    .get(&PropertyKey::from("token_cost"))
                    .and_then(|v| {
                        if let Value::Int64(n) = v {
                            Some(*n)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1)
                    .max(1);

                let new_count = act_count + 1;
                let new_utility = (utility * act_count as f64 + reward as f64) / new_count as f64;
                let new_efficiency = new_utility / token_cost as f64;

                pdb.db
                    .set_node_property(fid, "activation_count", Value::Int64(new_count));
                pdb.db
                    .set_node_property(fid, "utility", Value::Float64(new_utility));
                pdb.db
                    .set_node_property(fid, "cost_efficiency", Value::Float64(new_efficiency));
            }
        }

        // (3) Update avg_reward of :Pattern nodes that produced used facts
        for &fid in used_fact_ids {
            // Follow EXTRACTS edges backwards: Pattern -EXTRACTS-> Fact
            // We need to find patterns that point TO this fact
            for &pnid in &store.nodes_by_label("Pattern") {
                let points_to_fact = store
                    .edges_from(pnid, Direction::Outgoing)
                    .any(|(target, _)| target == fid);
                if points_to_fact {
                    if let Some(pnode) = store.get_node(pnid) {
                        let hits = pnode
                            .properties
                            .get(&PropertyKey::from("hit_count"))
                            .and_then(|v| {
                                if let Value::Int64(n) = v {
                                    Some(*n)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(1)
                            .max(1);
                        let avg = pnode
                            .properties
                            .get(&PropertyKey::from("avg_reward"))
                            .and_then(|v| {
                                if let Value::Float64(f) = v {
                                    Some(*f)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(0.0);
                        let new_avg = (avg * (hits - 1) as f64 + reward as f64) / hits as f64;
                        pdb.db
                            .set_node_property(pnid, "avg_reward", Value::Float64(new_avg));
                    }
                }
            }
        }

        // (4) REINFORCES between co-used facts if reward is high
        pdb.create_reinforces(used_fact_ids, reward as f64);
    }
}
