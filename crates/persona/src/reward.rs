use crate::PersonaDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;
use std::collections::{HashMap, HashSet};

// ═══════════════════════════════════════════════════════════════════════════════
// Ξ(t) T3 — RewardDetector: implicit token-level reward signals
// ═══════════════════════════════════════════════════════════════════════════════

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

/// Token-level reward detector. Computes reward [-1.0, +1.0] from user's response
/// to evaluate the quality of the previous LLM answer. Zero LLM decode overhead.
pub struct RewardDetector {
    /// token_id → polarity (-1.0 to +1.0), loaded from :RewardToken nodes
    pub token_polarity: HashMap<i32, f32>,
    /// Previous turn's token IDs for reformulation detection
    pub prev_query_tokens: Vec<i32>,
}

impl RewardDetector {
    /// Seed default :RewardToken nodes if none exist in PersonaDB.
    pub fn seed_default_reward_tokens(pdb: &PersonaDB) {
        let store = pdb.db.store();
        if !store.nodes_by_label("RewardToken").is_empty() {
            return;
        }

        let defaults: &[(&str, f32, &str)] = &[
            // French positive
            ("merci", 0.8, "fr"),
            ("oui", 0.3, "fr"),
            ("parfait", 0.9, "fr"),
            ("super", 0.7, "fr"),
            ("exactement", 0.8, "fr"),
            ("génial", 0.8, "fr"),
            ("excellent", 0.9, "fr"),
            ("bravo", 0.7, "fr"),
            ("bien joué", 0.7, "fr"),
            ("c'est ça", 0.6, "fr"),
            ("top", 0.6, "fr"),
            ("nickel", 0.7, "fr"),
            // French negative
            ("non", -0.3, "fr"),
            ("faux", -0.7, "fr"),
            ("incorrect", -0.8, "fr"),
            ("pas ça", -0.6, "fr"),
            ("c'est faux", -0.8, "fr"),
            ("erreur", -0.6, "fr"),
            ("nul", -0.5, "fr"),
            ("mauvais", -0.5, "fr"),
            // English positive
            ("thanks", 0.8, "en"),
            ("thank you", 0.9, "en"),
            ("yes", 0.3, "en"),
            ("perfect", 0.9, "en"),
            ("great", 0.7, "en"),
            ("excellent", 0.9, "en"),
            ("exactly", 0.8, "en"),
            ("correct", 0.6, "en"),
            ("awesome", 0.7, "en"),
            ("nice", 0.5, "en"),
            ("good", 0.4, "en"),
            // English negative
            ("no", -0.3, "en"),
            ("wrong", -0.7, "en"),
            ("incorrect", -0.8, "en"),
            ("bad", -0.5, "en"),
            ("not right", -0.6, "en"),
            ("terrible", -0.8, "en"),
            // Spanish
            ("gracias", 0.8, "es"),
            ("sí", 0.3, "es"),
            ("perfecto", 0.9, "es"),
            ("excelente", 0.9, "es"),
            ("no", -0.3, "es"),
            ("mal", -0.5, "es"),
            ("incorrecto", -0.8, "es"),
            // German
            ("danke", 0.8, "de"),
            ("ja", 0.3, "de"),
            ("perfekt", 0.9, "de"),
            ("nein", -0.3, "de"),
            ("falsch", -0.7, "de"),
            // Portuguese
            ("obrigado", 0.8, "pt"),
            ("obrigada", 0.8, "pt"),
            ("sim", 0.3, "pt"),
            ("perfeito", 0.9, "pt"),
            ("não", -0.3, "pt"),
            ("errado", -0.7, "pt"),
        ];

        for &(word, polarity, lang) in defaults {
            pdb.db.create_node_with_props(
                &["RewardToken"],
                [
                    ("word", Value::String(word.to_string().into())),
                    ("polarity", Value::Float64(polarity as f64)),
                    ("lang", Value::String(lang.to_string().into())),
                ],
            );
        }
    }

    /// Build RewardDetector by loading :RewardToken nodes and tokenizing them.
    pub fn new(pdb: &PersonaDB, engine: &dyn kv_registry::Tokenizer) -> Self {
        let store = pdb.db.store();
        let mut token_polarity: HashMap<i32, f32> = HashMap::new();

        for &nid in &store.nodes_by_label("RewardToken") {
            if let Some(node) = store.get_node(nid) {
                let word = node
                    .properties
                    .get(&PropertyKey::from("word"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let polarity = node
                    .properties
                    .get(&PropertyKey::from("polarity"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f as f32)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.0);

                if word.is_empty() {
                    continue;
                }

                // Tokenize the word — each token ID gets the polarity
                if let Ok(tokens) = engine.tokenize(word, false, false) {
                    for &tid in &tokens {
                        // If a token appears in multiple words, keep the strongest polarity
                        let entry = token_polarity.entry(tid).or_insert(0.0);
                        if polarity.abs() > entry.abs() {
                            *entry = polarity;
                        }
                    }
                }
            }
        }

        RewardDetector {
            token_polarity,
            prev_query_tokens: Vec::new(),
        }
    }

    /// Compute reward [-1.0, +1.0] for the previous turn based on the current user input.
    ///
    /// Signals (5 total, weighted):
    /// 1. Token polarity (0.30) — overlap with reward tokens from graph
    /// 2. Reformulation (0.20) — cosine > 0.85 on token bags → user is rephrasing = bad
    /// 3. Engagement (0.10) — longer sessions → small positive signal
    /// 4. Factual success (0.25) — did the response use injected facts?
    /// 5. Entropy signal (0.15) — low entropy = confident generation = good context
    ///
    /// New optional params (None = backward compatible):
    /// - `injected_facts`: the (key, value) facts that were in the header
    /// - `response_text`: the LLM response text (for fact matching)
    /// - `avg_entropy`: average entropy from GenerationSignals (passed as f32 to avoid coupling)
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

        // (1) Token polarity signal
        let mut polarity_sum: f32 = 0.0;
        let mut polarity_count: u32 = 0;
        for &tid in user_tokens {
            if let Some(&pol) = self.token_polarity.get(&tid) {
                polarity_sum += pol;
                polarity_count += 1;
            }
        }
        let token_signal = if polarity_count > 0 {
            (polarity_sum / polarity_count.max(1) as f32).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        // (2) Reformulation detection
        let reformulation_penalty = if !self.prev_query_tokens.is_empty() && !user_tokens.is_empty()
        {
            let cos = Self::bag_cosine(user_tokens, &self.prev_query_tokens);
            if cos > 0.85 { -0.3 } else { 0.0 }
        } else {
            0.0
        };

        // (3) Engagement bonus
        let engagement = 0.02 * (turn_count.min(20) as f32);

        // (4) Factual success — did the response use injected facts?
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

        // (5) Entropy signal — low entropy = confident = good context
        let entropy_signal = match avg_entropy {
            Some(e) if e < 1.5 => 0.1,  // confident generation
            Some(e) if e > 3.0 => -0.1, // very uncertain
            _ => 0.0,                   // neutral or not provided
        };

        self.prev_query_tokens = user_tokens.to_vec();

        // Weighted combination: 0.30 + 0.20 + 0.10 + 0.25 + 0.15 = 1.0
        let reward = (0.30 * token_signal
            + 0.20 * reformulation_penalty
            + 0.10 * engagement
            + 0.25 * factual_signal
            + 0.15 * entropy_signal)
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
