//! FormulaSelector: contextual selection of attention formulas.
//!
//! Scores candidate formulas using:
//!   score = 0.5 × relevance + 0.35 × track_record - 0.15 × cost_penalty
//!
//! Then samples via softmax with adaptive temperature:
//! - T=2.0 early (exploration), decays toward T=0.3 (exploitation).

use crate::attn_dsl::{self, AttnOp};

/// A candidate formula with its metadata from PersonaDB.
#[derive(Debug, Clone)]
pub struct FormulaCandidate {
    /// Formula name (e.g., "F0-Identity").
    pub name: String,
    /// JSON-serialized AttnOp DSL.
    pub dsl_json: String,
    /// Running average reward when this formula was active.
    pub avg_reward: f64,
    /// Ξ(t) energy level [0, 2].
    pub energy: f64,
    /// Total times this formula was selected.
    pub activation_count: i64,
    /// Evolution generation (0=seed).
    pub generation: i64,
    /// Context affinity tags.
    pub context_affinity: Vec<String>,
    /// Whether this formula is active.
    pub active: bool,
}

/// Result of formula selection.
#[derive(Debug, Clone)]
pub struct SelectedFormula {
    /// The selected AttnOp, deserialized from JSON.
    pub op: AttnOp,
    /// Name of the selected formula.
    pub name: String,
    /// Index into the candidates list.
    pub candidate_index: usize,
    /// Score that was computed for this formula.
    pub score: f64,
}

/// Adaptive temperature for softmax sampling.
///
/// Starts at `T_MAX` (exploration) and decays toward `T_MIN` as
/// total activations across all formulas increase.
#[derive(Debug, Clone)]
pub struct AdaptiveTemperature {
    /// Maximum temperature (high exploration).
    pub t_max: f64,
    /// Minimum temperature (exploitation).
    pub t_min: f64,
    /// Activation count at which temperature reaches midpoint.
    pub midpoint_activations: f64,
}

impl Default for AdaptiveTemperature {
    fn default() -> Self {
        Self {
            t_max: 2.0,
            t_min: 0.3,
            midpoint_activations: 100.0,
        }
    }
}

impl AdaptiveTemperature {
    /// Compute temperature from total activations across all formulas.
    ///
    /// Uses sigmoid decay: T = T_min + (T_max - T_min) / (1 + total/midpoint)
    pub fn temperature(&self, total_activations: i64) -> f64 {
        let t = total_activations as f64;
        self.t_min + (self.t_max - self.t_min) / (1.0 + t / self.midpoint_activations)
    }
}

/// FormulaSelector: scores and selects attention formulas.
pub struct FormulaSelector {
    /// Temperature schedule for softmax sampling.
    pub temperature: AdaptiveTemperature,
    /// Maximum estimated cost used for normalization.
    pub max_cost: f32,
}

impl Default for FormulaSelector {
    fn default() -> Self {
        Self {
            temperature: AdaptiveTemperature::default(),
            max_cost: 10.0, // QueryDelegate(5) + BiasAdd(2) = 7; 10 gives headroom
        }
    }
}

/// Scoring weights (from RFC §2.2).
const W_RELEVANCE: f64 = 0.50;
const W_TRACK_RECORD: f64 = 0.35;
const W_COST_PENALTY: f64 = 0.15;

impl FormulaSelector {
    /// Create a new selector with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Score a single formula candidate given the current context.
    ///
    /// Returns: `0.5 × relevance + 0.35 × track_record - 0.15 × cost_penalty`
    pub fn score(
        &self,
        candidate: &FormulaCandidate,
        context_tags: &[&str],
    ) -> f64 {
        // ── Relevance: context affinity match ──
        // Count how many of the candidate's affinity tags match the context.
        let affinity_matches = if candidate.context_affinity.is_empty() || context_tags.is_empty() {
            0.5 // neutral: no tags = equally relevant to everything
        } else {
            let matches = candidate
                .context_affinity
                .iter()
                .filter(|tag| context_tags.contains(&tag.as_str()))
                .count();
            (matches as f64 / candidate.context_affinity.len() as f64).max(0.1)
        };

        // ── Track record: avg_reward × energy ──
        // Normalize to [0, 1]: avg_reward ∈ [-1, 1], energy ∈ [0, 2]
        // shift reward to [0, 1], energy to [0, 1]
        let reward_norm = (candidate.avg_reward + 1.0) / 2.0; // [-1,1] → [0,1]
        let energy_norm = candidate.energy / 2.0; // [0,2] → [0,1]
        let track_record = reward_norm * energy_norm;

        // ── Cost penalty ──
        let cost = match serde_json::from_str::<AttnOp>(&candidate.dsl_json) {
            Ok(op) => attn_dsl::estimated_cost(&op) / self.max_cost,
            Err(_) => 1.0, // unparseable = maximum penalty
        };

        W_RELEVANCE * affinity_matches + W_TRACK_RECORD * track_record - W_COST_PENALTY * cost as f64
    }

    /// Select a formula from candidates via softmax sampling.
    ///
    /// Uses adaptive temperature and a deterministic seed for reproducibility.
    /// Falls back to Identity if no viable candidate exists.
    pub fn select(
        &self,
        candidates: &[FormulaCandidate],
        context_tags: &[&str],
        seed: u64,
    ) -> SelectedFormula {
        // Filter active candidates with energy > 0
        let viable: Vec<(usize, &FormulaCandidate)> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.active && c.energy > 0.0)
            .collect();

        // Fallback: no viable candidates → Identity
        if viable.is_empty() {
            return SelectedFormula {
                op: AttnOp::Identity,
                name: "fallback-Identity".to_string(),
                candidate_index: 0,
                score: 0.0,
            };
        }

        // Compute total activations for temperature
        let total_activations: i64 = candidates.iter().map(|c| c.activation_count).sum();
        let temp = self.temperature.temperature(total_activations);

        // Score all viable candidates
        let scores: Vec<f64> = viable
            .iter()
            .map(|(_, c)| self.score(c, context_tags))
            .collect();

        // Softmax with temperature
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores
            .iter()
            .map(|&s| ((s - max_score) / temp).exp())
            .collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            // Numerical issue → fallback
            return SelectedFormula {
                op: AttnOp::Identity,
                name: "fallback-Identity".to_string(),
                candidate_index: 0,
                score: 0.0,
            };
        }

        let probs: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Deterministic sampling from seed
        let r = pseudo_random(seed);
        let mut cumulative = 0.0;
        let mut selected_idx = viable.len() - 1; // default: last
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                selected_idx = i;
                break;
            }
        }

        let (orig_idx, candidate) = &viable[selected_idx];
        let op = serde_json::from_str::<AttnOp>(&candidate.dsl_json)
            .unwrap_or(AttnOp::Identity);

        SelectedFormula {
            op,
            name: candidate.name.clone(),
            candidate_index: *orig_idx,
            score: scores[selected_idx],
        }
    }
}

/// Simple pseudo-random in [0, 1) from a u64 seed.
/// Splitmix64 single iteration, then normalize.
fn pseudo_random(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^= z >> 31;
    (z as f64) / (u64::MAX as f64)
}

// ─── Natural Language Explanation ────────────────────────────────────────────

/// Translate an AttnOp formula into a human-readable description.
///
/// Used by :Self nodes so the model can describe its own attention strategy.
pub fn dsl_to_natural_language(op: &AttnOp) -> String {
    use crate::attn_dsl::*;
    match op {
        AttnOp::Identity => "Attention standard (pas de modification)".to_string(),
        AttnOp::Mask { condition } => {
            let cond = match condition {
                MaskCondition::GraphDistanceAbove(d) => {
                    format!("masque les tokens à plus de {} hops dans le graphe", d)
                }
                MaskCondition::EnergyBelow(t) => {
                    format!("masque les connexions à énergie synaptique < {:.1}", t)
                }
                MaskCondition::NoPath => {
                    "masque les paires de nœuds sans chemin".to_string()
                }
            };
            format!("Filtre sélectif : {}", cond)
        }
        AttnOp::BiasAdd { source, weight } => {
            let src = bias_source_nl(source);
            if *weight > 0.0 {
                format!("Amplifie l'attention via {} (×{:.1})", src, weight)
            } else {
                format!("Atténue l'attention via {} (×{:.1})", src, weight)
            }
        }
        AttnOp::BiasScale { source, weight } => {
            let src = bias_source_nl(source);
            format!("Mise à l'échelle ×{:.1} basée sur {}", weight, src)
        }
        AttnOp::WarpQ { delta_source, alpha } => {
            let src = delta_source_nl(delta_source);
            format!(
                "Déformation query via {} (α={:.1}) — modifie ce que le modèle cherche",
                src, alpha
            )
        }
        AttnOp::WarpK { delta_source, alpha } => {
            let src = delta_source_nl(delta_source);
            format!(
                "Déformation key via {} (α={:.1}) — modifie ce qui est trouvé",
                src, alpha
            )
        }
        AttnOp::QueryDelegate {
            entropy_threshold,
            query_type,
            max_inject_tokens,
        } => {
            let qt = query_type_nl(query_type);
            format!(
                "Délègue une sous-requête ({}) si entropie > {:.1} (max {} tokens)",
                qt, entropy_threshold, max_inject_tokens
            )
        }
        AttnOp::Sequence(ops) => {
            if ops.len() == 1 {
                dsl_to_natural_language(&ops[0])
            } else {
                let descs: Vec<String> = ops.iter().map(|o| dsl_to_natural_language(o)).collect();
                format!("Séquence : {}", descs.join(" → "))
            }
        }
        AttnOp::PerHead(mappings) => {
            format!(
                "Routage par tête : {} groupes avec stratégies différentes",
                mappings.len()
            )
        }
        AttnOp::Conditional {
            condition,
            then_op,
            else_op,
        } => {
            let cond = runtime_cond_nl(condition);
            format!(
                "Si {} : {} ; sinon : {}",
                cond,
                dsl_to_natural_language(then_op),
                dsl_to_natural_language(else_op)
            )
        }
    }
}

fn bias_source_nl(source: &crate::attn_dsl::BiasSource) -> String {
    use crate::attn_dsl::BiasSource;
    match source {
        BiasSource::GraphDistance { max_hops } => {
            format!("distance graphe (max {} hops)", max_hops)
        }
        BiasSource::SynapseEnergy => "énergie synaptique".to_string(),
        BiasSource::Coactivation => "fréquence de co-activation".to_string(),
        BiasSource::TemporalDecay { half_life } => {
            format!("décroissance temporelle (t½={:.0})", half_life)
        }
        BiasSource::GnnPairScore { layer } => {
            format!("score GNN couche {}", layer)
        }
        BiasSource::Constant(v) => format!("constante {:.1}", v),
        BiasSource::Product(a, b) => {
            format!("{} × {}", bias_source_nl(a), bias_source_nl(b))
        }
    }
}

fn delta_source_nl(source: &crate::attn_dsl::DeltaSource) -> String {
    use crate::attn_dsl::DeltaSource;
    match source {
        DeltaSource::GnnDelta => "GNN".to_string(),
        DeltaSource::NeighborMean { edge_type, .. } => {
            if let Some(et) = edge_type {
                format!("moyenne voisins ({})", et)
            } else {
                "moyenne voisins".to_string()
            }
        }
        DeltaSource::CausalChain { depth, decay } => {
            format!("chaîne causale (profondeur={}, decay={:.1})", depth, decay)
        }
    }
}

fn query_type_nl(qt: &crate::attn_dsl::QueryType) -> String {
    use crate::attn_dsl::QueryType;
    match qt {
        QueryType::Lookup => "lookup".to_string(),
        QueryType::Neighbors { .. } => "voisins".to_string(),
        QueryType::CausalTrace { .. } => "trace causale".to_string(),
        QueryType::Compute => "calcul".to_string(),
        QueryType::FreeformGQL => "requête GQL libre".to_string(),
    }
}

fn runtime_cond_nl(cond: &crate::attn_dsl::RuntimeCondition) -> String {
    use crate::attn_dsl::RuntimeCondition;
    match cond {
        RuntimeCondition::Uncertainty { threshold } => {
            format!("incertitude > {:.1}", threshold)
        }
        RuntimeCondition::TokenCount { min, max } => {
            format!("tokens dans [{}, {}]", min, max)
        }
        RuntimeCondition::ContextType(ct) => format!("contexte = '{}'", ct),
        RuntimeCondition::GraphDensity { threshold } => {
            format!("densité graphe > {:.2}", threshold)
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        name: &str,
        dsl_json: &str,
        avg_reward: f64,
        energy: f64,
        activation_count: i64,
        affinity: &[&str],
    ) -> FormulaCandidate {
        FormulaCandidate {
            name: name.to_string(),
            dsl_json: dsl_json.to_string(),
            avg_reward,
            energy,
            activation_count,
            generation: 0,
            context_affinity: affinity.iter().map(|s| s.to_string()).collect(),
            active: true,
        }
    }

    // ── Scoring ──

    #[test]
    fn test_score_high_reward_low_cost_beats_inverse() {
        let selector = FormulaSelector::new();

        let good = make_candidate(
            "good",
            r#""Identity""#,
            0.8,  // high reward
            1.5,  // high energy
            50,
            &["math"],
        );
        let bad = make_candidate(
            "bad",
            r#"{"QueryDelegate":{"entropy_threshold":1.0,"query_type":"Compute","max_inject_tokens":64}}"#,
            -0.5, // low reward
            0.3,  // low energy
            50,
            &["creative"],
        );

        let context = &["math"];
        let score_good = selector.score(&good, context);
        let score_bad = selector.score(&bad, context);

        assert!(
            score_good > score_bad,
            "good={score_good:.4} should beat bad={score_bad:.4}"
        );
    }

    #[test]
    fn test_score_affinity_matters() {
        let selector = FormulaSelector::new();

        let matching = make_candidate("match", r#""Identity""#, 0.0, 1.0, 10, &["math"]);
        let non_matching = make_candidate("nomatch", r#""Identity""#, 0.0, 1.0, 10, &["creative"]);

        let score_match = selector.score(&matching, &["math"]);
        let score_nomatch = selector.score(&non_matching, &["math"]);

        assert!(
            score_match > score_nomatch,
            "matching={score_match:.4} should beat non-matching={score_nomatch:.4}"
        );
    }

    #[test]
    fn test_score_no_tags_neutral() {
        let selector = FormulaSelector::new();
        let c1 = make_candidate("a", r#""Identity""#, 0.0, 1.0, 10, &[]);
        let c2 = make_candidate("b", r#""Identity""#, 0.0, 1.0, 10, &["math"]);

        // No context tags → both get neutral affinity
        let s1 = selector.score(&c1, &[]);
        let s2 = selector.score(&c2, &[]);

        // Both should have the same relevance component (0.5)
        assert!((s1 - s2).abs() < 0.01, "s1={s1:.4} s2={s2:.4}");
    }

    // ── Temperature ──

    #[test]
    fn test_temperature_initial() {
        let at = AdaptiveTemperature::default();
        assert!((at.temperature(0) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_temperature_decays() {
        let at = AdaptiveTemperature::default();
        let t0 = at.temperature(0);
        let t100 = at.temperature(100);
        let t500 = at.temperature(500);

        assert!(t0 > t100, "t0={t0:.2} > t100={t100:.2}");
        assert!(t100 > t500, "t100={t100:.2} > t500={t500:.2}");
        assert!(t500 > 0.3, "t500={t500:.2} > 0.3 (min)");
        assert!(t500 < 1.0, "t500={t500:.2} < 1.0 (should be well decayed)");
    }

    #[test]
    fn test_temperature_converges_to_min() {
        let at = AdaptiveTemperature::default();
        let t_huge = at.temperature(100_000);
        assert!(
            (t_huge - 0.3).abs() < 0.05,
            "t_huge={t_huge:.4} should be near 0.3"
        );
    }

    // ── Selection ──

    #[test]
    fn test_select_empty_fallback() {
        let selector = FormulaSelector::new();
        let result = selector.select(&[], &["math"], 42);
        assert_eq!(result.name, "fallback-Identity");
        assert_eq!(result.op, AttnOp::Identity);
    }

    #[test]
    fn test_select_all_dead_fallback() {
        let selector = FormulaSelector::new();
        let mut c = make_candidate("dead", r#""Identity""#, 0.5, 0.0, 100, &[]);
        c.active = true; // active but energy=0 → not viable
        let result = selector.select(&[c], &[], 42);
        assert_eq!(result.name, "fallback-Identity");
    }

    #[test]
    fn test_select_all_inactive_fallback() {
        let selector = FormulaSelector::new();
        let mut c = make_candidate("inactive", r#""Identity""#, 0.5, 1.0, 100, &[]);
        c.active = false;
        let result = selector.select(&[c], &[], 42);
        assert_eq!(result.name, "fallback-Identity");
    }

    #[test]
    fn test_select_single_candidate() {
        let selector = FormulaSelector::new();
        let c = make_candidate("only", r#""Identity""#, 0.5, 1.0, 10, &[]);
        let result = selector.select(&[c], &[], 42);
        assert_eq!(result.name, "only");
    }

    #[test]
    fn test_select_deterministic() {
        let selector = FormulaSelector::new();
        let candidates = vec![
            make_candidate("F0", r#""Identity""#, 0.2, 1.0, 10, &["math"]),
            make_candidate(
                "F1",
                r#"{"BiasAdd":{"source":{"GraphDistance":{"max_hops":4}},"weight":1.0}}"#,
                0.6,
                1.5,
                30,
                &["math", "factual"],
            ),
            make_candidate(
                "F2",
                r#"{"WarpQ":{"delta_source":"GnnDelta","alpha":0.5}}"#,
                -0.1,
                0.8,
                5,
                &["complex"],
            ),
        ];

        // Same seed → same result
        let r1 = selector.select(&candidates, &["math"], 12345);
        let r2 = selector.select(&candidates, &["math"], 12345);
        assert_eq!(r1.name, r2.name);
        assert_eq!(r1.candidate_index, r2.candidate_index);
    }

    #[test]
    fn test_select_different_seeds_can_differ() {
        let selector = FormulaSelector::new();
        let candidates = vec![
            make_candidate("F0", r#""Identity""#, 0.3, 1.0, 10, &[]),
            make_candidate("F1", r#""Identity""#, 0.3, 1.0, 10, &[]),
            make_candidate("F2", r#""Identity""#, 0.3, 1.0, 10, &[]),
        ];

        // With equal scores and high temperature, different seeds should
        // eventually select different formulas
        let mut selected_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for seed in 0..100u64 {
            let r = selector.select(&candidates, &[], seed);
            selected_names.insert(r.name);
        }
        // With 3 equal candidates and 100 seeds, we should see more than 1
        assert!(
            selected_names.len() > 1,
            "expected diversity, got {:?}",
            selected_names
        );
    }

    #[test]
    fn test_select_high_reward_preferred() {
        let selector = FormulaSelector::new();
        let candidates = vec![
            make_candidate("weak", r#""Identity""#, -0.5, 0.5, 200, &["math"]),
            make_candidate("strong", r#""Identity""#, 0.9, 1.8, 200, &["math"]),
        ];

        // With 200 activations, temperature is lower → exploitation.
        // "strong" should be selected most of the time.
        let mut strong_count = 0;
        for seed in 0..100u64 {
            let r = selector.select(&candidates, &["math"], seed);
            if r.name == "strong" {
                strong_count += 1;
            }
        }
        assert!(
            strong_count > 60,
            "strong should win >60% of the time, got {strong_count}%"
        );
    }

    #[test]
    fn test_dsl_to_nl_identity() {
        let nl = dsl_to_natural_language(&AttnOp::Identity);
        assert!(nl.contains("standard"), "got: {nl}");
    }

    #[test]
    fn test_dsl_to_nl_mask() {
        use crate::attn_dsl::MaskCondition;
        let op = AttnOp::Mask {
            condition: MaskCondition::GraphDistanceAbove(3),
        };
        let nl = dsl_to_natural_language(&op);
        assert!(nl.contains("3 hops"), "got: {nl}");
    }

    #[test]
    fn test_dsl_to_nl_bias_add() {
        use crate::attn_dsl::BiasSource;
        let op = AttnOp::BiasAdd {
            source: BiasSource::SynapseEnergy,
            weight: -0.5,
        };
        let nl = dsl_to_natural_language(&op);
        assert!(nl.contains("Atténue"), "got: {nl}");
        assert!(nl.contains("synaptique"), "got: {nl}");
    }

    #[test]
    fn test_dsl_to_nl_sequence() {
        use crate::attn_dsl::{BiasSource, MaskCondition};
        let op = AttnOp::Sequence(vec![
            AttnOp::Mask {
                condition: MaskCondition::GraphDistanceAbove(2),
            },
            AttnOp::BiasAdd {
                source: BiasSource::SynapseEnergy,
                weight: 0.5,
            },
        ]);
        let nl = dsl_to_natural_language(&op);
        assert!(nl.contains("Séquence"), "got: {nl}");
        assert!(nl.contains("→"), "got: {nl}");
    }
}
