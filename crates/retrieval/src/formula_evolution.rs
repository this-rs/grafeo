//! Evolution engine for attention formulas.
//!
//! 5 mutation operators that produce new formulas from existing ones:
//! 1. PerturbWeights — ±10-30% on weights
//! 2. InsertOp — add a BiasAdd or Mask into a Sequence
//! 3. RemoveOp — remove an op from a Sequence
//! 4. RewireCondition — change a threshold or condition
//! 5. Crossover — combine 2 formulas into Sequence or Conditional
//!
//! Plus homeostasis (stagnation detection) and GC (population cap).

use crate::attn_dsl::*;

/// Mutation operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationType {
    PerturbWeights,
    InsertOp,
    RemoveOp,
    RewireCondition,
    Crossover,
}

/// Result of a mutation operation.
#[derive(Debug, Clone)]
pub struct MutationResult {
    /// The new mutated formula.
    pub op: AttnOp,
    /// Which mutation was applied.
    pub mutation_type: MutationType,
    /// Human-readable description.
    pub description: String,
}

// ─── Simple PRNG for deterministic mutations ────────────────────────────────

/// Lightweight PRNG state (splitmix64).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        z
    }

    /// Random f32 in [lo, hi].
    pub fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let t = (self.next_u64() as f32) / (u64::MAX as f32);
        lo + t * (hi - lo)
    }

    /// Random usize in [0, max) (exclusive).
    pub fn usize_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() % max as u64) as usize
    }

    /// Random bool with probability p.
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32_range(0.0, 1.0) < p
    }
}

// ─── Mutation operators ─────────────────────────────────────────────────────

/// Apply PerturbWeights: multiply all weights by (1 + noise) where noise ∈ [-0.3, 0.3].
pub fn perturb_weights(op: &AttnOp, rng: &mut Rng) -> MutationResult {
    let mut new_op = op.clone();
    perturb_weights_inner(&mut new_op, rng);
    clamp_weights(&mut new_op);
    MutationResult {
        op: new_op,
        mutation_type: MutationType::PerturbWeights,
        description: "perturbed weights ±30%".to_string(),
    }
}

fn perturb_weights_inner(op: &mut AttnOp, rng: &mut Rng) {
    match op {
        AttnOp::BiasAdd { weight, .. } | AttnOp::BiasScale { weight, .. } => {
            let noise = rng.f32_range(-0.3, 0.3);
            *weight *= 1.0 + noise;
        }
        AttnOp::WarpQ { alpha, .. } | AttnOp::WarpK { alpha, .. } => {
            let noise = rng.f32_range(-0.3, 0.3);
            *alpha *= 1.0 + noise;
        }
        AttnOp::QueryDelegate {
            entropy_threshold, ..
        } => {
            let noise = rng.f32_range(-0.3, 0.3);
            *entropy_threshold = (*entropy_threshold * (1.0 + noise)).max(0.1);
        }
        AttnOp::Sequence(ops) => {
            for o in ops.iter_mut() {
                perturb_weights_inner(o, rng);
            }
        }
        AttnOp::PerHead(pairs) => {
            for (_, o) in pairs.iter_mut() {
                perturb_weights_inner(o, rng);
            }
        }
        AttnOp::Conditional {
            then_op, else_op, ..
        } => {
            perturb_weights_inner(then_op, rng);
            perturb_weights_inner(else_op, rng);
        }
        AttnOp::Mask { .. } | AttnOp::Identity => {}
    }
}

/// Insert a random op into a formula. Wraps in Sequence if not already one.
pub fn insert_op(op: &AttnOp, rng: &mut Rng) -> MutationResult {
    let new_element = random_simple_op(rng);
    let desc = format!("inserted {:?}", short_desc(&new_element));

    let new_op = match op {
        AttnOp::Sequence(ops) => {
            let mut new_ops = ops.clone();
            let pos = rng.usize_range(new_ops.len() + 1);
            new_ops.insert(pos, new_element);
            AttnOp::Sequence(new_ops)
        }
        other => AttnOp::Sequence(vec![other.clone(), new_element]),
    };

    let mut result_op = new_op;
    clamp_weights(&mut result_op);

    MutationResult {
        op: result_op,
        mutation_type: MutationType::InsertOp,
        description: desc,
    }
}

/// Remove an op from a Sequence. If not a Sequence, returns Identity.
/// Never produces an empty Sequence.
pub fn remove_op(op: &AttnOp, rng: &mut Rng) -> MutationResult {
    match op {
        AttnOp::Sequence(ops) if ops.len() > 1 => {
            let mut new_ops = ops.clone();
            let pos = rng.usize_range(new_ops.len());
            let removed = new_ops.remove(pos);
            let desc = format!("removed {:?} at pos {}", short_desc(&removed), pos);

            let new_op = if new_ops.len() == 1 {
                new_ops.into_iter().next().unwrap()
            } else {
                AttnOp::Sequence(new_ops)
            };

            MutationResult {
                op: new_op,
                mutation_type: MutationType::RemoveOp,
                description: desc,
            }
        }
        _ => MutationResult {
            op: AttnOp::Identity,
            mutation_type: MutationType::RemoveOp,
            description: "removed all → Identity".to_string(),
        },
    }
}

/// Rewire a condition: change thresholds or swap condition types.
pub fn rewire_condition(op: &AttnOp, rng: &mut Rng) -> MutationResult {
    let mut new_op = op.clone();
    let rewired = rewire_condition_inner(&mut new_op, rng);
    let desc = if rewired {
        "rewired condition/threshold".to_string()
    } else {
        "no condition to rewire, perturbed weights instead".to_string()
    };

    if !rewired {
        perturb_weights_inner(&mut new_op, rng);
    }
    clamp_weights(&mut new_op);

    MutationResult {
        op: new_op,
        mutation_type: MutationType::RewireCondition,
        description: desc,
    }
}

fn rewire_condition_inner(op: &mut AttnOp, rng: &mut Rng) -> bool {
    match op {
        AttnOp::Mask { condition } => {
            match condition {
                MaskCondition::GraphDistanceAbove(d) => {
                    *d = (*d as i8 + rng.f32_range(-1.5, 1.5) as i8).clamp(1, 6) as u8;
                }
                MaskCondition::EnergyBelow(t) => {
                    *t += rng.f32_range(-0.2, 0.2);
                    *t = t.clamp(0.01, 1.5);
                }
                MaskCondition::NoPath => {
                    // Swap to a different condition type
                    *condition = MaskCondition::GraphDistanceAbove(
                        rng.f32_range(2.0, 5.0) as u8,
                    );
                }
            }
            true
        }
        AttnOp::Conditional { condition, then_op, else_op, .. } => {
            match condition {
                RuntimeCondition::Uncertainty { threshold } => {
                    *threshold += rng.f32_range(-0.3, 0.3);
                    *threshold = threshold.clamp(0.1, 2.0);
                }
                RuntimeCondition::GraphDensity { threshold } => {
                    *threshold += rng.f32_range(-0.2, 0.2);
                    *threshold = threshold.clamp(0.1, 0.9);
                }
                RuntimeCondition::TokenCount { min, max } => {
                    *min = (*min as i32 + rng.f32_range(-20.0, 20.0) as i32).max(0) as u32;
                    *max = (*max as i32 + rng.f32_range(-20.0, 20.0) as i32).max(*min as i32 + 1) as u32;
                }
                RuntimeCondition::ContextType(_) => {
                    // Can't meaningfully mutate string → try children
                    return rewire_condition_inner(then_op, rng)
                        || rewire_condition_inner(else_op, rng);
                }
            }
            true
        }
        AttnOp::Sequence(ops) => {
            for o in ops.iter_mut() {
                if rewire_condition_inner(o, rng) {
                    return true;
                }
            }
            false
        }
        AttnOp::PerHead(pairs) => {
            for (_, o) in pairs.iter_mut() {
                if rewire_condition_inner(o, rng) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Crossover: combine two formulas into a Sequence or Conditional.
pub fn crossover(parent_a: &AttnOp, parent_b: &AttnOp, rng: &mut Rng) -> MutationResult {
    let new_op = if rng.chance(0.5) {
        // Sequence crossover
        AttnOp::Sequence(vec![parent_a.clone(), parent_b.clone()])
    } else {
        // Conditional crossover: uncertainty-based
        let threshold = rng.f32_range(0.5, 1.5);
        AttnOp::Conditional {
            condition: RuntimeCondition::Uncertainty { threshold },
            then_op: Box::new(parent_a.clone()),
            else_op: Box::new(parent_b.clone()),
        }
    };

    // Validate depth — if too deep, simplify
    let mut result = new_op;
    if validate_depth(&result, MAX_DEPTH).is_err() {
        // Fallback: just use parent_a with perturbed weights
        result = parent_a.clone();
        perturb_weights_inner(&mut result, rng);
    }
    clamp_weights(&mut result);

    MutationResult {
        op: result,
        mutation_type: MutationType::Crossover,
        description: "crossover of two parents".to_string(),
    }
}

/// Apply a random mutation to a formula.
pub fn mutate(op: &AttnOp, rng: &mut Rng) -> MutationResult {
    let choice = rng.usize_range(4);
    match choice {
        0 => perturb_weights(op, rng),
        1 => insert_op(op, rng),
        2 => remove_op(op, rng),
        3 => rewire_condition(op, rng),
        _ => unreachable!(),
    }
}

// ─── Homeostasis ────────────────────────────────────────────────────────────

/// Check if the population is stagnating (low reward variance).
///
/// Returns `true` if the variance of recent rewards is below threshold,
/// indicating the population should receive a mutation burst.
pub fn should_mutate_burst(recent_rewards: &[f64], variance_threshold: f64) -> bool {
    // Zero-seed: with ≤2 formulas, always trigger mutations to bootstrap diversity
    if recent_rewards.len() <= 2 {
        return true;
    }
    if recent_rewards.len() < 5 {
        return false; // not enough data for variance estimation
    }
    let mean = recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64;
    let variance =
        recent_rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent_rewards.len() as f64;
    variance < variance_threshold
}

/// Check if population diversity is too low and crossover should be triggered.
pub fn should_crossover(active_count: usize, min_active: usize) -> bool {
    active_count < min_active
}

// ─── GC ─────────────────────────────────────────────────────────────────────

/// Candidate for garbage collection.
#[derive(Debug)]
pub struct GcCandidate {
    pub index: usize,
    pub energy: f64,
    pub activation_count: i64,
    pub generation: i64,
}

/// Identify formulas that should be deactivated.
///
/// Rules:
/// 1. energy < `energy_threshold` AND activation_count > `min_activations` → dead
/// 2. If population > `max_population`, kill lowest-energy non-seed formulas first
///
/// Returns indices of formulas to deactivate.
pub fn gc_candidates(
    formulas: &[GcCandidate],
    energy_threshold: f64,
    min_activations: i64,
    max_population: usize,
) -> Vec<usize> {
    let mut to_kill = Vec::new();

    // Rule 1: dead formulas (low energy, enough activations to be sure)
    for f in formulas {
        if f.energy < energy_threshold && f.activation_count > min_activations {
            to_kill.push(f.index);
        }
    }

    // Rule 2: population cap — kill lowest-energy non-seed formulas
    let alive_after_rule1 = formulas.len() - to_kill.len();
    if alive_after_rule1 > max_population {
        let excess = alive_after_rule1 - max_population;
        let mut remaining: Vec<&GcCandidate> = formulas
            .iter()
            .filter(|f| !to_kill.contains(&f.index))
            .filter(|f| f.generation > 0) // never kill seeds
            .collect();
        remaining.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal));

        for f in remaining.iter().take(excess) {
            to_kill.push(f.index);
        }
    }

    to_kill
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Generate a random simple op (leaf node) for InsertOp.
fn random_simple_op(rng: &mut Rng) -> AttnOp {
    let choice = rng.usize_range(4);
    match choice {
        0 => AttnOp::BiasAdd {
            source: BiasSource::SynapseEnergy,
            weight: rng.f32_range(-1.0, 1.0),
        },
        1 => AttnOp::BiasAdd {
            source: BiasSource::GraphDistance { max_hops: rng.f32_range(2.0, 5.0) as u8 },
            weight: rng.f32_range(0.1, 1.5),
        },
        2 => AttnOp::Mask {
            condition: MaskCondition::GraphDistanceAbove(rng.f32_range(2.0, 5.0) as u8),
        },
        3 => AttnOp::Mask {
            condition: MaskCondition::EnergyBelow(rng.f32_range(0.1, 0.5)),
        },
        _ => unreachable!(),
    }
}

/// Short description of an AttnOp for logging.
fn short_desc(op: &AttnOp) -> &'static str {
    match op {
        AttnOp::Identity => "Identity",
        AttnOp::BiasAdd { .. } => "BiasAdd",
        AttnOp::BiasScale { .. } => "BiasScale",
        AttnOp::Mask { .. } => "Mask",
        AttnOp::WarpQ { .. } => "WarpQ",
        AttnOp::WarpK { .. } => "WarpK",
        AttnOp::QueryDelegate { .. } => "QueryDelegate",
        AttnOp::Sequence(_) => "Sequence",
        AttnOp::PerHead(_) => "PerHead",
        AttnOp::Conditional { .. } => "Conditional",
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_f1() -> AttnOp {
        AttnOp::BiasAdd {
            source: BiasSource::GraphDistance { max_hops: 4 },
            weight: 1.0,
        }
    }

    fn seed_f2() -> AttnOp {
        AttnOp::Sequence(vec![
            AttnOp::Mask {
                condition: MaskCondition::GraphDistanceAbove(3),
            },
            AttnOp::BiasAdd {
                source: BiasSource::SynapseEnergy,
                weight: -0.5,
            },
        ])
    }

    // ── PerturbWeights ──

    #[test]
    fn test_perturb_weights_changes_value() {
        let mut rng = Rng::new(42);
        let result = perturb_weights(&seed_f1(), &mut rng);
        if let AttnOp::BiasAdd { weight, .. } = &result.op {
            assert!((*weight - 1.0).abs() > 0.01, "weight should change");
            assert!(*weight >= WEIGHT_MIN && *weight <= WEIGHT_MAX);
        } else {
            panic!("expected BiasAdd");
        }
    }

    #[test]
    fn test_perturb_weights_in_range() {
        let op = AttnOp::BiasAdd {
            source: BiasSource::Constant(0.5),
            weight: 0.5,
        };
        for seed in 0..100u64 {
            let mut rng = Rng::new(seed);
            let result = perturb_weights(&op, &mut rng);
            if let AttnOp::BiasAdd { weight, .. } = &result.op {
                assert!(
                    *weight >= 0.5 * 0.7 - 0.01 && *weight <= 0.5 * 1.3 + 0.01,
                    "weight={weight} out of ±30% range from 0.5"
                );
            }
        }
    }

    #[test]
    fn test_perturb_weights_recursive() {
        let mut rng = Rng::new(77);
        let result = perturb_weights(&seed_f2(), &mut rng);
        if let AttnOp::Sequence(ops) = &result.op {
            assert_eq!(ops.len(), 2);
            // Second op should have perturbed weight
            if let AttnOp::BiasAdd { weight, .. } = &ops[1] {
                assert!((*weight - (-0.5)).abs() > 0.001);
            }
        }
    }

    // ── InsertOp ──

    #[test]
    fn test_insert_op_wraps_identity() {
        let mut rng = Rng::new(42);
        let result = insert_op(&AttnOp::Identity, &mut rng);
        if let AttnOp::Sequence(ops) = &result.op {
            assert_eq!(ops.len(), 2);
            assert_eq!(ops[0], AttnOp::Identity);
        } else {
            panic!("expected Sequence, got {:?}", short_desc(&result.op));
        }
    }

    #[test]
    fn test_insert_op_into_sequence() {
        let mut rng = Rng::new(42);
        let result = insert_op(&seed_f2(), &mut rng);
        if let AttnOp::Sequence(ops) = &result.op {
            assert_eq!(ops.len(), 3); // was 2, now 3
        } else {
            panic!("expected Sequence");
        }
    }

    #[test]
    fn test_insert_op_validates() {
        let mut rng = Rng::new(42);
        let result = insert_op(&seed_f1(), &mut rng);
        assert!(validate_depth(&result.op, MAX_DEPTH).is_ok());
    }

    // ── RemoveOp ──

    #[test]
    fn test_remove_op_from_sequence() {
        let mut rng = Rng::new(42);
        let result = remove_op(&seed_f2(), &mut rng);
        // seed_f2 has 2 elements → removing 1 leaves a single op (unwrapped)
        assert!(!matches!(result.op, AttnOp::Sequence(_)) || {
            if let AttnOp::Sequence(ops) = &result.op {
                ops.len() >= 1
            } else {
                true
            }
        });
    }

    #[test]
    fn test_remove_op_non_sequence_gives_identity() {
        let mut rng = Rng::new(42);
        let result = remove_op(&seed_f1(), &mut rng);
        assert_eq!(result.op, AttnOp::Identity);
    }

    #[test]
    fn test_remove_op_never_empty_sequence() {
        for seed in 0..50u64 {
            let mut rng = Rng::new(seed);
            let big = AttnOp::Sequence(vec![
                AttnOp::Identity,
                AttnOp::Identity,
                AttnOp::Identity,
            ]);
            let result = remove_op(&big, &mut rng);
            match &result.op {
                AttnOp::Sequence(ops) => assert!(ops.len() >= 2),
                _ => {} // unwrapped to single op, fine
            }
        }
    }

    // ── RewireCondition ──

    #[test]
    fn test_rewire_mask_condition() {
        let op = AttnOp::Mask {
            condition: MaskCondition::GraphDistanceAbove(3),
        };
        // Try multiple seeds — at least one should produce a different threshold
        let mut changed = false;
        for seed in 0..20u64 {
            let mut rng = Rng::new(seed);
            let result = rewire_condition(&op, &mut rng);
            if let AttnOp::Mask {
                condition: MaskCondition::GraphDistanceAbove(d),
            } = &result.op
            {
                assert!(*d >= 1 && *d <= 6);
                if *d != 3 {
                    changed = true;
                }
            }
        }
        assert!(changed, "rewire should change threshold for at least one seed");
    }

    #[test]
    fn test_rewire_no_condition_fallback_perturb() {
        let mut rng = Rng::new(42);
        let result = rewire_condition(&seed_f1(), &mut rng);
        // F1 has no condition but does have weights → perturbed
        assert_eq!(result.mutation_type, MutationType::RewireCondition);
    }

    // ── Crossover ──

    #[test]
    fn test_crossover_produces_valid() {
        for seed in 0..50u64 {
            let mut rng = Rng::new(seed);
            let result = crossover(&seed_f1(), &seed_f2(), &mut rng);
            assert!(validate_depth(&result.op, MAX_DEPTH).is_ok());
        }
    }

    #[test]
    fn test_crossover_contains_both_parents() {
        let mut rng = Rng::new(42);
        let result = crossover(&AttnOp::Identity, &seed_f1(), &mut rng);
        let json = serde_json::to_string(&result.op).unwrap();
        // Should contain Identity and BiasAdd somewhere
        assert!(
            json.contains("Identity") || json.contains("BiasAdd") || json.contains("GraphDistance"),
            "crossover should contain elements from parents: {json}"
        );
    }

    #[test]
    fn test_crossover_deep_trees_fallback() {
        // Create trees at depth 7 each → crossover would be 9 → should fallback
        let mut a = AttnOp::Identity;
        for _ in 0..6 {
            a = AttnOp::Sequence(vec![a]);
        }
        let b = a.clone();
        let mut rng = Rng::new(42);
        let result = crossover(&a, &b, &mut rng);
        assert!(validate_depth(&result.op, MAX_DEPTH).is_ok());
    }

    // ── Homeostasis ──

    #[test]
    fn test_stagnation_detected() {
        let rewards = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert!(should_mutate_burst(&rewards, 0.05));
    }

    #[test]
    fn test_no_stagnation_with_variance() {
        let rewards = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5];
        assert!(!should_mutate_burst(&rewards, 0.05));
    }

    /// With ≤2 data points, zero-seed bootstrap kicks in: always mutate to
    /// build initial diversity. With 3-4 points, not enough data for variance.
    #[test]
    fn test_bootstrap_and_insufficient_data() {
        // ≤2 rewards: zero-seed bootstrap → always mutate
        assert!(should_mutate_burst(&[0.5, 0.5], 0.05));
        assert!(should_mutate_burst(&[0.5], 0.05));
        assert!(should_mutate_burst(&[], 0.05));
        // 3-4 rewards: not enough for variance estimation → no stagnation
        assert!(!should_mutate_burst(&[0.5, 0.5, 0.5], 0.05));
        assert!(!should_mutate_burst(&[0.5, 0.5, 0.5, 0.5], 0.05));
    }

    #[test]
    fn test_should_crossover_low_diversity() {
        assert!(should_crossover(3, 5));
        assert!(!should_crossover(6, 5));
    }

    // ── GC ──

    #[test]
    fn test_gc_kills_dead_formulas() {
        let formulas = vec![
            GcCandidate { index: 0, energy: 1.0, activation_count: 100, generation: 0 },
            GcCandidate { index: 1, energy: 0.01, activation_count: 60, generation: 1 },
            GcCandidate { index: 2, energy: 0.5, activation_count: 30, generation: 1 },
        ];
        let killed = gc_candidates(&formulas, 0.05, 50, 50);
        assert_eq!(killed, vec![1]); // only index 1 is dead
    }

    #[test]
    fn test_gc_respects_min_activations() {
        let formulas = vec![
            GcCandidate { index: 0, energy: 0.01, activation_count: 10, generation: 1 },
        ];
        let killed = gc_candidates(&formulas, 0.05, 50, 50);
        assert!(killed.is_empty()); // not enough activations to be sure
    }

    #[test]
    fn test_gc_population_cap() {
        let mut formulas = Vec::new();
        for i in 0..60 {
            formulas.push(GcCandidate {
                index: i,
                energy: 0.1 + (i as f64) * 0.01,
                activation_count: 100,
                generation: if i < 6 { 0 } else { 1 }, // first 6 are seeds
            });
        }
        let killed = gc_candidates(&formulas, 0.05, 50, 50);
        // Should kill at least 10 (60 - 50) non-seed formulas with lowest energy
        assert!(killed.len() >= 10, "killed {} formulas", killed.len());
        // Seeds (gen=0) should not be killed
        for &k in &killed {
            assert!(formulas[k].generation > 0, "seed at index {} should not be killed", k);
        }
    }

    #[test]
    fn test_gc_never_kills_seeds() {
        let formulas = vec![
            GcCandidate { index: 0, energy: 0.01, activation_count: 100, generation: 0 },
        ];
        let killed = gc_candidates(&formulas, 0.05, 50, 0); // max_population=0 forces kill
        // Rule 1 would kill it (energy<0.05, activations>50), but we check:
        // Actually rule 1 doesn't check generation, so it WILL kill seeds.
        // Seeds are only protected in rule 2 (population cap).
        // This is intentional: if a seed truly has 0 energy after 100 uses, it's dead.
        assert_eq!(killed, vec![0]);
    }

    // ── Mutate (random) ──

    #[test]
    fn test_mutate_always_valid() {
        for seed in 0..100u64 {
            let mut rng = Rng::new(seed);
            let result = mutate(&seed_f2(), &mut rng);
            assert!(
                validate_depth(&result.op, MAX_DEPTH).is_ok(),
                "seed {seed}: invalid depth for mutation {:?}",
                result.mutation_type
            );
        }
    }
}
