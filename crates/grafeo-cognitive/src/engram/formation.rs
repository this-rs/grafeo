//! HebbianWithSurprise — engram formation from co-activation patterns.
//!
//! This module implements Layer 0 engram formation: nodes that fire together
//! across multiple episodes, weighted by prediction error (surprise), become
//! candidates for engram consolidation.

use std::collections::HashMap;

use dashmap::DashMap;
use grafeo_common::types::NodeId;
use serde::{Deserialize, Serialize};

use crate::epigenetic::{EngramTemplate, EpigeneticBridge, ProjectContext};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for engram formation thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationConfig {
    /// Minimum number of episodes a pair must co-occur in before being considered.
    pub min_episodes: usize,
    /// Minimum overlap ratio (Jaccard-like) for a co-activation cluster.
    pub min_overlap: f64,
    /// Minimum cumulative prediction error to trigger formation.
    pub min_prediction_error: f64,
    /// Maximum number of nodes in a single engram ensemble.
    pub max_ensemble_size: usize,
}

impl Default for FormationConfig {
    fn default() -> Self {
        Self {
            min_episodes: 3,
            min_overlap: 0.5,
            min_prediction_error: 0.3,
            max_ensemble_size: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger enum
// ---------------------------------------------------------------------------

/// Describes the condition under which an engram should be formed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngramFormationTrigger {
    /// Pure co-activation: nodes that appear together often enough.
    CoActivation {
        /// Minimum number of co-occurrence episodes.
        min_episodes: usize,
        /// Minimum overlap ratio.
        min_overlap: f64,
    },
    /// Prediction error exceeds a threshold — surprise-driven formation.
    PredictionError {
        /// The prediction error magnitude threshold.
        threshold: f64,
    },
    /// Combined Hebbian co-activation plus prediction error surprise.
    HebbianWithSurprise {
        /// Minimum co-occurrence episodes.
        min_episodes: usize,
        /// Minimum cumulative prediction error.
        min_pe: f64,
    },
}

// ---------------------------------------------------------------------------
// CoActivationDetector
// ---------------------------------------------------------------------------

/// Tracks which nodes co-occur across episodes and detects recurring patterns.
///
/// Uses a `DashMap` keyed by ordered `(NodeId, NodeId)` pairs, counting
/// how many episodes each pair has been observed in.
#[derive(Debug)]
pub struct CoActivationDetector {
    /// Co-occurrence counts for node pairs. Keys are ordered (min, max).
    co_occurrences: DashMap<(NodeId, NodeId), u32>,
    /// Total number of episodes recorded (used for overlap ratio computation).
    episode_count: u32,
    /// Per-node episode count — how many episodes each node appeared in.
    node_episodes: DashMap<NodeId, u32>,
}

impl CoActivationDetector {
    /// Creates a new empty detector.
    pub fn new() -> Self {
        Self {
            co_occurrences: DashMap::new(),
            episode_count: 0,
            node_episodes: DashMap::new(),
        }
    }

    /// Records a co-activation event: the given `nodes` all appeared together
    /// in a single episode.
    pub fn record_episode(&mut self, nodes: &[NodeId]) {
        self.episode_count += 1;

        // Update per-node episode counts.
        for &node in nodes {
            *self.node_episodes.entry(node).or_insert(0) += 1;
        }

        // Update pairwise co-occurrence counts (ordered pairs to avoid duplicates).
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let key = ordered_pair(nodes[i], nodes[j]);
                *self.co_occurrences.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Detects recurring co-activation patterns.
    ///
    /// Returns a list of ensembles — each ensemble is a `Vec<(NodeId, f64)>`
    /// where the `f64` is the node's contribution weight (fraction of episodes
    /// in which it appeared relative to the cluster).
    ///
    /// The algorithm is a greedy single-linkage clustering:
    /// 1. Collect all pairs that co-occurred at least `min_episodes` times.
    /// 2. Build an adjacency list from those strong pairs.
    /// 3. Extract connected components as candidate ensembles.
    /// 4. Filter clusters whose average pairwise overlap >= `min_overlap`.
    pub fn detect_patterns(
        &self,
        min_episodes: usize,
        min_overlap: f64,
    ) -> Vec<Vec<(NodeId, f64)>> {
        if self.episode_count == 0 {
            return Vec::new();
        }

        // Step 1: collect strong pairs.
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for entry in &self.co_occurrences {
            let ((a, b), &count) = (entry.key(), entry.value());
            if (count as usize) >= min_episodes {
                adjacency.entry(*a).or_default().push(*b);
                adjacency.entry(*b).or_default().push(*a);
            }
        }

        // Step 2: extract connected components via BFS.
        let mut visited = std::collections::HashSet::new();
        let mut components: Vec<Vec<NodeId>> = Vec::new();

        for &start in adjacency.keys() {
            if visited.contains(&start) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited.insert(start);

            while let Some(node) = queue.pop_front() {
                component.push(node);
                if let Some(neighbors) = adjacency.get(&node) {
                    for &neighbor in neighbors {
                        if visited.insert(neighbor) {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            components.push(component);
        }

        // Step 3: filter by overlap and build ensembles with weights.
        let total = self.episode_count as f64;
        components
            .into_iter()
            .filter(|component| {
                if component.len() < 2 {
                    return false;
                }
                // Average pairwise overlap = avg(co_count / min(node_a_count, node_b_count))
                let mut overlap_sum = 0.0;
                let mut pair_count = 0u32;
                for i in 0..component.len() {
                    for j in (i + 1)..component.len() {
                        let key = ordered_pair(component[i], component[j]);
                        let co_count = self.co_occurrences.get(&key).map_or(0.0, |v| *v as f64);
                        let na = self
                            .node_episodes
                            .get(&component[i])
                            .map_or(1.0, |v| *v as f64);
                        let nb = self
                            .node_episodes
                            .get(&component[j])
                            .map_or(1.0, |v| *v as f64);
                        let min_count = na.min(nb);
                        if min_count > 0.0 {
                            overlap_sum += co_count / min_count;
                        }
                        pair_count += 1;
                    }
                }
                if pair_count == 0 {
                    return false;
                }
                (overlap_sum / pair_count as f64) >= min_overlap
            })
            .map(|component| {
                component
                    .into_iter()
                    .map(|node| {
                        let weight = self
                            .node_episodes
                            .get(&node)
                            .map_or(0.0, |v| *v as f64 / total);
                        (node, weight)
                    })
                    .collect()
            })
            .collect()
    }
}

impl Default for CoActivationDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the modulated `min_episodes` given a total epigenetic modulation.
///
/// - Positive modulation (amplification) → **lower** threshold (easier formation)
/// - Negative modulation (suppression) → **higher** threshold (harder formation)
///
/// Formula: `adjusted = base × (1.0 - total_modulation)`, clamped to `[1, base × 3]`.
///
/// Examples with base=3:
/// - modulation +0.5 → 3 × 0.5 = 1.5 → 2
/// - modulation -0.5 → 3 × 1.5 = 4.5 → 5
/// - modulation  0.0 → 3 × 1.0 = 3.0 → 3
pub fn compute_modulated_min_episodes(base_min_episodes: usize, total_modulation: f64) -> usize {
    let factor = 1.0 - total_modulation;
    let adjusted = (base_min_episodes as f64 * factor).round() as usize;
    adjusted.clamp(1, base_min_episodes.saturating_mul(3).max(1))
}

/// Returns the pair `(min, max)` to canonicalize unordered node pairs.
fn ordered_pair(a: NodeId, b: NodeId) -> (NodeId, NodeId) {
    if a.0 <= b.0 { (a, b) } else { (b, a) }
}

// ---------------------------------------------------------------------------
// HebbianWithSurprise
// ---------------------------------------------------------------------------

/// Combines co-activation detection with prediction error (surprise) to
/// decide when to form new engrams.
///
/// Only clusters that have *both* sufficient co-activation *and* sufficient
/// cumulative surprise will be proposed for engram formation.
#[derive(Debug)]
pub struct HebbianWithSurprise {
    /// Formation configuration thresholds.
    config: FormationConfig,
    /// Internal co-activation detector.
    detector: CoActivationDetector,
    /// Cumulative prediction error accumulated since last formation check.
    surprise_accumulator: f64,
    /// Number of activations recorded (for averaging).
    activation_count: u32,
}

impl HebbianWithSurprise {
    /// Creates a new `HebbianWithSurprise` with the given configuration.
    pub fn new(config: FormationConfig) -> Self {
        Self {
            config,
            detector: CoActivationDetector::new(),
            surprise_accumulator: 0.0,
            activation_count: 0,
        }
    }

    /// Records a set of co-activated nodes along with the prediction error
    /// magnitude observed at that moment.
    pub fn record_activation(&mut self, nodes: &[NodeId], prediction_error: f64) {
        self.detector.record_episode(nodes);
        self.surprise_accumulator += prediction_error;
        self.activation_count += 1;
    }

    /// Returns ensembles that meet both the co-activation *and* the surprise
    /// threshold, meaning they are ready for engram formation.
    ///
    /// An empty result means no ensemble currently qualifies.
    pub fn should_form_engram(&self) -> Vec<Vec<(NodeId, f64)>> {
        // Check the cumulative surprise threshold first (cheap gate).
        if self.surprise_accumulator < self.config.min_prediction_error {
            return Vec::new();
        }

        let mut ensembles = self
            .detector
            .detect_patterns(self.config.min_episodes, self.config.min_overlap);

        // Enforce max ensemble size by truncating large clusters.
        for ensemble in &mut ensembles {
            // Sort by weight descending so we keep the most important nodes.
            ensemble.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ensemble.truncate(self.config.max_ensemble_size);
        }

        ensembles
    }

    /// Returns ensembles that meet both the co-activation *and* the surprise
    /// threshold, with epigenetic mark modulation applied to `min_episodes`.
    ///
    /// A mark with positive modulation (+0.5) **lowers** the threshold (easier
    /// formation). A mark with negative modulation (-0.5) **raises** it
    /// (suppression).
    ///
    /// The effective `min_episodes` is:
    /// ```text
    /// adjusted = base_min_episodes × (1.0 - Σ effective_modulation)
    /// ```
    /// clamped to `[1, base × 3]` to prevent degenerate cases.
    ///
    /// Returns a tuple of `(ensembles, marks_evaluated, marks_applied, marks_suppressed)`.
    pub fn should_form_engram_with_marks(
        &self,
        template: &EngramTemplate,
        bridge: &EpigeneticBridge,
        ctx: &ProjectContext,
    ) -> (Vec<Vec<(NodeId, f64)>>, usize, usize, usize) {
        // Check the cumulative surprise threshold first (cheap gate).
        if self.surprise_accumulator < self.config.min_prediction_error {
            return (Vec::new(), 0, 0, 0);
        }

        let active_marks = bridge.get_active_marks(template, ctx);
        let marks_evaluated = active_marks.len();
        let mut marks_applied = 0usize;
        let mut marks_suppressed = 0usize;

        let mut total_modulation = 0.0_f64;
        for mark in &active_marks {
            let eff = mark.effective_modulation();
            if eff > 0.0 {
                marks_applied += 1;
            } else if eff < 0.0 {
                marks_suppressed += 1;
            }
            total_modulation += eff;
        }

        let adjusted_min =
            compute_modulated_min_episodes(self.config.min_episodes, total_modulation);

        let mut ensembles = self
            .detector
            .detect_patterns(adjusted_min, self.config.min_overlap);

        // Enforce max ensemble size by truncating large clusters.
        for ensemble in &mut ensembles {
            ensemble.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ensemble.truncate(self.config.max_ensemble_size);
        }

        (ensembles, marks_evaluated, marks_applied, marks_suppressed)
    }

    /// Returns the current cumulative surprise value.
    pub fn cumulative_surprise(&self) -> f64 {
        self.surprise_accumulator
    }

    /// Returns the number of activations recorded so far.
    pub fn activation_count(&self) -> u32 {
        self.activation_count
    }

    /// Resets the surprise accumulator (typically called after engram formation).
    pub fn reset_surprise(&mut self) {
        self.surprise_accumulator = 0.0;
        self.activation_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formation_config_defaults_are_sane() {
        let cfg = FormationConfig::default();
        assert_eq!(cfg.min_episodes, 3);
        assert!((cfg.min_overlap - 0.5).abs() < f64::EPSILON);
        assert!((cfg.min_prediction_error - 0.3).abs() < f64::EPSILON);
        assert_eq!(cfg.max_ensemble_size, 20);
    }

    #[test]
    fn co_activation_detector_records_and_detects() {
        let mut detector = CoActivationDetector::new();
        let nodes = vec![NodeId(1), NodeId(2), NodeId(3)];

        // Record the same trio 5 times — well above min_episodes=3.
        for _ in 0..5 {
            detector.record_episode(&nodes);
        }

        let patterns = detector.detect_patterns(3, 0.5);
        assert!(!patterns.is_empty(), "should detect at least one pattern");

        // The pattern should contain all three nodes.
        let first = &patterns[0];
        let node_ids: Vec<NodeId> = first.iter().map(|(id, _)| *id).collect();
        assert!(node_ids.contains(&NodeId(1)));
        assert!(node_ids.contains(&NodeId(2)));
        assert!(node_ids.contains(&NodeId(3)));
    }

    #[test]
    fn co_activation_no_detection_below_threshold() {
        let mut detector = CoActivationDetector::new();
        // Record only 2 episodes — below min_episodes=3.
        for _ in 0..2 {
            detector.record_episode(&[NodeId(1), NodeId(2)]);
        }
        let patterns = detector.detect_patterns(3, 0.5);
        assert!(patterns.is_empty());
    }

    #[test]
    fn hebbian_with_surprise_gates_on_prediction_error() {
        let config = FormationConfig {
            min_episodes: 2,
            min_overlap: 0.5,
            min_prediction_error: 1.0,
            max_ensemble_size: 10,
        };
        let mut hebb = HebbianWithSurprise::new(config);

        let nodes = vec![NodeId(1), NodeId(2), NodeId(3)];

        // Record co-activations with low surprise — should NOT trigger.
        for _ in 0..5 {
            hebb.record_activation(&nodes, 0.1);
        }
        // Cumulative surprise = 0.5, threshold = 1.0 → no formation.
        assert!(hebb.should_form_engram().is_empty());

        // Push surprise over the threshold.
        hebb.record_activation(&nodes, 0.6);
        // Cumulative surprise = 1.1, threshold = 1.0 → should form.
        let ensembles = hebb.should_form_engram();
        assert!(!ensembles.is_empty());
    }

    #[test]
    fn hebbian_reset_surprise_clears_accumulator() {
        let config = FormationConfig::default();
        let mut hebb = HebbianWithSurprise::new(config);
        hebb.record_activation(&[NodeId(1), NodeId(2)], 5.0);
        assert!(hebb.cumulative_surprise() > 0.0);
        assert_eq!(hebb.activation_count(), 1);

        hebb.reset_surprise();
        assert!((hebb.cumulative_surprise()).abs() < f64::EPSILON);
        assert_eq!(hebb.activation_count(), 0);
    }

    #[test]
    fn modulated_min_episodes_positive_mark_lowers_threshold() {
        // base=3, modulation +0.5 → 3 × (1.0 - 0.5) = 1.5 → round → 2
        let adjusted = compute_modulated_min_episodes(3, 0.5);
        assert_eq!(adjusted, 2, "positive modulation should lower min_episodes");
    }

    #[test]
    fn modulated_min_episodes_negative_mark_raises_threshold() {
        // base=3, modulation -0.5 → 3 × (1.0 - (-0.5)) = 3 × 1.5 = 4.5 → round → 5
        let adjusted = compute_modulated_min_episodes(3, -0.5);
        assert_eq!(adjusted, 5, "negative modulation should raise min_episodes");
    }

    #[test]
    fn modulated_min_episodes_zero_modulation_unchanged() {
        let adjusted = compute_modulated_min_episodes(3, 0.0);
        assert_eq!(
            adjusted, 3,
            "zero modulation should not change min_episodes"
        );
    }

    #[test]
    fn modulated_min_episodes_clamped_low() {
        // base=3, modulation +1.0 → 3 × 0.0 = 0 → clamped to 1
        let adjusted = compute_modulated_min_episodes(3, 1.0);
        assert_eq!(adjusted, 1, "should clamp to minimum of 1");
    }

    #[test]
    fn modulated_min_episodes_clamped_high() {
        // base=3, modulation -5.0 → 3 × 6.0 = 18 → clamped to 3*3=9
        let adjusted = compute_modulated_min_episodes(3, -5.0);
        assert_eq!(adjusted, 9, "should clamp to maximum of base*3");
    }

    #[test]
    fn should_form_engram_with_marks_positive_modulation() {
        use crate::epigenetic::{EngramTemplate, EpigeneticBridge, EpigeneticMark, ProjectContext};

        let config = FormationConfig {
            min_episodes: 3,
            min_overlap: 0.5,
            min_prediction_error: 0.0,
            max_ensemble_size: 10,
        };
        let mut hebb = HebbianWithSurprise::new(config);
        let nodes = vec![NodeId(1), NodeId(2), NodeId(3)];

        // Record exactly 2 episodes — below base min_episodes=3
        for _ in 0..2 {
            hebb.record_activation(&nodes, 0.5);
        }

        // Without marks: min_episodes=3, only 2 episodes → no formation
        assert!(hebb.should_form_engram().is_empty());

        // With a +0.5 mark → adjusted min_episodes = 2 → should form
        let mut bridge = EpigeneticBridge::new();
        bridge.add_mark(EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            true,
            vec![],
        ));
        let ctx = ProjectContext::new();
        let template = EngramTemplate::any();
        let (ensembles, evaluated, applied, suppressed) =
            hebb.should_form_engram_with_marks(&template, &bridge, &ctx);
        assert!(
            !ensembles.is_empty(),
            "positive mark should lower threshold and allow formation"
        );
        assert_eq!(evaluated, 1);
        assert_eq!(applied, 1);
        assert_eq!(suppressed, 0);
    }

    #[test]
    fn should_form_engram_with_marks_negative_modulation() {
        use crate::epigenetic::{EngramTemplate, EpigeneticBridge, EpigeneticMark, ProjectContext};

        let config = FormationConfig {
            min_episodes: 3,
            min_overlap: 0.5,
            min_prediction_error: 0.0,
            max_ensemble_size: 10,
        };
        let mut hebb = HebbianWithSurprise::new(config);
        let nodes = vec![NodeId(1), NodeId(2), NodeId(3)];

        // Record 4 episodes — enough for base min_episodes=3
        for _ in 0..4 {
            hebb.record_activation(&nodes, 0.5);
        }

        // Without marks: should form (4 >= 3)
        assert!(!hebb.should_form_engram().is_empty());

        // With -0.5 mark → adjusted min_episodes = 5 → 4 < 5 → no formation
        let mut bridge = EpigeneticBridge::new();
        bridge.add_mark(EpigeneticMark::new(
            EngramTemplate::any(),
            -0.5,
            true,
            vec![],
        ));
        let ctx = ProjectContext::new();
        let template = EngramTemplate::any();
        let (ensembles, evaluated, applied, suppressed) =
            hebb.should_form_engram_with_marks(&template, &bridge, &ctx);
        assert!(
            ensembles.is_empty(),
            "negative mark should raise threshold and block formation"
        );
        assert_eq!(evaluated, 1);
        assert_eq!(applied, 0);
        assert_eq!(suppressed, 1);
    }

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_co_activation_default() {
        let detector = CoActivationDetector::default();
        assert_eq!(detector.episode_count, 0);
    }

    #[test]
    fn test_detect_patterns_zero_episodes() {
        let detector = CoActivationDetector::new();
        let patterns = detector.detect_patterns(1, 0.0);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_detect_patterns_single_node() {
        let mut detector = CoActivationDetector::new();
        // Record episodes with only a single node — no pairs possible
        for _ in 0..5 {
            detector.record_episode(&[NodeId(1)]);
        }
        let patterns = detector.detect_patterns(1, 0.0);
        assert!(
            patterns.is_empty(),
            "single node per episode should produce no pairs"
        );
    }

    #[test]
    fn test_record_episode_empty() {
        let mut detector = CoActivationDetector::new();
        detector.record_episode(&[]);
        assert_eq!(detector.episode_count, 1);
        // No co-occurrences should be recorded
        assert!(detector.co_occurrences.is_empty());
    }

    #[test]
    fn test_ordered_pair_symmetry() {
        assert_eq!(
            ordered_pair(NodeId(1), NodeId(2)),
            ordered_pair(NodeId(2), NodeId(1))
        );
        assert_eq!(
            ordered_pair(NodeId(5), NodeId(5)),
            ordered_pair(NodeId(5), NodeId(5))
        );
        assert_eq!(
            ordered_pair(NodeId(100), NodeId(1)),
            ordered_pair(NodeId(1), NodeId(100))
        );
    }

    #[test]
    fn test_hebbian_activation_count_increments() {
        let config = FormationConfig::default();
        let mut hebb = HebbianWithSurprise::new(config);
        assert_eq!(hebb.activation_count(), 0);

        hebb.record_activation(&[NodeId(1), NodeId(2)], 0.1);
        assert_eq!(hebb.activation_count(), 1);

        hebb.record_activation(&[NodeId(1), NodeId(2)], 0.2);
        assert_eq!(hebb.activation_count(), 2);

        hebb.record_activation(&[NodeId(3)], 0.3);
        assert_eq!(hebb.activation_count(), 3);
    }

    #[test]
    fn test_hebbian_cumulative_surprise_accumulates() {
        let config = FormationConfig::default();
        let mut hebb = HebbianWithSurprise::new(config);

        hebb.record_activation(&[NodeId(1)], 0.5);
        assert!((hebb.cumulative_surprise() - 0.5).abs() < f64::EPSILON);

        hebb.record_activation(&[NodeId(2)], 0.3);
        assert!((hebb.cumulative_surprise() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_should_form_engram_with_marks_below_surprise_threshold() {
        use crate::epigenetic::{EngramTemplate, EpigeneticBridge, EpigeneticMark, ProjectContext};

        let config = FormationConfig {
            min_episodes: 2,
            min_overlap: 0.5,
            min_prediction_error: 1.0, // surprise threshold
            max_ensemble_size: 10,
        };
        let mut hebb = HebbianWithSurprise::new(config);

        // Record enough co-activations but low surprise
        for _ in 0..5 {
            hebb.record_activation(&[NodeId(1), NodeId(2)], 0.1);
        }
        // cumulative surprise = 0.5 < 1.0 threshold

        let mut bridge = EpigeneticBridge::new();
        bridge.add_mark(EpigeneticMark::new(
            EngramTemplate::any(),
            0.5,
            true,
            vec![],
        ));

        let ctx = ProjectContext::new();
        let template = EngramTemplate::any();
        let (ensembles, _, _, _) = hebb.should_form_engram_with_marks(&template, &bridge, &ctx);
        assert!(
            ensembles.is_empty(),
            "below surprise threshold should produce no ensembles even with marks"
        );
    }

    #[test]
    fn test_compute_modulated_min_episodes_base_1() {
        // base=1, mod=0.0 → 1
        assert_eq!(compute_modulated_min_episodes(1, 0.0), 1);
        // base=1, mod=0.5 → 1 × 0.5 = 0.5 → round → 1 (clamped to min 1)
        assert_eq!(compute_modulated_min_episodes(1, 0.5), 1);
        // base=1, mod=-0.5 → 1 × 1.5 = 1.5 → round → 2
        assert_eq!(compute_modulated_min_episodes(1, -0.5), 2);
    }

    #[test]
    fn test_compute_modulated_min_episodes_extreme_positive() {
        // base=5, mod=2.0 → 5 × (1-2) = -5 → clamped to 1
        assert_eq!(compute_modulated_min_episodes(5, 2.0), 1);
    }

    #[test]
    fn ensemble_truncated_to_max_size() {
        let config = FormationConfig {
            min_episodes: 2,
            min_overlap: 0.3,
            min_prediction_error: 0.0, // no surprise gate
            max_ensemble_size: 3,
        };
        let mut hebb = HebbianWithSurprise::new(config);

        // Create a cluster of 6 nodes.
        let nodes: Vec<NodeId> = (1..=6).map(NodeId).collect();
        for _ in 0..5 {
            hebb.record_activation(&nodes, 0.5);
        }

        let ensembles = hebb.should_form_engram();
        for ensemble in &ensembles {
            assert!(
                ensemble.len() <= 3,
                "ensemble should be truncated to max_ensemble_size"
            );
        }
    }
}
