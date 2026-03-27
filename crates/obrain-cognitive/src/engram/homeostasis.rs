//! Homeostasis engine — Layer ⊥ (basic version for Phase 1).
//!
//! Implements synaptic scaling and meta-plasticity to prevent runaway
//! strengthening/weakening of engrams. This is the "immune regulation"
//! of the memory system: it keeps overall memory load in balance.

use std::collections::VecDeque;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::store::EngramStore;

#[cfg(feature = "immune")]
use crate::immune::ImmuneSystem;

#[cfg(feature = "stigmergy")]
use crate::stigmergy::StigmergicEngine;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the homeostasis engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisConfig {
    /// Target mean strength across all engrams.
    /// The engine will nudge the global mean toward this value.
    pub target_mean_strength: f64,

    /// How aggressively to scale strengths per sweep.
    /// Higher values correct faster but risk oscillation.
    pub scaling_rate: f64,

    /// How often to run a homeostatic sweep, in seconds.
    pub sweep_interval_secs: u64,

    /// Number of recent recall outcomes to track for meta-plasticity.
    pub meta_plasticity_window: usize,

    /// Pheromone decay rate per sweep cycle (0.0–1.0). Default: 0.95.
    /// After 100 cycles, a pheromone of 1.0 decays to ~0.006.
    pub pheromone_decay_rate: f64,

    /// Noise magnitude for anti-lock-in (fraction, e.g. 0.05 = ±5%).
    pub anti_lock_in_noise: f64,
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            target_mean_strength: 0.5,
            scaling_rate: 0.1,
            sweep_interval_secs: 3600,
            meta_plasticity_window: 100,
            pheromone_decay_rate: 0.95,
            anti_lock_in_noise: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// HomeostasisEngine
// ---------------------------------------------------------------------------

/// The homeostasis engine maintains global memory balance.
///
/// Two mechanisms:
/// - **Synaptic scaling**: normalizes all engram strengths toward a target mean.
/// - **Meta-plasticity**: adjusts a learning-rate multiplier based on recent
///   recall success rate (BCM-like rule).
#[derive(Debug)]
pub struct HomeostasisEngine {
    config: HomeostasisConfig,
    /// Ring buffer of recent recall outcomes (true = useful, false = not useful).
    recall_outcomes: Mutex<VecDeque<bool>>,
    /// Monotonic counter used as seed for anti-lock-in noise.
    /// Used by `sweep()` which requires a pheromone `Engine`.
    #[allow(dead_code)]
    sweep_counter: std::sync::atomic::AtomicU64,
}

impl HomeostasisEngine {
    /// Creates a new homeostasis engine with the given configuration.
    pub fn new(config: HomeostasisConfig) -> Self {
        let window = config.meta_plasticity_window;
        Self {
            config,
            recall_outcomes: Mutex::new(VecDeque::with_capacity(window)),
            sweep_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Normalize all engram strengths toward `target_mean_strength`.
    ///
    /// - If the current mean is above target, all strengths are scaled down.
    /// - If the current mean is below target, all strengths are scaled up.
    /// - All strengths are clamped to \[0.0, 1.0\] after adjustment.
    pub fn synaptic_scaling(&self, store: &EngramStore) {
        let engrams = store.list();
        if engrams.is_empty() {
            return;
        }

        let sum: f64 = engrams.iter().map(|e| e.strength).sum();
        let mean = sum / engrams.len() as f64;
        let target = self.config.target_mean_strength;

        if (mean - target).abs() < f64::EPSILON {
            return;
        }

        let factor = if mean > target {
            1.0 - self.config.scaling_rate * (mean - target)
        } else {
            1.0 + self.config.scaling_rate * (target - mean)
        };

        for engram in &engrams {
            let id = engram.id;
            store.update(id, |e| {
                e.strength = (e.strength * factor).clamp(0.0, 1.0);
            });
        }
    }

    /// Returns a learning-rate multiplier based on recent recall success rate.
    ///
    /// The multiplier follows a BCM-inspired rule:
    /// - High success rate (> 0.8): reduce learning rate (already good) -> \[0.5, 1.0\]
    /// - Low success rate (< 0.3): increase learning rate (needs more learning) -> \[1.0, 2.0\]
    /// - Moderate success rate: neutral multiplier around 1.0
    ///
    /// Returns a value in \[0.5, 2.0\].
    pub fn meta_plasticity_factor(&self) -> f64 {
        let outcomes = self.recall_outcomes.lock();
        if outcomes.is_empty() {
            return 1.0;
        }

        let successes = outcomes.iter().filter(|&&b| b).count();
        let success_rate = successes as f64 / outcomes.len() as f64;

        if success_rate > 0.8 {
            // Good recall: reduce learning rate.
            // Linearly interpolate from 1.0 at success_rate=0.8 to 0.5 at success_rate=1.0.
            let t = (success_rate - 0.8) / 0.2;
            1.0 - t * 0.5
        } else if success_rate < 0.3 {
            // Poor recall: increase learning rate.
            // Linearly interpolate from 1.0 at success_rate=0.3 to 2.0 at success_rate=0.0.
            let t = (0.3 - success_rate) / 0.3;
            1.0 + t * 1.0
        } else {
            // Moderate: neutral.
            1.0
        }
    }

    /// Records the outcome of a recall for meta-plasticity tracking.
    pub fn record_recall_outcome(&self, was_useful: bool) {
        let mut outcomes = self.recall_outcomes.lock();
        if outcomes.len() >= self.config.meta_plasticity_window {
            outcomes.pop_front();
        }
        outcomes.push_back(was_useful);
    }

    /// Runs all homeostatic mechanisms on the engram store only.
    ///
    /// Call this periodically (e.g., every `sweep_interval_secs`).
    pub fn sweep(&self, store: &EngramStore) {
        self.synaptic_scaling(store);
    }

    /// Runs all homeostatic mechanisms including stigmergic regulation.
    ///
    /// 1. Synaptic scaling on engrams
    /// 2. Pheromone evaporation (temporal decay)
    /// 3. Anti-lock-in noise injection on dominant pheromones
    #[cfg(feature = "stigmergy")]
    pub fn sweep_with_stigmergy(&self, store: &EngramStore, engine: &StigmergicEngine) {
        // 1. Engram synaptic scaling
        self.synaptic_scaling(store);

        // 2. Pheromone evaporation
        engine.evaporate_all(self.config.pheromone_decay_rate);

        // 3. Anti-lock-in: inject noise on dominant pheromones
        let seed = self
            .sweep_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        engine.prevent_lock_in(self.config.anti_lock_in_noise, seed);
    }

    /// T-reg homeostatic regulation: delegate to [`ImmuneSystem::regulate_immune`].
    ///
    /// For each detector with `false_positive_rate > 30%`, shrink
    /// `affinity_radius × 0.8`. Returns the list of regulated detector IDs.
    #[cfg(feature = "immune")]
    pub fn regulate_immune(&self, immune: &ImmuneSystem) -> Vec<(crate::immune::DetectorId, f64)> {
        immune.regulate_immune()
    }

    /// Runs all homeostatic mechanisms including immune regulation.
    ///
    /// 1. Synaptic scaling on engrams
    /// 2. T-reg regulation on immune detectors
    #[cfg(feature = "immune")]
    pub fn sweep_with_immune(
        &self,
        store: &EngramStore,
        immune: &ImmuneSystem,
    ) -> Vec<(crate::immune::DetectorId, f64)> {
        self.synaptic_scaling(store);
        self.regulate_immune(immune)
    }

    /// Returns the configured pheromone decay rate.
    pub fn pheromone_decay_rate(&self) -> f64 {
        self.config.pheromone_decay_rate
    }

    /// Returns the configured anti-lock-in noise magnitude.
    pub fn anti_lock_in_noise(&self) -> f64 {
        self.config.anti_lock_in_noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::types::{Engram, EngramId};
    use obrain_common::types::NodeId;

    fn test_store() -> EngramStore {
        EngramStore::new(None)
    }

    #[test]
    fn synaptic_scaling_brings_mean_toward_target() {
        let store = test_store();
        // Insert engrams with high strengths (mean = 0.9).
        for i in 0..10 {
            let id = EngramId(i);
            let mut e = Engram::new(id, vec![(NodeId(i), 1.0)]);
            e.strength = 0.9;
            store.insert(e);
        }

        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        engine.synaptic_scaling(&store);

        // After scaling, all strengths should be lower.
        for engram in store.list() {
            assert!(engram.strength < 0.9, "strength should have decreased");
        }
    }

    #[test]
    fn synaptic_scaling_scales_up_weak_engrams() {
        let store = test_store();
        for i in 0..10 {
            let id = EngramId(i);
            let mut e = Engram::new(id, vec![(NodeId(i), 1.0)]);
            e.strength = 0.1;
            store.insert(e);
        }

        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        engine.synaptic_scaling(&store);

        for engram in store.list() {
            assert!(engram.strength > 0.1, "strength should have increased");
        }
    }

    #[test]
    fn synaptic_scaling_clamps_to_unit_interval() {
        let store = test_store();
        let id = EngramId(1);
        let mut e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        e.strength = 1.0;
        store.insert(e);

        let config = HomeostasisConfig {
            target_mean_strength: 0.99,
            scaling_rate: 100.0, // extreme rate
            ..Default::default()
        };
        let engine = HomeostasisEngine::new(config);
        engine.synaptic_scaling(&store);

        let updated = store.get(id).unwrap();
        assert!(updated.strength >= 0.0 && updated.strength <= 1.0);
    }

    #[test]
    fn meta_plasticity_neutral_when_empty() {
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        assert!((engine.meta_plasticity_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn meta_plasticity_reduces_rate_on_high_success() {
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        for _ in 0..100 {
            engine.record_recall_outcome(true);
        }
        let factor = engine.meta_plasticity_factor();
        assert!(factor < 1.0, "factor should be < 1.0 for high success rate");
        assert!(factor >= 0.5, "factor should be >= 0.5");
    }

    #[test]
    fn meta_plasticity_increases_rate_on_low_success() {
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        for _ in 0..100 {
            engine.record_recall_outcome(false);
        }
        let factor = engine.meta_plasticity_factor();
        assert!(factor > 1.0, "factor should be > 1.0 for low success rate");
        assert!(factor <= 2.0, "factor should be <= 2.0");
    }

    #[test]
    fn meta_plasticity_window_is_bounded() {
        let config = HomeostasisConfig {
            meta_plasticity_window: 5,
            ..Default::default()
        };
        let engine = HomeostasisEngine::new(config);

        // Fill with successes.
        for _ in 0..5 {
            engine.record_recall_outcome(true);
        }
        // Now add failures which should push out the successes.
        for _ in 0..5 {
            engine.record_recall_outcome(false);
        }

        let outcomes = engine.recall_outcomes.lock();
        assert_eq!(outcomes.len(), 5);
        assert!(outcomes.iter().all(|&b| !b));
    }

    #[test]
    fn sweep_does_not_panic_on_empty_store() {
        let store = test_store();
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        engine.sweep(&store); // should not panic
    }

    // ---- Immune T-reg integration tests ----

    #[cfg(feature = "immune")]
    #[test]
    fn t_reg_via_homeostasis_regulates_high_fp() {
        use crate::immune::{ImmuneDetector, ImmuneSystem, ShapeDescriptor};

        let store = test_store();
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        let immune = ImmuneSystem::new();

        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 1.0, EngramId(1));
        det.false_positive_rate = 0.4;
        let id = immune.register(det);

        let regulated = engine.sweep_with_immune(&store, &immune);
        assert_eq!(regulated.len(), 1);
        assert_eq!(regulated[0].0, id);

        let det = immune.get(&id).unwrap();
        assert!(
            (det.affinity_radius - 0.8).abs() < 1e-10,
            "radius should be shrunk to 0.8"
        );
    }

    #[cfg(feature = "immune")]
    #[test]
    fn t_reg_via_homeostasis_no_regulation_when_healthy() {
        use crate::immune::{ImmuneDetector, ImmuneSystem, ShapeDescriptor};

        let store = test_store();
        let engine = HomeostasisEngine::new(HomeostasisConfig::default());
        let immune = ImmuneSystem::new();

        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 1.0, EngramId(1));
        det.false_positive_rate = 0.1; // healthy
        immune.register(det);

        let regulated = engine.sweep_with_immune(&store, &immune);
        assert!(
            regulated.is_empty(),
            "healthy detector should not be regulated"
        );
    }

    // ---- Stigmergy integration tests ----

    #[cfg(feature = "stigmergy")]
    mod stigmergy_tests {
        use super::*;
        use crate::engram::traits::EdgeAnnotator;
        use crate::stigmergy::{StigmergicEngine, TrailType};
        use obrain_common::types::EdgeId;

        struct NoopAnnotator;
        impl EdgeAnnotator for NoopAnnotator {
            fn annotate(&self, _edge: EdgeId, _key: &str, _value: f64) {}
            fn get_annotation(&self, _edge: EdgeId, _key: &str) -> Option<f64> {
                None
            }
            fn remove_annotation(&self, _edge: EdgeId, _key: &str) {}
        }

        fn make_stigmergic_engine() -> StigmergicEngine {
            StigmergicEngine::new(Box::new(NoopAnnotator))
        }

        #[test]
        fn homeostasis_with_stigmergy_evaporates() {
            let store = test_store();
            let stig = make_stigmergic_engine();
            let homeo = HomeostasisEngine::new(HomeostasisConfig::default());

            // Deposit a pheromone
            stig.deposit(EdgeId::new(1), TrailType::Query, 1.0);

            // Run sweep with stigmergy
            homeo.sweep_with_stigmergy(&store, &stig);

            // Pheromone should have decayed by 0.95 (plus possible noise)
            let val = stig.read(EdgeId::new(1), TrailType::Query);
            // Single entry = it's the max, so noise is also applied.
            // After decay: 0.95, then noise of ±5% → [0.9025, 0.9975]
            assert!(val < 1.0, "Pheromone should have decayed, got {val}");
            assert!(
                val > 0.85,
                "Pheromone should not have decayed too much, got {val}"
            );
        }

        #[test]
        fn homeostasis_with_stigmergy_full_cycle() {
            let store = test_store();
            let stig = make_stigmergic_engine();

            // Add engrams with high strengths
            for i in 0..5 {
                let id = EngramId(i);
                let mut e = Engram::new(id, vec![(NodeId(i), 1.0)]);
                e.strength = 0.9;
                store.insert(e);
            }

            // Add pheromones: one dominant, others weak
            stig.deposit(EdgeId::new(1), TrailType::Query, 100.0);
            stig.deposit(EdgeId::new(2), TrailType::Query, 50.0);

            let homeo = HomeostasisEngine::new(HomeostasisConfig::default());
            homeo.sweep_with_stigmergy(&store, &stig);

            // Engram strengths should have been scaled
            for e in store.list() {
                assert!(e.strength < 0.9, "Engram strength should decrease");
            }

            // Pheromone should have decayed
            let val1 = stig.read(EdgeId::new(1), TrailType::Query);
            let val2 = stig.read(EdgeId::new(2), TrailType::Query);
            assert!(val1 < 100.0, "Dominant pheromone should decay");
            assert!(val2 < 50.0, "Weak pheromone should decay");
        }

        #[test]
        fn anti_lock_in_perturbs_max_pheromone() {
            let stig = make_stigmergic_engine();

            // All pheromones at max
            stig.deposit(EdgeId::new(1), TrailType::Query, 100.0);
            stig.deposit(EdgeId::new(2), TrailType::Query, 100.0);

            let before = stig.read(EdgeId::new(1), TrailType::Query);
            stig.prevent_lock_in(0.05, 42);
            let after = stig.read(EdgeId::new(1), TrailType::Query);

            // Should have changed by ±5%
            let delta = (after - before).abs();
            assert!(
                delta <= before * 0.05 + f64::EPSILON,
                "Noise should be within ±5%, delta = {delta}"
            );
        }
    }
}
