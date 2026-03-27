//! EngramManager — the main coordinator for the engram system.
//!
//! Owns all subsystems (formation, decay, recall, homeostasis, metrics)
//! and provides a unified API for the rest of obrain-cognitive.

use std::sync::Arc;
use std::time::SystemTime;

use obrain_common::types::NodeId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use super::decay::{FsrsConfig, FsrsScheduler, ReviewRating};
use super::formation::{FormationConfig, HebbianWithSurprise};
use super::homeostasis::{HomeostasisConfig, HomeostasisEngine};
use super::observe::EngramMetricsCollector;
use super::recall::{RecallEngine, RecallResult, WarmupConfig, WarmupSelector};
use super::spectral::SpectralEncoder;
use super::store::EngramStore;
use super::traits::VectorIndex;
use super::types::{Engram, EngramHorizon, EngramId, RecallEvent, RecallFeedback};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for the [`EngramManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramManagerConfig {
    /// Configuration for Hebbian formation with surprise modulation.
    pub formation: FormationConfig,
    /// Configuration for the FSRS decay scheduler.
    pub fsrs: FsrsConfig,
    /// Configuration for homeostatic regulation.
    pub homeostasis: HomeostasisConfig,
    /// Configuration for warm-up recall selection.
    pub warmup: WarmupConfig,
}

impl Default for EngramManagerConfig {
    fn default() -> Self {
        Self {
            formation: FormationConfig::default(),
            fsrs: FsrsConfig::default(),
            homeostasis: HomeostasisConfig::default(),
            warmup: WarmupConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// EngramManager
// ---------------------------------------------------------------------------

/// Central coordinator for the engram memory system.
///
/// Holds all subsystems behind `Arc` or by value and exposes a
/// `&self`-only API suitable for concurrent use.
pub struct EngramManager {
    /// The shared engram store (DashMap-backed).
    store: Arc<EngramStore>,
    /// Hebbian + surprise formation engine (behind Mutex for interior mutability).
    formation: Mutex<HebbianWithSurprise>,
    /// FSRS-based decay scheduler.
    fsrs: FsrsScheduler,
    /// Spectral signature encoder.
    spectral: SpectralEncoder,
    /// Content-addressable recall engine.
    recall_engine: RecallEngine,
    /// Warmup configuration.
    warmup_config: WarmupConfig,
    /// Homeostatic regulation engine.
    homeostasis: HomeostasisEngine,
    /// Lock-free metrics collector.
    metrics: Arc<EngramMetricsCollector>,
    /// Vector index for spectral similarity search.
    vector_index: Arc<dyn VectorIndex>,
}

impl EngramManager {
    /// Creates a new `EngramManager` from the given configuration.
    ///
    /// The `vector_index` is injected so callers can provide an
    /// in-memory brute-force index or a production HNSW backend.
    pub fn new(config: EngramManagerConfig, vector_index: Arc<dyn VectorIndex>) -> Self {
        let store = Arc::new(EngramStore::new(None));
        let metrics = Arc::new(EngramMetricsCollector::new());

        Self {
            store,
            formation: Mutex::new(HebbianWithSurprise::new(config.formation)),
            fsrs: FsrsScheduler::new(config.fsrs),
            spectral: SpectralEncoder::new(),
            recall_engine: RecallEngine::new(),
            warmup_config: config.warmup,
            homeostasis: HomeostasisEngine::new(config.homeostasis),
            metrics,
            vector_index,
        }
    }

    /// Returns a reference to the shared engram store.
    pub fn store(&self) -> &Arc<EngramStore> {
        &self.store
    }

    /// Returns a reference to the metrics collector.
    pub fn metrics(&self) -> &Arc<EngramMetricsCollector> {
        &self.metrics
    }

    // -----------------------------------------------------------------
    // Activation & Formation
    // -----------------------------------------------------------------

    /// Records a co-activation event for the given node ensemble.
    ///
    /// Delegates to the formation engine and, if threshold is crossed,
    /// creates new engrams, encodes spectral signatures, and indexes
    /// them in the vector index.
    #[instrument(skip(self), fields(nodes = nodes.len(), prediction_error))]
    pub fn record_activation(&self, nodes: &[NodeId], prediction_error: f64) {
        let mut formation = self.formation.lock();
        formation.record_activation(nodes, prediction_error);

        // Check if any patterns are ready to become engrams.
        let candidates = formation.should_form_engram();
        drop(formation);
        for ensemble in candidates {
            let id = self.store.next_id();
            let mut engram = Engram::new(id, ensemble.clone());

            // Encode spectral signature.
            let signature = self.spectral.encode(&ensemble);
            engram.spectral_signature.clone_from(&signature);

            // Insert into store and vector index.
            self.store.insert(engram);
            self.vector_index.upsert(&id.to_string(), &signature);

            self.metrics.record_formation();
            debug!(%id, "engram formed");
        }
    }

    // -----------------------------------------------------------------
    // Recall
    // -----------------------------------------------------------------

    /// Performs content-addressable recall from the given cue nodes.
    ///
    /// Returns the top `k` matching engrams ordered by relevance.
    #[instrument(skip(self), fields(cues = cues.len(), k))]
    pub fn recall(&self, cues: &[NodeId], k: usize) -> Vec<RecallResult> {
        let results = self.recall_engine.recall(
            &self.store,
            cues,
            self.vector_index.as_ref(),
            &self.spectral,
            k,
        );
        for _ in &results {
            self.metrics.record_recall(true);
        }
        results
    }

    /// Warm-up recall: proactively retrieve engrams likely to be useful
    /// given the current cue context, with MMR diversity.
    ///
    /// Returns [`super::recall::ActivatedEngram`]s with the dominant engram in full detail
    /// and secondaries in summary form.
    #[instrument(skip(self), fields(cues = cues.len()))]
    pub fn warmup(&self, cues: &[NodeId]) -> Vec<super::recall::ActivatedEngram> {
        WarmupSelector::select_warmup_engrams(
            &self.store,
            cues,
            self.vector_index.as_ref(),
            &self.spectral,
            &self.warmup_config,
        )
    }

    // -----------------------------------------------------------------
    // Feedback
    // -----------------------------------------------------------------

    /// Mark an engram as having been used (positive feedback).
    ///
    /// Strengthens the engram, records a recall event, and feeds the
    /// homeostasis meta-plasticity tracker.
    pub fn mark_used(&self, engram_id: EngramId) {
        self.store.update(engram_id, |e| {
            e.recall_count += 1;
            e.last_activated = SystemTime::now();
            e.recall_history.push(RecallEvent {
                timestamp: SystemTime::now(),
                cues: Vec::new(),
                feedback: Some(RecallFeedback::Used),
            });
            // Boost strength slightly (clamped by homeostasis later).
            e.strength = (e.strength + 0.05).min(1.0);
        });
        self.metrics.record_recall(true);
        self.homeostasis.record_recall_outcome(true);
    }

    /// Mark an engram as rejected (negative feedback).
    ///
    /// Weakens the engram slightly and records the rejection.
    pub fn mark_rejected(&self, engram_id: EngramId) {
        self.store.update(engram_id, |e| {
            e.recall_history.push(RecallEvent {
                timestamp: SystemTime::now(),
                cues: Vec::new(),
                feedback: Some(RecallFeedback::Rejected),
            });
            // Slight strength penalty.
            e.strength = (e.strength - 0.02).max(0.0);
        });
        self.metrics.record_recall(false);
        self.homeostasis.record_recall_outcome(false);
    }

    // -----------------------------------------------------------------
    // Periodic Maintenance
    // -----------------------------------------------------------------

    /// Periodic maintenance tick.
    ///
    /// Should be called on a timer (e.g., every N seconds). Runs:
    /// 1. Homeostasis sweep (synaptic scaling).
    /// 2. FSRS decay for engrams that haven't been reviewed recently.
    /// 3. Horizon promotion/demotion based on strength.
    /// 4. Metrics update.
    #[instrument(skip(self))]
    pub fn tick(&self) {
        // 1. Homeostasis sweep.
        self.homeostasis.sweep(&self.store);
        self.metrics.record_homeostasis_sweep();

        // 2. FSRS decay pass — apply decay to engrams past their review date.
        let engrams = self.store.list();
        let now = SystemTime::now();
        for engram in &engrams {
            let elapsed_days = engram
                .fsrs_state
                .last_review
                .and_then(|lr| now.duration_since(lr).ok())
                .map_or(0.0, |d| d.as_secs_f64() / 86400.0);

            // Check if past scheduled review (or if never reviewed and old enough).
            let is_due = engram
                .fsrs_state
                .next_review
                .map_or(elapsed_days > 1.0, |nr| now >= nr);

            if is_due {
                let id = engram.id;
                self.store.update(id, |e| {
                    let days = e
                        .fsrs_state
                        .last_review
                        .and_then(|lr| now.duration_since(lr).ok())
                        .map_or(1.0, |d| d.as_secs_f64() / 86400.0);
                    let new_state = self.fsrs.review(&e.fsrs_state, ReviewRating::Again, days);
                    e.fsrs_state = new_state;
                    e.strength = (e.strength * 0.9).max(0.0);
                });

                // Check if engram has decayed below threshold.
                if let Some(updated) = self.store.get(id)
                    && updated.strength < 0.05
                {
                    self.store.remove(id);
                    self.vector_index.remove(&id.to_string());
                    self.metrics.record_decay();
                    debug!(%id, "engram decayed and removed");
                }
            }
        }

        // 3. Horizon promotion/demotion.
        let engrams = self.store.list();
        for engram in &engrams {
            let id = engram.id;
            let new_horizon = if engram.strength >= 0.8 && engram.recall_count >= 3 {
                EngramHorizon::Consolidated
            } else if engram.strength < 0.2 {
                EngramHorizon::Archived
            } else {
                EngramHorizon::Operational
            };

            if new_horizon != engram.horizon {
                self.store.update(id, |e| {
                    e.horizon = new_horizon;
                });

                if let Some(updated) = self.store.get(id)
                    && updated.is_crystallization_candidate()
                {
                    self.metrics.record_crystallization();
                    debug!(%id, "engram crystallization candidate");
                }
            }
        }

        // 4. Update mean strength metric.
        let engrams = self.store.list();
        if !engrams.is_empty() {
            let sum: f64 = engrams.iter().map(|e| e.strength).sum();
            let mean = sum / engrams.len() as f64;
            self.metrics.update_mean_strength(mean);
        }

        debug!(active = self.store.count(), "tick complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::traits::InMemoryVectorIndex;

    fn make_manager() -> EngramManager {
        let config = EngramManagerConfig::default();
        let index = Arc::new(InMemoryVectorIndex::new());
        EngramManager::new(config, index)
    }

    #[test]
    fn manager_can_be_constructed() {
        let _m = make_manager();
    }

    #[test]
    fn mark_used_strengthens_engram() {
        let m = make_manager();
        let id = m.store.next_id();
        let mut e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        e.strength = 0.5;
        m.store.insert(e);

        m.mark_used(id);

        let updated = m.store.get(id).unwrap();
        assert!(updated.strength > 0.5);
        assert_eq!(updated.recall_count, 1);
        assert_eq!(
            updated.recall_history.last().unwrap().feedback,
            Some(RecallFeedback::Used)
        );
    }

    #[test]
    fn mark_rejected_weakens_engram() {
        let m = make_manager();
        let id = m.store.next_id();
        let mut e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        e.strength = 0.5;
        m.store.insert(e);

        m.mark_rejected(id);

        let updated = m.store.get(id).unwrap();
        assert!(updated.strength < 0.5);
    }

    #[test]
    fn tick_does_not_panic_on_empty_store() {
        let m = make_manager();
        m.tick();
    }

    #[test]
    fn tick_promotes_strong_engrams() {
        let m = make_manager();
        let id = m.store.next_id();
        let mut e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        e.strength = 0.85;
        e.recall_count = 5;
        m.store.insert(e);

        m.tick();

        let updated = m.store.get(id).unwrap();
        assert_eq!(updated.horizon, EngramHorizon::Consolidated);
    }

    #[test]
    fn tick_archives_weak_engrams() {
        let m = make_manager();
        let id = m.store.next_id();
        let mut e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        e.strength = 0.15;
        m.store.insert(e);

        m.tick();

        let updated = m.store.get(id).unwrap();
        assert_eq!(updated.horizon, EngramHorizon::Archived);
    }

    #[test]
    fn metrics_reflect_operations() {
        let m = make_manager();
        let id = m.store.next_id();
        let e = Engram::new(id, vec![(NodeId(1), 1.0)]);
        m.store.insert(e);

        m.mark_used(id);
        m.mark_rejected(id);

        let snap = m.metrics.snapshot();
        assert_eq!(snap.recalls_successful, 1);
        assert_eq!(snap.recalls_rejected, 1);
        assert_eq!(snap.recalls_attempted, 2);
    }
}
