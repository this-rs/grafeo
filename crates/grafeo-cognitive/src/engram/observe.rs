//! CognitiveMetrics — lock-free observability and feedback loop.
//!
//! All counters use `AtomicU64` with `Relaxed` ordering for maximum
//! throughput. The `f64` mean-strength metric is stored as a bit-cast
//! `u64` inside an `AtomicU64`.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CognitiveMetrics — raw atomic counters
// ---------------------------------------------------------------------------

/// Raw atomic counters for the engram cognitive subsystem.
///
/// All fields are `AtomicU64` and all operations are lock-free.
/// For `mean_strength`, the `f64` is stored via `f64::to_bits` / `f64::from_bits`.
#[derive(Debug)]
pub struct CognitiveMetrics {
    /// Number of currently active (non-decayed) engrams.
    pub engrams_active: AtomicU64,
    /// Total engrams ever formed.
    pub engrams_formed: AtomicU64,
    /// Total engrams that have decayed (strength -> 0).
    pub engrams_decayed: AtomicU64,
    /// Total engrams recalled (served from memory).
    pub engrams_recalled: AtomicU64,
    /// Total engrams crystallized (promoted to permanent notes).
    pub engrams_crystallized: AtomicU64,
    /// Total formation attempts (may or may not produce an engram).
    pub formations_attempted: AtomicU64,
    /// Total recall attempts.
    pub recalls_attempted: AtomicU64,
    /// Total successful recalls (marked as used by the consumer).
    pub recalls_successful: AtomicU64,
    /// Total rejected recalls (marked as not useful).
    pub recalls_rejected: AtomicU64,
    /// Total homeostasis sweeps executed.
    pub homeostasis_sweeps: AtomicU64,
    /// Total prediction errors recorded.
    pub prediction_errors_total: AtomicU64,
    /// Mean engram strength, stored as `f64::to_bits`.
    pub mean_strength: AtomicU64,
}

impl Default for CognitiveMetrics {
    fn default() -> Self {
        Self {
            engrams_active: AtomicU64::new(0),
            engrams_formed: AtomicU64::new(0),
            engrams_decayed: AtomicU64::new(0),
            engrams_recalled: AtomicU64::new(0),
            engrams_crystallized: AtomicU64::new(0),
            formations_attempted: AtomicU64::new(0),
            recalls_attempted: AtomicU64::new(0),
            recalls_successful: AtomicU64::new(0),
            recalls_rejected: AtomicU64::new(0),
            homeostasis_sweeps: AtomicU64::new(0),
            prediction_errors_total: AtomicU64::new(0),
            mean_strength: AtomicU64::new(0.0_f64.to_bits()),
        }
    }
}

// ---------------------------------------------------------------------------
// CognitiveMetricsSnapshot — plain data struct for export
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all cognitive metrics.
///
/// This is a plain, cloneable struct suitable for serialization,
/// logging, or display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMetricsSnapshot {
    pub engrams_active: u64,
    pub engrams_formed: u64,
    pub engrams_decayed: u64,
    pub engrams_recalled: u64,
    pub engrams_crystallized: u64,
    pub formations_attempted: u64,
    pub recalls_attempted: u64,
    pub recalls_successful: u64,
    pub recalls_rejected: u64,
    pub homeostasis_sweeps: u64,
    pub prediction_errors_total: u64,
    pub mean_strength: f64,
}

// ---------------------------------------------------------------------------
// EngramMetricsCollector — ergonomic wrapper
// ---------------------------------------------------------------------------

/// Lock-free metrics collector for the engram subsystem.
///
/// Wraps [`CognitiveMetrics`] with ergonomic recording methods.
/// All operations are atomic and suitable for concurrent use from
/// multiple threads without any locking.
#[derive(Debug)]
pub struct EngramMetricsCollector {
    metrics: CognitiveMetrics,
}

impl EngramMetricsCollector {
    /// Creates a new collector with all counters initialised to zero.
    pub fn new() -> Self {
        Self {
            metrics: CognitiveMetrics::default(),
        }
    }

    /// Records that a new engram was formed.
    pub fn record_formation(&self) {
        self.metrics
            .formations_attempted
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .engrams_formed
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .engrams_active
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records that an engram has decayed (removed from active set).
    pub fn record_decay(&self) {
        self.metrics
            .engrams_decayed
            .fetch_add(1, Ordering::Relaxed);
        // Saturating decrement for the active counter.
        let _ = self
            .metrics
            .engrams_active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                if v > 0 {
                    Some(v - 1)
                } else {
                    Some(0)
                }
            });
    }

    /// Records a recall attempt and its outcome.
    pub fn record_recall(&self, was_successful: bool) {
        self.metrics
            .recalls_attempted
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .engrams_recalled
            .fetch_add(1, Ordering::Relaxed);
        if was_successful {
            self.metrics
                .recalls_successful
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.metrics
                .recalls_rejected
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records that an engram was crystallized into a permanent note.
    pub fn record_crystallization(&self) {
        self.metrics
            .engrams_crystallized
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records that a homeostasis sweep was executed.
    pub fn record_homeostasis_sweep(&self) {
        self.metrics
            .homeostasis_sweeps
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a prediction error event.
    pub fn record_prediction_error(&self) {
        self.metrics
            .prediction_errors_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Updates the mean engram strength metric.
    pub fn update_mean_strength(&self, mean: f64) {
        self.metrics
            .mean_strength
            .store(mean.to_bits(), Ordering::Relaxed);
    }

    /// Takes a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> CognitiveMetricsSnapshot {
        CognitiveMetricsSnapshot {
            engrams_active: self.metrics.engrams_active.load(Ordering::Relaxed),
            engrams_formed: self.metrics.engrams_formed.load(Ordering::Relaxed),
            engrams_decayed: self.metrics.engrams_decayed.load(Ordering::Relaxed),
            engrams_recalled: self.metrics.engrams_recalled.load(Ordering::Relaxed),
            engrams_crystallized: self.metrics.engrams_crystallized.load(Ordering::Relaxed),
            formations_attempted: self.metrics.formations_attempted.load(Ordering::Relaxed),
            recalls_attempted: self.metrics.recalls_attempted.load(Ordering::Relaxed),
            recalls_successful: self.metrics.recalls_successful.load(Ordering::Relaxed),
            recalls_rejected: self.metrics.recalls_rejected.load(Ordering::Relaxed),
            homeostasis_sweeps: self.metrics.homeostasis_sweeps.load(Ordering::Relaxed),
            prediction_errors_total: self
                .metrics
                .prediction_errors_total
                .load(Ordering::Relaxed),
            mean_strength: f64::from_bits(self.metrics.mean_strength.load(Ordering::Relaxed)),
        }
    }
}

impl Default for EngramMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_collector_has_zero_counters() {
        let c = EngramMetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.engrams_active, 0);
        assert_eq!(s.engrams_formed, 0);
        assert_eq!(s.engrams_decayed, 0);
        assert_eq!(s.engrams_recalled, 0);
        assert_eq!(s.engrams_crystallized, 0);
        assert_eq!(s.formations_attempted, 0);
        assert_eq!(s.recalls_attempted, 0);
        assert_eq!(s.recalls_successful, 0);
        assert_eq!(s.recalls_rejected, 0);
        assert_eq!(s.homeostasis_sweeps, 0);
        assert_eq!(s.prediction_errors_total, 0);
        assert!((s.mean_strength - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_formation_increments_counters() {
        let c = EngramMetricsCollector::new();
        c.record_formation();
        c.record_formation();
        let s = c.snapshot();
        assert_eq!(s.engrams_formed, 2);
        assert_eq!(s.formations_attempted, 2);
        assert_eq!(s.engrams_active, 2);
    }

    #[test]
    fn record_decay_decrements_active() {
        let c = EngramMetricsCollector::new();
        c.record_formation();
        c.record_formation();
        c.record_decay();
        let s = c.snapshot();
        assert_eq!(s.engrams_active, 1);
        assert_eq!(s.engrams_decayed, 1);
    }

    #[test]
    fn record_decay_saturates_at_zero() {
        let c = EngramMetricsCollector::new();
        c.record_decay();
        let s = c.snapshot();
        assert_eq!(s.engrams_active, 0);
        assert_eq!(s.engrams_decayed, 1);
    }

    #[test]
    fn record_recall_tracks_success_and_rejection() {
        let c = EngramMetricsCollector::new();
        c.record_recall(true);
        c.record_recall(true);
        c.record_recall(false);
        let s = c.snapshot();
        assert_eq!(s.recalls_attempted, 3);
        assert_eq!(s.engrams_recalled, 3);
        assert_eq!(s.recalls_successful, 2);
        assert_eq!(s.recalls_rejected, 1);
    }

    #[test]
    fn record_crystallization() {
        let c = EngramMetricsCollector::new();
        c.record_crystallization();
        assert_eq!(c.snapshot().engrams_crystallized, 1);
    }

    #[test]
    fn record_homeostasis_sweep() {
        let c = EngramMetricsCollector::new();
        c.record_homeostasis_sweep();
        c.record_homeostasis_sweep();
        assert_eq!(c.snapshot().homeostasis_sweeps, 2);
    }

    #[test]
    fn record_prediction_error() {
        let c = EngramMetricsCollector::new();
        c.record_prediction_error();
        assert_eq!(c.snapshot().prediction_errors_total, 1);
    }

    #[test]
    fn update_mean_strength_roundtrips() {
        let c = EngramMetricsCollector::new();
        c.update_mean_strength(0.42);
        let s = c.snapshot();
        assert!((s.mean_strength - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_is_serializable() {
        let c = EngramMetricsCollector::new();
        c.record_formation();
        c.update_mean_strength(0.75);
        let s = c.snapshot();
        let json = serde_json::to_string(&s).expect("should serialize");
        let deser: CognitiveMetricsSnapshot =
            serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(deser.engrams_formed, 1);
        assert!((deser.mean_strength - 0.75).abs() < f64::EPSILON);
    }
}
