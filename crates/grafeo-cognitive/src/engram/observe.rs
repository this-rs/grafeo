//! CognitiveMetrics — lock-free observability and feedback loop.
//!
//! All counters use `AtomicU64` with `Relaxed` ordering for maximum
//! throughput. The `f64` mean-strength metric is stored as a bit-cast
//! `u64` inside an `AtomicU64`.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stigmergic metrics — pure functions on pheromone distributions
// ---------------------------------------------------------------------------

/// Computes the normalized Shannon entropy of a pheromone distribution.
///
/// H_norm = H / log(N), where H = -Σ p_i × log(p_i).
///
/// Returns a value in `[0.0, 1.0]`:
/// - `1.0` = perfectly uniform distribution (healthy diversity)
/// - `0.0` = all mass on a single value (lock-in)
///
/// Edge cases:
/// - Empty slice → `0.0`
/// - Single element → `1.0` (trivially uniform)
/// - All zeros → `0.0`
pub fn compute_pheromone_entropy(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return if values.len() == 1 && values[0] > 0.0 {
            1.0
        } else {
            0.0
        };
    }

    let total: f64 = values.iter().copied().filter(|v| *v > 0.0).sum();
    if total <= 0.0 {
        return 0.0;
    }

    let n = values.iter().filter(|v| **v > 0.0).count();
    if n <= 1 {
        return 0.0;
    }

    let mut h = 0.0_f64;
    for &v in values {
        if v > 0.0 {
            let p = v / total;
            h -= p * p.ln();
        }
    }

    let max_h = (n as f64).ln();
    if max_h <= 0.0 {
        return 0.0;
    }

    (h / max_h).clamp(0.0, 1.0)
}

/// Computes the max/mean ratio of pheromone values.
///
/// Target: `< 10.0`. A ratio above this threshold indicates that a single
/// pheromone dominates the distribution, signaling potential lock-in.
///
/// Edge cases:
/// - Empty slice → `0.0`
/// - All zeros → `0.0`
/// - Single non-zero element → `1.0`
pub fn compute_max_pheromone_ratio(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut max = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut count = 0usize;

    for &v in values {
        if v > 0.0 {
            sum += v;
            count += 1;
            if v > max {
                max = v;
            }
        }
    }

    if count == 0 || sum <= 0.0 {
        return 0.0;
    }

    let mean = sum / count as f64;
    if mean <= 0.0 {
        return 0.0;
    }

    max / mean
}

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
    /// Normalized Shannon entropy of pheromone distribution, stored as `f64::to_bits`.
    /// High (→1.0) = healthy diversity, Low (→0.0) = lock-in.
    pub pheromone_entropy: AtomicU64,
    /// Max/mean ratio of pheromone values, stored as `f64::to_bits`.
    /// Target < 10.0. Above = single pheromone dominates.
    pub max_pheromone_ratio: AtomicU64,

    // -- Immune metrics (Layer 1) --
    /// Total immune detections (scans that matched).
    pub immune_detections_total: AtomicU64,
    /// Total immune detections marked as false positives (rejected).
    pub immune_detections_rejected: AtomicU64,
    /// Number of active immune detectors.
    pub immune_detector_count: AtomicU64,

    // -- Precision β (Hopfield Layer 3+4) --
    /// Average precision β across all active engrams, stored as `f64::to_bits`.
    pub avg_precision_beta: AtomicU64,

    // -- Epigenetic metrics (Layer 5) --
    /// Total epigenetic marks evaluated during formation decisions.
    pub marks_evaluated: AtomicU64,
    /// Total epigenetic marks with positive modulation that were applied.
    pub marks_applied: AtomicU64,
    /// Total epigenetic marks with negative modulation (suppression) that were applied.
    pub marks_suppressed: AtomicU64,
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
            pheromone_entropy: AtomicU64::new(0.0_f64.to_bits()),
            max_pheromone_ratio: AtomicU64::new(0.0_f64.to_bits()),
            immune_detections_total: AtomicU64::new(0),
            immune_detections_rejected: AtomicU64::new(0),
            immune_detector_count: AtomicU64::new(0),
            avg_precision_beta: AtomicU64::new(0.0_f64.to_bits()),
            marks_evaluated: AtomicU64::new(0),
            marks_applied: AtomicU64::new(0),
            marks_suppressed: AtomicU64::new(0),
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
    /// Normalized Shannon entropy of pheromone distribution (0.0–1.0).
    pub pheromone_entropy: f64,
    /// Max/mean ratio of pheromone values (target < 10.0).
    pub max_pheromone_ratio: f64,

    // -- Immune metrics (Layer 1) --
    /// Immune false-positive rate: rejected / total detections. Target < 20%.
    pub immune_fp_rate: f64,
    /// Number of active immune detectors.
    pub immune_detector_count: u64,

    // -- Precision β (Hopfield Layer 3+4) --
    /// Average precision β across all active engrams.
    pub avg_precision_beta: f64,

    // -- Epigenetic metrics (Layer 5) --
    /// Total epigenetic marks evaluated during formation decisions.
    pub marks_evaluated: u64,
    /// Total epigenetic marks with positive modulation applied.
    pub marks_applied: u64,
    /// Total epigenetic marks with negative modulation (suppression) applied.
    pub marks_suppressed: u64,
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
        self.metrics.engrams_formed.fetch_add(1, Ordering::Relaxed);
        self.metrics.engrams_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Records that an engram has decayed (removed from active set).
    pub fn record_decay(&self) {
        self.metrics.engrams_decayed.fetch_add(1, Ordering::Relaxed);
        // Saturating decrement for the active counter.
        let _ =
            self.metrics
                .engrams_active
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    if v > 0 { Some(v - 1) } else { Some(0) }
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

    /// Updates pheromone entropy from a distribution of pheromone values.
    ///
    /// Computes normalized Shannon entropy and stores the result.
    pub fn update_pheromone_entropy(&self, pheromone_values: &[f64]) {
        let entropy = compute_pheromone_entropy(pheromone_values);
        self.metrics
            .pheromone_entropy
            .store(entropy.to_bits(), Ordering::Relaxed);
    }

    /// Updates max/mean pheromone ratio from a distribution of pheromone values.
    ///
    /// Computes max/mean and stores the result.
    pub fn update_max_pheromone_ratio(&self, pheromone_values: &[f64]) {
        let ratio = compute_max_pheromone_ratio(pheromone_values);
        self.metrics
            .max_pheromone_ratio
            .store(ratio.to_bits(), Ordering::Relaxed);
    }

    // -- Immune metrics --

    /// Records an immune detection event. If `rejected` is true the detection
    /// was a false positive (the agent overrode it).
    pub fn record_immune_detection(&self, rejected: bool) {
        self.metrics
            .immune_detections_total
            .fetch_add(1, Ordering::Relaxed);
        if rejected {
            self.metrics
                .immune_detections_rejected
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Updates the current immune detector count.
    pub fn update_immune_detector_count(&self, count: usize) {
        self.metrics
            .immune_detector_count
            .store(count as u64, Ordering::Relaxed);
    }

    /// Returns the current immune false-positive rate (rejected / total).
    /// Returns 0.0 when there are no detections.
    pub fn immune_fp_rate(&self) -> f64 {
        let total = self.metrics.immune_detections_total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let rejected = self
            .metrics
            .immune_detections_rejected
            .load(Ordering::Relaxed);
        rejected as f64 / total as f64
    }

    // -- Precision β --

    /// Updates the average precision β across all active engrams.
    pub fn update_avg_precision_beta(&self, avg_beta: f64) {
        self.metrics
            .avg_precision_beta
            .store(avg_beta.to_bits(), Ordering::Relaxed);
    }

    // -- Epigenetic metrics --

    /// Records the outcome of epigenetic mark evaluation during formation.
    ///
    /// `evaluated` is the total number of marks checked.
    /// `applied` is the number with positive modulation.
    /// `suppressed` is the number with negative modulation.
    pub fn record_epigenetic_marks(&self, evaluated: usize, applied: usize, suppressed: usize) {
        self.metrics
            .marks_evaluated
            .fetch_add(evaluated as u64, Ordering::Relaxed);
        self.metrics
            .marks_applied
            .fetch_add(applied as u64, Ordering::Relaxed);
        self.metrics
            .marks_suppressed
            .fetch_add(suppressed as u64, Ordering::Relaxed);
    }

    /// Convenience: updates both pheromone entropy and max/mean ratio at once.
    pub fn update_stigmergy_metrics(&self, pheromone_values: &[f64]) {
        self.update_pheromone_entropy(pheromone_values);
        self.update_max_pheromone_ratio(pheromone_values);
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
            prediction_errors_total: self.metrics.prediction_errors_total.load(Ordering::Relaxed),
            mean_strength: f64::from_bits(self.metrics.mean_strength.load(Ordering::Relaxed)),
            pheromone_entropy: f64::from_bits(
                self.metrics.pheromone_entropy.load(Ordering::Relaxed),
            ),
            max_pheromone_ratio: f64::from_bits(
                self.metrics.max_pheromone_ratio.load(Ordering::Relaxed),
            ),
            immune_fp_rate: self.immune_fp_rate(),
            immune_detector_count: self.metrics.immune_detector_count.load(Ordering::Relaxed),
            avg_precision_beta: f64::from_bits(
                self.metrics.avg_precision_beta.load(Ordering::Relaxed),
            ),
            marks_evaluated: self.metrics.marks_evaluated.load(Ordering::Relaxed),
            marks_applied: self.metrics.marks_applied.load(Ordering::Relaxed),
            marks_suppressed: self.metrics.marks_suppressed.load(Ordering::Relaxed),
        }
    }
}

impl Default for EngramMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HomeostasisSignal — metrics → Layer ⊥ regulation decisions
// ---------------------------------------------------------------------------

/// Threshold for immune false-positive rate triggering T-reg regulation.
pub const IMMUNE_FP_RATE_THRESHOLD: f64 = 0.20;

/// Threshold for avg_precision_beta below which meta-plasticity is boosted.
/// When the global system confidence is very low, the system needs to learn
/// more aggressively.
pub const LOW_PRECISION_BETA_THRESHOLD: f64 = 0.5;

/// Regulation signals derived from cognitive metrics.
///
/// These signals feed into the HomeostasisScheduler (Layer ⊥) to trigger
/// corrective actions:
/// - `should_regulate_immune` → call `regulate_immune()` to shrink detector radii
/// - `meta_plasticity_boost` → multiplier for the learning rate (> 1.0 = more learning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisSignal {
    /// If true, the immune FP rate exceeds [`IMMUNE_FP_RATE_THRESHOLD`] and
    /// the HomeostasisScheduler should call `regulate_immune()` to reduce
    /// affinity radii of high-FP detectors.
    pub should_regulate_immune: bool,

    /// Meta-plasticity multiplier derived from `avg_precision_beta`.
    /// - `1.0` = no adjustment (healthy system)
    /// - `> 1.0` = increase learning rate (low confidence, needs exploration)
    ///
    /// Formula: when `avg_precision_beta < LOW_PRECISION_BETA_THRESHOLD`,
    /// boost = 1.0 + (threshold - β) / threshold, capped at 2.0.
    pub meta_plasticity_boost: f64,

    /// The raw immune FP rate that triggered the signal.
    pub immune_fp_rate: f64,

    /// The raw avg_precision_beta that triggered the signal.
    pub avg_precision_beta: f64,
}

impl HomeostasisSignal {
    /// Derive regulation signals from a metrics snapshot.
    ///
    /// - `immune_fp_rate > 20%` → `should_regulate_immune = true`
    /// - `avg_precision_beta < 0.5` → `meta_plasticity_boost > 1.0`
    pub fn from_metrics(snapshot: &CognitiveMetricsSnapshot) -> Self {
        let should_regulate_immune = snapshot.immune_fp_rate > IMMUNE_FP_RATE_THRESHOLD;

        let meta_plasticity_boost = if snapshot.avg_precision_beta < LOW_PRECISION_BETA_THRESHOLD {
            // Linear boost: 0.0 β → 2.0x, threshold β → 1.0x
            let ratio = (LOW_PRECISION_BETA_THRESHOLD - snapshot.avg_precision_beta)
                / LOW_PRECISION_BETA_THRESHOLD;
            (1.0 + ratio).min(2.0)
        } else {
            1.0
        };

        Self {
            should_regulate_immune,
            meta_plasticity_boost,
            immune_fp_rate: snapshot.immune_fp_rate,
            avg_precision_beta: snapshot.avg_precision_beta,
        }
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

    // -----------------------------------------------------------------------
    // Pheromone entropy tests
    // -----------------------------------------------------------------------

    #[test]
    fn pheromone_entropy_uniform_distribution() {
        // Uniform distribution → entropy ≈ 1.0
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let entropy = compute_pheromone_entropy(&values);
        assert!(
            (entropy - 1.0).abs() < 1e-10,
            "uniform distribution should have entropy ≈ 1.0, got {entropy}"
        );
    }

    #[test]
    fn pheromone_entropy_single_dominant() {
        // One pheromone dominates → entropy < 0.3
        let values = vec![100.0, 0.01, 0.01, 0.01, 0.01];
        let entropy = compute_pheromone_entropy(&values);
        assert!(
            entropy < 0.3,
            "dominant pheromone should have entropy < 0.3, got {entropy}"
        );
    }

    #[test]
    fn pheromone_entropy_empty() {
        assert_eq!(compute_pheromone_entropy(&[]), 0.0);
    }

    #[test]
    fn pheromone_entropy_single_nonzero() {
        assert_eq!(compute_pheromone_entropy(&[5.0]), 1.0);
    }

    #[test]
    fn pheromone_entropy_all_zeros() {
        assert_eq!(compute_pheromone_entropy(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn pheromone_entropy_with_zeros_mixed() {
        // Only non-zero values count; two equal non-zero → entropy = 1.0
        let values = vec![0.0, 5.0, 0.0, 5.0, 0.0];
        let entropy = compute_pheromone_entropy(&values);
        assert!(
            (entropy - 1.0).abs() < 1e-10,
            "two equal non-zero among zeros → entropy ≈ 1.0, got {entropy}"
        );
    }

    // -----------------------------------------------------------------------
    // Max pheromone ratio tests
    // -----------------------------------------------------------------------

    #[test]
    fn pheromone_ratio_uniform() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let ratio = compute_max_pheromone_ratio(&values);
        assert!(
            (ratio - 1.0).abs() < 1e-10,
            "uniform → ratio = 1.0, got {ratio}"
        );
    }

    #[test]
    fn pheromone_ratio_dominant() {
        // max = 100, mean = (100+1+1+1)/4 = 25.75 → ratio ≈ 3.88
        let values = vec![100.0, 1.0, 1.0, 1.0];
        let ratio = compute_max_pheromone_ratio(&values);
        assert!(ratio > 1.0, "should be > 1.0, got {ratio}");
        assert!(ratio < 10.0, "4 values, max 100 → ratio < 10, got {ratio}");
    }

    #[test]
    fn pheromone_ratio_extreme_lock_in() {
        // max = 1000, others = 0.1 → mean ≈ (1000+0.1*9)/10 = 100.09 → ratio ≈ 9.99
        // Actually with only non-zero: all 10 are non-zero
        let mut values = vec![0.1; 9];
        values.push(1000.0);
        let ratio = compute_max_pheromone_ratio(&values);
        assert!(
            ratio > 5.0,
            "extreme lock-in should have high ratio, got {ratio}"
        );
    }

    #[test]
    fn pheromone_ratio_empty() {
        assert_eq!(compute_max_pheromone_ratio(&[]), 0.0);
    }

    #[test]
    fn pheromone_ratio_all_zeros() {
        assert_eq!(compute_max_pheromone_ratio(&[0.0, 0.0]), 0.0);
    }

    // -----------------------------------------------------------------------
    // Integration: metrics_stigmergy
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_stigmergy_integration() {
        let c = EngramMetricsCollector::new();

        // Uniform pheromones
        let uniform = vec![1.0, 1.0, 1.0, 1.0];
        c.update_stigmergy_metrics(&uniform);
        let s = c.snapshot();
        assert!(
            (s.pheromone_entropy - 1.0).abs() < 1e-10,
            "entropy should be ≈ 1.0 for uniform, got {}",
            s.pheromone_entropy
        );
        assert!(
            (s.max_pheromone_ratio - 1.0).abs() < 1e-10,
            "ratio should be 1.0 for uniform, got {}",
            s.max_pheromone_ratio
        );

        // Now skewed pheromones
        let skewed = vec![100.0, 0.01, 0.01, 0.01];
        c.update_stigmergy_metrics(&skewed);
        let s2 = c.snapshot();
        assert!(
            s2.pheromone_entropy < 0.3,
            "entropy should be < 0.3 for skewed, got {}",
            s2.pheromone_entropy
        );
        assert!(
            s2.max_pheromone_ratio > 1.0,
            "ratio should be > 1.0 for skewed, got {}",
            s2.max_pheromone_ratio
        );
    }

    // -----------------------------------------------------------------------
    // Immune metrics tests (metrics_immune)
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_immune_fp_rate_zero_when_no_detections() {
        let c = EngramMetricsCollector::new();
        let s = c.snapshot();
        assert!((s.immune_fp_rate - 0.0).abs() < f64::EPSILON);
        assert_eq!(s.immune_detector_count, 0);
    }

    #[test]
    fn metrics_immune_fp_rate_computed_correctly() {
        let c = EngramMetricsCollector::new();
        // 10 detections, 3 rejected → FP rate = 0.3
        for _ in 0..7 {
            c.record_immune_detection(false);
        }
        for _ in 0..3 {
            c.record_immune_detection(true);
        }
        let s = c.snapshot();
        assert!(
            (s.immune_fp_rate - 0.3).abs() < 1e-10,
            "FP rate should be 0.3, got {}",
            s.immune_fp_rate
        );
    }

    #[test]
    fn metrics_immune_detector_count_updates() {
        let c = EngramMetricsCollector::new();
        c.update_immune_detector_count(42);
        assert_eq!(c.snapshot().immune_detector_count, 42);
        c.update_immune_detector_count(0);
        assert_eq!(c.snapshot().immune_detector_count, 0);
    }

    #[test]
    fn metrics_immune_all_rejected() {
        let c = EngramMetricsCollector::new();
        for _ in 0..5 {
            c.record_immune_detection(true);
        }
        let s = c.snapshot();
        assert!(
            (s.immune_fp_rate - 1.0).abs() < f64::EPSILON,
            "All rejected → FP rate should be 1.0, got {}",
            s.immune_fp_rate
        );
    }

    #[test]
    fn metrics_immune_none_rejected() {
        let c = EngramMetricsCollector::new();
        for _ in 0..5 {
            c.record_immune_detection(false);
        }
        let s = c.snapshot();
        assert!(
            (s.immune_fp_rate - 0.0).abs() < f64::EPSILON,
            "None rejected → FP rate should be 0.0, got {}",
            s.immune_fp_rate
        );
    }

    // -----------------------------------------------------------------------
    // Precision β tests (metrics_precision)
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_precision_beta_default_zero() {
        let c = EngramMetricsCollector::new();
        let s = c.snapshot();
        assert!((s.avg_precision_beta - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_precision_beta_roundtrips() {
        let c = EngramMetricsCollector::new();
        c.update_avg_precision_beta(3.14);
        let s = c.snapshot();
        assert!(
            (s.avg_precision_beta - 3.14).abs() < f64::EPSILON,
            "avg_precision_beta should be 3.14, got {}",
            s.avg_precision_beta
        );
    }

    #[test]
    fn metrics_precision_beta_updates_overwrite() {
        let c = EngramMetricsCollector::new();
        c.update_avg_precision_beta(1.0);
        c.update_avg_precision_beta(2.5);
        let s = c.snapshot();
        assert!(
            (s.avg_precision_beta - 2.5).abs() < f64::EPSILON,
            "should reflect last update, got {}",
            s.avg_precision_beta
        );
    }

    // -----------------------------------------------------------------------
    // Homeostasis integration tests (metrics_to_homeostasis_immune)
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_to_homeostasis_immune_high_fp_triggers_regulation() {
        let c = EngramMetricsCollector::new();
        // 25% FP rate → above 20% threshold → should trigger regulate_immune
        for _ in 0..75 {
            c.record_immune_detection(false);
        }
        for _ in 0..25 {
            c.record_immune_detection(true);
        }
        let s = c.snapshot();
        let regulation = HomeostasisSignal::from_metrics(&s);
        assert!(
            regulation.should_regulate_immune,
            "FP rate {} > 0.20 should trigger immune regulation",
            s.immune_fp_rate
        );
    }

    #[test]
    fn metrics_to_homeostasis_immune_low_fp_no_regulation() {
        let c = EngramMetricsCollector::new();
        // 10% FP rate → below 20% threshold → no regulation
        for _ in 0..90 {
            c.record_immune_detection(false);
        }
        for _ in 0..10 {
            c.record_immune_detection(true);
        }
        let s = c.snapshot();
        let regulation = HomeostasisSignal::from_metrics(&s);
        assert!(
            !regulation.should_regulate_immune,
            "FP rate {} <= 0.20 should NOT trigger immune regulation",
            s.immune_fp_rate
        );
    }

    #[test]
    fn metrics_to_homeostasis_immune_low_precision_increases_plasticity() {
        let c = EngramMetricsCollector::new();
        c.update_avg_precision_beta(0.1); // very low β
        let s = c.snapshot();
        let regulation = HomeostasisSignal::from_metrics(&s);
        assert!(
            regulation.meta_plasticity_boost > 1.0,
            "Low avg_precision_beta should boost meta_plasticity, got {}",
            regulation.meta_plasticity_boost
        );
    }

    #[test]
    fn metrics_to_homeostasis_immune_high_precision_no_boost() {
        let c = EngramMetricsCollector::new();
        c.update_avg_precision_beta(5.0); // healthy β
        let s = c.snapshot();
        let regulation = HomeostasisSignal::from_metrics(&s);
        assert!(
            (regulation.meta_plasticity_boost - 1.0).abs() < f64::EPSILON,
            "High avg_precision_beta should not boost meta_plasticity, got {}",
            regulation.meta_plasticity_boost
        );
    }

    #[test]
    fn metrics_to_homeostasis_immune_combined() {
        let c = EngramMetricsCollector::new();
        // High FP + low precision → both triggers fire
        for _ in 0..60 {
            c.record_immune_detection(false);
        }
        for _ in 0..40 {
            c.record_immune_detection(true);
        }
        c.update_avg_precision_beta(0.05);
        let s = c.snapshot();
        let regulation = HomeostasisSignal::from_metrics(&s);
        assert!(regulation.should_regulate_immune);
        assert!(regulation.meta_plasticity_boost > 1.0);
    }

    // -----------------------------------------------------------------------
    // Epigenetic metrics tests (marks_evaluated, marks_applied, marks_suppressed)
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_epigenetic_defaults_zero() {
        let c = EngramMetricsCollector::new();
        let s = c.snapshot();
        assert_eq!(s.marks_evaluated, 0);
        assert_eq!(s.marks_applied, 0);
        assert_eq!(s.marks_suppressed, 0);
    }

    #[test]
    fn metrics_epigenetic_records_correctly() {
        let c = EngramMetricsCollector::new();
        c.record_epigenetic_marks(5, 3, 1);
        let s = c.snapshot();
        assert_eq!(s.marks_evaluated, 5);
        assert_eq!(s.marks_applied, 3);
        assert_eq!(s.marks_suppressed, 1);
    }

    #[test]
    fn metrics_epigenetic_accumulates() {
        let c = EngramMetricsCollector::new();
        c.record_epigenetic_marks(3, 2, 1);
        c.record_epigenetic_marks(2, 1, 0);
        let s = c.snapshot();
        assert_eq!(s.marks_evaluated, 5);
        assert_eq!(s.marks_applied, 3);
        assert_eq!(s.marks_suppressed, 1);
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
