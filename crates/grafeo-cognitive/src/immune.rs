//! # Immune System — Layer 1: Anomaly Detection via Shape Space
//!
//! Inspired by the adaptive immune system (Jerne's idiotypic network, Nobel 1984).
//!
//! The immune layer generalises from specific engram experiences to detect
//! *structurally similar* patterns that have never been seen before. A gotcha
//! on `WhereBuilder + injection` matures into a detector for `*Builder + unsanitised input`.
//!
//! ## Core concepts
//!
//! - **ShapeDescriptor** — a normalised feature vector (L2 norm = 1.0) extracted from
//!   structural properties of an engram's ensemble (relation types, degrees, labels).
//! - **ImmuneDetector** — watches a region of shape space with a configurable
//!   `affinity_radius`. Matches any target whose euclidean distance falls within radius.
//! - **ImmuneSystem** — registry of detectors. `scan()` tests every detector against
//!   the current context and returns a list of `Detection` results.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use crate::engram::EngramId;

// ═══════════════════════════════════════════════════════════════════════════════
// DetectorId
// ═══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an immune detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DetectorId(pub u64);

static NEXT_DETECTOR_ID: AtomicU64 = AtomicU64::new(1);

impl DetectorId {
    /// Allocate a fresh, globally unique detector ID.
    pub fn next() -> Self {
        Self(NEXT_DETECTOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for DetectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "detector:{}", self.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ShapeDescriptor
// ═══════════════════════════════════════════════════════════════════════════════

/// A region in the feature space — a normalised vector of features extracted
/// from an engram's structural properties (relation types, degrees, labels).
///
/// # Invariants
///
/// - The vector is L2-normalised (‖v‖₂ = 1.0) after construction.
/// - An empty descriptor is valid but matches nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeDescriptor {
    /// The normalised feature vector.
    features: Vec<f64>,
}

impl ShapeDescriptor {
    /// Create a new `ShapeDescriptor` from a raw feature vector.
    ///
    /// The vector is automatically L2-normalised. If the input is the zero
    /// vector (or empty), the descriptor stores an empty vector.
    pub fn new(raw: Vec<f64>) -> Self {
        let norm = l2_norm(&raw);
        if norm < 1e-12 || raw.is_empty() {
            return Self {
                features: Vec::new(),
            };
        }
        let features = raw.iter().map(|x| x / norm).collect();
        Self { features }
    }

    /// Returns the normalised feature vector.
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// Dimensionality of the descriptor.
    pub fn dim(&self) -> usize {
        self.features.len()
    }

    /// Returns true if the descriptor is empty (zero-dimensional).
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Euclidean distance to another descriptor.
    ///
    /// If dimensions differ, the shorter vector is zero-padded.
    pub fn euclidean_distance(&self, other: &ShapeDescriptor) -> f64 {
        let max_dim = self.features.len().max(other.features.len());
        let mut sum_sq = 0.0_f64;
        for i in 0..max_dim {
            let a = self.features.get(i).copied().unwrap_or(0.0);
            let b = other.features.get(i).copied().unwrap_or(0.0);
            let diff = a - b;
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Cosine similarity (dot product of two L2-normalised vectors).
    pub fn cosine_similarity(&self, other: &ShapeDescriptor) -> f64 {
        if self.is_empty() || other.is_empty() {
            return 0.0;
        }
        let min_dim = self.features.len().min(other.features.len());
        let mut dot = 0.0_f64;
        for i in 0..min_dim {
            dot += self.features[i] * other.features[i];
        }
        dot.clamp(-1.0, 1.0)
    }

    /// Build a `ShapeDescriptor` from a spectral signature (engram).
    ///
    /// This is the primary factory method: the spectral signature already
    /// captures the structural essence of the engram ensemble.
    pub fn from_spectral_signature(spectral: &[f64]) -> Self {
        Self::new(spectral.to_vec())
    }

    /// Interpolate towards another descriptor by `rate` ∈ [0, 1].
    ///
    /// Returns a new `ShapeDescriptor` that is `(1-rate)*self + rate*other`,
    /// re-normalised to L2 = 1.0. Dimensions are zero-padded if needed.
    pub fn interpolate_towards(&self, other: &ShapeDescriptor, rate: f64) -> ShapeDescriptor {
        let rate = rate.clamp(0.0, 1.0);
        let max_dim = self.features.len().max(other.features.len());
        let mut raw = Vec::with_capacity(max_dim);
        for i in 0..max_dim {
            let a = self.features.get(i).copied().unwrap_or(0.0);
            let b = other.features.get(i).copied().unwrap_or(0.0);
            raw.push((1.0 - rate) * a + rate * b);
        }
        ShapeDescriptor::new(raw)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ImmuneDetector
// ═══════════════════════════════════════════════════════════════════════════════

/// An immune detector watches a region of shape space for structurally
/// similar patterns. Inspired by the adaptive immune system's B-cell
/// receptors and affinity maturation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmuneDetector {
    /// Unique identifier.
    pub id: DetectorId,

    /// The shape this detector recognises — a region of the feature space.
    pub shape: ShapeDescriptor,

    /// Maximum euclidean distance for a match. Smaller = more specific.
    pub affinity_radius: f64,

    /// History of shape mutations (somatic hypermutation). Each entry is a
    /// previous shape before mutation broadened the detector.
    pub mutation_history: Vec<ShapeDescriptor>,

    /// Clonal expansion count — how many times this detector has been
    /// successfully matched (→ clonal selection).
    pub clone_count: u32,

    /// Empirical false-positive rate. Used by T-reg homeostasis to shrink
    /// the affinity_radius when FP > 30%.
    pub false_positive_rate: f64,

    /// The source engram this detector was originally derived from.
    pub source_engram: EngramId,

    /// When this detector was created.
    pub created_at: SystemTime,
}

impl ImmuneDetector {
    /// Create a new detector with the given shape and source engram.
    pub fn new(shape: ShapeDescriptor, affinity_radius: f64, source_engram: EngramId) -> Self {
        Self {
            id: DetectorId::next(),
            shape,
            affinity_radius,
            mutation_history: Vec::new(),
            clone_count: 0,
            false_positive_rate: 0.0,
            source_engram,
            created_at: SystemTime::now(),
        }
    }

    /// Test whether `target` falls within this detector's affinity radius.
    ///
    /// Returns `true` if the euclidean distance between this detector's shape
    /// and the target shape is strictly less than `affinity_radius`.
    pub fn matches(&self, target: &ShapeDescriptor) -> bool {
        if self.shape.is_empty() || target.is_empty() {
            return false;
        }
        self.shape.euclidean_distance(target) < self.affinity_radius
    }

    /// Record a successful match — increment clone count (clonal expansion).
    pub fn record_match(&mut self) {
        self.clone_count = self.clone_count.saturating_add(1);
    }

    /// Mutate the detector's shape to broaden coverage (somatic hypermutation).
    ///
    /// The old shape is saved in `mutation_history` and the new shape replaces it.
    pub fn mutate(&mut self, new_shape: ShapeDescriptor) {
        self.mutation_history.push(self.shape.clone());
        self.shape = new_shape;
    }

    /// Somatic hypermutation: move the detector shape 10% towards a variant.
    ///
    /// - Saves the current shape in `mutation_history`.
    /// - Replaces `shape` with an interpolation `(1-rate)*self + rate*variant`.
    /// - Increments `clone_count` (clonal expansion).
    ///
    /// The default rate is [`DEFAULT_MUTATION_RATE`] (0.1).
    pub fn mutate_towards(&mut self, variant: &ShapeDescriptor) {
        self.mutate_towards_with_rate(variant, DEFAULT_MUTATION_RATE);
    }

    /// Like [`mutate_towards`](Self::mutate_towards) but with a custom rate.
    pub fn mutate_towards_with_rate(&mut self, variant: &ShapeDescriptor, rate: f64) {
        let new_shape = self.shape.interpolate_towards(variant, rate);
        self.mutation_history.push(self.shape.clone());
        self.shape = new_shape;
        self.clone_count = self.clone_count.saturating_add(1);
    }

    /// Expand affinity radius by `factor` (multiplicative), clamped to `max_radius`.
    ///
    /// Returns the new radius.
    pub fn expand_radius(&mut self, factor: f64, max_radius: f64) -> f64 {
        self.affinity_radius = (self.affinity_radius * factor).min(max_radius);
        self.affinity_radius
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Detection — result of a scan
// ═══════════════════════════════════════════════════════════════════════════════

/// A detection event produced when an immune detector matches a context.
#[derive(Debug, Clone)]
pub struct Detection {
    /// The detector that fired.
    pub detector_id: DetectorId,

    /// The source engram this detector generalises.
    pub source_engram: EngramId,

    /// Euclidean distance between the detector shape and the context.
    pub distance: f64,

    /// The affinity radius of the detector at match time.
    pub affinity_radius: f64,
}

impl Detection {
    /// Confidence ∈ (0.0, 1.0] — closer match → higher confidence.
    pub fn confidence(&self) -> f64 {
        if self.affinity_radius <= 0.0 {
            return 1.0;
        }
        (1.0 - self.distance / self.affinity_radius).clamp(0.0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ImmuneSystem — registry + scan
// ═══════════════════════════════════════════════════════════════════════════════

/// Default conservative affinity radius for auto-generated detectors.
pub const DEFAULT_AFFINITY_RADIUS: f64 = 0.3;

/// Default maximum affinity radius — ceiling for `expand_radius`.
pub const DEFAULT_MAX_AFFINITY_RADIUS: f64 = 2.0;

/// Default interpolation rate for `mutate_towards` (10% step towards variant).
pub const DEFAULT_MUTATION_RATE: f64 = 0.1;

/// Lower bound of the partial-match zone (fraction of affinity_radius).
pub const PARTIAL_MATCH_LOWER: f64 = 0.8;

/// Upper bound of the partial-match zone (fraction of affinity_radius).
pub const PARTIAL_MATCH_UPPER: f64 = 1.2;

/// The immune system maintains a registry of detectors and can scan a context
/// (represented as a `ShapeDescriptor`) to find matching detectors.
pub struct ImmuneSystem {
    /// Detector registry keyed by `DetectorId`.
    detectors: DashMap<DetectorId, ImmuneDetector>,
}

impl ImmuneSystem {
    /// Create an empty immune system.
    pub fn new() -> Self {
        Self {
            detectors: DashMap::new(),
        }
    }

    /// Register a new detector.
    pub fn register(&self, detector: ImmuneDetector) -> DetectorId {
        let id = detector.id;
        self.detectors.insert(id, detector);
        id
    }

    /// Remove a detector by ID.
    pub fn remove(&self, id: &DetectorId) -> Option<ImmuneDetector> {
        self.detectors.remove(id).map(|(_, d)| d)
    }

    /// Get a reference to a detector (via DashMap Ref guard).
    pub fn get(
        &self,
        id: &DetectorId,
    ) -> Option<dashmap::mapref::one::Ref<'_, DetectorId, ImmuneDetector>> {
        self.detectors.get(id)
    }

    /// Number of registered detectors.
    pub fn detector_count(&self) -> usize {
        self.detectors.len()
    }

    /// Scan a context shape against all detectors.
    ///
    /// Returns a list of [`Detection`] for every detector whose shape matches
    /// the context within its affinity radius.
    pub fn scan(&self, context: &ShapeDescriptor) -> Vec<Detection> {
        let mut detections = Vec::new();
        for entry in self.detectors.iter() {
            let detector = entry.value();
            if detector.matches(context) {
                let distance = detector.shape.euclidean_distance(context);
                detections.push(Detection {
                    detector_id: detector.id,
                    source_engram: detector.source_engram,
                    distance,
                    affinity_radius: detector.affinity_radius,
                });
            }
        }
        // Sort by distance ascending (closest match first)
        detections.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        detections
    }

    /// Get a mutable reference to a detector (via DashMap RefMut guard).
    pub fn get_mut(
        &self,
        id: &DetectorId,
    ) -> Option<dashmap::mapref::one::RefMut<'_, DetectorId, ImmuneDetector>> {
        self.detectors.get_mut(id)
    }

    /// Scan and trigger affinity maturation on partial matches.
    ///
    /// A *partial match* is defined as a target whose distance `d` satisfies:
    ///   `radius * PARTIAL_MATCH_LOWER ≤ d ≤ radius * PARTIAL_MATCH_UPPER`
    ///
    /// For each partial match:
    /// 1. `mutate_towards(context)` — somatic hypermutation (10% step).
    /// 2. `expand_radius(factor, max_radius)` — broaden detection zone.
    ///
    /// Returns the list of detector IDs that underwent maturation.
    pub fn on_detection(
        &self,
        context: &ShapeDescriptor,
        expansion_factor: f64,
        max_radius: f64,
    ) -> Vec<DetectorId> {
        let mut matured = Vec::new();

        // Collect candidates first to avoid holding DashMap iterators while mutating.
        let candidates: Vec<(DetectorId, f64, f64)> = self
            .detectors
            .iter()
            .filter_map(|entry| {
                let det = entry.value();
                if det.shape.is_empty() || context.is_empty() {
                    return None;
                }
                let dist = det.shape.euclidean_distance(context);
                let lower = det.affinity_radius * PARTIAL_MATCH_LOWER;
                let upper = det.affinity_radius * PARTIAL_MATCH_UPPER;
                if dist >= lower && dist <= upper {
                    Some((det.id, dist, det.affinity_radius))
                } else {
                    None
                }
            })
            .collect();

        for (id, _dist, _radius) in candidates {
            if let Some(mut det) = self.detectors.get_mut(&id) {
                det.mutate_towards(context);
                det.expand_radius(expansion_factor, max_radius);
                matured.push(id);
            }
        }

        matured
    }

    /// Automatically create a detector from a scarred engram.
    ///
    /// The detector's shape is derived from the engram's spectral signature,
    /// and the initial affinity radius is conservative (`DEFAULT_AFFINITY_RADIUS`).
    pub fn create_detector_from_scar(
        &self,
        spectral_signature: &[f64],
        source_engram: EngramId,
    ) -> DetectorId {
        let shape = ShapeDescriptor::from_spectral_signature(spectral_signature);
        let detector = ImmuneDetector::new(shape, DEFAULT_AFFINITY_RADIUS, source_engram);
        self.register(detector)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Idiotypic network — detectors recognising each other (Jerne, 1974)
    // ─────────────────────────────────────────────────────────────────────

    /// Cosine-similarity threshold above which two detectors are considered
    /// part of the same idiotypic family (`:RECOGNIZES` relation).
    pub const IDIOTYPIC_THRESHOLD: f64 = 0.7;

    /// Propagation factor for idiotypic activation (1-hop only).
    pub const IDIOTYPIC_PROPAGATION_FACTOR: f64 = 0.5;

    /// Find all pairs of detectors whose shapes have cosine similarity > threshold.
    ///
    /// Returns a list of `(DetectorId, DetectorId, cosine_similarity)` triples.
    /// Each pair appears only once (A,B but not B,A).
    pub fn find_idiotypic_pairs(&self, threshold: f64) -> Vec<(DetectorId, DetectorId, f64)> {
        let all: Vec<(DetectorId, ShapeDescriptor)> = self
            .detectors
            .iter()
            .map(|e| (e.key().clone(), e.value().shape.clone()))
            .collect();

        let mut pairs = Vec::new();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let sim = all[i].1.cosine_similarity(&all[j].1);
                if sim > threshold {
                    pairs.push((all[i].0, all[j].0, sim));
                }
            }
        }
        pairs
    }

    /// Get the idiotypic neighbours of a detector (cosine > [`IDIOTYPIC_THRESHOLD`]).
    ///
    /// Returns `(DetectorId, cosine_similarity)` for each neighbour.
    pub fn idiotypic_neighbours(&self, id: &DetectorId) -> Vec<(DetectorId, f64)> {
        let shape = match self.detectors.get(id) {
            Some(d) => d.shape.clone(),
            None => return Vec::new(),
        };

        self.detectors
            .iter()
            .filter_map(|entry| {
                let other_id = *entry.key();
                if other_id == *id {
                    return None;
                }
                let sim = shape.cosine_similarity(&entry.value().shape);
                if sim > Self::IDIOTYPIC_THRESHOLD {
                    Some((other_id, sim))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Idiotypic propagation: when a detector is activated, partially activate
    /// its idiotypic neighbours (1-hop only, no cascade).
    ///
    /// "Activation" here means incrementing `clone_count` by 1 and recording
    /// the activation. Returns the list of detector IDs that were propagated to,
    /// along with the propagation weight (similarity × [`IDIOTYPIC_PROPAGATION_FACTOR`]).
    pub fn idiotypic_propagate(&self, activated_id: &DetectorId) -> Vec<(DetectorId, f64)> {
        let neighbours = self.idiotypic_neighbours(activated_id);
        let mut propagated = Vec::new();

        for (neighbour_id, similarity) in neighbours {
            let weight = similarity * Self::IDIOTYPIC_PROPAGATION_FACTOR;
            if let Some(mut det) = self.detectors.get_mut(&neighbour_id) {
                det.clone_count = det.clone_count.saturating_add(1);
                propagated.push((neighbour_id, weight));
            }
        }

        propagated
    }

    // ─────────────────────────────────────────────────────────────────────
    // T-reg — regulatory suppression of overly promiscuous detectors
    // ─────────────────────────────────────────────────────────────────────

    /// False-positive rate threshold above which T-reg kicks in.
    pub const T_REG_FP_THRESHOLD: f64 = 0.3;

    /// Shrink factor applied to affinity_radius when FP rate is too high.
    pub const T_REG_SHRINK_FACTOR: f64 = 0.8;

    /// T-reg homeostatic regulation: for each detector with
    /// `false_positive_rate > threshold`, shrink `affinity_radius` by `factor`.
    ///
    /// Returns the list of regulated detector IDs and their new radii.
    pub fn regulate_immune(&self) -> Vec<(DetectorId, f64)> {
        self.regulate_immune_with(Self::T_REG_FP_THRESHOLD, Self::T_REG_SHRINK_FACTOR)
    }

    /// Like [`regulate_immune`](Self::regulate_immune) but with custom thresholds.
    pub fn regulate_immune_with(&self, fp_threshold: f64, shrink_factor: f64) -> Vec<(DetectorId, f64)> {
        let candidates: Vec<DetectorId> = self
            .detectors
            .iter()
            .filter(|e| e.value().false_positive_rate > fp_threshold)
            .map(|e| *e.key())
            .collect();

        let mut regulated = Vec::new();
        for id in candidates {
            if let Some(mut det) = self.detectors.get_mut(&id) {
                det.affinity_radius *= shrink_factor;
                regulated.push((id, det.affinity_radius));
            }
        }

        regulated
    }

    /// Mark a detection as rejected (false positive).
    ///
    /// Updates the detector's `false_positive_rate` using an exponential
    /// moving average: `fp = 0.9 * fp + 0.1 * 1.0`.
    pub fn mark_rejected(&self, detector_id: &DetectorId) {
        if let Some(mut det) = self.detectors.get_mut(detector_id) {
            det.false_positive_rate = 0.9 * det.false_positive_rate + 0.1;
        }
    }

    /// Mark a detection as confirmed (true positive).
    ///
    /// Updates the detector's `false_positive_rate` using an exponential
    /// moving average: `fp = 0.9 * fp + 0.1 * 0.0 = 0.9 * fp`.
    pub fn mark_confirmed(&self, detector_id: &DetectorId) {
        if let Some(mut det) = self.detectors.get_mut(detector_id) {
            det.false_positive_rate *= 0.9;
            det.record_match();
        }
    }

    /// Returns all detectors as a snapshot (cloned).
    pub fn list_detectors(&self) -> Vec<ImmuneDetector> {
        self.detectors.iter().map(|e| e.value().clone()).collect()
    }
}

impl Default for ImmuneSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// L2 norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // ShapeDescriptor tests
    // -------------------------------------------------------------------

    #[test]
    fn shape_descriptor_normalisation() {
        let sd = ShapeDescriptor::new(vec![3.0, 4.0]);
        let norm: f64 = sd.features().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "L2 norm should be 1.0, got {norm}"
        );
        assert!((sd.features()[0] - 0.6).abs() < 1e-10);
        assert!((sd.features()[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn shape_descriptor_zero_vector() {
        let sd = ShapeDescriptor::new(vec![0.0, 0.0, 0.0]);
        assert!(sd.is_empty(), "zero vector should produce empty descriptor");
    }

    #[test]
    fn shape_descriptor_empty() {
        let sd = ShapeDescriptor::new(vec![]);
        assert!(sd.is_empty());
        assert_eq!(sd.dim(), 0);
    }

    #[test]
    fn shape_descriptor_euclidean_distance() {
        let a = ShapeDescriptor::new(vec![1.0, 0.0]);
        let b = ShapeDescriptor::new(vec![0.0, 1.0]);
        // Both are unit vectors, distance = sqrt(2)
        let dist = a.euclidean_distance(&b);
        assert!((dist - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn shape_descriptor_self_distance_zero() {
        let a = ShapeDescriptor::new(vec![1.0, 2.0, 3.0]);
        assert!(a.euclidean_distance(&a) < 1e-10);
    }

    #[test]
    fn shape_descriptor_cosine_similarity_same() {
        let a = ShapeDescriptor::new(vec![1.0, 2.0, 3.0]);
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn shape_descriptor_from_spectral() {
        let spectral = vec![0.5, 0.5, 0.5, 0.5];
        let sd = ShapeDescriptor::from_spectral_signature(&spectral);
        assert_eq!(sd.dim(), 4);
        let norm: f64 = sd.features().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // ImmuneDetector match tests
    // -------------------------------------------------------------------

    #[test]
    fn immune_detector_match_within_radius() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let detector = ImmuneDetector::new(shape, 0.5, EngramId(1));

        // Target very close to detector shape
        let target = ShapeDescriptor::new(vec![0.95, 0.05, 0.0]);
        assert!(detector.matches(&target), "close target should match");
    }

    #[test]
    fn immune_detector_match_outside_radius() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let detector = ImmuneDetector::new(shape, 0.1, EngramId(1));

        // Target orthogonal — distance ≈ sqrt(2) >> 0.1
        let target = ShapeDescriptor::new(vec![0.0, 1.0, 0.0]);
        assert!(
            !detector.matches(&target),
            "distant target should NOT match"
        );
    }

    #[test]
    fn immune_detector_match_exact() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let detector = ImmuneDetector::new(shape.clone(), 0.5, EngramId(1));

        // Exact same shape — distance = 0 < 0.5
        assert!(detector.matches(&shape), "exact same shape should match");
    }

    #[test]
    fn immune_detector_match_empty_target() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let detector = ImmuneDetector::new(shape, 0.5, EngramId(1));
        let empty = ShapeDescriptor::new(vec![]);
        assert!(!detector.matches(&empty), "empty target should not match");
    }

    #[test]
    fn immune_detector_match_boundary() {
        // Exactly at the boundary — should NOT match (strictly less than)
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let detector = ImmuneDetector::new(shape.clone(), 0.0, EngramId(1));
        assert!(
            !detector.matches(&shape),
            "radius=0 with identical shape: distance=0, 0 < 0 is false"
        );
    }

    #[test]
    fn immune_detector_mutation() {
        let shape1 = ShapeDescriptor::new(vec![1.0, 0.0]);
        let mut detector = ImmuneDetector::new(shape1, 0.3, EngramId(1));

        let shape2 = ShapeDescriptor::new(vec![0.7, 0.7]);
        detector.mutate(shape2.clone());

        assert_eq!(detector.mutation_history.len(), 1);
        assert_eq!(detector.shape, shape2);
    }

    // -------------------------------------------------------------------
    // ImmuneSystem scan tests
    // -------------------------------------------------------------------

    #[test]
    fn immune_scan_no_detectors() {
        let system = ImmuneSystem::new();
        let ctx = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let results = system.scan(&ctx);
        assert!(results.is_empty());
    }

    #[test]
    fn immune_scan_single_match() {
        let system = ImmuneSystem::new();
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        system.register(ImmuneDetector::new(shape, 0.5, EngramId(42)));

        let ctx = ShapeDescriptor::new(vec![0.95, 0.05, 0.0]);
        let results = system.scan(&ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_engram, EngramId(42));
        assert!(results[0].confidence() > 0.5);
    }

    #[test]
    fn immune_scan_multiple_detectors() {
        let system = ImmuneSystem::new();

        // Detector A — matches
        let shape_a = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        system.register(ImmuneDetector::new(shape_a, 0.5, EngramId(1)));

        // Detector B — does NOT match (orthogonal, radius too small)
        let shape_b = ShapeDescriptor::new(vec![0.0, 1.0, 0.0]);
        system.register(ImmuneDetector::new(shape_b, 0.1, EngramId(2)));

        // Detector C — matches (close enough with large radius)
        let shape_c = ShapeDescriptor::new(vec![0.8, 0.2, 0.0]);
        system.register(ImmuneDetector::new(shape_c, 1.0, EngramId(3)));

        let ctx = ShapeDescriptor::new(vec![0.9, 0.1, 0.0]);
        let results = system.scan(&ctx);

        // A and C should match, B should not
        assert_eq!(
            results.len(),
            2,
            "expected 2 detections, got {}",
            results.len()
        );
        let engram_ids: Vec<_> = results.iter().map(|d| d.source_engram).collect();
        assert!(engram_ids.contains(&EngramId(1)));
        assert!(engram_ids.contains(&EngramId(3)));
    }

    #[test]
    fn immune_scan_sorted_by_distance() {
        let system = ImmuneSystem::new();

        // Detector A — far
        let shape_a = ShapeDescriptor::new(vec![0.7, 0.7, 0.0]);
        system.register(ImmuneDetector::new(shape_a, 2.0, EngramId(1)));

        // Detector B — close
        let shape_b = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        system.register(ImmuneDetector::new(shape_b, 2.0, EngramId(2)));

        let ctx = ShapeDescriptor::new(vec![0.95, 0.05, 0.0]);
        let results = system.scan(&ctx);
        assert_eq!(results.len(), 2);
        // First result should be the closest
        assert!(results[0].distance <= results[1].distance);
    }

    // -------------------------------------------------------------------
    // Auto-create detector from scarred engram
    // -------------------------------------------------------------------

    #[test]
    fn immune_auto_create_from_scar() {
        let system = ImmuneSystem::new();
        assert_eq!(system.detector_count(), 0);

        // Simulate: engram #99 has a scar, spectral signature = [0.5, 0.3, 0.8, 0.1]
        let spectral = vec![0.5, 0.3, 0.8, 0.1];
        let source_engram = EngramId(99);

        let detector_id = system.create_detector_from_scar(&spectral, source_engram);
        assert_eq!(system.detector_count(), 1);

        let detector = system.get(&detector_id).unwrap();
        assert_eq!(detector.source_engram, source_engram);
        assert!((detector.affinity_radius - DEFAULT_AFFINITY_RADIUS).abs() < 1e-10);

        // The detector shape should be the normalised spectral signature
        let expected = ShapeDescriptor::from_spectral_signature(&spectral);
        assert_eq!(detector.shape, expected);

        // Test that the detector can match a similar context
        let similar_ctx = ShapeDescriptor::new(vec![0.52, 0.28, 0.82, 0.08]);
        assert!(
            detector.matches(&similar_ctx),
            "detector should match a similar context within default radius"
        );
    }

    #[test]
    fn immune_auto_create_detector_matches_original() {
        let system = ImmuneSystem::new();

        let spectral = vec![1.0, 2.0, 3.0, 4.0];
        let source = EngramId(7);
        let did = system.create_detector_from_scar(&spectral, source);

        // The original spectral signature should match its own detector
        let original = ShapeDescriptor::from_spectral_signature(&spectral);
        let det = system.get(&did).unwrap();
        assert!(
            det.matches(&original),
            "detector should match the original spectral signature"
        );
    }

    // -------------------------------------------------------------------
    // Detection confidence
    // -------------------------------------------------------------------

    #[test]
    fn detection_confidence_exact_match() {
        let d = Detection {
            detector_id: DetectorId(1),
            source_engram: EngramId(1),
            distance: 0.0,
            affinity_radius: 0.5,
        };
        assert!((d.confidence() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn detection_confidence_edge() {
        let d = Detection {
            detector_id: DetectorId(1),
            source_engram: EngramId(1),
            distance: 0.25,
            affinity_radius: 0.5,
        };
        assert!((d.confidence() - 0.5).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Hypermutation tests (mutate_towards)
    // -------------------------------------------------------------------

    #[test]
    fn hypermutation_reduces_distance_to_variant() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let variant = ShapeDescriptor::new(vec![0.0, 1.0, 0.0]);
        let mut detector = ImmuneDetector::new(shape.clone(), 0.5, EngramId(1));

        let dist_before = detector.shape.euclidean_distance(&variant);
        detector.mutate_towards(&variant);
        let dist_after = detector.shape.euclidean_distance(&variant);

        assert!(
            dist_after < dist_before,
            "after mutation, distance to variant should decrease: {dist_before} -> {dist_after}"
        );
    }

    #[test]
    fn hypermutation_saves_history_and_increments_clone() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let variant = ShapeDescriptor::new(vec![0.0, 1.0]);
        let mut detector = ImmuneDetector::new(shape.clone(), 0.5, EngramId(1));

        assert_eq!(detector.mutation_history.len(), 0);
        assert_eq!(detector.clone_count, 0);

        detector.mutate_towards(&variant);

        assert_eq!(detector.mutation_history.len(), 1);
        assert_eq!(detector.clone_count, 1);
        // History should contain the original shape
        assert_eq!(detector.mutation_history[0], shape);
    }

    #[test]
    fn hypermutation_multiple_steps_converge() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let variant = ShapeDescriptor::new(vec![0.0, 0.0, 1.0]);
        let mut detector = ImmuneDetector::new(shape, 0.5, EngramId(1));

        let initial_dist = detector.shape.euclidean_distance(&variant);

        // Apply 5 mutations — distance should monotonically decrease
        let mut prev_dist = initial_dist;
        for i in 0..5 {
            detector.mutate_towards(&variant);
            let d = detector.shape.euclidean_distance(&variant);
            assert!(
                d < prev_dist,
                "step {i}: distance should decrease: {prev_dist} -> {d}"
            );
            prev_dist = d;
        }

        assert_eq!(detector.mutation_history.len(), 5);
        assert_eq!(detector.clone_count, 5);
    }

    // -------------------------------------------------------------------
    // Affinity expansion tests (expand_radius)
    // -------------------------------------------------------------------

    #[test]
    fn affinity_expansion_basic() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let mut detector = ImmuneDetector::new(shape, 0.3, EngramId(1));

        let r = detector.expand_radius(1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        assert!((r - 0.33).abs() < 0.01, "radius should grow by 10%: {r}");
    }

    #[test]
    fn affinity_expansion_capped_at_max() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let mut detector = ImmuneDetector::new(shape, 1.8, EngramId(1));

        // 3 expansions at 1.1x — should cap at 2.0
        for _ in 0..3 {
            detector.expand_radius(1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        }
        assert!(
            detector.affinity_radius <= DEFAULT_MAX_AFFINITY_RADIUS,
            "radius {} should not exceed max {}",
            detector.affinity_radius,
            DEFAULT_MAX_AFFINITY_RADIUS,
        );
        assert!(
            (detector.affinity_radius - DEFAULT_MAX_AFFINITY_RADIUS).abs() < 1e-10,
            "radius should be capped at exactly max"
        );
    }

    #[test]
    fn affinity_expansion_three_exposures() {
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut detector = ImmuneDetector::new(shape, 0.3, EngramId(1));

        let initial = detector.affinity_radius;
        detector.expand_radius(1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        detector.expand_radius(1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        detector.expand_radius(1.1, DEFAULT_MAX_AFFINITY_RADIUS);

        let expected = (initial * 1.1 * 1.1 * 1.1).min(DEFAULT_MAX_AFFINITY_RADIUS);
        assert!(
            (detector.affinity_radius - expected).abs() < 1e-10,
            "after 3 expositions, radius should be {expected}, got {}",
            detector.affinity_radius,
        );
        assert!(detector.affinity_radius > initial);
    }

    // -------------------------------------------------------------------
    // Maturation pipeline tests (on_detection)
    // -------------------------------------------------------------------

    #[test]
    fn maturation_pipeline_partial_match_triggers_mutation() {
        let system = ImmuneSystem::new();

        // Detector with radius 1.0 — partial zone is [0.8, 1.2]
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let det_id = system.register(ImmuneDetector::new(shape.clone(), 1.0, EngramId(1)));

        // We need a context whose distance is between 0.8 and 1.2.
        // shape = [1,0,0] normalised. Build a context at known distance ~1.0.
        // Two unit vectors at angle θ have distance 2*sin(θ/2).
        // For dist=1.0, θ = 2*arcsin(0.5) = π/3 = 60°.
        let angle = std::f64::consts::FRAC_PI_3; // 60 degrees → distance = 1.0
        let ctx = ShapeDescriptor::new(vec![angle.cos(), angle.sin(), 0.0]);
        let dist = shape.euclidean_distance(&ctx);

        // Verify the distance is in partial zone
        assert!(
            dist >= 1.0 * PARTIAL_MATCH_LOWER && dist <= 1.0 * PARTIAL_MATCH_UPPER,
            "distance {dist} should be in partial zone [0.8, 1.2]"
        );

        let matured = system.on_detection(&ctx, 1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        assert_eq!(matured.len(), 1, "one detector should mature");
        assert_eq!(matured[0], det_id);

        let det = system.get(&det_id).unwrap();
        assert_eq!(det.clone_count, 1, "clone_count should increment");
        assert_eq!(
            det.mutation_history.len(),
            1,
            "history should have one entry"
        );
        assert!(
            det.affinity_radius > 1.0,
            "radius should have expanded from 1.0"
        );
        // After mutation, shape should be closer to ctx
        assert!(
            det.shape.euclidean_distance(&ctx) < shape.euclidean_distance(&ctx),
            "mutated shape should be closer to the variant"
        );
    }

    #[test]
    fn maturation_pipeline_exact_match_no_maturation() {
        let system = ImmuneSystem::new();
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        system.register(ImmuneDetector::new(shape.clone(), 1.0, EngramId(1)));

        // Exact match — distance ≈ 0, which is < 0.8 (lower bound)
        let matured = system.on_detection(&shape, 1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        assert!(
            matured.is_empty(),
            "exact match should NOT trigger maturation (not a partial match)"
        );
    }

    #[test]
    fn maturation_pipeline_far_miss_no_maturation() {
        let system = ImmuneSystem::new();
        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        system.register(ImmuneDetector::new(shape, 0.3, EngramId(1)));

        // Orthogonal — distance ≈ sqrt(2) >> 0.3*1.2
        let far = ShapeDescriptor::new(vec![0.0, 1.0, 0.0]);
        let matured = system.on_detection(&far, 1.1, DEFAULT_MAX_AFFINITY_RADIUS);
        assert!(matured.is_empty(), "far miss should NOT trigger maturation");
    }

    // -------------------------------------------------------------------
    // Idiotypic network tests
    // -------------------------------------------------------------------

    #[test]
    fn idiotypic_network_finds_similar_pairs() {
        let system = ImmuneSystem::new();

        // A and B are very similar (cosine ~ 0.99)
        let shape_a = ShapeDescriptor::new(vec![1.0, 0.1, 0.0]);
        let shape_b = ShapeDescriptor::new(vec![1.0, 0.15, 0.0]);
        // C is orthogonal (cosine ~ 0)
        let shape_c = ShapeDescriptor::new(vec![0.0, 0.0, 1.0]);

        system.register(ImmuneDetector::new(shape_a, 0.5, EngramId(1)));
        system.register(ImmuneDetector::new(shape_b, 0.5, EngramId(2)));
        system.register(ImmuneDetector::new(shape_c, 0.5, EngramId(3)));

        let pairs = system.find_idiotypic_pairs(ImmuneSystem::IDIOTYPIC_THRESHOLD);
        assert_eq!(pairs.len(), 1, "only A-B should be an idiotypic pair");
        assert!(
            pairs[0].2 > ImmuneSystem::IDIOTYPIC_THRESHOLD,
            "cosine should be above threshold"
        );
    }

    #[test]
    fn idiotypic_neighbours_returns_only_close() {
        let system = ImmuneSystem::new();

        let shape_a = ShapeDescriptor::new(vec![1.0, 0.1, 0.0]);
        let id_a = system.register(ImmuneDetector::new(shape_a, 0.5, EngramId(1)));

        // B is similar to A
        let shape_b = ShapeDescriptor::new(vec![1.0, 0.2, 0.0]);
        let id_b = system.register(ImmuneDetector::new(shape_b, 0.5, EngramId(2)));

        // C is orthogonal
        let shape_c = ShapeDescriptor::new(vec![0.0, 0.0, 1.0]);
        system.register(ImmuneDetector::new(shape_c, 0.5, EngramId(3)));

        let neighbours = system.idiotypic_neighbours(&id_a);
        assert_eq!(neighbours.len(), 1);
        assert_eq!(neighbours[0].0, id_b);
    }

    #[test]
    fn idiotypic_propagation_activates_neighbours_not_unrelated() {
        let system = ImmuneSystem::new();

        // A and B are similar
        let shape_a = ShapeDescriptor::new(vec![1.0, 0.05, 0.0]);
        let id_a = system.register(ImmuneDetector::new(shape_a, 0.5, EngramId(1)));

        let shape_b = ShapeDescriptor::new(vec![1.0, 0.1, 0.0]);
        let id_b = system.register(ImmuneDetector::new(shape_b, 0.5, EngramId(2)));

        // C is orthogonal
        let shape_c = ShapeDescriptor::new(vec![0.0, 0.0, 1.0]);
        let id_c = system.register(ImmuneDetector::new(shape_c, 0.5, EngramId(3)));

        // Activate A → should propagate to B but NOT C
        let propagated = system.idiotypic_propagate(&id_a);
        let propagated_ids: Vec<DetectorId> = propagated.iter().map(|p| p.0).collect();

        assert!(
            propagated_ids.contains(&id_b),
            "B should be activated by A"
        );
        assert!(
            !propagated_ids.contains(&id_c),
            "C should NOT be activated by A"
        );

        // B's clone_count should have incremented
        let det_b = system.get(&id_b).unwrap();
        assert_eq!(det_b.clone_count, 1, "B clone_count should be 1");

        // C's clone_count should remain 0
        let det_c = system.get(&id_c).unwrap();
        assert_eq!(det_c.clone_count, 0, "C clone_count should remain 0");
    }

    #[test]
    fn idiotypic_propagation_weight_is_bounded() {
        let system = ImmuneSystem::new();

        // Two nearly identical detectors (cosine ~ 1.0)
        let shape_a = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let id_a = system.register(ImmuneDetector::new(shape_a.clone(), 0.5, EngramId(1)));
        system.register(ImmuneDetector::new(shape_a, 0.5, EngramId(2)));

        let propagated = system.idiotypic_propagate(&id_a);
        assert_eq!(propagated.len(), 1);
        // Weight = similarity × 0.5 — for cosine ~1.0 → weight ~0.5
        assert!(
            propagated[0].1 <= ImmuneSystem::IDIOTYPIC_PROPAGATION_FACTOR + 0.01,
            "weight should be ≤ propagation factor, got {}",
            propagated[0].1
        );
        assert!(propagated[0].1 > 0.3, "weight should be substantial for similar detectors");
    }

    // -------------------------------------------------------------------
    // T-reg regulation tests
    // -------------------------------------------------------------------

    #[test]
    fn t_reg_shrinks_high_fp_detector() {
        let system = ImmuneSystem::new();

        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 1.0, EngramId(1));
        det.false_positive_rate = 0.4; // above 0.3 threshold
        let id = system.register(det);

        let regulated = system.regulate_immune();
        assert_eq!(regulated.len(), 1);
        assert_eq!(regulated[0].0, id);

        let det = system.get(&id).unwrap();
        assert!(
            (det.affinity_radius - 0.8).abs() < 1e-10,
            "radius should be 1.0 × 0.8 = 0.8, got {}",
            det.affinity_radius
        );
    }

    #[test]
    fn t_reg_ignores_low_fp_detector() {
        let system = ImmuneSystem::new();

        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 1.0, EngramId(1));
        det.false_positive_rate = 0.1; // below threshold
        system.register(det);

        let regulated = system.regulate_immune();
        assert!(regulated.is_empty(), "low FP detector should not be regulated");
    }

    #[test]
    fn t_reg_repeated_regulation_shrinks_progressively() {
        let system = ImmuneSystem::new();

        let shape = ShapeDescriptor::new(vec![1.0, 0.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 1.0, EngramId(1));
        det.false_positive_rate = 0.5; // stays high
        let id = system.register(det);

        // 3 rounds of regulation
        for _ in 0..3 {
            system.regulate_immune();
        }

        let det = system.get(&id).unwrap();
        let expected = 1.0 * 0.8 * 0.8 * 0.8; // 0.512
        assert!(
            (det.affinity_radius - expected).abs() < 1e-10,
            "after 3 regulations, radius should be {expected}, got {}",
            det.affinity_radius
        );
    }

    #[test]
    fn mark_rejected_increases_fp_rate() {
        let system = ImmuneSystem::new();
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let det = ImmuneDetector::new(shape, 0.5, EngramId(1));
        let id = system.register(det);

        // Initial FP rate is 0.0
        assert!((system.get(&id).unwrap().false_positive_rate - 0.0).abs() < 1e-10);

        // Mark 5 rejections — FP rate should climb
        for _ in 0..5 {
            system.mark_rejected(&id);
        }

        let fp = system.get(&id).unwrap().false_positive_rate;
        assert!(fp > 0.3, "after 5 rejections, FP rate should be > 0.3, got {fp}");
    }

    #[test]
    fn mark_confirmed_decreases_fp_rate() {
        let system = ImmuneSystem::new();
        let shape = ShapeDescriptor::new(vec![1.0, 0.0]);
        let mut det = ImmuneDetector::new(shape, 0.5, EngramId(1));
        det.false_positive_rate = 0.5;
        let id = system.register(det);

        // Confirm 5 times — FP rate should decrease
        for _ in 0..5 {
            system.mark_confirmed(&id);
        }

        let fp = system.get(&id).unwrap().false_positive_rate;
        assert!(fp < 0.5, "after confirmations, FP rate should decrease, got {fp}");
        // Also clone_count should have incremented
        assert_eq!(system.get(&id).unwrap().clone_count, 5);
    }
}
