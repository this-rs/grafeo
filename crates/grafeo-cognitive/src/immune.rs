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
    pub fn new(
        shape: ShapeDescriptor,
        affinity_radius: f64,
        source_engram: EngramId,
    ) -> Self {
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
    pub fn get(&self, id: &DetectorId) -> Option<dashmap::mapref::one::Ref<'_, DetectorId, ImmuneDetector>> {
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
        detections.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        detections
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
        assert!((norm - 1.0).abs() < 1e-10, "L2 norm should be 1.0, got {norm}");
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
        assert!(!detector.matches(&target), "distant target should NOT match");
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
        assert_eq!(results.len(), 2, "expected 2 detections, got {}", results.len());
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
}
