//! Crystallization — automatic engram → note promotion.
//!
//! When an engram reaches high strength (> 0.85) after multiple reactivations
//! (≥ 5), the system proposes crystallization: generating a textual summary
//! and creating a `:CRYSTALLIZED_IN` relationship to a Note node.
//!
//! The `CrystallizationDetector` scans engrams and emits proposals.
//! The `crystallize` function performs the actual promotion via `CognitiveStorage`.

use std::collections::HashMap;

use obrain_common::types::{EdgeId, NodeId, Value};
use tracing::debug;

use super::observe::EngramMetricsCollector;
use super::store::EngramStore;
use super::traits::CognitiveStorage;
use super::types::{Engram, EngramId};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for crystallization detection.
#[derive(Debug, Clone)]
pub struct CrystallizationConfig {
    /// Minimum strength to be eligible for crystallization (default: 0.85).
    pub strength_threshold: f64,
    /// Minimum number of recalls required (default: 5).
    pub min_recall_count: u32,
}

impl Default for CrystallizationConfig {
    fn default() -> Self {
        Self {
            strength_threshold: 0.85,
            min_recall_count: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// CrystallizationProposal
// ---------------------------------------------------------------------------

/// A proposal to crystallize an engram into a permanent note.
#[derive(Debug, Clone)]
pub struct CrystallizationProposal {
    /// The engram that should be crystallized.
    pub engram_id: EngramId,
    /// The generated summary text for the note content.
    pub summary: String,
    /// The engram's current strength at time of proposal.
    pub strength: f64,
    /// The engram's recall count at time of proposal.
    pub recall_count: u32,
    /// The engram's valence.
    pub valence: f64,
}

// ---------------------------------------------------------------------------
// CrystallizationDetector
// ---------------------------------------------------------------------------

/// Detects engrams eligible for crystallization (strength > threshold AND
/// recall_count >= min_recalls) and emits proposals.
#[derive(Debug)]
pub struct CrystallizationDetector {
    config: CrystallizationConfig,
}

impl CrystallizationDetector {
    /// Creates a new detector with the given configuration.
    pub fn new(config: CrystallizationConfig) -> Self {
        Self { config }
    }

    /// Creates a new detector with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CrystallizationConfig::default())
    }

    /// Checks whether a single engram is eligible for crystallization.
    pub fn is_eligible(&self, engram: &Engram) -> bool {
        engram.strength > self.config.strength_threshold
            && engram.recall_count >= self.config.min_recall_count
    }

    /// Scans all engrams in the store and returns proposals for eligible ones.
    pub fn detect(&self, store: &EngramStore) -> Vec<CrystallizationProposal> {
        store
            .list()
            .into_iter()
            .filter(|e| self.is_eligible(e))
            .map(|e| CrystallizationProposal {
                engram_id: e.id,
                summary: generate_summary(&e),
                strength: e.strength,
                recall_count: e.recall_count,
                valence: e.valence,
            })
            .collect()
    }

    /// Checks a single engram and returns a proposal if eligible.
    pub fn check_engram(&self, engram: &Engram) -> Option<CrystallizationProposal> {
        if self.is_eligible(engram) {
            Some(CrystallizationProposal {
                engram_id: engram.id,
                summary: generate_summary(engram),
                strength: engram.strength,
                recall_count: engram.recall_count,
                valence: engram.valence,
            })
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// generate_summary — textual summary of an engram
// ---------------------------------------------------------------------------

/// Generates a human-readable summary of an engram for crystallization.
///
/// The summary includes:
/// - The ensemble of nodes (IDs and weights)
/// - A lesson indicator derived from valence
/// - A valence indicator (warning for negative, success for positive)
///
/// # Examples
///
/// An engram with 3 nodes and valence -0.8 produces a summary containing
/// all node labels and a "⚠ warning" indicator.
pub fn generate_summary(engram: &Engram) -> String {
    let mut parts = Vec::new();

    // Header with engram identity
    parts.push(format!("Engram {} — Crystallized Pattern", engram.id));

    // Ensemble description
    let node_labels: Vec<String> = engram
        .ensemble
        .iter()
        .map(|(node_id, weight)| format!("node:{} (w={:.2})", node_id.0, weight))
        .collect();
    parts.push(format!("Ensemble: [{}]", node_labels.join(", ")));

    // Lesson derived from valence and recall history
    let lesson = derive_lesson(engram);
    parts.push(format!("Lesson: {}", lesson));

    // Valence indicator
    let valence_indicator = valence_label(engram.valence);
    parts.push(format!(
        "Valence: {:.2} — {}",
        engram.valence, valence_indicator
    ));

    // Strength and recall stats
    parts.push(format!(
        "Strength: {:.2} | Recalls: {} | Horizon: {}",
        engram.strength, engram.recall_count, engram.horizon
    ));

    parts.join("\n")
}

/// Derives a lesson string from the engram's valence and ensemble.
fn derive_lesson(engram: &Engram) -> String {
    let node_count = engram.ensemble.len();
    if engram.valence < -0.5 {
        format!(
            "Recurring negative pattern across {} nodes — avoid or mitigate",
            node_count
        )
    } else if engram.valence > 0.5 {
        format!(
            "Proven successful pattern across {} nodes — replicate",
            node_count
        )
    } else {
        format!(
            "Neutral pattern across {} nodes — observed correlation",
            node_count
        )
    }
}

/// Returns a human-readable label for a valence value.
fn valence_label(valence: f64) -> &'static str {
    if valence < -0.5 {
        "⚠ warning"
    } else if valence < -0.1 {
        "caution"
    } else if valence > 0.5 {
        "✓ success"
    } else if valence > 0.1 {
        "positive"
    } else {
        "neutral"
    }
}

// ---------------------------------------------------------------------------
// crystallize — execute the crystallization
// ---------------------------------------------------------------------------

/// The label used for crystallized note nodes.
pub const LABEL_CRYSTALLIZED_NOTE: &str = "Note";

/// The relationship type for engram → note crystallization.
pub const REL_CRYSTALLIZED_IN: &str = "CRYSTALLIZED_IN";

/// Result of a successful crystallization.
#[derive(Debug, Clone)]
pub struct CrystallizationResult {
    /// The engram that was crystallized.
    pub engram_id: EngramId,
    /// The NodeId of the created Note node.
    pub note_id: NodeId,
    /// The EdgeId of the CRYSTALLIZED_IN relationship.
    pub edge_id: EdgeId,
    /// The summary that was written to the note.
    pub summary: String,
}

/// Executes crystallization: creates a Note node with the engram summary
/// and a `:CRYSTALLIZED_IN` edge from the engram's storage node to the note.
///
/// # Arguments
/// - `storage` — the cognitive storage backend for creating nodes/edges
/// - `engram` — the engram to crystallize
/// - `engram_node_id` — the NodeId of the engram in the graph (for creating the edge)
/// - `metrics` — metrics collector to increment `crystallization_proposals`
///
/// Returns the crystallization result with IDs of created entities.
pub fn crystallize(
    storage: &dyn CognitiveStorage,
    engram: &Engram,
    engram_node_id: NodeId,
    metrics: &EngramMetricsCollector,
) -> CrystallizationResult {
    let summary = generate_summary(engram);

    // 1. Create the Note node
    let mut note_props = HashMap::new();
    note_props.insert("content".to_string(), Value::String(summary.clone().into()));
    note_props.insert(
        "source".to_string(),
        Value::String(format!("crystallized:{}", engram.id).into()),
    );
    note_props.insert("valence".to_string(), Value::Float64(engram.valence));
    note_props.insert("strength".to_string(), Value::Float64(engram.strength));

    let note_id = storage.create_node(LABEL_CRYSTALLIZED_NOTE, &note_props);

    // 2. Create the CRYSTALLIZED_IN edge
    let mut edge_props = HashMap::new();
    edge_props.insert("strength".to_string(), Value::Float64(engram.strength));
    edge_props.insert(
        "recall_count".to_string(),
        Value::Int64(engram.recall_count as i64),
    );

    let edge_id = storage.create_edge(engram_node_id, note_id, REL_CRYSTALLIZED_IN, &edge_props);

    // 3. Increment crystallization_proposals metric
    metrics.record_crystallization();

    debug!(
        engram_id = %engram.id,
        note_id = ?note_id,
        "engram crystallized into note"
    );

    CrystallizationResult {
        engram_id: engram.id,
        note_id,
        edge_id,
        summary,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::types::{Engram, EngramId};

    // -- In-memory CognitiveStorage for tests --

    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockStorage {
        next_node: AtomicU64,
        next_edge: AtomicU64,
        nodes: parking_lot::Mutex<Vec<(NodeId, String, HashMap<String, Value>)>>,
        edges: parking_lot::Mutex<Vec<(EdgeId, NodeId, NodeId, String, HashMap<String, Value>)>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                next_node: AtomicU64::new(1000),
                next_edge: AtomicU64::new(2000),
                nodes: parking_lot::Mutex::new(Vec::new()),
                edges: parking_lot::Mutex::new(Vec::new()),
            }
        }
    }

    impl CognitiveStorage for MockStorage {
        fn create_node(&self, label: &str, properties: &HashMap<String, Value>) -> NodeId {
            let id = NodeId(self.next_node.fetch_add(1, Ordering::Relaxed));
            self.nodes
                .lock()
                .push((id, label.to_string(), properties.clone()));
            id
        }

        fn create_edge(
            &self,
            from: NodeId,
            to: NodeId,
            rel_type: &str,
            properties: &HashMap<String, Value>,
        ) -> EdgeId {
            let id = EdgeId(self.next_edge.fetch_add(1, Ordering::Relaxed));
            self.edges
                .lock()
                .push((id, from, to, rel_type.to_string(), properties.clone()));
            id
        }

        fn query_nodes(
            &self,
            _label: &str,
            _filter: Option<&super::super::traits::CognitiveFilter>,
        ) -> Vec<super::super::traits::CognitiveNode> {
            Vec::new()
        }

        fn update_node(&self, _id: NodeId, _properties: &HashMap<String, Value>) {}
        fn delete_node(&self, _id: NodeId) {}
        fn delete_edge(&self, _id: EdgeId) {}

        fn query_edges(
            &self,
            _from: NodeId,
            _rel_type: &str,
        ) -> Vec<super::super::traits::CognitiveEdge> {
            Vec::new()
        }
    }

    // -- Helper to build test engrams --

    fn make_engram(strength: f64, recall_count: u32, valence: f64, node_count: usize) -> Engram {
        let ensemble: Vec<(NodeId, f64)> = (0..node_count)
            .map(|i| (NodeId(i as u64 + 1), 1.0 / (i as f64 + 1.0)))
            .collect();
        let mut e = Engram::new(EngramId(42), ensemble);
        e.strength = strength;
        e.recall_count = recall_count;
        e.valence = valence;
        e
    }

    // -----------------------------------------------------------------------
    // Step 1: CrystallizationDetector tests
    // -----------------------------------------------------------------------

    #[test]
    fn detector_eligible_high_strength_high_recalls() {
        let detector = CrystallizationDetector::with_defaults();
        // strength 0.9 > 0.85, recall_count 6 >= 5 → eligible
        let engram = make_engram(0.9, 6, 0.0, 3);
        assert!(detector.is_eligible(&engram));
        let proposal = detector.check_engram(&engram);
        assert!(proposal.is_some());
    }

    #[test]
    fn detector_not_eligible_low_recalls() {
        let detector = CrystallizationDetector::with_defaults();
        // strength 0.9 > 0.85, but recall_count 3 < 5 → NOT eligible
        let engram = make_engram(0.9, 3, 0.0, 3);
        assert!(!detector.is_eligible(&engram));
        let proposal = detector.check_engram(&engram);
        assert!(proposal.is_none());
    }

    #[test]
    fn detector_not_eligible_low_strength() {
        let detector = CrystallizationDetector::with_defaults();
        // strength 0.5 <= 0.85 → NOT eligible regardless of recalls
        let engram = make_engram(0.5, 10, 0.0, 3);
        assert!(!detector.is_eligible(&engram));
    }

    #[test]
    fn detector_boundary_not_eligible_at_exact_threshold() {
        let detector = CrystallizationDetector::with_defaults();
        // strength == 0.85 is NOT > 0.85 → not eligible
        let engram = make_engram(0.85, 5, 0.0, 3);
        assert!(!detector.is_eligible(&engram));
    }

    #[test]
    fn detector_boundary_just_above_threshold() {
        let detector = CrystallizationDetector::with_defaults();
        // strength 0.851 > 0.85, recall_count 5 >= 5 → eligible
        let engram = make_engram(0.851, 5, 0.0, 3);
        assert!(detector.is_eligible(&engram));
    }

    #[test]
    fn detector_scan_store() {
        let store = EngramStore::new(None);
        // Insert one eligible and one not eligible
        let mut e1 = make_engram(0.9, 6, 0.3, 2);
        e1.id = store.next_id();
        store.insert(e1);

        let mut e2 = make_engram(0.5, 2, 0.0, 2);
        e2.id = store.next_id();
        store.insert(e2);

        let detector = CrystallizationDetector::with_defaults();
        let proposals = detector.detect(&store);
        assert_eq!(proposals.len(), 1);
        assert!(proposals[0].strength > 0.85);
    }

    // -----------------------------------------------------------------------
    // Step 2: generate_summary tests
    // -----------------------------------------------------------------------

    #[test]
    fn summary_contains_node_labels() {
        let engram = make_engram(0.9, 6, -0.8, 3);
        let summary = generate_summary(&engram);
        // Should contain all 3 node IDs
        assert!(summary.contains("node:1"), "summary should contain node:1");
        assert!(summary.contains("node:2"), "summary should contain node:2");
        assert!(summary.contains("node:3"), "summary should contain node:3");
    }

    #[test]
    fn summary_negative_valence_shows_warning() {
        let engram = make_engram(0.9, 6, -0.8, 3);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("warning"),
            "negative valence should show warning indicator, got: {}",
            summary
        );
    }

    #[test]
    fn summary_positive_valence_shows_success() {
        let engram = make_engram(0.9, 6, 0.8, 2);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("success"),
            "positive valence should show success indicator, got: {}",
            summary
        );
    }

    #[test]
    fn summary_neutral_valence() {
        let engram = make_engram(0.9, 6, 0.0, 2);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("neutral"),
            "neutral valence should show neutral, got: {}",
            summary
        );
    }

    #[test]
    fn summary_contains_valence_value() {
        let engram = make_engram(0.9, 6, -0.8, 3);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("-0.80"),
            "summary should contain valence value, got: {}",
            summary
        );
    }

    #[test]
    fn summary_contains_lesson_for_negative() {
        let engram = make_engram(0.9, 6, -0.8, 3);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("avoid or mitigate"),
            "negative lesson should advise avoidance, got: {}",
            summary
        );
    }

    // -----------------------------------------------------------------------
    // Step 3: crystallize + CRYSTALLIZED_IN + metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn crystallize_creates_note_and_edge() {
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();
        let engram = make_engram(0.9, 6, -0.5, 3);
        let engram_node_id = NodeId(100);

        let result = crystallize(&storage, &engram, engram_node_id, &metrics);

        // Check note was created
        let nodes = storage.nodes.lock();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].1, "Note");
        assert!(nodes[0].2.contains_key("content"));

        // Check CRYSTALLIZED_IN edge was created
        let edges = storage.edges.lock();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].1, engram_node_id); // from
        assert_eq!(edges[0].2, result.note_id); // to
        assert_eq!(edges[0].3, "CRYSTALLIZED_IN");

        // Check result
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn crystallize_increments_metrics() {
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();
        let engram = make_engram(0.9, 6, 0.3, 2);
        let engram_node_id = NodeId(100);

        assert_eq!(metrics.snapshot().engrams_crystallized, 0);

        crystallize(&storage, &engram, engram_node_id, &metrics);

        assert_eq!(metrics.snapshot().engrams_crystallized, 1);
    }

    #[test]
    fn crystallize_multiple_increments_metrics() {
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();

        for i in 0..3 {
            let mut engram = make_engram(0.9, 6, 0.3, 2);
            engram.id = EngramId(i);
            crystallize(&storage, &engram, NodeId(100 + i), &metrics);
        }

        assert_eq!(metrics.snapshot().engrams_crystallized, 3);
    }

    // -----------------------------------------------------------------------
    // Additional coverage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_config_thresholds() {
        let config = CrystallizationConfig {
            strength_threshold: 0.5,
            min_recall_count: 2,
        };
        let detector = CrystallizationDetector::new(config);

        // strength 0.6 > 0.5, recall 2 >= 2 → eligible
        let eligible = make_engram(0.6, 2, 0.0, 2);
        assert!(detector.is_eligible(&eligible));

        // strength 0.4 <= 0.5 → not eligible
        let not_eligible = make_engram(0.4, 5, 0.0, 2);
        assert!(!detector.is_eligible(&not_eligible));
    }

    #[test]
    fn test_valence_label_caution_range() {
        // -0.3 is between -0.5 and -0.1 → "caution"
        assert_eq!(valence_label(-0.3), "caution");
    }

    #[test]
    fn test_valence_label_positive_range() {
        // 0.3 is between 0.1 and 0.5 → "positive"
        assert_eq!(valence_label(0.3), "positive");
    }

    #[test]
    fn test_valence_label_boundary_negative_05() {
        // -0.5 is NOT < -0.5, so it falls into "caution" range (-0.5 < -0.1 is true)
        assert_eq!(valence_label(-0.5), "caution");
    }

    #[test]
    fn test_valence_label_boundary_positive_05() {
        // 0.5 is NOT > 0.5, so it falls into "positive" range (0.5 > 0.1 is true)
        assert_eq!(valence_label(0.5), "positive");
    }

    #[test]
    fn test_summary_contains_strength_and_recalls() {
        let engram = make_engram(0.92, 7, 0.0, 2);
        let summary = generate_summary(&engram);
        assert!(
            summary.contains("Strength: 0.92"),
            "summary should contain strength: {summary}"
        );
        assert!(
            summary.contains("Recalls: 7"),
            "summary should contain recalls: {summary}"
        );
    }

    #[test]
    fn test_summary_empty_ensemble() {
        let engram = make_engram(0.9, 6, 0.0, 0);
        let summary = generate_summary(&engram);
        // Should not panic with 0 nodes, ensemble should be "[]"
        assert!(
            summary.contains("Ensemble: []"),
            "empty ensemble should produce empty brackets: {summary}"
        );
    }

    #[test]
    fn test_crystallize_note_properties_correct() {
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();
        let engram = make_engram(0.91, 8, -0.3, 2);
        let engram_node_id = NodeId(200);

        let result = crystallize(&storage, &engram, engram_node_id, &metrics);

        let nodes = storage.nodes.lock();
        let props = &nodes[0].2;
        // Check "content" is a non-empty string
        if let Value::String(s) = props.get("content").unwrap() {
            assert!(!s.is_empty());
        } else {
            panic!("content should be String");
        }
        // Check "source" = "crystallized:42"
        assert_eq!(
            props.get("source"),
            Some(&Value::String(format!("crystallized:{}", engram.id).into()))
        );
        // Check valence and strength
        assert_eq!(props.get("valence"), Some(&Value::Float64(-0.3)));
        assert_eq!(props.get("strength"), Some(&Value::Float64(0.91)));
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_crystallize_edge_properties_correct() {
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();
        let engram = make_engram(0.88, 6, 0.2, 2);
        let engram_node_id = NodeId(300);

        crystallize(&storage, &engram, engram_node_id, &metrics);

        let edges = storage.edges.lock();
        let props = &edges[0].4;
        assert_eq!(props.get("strength"), Some(&Value::Float64(0.88)));
        assert_eq!(props.get("recall_count"), Some(&Value::Int64(6)));
    }

    #[test]
    fn test_detector_many_engrams_in_store() {
        let store = EngramStore::new(None);
        let detector = CrystallizationDetector::with_defaults();

        // Insert 100 engrams, only 10 eligible (strength > 0.85, recall >= 5)
        for i in 0..100u64 {
            let mut e = if i < 10 {
                make_engram(0.9, 6, 0.0, 2)
            } else {
                make_engram(0.5, 2, 0.0, 2)
            };
            e.id = store.next_id();
            store.insert(e);
        }

        let proposals = detector.detect(&store);
        assert_eq!(proposals.len(), 10);
    }

    #[test]
    fn test_check_engram_proposal_fields_match() {
        let detector = CrystallizationDetector::with_defaults();
        let engram = make_engram(0.95, 10, -0.7, 3);
        let proposal = detector.check_engram(&engram).unwrap();

        assert!((proposal.strength - 0.95).abs() < f64::EPSILON);
        assert_eq!(proposal.recall_count, 10);
        assert!((proposal.valence - (-0.7)).abs() < f64::EPSILON);
        assert_eq!(proposal.engram_id, engram.id);
    }

    #[test]
    fn full_flow_detect_and_crystallize() {
        let store = EngramStore::new(None);
        let storage = MockStorage::new();
        let metrics = EngramMetricsCollector::new();

        // Add eligible engram
        let mut e = make_engram(0.9, 6, -0.7, 3);
        e.id = store.next_id();
        let engram_node_id = NodeId(e.id.0);
        store.insert(e.clone());

        // Add non-eligible engram
        let mut e2 = make_engram(0.3, 1, 0.0, 2);
        e2.id = store.next_id();
        store.insert(e2);

        // Detect
        let detector = CrystallizationDetector::with_defaults();
        let proposals = detector.detect(&store);
        assert_eq!(proposals.len(), 1);

        // Crystallize the eligible one
        let engram = store.get(proposals[0].engram_id).unwrap();
        let result = crystallize(&storage, &engram, engram_node_id, &metrics);

        // Verify
        assert_eq!(metrics.snapshot().engrams_crystallized, 1);
        assert!(result.summary.contains("warning"));

        let edges = storage.edges.lock();
        assert_eq!(edges[0].3, "CRYSTALLIZED_IN");
    }
}
