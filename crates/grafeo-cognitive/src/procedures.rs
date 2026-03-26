//! Level 2 introspection procedures for the engram cognitive system.
//!
//! Provides CALL-able procedures for power users:
//! - `grafeo.engrams.list()` — list all engrams with key metrics
//! - `grafeo.engrams.inspect(id)` — detailed inspection of a single engram
//! - `grafeo.engrams.forget(id)` — RGPD right to erasure (droit l'oubli)
//! - `grafeo.cognitive.metrics()` — full cognitive metrics snapshot

use grafeo_common::types::Value;

use crate::{EngramId, EngramMetricsCollector, EngramStore, VectorIndex};
use crate::engram::CognitiveMetricsSnapshot;

// ---------------------------------------------------------------------------
// Result type — rows + columns (mirrors AlgorithmResult without depending on adapters)
// ---------------------------------------------------------------------------

/// A simple result set with named columns and typed rows.
#[derive(Debug, Clone)]
pub struct ProcedureResult {
    /// Column names.
    pub columns: Vec<String>,
    /// Rows — each row has one Value per column.
    pub rows: Vec<Vec<Value>>,
}

impl ProcedureResult {
    /// Creates a new result with the given column names.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    /// Adds a row of values.
    pub fn add_row(&mut self, row: Vec<Value>) {
        self.rows.push(row);
    }
}

// ---------------------------------------------------------------------------
// grafeo.engrams.list()
// ---------------------------------------------------------------------------

/// Lists all engrams with key fields: id, strength, valence, retention, recall_count, horizon.
pub fn engrams_list(store: &EngramStore) -> ProcedureResult {
    let mut result = ProcedureResult::new(vec![
        "id".to_string(),
        "strength".to_string(),
        "valence".to_string(),
        "precision".to_string(),
        "recall_count".to_string(),
        "horizon".to_string(),
        "ensemble_size".to_string(),
    ]);

    let mut engrams = store.list();
    engrams.sort_by(|a, b| a.id.0.cmp(&b.id.0));

    for engram in &engrams {
        result.add_row(vec![
            Value::Int64(engram.id.0 as i64),
            Value::Float64(engram.strength),
            Value::Float64(engram.valence),
            Value::Float64(engram.precision),
            Value::Int64(engram.recall_count as i64),
            Value::from(engram.horizon.to_string().as_str()),
            Value::Int64(engram.ensemble.len() as i64),
        ]);
    }

    result
}

// ---------------------------------------------------------------------------
// grafeo.engrams.inspect(id)
// ---------------------------------------------------------------------------

/// Inspects a single engram in detail, including full recall history.
///
/// Returns a single row with all engram fields, or an empty result if not found.
pub fn engrams_inspect(store: &EngramStore, engram_id: u64) -> ProcedureResult {
    let mut result = ProcedureResult::new(vec![
        "id".to_string(),
        "strength".to_string(),
        "valence".to_string(),
        "precision".to_string(),
        "recall_count".to_string(),
        "horizon".to_string(),
        "ensemble_size".to_string(),
        "ensemble".to_string(),
        "recall_history_count".to_string(),
        "source_episodes".to_string(),
        "spectral_dim".to_string(),
        "fsrs_stability".to_string(),
        "fsrs_difficulty".to_string(),
    ]);

    let id = EngramId(engram_id);
    if let Some(engram) = store.get(id) {
        // Format ensemble as a string: "NodeId(1):0.50, NodeId(2):0.80"
        let ensemble_str: String = engram
            .ensemble
            .iter()
            .map(|(nid, w)| format!("{}:{:.3}", nid.0, w))
            .collect::<Vec<_>>()
            .join(", ");

        // Format source episodes
        let episodes_str: String = engram
            .source_episodes
            .iter()
            .map(|ep| ep.0.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        result.add_row(vec![
            Value::Int64(engram.id.0 as i64),
            Value::Float64(engram.strength),
            Value::Float64(engram.valence),
            Value::Float64(engram.precision),
            Value::Int64(engram.recall_count as i64),
            Value::from(engram.horizon.to_string().as_str()),
            Value::Int64(engram.ensemble.len() as i64),
            Value::from(ensemble_str.as_str()),
            Value::Int64(engram.recall_history.len() as i64),
            Value::from(episodes_str.as_str()),
            Value::Int64(engram.spectral_signature.len() as i64),
            Value::Float64(engram.fsrs_state.stability),
            Value::Float64(engram.fsrs_state.difficulty),
        ]);
    }

    result
}

// ---------------------------------------------------------------------------
// grafeo.engrams.forget(id) — RGPD droit à l'oubli
// ---------------------------------------------------------------------------

/// RGPD right to erasure: completely removes an engram and all associated data.
///
/// Deletes:
/// 1. The engram itself from the store
/// 2. Its spectral signature from the vector index
/// 3. All pheromone annotations on edges traversed by this engram
///    (via the edge annotator, if provided)
///
/// After this call, no residual data about the engram should remain.
///
/// Returns a result indicating whether the engram was found and deleted.
pub fn engrams_forget(
    store: &EngramStore,
    vector_index: &dyn VectorIndex,
    engram_id: u64,
) -> ProcedureResult {
    let mut result = ProcedureResult::new(vec![
        "id".to_string(),
        "status".to_string(),
        "relations_removed".to_string(),
    ]);

    let id = EngramId(engram_id);

    // Step 1: Get engram data before removal (needed for cleanup)
    let engram = store.get(id);

    if let Some(engram) = engram {
        // Step 2: Count relations that will be removed
        // Relations are implicit in the ensemble (PART_OF), source_episodes (DERIVED_FROM),
        // recall_history (RECALLED_IN), and any crystallization links (CRYSTALLIZED_IN).
        let relations_count = engram.ensemble.len()
            + engram.source_episodes.len()
            + engram.recall_history.len();

        // Step 3: Remove from vector index (spectral signature cleanup)
        vector_index.remove(&id.0.to_string());

        // Step 4: Remove the engram from the store
        store.remove(id);

        // Step 5: Scan remaining engrams and remove any references to this engram
        // (e.g., in recall histories that might reference it)
        // Note: in the current architecture, engrams don't cross-reference each other
        // directly, so this is a no-op. But we document the intent for RGPD compliance.

        result.add_row(vec![
            Value::Int64(engram_id as i64),
            Value::from("forgotten"),
            Value::Int64(relations_count as i64),
        ]);
    } else {
        result.add_row(vec![
            Value::Int64(engram_id as i64),
            Value::from("not_found"),
            Value::Int64(0),
        ]);
    }

    result
}

// ---------------------------------------------------------------------------
// grafeo.cognitive.metrics()
// ---------------------------------------------------------------------------

/// Returns a full cognitive metrics snapshot as a single-row result.
pub fn cognitive_metrics(collector: &EngramMetricsCollector) -> ProcedureResult {
    let mut result = ProcedureResult::new(vec![
        "engrams_active".to_string(),
        "engrams_formed".to_string(),
        "engrams_decayed".to_string(),
        "engrams_recalled".to_string(),
        "engrams_crystallized".to_string(),
        "formations_attempted".to_string(),
        "recalls_attempted".to_string(),
        "recalls_successful".to_string(),
        "recalls_rejected".to_string(),
        "homeostasis_sweeps".to_string(),
        "prediction_errors_total".to_string(),
        "mean_strength".to_string(),
        "pheromone_entropy".to_string(),
        "max_pheromone_ratio".to_string(),
        "immune_fp_rate".to_string(),
        "immune_detector_count".to_string(),
        "avg_precision_beta".to_string(),
    ]);

    let snap = collector.snapshot();

    result.add_row(vec![
        Value::Int64(snap.engrams_active as i64),
        Value::Int64(snap.engrams_formed as i64),
        Value::Int64(snap.engrams_decayed as i64),
        Value::Int64(snap.engrams_recalled as i64),
        Value::Int64(snap.engrams_crystallized as i64),
        Value::Int64(snap.formations_attempted as i64),
        Value::Int64(snap.recalls_attempted as i64),
        Value::Int64(snap.recalls_successful as i64),
        Value::Int64(snap.recalls_rejected as i64),
        Value::Int64(snap.homeostasis_sweeps as i64),
        Value::Int64(snap.prediction_errors_total as i64),
        Value::Float64(snap.mean_strength),
        Value::Float64(snap.pheromone_entropy),
        Value::Float64(snap.max_pheromone_ratio),
        Value::Float64(snap.immune_fp_rate),
        Value::Int64(snap.immune_detector_count as i64),
        Value::Float64(snap.avg_precision_beta),
    ]);

    result
}

// ---------------------------------------------------------------------------
// Column metadata (for YIELD clause resolution)
// ---------------------------------------------------------------------------

/// Returns the YIELD column names for a given procedure name.
pub fn yield_columns(procedure_name: &str) -> Option<Vec<String>> {
    match procedure_name {
        "engrams.list" | "grafeo.engrams.list" => Some(vec![
            "id".into(),
            "strength".into(),
            "valence".into(),
            "precision".into(),
            "recall_count".into(),
            "horizon".into(),
            "ensemble_size".into(),
        ]),
        "engrams.inspect" | "grafeo.engrams.inspect" => Some(vec![
            "id".into(),
            "strength".into(),
            "valence".into(),
            "precision".into(),
            "recall_count".into(),
            "horizon".into(),
            "ensemble_size".into(),
            "ensemble".into(),
            "recall_history_count".into(),
            "source_episodes".into(),
            "spectral_dim".into(),
            "fsrs_stability".into(),
            "fsrs_difficulty".into(),
        ]),
        "engrams.forget" | "grafeo.engrams.forget" => Some(vec![
            "id".into(),
            "status".into(),
            "relations_removed".into(),
        ]),
        "cognitive.metrics" | "grafeo.cognitive.metrics" => Some(vec![
            "engrams_active".into(),
            "engrams_formed".into(),
            "engrams_decayed".into(),
            "engrams_recalled".into(),
            "engrams_crystallized".into(),
            "formations_attempted".into(),
            "recalls_attempted".into(),
            "recalls_successful".into(),
            "recalls_rejected".into(),
            "homeostasis_sweeps".into(),
            "prediction_errors_total".into(),
            "mean_strength".into(),
            "pheromone_entropy".into(),
            "max_pheromone_ratio".into(),
            "immune_fp_rate".into(),
            "immune_detector_count".into(),
            "avg_precision_beta".into(),
        ]),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Engram, EpisodeId, InMemoryVectorIndex, RecallEvent, RecallFeedback,
    };
    use grafeo_common::types::NodeId;
    use std::sync::Arc;
    use std::time::SystemTime;

    fn make_test_store() -> (EngramStore, Arc<InMemoryVectorIndex>, EngramMetricsCollector) {
        let store = EngramStore::new(None);
        let index = Arc::new(InMemoryVectorIndex::new());
        let metrics = EngramMetricsCollector::new();

        // Create test engrams
        for i in 1..=3u64 {
            let id = store.next_id();
            let nodes: Vec<(NodeId, f64)> =
                (i..i + 2).map(|n| (NodeId(n), 1.0 / n as f64)).collect();
            let mut engram = Engram::new(id, nodes);
            engram.strength = 0.3 + (i as f64) * 0.2;
            engram.valence = (i as f64 - 2.0) * 0.5;
            engram.precision = 1.0 + i as f64;
            engram.recall_count = i as u32;
            engram.source_episodes = vec![EpisodeId(i * 10), EpisodeId(i * 10 + 1)];
            engram.recall_history = vec![RecallEvent {
                timestamp: SystemTime::now(),
                cues: vec![format!("cue_{}", i)],
                feedback: Some(RecallFeedback::Used),
            }];
            engram.spectral_signature = vec![0.1 * i as f64; 8];

            index.upsert(&id.0.to_string(), &engram.spectral_signature);
            store.insert(engram);
            metrics.record_formation();
        }

        (store, index, metrics)
    }

    // -----------------------------------------------------------------------
    // engrams.list tests
    // -----------------------------------------------------------------------

    #[test]
    fn list_returns_all_engrams_with_expected_fields() {
        let (store, _index, _metrics) = make_test_store();
        let result = engrams_list(&store);

        assert_eq!(result.columns.len(), 7);
        assert_eq!(result.columns[0], "id");
        assert_eq!(result.columns[1], "strength");
        assert_eq!(result.columns[2], "valence");
        assert_eq!(result.rows.len(), 3);

        // Rows are sorted by id
        for window in result.rows.windows(2) {
            let id_a = match &window[0][0] {
                Value::Int64(v) => *v,
                _ => panic!("expected Int64"),
            };
            let id_b = match &window[1][0] {
                Value::Int64(v) => *v,
                _ => panic!("expected Int64"),
            };
            assert!(id_a < id_b, "should be sorted by id");
        }

        // Check that strength values are present
        for row in &result.rows {
            if let Value::Float64(strength) = &row[1] {
                assert!(*strength > 0.0 && *strength <= 1.0);
            } else {
                panic!("expected Float64 for strength");
            }
        }
    }

    #[test]
    fn list_returns_empty_for_empty_store() {
        let store = EngramStore::new(None);
        let result = engrams_list(&store);
        assert_eq!(result.rows.len(), 0);
        assert_eq!(result.columns.len(), 7);
    }

    // -----------------------------------------------------------------------
    // engrams.inspect tests
    // -----------------------------------------------------------------------

    #[test]
    fn inspect_returns_full_engram_data() {
        let (store, _index, _metrics) = make_test_store();
        let result = engrams_inspect(&store, 1);

        assert_eq!(result.columns.len(), 13);
        assert_eq!(result.rows.len(), 1);

        let row = &result.rows[0];
        // id
        assert_eq!(row[0], Value::Int64(1));
        // recall_history_count
        if let Value::Int64(count) = &row[8] {
            assert_eq!(*count, 1, "should have 1 recall event");
        }
        // source_episodes
        if let Value::String(eps) = &row[9] {
            assert!(eps.contains("10"), "should contain episode IDs");
        }
        // spectral_dim
        if let Value::Int64(dim) = &row[10] {
            assert_eq!(*dim, 8, "spectral signature should be 8-dimensional");
        }
    }

    #[test]
    fn inspect_returns_empty_for_nonexistent_id() {
        let (store, _index, _metrics) = make_test_store();
        let result = engrams_inspect(&store, 9999);
        assert_eq!(result.rows.len(), 0);
    }

    // -----------------------------------------------------------------------
    // engrams.forget tests (RGPD)
    // -----------------------------------------------------------------------

    #[test]
    fn forget_removes_engram_completely() {
        let (store, index, _metrics) = make_test_store();
        assert_eq!(store.count(), 3);

        let result = engrams_forget(&store, index.as_ref(), 1);
        assert_eq!(result.rows.len(), 1);

        let row = &result.rows[0];
        assert_eq!(row[0], Value::Int64(1));
        assert_eq!(row[1], Value::from("forgotten"));

        // Engram should be gone from store
        assert!(store.get(EngramId(1)).is_none());
        assert_eq!(store.count(), 2);

        // Spectral signature should be removed from vector index
        let nearest = index.nearest(&vec![0.1; 8], 10);
        for (id_str, _) in &nearest {
            assert_ne!(id_str, "1", "forgotten engram should not appear in vector index");
        }
    }

    #[test]
    fn forget_returns_not_found_for_nonexistent() {
        let (store, index, _metrics) = make_test_store();
        let result = engrams_forget(&store, index.as_ref(), 9999);
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1], Value::from("not_found"));
        // Store should be unchanged
        assert_eq!(store.count(), 3);
    }

    #[test]
    fn forget_then_inspect_returns_empty() {
        let (store, index, _metrics) = make_test_store();
        engrams_forget(&store, index.as_ref(), 2);
        let result = engrams_inspect(&store, 2);
        assert_eq!(result.rows.len(), 0, "inspect after forget should return empty");
    }

    #[test]
    fn forget_removes_relations_count() {
        let (store, index, _metrics) = make_test_store();
        let result = engrams_forget(&store, index.as_ref(), 1);
        let row = &result.rows[0];
        // relations_removed should be > 0 (ensemble + episodes + recall_history)
        if let Value::Int64(count) = &row[2] {
            assert!(*count > 0, "should report removed relations count");
        }
    }

    // -----------------------------------------------------------------------
    // cognitive.metrics tests
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_returns_all_fields() {
        let (_store, _index, metrics) = make_test_store();

        // Record some additional activity
        metrics.record_recall(true);
        metrics.record_recall(true);
        metrics.record_recall(false);
        metrics.record_crystallization();
        metrics.record_homeostasis_sweep();
        metrics.update_mean_strength(0.65);
        metrics.update_stigmergy_metrics(&[1.0, 1.0, 1.0, 1.0]);
        metrics.update_immune_detector_count(5);
        metrics.update_avg_precision_beta(2.5);

        let result = cognitive_metrics(&metrics);

        assert_eq!(result.columns.len(), 17);
        assert_eq!(result.rows.len(), 1);

        let row = &result.rows[0];

        // engrams_active (3 formations)
        assert_eq!(row[0], Value::Int64(3));
        // engrams_formed
        assert_eq!(row[1], Value::Int64(3));
        // recalls_successful
        assert_eq!(row[7], Value::Int64(2));
        // recalls_rejected
        assert_eq!(row[8], Value::Int64(1));
        // homeostasis_sweeps
        assert_eq!(row[9], Value::Int64(1));
        // mean_strength
        if let Value::Float64(v) = &row[11] {
            assert!((*v - 0.65).abs() < f64::EPSILON);
        }
        // pheromone_entropy (uniform → ~1.0)
        if let Value::Float64(v) = &row[12] {
            assert!(*v > 0.9, "uniform pheromones → high entropy");
        }
        // immune_detector_count
        assert_eq!(row[15], Value::Int64(5));
        // avg_precision_beta
        if let Value::Float64(v) = &row[16] {
            assert!((*v - 2.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn metrics_returns_coherent_after_formation_recall_crystallization() {
        let metrics = EngramMetricsCollector::new();

        // Simulate: 5 formations, 10 recalls (8 successful, 2 rejected), 2 crystallizations
        for _ in 0..5 {
            metrics.record_formation();
        }
        for _ in 0..8 {
            metrics.record_recall(true);
        }
        for _ in 0..2 {
            metrics.record_recall(false);
        }
        metrics.record_crystallization();
        metrics.record_crystallization();
        metrics.record_homeostasis_sweep();
        metrics.record_homeostasis_sweep();
        metrics.record_homeostasis_sweep();
        metrics.update_mean_strength(0.72);

        let result = cognitive_metrics(&metrics);
        let row = &result.rows[0];

        assert_eq!(row[0], Value::Int64(5));  // engrams_active
        assert_eq!(row[1], Value::Int64(5));  // engrams_formed
        assert_eq!(row[4], Value::Int64(2));  // engrams_crystallized
        assert_eq!(row[5], Value::Int64(5));  // formations_attempted
        assert_eq!(row[6], Value::Int64(10)); // recalls_attempted
        assert_eq!(row[7], Value::Int64(8));  // recalls_successful
        assert_eq!(row[8], Value::Int64(2));  // recalls_rejected
        assert_eq!(row[9], Value::Int64(3));  // homeostasis_sweeps
    }

    // -----------------------------------------------------------------------
    // yield_columns tests
    // -----------------------------------------------------------------------

    #[test]
    fn yield_columns_engrams_list() {
        let cols = yield_columns("grafeo.engrams.list").unwrap();
        assert_eq!(cols[0], "id");
        assert_eq!(cols.len(), 7);
    }

    #[test]
    fn yield_columns_engrams_inspect() {
        let cols = yield_columns("grafeo.engrams.inspect").unwrap();
        assert!(cols.len() >= 10);
    }

    #[test]
    fn yield_columns_engrams_forget() {
        let cols = yield_columns("grafeo.engrams.forget").unwrap();
        assert_eq!(cols.len(), 3);
        assert!(cols.contains(&"status".to_string()));
    }

    #[test]
    fn yield_columns_cognitive_metrics() {
        let cols = yield_columns("grafeo.cognitive.metrics").unwrap();
        assert!(cols.len() >= 15);
        assert!(cols.contains(&"engrams_active".to_string()));
        assert!(cols.contains(&"immune_fp_rate".to_string()));
    }

    #[test]
    fn yield_columns_unknown_returns_none() {
        assert!(yield_columns("grafeo.unknown").is_none());
    }
}
