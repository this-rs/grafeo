//! Integration tests for Level 2 engram introspection procedures.
//!
//! Tests both Cypher-style and GQL-style CALL syntax (both use the same
//! dispatch path via `try_execute_cognitive_procedure`).

#![cfg(feature = "cognitive-engram")]

use grafeo_cognitive::{
    CognitiveConfig, CognitiveEngineBuilder, Engram, EngramMetricsCollector, EngramStore,
    InMemoryVectorIndex, VectorIndex,
};
use grafeo_common::types::{NodeId, Value};
use grafeo_engine::cognitive_procedures::try_execute_cognitive_procedure;
use grafeo_engine::query::plan::LogicalExpression;
use std::sync::Arc;

/// Helper: create a cognitive engine with the engram subsystem wired up.
fn make_engine_with_engrams() -> Arc<dyn grafeo_cognitive::CognitiveEngine> {
    let bus = grafeo_reactive::MutationBus::new();
    let scheduler = grafeo_reactive::Scheduler::new(&bus, grafeo_reactive::BatchConfig::default());
    let mut config = CognitiveConfig::new();
    config.energy.enabled = true;
    let mut engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

    // Wire up the engram subsystem
    let store = Arc::new(EngramStore::new(None));
    let metrics = Arc::new(EngramMetricsCollector::new());
    let vector_index = Arc::new(InMemoryVectorIndex::new());

    // Insert test engrams
    for i in 1..=3u64 {
        let id = store.next_id();
        let nodes: Vec<(NodeId, f64)> = (i..i + 2).map(|n| (NodeId(n), 1.0 / n as f64)).collect();
        let mut engram = Engram::new(id, nodes);
        engram.strength = 0.3 + (i as f64) * 0.2;
        engram.valence = (i as f64 - 2.0) * 0.5;
        engram.precision = 1.0 + i as f64;
        engram.recall_count = i as u32;
        engram.spectral_signature = vec![0.1 * i as f64; 8];
        vector_index.upsert(&id.0.to_string(), &engram.spectral_signature);
        store.insert(engram);
        metrics.record_formation();
    }

    // Record some recall activity for metrics
    metrics.record_recall(true);
    metrics.record_recall(true);
    metrics.record_recall(false);
    metrics.record_crystallization();
    metrics.update_mean_strength(0.65);

    engine.set_engram_subsystem(store, metrics, vector_index);
    Arc::new(engine)
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.list() — Cypher syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_list_returns_all_engrams_cypher() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("grafeo.engrams.list", &[], &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();

    assert_eq!(algo_result.columns[0], "id");
    assert_eq!(algo_result.columns[1], "strength");
    assert_eq!(algo_result.columns[2], "valence");
    assert_eq!(algo_result.columns.len(), 7);
    assert_eq!(algo_result.rows.len(), 3, "should list all 3 engrams");
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.list() — GQL syntax (same dispatch)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_list_returns_all_engrams_gql() {
    let engine = make_engine_with_engrams();
    // GQL uses the short name
    let result = try_execute_cognitive_procedure("engrams.list", &[], &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.rows.len(), 3);
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.inspect(id) — Cypher syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_inspect_returns_full_detail_cypher() {
    let engine = make_engine_with_engrams();
    let args = vec![LogicalExpression::Literal(Value::Int64(1))];
    let result = try_execute_cognitive_procedure("grafeo.engrams.inspect", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();

    assert_eq!(algo_result.columns.len(), 13);
    assert_eq!(algo_result.rows.len(), 1);
    assert_eq!(algo_result.rows[0][0], Value::Int64(1)); // id
}

#[tokio::test]
async fn engrams_inspect_returns_empty_for_nonexistent() {
    let engine = make_engine_with_engrams();
    let args = vec![LogicalExpression::Literal(Value::Int64(9999))];
    let result = try_execute_cognitive_procedure("grafeo.engrams.inspect", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.rows.len(), 0);
}

#[tokio::test]
async fn engrams_inspect_rejects_missing_args() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("grafeo.engrams.inspect", &[], &engine);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.inspect(id) — GQL syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_inspect_gql() {
    let engine = make_engine_with_engrams();
    let args = vec![LogicalExpression::Literal(Value::Int64(2))];
    let result = try_execute_cognitive_procedure("engrams.inspect", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.rows.len(), 1);
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.forget(id) — Cypher syntax (RGPD)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_forget_removes_engram_cypher() {
    let engine = make_engine_with_engrams();

    // Verify the engram exists first
    let args_inspect = vec![LogicalExpression::Literal(Value::Int64(1))];
    let before = try_execute_cognitive_procedure("grafeo.engrams.inspect", &args_inspect, &engine)
        .unwrap()
        .unwrap();
    assert_eq!(before.rows.len(), 1, "engram 1 should exist before forget");

    // Forget it
    let args = vec![LogicalExpression::Literal(Value::Int64(1))];
    let result = try_execute_cognitive_procedure("grafeo.engrams.forget", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();

    assert_eq!(
        algo_result.columns,
        vec!["id", "status", "relations_removed"]
    );
    assert_eq!(algo_result.rows.len(), 1);
    assert_eq!(algo_result.rows[0][1], Value::from("forgotten"));

    // Verify it's gone
    let after = try_execute_cognitive_procedure("grafeo.engrams.inspect", &args_inspect, &engine)
        .unwrap()
        .unwrap();
    assert_eq!(
        after.rows.len(),
        0,
        "engram 1 should not exist after forget"
    );

    // List should now have 2 engrams
    let list = try_execute_cognitive_procedure("grafeo.engrams.list", &[], &engine)
        .unwrap()
        .unwrap();
    assert_eq!(list.rows.len(), 2);
}

#[tokio::test]
async fn engrams_forget_nonexistent_returns_not_found() {
    let engine = make_engine_with_engrams();
    let args = vec![LogicalExpression::Literal(Value::Int64(9999))];
    let result = try_execute_cognitive_procedure("grafeo.engrams.forget", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.rows[0][1], Value::from("not_found"));
}

#[tokio::test]
async fn engrams_forget_rejects_missing_args() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("grafeo.engrams.forget", &[], &engine);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// CALL grafeo.engrams.forget(id) — GQL syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engrams_forget_gql() {
    let engine = make_engine_with_engrams();
    let args = vec![LogicalExpression::Literal(Value::Int64(2))];
    let result = try_execute_cognitive_procedure("engrams.forget", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.rows[0][1], Value::from("forgotten"));
}

// ---------------------------------------------------------------------------
// CALL grafeo.cognitive.metrics() — Cypher syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cognitive_metrics_returns_all_fields_cypher() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("grafeo.cognitive.metrics", &[], &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();

    assert_eq!(algo_result.columns.len(), 20);
    assert_eq!(algo_result.rows.len(), 1);

    // Check specific values
    let row = &algo_result.rows[0];
    assert_eq!(row[0], Value::Int64(3)); // engrams_active
    assert_eq!(row[1], Value::Int64(3)); // engrams_formed
    assert_eq!(row[7], Value::Int64(2)); // recalls_successful
    assert_eq!(row[8], Value::Int64(1)); // recalls_rejected
    assert_eq!(row[4], Value::Int64(1)); // engrams_crystallized
}

// ---------------------------------------------------------------------------
// CALL grafeo.cognitive.metrics() — GQL syntax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cognitive_metrics_gql() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("cognitive.metrics", &[], &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.columns.len(), 20);
    assert_eq!(algo_result.rows.len(), 1);
}

// ---------------------------------------------------------------------------
// Unknown procedure returns None
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_procedure_returns_none() {
    let engine = make_engine_with_engrams();
    let result = try_execute_cognitive_procedure("grafeo.engrams.unknown", &[], &engine).unwrap();
    assert!(result.is_none());
}
