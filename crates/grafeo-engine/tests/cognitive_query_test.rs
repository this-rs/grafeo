//! Tests for cognitive query integration: UDFs and CALL procedures.

#![cfg(feature = "cognitive")]

use grafeo_cognitive::{CognitiveConfig, CognitiveEngine, CognitiveEngineBuilder};
use grafeo_common::types::{NodeId, Value};
use grafeo_engine::cognitive_procedures::try_execute_cognitive_procedure;
use grafeo_engine::cognitive_udfs::{register_cognitive_udfs, EnergyUdf, RiskUdf, SynapsesUdf};
use grafeo_engine::query::plan::LogicalExpression;

use grafeo_adapters::plugins::{PluginRegistry, UserDefinedFunction};
use std::sync::Arc;

/// Helper: create a cognitive engine (as dyn CognitiveEngine) with energy enabled.
/// Must be called within a tokio runtime (Scheduler requires it).
fn make_engine() -> Arc<dyn CognitiveEngine> {
    let bus = grafeo_reactive::MutationBus::new();
    let scheduler =
        grafeo_reactive::Scheduler::new(&bus, grafeo_reactive::BatchConfig::default());
    let mut config = CognitiveConfig::new();
    config.energy.enabled = true;
    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);
    Arc::new(engine) as Arc<dyn CognitiveEngine>
}

// ---------------------------------------------------------------------------
// UDF: grafeo.energy()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn energy_udf_returns_zero_for_unknown_node() {
    let engine = make_engine();
    let udf = EnergyUdf::new(Arc::clone(&engine));
    let result = udf.evaluate(&[Value::Int64(999)]).unwrap();
    match result {
        Value::Float64(v) => assert!(v >= 0.0),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

#[tokio::test]
async fn energy_udf_returns_boosted_energy() {
    let engine = make_engine();
    if let Some(store) = engine.energy_store() {
        store.boost(NodeId(42), 5.0);
    }
    let udf = EnergyUdf::new(engine);
    let result = udf.evaluate(&[Value::Int64(42)]).unwrap();
    match result {
        Value::Float64(v) => assert!(v >= 4.99, "Expected >= 4.99, got {}", v),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

#[tokio::test]
async fn energy_udf_rejects_missing_args() {
    let engine = make_engine();
    let udf = EnergyUdf::new(engine);
    let result = udf.evaluate(&[]);
    assert!(result.is_err());
}

#[tokio::test]
async fn energy_udf_name() {
    let engine = make_engine();
    let udf = EnergyUdf::new(engine);
    assert_eq!(udf.name(), "grafeo.energy");
}

// ---------------------------------------------------------------------------
// UDF: grafeo.risk()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn risk_udf_returns_zero_without_fabric() {
    let engine = make_engine();
    let udf = RiskUdf::new(engine);
    let result = udf.evaluate(&[Value::Int64(1)]).unwrap();
    match result {
        Value::Float64(v) => assert!((v - 0.0).abs() < f64::EPSILON),
        other => panic!("Expected Float64, got {:?}", other),
    }
}

#[tokio::test]
async fn risk_udf_name() {
    let engine = make_engine();
    let udf = RiskUdf::new(engine);
    assert_eq!(udf.name(), "grafeo.risk");
}

// ---------------------------------------------------------------------------
// UDF: grafeo.synapses()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn synapses_udf_returns_empty_list_without_synapse() {
    let engine = make_engine();
    let udf = SynapsesUdf::new(engine);
    let result = udf.evaluate(&[Value::Int64(1)]).unwrap();
    match result {
        Value::List(list) => assert!(list.is_empty()),
        other => panic!("Expected List, got {:?}", other),
    }
}

#[tokio::test]
async fn synapses_udf_name() {
    let engine = make_engine();
    let udf = SynapsesUdf::new(engine);
    assert_eq!(udf.name(), "grafeo.synapses");
}

// ---------------------------------------------------------------------------
// UDF registration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn register_cognitive_udfs_adds_three_udfs() {
    let engine = make_engine();
    let registry = PluginRegistry::new();
    register_cognitive_udfs(&registry, engine);

    assert!(registry.get_udf("grafeo.energy").is_some());
    assert!(registry.get_udf("grafeo.risk").is_some());
    assert!(registry.get_udf("grafeo.synapses").is_some());
    assert_eq!(registry.list_udfs().len(), 3);
}

// ---------------------------------------------------------------------------
// CALL procedures
// ---------------------------------------------------------------------------

#[tokio::test]
async fn spread_procedure_rejects_missing_args() {
    let engine = make_engine();
    let result = try_execute_cognitive_procedure("grafeo.cognitive.spread", &[], &engine);
    assert!(result.is_err());
}

#[tokio::test]
async fn spread_procedure_returns_result_for_valid_args() {
    let engine = make_engine();
    let args = vec![LogicalExpression::Literal(Value::Int64(1))];
    let result = try_execute_cognitive_procedure("grafeo.cognitive.spread", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.columns.len(), 2); // "activated", "energy"
}

#[tokio::test]
async fn distill_procedure_returns_result() {
    let engine = make_engine();
    let args = vec![]; // default min_weight = 0.1
    let result = try_execute_cognitive_procedure("grafeo.cognitive.distill", &args, &engine);
    assert!(result.is_ok());
    let algo_result = result.unwrap().unwrap();
    assert_eq!(algo_result.columns.len(), 1); // "artifact"
}

#[tokio::test]
async fn unknown_procedure_returns_none() {
    let engine = make_engine();
    let result =
        try_execute_cognitive_procedure("grafeo.cognitive.unknown", &[], &engine).unwrap();
    assert!(result.is_none());
}
