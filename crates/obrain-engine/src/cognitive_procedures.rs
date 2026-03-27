//! Cognitive procedure implementations for `CALL grafeo.cognitive.*`.
//!
//! These procedures provide query-time access to the cognitive subsystems:
//! - `CALL grafeo.cognitive.spread(node, {hops: 3}) YIELD activated, energy`
//! - `CALL grafeo.cognitive.distill({min_weight: 0.1}) YIELD artifact`
//!
//! They are resolved in the planner when the `cognitive` feature is enabled.

use grafeo_adapters::plugins::AlgorithmResult;
use grafeo_cognitive::CognitiveEngine;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use std::sync::Arc;

use crate::query::plan::LogicalExpression;

/// Checks if a procedure name matches a cognitive procedure and executes it.
///
/// Returns `Some(AlgorithmResult)` if the name matched, `None` otherwise.
pub fn try_execute_cognitive_procedure(
    resolved_name: &str,
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<Option<AlgorithmResult>> {
    match resolved_name {
        "grafeo.cognitive.spread" => {
            let result = execute_spread(arguments, engine)?;
            Ok(Some(result))
        }
        "grafeo.cognitive.distill" => {
            let result = execute_distill(arguments, engine)?;
            Ok(Some(result))
        }
        "grafeo.cognitive.search" => {
            let result = execute_search(arguments, engine)?;
            Ok(Some(result))
        }
        // Level 2 — Engram introspection procedures
        "grafeo.engrams.list" | "engrams.list" => {
            let result = execute_engrams_list(engine)?;
            Ok(Some(result))
        }
        "grafeo.engrams.inspect" | "engrams.inspect" => {
            let result = execute_engrams_inspect(arguments, engine)?;
            Ok(Some(result))
        }
        "grafeo.engrams.forget" | "engrams.forget" => {
            let result = execute_engrams_forget(arguments, engine)?;
            Ok(Some(result))
        }
        "grafeo.cognitive.metrics" | "cognitive.metrics" => {
            let result = execute_cognitive_metrics(engine)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// `CALL grafeo.cognitive.spread(node_id, {hops: N}) YIELD activated, energy`
///
/// Performs spreading activation from a source node and returns all activated
/// nodes with their energy values.
fn execute_spread(
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<AlgorithmResult> {
    // Parse arguments: first is node ID, second is optional config map
    let source_node_id = match arguments.first() {
        Some(LogicalExpression::Literal(Value::Int64(id))) => *id as u64,
        Some(_) => {
            return Err(Error::Internal(
                "grafeo.cognitive.spread(): first argument must be a node ID (integer)".to_string(),
            ));
        }
        None => {
            return Err(Error::Internal(
                "grafeo.cognitive.spread(): requires at least 1 argument (node ID)".to_string(),
            ));
        }
    };

    let max_hops = extract_map_int(arguments.get(1), "hops").unwrap_or(3) as u32;

    let mut result = AlgorithmResult::new(vec!["activated".to_string(), "energy".to_string()]);

    // Perform spreading activation if synapse subsystem is available
    #[cfg(feature = "cognitive")]
    {
        use grafeo_cognitive::activation::{SpreadConfig, SynapseActivationSource, spread_single};
        use grafeo_common::types::NodeId;

        if let Some(synapse_store) = engine.synapse_store() {
            let config = SpreadConfig::default().with_max_hops(max_hops);
            let source = SynapseActivationSource::new(Arc::clone(synapse_store));
            let activation_map = spread_single(NodeId(source_node_id), 1.0, &source, &config);

            for (node_id, energy) in activation_map {
                result.add_row(vec![Value::Int64(node_id.0 as i64), Value::Float64(energy)]);
            }
        }
    }

    let _ = (source_node_id, max_hops, engine);

    Ok(result)
}

/// `CALL grafeo.cognitive.distill({min_weight: 0.1}) YIELD artifact`
///
/// Distills the knowledge graph by extracting high-weight synaptic connections
/// as knowledge artifacts. Each artifact is a map with source, target, weight.
fn execute_distill(
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<AlgorithmResult> {
    let min_weight = extract_map_float(arguments.first(), "min_weight").unwrap_or(0.1);

    let mut result = AlgorithmResult::new(vec!["artifact".to_string()]);

    #[cfg(feature = "cognitive")]
    {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        if let Some(synapse_store) = engine.synapse_store() {
            let all = synapse_store.snapshot();
            for synapse in all {
                if synapse.current_weight() >= min_weight {
                    let mut map = BTreeMap::new();
                    map.insert(
                        PropertyKey::from("source"),
                        Value::Int64(synapse.source.0 as i64),
                    );
                    map.insert(
                        PropertyKey::from("target"),
                        Value::Int64(synapse.target.0 as i64),
                    );
                    map.insert(
                        PropertyKey::from("weight"),
                        Value::Float64(synapse.current_weight()),
                    );
                    result.add_row(vec![Value::Map(Arc::new(map))]);
                }
            }
        }
    }

    let _ = (min_weight, engine);

    Ok(result)
}

/// `CALL grafeo.cognitive.search({query_embedding: [...], weights: {energy: 0.3, ...}, limit: 10})`
///
/// Performs a multi-signal search combining vector similarity, energy, topology,
/// and synapse traversal. Returns scored nodes with per-signal breakdown.
fn execute_search(
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<AlgorithmResult> {
    use grafeo_cognitive::search::{SearchConfig, SearchPipeline, SearchWeights};
    use grafeo_common::types::NodeId;

    // Parse the config map (first argument)
    let config_arg = arguments.first();

    // Extract query_embedding (required)
    let query_embedding =
        extract_map_float_list(config_arg, "query_embedding").ok_or_else(|| {
            Error::Internal(
                "grafeo.cognitive.search(): requires query_embedding (list of floats)".to_string(),
            )
        })?;

    if query_embedding.is_empty() {
        return Err(Error::Internal(
            "grafeo.cognitive.search(): query_embedding must not be empty".to_string(),
        ));
    }

    // Extract optional parameters
    let limit = extract_map_int(config_arg, "limit").unwrap_or(10) as usize;

    // Extract weights from nested map
    let w_similarity = extract_nested_map_float(config_arg, "weights", "similarity").unwrap_or(0.3);
    let w_energy = extract_nested_map_float(config_arg, "weights", "energy").unwrap_or(0.3);
    let w_topology = extract_nested_map_float(config_arg, "weights", "topology").unwrap_or(0.3);
    let w_synapse = extract_nested_map_float(config_arg, "weights", "synapse").unwrap_or(0.1);

    let search_config = SearchConfig {
        weights: SearchWeights::new(w_similarity, w_energy, w_topology, w_synapse),
        limit,
        ..Default::default()
    };

    // For now, simulate vector candidates from query_embedding
    // In a full integration, this would delegate to the HNSW index on GrafeoDB.
    // We create placeholder candidates — the actual vector search integration
    // happens at the GrafeoDB level when wiring the procedure.
    let vector_candidates: Vec<(NodeId, f64)> = Vec::new();

    // Build the pipeline from cognitive engine stores
    #[allow(unused_mut)]
    let mut pipeline = SearchPipeline::new();

    #[cfg(feature = "cognitive")]
    {
        if let Some(energy_store) = engine.energy_store() {
            pipeline = pipeline.with_energy_store(Arc::clone(energy_store));
        }
        if let Some(synapse_store) = engine.synapse_store() {
            pipeline = pipeline.with_synapse_store(Arc::clone(synapse_store));
        }
    }

    // Fabric store access requires the fabric feature on grafeo-cognitive
    // which is enabled via cognitive-fabric feature on grafeo-engine.
    // We use cfg to conditionally access it.
    #[cfg(feature = "cognitive-fabric")]
    {
        if let Some(fabric_store) = engine.fabric_store() {
            pipeline = pipeline.with_fabric_store(Arc::clone(fabric_store));
        }
    }

    let _ = (&query_embedding, engine);

    let results = pipeline.search(&vector_candidates, &search_config);

    let mut algo_result = AlgorithmResult::new(vec![
        "node_id".to_string(),
        "score".to_string(),
        "signal_similarity".to_string(),
        "signal_energy".to_string(),
        "signal_topology".to_string(),
        "signal_synapse".to_string(),
    ]);

    for r in results {
        algo_result.add_row(vec![
            Value::Int64(r.node_id.0 as i64),
            Value::Float64(r.score),
            Value::Float64(r.signal_similarity),
            Value::Float64(r.signal_energy),
            Value::Float64(r.signal_topology),
            Value::Float64(r.signal_synapse),
        ]);
    }

    Ok(algo_result)
}

// ============================================================================
// Level 2 — Engram introspection procedures
// ============================================================================

/// `CALL grafeo.engrams.list() YIELD id, strength, valence, precision, recall_count, horizon, ensemble_size`
///
/// Lists all engrams in the store with their key metrics.
fn execute_engrams_list(engine: &Arc<dyn CognitiveEngine>) -> Result<AlgorithmResult> {
    #[cfg(feature = "cognitive-engram")]
    {
        if let Some(store) = engine.engram_store() {
            let proc_result = grafeo_cognitive::procedures::engrams_list(store);
            return Ok(procedure_result_to_algorithm_result(proc_result));
        }
    }
    let _ = engine;

    // Engram subsystem not available — return empty result with correct columns
    Ok(AlgorithmResult::new(vec![
        "id".into(),
        "strength".into(),
        "valence".into(),
        "precision".into(),
        "recall_count".into(),
        "horizon".into(),
        "ensemble_size".into(),
    ]))
}

/// `CALL grafeo.engrams.inspect(id) YIELD ...`
///
/// Returns full detail for a single engram including recall history.
fn execute_engrams_inspect(
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<AlgorithmResult> {
    let engram_id = match arguments.first() {
        Some(LogicalExpression::Literal(Value::Int64(id))) => *id as u64,
        Some(_) => {
            return Err(Error::Internal(
                "grafeo.engrams.inspect(): argument must be an engram ID (integer)".to_string(),
            ));
        }
        None => {
            return Err(Error::Internal(
                "grafeo.engrams.inspect(): requires 1 argument (engram ID)".to_string(),
            ));
        }
    };

    #[cfg(feature = "cognitive-engram")]
    {
        if let Some(store) = engine.engram_store() {
            let proc_result = grafeo_cognitive::procedures::engrams_inspect(store, engram_id);
            return Ok(procedure_result_to_algorithm_result(proc_result));
        }
    }
    let _ = (engram_id, engine);

    Ok(AlgorithmResult::new(vec![
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
    ]))
}

/// `CALL grafeo.engrams.forget(id) YIELD id, status, relations_removed`
///
/// RGPD right to erasure: completely removes an engram and all associated data.
fn execute_engrams_forget(
    arguments: &[LogicalExpression],
    engine: &Arc<dyn CognitiveEngine>,
) -> Result<AlgorithmResult> {
    let engram_id = match arguments.first() {
        Some(LogicalExpression::Literal(Value::Int64(id))) => *id as u64,
        Some(_) => {
            return Err(Error::Internal(
                "grafeo.engrams.forget(): argument must be an engram ID (integer)".to_string(),
            ));
        }
        None => {
            return Err(Error::Internal(
                "grafeo.engrams.forget(): requires 1 argument (engram ID)".to_string(),
            ));
        }
    };

    #[cfg(feature = "cognitive-engram")]
    {
        if let (Some(store), Some(vi)) = (engine.engram_store(), engine.vector_index()) {
            let proc_result =
                grafeo_cognitive::procedures::engrams_forget(store, vi.as_ref(), engram_id);
            return Ok(procedure_result_to_algorithm_result(proc_result));
        }
    }
    let _ = (engram_id, engine);

    let mut result = AlgorithmResult::new(vec![
        "id".into(),
        "status".into(),
        "relations_removed".into(),
    ]);
    result.add_row(vec![
        Value::Int64(engram_id as i64),
        Value::from("not_available"),
        Value::Int64(0),
    ]);
    Ok(result)
}

/// `CALL grafeo.cognitive.metrics() YIELD engrams_active, engrams_formed, ...`
///
/// Returns full cognitive metrics snapshot.
fn execute_cognitive_metrics(engine: &Arc<dyn CognitiveEngine>) -> Result<AlgorithmResult> {
    #[cfg(feature = "cognitive-engram")]
    {
        if let Some(metrics) = engine.engram_metrics() {
            let proc_result = grafeo_cognitive::procedures::cognitive_metrics(metrics);
            return Ok(procedure_result_to_algorithm_result(proc_result));
        }
    }
    let _ = engine;

    Ok(AlgorithmResult::new(vec![
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
        "marks_evaluated".into(),
        "marks_applied".into(),
        "marks_suppressed".into(),
    ]))
}

/// Converts a `grafeo_cognitive::procedures::ProcedureResult` to an `AlgorithmResult`.
#[cfg(feature = "cognitive-engram")]
fn procedure_result_to_algorithm_result(
    proc_result: grafeo_cognitive::procedures::ProcedureResult,
) -> AlgorithmResult {
    let mut result = AlgorithmResult::new(proc_result.columns);
    for row in proc_result.rows {
        result.add_row(row);
    }
    result
}

/// Extracts an integer from a map expression argument: `{key: value}`.
fn extract_map_int(arg: Option<&LogicalExpression>, key: &str) -> Option<i64> {
    if let Some(LogicalExpression::Map(entries)) = arg {
        for (k, v) in entries {
            if k == key
                && let LogicalExpression::Literal(Value::Int64(n)) = v
            {
                return Some(*n);
            }
        }
    }
    None
}

/// Extracts a float from a map expression argument: `{key: value}`.
fn extract_map_float(arg: Option<&LogicalExpression>, key: &str) -> Option<f64> {
    if let Some(LogicalExpression::Map(entries)) = arg {
        for (k, v) in entries {
            if k == key {
                if let LogicalExpression::Literal(Value::Float64(f)) = v {
                    return Some(*f);
                }
                if let LogicalExpression::Literal(Value::Int64(n)) = v {
                    return Some(*n as f64);
                }
            }
        }
    }
    None
}

/// Extracts a list of floats from a map expression argument: `{key: [1.0, 2.0]}`.
fn extract_map_float_list(arg: Option<&LogicalExpression>, key: &str) -> Option<Vec<f64>> {
    if let Some(LogicalExpression::Map(entries)) = arg {
        for (k, v) in entries {
            if k == key
                && let LogicalExpression::List(items) = v
            {
                let mut floats = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        LogicalExpression::Literal(Value::Float64(f)) => floats.push(*f),
                        LogicalExpression::Literal(Value::Int64(n)) => {
                            floats.push(*n as f64);
                        }
                        _ => return None,
                    }
                }
                return Some(floats);
            }
        }
    }
    None
}

/// Extracts a float from a nested map: `{outer_key: {inner_key: value}}`.
fn extract_nested_map_float(
    arg: Option<&LogicalExpression>,
    outer_key: &str,
    inner_key: &str,
) -> Option<f64> {
    if let Some(LogicalExpression::Map(entries)) = arg {
        for (k, v) in entries {
            if k == outer_key
                && let LogicalExpression::Map(inner_entries) = v
            {
                for (ik, iv) in inner_entries {
                    if ik == inner_key {
                        if let LogicalExpression::Literal(Value::Float64(f)) = iv {
                            return Some(*f);
                        }
                        if let LogicalExpression::Literal(Value::Int64(n)) = iv {
                            return Some(*n as f64);
                        }
                    }
                }
            }
        }
    }
    None
}
