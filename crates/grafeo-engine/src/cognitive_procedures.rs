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

/// Extracts an integer from a map expression argument: `{key: value}`.
fn extract_map_int(arg: Option<&LogicalExpression>, key: &str) -> Option<i64> {
    if let Some(LogicalExpression::Map(entries)) = arg {
        for (k, v) in entries {
            if k == key {
                if let LogicalExpression::Literal(Value::Int64(n)) = v {
                    return Some(*n);
                }
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
