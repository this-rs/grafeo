//! Cognitive UDF implementations — `obrain.energy()`, `obrain.risk()`, `obrain.synapses()`.
//!
//! These UDFs bridge the `obrain-cognitive` stores with the query engine's UDF
//! registry. They are registered when the `cognitive` feature flag is active.

use obrain_adapters::plugins::UserDefinedFunction;
use obrain_cognitive::CognitiveEngine;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_common::utils::error::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// obrain.energy(node) → Float64
// ---------------------------------------------------------------------------

/// UDF that returns the current energy of a node.
pub struct EnergyUdf {
    engine: Arc<dyn CognitiveEngine>,
}

impl EnergyUdf {
    /// Creates a new energy UDF backed by the given cognitive engine.
    pub fn new(engine: Arc<dyn CognitiveEngine>) -> Self {
        Self { engine }
    }
}

impl UserDefinedFunction for EnergyUdf {
    fn name(&self) -> &str {
        "obrain.energy"
    }

    fn description(&self) -> &str {
        "Returns the current energy (activation level) of a node"
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        let node_id = extract_node_id(args, "obrain.energy")?;
        #[cfg(feature = "cognitive")]
        {
            if let Some(store) = self.engine.energy_store() {
                return Ok(Value::Float64(store.get_energy(node_id)));
            }
        }
        Ok(Value::Float64(0.0))
    }
}

// ---------------------------------------------------------------------------
// obrain.risk(node) → Float64
// ---------------------------------------------------------------------------

/// UDF that returns the risk score of a node (from fabric metrics).
pub struct RiskUdf {
    engine: Arc<dyn CognitiveEngine>,
}

impl RiskUdf {
    /// Creates a new risk UDF backed by the given cognitive engine.
    pub fn new(engine: Arc<dyn CognitiveEngine>) -> Self {
        Self { engine }
    }
}

impl UserDefinedFunction for RiskUdf {
    fn name(&self) -> &str {
        "obrain.risk"
    }

    fn description(&self) -> &str {
        "Returns the risk score of a node (churn × pagerank × knowledge_gap × betweenness)"
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        let node_id = extract_node_id(args, "obrain.risk")?;
        #[cfg(feature = "cognitive-fabric")]
        {
            if let Some(store) = self.engine.fabric_store() {
                let score = store.get_fabric_score(node_id);
                return Ok(Value::Float64(score.risk_score));
            }
        }
        let _ = node_id;
        Ok(Value::Float64(0.0))
    }
}

// ---------------------------------------------------------------------------
// obrain.synapses(node) → List<Map>
// ---------------------------------------------------------------------------

/// UDF that returns the list of synapses connected to a node.
pub struct SynapsesUdf {
    engine: Arc<dyn CognitiveEngine>,
}

impl SynapsesUdf {
    /// Creates a new synapses UDF backed by the given cognitive engine.
    pub fn new(engine: Arc<dyn CognitiveEngine>) -> Self {
        Self { engine }
    }
}

impl UserDefinedFunction for SynapsesUdf {
    fn name(&self) -> &str {
        "obrain.synapses"
    }

    fn description(&self) -> &str {
        "Returns the list of Hebbian synapses connected to a node"
    }

    fn evaluate(&self, args: &[Value]) -> Result<Value> {
        let node_id = extract_node_id(args, "obrain.synapses")?;
        #[cfg(feature = "cognitive")]
        {
            if let Some(store) = self.engine.synapse_store() {
                let synapses = store.list_synapses(node_id);
                let list: Vec<Value> = synapses
                    .into_iter()
                    .map(|s| {
                        let mut map = BTreeMap::new();
                        map.insert(PropertyKey::from("source"), Value::Int64(s.source.0 as i64));
                        map.insert(PropertyKey::from("target"), Value::Int64(s.target.0 as i64));
                        map.insert(
                            PropertyKey::from("weight"),
                            Value::Float64(s.current_weight()),
                        );
                        Value::Map(Arc::new(map))
                    })
                    .collect();
                return Ok(Value::List(list.into()));
            }
        }
        let _ = node_id;
        Ok(Value::List(Arc::from(Vec::<Value>::new().as_slice())))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts a `NodeId` from the first argument.
fn extract_node_id(args: &[Value], fn_name: &str) -> Result<NodeId> {
    if args.is_empty() {
        return Err(Error::Internal(format!(
            "{fn_name}() requires exactly 1 argument (node ID)"
        )));
    }
    match &args[0] {
        Value::Int64(id) => Ok(NodeId(*id as u64)),
        Value::Null => Ok(NodeId(0)), // return default for null
        other => Err(Error::Internal(format!(
            "{fn_name}(): expected node ID (Int64), got {other:?}"
        ))),
    }
}

/// Registers all cognitive UDFs with the given plugin registry.
///
/// Called during database initialization when `cognitive` feature is enabled.
pub fn register_cognitive_udfs(
    registry: &obrain_adapters::plugins::PluginRegistry,
    engine: Arc<dyn CognitiveEngine>,
) {
    registry.register_udf(Arc::new(EnergyUdf::new(Arc::clone(&engine))));
    registry.register_udf(Arc::new(RiskUdf::new(Arc::clone(&engine))));
    registry.register_udf(Arc::new(SynapsesUdf::new(engine)));
    tracing::info!("cognitive: registered UDFs (obrain.energy, obrain.risk, obrain.synapses)");
}
