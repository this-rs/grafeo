//! # Kernel Parameters
//!
//! Self-referential cognitive parameters stored as graph nodes.
//! Each parameter is a `:KernelParam` node in the graph, participating
//! in the same energy/synapse cycle as any other node.
//!
//! ## Default Parameters
//!
//! | Name | Default | Range | Description |
//! |------|---------|-------|-------------|
//! | propagation_decay | 0.3 | [0.05, 0.9] | Decay per hop in spreading activation |
//! | community_cohesion_threshold | 0.7 | [0.1, 1.0] | Accept boundary for propagation |
//! | max_hops | 3.0 | [1.0, 10.0] | Max propagation depth |
//! | min_propagated_energy | 0.1 | [0.01, 0.5] | Energy cutoff threshold |
//! | cristallization_sessions | 5.0 | [2.0, 20.0] | Sessions before crystallization |
//! | cristallization_energy | 0.8 | [0.3, 1.0] | Min energy for crystallization |
//! | dissolution_hit_rate | 0.05 | [0.01, 0.3] | Hit rate below which skills dissolve |
//! | context_budget_tokens | 750.0 | [100.0, 4000.0] | Max context injection budget |
//! | kernel_learning_rate | 0.1 | [0.001, 0.5] | Meta-learning rate for parameter adjustment |

use std::collections::HashMap;

use obrain_common::types::{NodeId, Value};

use crate::engram::traits::{CognitiveFilter, CognitiveNode, CognitiveStorage};
use crate::error::{CognitiveError, CognitiveResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label used for KernelParam nodes in the graph.
pub const LABEL_KERNEL_PARAM: &str = "KernelParam";

// Property keys for KernelParam nodes.
const PROP_NAME: &str = "kernel_name";
const PROP_VALUE: &str = "kernel_value";
const PROP_MIN: &str = "kernel_min";
const PROP_MAX: &str = "kernel_max";
const PROP_LR: &str = "kernel_lr";
const PROP_LAST_ADJUSTED: &str = "kernel_last_adjusted";
const PROP_ADJUSTMENT_COUNT: &str = "kernel_adjustment_count";

// ---------------------------------------------------------------------------
// KernelParam — runtime representation
// ---------------------------------------------------------------------------

/// A cognitive kernel parameter stored as a graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelParam {
    /// The node id in the graph (set after persistence).
    pub node_id: Option<NodeId>,
    /// Parameter name (unique identifier).
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Minimum allowed value.
    pub min_value: f64,
    /// Maximum allowed value.
    pub max_value: f64,
    /// Learning rate for parameter adjustment.
    pub learning_rate: f64,
    /// Timestamp of last adjustment (epoch millis), if any.
    pub last_adjusted: Option<u64>,
    /// Number of times this parameter has been adjusted.
    pub adjustment_count: u64,
}

impl KernelParam {
    /// Clamps `value` to `[min_value, max_value]`.
    fn clamped_value(&self, value: f64) -> f64 {
        value.clamp(self.min_value, self.max_value)
    }

    /// Converts this param into a property map for storage.
    fn to_properties(&self) -> HashMap<String, Value> {
        let mut props = HashMap::new();
        props.insert(PROP_NAME.to_string(), Value::from(self.name.as_str()));
        props.insert(PROP_VALUE.to_string(), Value::Float64(self.value));
        props.insert(PROP_MIN.to_string(), Value::Float64(self.min_value));
        props.insert(PROP_MAX.to_string(), Value::Float64(self.max_value));
        props.insert(PROP_LR.to_string(), Value::Float64(self.learning_rate));
        props.insert(
            PROP_LAST_ADJUSTED.to_string(),
            match self.last_adjusted {
                Some(ts) => Value::Int64(ts as i64),
                None => Value::Null,
            },
        );
        props.insert(
            PROP_ADJUSTMENT_COUNT.to_string(),
            Value::Int64(self.adjustment_count as i64),
        );
        props
    }

    /// Reconstructs a `KernelParam` from a `CognitiveNode`.
    fn from_cognitive_node(node: &CognitiveNode) -> Option<Self> {
        let name = node
            .properties
            .get(PROP_NAME)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())?;
        let value = node
            .properties
            .get(PROP_VALUE)
            .and_then(|v| v.as_float64())?;
        let min_value = node.properties.get(PROP_MIN).and_then(|v| v.as_float64())?;
        let max_value = node.properties.get(PROP_MAX).and_then(|v| v.as_float64())?;
        let learning_rate = node.properties.get(PROP_LR).and_then(|v| v.as_float64())?;
        let last_adjusted = node
            .properties
            .get(PROP_LAST_ADJUSTED)
            .and_then(|v| v.as_int64())
            .map(|ts| ts as u64);
        let adjustment_count = node
            .properties
            .get(PROP_ADJUSTMENT_COUNT)
            .and_then(|v| v.as_int64())
            .unwrap_or(0) as u64;

        Some(Self {
            node_id: Some(node.id),
            name,
            value,
            min_value,
            max_value,
            learning_rate,
            last_adjusted,
            adjustment_count,
        })
    }
}

// ---------------------------------------------------------------------------
// KernelParamDef — static default definitions
// ---------------------------------------------------------------------------

/// Definition of a default kernel parameter.
#[derive(Debug, Clone)]
pub struct KernelParamDef {
    /// Parameter name.
    pub name: &'static str,
    /// Default value.
    pub default_value: f64,
    /// Minimum allowed value.
    pub min_value: f64,
    /// Maximum allowed value.
    pub max_value: f64,
    /// Default learning rate.
    pub default_learning_rate: f64,
    /// Human-readable description.
    pub description: &'static str,
}

/// The 9 default kernel parameters.
pub const DEFAULT_PARAMS: &[KernelParamDef] = &[
    KernelParamDef {
        name: "propagation_decay",
        default_value: 0.3,
        min_value: 0.05,
        max_value: 0.9,
        default_learning_rate: 0.1,
        description: "Decay per hop in spreading activation",
    },
    KernelParamDef {
        name: "community_cohesion_threshold",
        default_value: 0.7,
        min_value: 0.1,
        max_value: 1.0,
        default_learning_rate: 0.1,
        description: "Accept boundary for propagation",
    },
    KernelParamDef {
        name: "max_hops",
        default_value: 3.0,
        min_value: 1.0,
        max_value: 10.0,
        default_learning_rate: 0.1,
        description: "Max propagation depth",
    },
    KernelParamDef {
        name: "min_propagated_energy",
        default_value: 0.1,
        min_value: 0.01,
        max_value: 0.5,
        default_learning_rate: 0.1,
        description: "Energy cutoff threshold",
    },
    KernelParamDef {
        name: "cristallization_sessions",
        default_value: 5.0,
        min_value: 2.0,
        max_value: 20.0,
        default_learning_rate: 0.1,
        description: "Sessions before crystallization",
    },
    KernelParamDef {
        name: "cristallization_energy",
        default_value: 0.8,
        min_value: 0.3,
        max_value: 1.0,
        default_learning_rate: 0.1,
        description: "Min energy for crystallization",
    },
    KernelParamDef {
        name: "dissolution_hit_rate",
        default_value: 0.05,
        min_value: 0.01,
        max_value: 0.3,
        default_learning_rate: 0.1,
        description: "Hit rate below which skills dissolve",
    },
    KernelParamDef {
        name: "context_budget_tokens",
        default_value: 750.0,
        min_value: 100.0,
        max_value: 4000.0,
        default_learning_rate: 0.1,
        description: "Max context injection budget",
    },
    KernelParamDef {
        name: "kernel_learning_rate",
        default_value: 0.1,
        min_value: 0.001,
        max_value: 0.5,
        default_learning_rate: 0.1,
        description: "Meta-learning rate for parameter adjustment",
    },
];

// ---------------------------------------------------------------------------
// CognitiveKernelConfig — runtime config built from graph params
// ---------------------------------------------------------------------------

/// Runtime configuration reconstructed from `:KernelParam` graph nodes.
///
/// This struct provides typed access to cognitive kernel parameters
/// for use by the cognitive engine at runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveKernelConfig {
    /// Decay per hop in spreading activation.
    pub propagation_decay: f64,
    /// Accept boundary for propagation.
    pub community_cohesion_threshold: f64,
    /// Max propagation depth.
    pub max_hops: u32,
    /// Energy cutoff threshold.
    pub min_propagated_energy: f64,
    /// Sessions before crystallization.
    pub cristallization_sessions: u32,
    /// Min energy for crystallization.
    pub cristallization_energy: f64,
    /// Hit rate below which skills dissolve.
    pub dissolution_hit_rate: f64,
    /// Max context injection budget.
    pub context_budget_tokens: u32,
    /// Meta-learning rate for parameter adjustment.
    pub kernel_learning_rate: f64,
}

impl Default for CognitiveKernelConfig {
    fn default() -> Self {
        Self {
            propagation_decay: 0.3,
            community_cohesion_threshold: 0.7,
            max_hops: 3,
            min_propagated_energy: 0.1,
            cristallization_sessions: 5,
            cristallization_energy: 0.8,
            dissolution_hit_rate: 0.05,
            context_budget_tokens: 750,
            kernel_learning_rate: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// KernelParamStore — CRUD operations via CognitiveStorage
// ---------------------------------------------------------------------------

/// Store for managing `:KernelParam` nodes via `CognitiveStorage`.
pub struct KernelParamStore;

impl KernelParamStore {
    /// Seeds the default kernel parameters if they do not already exist.
    ///
    /// For each entry in [`DEFAULT_PARAMS`], checks whether a node with
    /// matching `kernel_name` already exists. If not, creates it.
    /// Returns all kernel parameters (existing + newly created).
    pub fn seed_defaults(storage: &dyn CognitiveStorage) -> CognitiveResult<Vec<KernelParam>> {
        let existing = Self::list_params(storage)?;

        for def in DEFAULT_PARAMS {
            let already_exists = existing.iter().any(|p| p.name == def.name);
            if !already_exists {
                let param = KernelParam {
                    node_id: None,
                    name: def.name.to_string(),
                    value: def.default_value,
                    min_value: def.min_value,
                    max_value: def.max_value,
                    learning_rate: def.default_learning_rate,
                    last_adjusted: None,
                    adjustment_count: 0,
                };
                storage.create_node(LABEL_KERNEL_PARAM, &param.to_properties());
            }
        }

        // Return the full list (including newly created)
        Self::list_params(storage)
    }

    /// Retrieves a kernel parameter by name.
    pub fn get_param(
        storage: &dyn CognitiveStorage,
        name: &str,
    ) -> CognitiveResult<Option<KernelParam>> {
        let filter = CognitiveFilter::PropertyEquals(PROP_NAME.to_string(), Value::from(name));
        let nodes = storage.query_nodes(LABEL_KERNEL_PARAM, Some(&filter));

        Ok(nodes.first().and_then(KernelParam::from_cognitive_node))
    }

    /// Updates the value of a kernel parameter by name.
    ///
    /// The value is clamped to `[min_value, max_value]`. The `adjustment_count`
    /// is incremented and `last_adjusted` is set to the provided timestamp
    /// (or the current time if `None`).
    pub fn set_param(
        storage: &dyn CognitiveStorage,
        name: &str,
        value: f64,
        now_millis: Option<u64>,
    ) -> CognitiveResult<()> {
        let filter = CognitiveFilter::PropertyEquals(PROP_NAME.to_string(), Value::from(name));
        let nodes = storage.query_nodes(LABEL_KERNEL_PARAM, Some(&filter));

        let node = nodes
            .first()
            .ok_or_else(|| CognitiveError::Store(format!("kernel param not found: {name}")))?;

        let mut param = KernelParam::from_cognitive_node(node)
            .ok_or_else(|| CognitiveError::Store(format!("invalid kernel param node: {name}")))?;

        param.value = param.clamped_value(value);
        param.adjustment_count += 1;
        param.last_adjusted = Some(now_millis.unwrap_or(0));

        storage.update_node(node.id, &param.to_properties());

        Ok(())
    }

    /// Lists all kernel parameters.
    pub fn list_params(storage: &dyn CognitiveStorage) -> CognitiveResult<Vec<KernelParam>> {
        let nodes = storage.query_nodes(LABEL_KERNEL_PARAM, None);
        Ok(nodes
            .iter()
            .filter_map(KernelParam::from_cognitive_node)
            .collect())
    }

    /// Reconstructs a [`CognitiveKernelConfig`] from the graph parameters.
    ///
    /// Missing parameters fall back to their defaults.
    pub fn build_config(storage: &dyn CognitiveStorage) -> CognitiveResult<CognitiveKernelConfig> {
        let params = Self::list_params(storage)?;
        let defaults = CognitiveKernelConfig::default();

        let get = |name: &str, default: f64| -> f64 {
            params
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.value)
                .unwrap_or(default)
        };

        Ok(CognitiveKernelConfig {
            propagation_decay: get("propagation_decay", defaults.propagation_decay),
            community_cohesion_threshold: get(
                "community_cohesion_threshold",
                defaults.community_cohesion_threshold,
            ),
            max_hops: get("max_hops", defaults.max_hops as f64) as u32,
            min_propagated_energy: get("min_propagated_energy", defaults.min_propagated_energy),
            cristallization_sessions: get(
                "cristallization_sessions",
                defaults.cristallization_sessions as f64,
            ) as u32,
            cristallization_energy: get("cristallization_energy", defaults.cristallization_energy),
            dissolution_hit_rate: get("dissolution_hit_rate", defaults.dissolution_hit_rate),
            context_budget_tokens: get(
                "context_budget_tokens",
                defaults.context_budget_tokens as f64,
            ) as u32,
            kernel_learning_rate: get("kernel_learning_rate", defaults.kernel_learning_rate),
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::traits::{CognitiveEdge, CognitiveNode, CognitiveStorage};
    use obrain_common::types::{EdgeId, NodeId};
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    // -----------------------------------------------------------------------
    // In-memory mock CognitiveStorage
    // -----------------------------------------------------------------------

    struct MockStorage {
        nodes: Mutex<Vec<CognitiveNode>>,
        next_id: AtomicU64,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(Vec::new()),
                next_id: AtomicU64::new(1),
            }
        }
    }

    impl CognitiveStorage for MockStorage {
        fn create_node(&self, label: &str, properties: &HashMap<String, Value>) -> NodeId {
            let id = NodeId::from(self.next_id.fetch_add(1, Ordering::Relaxed));
            let node = CognitiveNode {
                id,
                label: label.to_string(),
                properties: properties.clone(),
            };
            self.nodes.lock().unwrap().push(node);
            id
        }

        fn create_edge(
            &self,
            _from: NodeId,
            _to: NodeId,
            _rel_type: &str,
            _properties: &HashMap<String, Value>,
        ) -> EdgeId {
            EdgeId::from(0_u64)
        }

        fn query_nodes(&self, label: &str, filter: Option<&CognitiveFilter>) -> Vec<CognitiveNode> {
            let nodes = self.nodes.lock().unwrap();
            nodes
                .iter()
                .filter(|n| n.label == label)
                .filter(|n| match filter {
                    None => true,
                    Some(CognitiveFilter::PropertyEquals(key, val)) => {
                        n.properties.get(key).map_or(false, |v| match (v, val) {
                            (Value::String(a), Value::String(b)) => a == b,
                            (Value::Float64(a), Value::Float64(b)) => a == b,
                            (Value::Int64(a), Value::Int64(b)) => a == b,
                            _ => false,
                        })
                    }
                    _ => true,
                })
                .cloned()
                .collect()
        }

        fn update_node(&self, id: NodeId, properties: &HashMap<String, Value>) {
            let mut nodes = self.nodes.lock().unwrap();
            if let Some(node) = nodes.iter_mut().find(|n| n.id == id) {
                for (k, v) in properties {
                    node.properties.insert(k.clone(), v.clone());
                }
            }
        }

        fn delete_node(&self, id: NodeId) {
            let mut nodes = self.nodes.lock().unwrap();
            nodes.retain(|n| n.id != id);
        }

        fn delete_edge(&self, _id: EdgeId) {}

        fn query_edges(&self, _from: NodeId, _rel_type: &str) -> Vec<CognitiveEdge> {
            Vec::new()
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn seed_defaults_creates_all_params() {
        let storage = MockStorage::new();
        let params = KernelParamStore::seed_defaults(&storage).unwrap();

        assert_eq!(params.len(), DEFAULT_PARAMS.len());
        for def in DEFAULT_PARAMS {
            let param = params.iter().find(|p| p.name == def.name);
            assert!(param.is_some(), "missing default param: {}", def.name);
            let param = param.unwrap();
            assert!(
                (param.value - def.default_value).abs() < f64::EPSILON,
                "wrong default for {}",
                def.name
            );
            assert!(
                (param.min_value - def.min_value).abs() < f64::EPSILON,
                "wrong min for {}",
                def.name
            );
            assert!(
                (param.max_value - def.max_value).abs() < f64::EPSILON,
                "wrong max for {}",
                def.name
            );
        }
    }

    #[test]
    fn seed_defaults_is_idempotent() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();
        let params = KernelParamStore::seed_defaults(&storage).unwrap();
        // Should not duplicate
        assert_eq!(params.len(), DEFAULT_PARAMS.len());
    }

    #[test]
    fn get_set_param_roundtrip() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();

        // Get initial value
        let param = KernelParamStore::get_param(&storage, "propagation_decay")
            .unwrap()
            .unwrap();
        assert!((param.value - 0.3).abs() < f64::EPSILON);
        assert_eq!(param.adjustment_count, 0);

        // Set new value
        KernelParamStore::set_param(&storage, "propagation_decay", 0.5, Some(12345)).unwrap();

        let param = KernelParamStore::get_param(&storage, "propagation_decay")
            .unwrap()
            .unwrap();
        assert!((param.value - 0.5).abs() < f64::EPSILON);
        assert_eq!(param.adjustment_count, 1);
        assert_eq!(param.last_adjusted, Some(12345));
    }

    #[test]
    fn set_param_clamps_value() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();

        // Set value above max (propagation_decay max = 0.9)
        KernelParamStore::set_param(&storage, "propagation_decay", 99.0, Some(1)).unwrap();
        let param = KernelParamStore::get_param(&storage, "propagation_decay")
            .unwrap()
            .unwrap();
        assert!((param.value - 0.9).abs() < f64::EPSILON);

        // Set value below min (propagation_decay min = 0.05)
        KernelParamStore::set_param(&storage, "propagation_decay", 0.001, Some(2)).unwrap();
        let param = KernelParamStore::get_param(&storage, "propagation_decay")
            .unwrap()
            .unwrap();
        assert!((param.value - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn adjustment_count_increments() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();

        for i in 0..5 {
            KernelParamStore::set_param(&storage, "max_hops", 4.0, Some(i)).unwrap();
        }

        let param = KernelParamStore::get_param(&storage, "max_hops")
            .unwrap()
            .unwrap();
        assert_eq!(param.adjustment_count, 5);
    }

    #[test]
    fn build_config_uses_graph_values() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();

        // Override some values
        KernelParamStore::set_param(&storage, "propagation_decay", 0.5, Some(1)).unwrap();
        KernelParamStore::set_param(&storage, "max_hops", 7.0, Some(1)).unwrap();
        KernelParamStore::set_param(&storage, "context_budget_tokens", 2000.0, Some(1)).unwrap();

        let config = KernelParamStore::build_config(&storage).unwrap();

        assert!((config.propagation_decay - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.max_hops, 7);
        assert_eq!(config.context_budget_tokens, 2000);
        // Unchanged defaults
        assert!((config.community_cohesion_threshold - 0.7).abs() < f64::EPSILON);
        assert!((config.cristallization_energy - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn build_config_defaults_when_empty() {
        let storage = MockStorage::new();
        // Don't seed — should fall back to defaults
        let config = KernelParamStore::build_config(&storage).unwrap();
        assert_eq!(config, CognitiveKernelConfig::default());
    }

    #[test]
    fn set_nonexistent_param_returns_error() {
        let storage = MockStorage::new();
        KernelParamStore::seed_defaults(&storage).unwrap();

        let result = KernelParamStore::set_param(&storage, "nonexistent", 1.0, Some(1));
        assert!(result.is_err());
    }
}
