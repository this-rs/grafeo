//! # Triphasic Bootstrap
//!
//! Solves the chicken-and-egg problem: the kernel is in the graph,
//! but to load the graph you need the kernel.
//!
//! - **Cold start**: graph empty -> hardcoded defaults
//! - **Warm start**: kernel params in graph -> load and build config
//! - **Hot start**: kernel in memory -> delta propagation only

use crate::engram::traits::CognitiveStorage;
use crate::error::CognitiveResult;
use crate::kernel_params::{CognitiveKernelConfig, KernelParamStore};

// ---------------------------------------------------------------------------
// Bootstrap types
// ---------------------------------------------------------------------------

/// Bootstrap mode detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMode {
    /// Graph is empty, seed from hardcoded defaults.
    Cold,
    /// Kernel params found in graph, load them.
    Warm,
    /// Config already in memory, no loading needed.
    Hot,
}

/// Result of a bootstrap operation.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Which bootstrap mode was used.
    pub mode: BootstrapMode,
    /// The resolved kernel config.
    pub config: CognitiveKernelConfig,
    /// Number of params loaded from graph (warm start).
    pub params_loaded: usize,
    /// Number of params created (cold start).
    pub params_created: usize,
}

// ---------------------------------------------------------------------------
// Bootstrap function
// ---------------------------------------------------------------------------

/// Performs bootstrap: detect mode, load/seed params, build config.
///
/// # Arguments
///
/// * `storage` — the cognitive graph store
/// * `existing_config` — if `Some`, the config is already in memory (hot start)
///
/// # Returns
///
/// A [`BootstrapResult`] with the resolved config and diagnostic counters.
pub fn bootstrap(
    storage: &dyn CognitiveStorage,
    existing_config: Option<&CognitiveKernelConfig>,
) -> CognitiveResult<BootstrapResult> {
    // Hot start: config already in memory
    if let Some(config) = existing_config {
        return Ok(BootstrapResult {
            mode: BootstrapMode::Hot,
            config: config.clone(),
            params_loaded: 0,
            params_created: 0,
        });
    }

    // Try warm start: load from graph
    let existing = KernelParamStore::list_params(storage)?;
    if !existing.is_empty() {
        let config = KernelParamStore::build_config(storage)?;
        return Ok(BootstrapResult {
            mode: BootstrapMode::Warm,
            config,
            params_loaded: existing.len(),
            params_created: 0,
        });
    }

    // Cold start: seed defaults
    let params = KernelParamStore::seed_defaults(storage)?;
    let config = KernelParamStore::build_config(storage)?;
    Ok(BootstrapResult {
        mode: BootstrapMode::Cold,
        config,
        params_loaded: 0,
        params_created: params.len(),
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::traits::{CognitiveEdge, CognitiveFilter, CognitiveNode, CognitiveStorage};
    use obrain_common::types::{EdgeId, NodeId, Value};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    // -----------------------------------------------------------------------
    // In-memory mock CognitiveStorage (same pattern as kernel_params tests)
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
        fn create_node(
            &self,
            label: &str,
            properties: &HashMap<String, Value>,
        ) -> NodeId {
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

        fn query_nodes(
            &self,
            label: &str,
            filter: Option<&CognitiveFilter>,
        ) -> Vec<CognitiveNode> {
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
    fn cold_start_seeds_defaults() {
        let storage = MockStorage::new();
        let result = bootstrap(&storage, None).unwrap();

        assert_eq!(result.mode, BootstrapMode::Cold);
        assert_eq!(result.params_loaded, 0);
        assert!(result.params_created > 0);
        assert_eq!(result.config, CognitiveKernelConfig::default());
    }

    #[test]
    fn warm_start_loads_existing() {
        let storage = MockStorage::new();

        // Seed defaults first (simulates a previous cold start)
        KernelParamStore::seed_defaults(&storage).unwrap();

        // Now bootstrap should detect warm start
        let result = bootstrap(&storage, None).unwrap();

        assert_eq!(result.mode, BootstrapMode::Warm);
        assert!(result.params_loaded > 0);
        assert_eq!(result.params_created, 0);
    }

    #[test]
    fn hot_start_uses_existing_config() {
        let storage = MockStorage::new();
        let config = CognitiveKernelConfig {
            propagation_decay: 0.5,
            max_hops: 7,
            ..CognitiveKernelConfig::default()
        };

        let result = bootstrap(&storage, Some(&config)).unwrap();

        assert_eq!(result.mode, BootstrapMode::Hot);
        assert_eq!(result.params_loaded, 0);
        assert_eq!(result.params_created, 0);
        assert!((result.config.propagation_decay - 0.5).abs() < f64::EPSILON);
        assert_eq!(result.config.max_hops, 7);
    }

    #[test]
    fn cold_start_config_matches_defaults() {
        let storage = MockStorage::new();
        let result = bootstrap(&storage, None).unwrap();
        let defaults = CognitiveKernelConfig::default();

        assert!(
            (result.config.propagation_decay - defaults.propagation_decay).abs()
                < f64::EPSILON
        );
        assert_eq!(result.config.max_hops, defaults.max_hops);
        assert_eq!(
            result.config.context_budget_tokens,
            defaults.context_budget_tokens
        );
    }

    #[test]
    fn warm_start_preserves_modifications() {
        let storage = MockStorage::new();

        // Cold start
        KernelParamStore::seed_defaults(&storage).unwrap();

        // Modify a param
        KernelParamStore::set_param(&storage, "propagation_decay", 0.6, Some(1000))
            .unwrap();

        // Warm start should pick up the modification
        let result = bootstrap(&storage, None).unwrap();

        assert_eq!(result.mode, BootstrapMode::Warm);
        assert!(
            (result.config.propagation_decay - 0.6).abs() < f64::EPSILON,
            "warm start should preserve modified param"
        );
    }
}
