//! Write-through trait for cognitive stores backed by a graph store.
//!
//! Every cognitive subsystem (energy, synapse, scar, co-change, fabric)
//! uses a [`DashMap`](dashmap::DashMap) as a hot cache and persists scores
//! to the underlying [`GraphStoreMut`] as node/edge properties. On restart,
//! the cache rebuilds lazily from those properties.
//!
//! This module defines the [`CognitiveStore`] trait that captures the
//! write-through contract shared across all cognitive subsystems.

use obrain_common::types::{NodeId, Value};
use obrain_core::graph::GraphStoreMut;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Property key constants
// ---------------------------------------------------------------------------

/// Node property key for energy scores.
pub const PROP_ENERGY: &str = "_cog_energy";

/// Edge property key for synapse weights.
pub const PROP_SYNAPSE_WEIGHT: &str = "_cog_synapse_weight";

/// Node property key for scar count.
pub const PROP_SCAR_COUNT: &str = "_cog_scar_count";

/// Node property key for scar intensity.
pub const PROP_SCAR_INTENSITY: &str = "_cog_scar_intensity";

/// Edge property key for co-change count.
pub const PROP_CO_CHANGE_COUNT: &str = "_cog_co_change_count";

/// Node property key for fabric risk score.
pub const PROP_FABRIC_RISK: &str = "_cog_risk_score";

/// Node property key for fabric mutation frequency (formerly "churn score").
#[deprecated(note = "use PROP_FABRIC_MUTATION_FREQ instead")]
pub const PROP_FABRIC_CHURN: &str = "_cog_churn_score";

/// Node property key for fabric annotation density (formerly "knowledge density").
#[deprecated(note = "use PROP_FABRIC_ANNOTATION_DENSITY instead")]
pub const PROP_FABRIC_DENSITY: &str = "_cog_knowledge_density";

/// Node property key for fabric mutation frequency.
pub const PROP_FABRIC_MUTATION_FREQ: &str = "_cog_mutation_frequency";

/// Node property key for fabric annotation density.
pub const PROP_FABRIC_ANNOTATION_DENSITY: &str = "_cog_annotation_density";

// ---------------------------------------------------------------------------
// CognitiveStore trait
// ---------------------------------------------------------------------------

/// Write-through interface for cognitive score persistence.
///
/// Cognitive stores implement this trait to expose a uniform
/// `get_score` / `set_score` / `delete_score` surface.  The
/// underlying [`GraphStoreMut`] is used as the source of truth;
/// the DashMap serves as a hot cache.
///
/// # Generic parameter
///
/// `S: GraphStoreMut` — the backing graph store type. In practice,
/// the stores use `Arc<dyn GraphStoreMut>` for dynamic dispatch.
pub trait CognitiveStore<S: GraphStoreMut + ?Sized = dyn GraphStoreMut>: Send + Sync {
    /// Returns the cached score for `node_id`, or loads it lazily
    /// from the graph store if not present in the hot cache.
    fn get_score(&self, node_id: NodeId) -> Option<f64>;

    /// Sets the score for `node_id` in both the hot cache and the
    /// backing graph store (write-through).
    fn set_score(&self, node_id: NodeId, value: f64);

    /// Deletes the score for `node_id` from both the hot cache and
    /// the backing graph store.
    fn delete_score(&self, node_id: NodeId);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reads a `Float64` node property from the graph store, returning `None`
/// if the node or property doesn't exist.
pub fn load_node_f64(store: &dyn GraphStoreMut, node_id: NodeId, key: &str) -> Option<f64> {
    let pk = obrain_common::types::PropertyKey::from(key);
    store
        .get_node_property(node_id, &pk)
        .and_then(|v| v.as_float64())
}

/// Persists a `Float64` node property to the graph store.
pub fn persist_node_f64(store: &dyn GraphStoreMut, node_id: NodeId, key: &str, value: f64) {
    store.set_node_property(node_id, key, Value::Float64(value));
}

/// Reads a `Float64` edge property from the graph store.
pub fn load_edge_f64(
    store: &dyn GraphStoreMut,
    edge_id: obrain_common::types::EdgeId,
    key: &str,
) -> Option<f64> {
    let pk = obrain_common::types::PropertyKey::from(key);
    store
        .get_edge_property(edge_id, &pk)
        .and_then(|v| v.as_float64())
}

/// Persists a `Float64` edge property to the graph store.
pub fn persist_edge_f64(
    store: &dyn GraphStoreMut,
    edge_id: obrain_common::types::EdgeId,
    key: &str,
    value: f64,
) {
    store.set_edge_property(edge_id, key, Value::Float64(value));
}

/// Wraps an optional `Arc<dyn GraphStoreMut>` for stores that may or may not
/// have a backing graph. When `None`, write-through is silently skipped.
pub type OptionalGraphStore = Option<Arc<dyn GraphStoreMut>>;
