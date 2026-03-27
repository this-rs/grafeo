//! Vector accessor trait for reading vectors by node ID.
//!
//! This module provides the [`VectorAccessor`] trait, which decouples vector
//! storage from vector indexing. The HNSW index is topology-only (neighbor
//! lists only, no stored vectors) and reads vectors through this trait from
//! [`PropertyStorage`], the single source of truth, halving memory usage
//! for vector workloads.
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::VectorAccessor;
//! use grafeo_common::types::NodeId;
//! use std::sync::Arc;
//!
//! // Closure-based accessor for tests
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> {
//!     Some(vec![1.0, 2.0, 3.0].into())
//! };
//! assert!(accessor.get_vector(NodeId::new(1)).is_some());
//! ```

use std::sync::Arc;

use grafeo_common::types::{NodeId, PropertyKey, Value};

use crate::graph::GraphStore;

/// Trait for reading vectors by node ID.
///
/// HNSW is topology-only: vectors live in property storage, not in
/// HNSW nodes. This trait provides the bridge for reading them.
pub trait VectorAccessor: Send + Sync {
    /// Returns the vector associated with the given node ID, if it exists.
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>>;
}

/// Reads vectors from a graph store's property storage for a given property key.
///
/// This is the primary accessor used by the engine when performing vector
/// operations. It reads directly from the property store, avoiding any
/// duplication.
pub struct PropertyVectorAccessor<'a> {
    store: &'a dyn GraphStore,
    property: PropertyKey,
}

impl<'a> PropertyVectorAccessor<'a> {
    /// Creates a new accessor for the given store and property key.
    #[must_use]
    pub fn new(store: &'a dyn GraphStore, property: impl Into<PropertyKey>) -> Self {
        Self {
            store,
            property: property.into(),
        }
    }
}

impl VectorAccessor for PropertyVectorAccessor<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        match self.store.get_node_property(id, &self.property) {
            Some(Value::Vector(v)) => Some(v),
            _ => None,
        }
    }
}

/// Blanket implementation for closures, useful in tests.
impl<F> VectorAccessor for F
where
    F: Fn(NodeId) -> Option<Arc<[f32]>> + Send + Sync,
{
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_closure_accessor() {
        let vectors: std::collections::HashMap<NodeId, Arc<[f32]>> = [
            (NodeId::new(1), Arc::from(vec![1.0_f32, 0.0, 0.0])),
            (NodeId::new(2), Arc::from(vec![0.0_f32, 1.0, 0.0])),
        ]
        .into_iter()
        .collect();

        let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { vectors.get(&id).cloned() };

        assert!(accessor.get_vector(NodeId::new(1)).is_some());
        assert_eq!(accessor.get_vector(NodeId::new(1)).unwrap().len(), 3);
        assert!(accessor.get_vector(NodeId::new(3)).is_none());
    }

    #[test]
    fn test_property_vector_accessor() {
        let store = LpgStore::new().unwrap();
        let id = store.create_node(&["Test"]);
        let vec_data: Arc<[f32]> = vec![1.0, 2.0, 3.0].into();
        store.set_node_property(id, "embedding", Value::Vector(vec_data.clone()));

        let accessor = PropertyVectorAccessor::new(&store, "embedding");
        let result = accessor.get_vector(id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), vec_data.as_ref());

        // Non-existent node
        assert!(accessor.get_vector(NodeId::new(999)).is_none());

        // Wrong property type
        store.set_node_property(id, "name", Value::from("hello"));
        let name_accessor = PropertyVectorAccessor::new(&store, "name");
        assert!(name_accessor.get_vector(id).is_none());
    }
}
