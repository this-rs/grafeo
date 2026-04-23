//! Integration tests for `obrain_core::index::vector::accessor::*`
//! (`VectorAccessor` trait, `PropertyVectorAccessor`, and the `Fn` blanket impl)
//! against the substrate backend.
//!
//! Relocated from `crates/obrain-core/src/index/vector/accessor.rs`'s in-crate
//! `#[cfg(test)] mod tests` block as part of T17 W4.p4. Same rationale as the
//! `operators_*` migration (see note tagged `t17 w4.p4 migration-pattern`):
//! a substrate-backed fixture cannot live inside `obrain-core` because the
//! dev-dep cycle produces two distinct compilation units of `obrain-core`,
//! breaking the `Arc<SubstrateStore> as Arc<dyn GraphStore>` trait cast.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test vector_accessor
//! ```

use std::sync::Arc;

use obrain_common::types::{NodeId, Value};
use obrain_core::graph::GraphStoreMut;
use obrain_core::index::vector::{PropertyVectorAccessor, VectorAccessor};
use obrain_substrate::SubstrateStore;

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
    let store = SubstrateStore::open_tempfile().unwrap();
    let id = store.create_node(&["Test"]);
    let vec_data: Arc<[f32]> = vec![1.0, 2.0, 3.0].into();
    store.set_node_property(id, "embedding", Value::Vector(vec_data.clone()));

    let accessor = PropertyVectorAccessor::new(&store, "embedding");
    let result = accessor.get_vector(id);
    assert!(result.is_some());
    assert_eq!(result.unwrap().as_ref(), vec_data.as_ref());

    // Non-existent node (use a NodeId that substrate has not allocated)
    assert!(accessor.get_vector(NodeId::new(999)).is_none());

    // Wrong property type
    store.set_node_property(id, "name", Value::from("hello"));
    let name_accessor = PropertyVectorAccessor::new(&store, "name");
    assert!(name_accessor.get_vector(id).is_none());
}
