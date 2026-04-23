//! Integration tests for `obrain_core::execution::operators::VectorJoinOperator`
//! against the substrate backend.
//!
//! ## Why this lives in obrain-substrate and not in obrain-core
//!
//! Relocated as part of T17 Step 3 W2/Class-2 follow-up (decision
//! `b1dfe229`). `obrain-core` cannot take `obrain-substrate` as a
//! dev-dependency without creating two distinct compilation units of
//! `obrain-core` (gotcha `598dda40`). The forward direction
//! (`obrain-substrate → obrain-core`) has no cycle, so moving operator
//! tests into an integration test of `obrain-substrate` is the
//! post-T17-cutover home for the LPG-era fixtures.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_vector_join
//! ```

use obrain_common::types::Value;
use obrain_core::execution::operators::{NodeListOperator, Operator, VectorJoinOperator};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_core::graph::GraphStore;
use obrain_core::index::vector::DistanceMetric;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

#[test]
fn test_vector_join_static_query() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes with vector embeddings
    let n1 = store.create_node(&["Item"]);
    let n2 = store.create_node(&["Item"]);
    let n3 = store.create_node(&["Item"]);

    store.set_node_property(n1, "embedding", Value::Vector(vec![1.0, 0.0, 0.0].into()));
    store.set_node_property(n2, "embedding", Value::Vector(vec![0.0, 1.0, 0.0].into()));
    store.set_node_property(n3, "embedding", Value::Vector(vec![0.9, 0.1, 0.0].into()));

    // Create a simple left operator that produces one row
    let left = Box::new(NodeListOperator::new(vec![n1], 1024));

    // Create vector join with query similar to n1
    let query = vec![1.0, 0.0, 0.0];
    let mut join = VectorJoinOperator::with_static_query(
        left,
        Arc::clone(&store) as Arc<dyn GraphStore>,
        query,
        "embedding",
        3,
        DistanceMetric::Euclidean,
    );

    // Should find all 3 items, sorted by distance
    let mut total_results = 0;
    while let Ok(Some(chunk)) = join.next() {
        total_results += chunk.row_count();
    }

    assert_eq!(total_results, 3);
    // Silence unused bindings; n2/n3 exist to populate the store.
    let _ = (n2, n3);
}

#[test]
fn test_vector_join_entity_to_entity() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create source nodes
    let src1 = store.create_node(&["Source"]);
    store.set_node_property(src1, "embedding", Value::Vector(vec![1.0, 0.0].into()));

    // Create target nodes
    let t1 = store.create_node(&["Target"]);
    let t2 = store.create_node(&["Target"]);
    store.set_node_property(t1, "embedding", Value::Vector(vec![0.9, 0.1].into()));
    store.set_node_property(t2, "embedding", Value::Vector(vec![0.0, 1.0].into()));

    // Left operator produces source node
    let left = Box::new(NodeListOperator::new(vec![src1], 1024));

    // Entity-to-entity join: find targets similar to source
    let mut join = VectorJoinOperator::entity_to_entity(
        left,
        Arc::clone(&store) as Arc<dyn GraphStore>,
        0, // node column
        "embedding",
        "embedding",
        2,
        DistanceMetric::Euclidean,
    )
    .with_right_label("Target");

    // Should find both targets
    let mut total_results = 0;
    while let Ok(Some(chunk)) = join.next() {
        total_results += chunk.row_count();
        // First result should be t1 (closer to src1)
        if total_results == 1 {
            let right_col = chunk.column(1).unwrap();
            let right_node = right_col.get_node_id(0).unwrap();
            assert_eq!(right_node, t1);
        }
    }

    assert_eq!(total_results, 2);
    let _ = t2;
}

#[test]
fn test_vector_join_with_distance_filter() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes with embeddings
    let n1 = store.create_node(&["Item"]);
    let n2 = store.create_node(&["Item"]);
    store.set_node_property(n1, "vec", Value::Vector(vec![1.0, 0.0].into()));
    store.set_node_property(n2, "vec", Value::Vector(vec![0.0, 1.0].into())); // Far away

    let left = Box::new(NodeListOperator::new(vec![n1], 1024));
    let query = vec![1.0, 0.0];

    let mut join = VectorJoinOperator::with_static_query(
        left,
        Arc::clone(&store) as Arc<dyn GraphStore>,
        query,
        "vec",
        10,
        DistanceMetric::Euclidean,
    )
    .with_max_distance(0.5); // Only very close matches

    // Should find only n1 (distance 0.0)
    let mut results = Vec::new();
    while let Ok(Some(chunk)) = join.next() {
        for i in 0..chunk.row_count() {
            let node = chunk.column(1).unwrap().get_node_id(i).unwrap();
            results.push(node);
        }
    }

    assert_eq!(results.len(), 1);
    assert_eq!(results[0], n1);
    let _ = n2;
}

#[test]
fn test_vector_join_name() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let left = Box::new(NodeListOperator::new(vec![], 1024));

    let join = VectorJoinOperator::with_static_query(
        left,
        store as Arc<dyn GraphStore>,
        vec![1.0],
        "embedding",
        5,
        DistanceMetric::Cosine,
    );

    assert_eq!(join.name(), "VectorJoin(BruteForce)");
}
