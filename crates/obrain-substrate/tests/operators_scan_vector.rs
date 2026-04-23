//! Integration tests for `obrain_core::execution::operators::VectorScanOperator`
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
//! cargo test -p obrain-substrate --test operators_scan_vector
//! ```

use obrain_common::types::Value;
use obrain_core::execution::operators::{Operator, VectorScanOperator};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_core::graph::GraphStore;
use obrain_core::index::vector::DistanceMetric;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

#[test]
fn test_vector_scan_brute_force() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes with vector embeddings
    let n1 = store.create_node(&["Document"]);
    let n2 = store.create_node(&["Document"]);
    let n3 = store.create_node(&["Document"]);

    // Set vector properties - n1 is closest to query
    store.set_node_property(n1, "embedding", Value::Vector(vec![0.1, 0.2, 0.3].into()));
    store.set_node_property(n2, "embedding", Value::Vector(vec![0.5, 0.6, 0.7].into()));
    store.set_node_property(n3, "embedding", Value::Vector(vec![0.9, 0.8, 0.7].into()));

    // Query vector similar to n1
    let query = vec![0.1, 0.2, 0.35];

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "embedding",
        query,
        2, // k=2
        DistanceMetric::Euclidean,
    )
    .with_label("Document");

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 2);

    // First result should be n1 (closest)
    let first_node = chunk.column(0).unwrap().get_node_id(0);
    assert_eq!(first_node, Some(n1));

    // Should be exhausted
    assert!(scan.next().unwrap().is_none());
    let _ = (n2, n3);
}

#[test]
fn test_vector_scan_reset() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let n1 = store.create_node(&["Doc"]);
    store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.2].into()));

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.1, 0.2],
        10,
        DistanceMetric::Cosine,
    );

    // First scan
    let chunk1 = scan.next().unwrap().unwrap();
    assert_eq!(chunk1.row_count(), 1);
    assert!(scan.next().unwrap().is_none());

    // Reset and scan again
    scan.reset();
    let chunk2 = scan.next().unwrap().unwrap();
    assert_eq!(chunk2.row_count(), 1);
    let _ = n1;
}

#[test]
fn test_vector_scan_with_distance_filter() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let n1 = store.create_node(&["Doc"]);
    let n2 = store.create_node(&["Doc"]);

    // n1 is very close, n2 is far
    store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.0].into()));
    store.set_node_property(n2, "vec", Value::Vector(vec![10.0, 10.0].into()));

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.0, 0.0],
        10,
        DistanceMetric::Euclidean,
    )
    .with_max_distance(1.0); // Only n1 should pass

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let node_id = chunk.column(0).unwrap().get_node_id(0);
    assert_eq!(node_id, Some(n1));
    let _ = n2;
}

#[test]
fn test_vector_scan_empty_results() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // No nodes with vectors
    store.create_node(&["Doc"]);

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "embedding",
        vec![0.1, 0.2],
        10,
        DistanceMetric::Cosine,
    );

    let result = scan.next().unwrap();
    assert!(result.is_none());
}

#[test]
fn test_vector_scan_name() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let brute_scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.1],
        10,
        DistanceMetric::Cosine,
    );
    assert_eq!(brute_scan.name(), "VectorScan(BruteForce)");
}

#[test]
fn test_vector_scan_with_min_similarity() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let n1 = store.create_node(&["Doc"]);
    let n2 = store.create_node(&["Doc"]);

    // Normalized vectors for cosine similarity
    // n1: [1, 0] - orthogonal to query
    // n2: [0.707, 0.707] - similar to query [0, 1]
    store.set_node_property(n1, "vec", Value::Vector(vec![1.0, 0.0].into()));
    store.set_node_property(n2, "vec", Value::Vector(vec![0.707, 0.707].into()));

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.0, 1.0], // Query: [0, 1]
        10,
        DistanceMetric::Cosine,
    )
    .with_min_similarity(0.5); // Filters out n1 (similarity ~0)

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);

    let node_id = chunk.column(0).unwrap().get_node_id(0);
    assert_eq!(node_id, Some(n2));
    let _ = n1;
}

#[test]
fn test_vector_scan_with_ef() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let n1 = store.create_node(&["Doc"]);
    store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.2].into()));

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.1, 0.2],
        10,
        DistanceMetric::Cosine,
    )
    .with_ef(128); // Higher ef (doesn't affect brute-force, but tests API)

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 1);
    let _ = n1;
}

#[test]
fn test_vector_scan_with_chunk_capacity() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create many nodes
    for i in 0..10 {
        let node = store.create_node(&["Doc"]);
        store.set_node_property(node, "vec", Value::Vector(vec![i as f32, 0.0].into()));
    }

    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.0, 0.0],
        10,
        DistanceMetric::Euclidean,
    )
    .with_chunk_capacity(3); // Small chunks

    // Should return multiple chunks
    let chunk1 = scan.next().unwrap().unwrap();
    assert_eq!(chunk1.row_count(), 3);

    let chunk2 = scan.next().unwrap().unwrap();
    assert_eq!(chunk2.row_count(), 3);

    let chunk3 = scan.next().unwrap().unwrap();
    assert_eq!(chunk3.row_count(), 3);

    let chunk4 = scan.next().unwrap().unwrap();
    assert_eq!(chunk4.row_count(), 1);

    assert!(scan.next().unwrap().is_none());
}

#[test]
fn test_vector_scan_no_label_filter() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes with different labels
    let n1 = store.create_node(&["TypeA"]);
    let n2 = store.create_node(&["TypeB"]);

    store.set_node_property(n1, "vec", Value::Vector(vec![0.1, 0.2].into()));
    store.set_node_property(n2, "vec", Value::Vector(vec![0.3, 0.4].into()));

    // Without label filter - should find both
    let mut scan = VectorScanOperator::brute_force(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        "vec",
        vec![0.0, 0.0],
        10,
        DistanceMetric::Euclidean,
    );

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 2);
    let _ = (n1, n2);
}

#[cfg(feature = "vector-index")]
#[test]
fn test_vector_scan_with_hnsw_index() {
    use obrain_core::index::vector::{HnswConfig, HnswIndex, PropertyVectorAccessor};

    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes and set vector properties FIRST (so accessor can read them)
    let n1 = store.create_node(&["Doc"]);
    let n2 = store.create_node(&["Doc"]);
    let n3 = store.create_node(&["Doc"]);

    let v1 = vec![0.1f32, 0.2, 0.3];
    let v2 = vec![0.5, 0.6, 0.7];
    let v3 = vec![0.9, 0.8, 0.7];

    store.set_node_property(n1, "vec", Value::Vector(v1.clone().into()));
    store.set_node_property(n2, "vec", Value::Vector(v2.clone().into()));
    store.set_node_property(n3, "vec", Value::Vector(v3.clone().into()));

    // Create HNSW index and insert using accessor
    let config = HnswConfig::new(3, DistanceMetric::Euclidean);
    let index = Arc::new(HnswIndex::new(config));
    let accessor = PropertyVectorAccessor::new(&*store, "vec");

    index.insert(n1, &v1, &accessor);
    index.insert(n2, &v2, &accessor);
    index.insert(n3, &v3, &accessor);

    // Search using index
    let query = vec![0.1f32, 0.2, 0.35];
    let mut scan = VectorScanOperator::with_index(
        Arc::clone(&store) as Arc<dyn GraphStore>,
        Arc::clone(&index),
        query,
        2,
    )
    .with_property("vec");

    assert_eq!(scan.name(), "VectorScan(HNSW)");

    let chunk = scan.next().unwrap().unwrap();
    assert_eq!(chunk.row_count(), 2);

    // First result should be n1 (closest)
    let first_node = chunk.column(0).unwrap().get_node_id(0);
    assert_eq!(first_node, Some(n1));
}
