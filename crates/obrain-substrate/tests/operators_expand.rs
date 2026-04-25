//! Integration tests for `obrain_core::execution::operators::ExpandOperator`
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
//! cargo test -p obrain-substrate --test operators_expand
//! ```

use obrain_common::types::NodeId;
use obrain_core::execution::operators::{ExpandOperator, Operator, ScanOperator};
use obrain_core::graph::Direction;
use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

/// Creates a new `SubstrateStore` wrapped in an `Arc` and returns both the
/// `GraphStoreMut` handle (for mutation) and a `GraphStore` trait-object
/// handle (for operators).
fn test_store() -> (Arc<dyn GraphStoreMut>, Arc<dyn GraphStore>) {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let dyn_store: Arc<dyn GraphStore> = Arc::clone(&store) as Arc<dyn GraphStore>;
    (store, dyn_store)
}

#[test]
fn test_expand_outgoing() {
    let (store, dyn_store) = test_store();

    // Create nodes
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    // Create edges: Alix -> Gus, Alix -> Vincent
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, vincent, "KNOWS");

    // Scan Alix only
    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

    let mut expand = ExpandOperator::new(
        Arc::clone(&dyn_store),
        scan,
        0, // source column
        Direction::Outgoing,
        vec![],
    );

    // Collect all results
    let mut results = Vec::new();
    while let Ok(Some(chunk)) = expand.next() {
        for i in 0..chunk.row_count() {
            let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
            let edge = chunk.column(1).unwrap().get_edge_id(i).unwrap();
            let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
            results.push((src, edge, dst));
        }
    }

    // Alix -> Gus, Alix -> Vincent
    assert_eq!(results.len(), 2);

    // All source nodes should be Alix
    for (src, _, _) in &results {
        assert_eq!(*src, alix);
    }

    // Target nodes should be Gus and Vincent
    let targets: Vec<NodeId> = results.iter().map(|(_, _, dst)| *dst).collect();
    assert!(targets.contains(&gus));
    assert!(targets.contains(&vincent));
}

#[test]
fn test_expand_with_edge_type_filter() {
    let (store, dyn_store) = test_store();

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let company = store.create_node(&["Company"]);

    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, company, "WORKS_AT");

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

    let mut expand = ExpandOperator::new(
        Arc::clone(&dyn_store),
        scan,
        0,
        Direction::Outgoing,
        vec!["KNOWS".to_string()],
    );

    let mut results = Vec::new();
    while let Ok(Some(chunk)) = expand.next() {
        for i in 0..chunk.row_count() {
            let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
            results.push(dst);
        }
    }

    // Only KNOWS edges should be followed
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], gus);
}

#[test]
fn test_expand_incoming() {
    let (store, dyn_store) = test_store();

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);

    store.create_edge(alix, gus, "KNOWS");

    // Scan Gus
    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Incoming, vec![]);

    let mut results = Vec::new();
    while let Ok(Some(chunk)) = expand.next() {
        for i in 0..chunk.row_count() {
            let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
            let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
            results.push((src, dst));
        }
    }

    // Gus <- Alix (Gus's incoming edge from Alix)
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, gus); // source in the expand is Gus
    assert_eq!(results[0].1, alix); // target is Alix (who points to Gus)
}

#[test]
fn test_expand_no_edges() {
    let (store, dyn_store) = test_store();

    store.create_node(&["Person"]);

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));

    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

    let result = expand.next().unwrap();
    assert!(result.is_none());
}

#[test]
fn test_expand_reset() {
    let (store, dyn_store) = test_store();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

    // First pass
    let mut count1 = 0;
    while let Ok(Some(chunk)) = expand.next() {
        count1 += chunk.row_count();
    }

    // Reset and run again
    expand.reset();
    let mut count2 = 0;
    while let Ok(Some(chunk)) = expand.next() {
        count2 += chunk.row_count();
    }

    assert_eq!(count1, count2);
    assert_eq!(count1, 1);
}

#[test]
fn test_expand_name() {
    let (_store, dyn_store) = test_store();
    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
    let expand = ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);
    assert_eq!(expand.name(), "Expand");
}

#[test]
fn test_expand_with_chunk_capacity() {
    let (store, dyn_store) = test_store();

    let a = store.create_node(&["Person"]);
    for _ in 0..5 {
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
    }

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![])
            .with_chunk_capacity(2);

    // With capacity 2 and 5 edges from node a, we should get multiple chunks
    let mut total = 0;
    let mut chunk_count = 0;
    while let Ok(Some(chunk)) = expand.next() {
        chunk_count += 1;
        total += chunk.row_count();
    }

    assert_eq!(total, 5);
    assert!(
        chunk_count >= 2,
        "Expected multiple chunks with small capacity"
    );
}

#[test]
fn test_expand_edge_type_case_insensitive() {
    let (store, dyn_store) = test_store();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    store.create_edge(a, b, "KNOWS");

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
    let mut expand = ExpandOperator::new(
        Arc::clone(&dyn_store),
        scan,
        0,
        Direction::Outgoing,
        vec!["knows".to_string()], // lowercase
    );

    let mut count = 0;
    while let Ok(Some(chunk)) = expand.next() {
        count += chunk.row_count();
    }

    // Should match case-insensitively
    assert_eq!(count, 1);
}

#[test]
fn test_expand_multiple_source_nodes() {
    let (store, dyn_store) = test_store();

    let a = store.create_node(&["Person"]);
    let b = store.create_node(&["Person"]);
    let c = store.create_node(&["Person"]);

    store.create_edge(a, c, "KNOWS");
    store.create_edge(b, c, "KNOWS");

    let scan = Box::new(ScanOperator::with_label(Arc::clone(&dyn_store), "Person"));
    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

    let mut results = Vec::new();
    while let Ok(Some(chunk)) = expand.next() {
        for i in 0..chunk.row_count() {
            let src = chunk.column(0).unwrap().get_node_id(i).unwrap();
            let dst = chunk.column(2).unwrap().get_node_id(i).unwrap();
            results.push((src, dst));
        }
    }

    // Both a->c and b->c
    assert_eq!(results.len(), 2);
}

#[test]
fn test_expand_empty_input() {
    let (_store, dyn_store) = test_store();

    // No nodes with this label
    let scan = Box::new(ScanOperator::with_label(
        Arc::clone(&dyn_store),
        "Nonexistent",
    ));
    let mut expand =
        ExpandOperator::new(Arc::clone(&dyn_store), scan, 0, Direction::Outgoing, vec![]);

    let result = expand.next().unwrap();
    assert!(result.is_none());
}
