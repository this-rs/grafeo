//! Integration tests for `obrain_core::execution::operators::FactorizedExpandOperator`
//! (and `FactorizedExpandChain`) against the substrate backend.
//!
//! Relocated from `crates/obrain-core/src/execution/operators/factorized_expand.rs`'s
//! in-crate `#[cfg(test)] mod tests` block as part of T17 W4.p4. See note
//! tagged `t17 w4.p4 migration-pattern` for rationale.
//!
//! Note: `SingleChunkOperator` is a private test helper in the original
//! source; it is re-inlined below to avoid widening the public API of
//! `obrain-core` for a test-only concern.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_factorized_expand
//! ```

use obrain_common::types::LogicalType;
use obrain_core::execution::chunk::DataChunk;
use obrain_core::execution::operators::{
    FactorizedExpandChain, FactorizedExpandOperator, Operator, OperatorResult, ScanOperator,
};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_core::graph::Direction;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

/// Helper operator that returns a single chunk once. Inlined from the
/// private `SingleChunkOperator` in the source (test-only, not part of
/// the crate's public surface).
struct SingleChunkOperator {
    chunk: Option<DataChunk>,
}

impl SingleChunkOperator {
    fn new(chunk: DataChunk) -> Self {
        Self { chunk: Some(chunk) }
    }
}

impl Operator for SingleChunkOperator {
    fn next(&mut self) -> OperatorResult {
        Ok(self.chunk.take())
    }

    fn reset(&mut self) {
        // Cannot reset - chunk is consumed
    }

    fn name(&self) -> &'static str {
        "SingleChunk"
    }
}

#[test]
fn test_factorized_expand_basic() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes
    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    // Alix knows Gus and Vincent
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, vincent, "KNOWS");

    let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

    let mut expand = FactorizedExpandOperator::new(
        store.clone(),
        scan,
        0,
        Direction::Outgoing,
        vec!["KNOWS".to_string()],
    );

    // Get factorized result
    let result = expand.next_factorized().unwrap();
    assert!(result.is_some());

    let chunk = result.unwrap();

    // Should have 2 levels: sources and neighbors
    assert_eq!(chunk.level_count(), 2);

    // Level 0 has 3 sources (Alix, Gus, Vincent)
    assert_eq!(chunk.level(0).unwrap().column_count(), 1);

    // Level 1 has edges and targets
    // Only Alix has outgoing KNOWS edges (to Gus and Vincent)
    // So we should have 2 edges total
    assert_eq!(chunk.level(1).unwrap().column_count(), 2);
}

#[test]
fn test_factorized_vs_flat_equivalence() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["Person"]);

    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, vincent, "KNOWS");
    store.create_edge(gus, vincent, "KNOWS");

    // Run factorized expand
    let scan1 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
    let mut factorized_expand =
        FactorizedExpandOperator::new(store.clone(), scan1, 0, Direction::Outgoing, vec![]);

    let factorized_result = factorized_expand.next_factorized().unwrap().unwrap();
    let flat_from_factorized = factorized_result.flatten();

    // Run regular expand (using the factorized operator's flat interface)
    let scan2 = Box::new(ScanOperator::with_label(store.clone(), "Person"));
    let mut regular_expand =
        FactorizedExpandOperator::new(store.clone(), scan2, 0, Direction::Outgoing, vec![]);

    let flat_result = regular_expand.next().unwrap().unwrap();

    // Both should have the same row count
    assert_eq!(
        flat_from_factorized.row_count(),
        flat_result.row_count(),
        "Factorized and flat should produce same row count"
    );
}

#[test]
fn test_factorized_expand_no_edges() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create nodes with no edges
    store.create_node(&["Person"]);
    store.create_node(&["Person"]);

    let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

    let mut expand =
        FactorizedExpandOperator::new(store.clone(), scan, 0, Direction::Outgoing, vec![]);

    let result = expand.next_factorized().unwrap();
    assert!(result.is_some());

    let chunk = result.unwrap();
    // Should only have the source level (no expansion level added when no edges)
    assert_eq!(chunk.level_count(), 1);
}

#[test]
fn test_factorized_chain_two_hop() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create a 2-hop graph: a -> b1, b2 -> c1, c2, c3, c4
    let a = store.create_node(&["Person"]);
    let b1 = store.create_node(&["Person"]);
    let b2 = store.create_node(&["Person"]);
    let c1 = store.create_node(&["Person"]);
    let c2 = store.create_node(&["Person"]);
    let c3 = store.create_node(&["Person"]);
    let c4 = store.create_node(&["Person"]);

    // a knows b1 and b2
    store.create_edge(a, b1, "KNOWS");
    store.create_edge(a, b2, "KNOWS");

    // b1 knows c1 and c2
    store.create_edge(b1, c1, "KNOWS");
    store.create_edge(b1, c2, "KNOWS");

    // b2 knows c3 and c4
    store.create_edge(b2, c3, "KNOWS");
    store.create_edge(b2, c4, "KNOWS");

    // Create source chunk with just node 'a'
    let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
    source_chunk.column_mut(0).unwrap().push_node_id(a);
    source_chunk.set_count(1);

    let source = Box::new(SingleChunkOperator::new(source_chunk));

    // Build 2-hop chain
    let chain = FactorizedExpandChain::new(store.clone(), source)
        .expand(0, Direction::Outgoing, vec!["KNOWS".to_string()])
        .unwrap()
        .expand(1, Direction::Outgoing, vec!["KNOWS".to_string()]) // column 1 is target from first expand
        .unwrap();

    let result = chain.finish().expect("Should have result");

    // Should have 3 levels: source (a), hop1 (b1,b2), hop2 (c1,c2,c3,c4)
    assert_eq!(result.level_count(), 3);

    // Physical size: 1 (source) + 2+2 (hop1 edges+targets) + 4+4 (hop2 edges+targets) = 13
    // vs flat which would be 4 rows * 5 columns = 20
    assert_eq!(result.physical_size(), 13);

    // Logical row count should be 4 (4 paths: a->b1->c1, a->b1->c2, a->b2->c3, a->b2->c4)
    assert_eq!(result.logical_row_count(), 4);

    // Flatten and verify
    let flat = result.flatten();
    assert_eq!(flat.row_count(), 4);
}

#[test]
fn test_factorized_expand_multi_edge_type_filter() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    let alix = store.create_node(&["Person"]);
    let gus = store.create_node(&["Person"]);
    let vincent = store.create_node(&["City"]);

    // Mixed edge types
    store.create_edge(alix, gus, "KNOWS");
    store.create_edge(alix, vincent, "LIVES_IN");
    store.create_edge(gus, vincent, "WORKS_AT");

    let scan = Box::new(ScanOperator::with_label(store.clone(), "Person"));

    // Filter for KNOWS and LIVES_IN (case-insensitive)
    let mut expand = FactorizedExpandOperator::new(
        store.clone(),
        scan,
        0,
        Direction::Outgoing,
        vec!["knows".to_string(), "lives_in".to_string()],
    );

    let result = expand.next_factorized().unwrap().unwrap();
    let flat = result.flatten();

    // From Alix: KNOWS (to Gus) and LIVES_IN (to Vincent) = 2 rows
    // From Gus: WORKS_AT is filtered out = 0 rows
    assert_eq!(flat.row_count(), 2);
}

#[test]
fn test_factorized_memory_savings() {
    let store = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Create a star graph: center connected to 10 leaves
    let center = store.create_node(&["Center"]);
    let mut leaves = Vec::new();
    for _ in 0..10 {
        let leaf = store.create_node(&["Leaf"]);
        store.create_edge(center, leaf, "POINTS_TO");
        leaves.push(leaf);
    }

    // Scan just the center
    let mut source_chunk = DataChunk::with_capacity(&[LogicalType::Node], 1);
    source_chunk.column_mut(0).unwrap().push_node_id(center);
    source_chunk.set_count(1);

    let single = Box::new(SingleChunkOperator::new(source_chunk));

    let mut expand =
        FactorizedExpandOperator::new(store.clone(), single, 0, Direction::Outgoing, vec![]);

    let factorized = expand.next_factorized().unwrap().unwrap();

    // Physical size should be 1 (source) + 10 (edges) + 10 (targets) = 21 values
    // vs flat which would be 10 rows * 3 columns = 30 values
    assert_eq!(factorized.physical_size(), 21);

    // But logical row count should be 10
    assert_eq!(factorized.logical_row_count(), 10);

    // Flatten and verify correctness
    let flat = factorized.flatten();
    assert_eq!(flat.row_count(), 10);
}
