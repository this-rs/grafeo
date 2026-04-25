//! Integration tests for
//! `obrain_core::execution::parallel::source::ParallelNodeScanSource`
//! against the substrate backend.
//!
//! ## Why this lives in obrain-substrate and not in obrain-core
//!
//! Relocated as part of T17 Step 3 W2/Class-2 follow-up (decision
//! `b1dfe229`). `obrain-core` cannot take `obrain-substrate` as a
//! dev-dependency without creating two distinct compilation units of
//! `obrain-core` (gotcha `598dda40`). The forward direction
//! (`obrain-substrate → obrain-core`) has no cycle, so moving
//! ParallelNodeScanSource tests into an integration test of
//! `obrain-substrate` is the post-T17-cutover home for the LPG-era
//! fixtures.
//!
//! Only the three LpgStore-coupled tests move here
//! (`test_parallel_node_scan_source`, `test_parallel_node_scan_partition`,
//! `test_parallel_node_scan_morsels`). The remaining non-graph tests
//! (ParallelVectorSource / RangeSource / ParallelChunkSource / triple_scan)
//! stay in obrain-core because they do not touch any GraphStore.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_parallel_source
//! ```

use obrain_core::execution::parallel::{Morsel, ParallelNodeScanSource, ParallelSource};
use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

#[test]
fn test_parallel_node_scan_source() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Add some nodes with labels
    for i in 0..100 {
        if i % 2 == 0 {
            store.create_node(&["Person", "Employee"]);
        } else {
            store.create_node(&["Person"]);
        }
    }

    // Test scan all nodes
    let source = ParallelNodeScanSource::new(Arc::clone(&store) as Arc<dyn GraphStore>);
    assert_eq!(source.total_rows(), Some(100));
    assert!(source.is_partitionable());
    assert_eq!(source.num_columns(), 1);

    // Test scan by label
    let source_person =
        ParallelNodeScanSource::with_label(Arc::clone(&store) as Arc<dyn GraphStore>, "Person");
    assert_eq!(source_person.total_rows(), Some(100));

    let source_employee =
        ParallelNodeScanSource::with_label(Arc::clone(&store) as Arc<dyn GraphStore>, "Employee");
    assert_eq!(source_employee.total_rows(), Some(50));
}

#[test]
fn test_parallel_node_scan_partition() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Add 100 nodes
    for _ in 0..100 {
        store.create_node(&[]);
    }

    let source = ParallelNodeScanSource::new(Arc::clone(&store) as Arc<dyn GraphStore>);

    // Create partition for rows 20-50
    let morsel = Morsel::new(0, 0, 20, 50);
    let mut partition = source.create_partition(&morsel);

    // Should produce 30 rows total
    let mut total: usize = 0;
    while let Ok(Some(chunk)) = partition.next_chunk(10) {
        total += chunk.len();
    }
    assert_eq!(total, 30);
}

#[test]
fn test_parallel_node_scan_morsels() {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());

    // Add 1000 nodes
    for _ in 0..1000 {
        store.create_node(&[]);
    }

    let source = ParallelNodeScanSource::new(Arc::clone(&store) as Arc<dyn GraphStore>);

    // Generate morsels with size 256
    let morsels = source.generate_morsels(256, 0);
    assert_eq!(morsels.len(), 4); // 1000 / 256 = 3 full + 1 partial

    // Verify morsels cover all rows
    let mut total_rows = 0;
    for morsel in &morsels {
        total_rows += morsel.end_row - morsel.start_row;
    }
    assert_eq!(total_rows, 1000);
}
