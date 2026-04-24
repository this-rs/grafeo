//! Integration tests for
//! `obrain_core::execution::operators::TypedDegreeTopKOperator`
//! against the substrate backend (T17h T9).
//!
//! Lives in obrain-substrate and not obrain-core for the same dev-dep
//! cycle reason as `operators_scan.rs` — see the header of that file.
//!
//! Run with :
//!
//! ```bash
//! cargo test -p obrain-substrate --test operators_typed_degree_topk
//! ```

use obrain_core::execution::operators::{
    Operator, TypedDegreeDirection, TypedDegreeTopKOperator,
};
use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_common::types::{NodeId, Value};

fn read_row(chunk: &obrain_core::execution::DataChunk, i: usize) -> (NodeId, i64) {
    let col_node = chunk.column(0).expect("node column");
    let col_deg = chunk.column(1).expect("degree column");
    let n = col_node.get_node_id(i).expect("node id at index");
    let d = match col_deg.get_value(i) {
        Some(Value::Int64(d)) => d,
        other => panic!("expected Int64, got {other:?}"),
    };
    (n, d)
}

/// Reads a 3-column row (Separate direction output : [Node, Int64 out,
/// Int64 in]).
fn read_row_separate(
    chunk: &obrain_core::execution::DataChunk,
    i: usize,
) -> (NodeId, i64, i64) {
    let col_node = chunk.column(0).expect("node column");
    let col_out = chunk.column(1).expect("out column");
    let col_in = chunk.column(2).expect("in column");
    let n = col_node.get_node_id(i).expect("node id at index");
    let out = match col_out.get_value(i) {
        Some(Value::Int64(v)) => v,
        other => panic!("expected Int64 for out, got {other:?}"),
    };
    let inc = match col_in.get_value(i) {
        Some(Value::Int64(v)) => v,
        other => panic!("expected Int64 for in, got {other:?}"),
    };
    (n, out, inc)
}
use obrain_substrate::SubstrateStore;
use std::sync::Arc;

fn setup() -> (Arc<dyn GraphStoreMut>, Vec<NodeId>) {
    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    // 5 File nodes with varied IMPORTS degrees :
    //   f0 : 0 out, 2 in  (total 2)
    //   f1 : 1 out, 0 in  (total 1)
    //   f2 : 3 out, 1 in  (total 4)
    //   f3 : 1 out, 1 in  (total 2)
    //   f4 : 2 out, 0 in  (total 2)
    let files: Vec<NodeId> = (0..5).map(|_| store.create_node(&["File"])).collect();
    // Also a non-File node to test label filtering.
    let _other = store.create_node(&["Package"]);

    // Edges — all `IMPORTS` type unless noted.
    store.create_edge(files[1], files[0], "IMPORTS"); // f1->f0
    store.create_edge(files[2], files[0], "IMPORTS"); // f2->f0
    store.create_edge(files[2], files[3], "IMPORTS"); // f2->f3
    store.create_edge(files[2], files[4], "IMPORTS"); // f2->f4
    store.create_edge(files[3], files[2], "IMPORTS"); // f3->f2
    store.create_edge(files[4], files[1], "IMPORTS"); // f4->f1  (wait, this adds f1.in)
    store.create_edge(files[4], files[2], "IMPORTS"); // f4->f2
    //
    // Let's recompute to match the plan above :
    //   f0 : out=0, in = {f1, f2} = 2
    //   f1 : out={f0} = 1, in = {f4} = 1   -- adjusted from the header
    //   f2 : out={f0, f3, f4} = 3, in = {f3, f4} = 2
    //   f3 : out={f2} = 1, in = {f2} = 1
    //   f4 : out={f1, f2} = 2, in = {f2} = 1
    // Separate CONTAINS edges to verify type filtering.
    store.create_edge(files[0], files[1], "CONTAINS");
    store.create_edge(files[0], files[2], "CONTAINS");
    store.create_edge(files[0], files[3], "CONTAINS");

    (store, files)
}

#[test]
fn typed_degree_topk_outgoing_only() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Outgoing,
        3,
    );

    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 3);

    // Expected out-degree per file under IMPORTS :
    //   f0=0, f1=1, f2=3, f3=1, f4=2
    //   Top 3 DESC by degree (ties by slot ASC) : [f2=3, f4=2, f1=1]
    let got: Vec<(NodeId, i64)> = (0..3).map(|i| read_row(&chunk, i)).collect();

    assert_eq!(got, vec![(files[2], 3), (files[4], 2), (files[1], 1)]);
    assert!(op.next().unwrap().is_none());
}

#[test]
fn typed_degree_topk_both_directions() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Both,
        5,
    );

    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 5);

    // out+in per file under IMPORTS :
    //   f0 : 0+2=2, f1 : 1+1=2, f2 : 3+2=5, f3 : 1+1=2, f4 : 2+1=3
    //   Top 5 DESC by total (ties by slot ASC) :
    //   [f2=5, f4=3, f0=2, f1=2, f3=2]
    let got: Vec<(NodeId, i64)> = (0..5).map(|i| read_row(&chunk, i)).collect();

    assert_eq!(
        got,
        vec![
            (files[2], 5),
            (files[4], 3),
            (files[0], 2),
            (files[1], 2),
            (files[3], 2),
        ]
    );
}

#[test]
fn typed_degree_topk_respects_edge_type_filter() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    // f0 has CONTAINS out-degree 3 (f1, f2, f3) — every other file has 0.
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("CONTAINS".to_string()),
        TypedDegreeDirection::Outgoing,
        1,
    );

    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 1);
    assert_eq!(read_row(&chunk, 0), (files[0], 3));
}

#[test]
fn typed_degree_topk_unknown_type_is_empty() {
    let (store, _files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("NONEXISTENT".to_string()),
        TypedDegreeDirection::Both,
        10,
    );
    // All degrees zero for an unknown edge type. The operator still
    // emits `min(k, |candidates|) = 5` rows with degree 0 — equivalent
    // to the slow path's output for this query.
    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 5);
    for i in 0..5 {
        let (_n, d) = read_row(&chunk, i);
        assert_eq!(d, 0);
    }
}

#[test]
fn typed_degree_topk_reset_replays() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Outgoing,
        2,
    );
    let first = op.next().unwrap().expect("chunk 1");
    assert_eq!(first.row_count(), 2);
    assert!(op.next().unwrap().is_none());

    op.reset();
    let second = op.next().unwrap().expect("chunk 2");
    assert_eq!(second.row_count(), 2);

    // Top-2 unchanged across reset : [f2=3, f4=2].
    let got: Vec<NodeId> = (0..2).map(|i| read_row(&second, i).0).collect();
    assert_eq!(got, vec![files[2], files[4]]);
}

#[test]
fn typed_degree_topk_k_larger_than_label_set() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    // K = 50 with only 5 File nodes — fast-path skips the heap.
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Outgoing,
        50,
    );
    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 5);
    let got: Vec<(NodeId, i64)> = (0..5).map(|i| read_row(&chunk, i)).collect();
    // Full sort : f2=3, f4=2, f1=1, f3=1, f0=0.
    assert_eq!(
        got,
        vec![
            (files[2], 3),
            (files[4], 2),
            (files[1], 1),
            (files[3], 1),
            (files[0], 0),
        ]
    );
}

/// T17h T9a — `Direction::Separate` emits 3 columns (node, out, in).
/// Ranking is on `out + in` (same as `Both`) so the row order matches
/// the `Both` test above, but the two counts are exposed separately.
#[test]
fn typed_degree_topk_separate_outputs_out_and_in() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Separate,
        5,
    );

    let chunk = op.next().unwrap().expect("one chunk");
    assert_eq!(chunk.row_count(), 5);
    assert_eq!(chunk.column_count(), 3, "Separate must emit 3 columns");

    // Expected per-file (out, in) under IMPORTS :
    //   f0 : (0, 2) total=2
    //   f1 : (1, 1) total=2
    //   f2 : (3, 2) total=5
    //   f3 : (1, 1) total=2
    //   f4 : (2, 1) total=3
    // Ranking : total DESC, slot ASC → [f2, f4, f0, f1, f3].
    let got: Vec<(NodeId, i64, i64)> = (0..5).map(|i| read_row_separate(&chunk, i)).collect();
    assert_eq!(
        got,
        vec![
            (files[2], 3, 2),
            (files[4], 2, 1),
            (files[0], 0, 2),
            (files[1], 1, 1),
            (files[3], 1, 1),
        ]
    );
    assert!(op.next().unwrap().is_none());
}

/// T17h T9a — reset on a Separate operator replays the exact same
/// 3-column output. Locks in that the triple-layout `results` vec is
/// rebuilt consistently across invocations.
#[test]
fn typed_degree_topk_separate_reset_replays() {
    let (store, files) = setup();
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut op = TypedDegreeTopKOperator::new(
        arc_store,
        Some("File".to_string()),
        Some("IMPORTS".to_string()),
        TypedDegreeDirection::Separate,
        2,
    );
    let first = op.next().unwrap().expect("chunk 1");
    assert_eq!(first.row_count(), 2);
    assert_eq!(first.column_count(), 3);
    assert!(op.next().unwrap().is_none());

    op.reset();
    let second = op.next().unwrap().expect("chunk 2");
    assert_eq!(second.row_count(), 2);
    assert_eq!(second.column_count(), 3);

    // Top-2 unchanged across reset : [(f2, 3, 2), (f4, 2, 1)].
    let got: Vec<(NodeId, i64, i64)> = (0..2).map(|i| read_row_separate(&second, i)).collect();
    assert_eq!(got, vec![(files[2], 3, 2), (files[4], 2, 1)]);
}

/// T17i T1 — concurrent `create_edge` on a fresh store must not
/// double-count the typed-degree columns. The deferred
/// `ensure_initialized` serialises the first init under a mutex ;
/// pre-init of the registry via `typed_degrees()` at the top of
/// `create_edge` (before `splice_edge_at_head`) guarantees the
/// rebuild scan runs on the pre-edge state for exactly one thread
/// and returns an empty registry for a fresh store, after which
/// every thread's explicit `incr_typed_*_degree(+1)` is the single
/// source of truth.
///
/// Regression anchor : prior to the T17i T1 fix, the first
/// `create_edge` path triggered `typed_degrees()` after
/// `splice_edge_at_head` ; the rebuild scan then found the new
/// edge already in the zone and incremented its column, and the
/// subsequent explicit `incr_typed_out_degree(+1)` added a second
/// unit — yielding 2 for a single `create_edge` call.
#[test]
fn typed_degrees_concurrent_init_no_double_count() {
    use std::sync::Arc;
    use std::sync::Barrier;

    const THREADS: usize = 8;
    const EDGES_PER_THREAD: usize = 50;

    let store: Arc<dyn GraphStoreMut> = Arc::new(SubstrateStore::open_tempfile().unwrap());
    // Allocate a pool of nodes that threads can pair up for edges.
    let nodes: Vec<NodeId> = (0..(THREADS * 2))
        .map(|_| store.create_node(&["File"]))
        .collect();

    let barrier = Arc::new(Barrier::new(THREADS));
    let mut handles = Vec::new();
    for t in 0..THREADS {
        let store = store.clone();
        let barrier = barrier.clone();
        let src = nodes[t * 2];
        let dst = nodes[t * 2 + 1];
        handles.push(std::thread::spawn(move || {
            // All threads wait, then release — the first `create_edge`
            // call races into `typed_degrees()` / `ensure_initialized`.
            barrier.wait();
            for _ in 0..EDGES_PER_THREAD {
                let _ = store.create_edge(src, dst, "IMPORTS");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // For each (src, dst) pair in the thread pool : out_degree must be
    // exactly EDGES_PER_THREAD (not 2× ; no misses either).
    let arc_store = store.clone() as Arc<dyn GraphStore>;
    let mut out_sum: i64 = 0;
    let mut in_sum: i64 = 0;
    for t in 0..THREADS {
        let src = nodes[t * 2];
        let dst = nodes[t * 2 + 1];
        let out = arc_store.out_degree_by_type(src, Some("IMPORTS"));
        let inn = arc_store.in_degree_by_type(dst, Some("IMPORTS"));
        assert_eq!(
            out, EDGES_PER_THREAD,
            "thread {t} src out_degree must be exactly {EDGES_PER_THREAD}"
        );
        assert_eq!(
            inn, EDGES_PER_THREAD,
            "thread {t} dst in_degree must be exactly {EDGES_PER_THREAD}"
        );
        out_sum += out as i64;
        in_sum += inn as i64;
    }
    let expected = (THREADS * EDGES_PER_THREAD) as i64;
    assert_eq!(out_sum, expected);
    assert_eq!(in_sum, expected);
}
