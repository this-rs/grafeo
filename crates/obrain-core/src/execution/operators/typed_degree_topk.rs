//! Typed-degree Top-K operator (T17h T9).
//!
//! Emits the `K` nodes with the largest degree filtered by a specific
//! edge type (and optionally summing in- and out-degree). Intended as
//! the physical replacement for the canonical pattern
//!
//! ```cypher
//! MATCH (n:Label)
//! OPTIONAL MATCH (n)-[:TYPE]->()         -- out branch
//! OPTIONAL MATCH ()-[:TYPE]->(n)         -- in  branch (optional)
//! WITH n, count(*) AS d                  -- (DISTINCT is OK when edges are
//!                                           unique per (src, dst) pair)
//! ORDER BY d DESC
//! LIMIT K
//! ```
//!
//! When the backend reports `supports_typed_degree() == true`, the
//! planner rewrites this pipeline to a single [`TypedDegreeTopKOperator`]
//! that pulls per-node counts from the per-edge-type degree column
//! (O(1) atomic load, T17h T8) — no edge walk, no aggregation state,
//! a constant-sized min-heap for the top-K selection.
//!
//! ## Semantic fidelity
//!
//! The typed-degree column counts **edges**, not distinct peers. On
//! graphs where `(src, dst)` pairs can carry more than one edge of the
//! same type (multi-graphs), `count(DISTINCT peer)` differs from this
//! operator's output. On Obrain's real corpora (PO / Wikipedia /
//! Megalaw), IMPORTS / CONTAINS / TOUCHES edges are unique per pair,
//! so the answer is exact. The planner should fall back to the slow
//! path when the backend cannot guarantee uniqueness (no such guard
//! exists today — noted as a T17i follow-up).
//!
//! ## Output schema
//!
//! Two columns :
//! - `0 : Node` — the node id
//! - `1 : Int64` — the degree count used for ranking
//!
//! Downstream operators (typically a `Project`) materialise node
//! properties (e.g. `f.path`) and any arithmetic (`imports + dependents
//! AS connections`). The operator stays small and composable.

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStore;
use obrain_common::types::{LogicalType, NodeId};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Count of times the Cypher planner rewrite
/// (`try_plan_typed_degree_topk`) has substituted the canonical
/// `most_connected` pipeline with a [`TypedDegreeTopKOperator`].
///
/// Used by the T9b TDD tests to verify that snapshot queries are
/// routed correctly : EXPLAIN emits the logical plan unchanged after
/// rewrite (cf. `plan.rs::explain_tree` — logical-level only), so we
/// cannot snapshot-match on physical operator names. This counter
/// gives the tests a deterministic signal instead.
///
/// Incremented by the planner in T9c. Reset by tests via
/// [`reset_typed_degree_rewrite_counter`] before each scenario.
pub static TYPED_DEGREE_REWRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Resets the rewrite counter to 0. For test use only.
pub fn reset_typed_degree_rewrite_counter() {
    TYPED_DEGREE_REWRITE_COUNTER.store(0, Ordering::Relaxed);
}

/// Reads the current value of the rewrite counter.
pub fn typed_degree_rewrite_counter() -> u64 {
    TYPED_DEGREE_REWRITE_COUNTER.load(Ordering::Relaxed)
}

/// Direction of the degree sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedDegreeDirection {
    /// Count only outgoing edges (`(n)-[:TYPE]->()`). Output schema :
    /// `[Node, Int64(out_count)]`.
    Outgoing,
    /// Count only incoming edges (`()-[:TYPE]->(n)`). Output schema :
    /// `[Node, Int64(in_count)]`.
    Incoming,
    /// Sum outgoing + incoming (`both`). Output schema :
    /// `[Node, Int64(out_count + in_count)]`.
    Both,
    /// Emit out_count and in_count as separate columns (T17h T9a).
    /// Ranking is on `out + in` (same as `Both`) but the downstream
    /// consumer gets both counts, which is required by the Cypher
    /// planner rewrite that recomposes `imports + dependents AS
    /// connections`. Output schema : `[Node, Int64(out), Int64(in)]`.
    Separate,
}

/// Top-K operator driven by per-edge-type degree counters.
pub struct TypedDegreeTopKOperator {
    store: Arc<dyn GraphStore>,
    /// Node label to scan. `None` means "all nodes" — the operator
    /// still runs but iterates `store.node_ids()` which is O(N).
    label: Option<String>,
    /// Edge type name to filter. `None` means "sum across all types"
    /// which delegates to the total (T5) degree column.
    edge_type: Option<String>,
    /// Direction : outgoing, incoming, or both.
    direction: TypedDegreeDirection,
    /// Top-K size.
    k: usize,
    /// Computed top-K rows, ordered by ranking DESC (ties by slot
    /// ASC). Each tuple is `(node, out_count, in_count)` so a single
    /// storage layout covers all four `TypedDegreeDirection` variants :
    /// Outgoing stores `(n, out, 0)`, Incoming `(n, 0, in)`, Both
    /// `(n, out+in, 0)`, Separate `(n, out, in)`. `next()` projects
    /// the right columns based on `direction`.
    results: Vec<(NodeId, i64, i64)>,
    /// Read position in `results`.
    position: usize,
    /// Per-chunk capacity on output.
    chunk_capacity: usize,
    /// Flipped to `true` when the first call finishes.
    computed: bool,
}

impl TypedDegreeTopKOperator {
    /// Creates a new typed-degree top-K operator.
    ///
    /// # Panics
    ///
    /// Does not panic ; invalid arguments (e.g. unknown `label` /
    /// `edge_type`) simply produce an empty result set at runtime.
    pub fn new(
        store: Arc<dyn GraphStore>,
        label: Option<String>,
        edge_type: Option<String>,
        direction: TypedDegreeDirection,
        k: usize,
    ) -> Self {
        Self {
            store,
            label,
            edge_type,
            direction,
            k,
            results: Vec::new(),
            position: 0,
            chunk_capacity: 2048,
            computed: false,
        }
    }

    /// Overrides the per-chunk output capacity (default 2048).
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Returns the Operator's output schema. Two columns for
    /// Outgoing / Incoming / Both, three for Separate.
    #[must_use]
    pub fn output_schema(&self) -> Vec<LogicalType> {
        match self.direction {
            TypedDegreeDirection::Separate => {
                vec![LogicalType::Node, LogicalType::Int64, LogicalType::Int64]
            }
            _ => vec![LogicalType::Node, LogicalType::Int64],
        }
    }

    /// Computes `(out_count, in_count)` for one node. Outgoing /
    /// Incoming fill the unused slot with 0 so the storage layout is
    /// uniform across variants. Both and Separate both fetch the two
    /// counts from the T8 per-edge-type degree registry.
    fn degree_pair(&self, node: NodeId) -> (i64, i64) {
        let et = self.edge_type.as_deref();
        let (fetch_out, fetch_in) = match self.direction {
            TypedDegreeDirection::Outgoing => (true, false),
            TypedDegreeDirection::Incoming => (false, true),
            TypedDegreeDirection::Both | TypedDegreeDirection::Separate => (true, true),
        };
        let out = if fetch_out {
            self.store.out_degree_by_type(node, et) as i64
        } else {
            0
        };
        let inc = if fetch_in {
            self.store.in_degree_by_type(node, et) as i64
        } else {
            0
        };
        (out, inc)
    }

    /// Ranking key used for top-K selection. Outgoing / Incoming rank
    /// on their single non-zero count ; Both and Separate rank on the
    /// sum (so Separate produces the same top-K as Both, only with
    /// more output columns).
    fn rank(&self, out: i64, inc: i64) -> i64 {
        match self.direction {
            TypedDegreeDirection::Outgoing => out,
            TypedDegreeDirection::Incoming => inc,
            TypedDegreeDirection::Both | TypedDegreeDirection::Separate => out + inc,
        }
    }

    /// Stored value for the "first Int64 column" across variants.
    /// For Both this is `out + inc` (so the 2-column schema keeps the
    /// sum) ; for the single-direction variants it's the non-zero
    /// count ; for Separate it's `out` (the second column will carry
    /// `inc`).
    fn stored_first_int(&self, out: i64, inc: i64) -> i64 {
        match self.direction {
            TypedDegreeDirection::Both => out + inc,
            TypedDegreeDirection::Separate | TypedDegreeDirection::Outgoing => out,
            TypedDegreeDirection::Incoming => inc,
        }
    }

    fn compute_topk(&mut self) {
        if self.computed {
            return;
        }
        self.computed = true;

        if self.k == 0 {
            return;
        }

        // Candidate node set — restricted to the label if provided.
        let candidates: Vec<NodeId> = match &self.label {
            Some(label) => self.store.nodes_by_label(label),
            None => self.store.node_ids(),
        };

        // Fast-path : K >= |candidates| — skip the heap, materialise
        // all rows and sort by ranking.
        if candidates.len() <= self.k {
            let mut rows: Vec<(NodeId, i64, i64)> = candidates
                .into_iter()
                .map(|n| {
                    let (out, inc) = self.degree_pair(n);
                    (n, out, inc)
                })
                .collect();
            rows.sort_by(|a, b| {
                let ra = self.rank(a.1, a.2);
                let rb = self.rank(b.1, b.2);
                rb.cmp(&ra).then_with(|| a.0.0.cmp(&b.0.0))
            });
            self.results = rows;
            return;
        }

        // Min-heap keyed on `(rank, Reverse<slot>)` — higher rank
        // wins ; within ties, smaller slot wins (so the output order
        // matches the stable final sort). Wrapping the whole tuple in
        // `Reverse` turns the max-heap into a min-heap, making
        // `peek()` return the weakest candidate for eviction.
        //
        // The heap stores the full `(node, out, inc)` triple on the
        // side in a parallel Vec indexed by slot to keep the heap
        // payload tiny. Simpler alternative : store the triple in the
        // heap entry. We pick the latter for clarity — K is bounded
        // by the user and usually tiny (K ≤ 100).
        type RankKey = (i64, Reverse<u64>);
        let mut heap: BinaryHeap<Reverse<(RankKey, i64, i64)>> =
            BinaryHeap::with_capacity(self.k + 1);
        for node in &candidates {
            let (out, inc) = self.degree_pair(*node);
            let r = self.rank(out, inc);
            let key: RankKey = (r, Reverse(node.0));
            let entry = (key, out, inc);
            if heap.len() < self.k {
                heap.push(Reverse(entry));
            } else if let Some(worst) = heap.peek() {
                // `worst.0.0` is the ranking key of the weakest
                // candidate. Evict it when the new entry beats it.
                if entry.0 > worst.0.0 {
                    heap.pop();
                    heap.push(Reverse(entry));
                }
            }
        }

        // Drain + sort by ranking DESC / slot ASC for stability.
        let mut rows: Vec<(NodeId, i64, i64)> = heap
            .into_iter()
            .map(|Reverse(((_, Reverse(slot)), out, inc))| (NodeId(slot), out, inc))
            .collect();
        rows.sort_by(|a, b| {
            let ra = self.rank(a.1, a.2);
            let rb = self.rank(b.1, b.2);
            rb.cmp(&ra).then_with(|| a.0.0.cmp(&b.0.0))
        });
        self.results = rows;
    }
}

impl Operator for TypedDegreeTopKOperator {
    fn next(&mut self) -> OperatorResult {
        self.compute_topk();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        let schema = self.output_schema();
        let end = (self.position + self.chunk_capacity).min(self.results.len());
        let count = end - self.position;
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        // Column 0 : node id (always).
        {
            let node_col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with matching schema");
            for i in self.position..end {
                node_col.push_node_id(self.results[i].0);
            }
        }
        // Column 1 : first Int64 (out / in / out+in depending on
        // direction, see `stored_first_int`).
        {
            let col1 = chunk
                .column_mut(1)
                .expect("column 1 exists: chunk created with matching schema");
            for i in self.position..end {
                let (_, out, inc) = self.results[i];
                col1.push_int64(self.stored_first_int(out, inc));
            }
        }
        // Column 2 (Separate only) : in_count.
        if matches!(self.direction, TypedDegreeDirection::Separate) {
            let col2 = chunk
                .column_mut(2)
                .expect("column 2 exists: Separate schema has 3 columns");
            for i in self.position..end {
                col2.push_int64(self.results[i].2);
            }
        }
        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.results.clear();
        self.computed = false;
    }

    fn name(&self) -> &'static str {
        "TypedDegreeTopK"
    }
}

// Tests live in `crates/obrain-substrate/tests/operators_typed_degree_topk.rs`
// (mirror the W2 Class-2 pattern : obrain-core can't pull obrain-substrate as
// a dev-dep, so integration tests against a real SubstrateStore live on the
// substrate side).
