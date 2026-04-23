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

/// Direction of the degree sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypedDegreeDirection {
    /// Count only outgoing edges (`(n)-[:TYPE]->()`).
    Outgoing,
    /// Count only incoming edges (`()-[:TYPE]->(n)`).
    Incoming,
    /// Sum outgoing + incoming (`both`).
    Both,
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
    /// Computed top-K rows, ordered by degree DESC. Populated lazily
    /// on the first `next()` call.
    results: Vec<(NodeId, i64)>,
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

    /// Returns the Operator's output schema. Always two columns :
    /// `[Node, Int64]`.
    #[must_use]
    pub fn output_schema() -> [LogicalType; 2] {
        [LogicalType::Node, LogicalType::Int64]
    }

    fn degree(&self, node: NodeId) -> i64 {
        let et = self.edge_type.as_deref();
        let out = matches!(
            self.direction,
            TypedDegreeDirection::Outgoing | TypedDegreeDirection::Both
        );
        let inc = matches!(
            self.direction,
            TypedDegreeDirection::Incoming | TypedDegreeDirection::Both
        );
        let mut d: i64 = 0;
        if out {
            d += self.store.out_degree_by_type(node, et) as i64;
        }
        if inc {
            d += self.store.in_degree_by_type(node, et) as i64;
        }
        d
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

        // Min-heap of size K keyed by degree (ties broken by slot id
        // DESC so the iteration order is stable across invocations on
        // the same backend). `Reverse<(degree, slot)>` puts the
        // smallest at the top of the max-heap, which is what we pop
        // when evicting.
        //
        // Fast-path : K >= |candidates| — skip the heap, sort the full
        // vec.
        if candidates.len() <= self.k {
            let mut rows: Vec<(NodeId, i64)> = candidates
                .into_iter()
                .map(|n| (n, self.degree(n)))
                .collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0 .0.cmp(&b.0 .0)));
            self.results = rows;
            return;
        }

        // Ranking : `(degree, Reverse<slot>)` — higher is better. Higher
        // degree wins ; within ties, the smaller slot wins (so the
        // output order matches the stable final sort). Wrapping the
        // whole tuple in `Reverse` turns the max-heap into a min-heap
        // on this ranking, so `peek()` is the weakest candidate.
        type Rank = (i64, Reverse<u64>);
        let mut heap: BinaryHeap<Reverse<Rank>> = BinaryHeap::with_capacity(self.k + 1);
        for node in &candidates {
            let d = self.degree(*node);
            let entry: Rank = (d, Reverse(node.0));
            if heap.len() < self.k {
                heap.push(Reverse(entry));
            } else if let Some(worst) = heap.peek() {
                // `worst.0` is the weakest candidate in the heap.
                // Evict it when the new entry beats it under the
                // ranking (degree DESC ; slot ASC on ties).
                if entry > worst.0 {
                    heap.pop();
                    heap.push(Reverse(entry));
                }
            }
        }

        // Drain into a vec sorted by degree DESC, then by slot ASC for
        // stability (same order the heap ranking selected for).
        let mut rows: Vec<(NodeId, i64)> = heap
            .into_iter()
            .map(|Reverse((d, Reverse(slot)))| (NodeId(slot), d))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0 .0.cmp(&b.0 .0)));
        self.results = rows;
    }
}

impl Operator for TypedDegreeTopKOperator {
    fn next(&mut self) -> OperatorResult {
        self.compute_topk();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        let schema = Self::output_schema();
        let end = (self.position + self.chunk_capacity).min(self.results.len());
        let count = end - self.position;
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        {
            let node_col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with matching schema");
            for i in self.position..end {
                node_col.push_node_id(self.results[i].0);
            }
        }
        {
            let deg_col = chunk
                .column_mut(1)
                .expect("column 1 exists: chunk created with matching schema");
            for i in self.position..end {
                deg_col.push_int64(self.results[i].1);
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
