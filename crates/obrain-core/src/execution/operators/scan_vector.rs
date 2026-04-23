//! Vector similarity scan operator.
//!
//! Performs approximate nearest neighbor (ANN) search using HNSW index
//! or brute-force search for small datasets.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStore;
use crate::index::vector::DistanceMetric;
use obrain_common::types::{LogicalType, NodeId, PropertyKey, Value};
use std::sync::Arc;

#[cfg(feature = "vector-index")]
use crate::index::vector::HnswIndex;

/// A scan operator that finds nodes by vector similarity.
///
/// This operator performs k-nearest neighbor search on vector embeddings
/// stored in node properties. It can use an HNSW index for O(log n) search
/// or fall back to brute-force O(n) search.
///
/// # Output Schema
///
/// Returns a DataChunk with two columns:
/// 1. `Node` - The matched node ID
/// 2. `Float64` - The distance/similarity score
///
/// # Example
///
/// ```no_run
/// use obrain_core::execution::operators::{Operator, VectorScanOperator};
/// use obrain_core::graph::traits::GraphStore;
/// use obrain_core::index::vector::DistanceMetric;
/// use std::sync::Arc;
///
/// // Any backend implementing `GraphStore` works here — post-T17 the
/// // canonical one is `obrain_substrate::SubstrateStore`, but this doc
/// // stays in `obrain-core` and therefore uses the trait only.
/// # fn open_store() -> Arc<dyn GraphStore> { unimplemented!() }
/// # fn example() -> Result<(), obrain_core::execution::operators::OperatorError> {
/// let store = open_store();
/// let query = vec![0.1f32, 0.2, 0.3];
/// let mut scan = VectorScanOperator::brute_force(
///     store, "embedding", query, 10, DistanceMetric::Cosine,
/// );
///
/// while let Some(chunk) = scan.next()? {
///     for i in 0..chunk.row_count() {
///         let node_id = chunk.column(0).and_then(|c| c.get_node_id(i));
///         let distance = chunk.column(1).and_then(|c| c.get_float64(i));
///         println!("Node {:?} at distance {:?}", node_id, distance);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub struct VectorScanOperator {
    /// The store to fetch node properties from (for brute-force).
    store: Arc<dyn GraphStore>,
    /// The HNSW index to search (None = brute-force).
    #[cfg(feature = "vector-index")]
    index: Option<Arc<HnswIndex>>,
    /// The query vector.
    query: Vec<f32>,
    /// Number of nearest neighbors to return.
    k: usize,
    /// Distance metric (for brute-force or metric override).
    metric: DistanceMetric,
    /// Property name containing the vector (for brute-force).
    property: String,
    /// Label filter (for brute-force).
    label: Option<String>,
    /// Minimum similarity threshold (filters results).
    min_similarity: Option<f32>,
    /// Maximum distance threshold (filters results).
    max_distance: Option<f32>,
    /// Search ef parameter (higher = more accurate but slower).
    ef: usize,
    /// Cached results from search.
    results: Vec<(NodeId, f32)>,
    /// Current position in results.
    position: usize,
    /// Whether search has been executed.
    executed: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Whether using index (for name() without feature gate).
    uses_index: bool,
}

impl VectorScanOperator {
    /// Creates a new vector scan operator using an HNSW index.
    ///
    /// # Arguments
    ///
    /// * `store` - The LPG store (used for property lookup if needed)
    /// * `index` - The HNSW index to search
    /// * `query` - The query vector
    /// * `k` - Number of nearest neighbors to return
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn with_index(
        store: Arc<dyn GraphStore>,
        index: Arc<HnswIndex>,
        query: Vec<f32>,
        k: usize,
    ) -> Self {
        Self {
            store,
            index: Some(index),
            query,
            k,
            metric: DistanceMetric::Cosine,
            property: String::new(),
            label: None,
            min_similarity: None,
            max_distance: None,
            ef: 64, // Default ef for search
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
            uses_index: true,
        }
    }

    /// Sets the property name (required for HNSW index vector accessor).
    #[must_use]
    pub fn with_property(mut self, property: impl Into<String>) -> Self {
        self.property = property.into();
        self
    }

    /// Creates a new vector scan operator for brute-force search.
    ///
    /// This is suitable for small datasets (< 10K vectors) where
    /// index overhead isn't worth it.
    ///
    /// # Arguments
    ///
    /// * `store` - The LPG store to scan
    /// * `property` - The property name containing vector embeddings
    /// * `query` - The query vector
    /// * `k` - Number of nearest neighbors to return
    /// * `metric` - Distance metric to use
    #[must_use]
    pub fn brute_force(
        store: Arc<dyn GraphStore>,
        property: impl Into<String>,
        query: Vec<f32>,
        k: usize,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            store,
            #[cfg(feature = "vector-index")]
            index: None,
            query,
            k,
            metric,
            property: property.into(),
            label: None,
            min_similarity: None,
            max_distance: None,
            ef: 64,
            results: Vec::new(),
            position: 0,
            executed: false,
            chunk_capacity: 2048,
            uses_index: false,
        }
    }

    /// Sets a label filter for brute-force search.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the search ef parameter (accuracy vs speed tradeoff).
    ///
    /// Higher values give more accurate results but slower search.
    /// Default is 64. For production use, 50-200 is typical.
    #[must_use]
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Sets a minimum similarity threshold.
    ///
    /// Results with similarity below this value will be filtered out.
    /// For cosine similarity, this should be in [-1, 1].
    #[must_use]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Sets a maximum distance threshold.
    ///
    /// Results with distance above this value will be filtered out.
    #[must_use]
    pub fn with_max_distance(mut self, threshold: f32) -> Self {
        self.max_distance = Some(threshold);
        self
    }

    /// Sets the chunk capacity for output batches.
    #[must_use]
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Executes the vector search (lazily on first next() call).
    fn execute_search(&mut self) {
        if self.executed {
            return;
        }
        self.executed = true;

        #[cfg(feature = "vector-index")]
        {
            if let Some(ref index) = self.index {
                // Use HNSW index with property accessor
                let accessor = crate::index::vector::PropertyVectorAccessor::new(
                    &*self.store,
                    &*self.property,
                );
                self.results = index.search_with_ef(&self.query, self.k, self.ef, &accessor);
                self.apply_filters();
                return;
            }
        }

        // Brute-force search over node properties
        self.results = self.brute_force_search();
        self.apply_filters();
    }

    /// Performs brute-force k-NN search over node properties.
    fn brute_force_search(&self) -> Vec<(NodeId, f32)> {
        use crate::index::vector::brute_force_knn;

        // Get nodes to search (optionally filtered by label)
        let node_ids = match &self.label {
            Some(label) => self.store.nodes_by_label(label),
            None => self.store.node_ids(),
        };

        // Collect vectors from node properties
        let vectors: Vec<(NodeId, Vec<f32>)> = node_ids
            .into_iter()
            .filter_map(|id| {
                self.store
                    .get_node_property(id, &PropertyKey::new(&self.property))
                    .and_then(|v| {
                        if let Value::Vector(vec) = v {
                            Some((id, vec.to_vec()))
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Run brute-force k-NN
        let iter = vectors.iter().map(|(id, v)| (*id, v.as_slice()));
        brute_force_knn(iter, &self.query, self.k, self.metric)
    }

    /// Applies similarity/distance filters to results.
    fn apply_filters(&mut self) {
        if self.min_similarity.is_none() && self.max_distance.is_none() {
            return;
        }

        self.results.retain(|(_, distance)| {
            // For cosine metric, convert distance to similarity
            let passes_similarity = match self.min_similarity {
                Some(threshold) if self.metric == DistanceMetric::Cosine => {
                    let similarity = 1.0 - distance;
                    similarity >= threshold
                }
                Some(_) => true, // Similarity filter only applies to cosine
                None => true,
            };

            let passes_distance = match self.max_distance {
                Some(threshold) => *distance <= threshold,
                None => true,
            };

            passes_similarity && passes_distance
        });
    }
}

impl Operator for VectorScanOperator {
    fn next(&mut self) -> OperatorResult {
        // Execute search on first call
        self.execute_search();

        if self.position >= self.results.len() {
            return Ok(None);
        }

        // Create output chunk with (NodeId, distance) schema
        let schema = [LogicalType::Node, LogicalType::Float64];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.results.len());
        let count = end - self.position;

        // Fill node ID column
        {
            let node_col = chunk
                .column_mut(0)
                .ok_or_else(|| OperatorError::ColumnNotFound("node column".into()))?;

            for i in self.position..end {
                let (node_id, _) = self.results[i];
                node_col.push_node_id(node_id);
            }
        }

        // Fill distance column
        {
            let dist_col = chunk
                .column_mut(1)
                .ok_or_else(|| OperatorError::ColumnNotFound("distance column".into()))?;

            for i in self.position..end {
                let (_, distance) = self.results[i];
                dist_col.push_float64(f64::from(distance));
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.results.clear();
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        if self.uses_index {
            "VectorScan(HNSW)"
        } else {
            "VectorScan(BruteForce)"
        }
    }
}

// Tests relocated to `crates/obrain-substrate/tests/operators_scan_vector.rs`
// (T17 Step 3 W2 Class-2 migration — decision `b1dfe229`). obrain-core cannot
// take obrain-substrate as a dev-dep (dev-dep cycle, gotcha `598dda40`), so
// the LPG-era fixtures are rebuilt against SubstrateStore in an integration
// test of obrain-substrate.
