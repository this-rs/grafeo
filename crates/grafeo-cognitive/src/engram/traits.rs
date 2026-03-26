//! Cognitive trait contract — the 5 traits that define the boundary between
//! the graph engine and the cognitive layer.
//!
//! These traits follow Option C architecture: `grafeo-cognitive` is an independent
//! crate that depends *only* on these traits, not on the engine internals.
//! The engine provides implementations via wrapper types.

use grafeo_common::types::{EdgeId, NodeId, Value};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Trait 1: QueryObserver — hooks into query execution
// ---------------------------------------------------------------------------

/// Observer for query execution events.
///
/// MutationObserver is already covered by `grafeo_reactive::MutationListener`.
/// This trait captures the *read* side: what queries are executed, what paths
/// are traversed, enabling the cognitive layer to track query patterns.
pub trait QueryObserver: Send + Sync {
    /// Called after a query is executed with its result statistics.
    fn on_query_executed(&self, query_text: &str, result_count: usize, duration: Duration);

    /// Called when an edge is traversed during query execution.
    fn on_traversal(&self, path: &[EdgeId]);
}

// ---------------------------------------------------------------------------
// Trait 2: EdgeAnnotator — float annotations on edges (pheromones, weights)
// ---------------------------------------------------------------------------

/// Annotate edges with float64 values without modifying user-visible properties.
///
/// Used by stigmergy (pheromones) and other layers that need to store
/// metadata on edges without polluting the user schema.
pub trait EdgeAnnotator: Send + Sync {
    /// Set a cognitive annotation on an edge.
    fn annotate(&self, edge: EdgeId, key: &str, value: f64);

    /// Read a cognitive annotation from an edge.
    fn get_annotation(&self, edge: EdgeId, key: &str) -> Option<f64>;

    /// Remove a cognitive annotation from an edge.
    fn remove_annotation(&self, edge: EdgeId, key: &str);
}

// ---------------------------------------------------------------------------
// Trait 3: VectorIndex — nearest-neighbor search for spectral signatures
// ---------------------------------------------------------------------------

/// Vector index for spectral signature similarity search.
///
/// The engram system uses spectral signatures (float vectors) to represent
/// engram patterns. This trait abstracts the nearest-neighbor search engine
/// (could be HNSW, brute-force, or the engine's built-in vector index).
pub trait VectorIndex: Send + Sync {
    /// Insert or update a vector with the given identifier.
    fn upsert(&self, id: &str, vector: &[f64]);

    /// Find the k nearest neighbors to the query vector.
    /// Returns (id, distance) pairs sorted by ascending distance.
    fn nearest(&self, query: &[f64], k: usize) -> Vec<(String, f64)>;

    /// Remove a vector by identifier.
    fn remove(&self, id: &str);

    /// Returns the dimensionality of stored vectors (0 if empty).
    fn dimensions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Trait 4: CognitiveStorage — CRUD for cognitive-internal nodes/edges
// ---------------------------------------------------------------------------

/// Storage interface for cognitive-internal graph entities.
///
/// Engrams, immune detectors, and other cognitive structures are stored as
/// graph nodes with special labels (`:Engram`, `:ImmuneDetector`, etc.).
/// This trait abstracts the storage layer so the cognitive crate doesn't
/// depend on engine internals.
pub trait CognitiveStorage: Send + Sync {
    /// Create a cognitive node with the given label and properties.
    fn create_node(&self, label: &str, properties: &HashMap<String, Value>) -> NodeId;

    /// Create a cognitive edge between two nodes.
    fn create_edge(
        &self,
        from: NodeId,
        to: NodeId,
        rel_type: &str,
        properties: &HashMap<String, Value>,
    ) -> EdgeId;

    /// Query cognitive nodes by label and optional property filter.
    fn query_nodes(&self, label: &str, filter: Option<&CognitiveFilter>) -> Vec<CognitiveNode>;

    /// Update properties on a cognitive node.
    fn update_node(&self, id: NodeId, properties: &HashMap<String, Value>);

    /// Delete a cognitive node and all its edges.
    fn delete_node(&self, id: NodeId);

    /// Delete a cognitive edge.
    fn delete_edge(&self, id: EdgeId);

    /// Query edges of a specific type from a source node.
    fn query_edges(&self, from: NodeId, rel_type: &str) -> Vec<CognitiveEdge>;
}

/// A filter for cognitive node queries.
#[derive(Debug, Clone)]
pub enum CognitiveFilter {
    /// Property equals a specific value.
    PropertyEquals(String, Value),
    /// Property greater than a float value.
    PropertyGt(String, f64),
    /// Property less than a float value.
    PropertyLt(String, f64),
    /// Compound AND filter.
    And(Vec<CognitiveFilter>),
}

/// A cognitive node returned from storage queries.
#[derive(Debug, Clone)]
pub struct CognitiveNode {
    /// The node identifier.
    pub id: NodeId,
    /// The node's label.
    pub label: String,
    /// The node's properties.
    pub properties: HashMap<String, Value>,
}

/// A cognitive edge returned from storage queries.
#[derive(Debug, Clone)]
pub struct CognitiveEdge {
    /// The edge identifier.
    pub id: EdgeId,
    /// Source node.
    pub from: NodeId,
    /// Target node.
    pub to: NodeId,
    /// Relationship type.
    pub rel_type: String,
    /// Edge properties.
    pub properties: HashMap<String, Value>,
}

// ---------------------------------------------------------------------------
// Trait 5: CognitiveObservability — feedback loop for mark_used/mark_rejected
// ---------------------------------------------------------------------------

/// Observability trait for feedback on cognitive suggestions.
///
/// When the cognitive layer proposes engrams during warm-up, the consumer
/// (e.g., the PO) can mark them as used or rejected, creating a feedback
/// loop that improves future suggestions.
pub trait CognitiveObservability: Send + Sync {
    /// Mark an engram as successfully used (positive feedback).
    fn mark_used(&self, engram_id: &str, context: &str);

    /// Mark an engram as rejected (negative feedback).
    fn mark_rejected(&self, engram_id: &str, reason: &str);
}

// ---------------------------------------------------------------------------
// In-memory implementations (for testing and standalone usage)
// ---------------------------------------------------------------------------

/// In-memory vector index using brute-force nearest-neighbor search.
#[derive(Debug, Default)]
pub struct InMemoryVectorIndex {
    vectors: parking_lot::RwLock<HashMap<String, Vec<f64>>>,
}

impl InMemoryVectorIndex {
    /// Creates a new empty in-memory vector index.
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorIndex for InMemoryVectorIndex {
    fn upsert(&self, id: &str, vector: &[f64]) {
        self.vectors.write().insert(id.to_string(), vector.to_vec());
    }

    fn nearest(&self, query: &[f64], k: usize) -> Vec<(String, f64)> {
        let vectors = self.vectors.read();
        let mut distances: Vec<(String, f64)> = vectors
            .iter()
            .map(|(id, vec)| {
                let dist = cosine_distance(query, vec);
                (id.clone(), dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    fn remove(&self, id: &str) {
        self.vectors.write().remove(id);
    }

    fn dimensions(&self) -> usize {
        self.vectors.read().values().next().map_or(0, |v| v.len())
    }
}

/// Cosine distance between two vectors (1 - cosine_similarity).
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

/// No-op query observer for contexts where query tracking is not needed.
#[derive(Debug, Default)]
pub struct NoopQueryObserver;

impl QueryObserver for NoopQueryObserver {
    fn on_query_executed(&self, _query_text: &str, _result_count: usize, _duration: Duration) {}
    fn on_traversal(&self, _path: &[EdgeId]) {}
}
