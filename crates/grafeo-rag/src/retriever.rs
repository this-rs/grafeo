//! Engram-based retriever — schema-agnostic graph retrieval via cognitive recall.
//!
//! The retriever converts a text query into graph node results through:
//! 1. **Text-to-cues**: scan graph nodes to find those whose text properties
//!    match query terms (schema-agnostic — reads all string properties).
//! 2. **Engram recall**: use the cue nodes for Hopfield spectral matching
//!    to find relevant engrams (consolidated memory traces).
//! 3. **Spreading activation**: BFS from engram nodes through synapses to
//!    discover related content with energy decay.
//! 4. **Node extraction**: extract text content from all activated nodes.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, RwLock};

use grafeo_cognitive::activation::{SpreadConfig, SynapseActivationSource, spread};
use grafeo_cognitive::engram::{
    EngramStore, RecallEngine, SpectralEncoder,
    traits::{InMemoryVectorIndex, VectorIndex},
};
use grafeo_cognitive::synapse::SynapseStore;
use grafeo_common::types::NodeId;
use grafeo_core::LpgStore;

use crate::config::RagConfig;
use crate::error::{RagError, RagResult};
use crate::traits::{RetrievalResult, RetrievalSource, RetrievedNode, Retriever};

/// Maximum distinct terms to index per node.
/// Prevents nodes with very long text from dominating the index.
const MAX_TERMS_PER_NODE: usize = 100;

/// An entry in the inverted text index: a node and its raw TF score.
/// IDF is computed at query time so it stays correct as the index evolves.
#[derive(Debug, Clone)]
struct IndexEntry {
    node_id: NodeId,
    /// Raw TF score: 1.0 + specificity (term_len / text_len).
    tf: f64,
}

/// Live inverted index with incremental update support.
///
/// Stores raw TF scores and document frequency counts. IDF is computed
/// at query time: `idf(term) = ln(total_nodes / df(term))`.
///
/// The index supports `index_node()` and `remove_node()` for incremental
/// updates without rebuilding from scratch.
#[derive(Debug)]
struct InvertedIndex {
    /// term → [(node_id, tf_score)]
    text_entries: HashMap<String, Vec<IndexEntry>>,

    /// term → number of distinct nodes containing this term (for IDF).
    term_doc_freq: HashMap<String, usize>,

    /// lowercase_label → [node_id]
    label_entries: HashMap<String, Vec<NodeId>>,

    /// label → total node count for that label.
    label_counts: HashMap<String, usize>,

    /// node → max label cardinality (for dampening lookup).
    node_label_cardinality: HashMap<NodeId, usize>,

    /// Per-node: which terms this node contributed (for removal).
    node_terms: HashMap<NodeId, Vec<String>>,

    /// Per-node: which labels this node has (for removal).
    node_labels: HashMap<NodeId, Vec<String>>,

    /// Total indexed nodes.
    total_nodes: usize,
}

impl InvertedIndex {
    fn new() -> Self {
        Self {
            text_entries: HashMap::new(),
            term_doc_freq: HashMap::new(),
            label_entries: HashMap::new(),
            label_counts: HashMap::new(),
            node_label_cardinality: HashMap::new(),
            node_terms: HashMap::new(),
            node_labels: HashMap::new(),
            total_nodes: 0,
        }
    }

    /// Add a single node to the index.
    ///
    /// If the node is already indexed, it is first removed then re-added
    /// (idempotent update).
    fn add_node(&mut self, node_id: NodeId, labels: &[String], properties: &[(String, String)]) {
        // Remove first if already present (idempotent update)
        if self.node_terms.contains_key(&node_id) {
            self.remove_node(node_id);
        }

        self.total_nodes += 1;

        // Index text properties
        let mut seen_terms: HashSet<String> = HashSet::new();
        let mut stored_terms: Vec<String> = Vec::new();

        for (_key, value) in properties {
            if value.is_empty() {
                continue;
            }
            let text_len = value.len().max(1);
            let terms = tokenize_text(&value.to_lowercase());
            for term in terms {
                if seen_terms.contains(&term) {
                    continue;
                }
                if seen_terms.len() >= MAX_TERMS_PER_NODE {
                    break;
                }
                let specificity = term.len() as f64 / text_len as f64;
                let tf = 1.0 + specificity;

                self.text_entries
                    .entry(term.clone())
                    .or_default()
                    .push(IndexEntry { node_id, tf });

                // Update doc freq: only if this is the first time this node has this term
                *self.term_doc_freq.entry(term.clone()).or_insert(0) += 1;

                seen_terms.insert(term.clone());
                stored_terms.push(term);
            }
            if seen_terms.len() >= MAX_TERMS_PER_NODE {
                break;
            }
        }

        // Index labels
        let mut stored_labels: Vec<String> = Vec::new();
        for label in labels {
            let lower = label.to_lowercase();
            self.label_entries
                .entry(lower.clone())
                .or_default()
                .push(node_id);
            *self.label_counts.entry(lower.clone()).or_insert(0) += 1;
            stored_labels.push(lower);
        }

        // Compute cardinality for this node (max label count)
        let max_card = stored_labels
            .iter()
            .filter_map(|l| self.label_counts.get(l))
            .max()
            .copied()
            .unwrap_or(1);
        self.node_label_cardinality.insert(node_id, max_card);

        // Store reverse mappings for removal
        self.node_terms.insert(node_id, stored_terms);
        self.node_labels.insert(node_id, stored_labels);
    }

    /// Remove a node from the index.
    fn remove_node(&mut self, node_id: NodeId) {
        // Remove from text index + update doc freq
        if let Some(terms) = self.node_terms.remove(&node_id) {
            for term in &terms {
                if let Some(entries) = self.text_entries.get_mut(term) {
                    entries.retain(|e| e.node_id != node_id);
                    if entries.is_empty() {
                        self.text_entries.remove(term);
                    }
                }
                if let Some(df) = self.term_doc_freq.get_mut(term) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        self.term_doc_freq.remove(term);
                    }
                }
            }
        }

        // Remove from label index + update counts
        if let Some(labels) = self.node_labels.remove(&node_id) {
            for label in &labels {
                if let Some(node_ids) = self.label_entries.get_mut(label) {
                    node_ids.retain(|id| *id != node_id);
                    if node_ids.is_empty() {
                        self.label_entries.remove(label);
                    }
                }
                if let Some(count) = self.label_counts.get_mut(label) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.label_counts.remove(label);
                    }
                }
            }
        }

        self.node_label_cardinality.remove(&node_id);
        self.total_nodes = self.total_nodes.saturating_sub(1);
    }

    /// Refresh label cardinality for all nodes.
    ///
    /// Called after bulk operations to ensure dampening values are current.
    fn refresh_cardinalities(&mut self) {
        for (node_id, labels) in &self.node_labels {
            let max_card = labels
                .iter()
                .filter_map(|l| self.label_counts.get(l))
                .max()
                .copied()
                .unwrap_or(1);
            self.node_label_cardinality.insert(*node_id, max_card);
        }
    }

    /// Query the index: return scored nodes for given terms.
    ///
    /// Computes IDF on-the-fly from current doc freq counts, applies label
    /// cardinality dampening, and returns the top `max_cues` results.
    fn query(&self, terms: &[String], max_cues: usize) -> Vec<(NodeId, f64)> {
        if terms.is_empty() || self.total_nodes == 0 {
            return Vec::new();
        }

        let n = self.total_nodes as f64;
        let mut node_scores: HashMap<NodeId, f64> = HashMap::new();

        for term in terms {
            // Text index lookup with on-the-fly IDF
            if let Some(entries) = self.text_entries.get(term.as_str()) {
                let df = self
                    .term_doc_freq
                    .get(term.as_str())
                    .copied()
                    .unwrap_or(1)
                    .max(1) as f64;
                let idf = (n / df).ln().max(0.1);

                for entry in entries {
                    *node_scores.entry(entry.node_id).or_default() += entry.tf * idf;
                }
            }

            // Label matching (bidirectional: term in label OR label in term)
            for (label, node_ids) in &self.label_entries {
                if label.contains(term.as_str()) || term.contains(label.as_str()) {
                    for node_id in node_ids {
                        *node_scores.entry(*node_id).or_default() += 0.5;
                    }
                }
            }
        }

        // Apply label cardinality dampening
        let mut scored: Vec<(NodeId, f64)> = node_scores
            .into_iter()
            .map(|(node_id, raw_score)| {
                let cardinality = self
                    .node_label_cardinality
                    .get(&node_id)
                    .copied()
                    .unwrap_or(1) as f64;
                let label_fraction = cardinality / n;
                let dampening = (1.0 + label_fraction * 10.0).ln().max(1.0);
                (node_id, raw_score / dampening)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_cues);
        scored
    }

    /// Return diagnostics: (total_nodes, distinct_terms, label_distribution).
    fn stats(&self) -> (usize, usize, Vec<(String, usize, f64)>) {
        let total = self.total_nodes.max(1);
        let terms = self.text_entries.len();
        let mut labels: Vec<(String, usize, f64)> = self
            .label_counts
            .iter()
            .map(|(label, &count)| {
                let frac = count as f64 / total as f64 * 100.0;
                (label.clone(), count, frac)
            })
            .collect();
        labels.sort_by(|a, b| b.1.cmp(&a.1));
        (total, terms, labels)
    }
}

// ─────────────────────────────────────────────────────────────────
// EngramRetriever
// ─────────────────────────────────────────────────────────────────

/// Schema-agnostic retriever backed by Grafeo's cognitive layer.
///
/// Uses engrams as the abstraction layer so it doesn't need to know
/// the database schema. Any GrafeoDB with a cognitive engine will work.
///
/// The internal inverted index is behind a `RwLock` and supports
/// incremental updates via `index_node()` / `remove_node()`.
/// A server can subscribe to post-commit mutations and call these
/// methods to keep the index always up-to-date.
pub struct EngramRetriever {
    /// The graph store for reading nodes/properties.
    graph: Arc<LpgStore>,

    /// The engram store (cognitive memory traces).
    engram_store: Arc<EngramStore>,

    /// Vector index for spectral similarity search.
    vector_index: Arc<dyn VectorIndex>,

    /// Spectral encoder for query vectors.
    spectral: Arc<SpectralEncoder>,

    /// Synapse store for spreading activation.
    synapse_store: Option<Arc<SynapseStore>>,

    /// Live inverted index (RwLock for concurrent read + incremental write).
    index: RwLock<InvertedIndex>,
}

impl EngramRetriever {
    /// Create a new retriever from cognitive components.
    ///
    /// Builds the inverted text index at construction time by scanning
    /// all graph nodes. For large graphs, consider `new_lazy()` +
    /// incremental `index_node()` calls instead.
    pub fn new(
        graph: Arc<LpgStore>,
        engram_store: Arc<EngramStore>,
        vector_index: Arc<dyn VectorIndex>,
        spectral: Arc<SpectralEncoder>,
        synapse_store: Option<Arc<SynapseStore>>,
    ) -> Self {
        let index = Self::build_full_index(&graph);
        Self {
            graph,
            engram_store,
            vector_index,
            spectral,
            synapse_store,
            index: RwLock::new(index),
        }
    }

    /// Create a retriever with default spectral encoder and in-memory vector index.
    ///
    /// This is the simplest setup — useful for testing or when the cognitive
    /// engine hasn't been fully initialized.
    pub fn with_defaults(
        graph: Arc<LpgStore>,
        engram_store: Arc<EngramStore>,
        synapse_store: Option<Arc<SynapseStore>>,
    ) -> Self {
        let index = Self::build_full_index(&graph);
        Self {
            graph,
            engram_store,
            vector_index: Arc::new(InMemoryVectorIndex::new()),
            spectral: Arc::new(SpectralEncoder::new()),
            synapse_store,
            index: RwLock::new(index),
        }
    }

    /// Create a retriever with an **empty** index.
    ///
    /// No initial scan — the caller is responsible for populating the
    /// index via `index_node()` calls (e.g. from a mutation stream).
    /// This is the preferred constructor for server integration where
    /// the index is built incrementally.
    pub fn new_lazy(
        graph: Arc<LpgStore>,
        engram_store: Arc<EngramStore>,
        synapse_store: Option<Arc<SynapseStore>>,
    ) -> Self {
        Self {
            graph,
            engram_store,
            vector_index: Arc::new(InMemoryVectorIndex::new()),
            spectral: Arc::new(SpectralEncoder::new()),
            synapse_store,
            index: RwLock::new(InvertedIndex::new()),
        }
    }

    /// Build the full index by scanning all graph nodes.
    fn build_full_index(graph: &LpgStore) -> InvertedIndex {
        let node_ids = graph.node_ids();
        let mut index = InvertedIndex::new();

        for node_id in &node_ids {
            if let Some(node) = graph.get_node(*node_id) {
                let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();
                let properties: Vec<(String, String)> = node
                    .properties
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string())))
                    .collect();
                index.add_node(*node_id, &labels, &properties);
            }
        }

        // After bulk load, refresh all cardinalities at once
        // (individual add_node only sees partial label_counts)
        index.refresh_cardinalities();

        index
    }

    // ── Incremental update API ──────────────────────────────────

    /// Index a single node (by reading it from the graph store).
    ///
    /// If the node is already indexed, it is updated in-place.
    /// Call this from a post-commit hook or mutation listener to keep
    /// the index in sync with the graph.
    pub fn index_node(&self, node_id: NodeId) {
        if let Some(node) = self.graph.get_node(node_id) {
            let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();
            let properties: Vec<(String, String)> = node
                .properties
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string())))
                .collect();

            let mut idx = self.index.write().unwrap();
            idx.add_node(node_id, &labels, &properties);
        }
    }

    /// Remove a node from the index.
    ///
    /// Call this when a node is deleted from the graph.
    pub fn remove_node(&self, node_id: NodeId) {
        let mut idx = self.index.write().unwrap();
        idx.remove_node(node_id);
    }

    /// Index multiple nodes at once (batch update).
    ///
    /// More efficient than calling `index_node()` in a loop because
    /// it holds the write lock for the entire batch and refreshes
    /// cardinalities once at the end.
    pub fn index_nodes(&self, node_ids: &[NodeId]) {
        let mut idx = self.index.write().unwrap();
        for &node_id in node_ids {
            if let Some(node) = self.graph.get_node(node_id) {
                let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();
                let properties: Vec<(String, String)> = node
                    .properties
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.to_string(), s.to_string())))
                    .collect();
                idx.add_node(node_id, &labels, &properties);
            }
        }
        idx.refresh_cardinalities();
    }

    /// Rebuild the entire index from the current graph state.
    ///
    /// Useful as a `/reindex` command or after major bulk operations.
    pub fn reindex(&self) {
        let new_index = Self::build_full_index(&self.graph);
        let mut idx = self.index.write().unwrap();
        *idx = new_index;
    }

    // ── Diagnostics ─────────────────────────────────────────────

    /// Return index statistics for diagnostics.
    ///
    /// Returns `(total_nodes, distinct_terms, label_distribution)` where
    /// `label_distribution` is a sorted vec of `(label, count, fraction%)`.
    pub fn index_stats(&self) -> (usize, usize, Vec<(String, usize, f64)>) {
        let idx = self.index.read().unwrap();
        idx.stats()
    }

    // ── Internal helpers ────────────────────────────────────────

    /// Convert a text query into cue NodeIds using the inverted index.
    fn text_to_cues(&self, query: &str, max_cues: usize) -> Vec<(NodeId, f64)> {
        let terms = tokenize_query(query);
        let idx = self.index.read().unwrap();
        idx.query(&terms, max_cues)
    }

    /// Extract text content from a node in a schema-agnostic way.
    fn extract_node_content(&self, node_id: NodeId) -> Option<RetrievedNode> {
        let node = self.graph.get_node(node_id)?;

        let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();

        let mut properties = HashMap::new();
        for (key, value) in node.properties.iter() {
            if let Some(s) = value.as_str() {
                if !s.is_empty() {
                    properties.insert(key.as_str().to_string(), s.to_string());
                }
            }
        }

        // Get outgoing relations
        let outgoing: Vec<(String, NodeId)> = self
            .graph
            .edges_from(node_id, grafeo_core::graph::Direction::Outgoing)
            .into_iter()
            .filter_map(|(target, edge_id)| {
                let edge = self.graph.get_edge(edge_id)?;
                Some((edge.edge_type.to_string(), target))
            })
            .collect();

        // Get incoming relations
        let incoming: Vec<(String, NodeId)> = self
            .graph
            .edges_from(node_id, grafeo_core::graph::Direction::Incoming)
            .into_iter()
            .filter_map(|(source, edge_id)| {
                let edge = self.graph.get_edge(edge_id)?;
                Some((edge.edge_type.to_string(), source))
            })
            .collect();

        Some(RetrievedNode {
            node_id,
            labels,
            properties,
            score: 0.0, // Will be set by caller
            source: RetrievalSource::SpreadingActivation {
                depth: 0,
                activation: 0.0,
            },
            outgoing_relations: outgoing,
            incoming_relations: incoming,
        })
    }
}

impl Retriever for EngramRetriever {
    fn retrieve(&self, query: &str, config: &RagConfig) -> RagResult<RetrievalResult> {
        // Step 1: Convert text query to cue nodes
        let cue_nodes = self.text_to_cues(query, config.max_engrams * 5);
        let cue_node_ids: Vec<NodeId> = cue_nodes.iter().map(|(id, _)| *id).collect();

        // Step 2: Engram recall via Hopfield
        let recall_engine = RecallEngine::new();
        let recall_results = recall_engine.recall(
            &self.engram_store,
            &cue_node_ids,
            self.vector_index.as_ref(),
            &self.spectral,
            config.max_engrams,
        );

        let engrams_matched = recall_results.len();

        // Collect all node IDs from recalled engrams
        let mut activated_nodes: HashMap<NodeId, (f64, RetrievalSource)> = HashMap::new();

        for result in &recall_results {
            for &(node_id, weight) in &result.engram.ensemble {
                let score = result.confidence * weight;
                let source = RetrievalSource::EngramRecall {
                    engram_id: result.engram_id.0,
                    confidence: result.confidence,
                };
                let entry = activated_nodes.entry(node_id).or_insert((0.0, source.clone()));
                if score > entry.0 {
                    *entry = (score, source);
                }
            }
        }

        // Step 3: Spreading activation from engram nodes through synapses
        if let Some(ref synapse_store) = self.synapse_store {
            let source_activation = SynapseActivationSource::new(Arc::clone(synapse_store));
            let spread_config = SpreadConfig::default()
                .with_max_hops(config.activation_depth)
                .with_decay_factor(config.activation_decay)
                .with_min_energy(config.min_activation_energy)
                .with_max_activated_nodes(config.max_activated_nodes);

            let sources: Vec<(NodeId, f64)> = activated_nodes
                .iter()
                .map(|(id, (score, _))| (*id, *score))
                .collect();

            if !sources.is_empty() {
                let activation_map = spread(&sources, &source_activation, &spread_config);

                for (node_id, activation) in &activation_map {
                    if !activated_nodes.contains_key(node_id)
                        && *activation >= config.min_activation_energy
                    {
                        activated_nodes.insert(
                            *node_id,
                            (
                                *activation,
                                RetrievalSource::SpreadingActivation {
                                    depth: 1,
                                    activation: *activation,
                                },
                            ),
                        );
                    }
                }
            }
        }

        // Also add cue nodes themselves (direct text matches)
        for (node_id, text_score) in &cue_nodes {
            if !activated_nodes.contains_key(node_id) {
                activated_nodes.insert(
                    *node_id,
                    (
                        *text_score * 0.5,
                        RetrievalSource::SpreadingActivation {
                            depth: 0,
                            activation: *text_score,
                        },
                    ),
                );
            }
        }

        let nodes_activated = activated_nodes.len();

        // Step 4: Extract content from all activated nodes
        let mut nodes: Vec<RetrievedNode> = Vec::new();
        let seen: HashSet<NodeId> = HashSet::new();

        for (node_id, (score, source)) in &activated_nodes {
            if seen.contains(node_id) {
                continue;
            }
            if let Some(mut retrieved) = self.extract_node_content(*node_id) {
                retrieved.score = *score;
                retrieved.source = source.clone();
                nodes.push(retrieved);
            }
        }

        // Sort by score descending
        nodes.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max_context_nodes
        nodes.truncate(config.max_context_nodes);

        if nodes.is_empty() && engrams_matched == 0 {
            if cue_nodes.is_empty() {
                return Err(RagError::NoEngramsFound(query.to_string()));
            }
        }

        Ok(RetrievalResult {
            nodes,
            engrams_matched,
            nodes_activated,
        })
    }
}

// ─────────────────────────────────────────────────────────────────
// Tokenization
// ─────────────────────────────────────────────────────────────────

/// Static stop word set — built once, reused across all queries.
static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "each", "few", "more", "most", "other", "some", "such", "no",
        "than", "too", "very", "just", "about", "also", "this", "that",
        "these", "those", "it", "its", "my", "your", "his", "her", "our",
        "their", "what", "which", "who", "whom", "where", "when", "why", "how",
        // French stop words
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "est",
        "en", "que", "qui", "dans", "pour", "par", "sur", "avec", "ce",
        "se", "son", "sa", "ses", "au", "aux", "ne", "pas", "plus",
        "sont", "ont", "fait", "être", "avoir", "il", "elle", "nous",
        "vous", "ils", "elles", "je", "tu", "on", "me", "te", "lui",
        "leur", "y", "si", "ou", "mais", "donc", "car", "ni",
        "quels", "quelles", "quel", "quelle", "comment", "combien",
    ]
    .into_iter()
    .collect()
});

/// Tokenize a text string into lowercase normalized terms.
///
/// Filters out common stop words (EN+FR) and very short terms (< 2 chars).
fn tokenize_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2 && !STOP_WORDS.contains(w))
        .map(String::from)
        .collect()
}

/// Tokenize a query string into lowercase search terms.
fn tokenize_query(query: &str) -> Vec<String> {
    tokenize_text(query)
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_index_add_and_query() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[
                ("name".into(), "Grafeo".into()),
                ("description".into(), "Graph database engine".into()),
            ],
        );
        idx.add_node(
            NodeId(2),
            &["Note".to_string()],
            &[("title".into(), "WAL bug in Grafeo".into())],
        );

        let results = idx.query(&tokenize_query("Grafeo"), 10);
        assert!(!results.is_empty(), "Should find nodes matching 'Grafeo'");
        // Both nodes mention "grafeo"
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn inverted_index_remove_node() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Grafeo".into())],
        );
        idx.add_node(
            NodeId(2),
            &["Note".to_string()],
            &[("title".into(), "Other thing".into())],
        );
        assert_eq!(idx.total_nodes, 2);

        idx.remove_node(NodeId(1));
        assert_eq!(idx.total_nodes, 1);

        let results = idx.query(&tokenize_query("Grafeo"), 10);
        assert!(results.is_empty(), "Removed node should not appear");
    }

    #[test]
    fn inverted_index_update_is_idempotent() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Grafeo".into())],
        );
        // Re-add same node with different content
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Grafeo v2".into())],
        );
        assert_eq!(idx.total_nodes, 1, "Should still be 1 node after update");

        let results = idx.query(&tokenize_query("Grafeo"), 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn label_dampening_reduces_dominant_labels() {
        let mut idx = InvertedIndex::new();

        // Add 100 ChatMessage nodes with "hello"
        for i in 0..100 {
            idx.add_node(
                NodeId(i),
                &["ChatMessage".to_string()],
                &[("content".into(), "hello world".into())],
            );
        }
        // Add 1 Note node with "hello"
        idx.add_node(
            NodeId(200),
            &["Note".to_string()],
            &[("content".into(), "hello important note".into())],
        );
        idx.refresh_cardinalities();

        let results = idx.query(&tokenize_query("hello"), 5);
        assert!(!results.is_empty());

        // The Note node should rank higher than any ChatMessage
        // because ChatMessage has 100/101 ≈ 99% cardinality → heavy dampening
        let top_node = results[0].0;
        assert_eq!(
            top_node,
            NodeId(200),
            "Note should rank above ChatMessages due to label dampening"
        );
    }

    #[test]
    fn idf_dampens_common_terms() {
        let n = 100.0_f64;
        let idf_rare = (n / 1.0).ln();
        let idf_common = (n / 90.0).ln();
        assert!(idf_rare > idf_common * 10.0, "Rare terms should score much higher");
    }

    #[test]
    fn label_dampening_formula() {
        let frac_90: f64 = 0.9;
        let dampening_90 = (1.0 + frac_90 * 10.0).ln().max(1.0);
        let frac_01: f64 = 0.01;
        let dampening_01 = (1.0 + frac_01 * 10.0).ln().max(1.0);

        assert!(dampening_90 > 2.0, "90% label should be heavily dampened: {}", dampening_90);
        assert!(dampening_01 < 1.2, "1% label should barely be dampened: {}", dampening_01);
        assert!(10.0 / dampening_90 < 10.0 / dampening_01);
    }

    #[test]
    fn index_stats_returns_distribution() {
        let mut idx = InvertedIndex::new();
        idx.add_node(NodeId(1), &["Project".into()], &[("name".into(), "A".into())]);
        idx.add_node(NodeId(2), &["Project".into()], &[("name".into(), "B".into())]);
        idx.add_node(NodeId(3), &["Note".into()], &[("title".into(), "C".into())]);

        let (total, _terms, labels) = idx.stats();
        assert_eq!(total, 3);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].0, "project"); // Most common first
        assert_eq!(labels[0].1, 2);
    }

    #[test]
    fn tokenize_filters_stopwords() {
        let terms = tokenize_query("What are the projects in this database?");
        assert!(!terms.contains(&"what".to_string()));
        assert!(!terms.contains(&"are".to_string()));
        assert!(!terms.contains(&"the".to_string()));
        assert!(terms.contains(&"projects".to_string()));
        assert!(terms.contains(&"database".to_string()));
    }

    #[test]
    fn tokenize_french() {
        let terms = tokenize_query("Quels sont les projets qui utilisent des plans?");
        assert!(!terms.contains(&"quels".to_string()));
        assert!(!terms.contains(&"sont".to_string()));
        assert!(!terms.contains(&"les".to_string()));
        assert!(terms.contains(&"projets".to_string()));
        assert!(terms.contains(&"utilisent".to_string()));
        assert!(terms.contains(&"plans".to_string()));
    }

    #[test]
    fn tokenize_short_words_filtered() {
        let terms = tokenize_query("a b cd ef");
        assert!(!terms.contains(&"a".to_string()));
        assert!(!terms.contains(&"b".to_string()));
        assert!(terms.contains(&"cd".to_string()));
        assert!(terms.contains(&"ef".to_string()));
    }
}
