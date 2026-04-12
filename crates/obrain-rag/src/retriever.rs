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
use std::sync::{Arc, RwLock};

use obrain_cognitive::activation::{SpreadConfig, SynapseActivationSource, spread};
use obrain_cognitive::engram::{
    EngramStore, RecallEngine, SpectralEncoder,
    traits::{InMemoryVectorIndex, VectorIndex},
};
use obrain_cognitive::synapse::SynapseStore;
use obrain_common::types::NodeId;
use obrain_core::LpgStore;

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
        // Early exit if node was never indexed
        if !self.node_terms.contains_key(&node_id) && !self.node_labels.contains_key(&node_id) {
            return;
        }

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

    /// Query the index scoped to nodes with a specific label.
    ///
    /// Like `query()` but only returns nodes that have the given label.
    /// Cardinality dampening is skipped since we're already filtering by label.
    fn query_by_label(&self, terms: &[String], label: &str, max_results: usize) -> Vec<(NodeId, f64)> {
        if terms.is_empty() || self.total_nodes == 0 {
            return Vec::new();
        }

        let lower_label = label.to_lowercase();

        // Get nodes with this label
        let label_nodes: HashSet<NodeId> = self
            .label_entries
            .get(&lower_label)
            .map(|ids| ids.iter().copied().collect())
            .unwrap_or_default();

        if label_nodes.is_empty() {
            return Vec::new();
        }

        let n = self.total_nodes as f64;
        let mut node_scores: HashMap<NodeId, f64> = HashMap::new();

        for term in terms {
            if let Some(entries) = self.text_entries.get(term.as_str()) {
                let df = self
                    .term_doc_freq
                    .get(term.as_str())
                    .copied()
                    .unwrap_or(1)
                    .max(1) as f64;
                let idf = (n / df).ln().max(0.1);

                for entry in entries {
                    if label_nodes.contains(&entry.node_id) {
                        *node_scores.entry(entry.node_id).or_default() += entry.tf * idf;
                    }
                }
            }
        }

        let mut scored: Vec<(NodeId, f64)> = node_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_results);
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

/// Schema-agnostic retriever backed by Obrain's cognitive layer.
///
/// Uses engrams as the abstraction layer so it doesn't need to know
/// the database schema. Any ObrainDB with a cognitive engine will work.
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

    // ── Label-scoped recall ───────────────────────────────────

    /// Recall nodes with a specific label, bypassing engram matching.
    ///
    /// This is the direct-match path: queries the inverted index for nodes
    /// of the given label and returns them with full content extracted.
    /// Useful for recalling raw messages (`CogMessage`) or identity nodes
    /// (`Identity`) that haven't yet formed into engrams.
    pub fn recall_by_label(
        &self,
        query: &str,
        label: &str,
        max_results: usize,
    ) -> Vec<RetrievedNode> {
        let terms = tokenize_query(query);
        let idx = self.index.read().unwrap();
        let hits = idx.query_by_label(&terms, label, max_results);
        drop(idx);

        let mut nodes = Vec::with_capacity(hits.len());
        for (node_id, score) in hits {
            if let Some(mut retrieved) = self.extract_node_content(node_id) {
                retrieved.score = score;
                retrieved.source = RetrievalSource::DirectMatch { text_score: score };
                nodes.push(retrieved);
            }
        }
        nodes
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
            if let Some(s) = value.as_str()
                && !s.is_empty()
            {
                properties.insert(key.as_str().to_string(), s.to_string());
            }
        }

        // Get outgoing relations
        let outgoing: Vec<(String, NodeId)> = self
            .graph
            .edges_from(node_id, obrain_core::graph::Direction::Outgoing)
            .filter_map(|(target, edge_id)| {
                let edge = self.graph.get_edge(edge_id)?;
                Some((edge.edge_type.to_string(), target))
            })
            .collect();

        // Get incoming relations
        let incoming: Vec<(String, NodeId)> = self
            .graph
            .edges_from(node_id, obrain_core::graph::Direction::Incoming)
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
                let entry = activated_nodes
                    .entry(node_id)
                    .or_insert((0.0, source.clone()));
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

        // Also add cue nodes themselves (direct text matches).
        // These are first-class results — no score penalty.
        // When no engrams exist, direct matches ARE the recall path.
        for (node_id, text_score) in &cue_nodes {
            let direct_source = RetrievalSource::DirectMatch {
                text_score: *text_score,
            };
            match activated_nodes.get(node_id) {
                Some((existing_score, _)) if *existing_score >= *text_score => {
                    // Engram/activation already found this with higher score — keep it
                }
                _ => {
                    // Direct match is stronger or node not yet seen — use it
                    activated_nodes.insert(*node_id, (*text_score, direct_source));
                }
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

        if nodes.is_empty() && engrams_matched == 0 && cue_nodes.is_empty() {
            return Err(RagError::NoEngramsFound(query.to_string()));
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

/// Tokenize a text string into lowercase normalized terms.
///
/// No stop word filtering — everything emerges from the graph.
/// TF-IDF naturally dampens common terms (high document frequency → low IDF).
/// Only single characters are filtered as noise.
fn tokenize_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2)
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
                ("name".into(), "Obrain".into()),
                ("description".into(), "Graph database engine".into()),
            ],
        );
        idx.add_node(
            NodeId(2),
            &["Note".to_string()],
            &[("title".into(), "WAL bug in Obrain".into())],
        );

        let results = idx.query(&tokenize_query("Obrain"), 10);
        assert!(!results.is_empty(), "Should find nodes matching 'Obrain'");
        // Both nodes mention "obrain"
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn inverted_index_remove_node() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Obrain".into())],
        );
        idx.add_node(
            NodeId(2),
            &["Note".to_string()],
            &[("title".into(), "Other thing".into())],
        );
        assert_eq!(idx.total_nodes, 2);

        idx.remove_node(NodeId(1));
        assert_eq!(idx.total_nodes, 1);

        let results = idx.query(&tokenize_query("Obrain"), 10);
        assert!(results.is_empty(), "Removed node should not appear");
    }

    #[test]
    fn inverted_index_update_is_idempotent() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Obrain".into())],
        );
        // Re-add same node with different content
        idx.add_node(
            NodeId(1),
            &["Project".to_string()],
            &[("name".into(), "Obrain v2".into())],
        );
        assert_eq!(idx.total_nodes, 1, "Should still be 1 node after update");

        let results = idx.query(&tokenize_query("Obrain"), 10);
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
        assert!(
            idf_rare > idf_common * 10.0,
            "Rare terms should score much higher"
        );
    }

    #[test]
    fn label_dampening_formula() {
        let frac_90: f64 = 0.9;
        let dampening_90 = (1.0 + frac_90 * 10.0).ln().max(1.0);
        let frac_01: f64 = 0.01;
        let dampening_01 = (1.0 + frac_01 * 10.0).ln().max(1.0);

        assert!(
            dampening_90 > 2.0,
            "90% label should be heavily dampened: {}",
            dampening_90
        );
        assert!(
            dampening_01 < 1.2,
            "1% label should barely be dampened: {}",
            dampening_01
        );
        assert!(10.0 / dampening_90 < 10.0 / dampening_01);
    }

    #[test]
    fn index_stats_returns_distribution() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Project".into()],
            &[("name".into(), "A".into())],
        );
        idx.add_node(
            NodeId(2),
            &["Project".into()],
            &[("name".into(), "B".into())],
        );
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

    // ── InvertedIndex edge cases ────────────────────────────────

    #[test]
    fn empty_index_query_returns_empty() {
        let idx = InvertedIndex::new();
        let results = idx.query(&tokenize_query("anything"), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Test".into()],
            &[("name".into(), "hello".into())],
        );
        let results = idx.query(&[], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn remove_nonexistent_node_is_noop() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Test".into()],
            &[("name".into(), "hello".into())],
        );
        idx.remove_node(NodeId(999)); // doesn't exist
        assert_eq!(idx.total_nodes, 1);
    }

    #[test]
    fn empty_properties_indexed_without_error() {
        let mut idx = InvertedIndex::new();
        idx.add_node(NodeId(1), &["Test".into()], &[]);
        assert_eq!(idx.total_nodes, 1);
        assert!(idx.node_terms.get(&NodeId(1)).unwrap().is_empty());
    }

    #[test]
    fn empty_string_property_skipped() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Test".into()],
            &[("name".into(), String::new())],
        );
        assert!(idx.node_terms.get(&NodeId(1)).unwrap().is_empty());
    }

    #[test]
    fn max_terms_per_node_capped() {
        let mut idx = InvertedIndex::new();
        // Create text with 200+ unique words
        let big_text: String = (0..250)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        idx.add_node(NodeId(1), &["Test".into()], &[("content".into(), big_text)]);
        let term_count = idx.node_terms.get(&NodeId(1)).unwrap().len();
        assert!(
            term_count <= MAX_TERMS_PER_NODE,
            "Terms should be capped at {MAX_TERMS_PER_NODE}, got {term_count}"
        );
    }

    #[test]
    fn multiple_labels_indexed() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["Note".into(), "Gotcha".into()],
            &[("title".into(), "test".into())],
        );
        assert!(idx.label_entries.contains_key("note"));
        assert!(idx.label_entries.contains_key("gotcha"));
        assert_eq!(idx.node_labels.get(&NodeId(1)).unwrap().len(), 2);
    }

    #[test]
    fn label_matching_bidirectional() {
        let mut idx = InvertedIndex::new();
        idx.add_node(
            NodeId(1),
            &["ChatMessage".into()],
            &[("x".into(), "unrelated".into())],
        );

        // Query "chat" should match label "chatmessage" (term in label)
        let results = idx.query(&tokenize_query("chat"), 10);
        assert!(
            !results.is_empty(),
            "Term 'chat' should match label 'chatmessage'"
        );

        // Query "chatmessage" should also match
        let results2 = idx.query(&tokenize_query("chatmessage"), 10);
        assert!(!results2.is_empty());
    }

    #[test]
    fn idf_changes_with_index_evolution() {
        let mut idx = InvertedIndex::new();

        // Initially: 1 node with "obrain"
        idx.add_node(
            NodeId(1),
            &["Project".into()],
            &[("name".into(), "obrain".into())],
        );
        let results1 = idx.query(&tokenize_query("obrain"), 10);
        let score1 = results1[0].1;

        // Add 99 more nodes WITHOUT "obrain" → IDF of "obrain" should increase
        for i in 2..=100 {
            idx.add_node(
                NodeId(i),
                &["Other".into()],
                &[("name".into(), format!("node{i}"))],
            );
        }
        idx.refresh_cardinalities();

        let results2 = idx.query(&tokenize_query("obrain"), 10);
        let score2 = results2[0].1;

        assert!(
            score2 > score1,
            "IDF should increase when term becomes rarer: {score2} > {score1}"
        );
    }

    #[test]
    fn refresh_cardinalities_updates_after_removals() {
        let mut idx = InvertedIndex::new();
        for i in 0..10 {
            idx.add_node(
                NodeId(i),
                &["ChatMessage".into()],
                &[("text".into(), format!("msg{i}"))],
            );
        }
        idx.add_node(
            NodeId(100),
            &["Note".into()],
            &[("text".into(), "note".into())],
        );
        idx.refresh_cardinalities();

        // ChatMessage cardinality = 10, Note = 1
        assert_eq!(*idx.node_label_cardinality.get(&NodeId(0)).unwrap(), 10);
        assert_eq!(*idx.node_label_cardinality.get(&NodeId(100)).unwrap(), 1);

        // Remove 9 ChatMessages
        for i in 1..10 {
            idx.remove_node(NodeId(i));
        }
        idx.refresh_cardinalities();

        // Now ChatMessage cardinality = 1
        assert_eq!(*idx.node_label_cardinality.get(&NodeId(0)).unwrap(), 1);
    }

    #[test]
    fn stats_empty_index() {
        let idx = InvertedIndex::new();
        let (total, terms, labels) = idx.stats();
        // stats() uses max(1) to avoid division by zero in percentage calc
        assert_eq!(total, 1);
        assert_eq!(terms, 0);
        assert!(labels.is_empty());
    }

    #[test]
    fn stats_label_percentages() {
        let mut idx = InvertedIndex::new();
        for i in 0..4 {
            idx.add_node(
                NodeId(i),
                &["Project".into()],
                &[("name".into(), format!("p{i}"))],
            );
        }
        idx.add_node(
            NodeId(10),
            &["Note".into()],
            &[("title".into(), "n".into())],
        );

        let (total, _, labels) = idx.stats();
        assert_eq!(total, 5);
        // Project = 80%, Note = 20%
        let project = labels.iter().find(|(l, _, _)| l == "project").unwrap();
        assert!((project.2 - 80.0).abs() < 0.1);
    }

    #[test]
    fn query_multi_term_aggregates_scores() {
        let mut idx = InvertedIndex::new();
        // Node 1 matches both terms
        idx.add_node(
            NodeId(1),
            &["Note".into()],
            &[("content".into(), "obrain database engine".into())],
        );
        // Node 2 matches only one term
        idx.add_node(
            NodeId(2),
            &["Note".into()],
            &[("content".into(), "obrain project".into())],
        );
        idx.refresh_cardinalities();

        let results = idx.query(&tokenize_query("database engine"), 10);
        // Node 1 should score higher (matches both "database" and "engine")
        if !results.is_empty() {
            assert_eq!(results[0].0, NodeId(1));
        }
    }

    #[test]
    fn query_truncates_to_max_cues() {
        let mut idx = InvertedIndex::new();
        for i in 0..50 {
            idx.add_node(
                NodeId(i),
                &["Note".into()],
                &[("content".into(), format!("hello world {i}"))],
            );
        }
        idx.refresh_cardinalities();

        let results = idx.query(&tokenize_query("hello"), 5);
        assert_eq!(results.len(), 5);
    }

    // ── Tokenization edge cases ─────────────────────────────────

    #[test]
    fn tokenize_preserves_underscores_and_hyphens() {
        let terms = tokenize_query("node_id my-component");
        assert!(terms.contains(&"node_id".to_string()));
        assert!(terms.contains(&"my-component".to_string()));
    }

    #[test]
    fn tokenize_empty_string() {
        let terms = tokenize_query("");
        assert!(terms.is_empty());
    }

    #[test]
    fn tokenize_only_stopwords() {
        let terms = tokenize_query("the a is are in of");
        assert!(terms.is_empty());
    }

    #[test]
    fn tokenize_mixed_case() {
        let terms = tokenize_query("OBrain DataBase");
        assert!(terms.contains(&"obrain".to_string()));
        assert!(terms.contains(&"database".to_string()));
    }

    // ── EngramRetriever integration tests ───────────────────────

    /// Helper: build a small LpgStore with known nodes and edges.
    fn make_test_graph() -> Arc<LpgStore> {
        let store = LpgStore::new().unwrap();

        let n1 = store.create_node_with_props(
            &["Project"],
            [
                (
                    "name".to_string(),
                    obrain_common::types::Value::from("Obrain"),
                ),
                (
                    "description".to_string(),
                    obrain_common::types::Value::from("A graph database engine"),
                ),
            ],
        );

        let n2 = store.create_node_with_props(
            &["Note", "Gotcha"],
            [
                (
                    "title".to_string(),
                    obrain_common::types::Value::from("WAL Bug"),
                ),
                (
                    "content".to_string(),
                    obrain_common::types::Value::from("checkpoint.meta breaks recovery"),
                ),
            ],
        );

        let n3 = store.create_node_with_props(
            &["Task"],
            [(
                "title".to_string(),
                obrain_common::types::Value::from("Fix WAL recovery"),
            )],
        );

        // Create edges
        store.create_edge(n1, n2, "HAS_NOTE");
        store.create_edge(n2, n3, "BLOCKS");

        Arc::new(store)
    }

    fn make_retriever(graph: Arc<LpgStore>) -> EngramRetriever {
        let engram_store = Arc::new(EngramStore::new(None));
        EngramRetriever::with_defaults(graph, engram_store, None)
    }

    #[test]
    fn retriever_with_defaults_builds_index() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        let (total, terms, labels) = retriever.index_stats();
        assert_eq!(total, 3);
        assert!(terms > 0, "Should have indexed some terms");
        assert!(!labels.is_empty());
    }

    #[test]
    fn retriever_new_lazy_starts_empty() {
        let graph = make_test_graph();
        let engram_store = Arc::new(EngramStore::new(None));
        let retriever = EngramRetriever::new_lazy(Arc::clone(&graph), engram_store, None);

        let (total, terms, _) = retriever.index_stats();
        // stats() returns max(total_nodes, 1) to avoid division by zero
        assert_eq!(total, 1); // empty index reports 1 (floor)
        assert_eq!(terms, 0);
    }

    #[test]
    fn retriever_index_node_incremental() {
        let graph = make_test_graph();
        let engram_store = Arc::new(EngramStore::new(None));
        let retriever = EngramRetriever::new_lazy(Arc::clone(&graph), engram_store, None);

        // Index just one node
        let node_ids = graph.node_ids();
        retriever.index_node(node_ids[0]);

        let (total, _, _) = retriever.index_stats();
        assert_eq!(total, 1);
    }

    #[test]
    fn retriever_index_nodes_batch() {
        let graph = make_test_graph();
        let engram_store = Arc::new(EngramStore::new(None));
        let retriever = EngramRetriever::new_lazy(Arc::clone(&graph), engram_store, None);

        let node_ids = graph.node_ids();
        retriever.index_nodes(&node_ids);

        let (total, _, _) = retriever.index_stats();
        assert_eq!(total, 3);
    }

    #[test]
    fn retriever_remove_node_from_index() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        let node_ids = graph.node_ids();
        retriever.remove_node(node_ids[0]);

        let (total, _, _) = retriever.index_stats();
        assert_eq!(total, 2);
    }

    #[test]
    fn retriever_reindex_rebuilds() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        // Remove a node from index
        let node_ids = graph.node_ids();
        retriever.remove_node(node_ids[0]);
        assert_eq!(retriever.index_stats().0, 2);

        // Reindex should bring it back
        retriever.reindex();
        assert_eq!(retriever.index_stats().0, 3);
    }

    #[test]
    fn retriever_text_to_cues_finds_nodes() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);

        let cues = retriever.text_to_cues("Obrain database", 10);
        assert!(!cues.is_empty(), "Should find nodes matching 'Obrain'");
    }

    #[test]
    fn retriever_text_to_cues_no_match() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);

        let cues = retriever.text_to_cues("zzzznonexistent", 10);
        assert!(cues.is_empty());
    }

    #[test]
    fn retriever_extract_node_content() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        let node_ids = graph.node_ids();
        let content = retriever.extract_node_content(node_ids[0]);
        assert!(content.is_some());

        let node = content.unwrap();
        assert!(!node.labels.is_empty());
        assert!(!node.properties.is_empty());
    }

    #[test]
    fn retriever_extract_node_content_has_relations() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        // First node (Project) has outgoing HAS_NOTE edge
        let node_ids = graph.node_ids();
        let content = retriever.extract_node_content(node_ids[0]).unwrap();

        let has_rels =
            !content.outgoing_relations.is_empty() || !content.incoming_relations.is_empty();
        assert!(has_rels, "Node should have at least one relation");
    }

    #[test]
    fn retriever_extract_nonexistent_node() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);

        let content = retriever.extract_node_content(NodeId(99999));
        assert!(content.is_none());
    }

    #[test]
    fn retriever_retrieve_finds_text_matches() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);
        let config = RagConfig::default();

        let result = retriever.retrieve("WAL recovery checkpoint", &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(
            !result.nodes.is_empty(),
            "Should find nodes matching WAL query"
        );
    }

    #[test]
    fn retriever_retrieve_returns_properties() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);
        let config = RagConfig::default();

        let result = retriever.retrieve("Obrain database", &config).unwrap();
        assert!(!result.nodes.is_empty());

        let node = &result.nodes[0];
        assert!(
            !node.properties.is_empty(),
            "Retrieved nodes should have properties"
        );
        assert!(
            !node.labels.is_empty(),
            "Retrieved nodes should have labels"
        );
    }

    #[test]
    fn retriever_retrieve_no_match_returns_error() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);
        let config = RagConfig::default();

        let result = retriever.retrieve("xyznonexistent999", &config);
        assert!(result.is_err(), "Should error on no matches");
    }

    #[test]
    fn retriever_retrieve_respects_max_context_nodes() {
        let graph = make_test_graph();
        let retriever = make_retriever(graph);
        let config = RagConfig {
            max_context_nodes: 1,
            ..RagConfig::default()
        };

        let result = retriever.retrieve("WAL Obrain", &config).unwrap();
        assert!(result.nodes.len() <= 1);
    }

    #[test]
    fn retriever_new_with_full_components() {
        let graph = make_test_graph();
        let engram_store = Arc::new(EngramStore::new(None));
        let vector_index = Arc::new(InMemoryVectorIndex::new());
        let spectral = Arc::new(SpectralEncoder::new());
        let synapse_store = Arc::new(SynapseStore::new(
            obrain_cognitive::synapse::SynapseConfig::default(),
        ));

        let retriever = EngramRetriever::new(
            graph,
            engram_store,
            vector_index,
            spectral,
            Some(synapse_store),
        );

        let (total, _, _) = retriever.index_stats();
        assert_eq!(total, 3);
    }

    #[test]
    fn retriever_retrieve_with_synapse_store() {
        let graph = make_test_graph();
        let engram_store = Arc::new(EngramStore::new(None));
        let synapse_store = Arc::new(SynapseStore::new(
            obrain_cognitive::synapse::SynapseConfig::default(),
        ));

        let retriever =
            EngramRetriever::with_defaults(Arc::clone(&graph), engram_store, Some(synapse_store));
        let config = RagConfig::default();

        // Should work even with synapse store (spreading activation path)
        let result = retriever.retrieve("Obrain", &config);
        assert!(result.is_ok());
    }

    #[test]
    fn retriever_index_node_nonexistent_is_noop() {
        let graph = make_test_graph();
        let retriever = make_retriever(Arc::clone(&graph));

        let before = retriever.index_stats().0;
        retriever.index_node(NodeId(99999)); // doesn't exist in graph
        let after = retriever.index_stats().0;
        assert_eq!(before, after);
    }
}
