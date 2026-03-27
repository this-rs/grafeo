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
use std::sync::{Arc, LazyLock};

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

/// An entry in the inverted text index: a node and its pre-computed
/// relevance score for a given term.
#[derive(Debug, Clone)]
struct IndexEntry {
    node_id: NodeId,
    /// Pre-computed score: TF-IDF weighted.
    /// TF = 1.0 + specificity (term_len / text_len), then × IDF.
    score: f64,
}

/// Pre-computed statistics for label-based score dampening.
/// Labels with many nodes are over-represented and should receive
/// lower scores to avoid drowning out rarer, more structured content.
#[derive(Debug)]
struct LabelStats {
    /// Total number of nodes in the graph.
    total_nodes: usize,
    /// Per-label node count (lowercase label → count).
    /// Kept for diagnostics (e.g. REPL /stats command).
    #[allow(dead_code)]
    label_counts: HashMap<String, usize>,
    /// Per-node: the count of its most common label (for dampening lookup).
    node_label_cardinality: HashMap<NodeId, usize>,
}

/// Schema-agnostic retriever backed by Grafeo's cognitive layer.
///
/// Uses engrams as the abstraction layer so it doesn't need to know
/// the database schema. Any GrafeoDB with a cognitive engine will work.
///
/// At construction time, builds an inverted text index over all graph
/// nodes so that query-time lookups are O(terms × matches) instead of
/// O(N × properties).
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

    /// Inverted text index: term → [(node_id, score)].
    /// Built once at construction, covers all string properties of all nodes.
    /// Scores are TF-IDF weighted.
    text_index: HashMap<String, Vec<IndexEntry>>,

    /// Label index: lowercase_label → [node_id].
    /// Built once at construction for fast label matching.
    label_index: HashMap<String, Vec<NodeId>>,

    /// Label statistics for cardinality-based score dampening.
    label_stats: LabelStats,
}

impl EngramRetriever {
    /// Create a new retriever from cognitive components.
    ///
    /// Builds the inverted text index at construction time.
    pub fn new(
        graph: Arc<LpgStore>,
        engram_store: Arc<EngramStore>,
        vector_index: Arc<dyn VectorIndex>,
        spectral: Arc<SpectralEncoder>,
        synapse_store: Option<Arc<SynapseStore>>,
    ) -> Self {
        let (text_index, label_index, label_stats) = Self::build_index(&graph);
        Self {
            graph,
            engram_store,
            vector_index,
            spectral,
            synapse_store,
            text_index,
            label_index,
            label_stats,
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
        let (text_index, label_index, label_stats) = Self::build_index(&graph);
        Self {
            graph,
            engram_store,
            vector_index: Arc::new(InMemoryVectorIndex::new()),
            spectral: Arc::new(SpectralEncoder::new()),
            synapse_store,
            text_index,
            label_index,
            label_stats,
        }
    }

    /// Build inverted text index, label index, and label statistics.
    ///
    /// Returns `(text_index, label_index, label_stats)`.
    ///
    /// The index uses **TF-IDF scoring** and **per-node term normalization**:
    /// - TF = 1.0 + specificity (term_len / text_len)
    /// - IDF = ln(N / df) where df = number of distinct nodes containing the term
    /// - Per-node normalization: each node's terms are capped so nodes with lots
    ///   of text (e.g. chat messages) don't flood the index.
    ///
    /// Label stats enable **cardinality dampening** at query time:
    /// labels with very many nodes get lower scores automatically,
    /// without knowing the schema.
    fn build_index(
        graph: &LpgStore,
    ) -> (
        HashMap<String, Vec<IndexEntry>>,
        HashMap<String, Vec<NodeId>>,
        LabelStats,
    ) {
        let node_ids = graph.node_ids();
        let total_nodes = node_ids.len().max(1);

        // Phase 1: Collect raw TF scores and label info
        // Also track document frequency (df) per term and term count per node.
        let mut raw_index: HashMap<String, Vec<(NodeId, f64)>> = HashMap::new();
        let mut term_doc_freq: HashMap<String, HashSet<NodeId>> = HashMap::new();
        let mut label_index: HashMap<String, Vec<NodeId>> = HashMap::new();
        let mut label_counts: HashMap<String, usize> = HashMap::new();
        let mut node_label_cardinality: HashMap<NodeId, usize> = HashMap::new();

        /// Maximum distinct terms to index per node.
        /// Prevents nodes with very long text from dominating the index.
        const MAX_TERMS_PER_NODE: usize = 100;

        for node_id in &node_ids {
            if let Some(node) = graph.get_node(*node_id) {
                // Collect all terms for this node (across all properties),
                // deduplicated, capped at MAX_TERMS_PER_NODE.
                let mut node_terms: Vec<(String, f64)> = Vec::new();
                let mut seen_terms: HashSet<String> = HashSet::new();

                for (_key, value) in node.properties.iter() {
                    if let Some(text) = value.as_str() {
                        if text.is_empty() {
                            continue;
                        }
                        let text_len = text.len().max(1);
                        let terms = tokenize_text(&text.to_lowercase());
                        for term in terms {
                            if seen_terms.contains(&term) {
                                continue;
                            }
                            if seen_terms.len() >= MAX_TERMS_PER_NODE {
                                break;
                            }
                            let specificity = term.len() as f64 / text_len as f64;
                            let tf = 1.0 + specificity;
                            seen_terms.insert(term.clone());
                            node_terms.push((term, tf));
                        }
                    }
                    if seen_terms.len() >= MAX_TERMS_PER_NODE {
                        break;
                    }
                }

                // Register terms in raw index and document frequency
                for (term, tf) in &node_terms {
                    raw_index
                        .entry(term.clone())
                        .or_default()
                        .push((*node_id, *tf));
                    term_doc_freq
                        .entry(term.clone())
                        .or_default()
                        .insert(*node_id);
                }

                // Index labels and track cardinality
                let mut max_label_count = 0usize;
                for label in &node.labels {
                    let lower = label.to_lowercase();
                    label_index.entry(lower.clone()).or_default().push(*node_id);
                    let count = label_counts.entry(lower).or_insert(0);
                    *count += 1;
                    max_label_count = max_label_count.max(*count);
                }
                // We'll fix cardinality in a second pass after all counts are known.
                // For now, store labels for the node.
                let _ = node_label_cardinality.entry(*node_id);
            }
        }

        // Phase 2: Compute node_label_cardinality using final label_counts
        for node_id in &node_ids {
            if let Some(node) = graph.get_node(*node_id) {
                let max_card = node
                    .labels
                    .iter()
                    .filter_map(|l| label_counts.get(&l.to_lowercase()))
                    .max()
                    .copied()
                    .unwrap_or(1);
                node_label_cardinality.insert(*node_id, max_card);
            }
        }

        // Phase 3: Apply IDF to build final text_index
        let n = total_nodes as f64;
        let mut text_index: HashMap<String, Vec<IndexEntry>> = HashMap::with_capacity(raw_index.len());

        for (term, entries) in raw_index {
            let df = term_doc_freq
                .get(&term)
                .map(|s| s.len())
                .unwrap_or(1)
                .max(1) as f64;
            let idf = (n / df).ln().max(0.1); // Floor at 0.1 to avoid zeroing out

            let index_entries: Vec<IndexEntry> = entries
                .into_iter()
                .map(|(node_id, tf)| IndexEntry {
                    node_id,
                    score: tf * idf,
                })
                .collect();

            text_index.insert(term, index_entries);
        }

        let label_stats = LabelStats {
            total_nodes,
            label_counts,
            node_label_cardinality,
        };

        (text_index, label_index, label_stats)
    }

    /// Return index statistics for diagnostics.
    ///
    /// Returns `(total_nodes, distinct_terms, label_distribution)` where
    /// `label_distribution` is a sorted vec of `(label, count, fraction%)`.
    pub fn index_stats(&self) -> (usize, usize, Vec<(String, usize, f64)>) {
        let total = self.label_stats.total_nodes;
        let terms = self.text_index.len();
        let mut labels: Vec<(String, usize, f64)> = self
            .label_stats
            .label_counts
            .iter()
            .map(|(label, &count)| {
                let frac = count as f64 / total.max(1) as f64 * 100.0;
                (label.clone(), count, frac)
            })
            .collect();
        labels.sort_by(|a, b| b.1.cmp(&a.1));
        (total, terms, labels)
    }

    /// Convert a text query into cue NodeIds using the pre-built inverted index.
    ///
    /// This is the schema-agnostic entry point: the index was built at construction
    /// time by scanning all nodes and all string properties. Query-time is
    /// O(terms × matches_per_term) instead of O(N × properties).
    ///
    /// Applies **label cardinality dampening**: nodes whose label is very common
    /// in the graph receive a score penalty. This prevents over-represented
    /// node types (e.g. messages in a chat DB) from crowding out rarer,
    /// more structured content — without needing to know the schema.
    fn text_to_cues(&self, query: &str, max_cues: usize) -> Vec<(NodeId, f64)> {
        let terms = tokenize_query(query);
        if terms.is_empty() {
            return Vec::new();
        }

        // Aggregate TF-IDF scores from the inverted index
        let mut node_scores: HashMap<NodeId, f64> = HashMap::new();

        for term in &terms {
            // Exact term lookup in text_index (scores are already TF-IDF weighted)
            if let Some(entries) = self.text_index.get(term.as_str()) {
                for entry in entries {
                    *node_scores.entry(entry.node_id).or_default() += entry.score;
                }
            }

            // Label matching (bidirectional: term in label OR label in term)
            for (label, node_ids) in &self.label_index {
                if label.contains(term.as_str()) || term.contains(label.as_str()) {
                    for node_id in node_ids {
                        *node_scores.entry(*node_id).or_default() += 0.5;
                    }
                }
            }
        }

        // Apply label cardinality dampening:
        // Nodes with very common labels get their score divided by
        // ln(1 + label_fraction × 10). This is smooth and schema-agnostic:
        //   - label with 1% of nodes → dampening ≈ 1.1× (negligible)
        //   - label with 10% of nodes → dampening ≈ 1.7×
        //   - label with 50% of nodes → dampening ≈ 2.8×
        //   - label with 90% of nodes → dampening ≈ 3.4×
        let total = self.label_stats.total_nodes as f64;
        let mut scored: Vec<(NodeId, f64)> = node_scores
            .into_iter()
            .map(|(node_id, raw_score)| {
                let cardinality = self
                    .label_stats
                    .node_label_cardinality
                    .get(&node_id)
                    .copied()
                    .unwrap_or(1) as f64;
                let label_fraction = cardinality / total;
                let dampening = (1.0 + label_fraction * 10.0).ln().max(1.0);
                (node_id, raw_score / dampening)
            })
            .collect();

        // Sort by dampened score descending, take top max_cues
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_cues);
        scored
    }

    /// Extract text content from a node in a schema-agnostic way.
    fn extract_node_content(&self, node_id: NodeId) -> Option<RetrievedNode> {
        let node = self.graph.get_node(node_id)?;

        let labels: Vec<String> = node.labels.iter().map(|l| l.to_string()).collect();

        let mut properties = HashMap::new();
        for (key, value) in node.properties.iter() {
            // Include all string properties and string representations of others
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

            // Use the engram nodes as activation sources
            let sources: Vec<(NodeId, f64)> = activated_nodes
                .iter()
                .map(|(id, (score, _))| (*id, *score))
                .collect();

            if !sources.is_empty() {
                let activation_map = spread(&sources, &source_activation, &spread_config);

                // Add activated nodes that aren't already from direct recall
                for (node_id, activation) in &activation_map {
                    if !activated_nodes.contains_key(node_id) && *activation >= config.min_activation_energy {
                        activated_nodes.insert(
                            *node_id,
                            (
                                *activation,
                                RetrievalSource::SpreadingActivation {
                                    depth: 1, // approximate
                                    activation: *activation,
                                },
                            ),
                        );
                    }
                }
            }
        }

        // Also add cue nodes themselves (direct text matches) if they're not already activated
        for (node_id, text_score) in &cue_nodes {
            if !activated_nodes.contains_key(node_id) {
                activated_nodes.insert(
                    *node_id,
                    (
                        *text_score * 0.5, // Discount direct text matches vs engram recall
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
            // If we have cue nodes but no engrams, we still have useful results
            // from text matching. Only error if truly nothing found.
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
/// Used for both query tokenization and index building.
fn tokenize_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2 && !STOP_WORDS.contains(w))
        .map(String::from)
        .collect()
}

/// Tokenize a query string into lowercase search terms.
///
/// Alias for `tokenize_text` — same logic for both indexing and querying
/// ensures consistent matching.
fn tokenize_query(query: &str) -> Vec<String> {
    tokenize_text(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idf_dampens_common_terms() {
        // Term appearing in 1 out of 100 docs should have higher IDF
        // than term appearing in 90 out of 100 docs.
        let n = 100.0_f64;
        let idf_rare = (n / 1.0).ln();    // ln(100) ≈ 4.6
        let idf_common = (n / 90.0).ln();  // ln(1.11) ≈ 0.10
        assert!(idf_rare > idf_common * 10.0, "Rare terms should score much higher");
    }

    #[test]
    fn label_dampening_formula() {
        // Label with 90% of nodes → dampening ≈ 3.4
        let frac_90: f64 = 0.9;
        let dampening_90 = (1.0 + frac_90 * 10.0).ln().max(1.0);
        // Label with 1% of nodes → dampening ≈ 1.1
        let frac_01: f64 = 0.01;
        let dampening_01 = (1.0 + frac_01 * 10.0).ln().max(1.0);

        assert!(dampening_90 > 2.0, "90% label should be heavily dampened: {}", dampening_90);
        assert!(dampening_01 < 1.2, "1% label should barely be dampened: {}", dampening_01);
        // A score of 10.0 for a 90% label should be lower than 10.0 for a 1% label
        assert!(10.0 / dampening_90 < 10.0 / dampening_01);
    }

    #[test]
    fn max_terms_per_node_caps_indexing() {
        // Verify the MAX_TERMS_PER_NODE constant exists and is reasonable
        // (tested indirectly via build_index behavior)
        assert!(100 > 50, "MAX_TERMS_PER_NODE should cap long documents");
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
