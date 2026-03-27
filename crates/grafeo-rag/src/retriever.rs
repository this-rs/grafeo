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
use std::sync::Arc;

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

/// Schema-agnostic retriever backed by Grafeo's cognitive layer.
///
/// Uses engrams as the abstraction layer so it doesn't need to know
/// the database schema. Any GrafeoDB with a cognitive engine will work.
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
}

impl EngramRetriever {
    /// Create a new retriever from cognitive components.
    pub fn new(
        graph: Arc<LpgStore>,
        engram_store: Arc<EngramStore>,
        vector_index: Arc<dyn VectorIndex>,
        spectral: Arc<SpectralEncoder>,
        synapse_store: Option<Arc<SynapseStore>>,
    ) -> Self {
        Self {
            graph,
            engram_store,
            vector_index,
            spectral,
            synapse_store,
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
        Self {
            graph,
            engram_store,
            vector_index: Arc::new(InMemoryVectorIndex::new()),
            spectral: Arc::new(SpectralEncoder::new()),
            synapse_store,
        }
    }

    /// Convert a text query into cue NodeIds by scanning the graph for
    /// nodes whose text properties contain query terms.
    ///
    /// This is the schema-agnostic entry point: we don't know which labels
    /// or properties exist, so we scan all nodes and check all string properties.
    fn text_to_cues(&self, query: &str, max_cues: usize) -> Vec<(NodeId, f64)> {
        let terms = tokenize_query(query);
        if terms.is_empty() {
            return Vec::new();
        }

        let node_ids = self.graph.node_ids();
        let mut scored: Vec<(NodeId, f64)> = Vec::new();

        for node_id in &node_ids {
            if let Some(node) = self.graph.get_node(*node_id) {
                let mut node_score = 0.0f64;

                // Check all string properties for term matches
                for (_key, value) in node.properties.iter() {
                    if let Some(text) = value.as_str() {
                        let text_lower = text.to_lowercase();
                        for term in &terms {
                            if text_lower.contains(term.as_str()) {
                                // Score by term length relative to text length
                                // (longer matches are more specific)
                                let specificity = term.len() as f64 / text_lower.len().max(1) as f64;
                                node_score += 1.0 + specificity;
                            }
                        }
                    }
                }

                // Also check labels (bidirectional: term in label OR label in term)
                for label in &node.labels {
                    let label_lower = label.to_lowercase();
                    for term in &terms {
                        if label_lower.contains(term.as_str()) || term.contains(label_lower.as_str()) {
                            node_score += 0.5;
                        }
                    }
                }

                if node_score > 0.0 {
                    scored.push((*node_id, node_score));
                }
            }
        }

        // Sort by score descending, take top max_cues
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

/// Tokenize a query string into lowercase search terms.
///
/// Filters out common stop words and very short terms.
fn tokenize_query(query: &str) -> Vec<String> {
    const STOP_WORDS: &[&str] = &[
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
    ];

    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();

    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2 && !stop_set.contains(w))
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
