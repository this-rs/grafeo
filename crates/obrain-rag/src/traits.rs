//! Core traits for the RAG pipeline.
//!
//! The pipeline is decomposed into three pluggable stages:
//! 1. **Retriever** — finds relevant nodes from the graph given a query
//! 2. **ContextBuilder** — formats retrieved nodes into LLM-consumable text
//! 3. **FeedbackSink** — reinforces cognitive structures after LLM response

use crate::config::RagConfig;
use crate::error::RagResult;
use grafeo_common::types::NodeId;
use std::collections::HashMap;

/// A node retrieved by the RAG pipeline, with its activation score
/// and extracted content.
///
/// Content extraction is **schema-agnostic**: all string properties
/// are collected without hardcoding label names or property keys.
#[derive(Debug, Clone)]
pub struct RetrievedNode {
    /// The graph node ID.
    pub node_id: NodeId,

    /// Labels on this node (e.g. `["Project"]`, `["Note", "Gotcha"]`).
    pub labels: Vec<String>,

    /// All text-valued properties extracted from the node.
    /// Keys are property names, values are their string content.
    pub properties: HashMap<String, String>,

    /// Combined relevance score (engram recall × activation × energy).
    pub score: f64,

    /// How this node was found — direct engram hit or spreading activation.
    pub source: RetrievalSource,

    /// Outgoing relations: `(relation_type, target_node_id)`.
    pub outgoing_relations: Vec<(String, NodeId)>,

    /// Incoming relations: `(relation_type, source_node_id)`.
    pub incoming_relations: Vec<(String, NodeId)>,
}

/// How a node was found during retrieval.
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalSource {
    /// Direct hit from engram Hopfield recall.
    EngramRecall {
        /// The engram ID that matched.
        engram_id: u64,
        /// Recall confidence from Hopfield matching.
        confidence: f64,
    },
    /// Found via spreading activation from an engram node.
    SpreadingActivation {
        /// BFS depth from the nearest engram node.
        depth: u32,
        /// Activation energy at this node.
        activation: f64,
    },
}

/// The result of a retrieval operation.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Retrieved nodes, sorted by score (highest first).
    pub nodes: Vec<RetrievedNode>,

    /// Total number of engrams matched before activation.
    pub engrams_matched: usize,

    /// Total nodes activated before filtering/ranking.
    pub nodes_activated: usize,
}

/// The final RAG context ready to be injected into an LLM prompt.
#[derive(Debug, Clone)]
pub struct RagContext {
    /// Formatted text context for the LLM.
    pub text: String,

    /// Estimated token count.
    pub estimated_tokens: usize,

    /// Number of nodes included in the context.
    pub nodes_included: usize,

    /// Node IDs included (for feedback tracking).
    pub node_ids: Vec<NodeId>,

    /// Per-node text values for response-aware feedback.
    /// Each entry is `(node_id, [text_values])` — the significant string
    /// property values from that node. Used by `FeedbackSink` to detect
    /// which nodes were actually mentioned in the LLM response.
    pub node_texts: Vec<(NodeId, Vec<String>)>,
}

/// Retriever — finds relevant graph nodes given a text query.
///
/// The default implementation uses engram-based Hopfield recall
/// followed by spreading activation through synapses.
pub trait Retriever: Send + Sync {
    /// Retrieve relevant nodes for a query.
    fn retrieve(&self, query: &str, config: &RagConfig) -> RagResult<RetrievalResult>;
}

/// ContextBuilder — formats retrieved nodes into LLM-consumable text.
///
/// Handles ranking, token budgeting, and structured formatting.
pub trait ContextBuilder: Send + Sync {
    /// Build a text context from retrieved nodes.
    fn build(&self, result: &RetrievalResult, config: &RagConfig) -> RagResult<RagContext>;
}

/// FeedbackSink — reinforces cognitive structures after LLM response.
///
/// Given the context that was provided and the LLM's response,
/// strengthens synapses between co-activated concepts and boosts
/// energy of nodes whose content was useful.
pub trait FeedbackSink: Send + Sync {
    /// Provide feedback after an LLM response.
    ///
    /// - `context`: the RAG context that was injected into the prompt
    /// - `response`: the LLM's response text
    fn feedback(
        &self,
        context: &RagContext,
        response: &str,
        config: &RagConfig,
    ) -> RagResult<FeedbackStats>;
}

/// Statistics from a feedback operation.
#[derive(Debug, Clone, Default)]
pub struct FeedbackStats {
    /// Number of synapses reinforced.
    pub synapses_reinforced: usize,

    /// Number of nodes whose energy was boosted.
    pub nodes_boosted: usize,
}
