//! Conversation Fragment Registry — stores Q&A summaries as KV-cache-resident nodes.

use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

use crate::{Tokenizer, KvNodeRegistry};

/// Conversation fragments live in NodeId space 0xFFFF_0000_0000_xxxx
/// to avoid collision with graph NodeIds.
pub const CONV_NODE_BASE: u64 = 0xFFFF_0000_0000_0000;

/// A conversation fragment — a concise Q&A summary stored as a KV node.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ConvFragment {
    /// Virtual NodeId in the KV registry.
    pub node_id: NodeId,
    /// The question asked by the user.
    pub question: String,
    /// Key terms from the question (for matching).
    pub terms: Vec<String>,
    /// Concise summary text registered in the KV.
    pub kv_text: String,
    /// Which graph NodeIds were relevant to this Q&A (for attention links).
    pub related_graph_nodes: Vec<NodeId>,
    /// Turn number.
    pub turn: u32,
}

/// Manages conversation fragments as KV-cache-resident nodes.
pub struct ConvFragments {
    pub fragments: Vec<ConvFragment>,
    pub next_turn: u32,
}

impl ConvFragments {
    pub fn new() -> Self {
        Self { fragments: Vec::new(), next_turn: 0 }
    }

    /// Returns the NodeId of the last registered conversation fragment.
    pub fn last_node_id(&self) -> Option<NodeId> {
        self.fragments.last().map(|f| f.node_id)
    }

    /// Create a concise fragment from a Q&A exchange and register it in the KV.
    pub fn add_turn(
        &mut self,
        question: &str,
        answer: &str,
        related_nodes: &[NodeId],
        registry: &mut KvNodeRegistry,
        engine: &dyn Tokenizer,
        kv_capacity: i32,
    ) -> Result<NodeId> {
        let turn = self.next_turn;
        self.next_turn += 1;
        let node_id = NodeId(CONV_NODE_BASE + turn as u64);

        // Extract key terms from the question
        let terms: Vec<String> = question.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();

        // Build a concise fragment — question + first meaningful lines of answer
        let answer_summary = Self::summarize_answer(answer, 200);
        let kv_text = format!("[Conv Q{}] {}\n→ {}\n", turn + 1, question, answer_summary);

        // Estimate tokens and ensure capacity
        let est_tokens = (kv_text.len() as f64 / 3.5) as i32 + 5;
        let protected: HashSet<NodeId> = related_nodes.iter().copied()
            .chain(self.fragments.iter().map(|f| f.node_id))
            .collect();
        registry.ensure_capacity(est_tokens, kv_capacity, &protected, engine);

        // Encode and register in the KV cache
        let _n_tokens = engine.token_count(&kv_text)? as i32;
        registry.register(node_id, &kv_text, engine)?;

        let fragment = ConvFragment {
            node_id,
            question: question.to_string(),
            terms,
            kv_text,
            related_graph_nodes: related_nodes.to_vec(),
            turn,
        };

        self.fragments.push(fragment);
        Ok(node_id)
    }

    /// Extract the first meaningful lines of the answer as a summary.
    pub fn summarize_answer(answer: &str, max_chars: usize) -> String {
        let mut summary = String::new();
        for line in answer.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            if trimmed.starts_with("---") || trimmed.starts_with("===") { continue; }
            if trimmed.starts_with("```") { continue; }

            if !summary.is_empty() { summary.push(' '); }
            summary.push_str(trimmed);

            if summary.len() >= max_chars { break; }
        }
        if summary.len() > max_chars {
            summary = summary.chars().take(max_chars).collect();
            summary.push_str("…");
        }
        summary
    }

    /// Find conversation fragments relevant to a query.
    pub fn find_relevant(&self, query: &str, registry: &mut KvNodeRegistry) -> (Vec<NodeId>, HashMap<NodeId, HashSet<NodeId>>) {
        let query_lower = query.to_lowercase();
        let query_terms: HashSet<String> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();

        let stop_words: HashSet<&str> = [
            "les", "des", "est", "sont", "que", "qui", "quels", "quel", "quelle",
            "pour", "dans", "avec", "sur", "par", "une", "the", "and", "what",
            "which", "about", "from", "with", "donne", "détails", "details",
            "plus", "encore", "more", "tell", "show", "comment", "quoi",
        ].into_iter().collect();

        let meaningful_terms: HashSet<&String> = query_terms.iter()
            .filter(|t| !stop_words.contains(t.as_str()))
            .collect();

        let mut relevant_ids = Vec::new();
        let mut conv_adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        for frag in &self.fragments {
            if registry.get_slot(frag.node_id).is_none() { continue; }

            let frag_terms: HashSet<&String> = frag.terms.iter()
                .filter(|t| !stop_words.contains(t.as_str()))
                .collect();

            let overlap = meaningful_terms.intersection(&frag_terms).count();

            let text_lower = frag.kv_text.to_lowercase();
            let text_matches = meaningful_terms.iter()
                .filter(|t| text_lower.contains(t.as_str()))
                .count();

            let score = overlap * 2 + text_matches;

            if score > 0 {
                relevant_ids.push(frag.node_id);
                registry.touch(&[frag.node_id]);

                let related: HashSet<NodeId> = frag.related_graph_nodes.iter().copied().collect();
                conv_adjacency.insert(frag.node_id, related.clone());
                for &gn in &frag.related_graph_nodes {
                    conv_adjacency.entry(gn).or_default().insert(frag.node_id);
                }
            }
        }

        // Include the LAST fragment for context continuity
        if let Some(last) = self.fragments.last() {
            if !relevant_ids.contains(&last.node_id) {
                if registry.get_slot(last.node_id).is_some() {
                    relevant_ids.push(last.node_id);
                    registry.touch(&[last.node_id]);
                    let related: HashSet<NodeId> = last.related_graph_nodes.iter().copied().collect();
                    conv_adjacency.insert(last.node_id, related.clone());
                    for &gn in &last.related_graph_nodes {
                        conv_adjacency.entry(gn).or_default().insert(last.node_id);
                    }
                }
            }
        }

        (relevant_ids, conv_adjacency)
    }
}
