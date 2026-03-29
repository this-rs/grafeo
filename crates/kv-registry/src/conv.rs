//! Conversation Fragment Registry — 3-tier conversation memory.
//!
//! **HOT** (KV cache): max `max_fragments` fragments, model sees them via attention.
//! **WARM** (in-memory archive): evicted fragments with terms + text, searchable.
//!   When a query matches a WARM fragment, it's promoted back to HOT (re-encoded in KV).
//! **COLD** (PersonaDB): all messages stored permanently. Loaded into WARM at startup.

use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

use crate::{Tokenizer, KvNodeRegistry};

/// Conversation fragments live in NodeId space 0xFFFF_0000_0000_xxxx
/// to avoid collision with graph NodeIds.
pub const CONV_NODE_BASE: u64 = 0xFFFF_0000_0000_0000;

/// Maximum conversation fragments kept in KV cache (HOT tier).
const MAX_HOT_FRAGMENTS: usize = 10;

/// Maximum archived fragments in WARM tier (in-memory, not in KV).
/// Beyond this, oldest are dropped (they remain in PersonaDB = COLD).
const MAX_WARM_FRAGMENTS: usize = 200;

/// A conversation fragment — a concise Q&A summary.
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

/// Manages conversation fragments across 3 tiers: HOT (KV) → WARM (memory) → COLD (PersonaDB).
pub struct ConvFragments {
    /// HOT tier: fragments currently encoded in the KV cache.
    pub fragments: Vec<ConvFragment>,
    /// WARM tier: archived fragments (evicted from KV, searchable by terms).
    pub archived: Vec<ConvFragment>,
    pub next_turn: u32,
    pub max_fragments: usize,
}

impl ConvFragments {
    pub fn new() -> Self {
        Self {
            fragments: Vec::new(),
            archived: Vec::new(),
            next_turn: 0,
            max_fragments: MAX_HOT_FRAGMENTS,
        }
    }

    pub fn with_max_fragments(max: usize) -> Self {
        Self {
            fragments: Vec::new(),
            archived: Vec::new(),
            next_turn: 0,
            max_fragments: max.max(3),
        }
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

        // Extract key terms from the question + answer
        let terms = Self::extract_terms(question, answer);

        // Build a concise fragment — question + first meaningful lines of answer
        let answer_summary = Self::summarize_answer(answer, 200);
        let kv_text = format!("[Conv Q{}] {}\n→ {}\n", turn + 1, question, answer_summary);

        // ── Sliding window: evict oldest HOT fragments → WARM archive ──
        while self.fragments.len() >= self.max_fragments {
            let evicted = self.fragments.remove(0);
            registry.unregister(evicted.node_id);
            eprintln!(
                "[Conv] HOT→WARM: Q{} (node {:?}) — {} hot, {} warm",
                evicted.turn + 1, evicted.node_id,
                self.fragments.len(), self.archived.len() + 1,
            );
            self.archived.push(evicted);
        }

        // Cap WARM tier (oldest drop to COLD = PersonaDB only)
        while self.archived.len() > MAX_WARM_FRAGMENTS {
            let dropped = self.archived.remove(0);
            eprintln!(
                "[Conv] WARM→COLD: Q{} — only in PersonaDB now",
                dropped.turn + 1,
            );
        }

        // Estimate tokens and ensure capacity
        // Only protect RECENT fragments (last 5), not ALL of them
        let est_tokens = (kv_text.len() as f64 / 3.5) as i32 + 5;
        let recent_count = 5.min(self.fragments.len());
        let protected: HashSet<NodeId> = related_nodes.iter().copied()
            .chain(self.fragments[self.fragments.len() - recent_count..].iter().map(|f| f.node_id))
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

    /// Seed a WARM-tier fragment from PersonaDB history (startup restoration).
    /// These are NOT in the KV cache — they're searchable and promotable.
    pub fn seed_warm(
        &mut self,
        question: &str,
        answer: &str,
        related_nodes: &[NodeId],
    ) {
        let turn = self.next_turn;
        self.next_turn += 1;
        let node_id = NodeId(CONV_NODE_BASE + turn as u64);

        let terms = Self::extract_terms(question, answer);
        let answer_summary = Self::summarize_answer(answer, 200);
        let kv_text = format!("[Conv Q{}] {}\n→ {}\n", turn + 1, question, answer_summary);

        let fragment = ConvFragment {
            node_id,
            question: question.to_string(),
            terms,
            kv_text,
            related_graph_nodes: related_nodes.to_vec(),
            turn,
        };

        self.archived.push(fragment);

        // Cap WARM tier
        while self.archived.len() > MAX_WARM_FRAGMENTS {
            self.archived.remove(0);
        }
    }

    /// Extract key terms from both question and answer for matching.
    fn extract_terms(question: &str, answer: &str) -> Vec<String> {
        let combined = format!("{} {}", question, Self::summarize_answer(answer, 100));
        combined.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
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
    /// Searches HOT tier first, then WARM tier. WARM matches are promoted to HOT
    /// (re-encoded into KV cache).
    pub fn find_relevant(
        &mut self,
        query: &str,
        registry: &mut KvNodeRegistry,
        engine: &dyn Tokenizer,
        kv_capacity: i32,
    ) -> (Vec<NodeId>, HashMap<NodeId, HashSet<NodeId>>) {
        let meaningful_terms = Self::query_terms(query);
        let stop_words = Self::stop_words();

        let mut relevant_ids = Vec::new();
        let mut conv_adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();

        // ── Search HOT tier (in KV cache) ──
        for frag in &self.fragments {
            if registry.get_slot(frag.node_id).is_none() { continue; }

            let score = Self::score_fragment(frag, &meaningful_terms, &stop_words);
            if score > 0 {
                relevant_ids.push(frag.node_id);
                registry.touch(&[frag.node_id]);
                Self::add_adjacency(&mut conv_adjacency, frag);
            }
        }

        // ── Search WARM tier (archived, not in KV) → promote matches ──
        let mut promote_indices: Vec<(usize, usize)> = Vec::new(); // (index, score)
        for (i, frag) in self.archived.iter().enumerate() {
            let score = Self::score_fragment(frag, &meaningful_terms, &stop_words);
            if score > 0 {
                promote_indices.push((i, score));
            }
        }
        // Sort by score descending, promote top 3 max
        promote_indices.sort_by(|a, b| b.1.cmp(&a.1));
        promote_indices.truncate(3);
        // Sort indices in reverse order for safe removal
        promote_indices.sort_by(|a, b| b.0.cmp(&a.0));

        let mut promoted: Vec<ConvFragment> = Vec::new();
        for (idx, _score) in &promote_indices {
            promoted.push(self.archived.remove(*idx));
        }
        // Promote: re-encode into KV cache
        for frag in promoted {
            eprintln!(
                "[Conv] WARM→HOT: promoting Q{} '{}' (matched query)",
                frag.turn + 1, &frag.question.chars().take(50).collect::<String>(),
            );
            // Make room if needed
            while self.fragments.len() >= self.max_fragments {
                let evicted = self.fragments.remove(0);
                registry.unregister(evicted.node_id);
                self.archived.push(evicted);
            }
            // Re-encode
            let est_tokens = (frag.kv_text.len() as f64 / 3.5) as i32 + 5;
            let protected: HashSet<NodeId> = self.fragments.iter().map(|f| f.node_id).collect();
            registry.ensure_capacity(est_tokens, kv_capacity, &protected, engine);
            if let Ok(_) = registry.register(frag.node_id, &frag.kv_text, engine) {
                relevant_ids.push(frag.node_id);
                Self::add_adjacency(&mut conv_adjacency, &frag);
                self.fragments.push(frag);
            } else {
                // Re-encode failed (KV full even after eviction), put back in archive
                self.archived.push(frag);
            }
        }

        // Include the LAST fragment for context continuity
        if let Some(last) = self.fragments.last() {
            if !relevant_ids.contains(&last.node_id) {
                if registry.get_slot(last.node_id).is_some() {
                    relevant_ids.push(last.node_id);
                    registry.touch(&[last.node_id]);
                    Self::add_adjacency(&mut conv_adjacency, last);
                }
            }
        }

        (relevant_ids, conv_adjacency)
    }

    /// Compute match score for a fragment against query terms.
    fn score_fragment(frag: &ConvFragment, meaningful_terms: &HashSet<String>, stop_words: &HashSet<&str>) -> usize {
        let frag_terms: HashSet<&str> = frag.terms.iter()
            .map(|t| t.as_str())
            .filter(|t| !stop_words.contains(t))
            .collect();

        let overlap = meaningful_terms.iter()
            .filter(|t| frag_terms.contains(t.as_str()))
            .count();

        let text_lower = frag.kv_text.to_lowercase();
        let text_matches = meaningful_terms.iter()
            .filter(|t| text_lower.contains(t.as_str()))
            .count();

        overlap * 2 + text_matches
    }

    fn query_terms(query: &str) -> HashSet<String> {
        query.to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect()
    }

    fn stop_words() -> HashSet<&'static str> {
        [
            "les", "des", "est", "sont", "que", "qui", "quels", "quel", "quelle",
            "pour", "dans", "avec", "sur", "par", "une", "the", "and", "what",
            "which", "about", "from", "with", "donne", "détails", "details",
            "plus", "encore", "more", "tell", "show", "comment", "quoi",
        ].into_iter().collect()
    }

    fn add_adjacency(conv_adjacency: &mut HashMap<NodeId, HashSet<NodeId>>, frag: &ConvFragment) {
        let related: HashSet<NodeId> = frag.related_graph_nodes.iter().copied().collect();
        conv_adjacency.insert(frag.node_id, related.clone());
        for &gn in &frag.related_graph_nodes {
            conv_adjacency.entry(gn).or_default().insert(frag.node_id);
        }
    }

    /// Stats for debugging.
    pub fn stats(&self) -> (usize, usize) {
        (self.fragments.len(), self.archived.len())
    }
}
