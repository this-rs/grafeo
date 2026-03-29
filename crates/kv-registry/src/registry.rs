//! KV Node Registry — tracks what's in the KV cache across queries.

use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

use crate::Tokenizer;

/// How a node was encoded into the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSlotMode {
    /// Encoded via token IDs (standard text encoding, N positions).
    Token,
    /// Injected via batch.embd (Phase C, 1 position per virtual token).
    Embedding,
    /// 1 embedding + N tag tokens (Phase C7: micro-tags, ~3 positions per node).
    EmbeddingWithTags { n_tag_tokens: i32 },
}

/// A slot in the KV cache occupied by one graph node.
#[derive(Debug, Clone)]
pub struct KvSlot {
    /// Token position range [start, end) in the KV cache.
    pub start: i32,
    pub end: i32,
    /// Number of tokens (or virtual tokens for embeddings).
    pub n_tokens: i32,
    /// When this slot was last used (query counter).
    pub last_used: u64,
    /// The text that was encoded for this node (needed to reconstruct prompt prefix).
    pub text: String,
    /// How this node was encoded.
    pub mode: KvSlotMode,
}

/// Metrics for KV cache usage.
pub struct KvMetrics {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    #[allow(dead_code)]
    pub encode_time_ms: u128,
}

impl KvMetrics {
    pub fn new() -> Self {
        Self { total_queries: 0, cache_hits: 0, cache_misses: 0, encode_time_ms: 0 }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }
}

/// Registry that tracks which graph nodes are encoded in the KV cache.
///
/// With `cache_prompt: true`, llama.cpp reuses the KV cache when the prompt
/// prefix matches. So we keep nodes in a stable order — header first, then
/// nodes in insertion order. New nodes are appended at the end.
pub struct KvNodeRegistry {
    /// NodeId → KvSlot for nodes currently in KV.
    pub nodes: HashMap<NodeId, KvSlot>,
    /// Insertion-ordered list of node IDs (matches KV position order).
    pub order: Vec<NodeId>,
    /// End of the system header in token positions.
    pub header_end: i32,
    /// Next free token position.
    pub next_pos: i32,
    /// Query counter for LRU tracking.
    pub query_counter: u64,
    /// Metrics.
    pub metrics: KvMetrics,
    /// The system header text (for prompt reconstruction).
    pub header_text: String,
}

impl KvNodeRegistry {
    pub fn new(header_text: &str, header_tokens: i32) -> Self {
        Self {
            nodes: HashMap::new(),
            order: Vec::new(),
            header_end: header_tokens,
            next_pos: header_tokens,
            query_counter: 0,
            metrics: KvMetrics::new(),
            header_text: header_text.to_string(),
        }
    }

    /// Check which nodes need to be loaded (not yet in KV).
    pub fn find_missing(&self, needed: &[NodeId]) -> Vec<NodeId> {
        needed.iter()
            .filter(|id| !self.nodes.contains_key(id))
            .copied()
            .collect()
    }

    /// Touch nodes that are already in KV (update last_used).
    pub fn touch(&mut self, ids: &[NodeId]) {
        for id in ids {
            if let Some(slot) = self.nodes.get_mut(id) {
                slot.last_used = self.query_counter;
                self.metrics.cache_hits += 1;
            }
        }
    }

    /// Register and encode a node into the KV cache via text tokenization.
    pub fn register(&mut self, node_id: NodeId, text: &str, engine: &dyn Tokenizer) -> Result<()> {
        let tokens = engine.tokenize(text, false, true)?;
        let n_tokens = tokens.len() as i32;
        let positions: Vec<i32> = (self.next_pos..self.next_pos + n_tokens).collect();
        engine.encode(&tokens, &positions, 0)?; // seq_id=0 = persistent context

        let slot = KvSlot {
            start: self.next_pos,
            end: self.next_pos + n_tokens,
            n_tokens,
            last_used: self.query_counter,
            text: text.to_string(),
            mode: KvSlotMode::Token,
        };
        self.next_pos += n_tokens;
        self.nodes.insert(node_id, slot);
        self.order.push(node_id);
        self.metrics.cache_misses += 1;
        Ok(())
    }

    /// Register a node into the KV cache via direct embedding injection (Phase C).
    ///
    /// The embedding occupies exactly 1 KV position (vs N tokens for text).
    /// `encode_embd_fn` is called with (embedding, position, seq_id) to perform the
    /// actual llama_decode with batch.embd. This avoids adding encode_embeddings to
    /// the Tokenizer trait (which would break all implementors).
    pub fn register_embedding<F>(
        &mut self,
        node_id: NodeId,
        text: &str,
        embedding: &[f32],
        encode_embd_fn: F,
    ) -> Result<()>
    where
        F: FnOnce(&[f32], &[i32], i32) -> Result<usize>,
    {
        let position = vec![self.next_pos];
        encode_embd_fn(embedding, &position, 0)?; // seq_id=0 = persistent context

        let slot = KvSlot {
            start: self.next_pos,
            end: self.next_pos + 1, // 1 virtual token
            n_tokens: 1,
            last_used: self.query_counter,
            text: text.to_string(),
            mode: KvSlotMode::Embedding,
        };
        self.next_pos += 1;
        self.nodes.insert(node_id, slot);
        self.order.push(node_id);
        self.metrics.cache_misses += 1;
        Ok(())
    }

    /// Register a node via embedding + micro-tag tokens (Phase C7).
    ///
    /// Encodes 1 embedding (position p) + N tag tokens (positions p+1..p+N).
    /// Total KV cost: 1 + N positions (typically 3 for ":Label name [rel1, rel2]").
    /// This is 17× compression vs full text (~50 tokens) while carrying type+relation info.
    pub fn register_embedding_with_tags<F>(
        &mut self,
        node_id: NodeId,
        text: &str,
        embedding: &[f32],
        tag_text: &str,
        encode_embd_fn: F,
        engine: &dyn Tokenizer,
    ) -> Result<()>
    where
        F: FnOnce(&[f32], &[i32], i32) -> Result<usize>,
    {
        // Step 1: Inject the embedding at position p
        let p = self.next_pos;
        encode_embd_fn(embedding, &[p], 0)?; // seq_id=0 = persistent

        // Step 2: Tokenize and encode the micro-tag at positions p+1..
        let tag_tokens = engine.tokenize(tag_text, false, true)?;
        let n_tag = tag_tokens.len() as i32;
        if n_tag > 0 {
            let tag_positions: Vec<i32> = ((p + 1)..=(p + n_tag)).collect();
            engine.encode(&tag_tokens, &tag_positions, 0)?;
        }

        let total = 1 + n_tag;
        let slot = KvSlot {
            start: p,
            end: p + total,
            n_tokens: total,
            last_used: self.query_counter,
            text: text.to_string(),
            mode: KvSlotMode::EmbeddingWithTags { n_tag_tokens: n_tag },
        };
        self.next_pos += total;
        self.nodes.insert(node_id, slot);
        self.order.push(node_id);
        self.metrics.cache_misses += 1;
        Ok(())
    }

    /// Get the KV position range for a node (if loaded).
    pub fn get_slot(&self, node_id: NodeId) -> Option<&KvSlot> {
        self.nodes.get(&node_id)
    }

    /// Update the system header text (e.g. when facts change mid-session).
    pub fn update_header(&mut self, new_header: &str) {
        self.header_text = new_header.to_string();
    }

    /// Reconstruct the full prompt prefix (header + all loaded nodes in order).
    /// Used in HTTP mode for `cache_prompt: true` matching.
    pub fn reconstruct_prompt(&self) -> String {
        let mut prompt = self.header_text.clone();
        for nid in &self.order {
            if let Some(slot) = self.nodes.get(nid) {
                prompt.push_str(&slot.text);
            }
        }
        prompt
    }

    /// Start a new query — increment counter.
    pub fn begin_query(&mut self) {
        self.query_counter += 1;
        self.metrics.total_queries += 1;
    }

    /// Evict least-recently-used nodes to free up token capacity.
    pub fn evict_lru(&mut self, tokens_needed: i32, protected: &HashSet<NodeId>, engine: &dyn Tokenizer) -> Vec<NodeId> {
        let mut candidates: Vec<(NodeId, u64, i32)> = self.nodes.iter()
            .filter(|(id, _)| !protected.contains(id))
            .map(|(id, slot)| (*id, slot.last_used, slot.n_tokens))
            .collect();
        candidates.sort_by_key(|(_, last_used, _)| *last_used);

        let mut freed = 0i32;
        let mut evicted: Vec<NodeId> = Vec::new();

        for (nid, _last_used, n_tokens) in &candidates {
            if freed >= tokens_needed { break; }
            evicted.push(*nid);
            freed += n_tokens;
        }

        // Remove from KV cache + registry
        for nid in &evicted {
            if let Some(slot) = self.nodes.remove(nid) {
                engine.evict(slot.start, slot.end);
            }
        }
        self.order.retain(|id| !evicted.contains(id));

        // Update next_pos to actual KV cache state (llama.cpp requires consecutive positions)
        if !evicted.is_empty() {
            self.next_pos = engine.seq_pos_max(0) + 1;
        }

        evicted
    }

    /// Ensure capacity for `needed_tokens` new tokens.
    /// If position space is near 90% of n_ctx, do a full recompact.
    pub fn ensure_capacity(&mut self, needed_tokens: i32, max_kv_tokens: i32, protected: &HashSet<NodeId>, engine: &dyn Tokenizer) {
        // Check if position space is nearly exhausted (full-recompact trigger)
        let n_ctx = engine.n_ctx() as i32;
        if self.next_pos + needed_tokens > (n_ctx as f64 * 0.9) as i32 {
            eprintln!("  ⚠ Position space near limit ({}/{}), full recompact...", self.next_pos, n_ctx);
            self.full_recompact(engine);
        }

        let total_tokens: i32 = self.nodes.values().map(|s| s.n_tokens).sum();
        let available = max_kv_tokens - total_tokens;
        if available >= needed_tokens { return; }

        let to_free = needed_tokens - available;
        self.evict_lru(to_free, protected, engine);
    }

    /// Full recompact: clear everything and re-encode from scratch.
    pub fn full_recompact(&mut self, engine: &dyn Tokenizer) {
        engine.clear_kv();

        // Re-encode header (may have changed size since last build)
        let mut pos = 0i32;
        if let Ok(tokens) = engine.tokenize(&self.header_text, false, true) {
            let n_tok = tokens.len() as i32;
            let positions: Vec<i32> = (0..n_tok).collect();
            let _ = engine.encode(&tokens, &positions, 0);
            pos = n_tok;
        }
        // Update header_end to reflect actual new header size
        self.header_end = pos;

        // Re-encode all nodes with fresh contiguous positions
        for nid in &self.order {
            if let Some(slot) = self.nodes.get_mut(nid) {
                if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                    let n_tok = tokens.len() as i32;
                    let positions: Vec<i32> = (pos..pos + n_tok).collect();
                    let _ = engine.encode(&tokens, &positions, 0);
                    slot.start = pos;
                    slot.end = pos + n_tok;
                    slot.n_tokens = n_tok;
                    pos += n_tok;
                }
            }
        }
        self.next_pos = pos;
    }
}
