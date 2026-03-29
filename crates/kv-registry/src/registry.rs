//! KV Node Registry — tracks what's in the KV cache across queries.

use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

use crate::Tokenizer;
use crate::hilbert::HilbertLayout;

/// How a node was encoded into the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSlotMode {
    /// Encoded via token IDs (standard text encoding, N positions).
    Token,
    /// Injected via batch.embd (Phase C, 1 position per virtual token).
    Embedding,
    /// 1 embedding + N label-only tag tokens (Phase D tier Β: ":Label name", ~2 positions).
    EmbeddingWithMinimalTag { n_label_tokens: i32 },
    /// 1 embedding + N tag tokens (Phase C7/D tier Α: micro-tags, ~3-5 positions per node).
    EmbeddingWithTags { n_tag_tokens: i32 },
}

/// KV tier for dynamic resolution management (Phase D).
/// Tiers are orthogonal to position — changing tier doesn't change KV position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTier {
    /// Rich tags: 1 embedding + full micro-tag (":Label name [→rel1 t1, →rel2 t2]").
    /// Highest resolution, ~3-5 KV positions per node.
    Alpha,
    /// Minimal tags: 1 embedding + label only (":Person Thomas").
    /// Medium resolution, ~2 KV positions per node.
    Beta,
    /// Pure embedding: 1 embedding only.
    /// Lowest resolution, 1 KV position per node.
    Gamma,
}

/// Budget tracker for tier promotions per round.
/// Limits the number of promote operations to bound latency.
#[derive(Debug)]
pub struct TierBudget {
    pub max_promotions: u32,
    pub used: u32,
}

impl TierBudget {
    pub fn new(max_promotions: u32) -> Self {
        Self { max_promotions, used: 0 }
    }

    pub fn can_promote(&self) -> bool {
        self.used < self.max_promotions
    }

    pub fn consume(&mut self) -> bool {
        if self.used < self.max_promotions {
            self.used += 1;
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn remaining(&self) -> u32 {
        self.max_promotions.saturating_sub(self.used)
    }
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
    /// Dynamic resolution tier (Phase D).
    pub tier: KvTier,
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
    /// Phase D: tier promotion budget per round.
    pub tier_budget: TierBudget,
    /// Phase D: optional Hilbert layout for topology-aware KV positioning.
    /// When set, register_embedding* uses Hilbert positions instead of next_pos.
    pub hilbert_layout: Option<HilbertLayout>,
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
            tier_budget: TierBudget::new(50),
            hilbert_layout: None,
        }
    }

    /// Set the Hilbert layout for topology-aware KV positioning.
    pub fn set_hilbert_layout(&mut self, layout: HilbertLayout) {
        self.hilbert_layout = Some(layout);
    }

    /// Get the base KV position for a node: Hilbert position if available, else next_pos.
    /// When using Hilbert layout, next_pos is advanced past the max Hilbert position
    /// to keep Token-mode nodes (conversation, header) out of the Hilbert range.
    fn position_for(&mut self, node_id: NodeId) -> i32 {
        if let Some(ref layout) = self.hilbert_layout {
            if let Some(pos) = layout.get_position(node_id) {
                // Advance next_pos past this position if needed (for non-Hilbert nodes)
                let pos = pos as i32;
                if pos + 1 > self.next_pos {
                    self.next_pos = pos + 1;
                }
                return pos;
            }
        }
        self.next_pos
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
            tier: KvTier::Alpha, // Text nodes are always full resolution
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
        let p = self.position_for(node_id);
        encode_embd_fn(embedding, &[p], 0)?; // seq_id=0 = persistent context

        let slot = KvSlot {
            start: p,
            end: p + 1, // 1 virtual token
            n_tokens: 1,
            last_used: self.query_counter,
            text: text.to_string(),
            mode: KvSlotMode::Embedding,
            tier: KvTier::Gamma, // Pure embedding = lowest resolution
        };
        // Advance next_pos past this slot if it was from Hilbert
        if p + 1 > self.next_pos { self.next_pos = p + 1; }
        // If no Hilbert, advance normally
        if self.hilbert_layout.is_none() || self.hilbert_layout.as_ref().and_then(|l| l.get_position(node_id)).is_none() {
            self.next_pos = p + 1;
        }
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
        // Step 1: Inject the embedding at position p (Hilbert or sequential)
        let p = self.position_for(node_id);
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
            tier: KvTier::Alpha, // Embedding + full tags = highest resolution
        };
        // Advance next_pos past this slot
        if p + total > self.next_pos { self.next_pos = p + total; }
        if self.hilbert_layout.is_none() || self.hilbert_layout.as_ref().and_then(|l| l.get_position(node_id)).is_none() {
            self.next_pos = p + total;
        }
        self.nodes.insert(node_id, slot);
        self.order.push(node_id);
        self.metrics.cache_misses += 1;
        Ok(())
    }

    // ── Phase D: Tier promotion/demotion ──────────────────────────

    /// Demote a node from Alpha (rich tags) to Beta (minimal tag = label only).
    /// Removes tag tokens beyond the first label token via engine.evict().
    /// Slot: EmbeddingWithTags{n} → EmbeddingWithMinimalTag{1}, end = start+2.
    pub fn demote_to_beta(&mut self, node_id: NodeId, engine: &dyn Tokenizer) -> Result<()> {
        let slot = self.nodes.get_mut(&node_id)
            .ok_or_else(|| anyhow::anyhow!("demote_to_beta: node {:?} not in registry", node_id))?;
        match slot.mode {
            KvSlotMode::EmbeddingWithTags { n_tag_tokens } if n_tag_tokens > 1 => {
                // Keep embedding (start) + first label token (start+1), evict the rest
                engine.evict(slot.start + 2, slot.end);
                slot.end = slot.start + 2;
                slot.n_tokens = 2;
                slot.mode = KvSlotMode::EmbeddingWithMinimalTag { n_label_tokens: 1 };
                slot.tier = KvTier::Beta;
                Ok(())
            }
            KvSlotMode::EmbeddingWithTags { n_tag_tokens: 1 } => {
                // Already just 1 tag token — just relabel
                slot.mode = KvSlotMode::EmbeddingWithMinimalTag { n_label_tokens: 1 };
                slot.tier = KvTier::Beta;
                Ok(())
            }
            KvSlotMode::EmbeddingWithTags { n_tag_tokens: 0 } => {
                // No tags — go directly to Gamma
                slot.mode = KvSlotMode::Embedding;
                slot.tier = KvTier::Gamma;
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "demote_to_beta: node {:?} mode {:?} cannot demote to Beta",
                node_id, slot.mode
            )),
        }
    }

    /// Demote a node to Gamma (pure embedding). Removes all tag tokens.
    /// Works from Alpha (EmbeddingWithTags) or Beta (EmbeddingWithMinimalTag).
    pub fn demote_to_gamma(&mut self, node_id: NodeId, engine: &dyn Tokenizer) -> Result<()> {
        let slot = self.nodes.get_mut(&node_id)
            .ok_or_else(|| anyhow::anyhow!("demote_to_gamma: node {:?} not in registry", node_id))?;
        match slot.mode {
            KvSlotMode::EmbeddingWithTags { .. } | KvSlotMode::EmbeddingWithMinimalTag { .. } => {
                if slot.end > slot.start + 1 {
                    engine.evict(slot.start + 1, slot.end);
                }
                slot.end = slot.start + 1;
                slot.n_tokens = 1;
                slot.mode = KvSlotMode::Embedding;
                slot.tier = KvTier::Gamma;
                Ok(())
            }
            KvSlotMode::Embedding => {
                // Already Gamma
                slot.tier = KvTier::Gamma;
                Ok(())
            }
            KvSlotMode::Token => Err(anyhow::anyhow!(
                "demote_to_gamma: node {:?} is Token mode, cannot demote (evict instead)",
                node_id
            )),
        }
    }

    /// Promote a node from Gamma (pure embedding) to Beta (embedding + label token).
    /// Encodes 1 label token at position start+1.
    pub fn promote_to_beta(&mut self, node_id: NodeId, label_text: &str, engine: &dyn Tokenizer) -> Result<()> {
        if !self.tier_budget.consume() {
            return Err(anyhow::anyhow!("promote_to_beta: tier budget exhausted ({} used)", self.tier_budget.used));
        }
        let slot = self.nodes.get_mut(&node_id)
            .ok_or_else(|| anyhow::anyhow!("promote_to_beta: node {:?} not in registry", node_id))?;
        match slot.mode {
            KvSlotMode::Embedding => {
                let label_tokens = engine.tokenize(label_text, false, true)?;
                let n_label = label_tokens.len().min(3) as i32; // cap label to 3 tokens
                if n_label > 0 {
                    let label_positions: Vec<i32> = ((slot.start + 1)..=(slot.start + n_label)).collect();
                    engine.encode(&label_tokens[..n_label as usize], &label_positions, 0)?;
                }
                slot.end = slot.start + 1 + n_label;
                slot.n_tokens = 1 + n_label;
                slot.mode = KvSlotMode::EmbeddingWithMinimalTag { n_label_tokens: n_label };
                slot.tier = KvTier::Beta;
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "promote_to_beta: node {:?} mode {:?} is not Embedding/Gamma",
                node_id, slot.mode
            )),
        }
    }

    /// Promote a node to Alpha (embedding + full micro-tag).
    /// Encodes tag tokens at positions start+1..start+1+n_tag.
    /// Works from Gamma (Embedding) or Beta (EmbeddingWithMinimalTag).
    pub fn promote_to_alpha(&mut self, node_id: NodeId, tag_text: &str, engine: &dyn Tokenizer) -> Result<()> {
        if !self.tier_budget.consume() {
            return Err(anyhow::anyhow!("promote_to_alpha: tier budget exhausted ({} used)", self.tier_budget.used));
        }
        let slot = self.nodes.get_mut(&node_id)
            .ok_or_else(|| anyhow::anyhow!("promote_to_alpha: node {:?} not in registry", node_id))?;
        match slot.mode {
            KvSlotMode::Embedding | KvSlotMode::EmbeddingWithMinimalTag { .. } => {
                // First, evict any existing tags beyond the embedding
                if slot.end > slot.start + 1 {
                    engine.evict(slot.start + 1, slot.end);
                }
                // Encode full tag
                let tag_tokens = engine.tokenize(tag_text, false, true)?;
                let n_tag = tag_tokens.len() as i32;
                if n_tag > 0 {
                    let tag_positions: Vec<i32> = ((slot.start + 1)..=(slot.start + n_tag)).collect();
                    engine.encode(&tag_tokens, &tag_positions, 0)?;
                }
                slot.end = slot.start + 1 + n_tag;
                slot.n_tokens = 1 + n_tag;
                slot.mode = KvSlotMode::EmbeddingWithTags { n_tag_tokens: n_tag };
                slot.tier = KvTier::Alpha;
                Ok(())
            }
            KvSlotMode::EmbeddingWithTags { .. } => {
                // Already Alpha
                slot.tier = KvTier::Alpha;
                Ok(())
            }
            KvSlotMode::Token => Err(anyhow::anyhow!(
                "promote_to_alpha: node {:?} is Token mode, cannot promote",
                node_id
            )),
        }
    }

    /// Get the tier of a node (if loaded).
    pub fn get_tier(&self, node_id: NodeId) -> Option<KvTier> {
        self.nodes.get(&node_id).map(|s| s.tier)
    }

    /// Count nodes by tier.
    pub fn tier_distribution(&self) -> (usize, usize, usize) {
        let mut alpha = 0;
        let mut beta = 0;
        let mut gamma = 0;
        for slot in self.nodes.values() {
            match slot.tier {
                KvTier::Alpha => alpha += 1,
                KvTier::Beta => beta += 1,
                KvTier::Gamma => gamma += 1,
            }
        }
        (alpha, beta, gamma)
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

    /// Start a new query — increment counter and reset tier budget.
    pub fn begin_query(&mut self) {
        self.query_counter += 1;
        self.metrics.total_queries += 1;
        self.tier_budget.reset();
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
            self.full_recompact::<fn(&[f32], &[i32], i32) -> Result<usize>>(engine, None, None);
        }

        let total_tokens: i32 = self.nodes.values().map(|s| s.n_tokens).sum();
        let available = max_kv_tokens - total_tokens;
        if available >= needed_tokens { return; }

        let to_free = needed_tokens - available;
        self.evict_lru(to_free, protected, engine);
    }

    /// Full recompact: clear everything and re-encode from scratch.
    ///
    /// For Token-mode nodes: re-tokenize from stored text.
    /// For Embedding/Tag nodes: requires `encode_embd_fn` to re-inject embeddings
    /// from the provided cache. If no cache, embedding nodes are re-encoded as text fallback.
    pub fn full_recompact<F>(
        &mut self,
        engine: &dyn Tokenizer,
        encode_embd_fn: Option<F>,
        get_embedding: Option<&dyn Fn(NodeId) -> Option<Vec<f32>>>,
    ) where
        F: Fn(&[f32], &[i32], i32) -> Result<usize>,
    {
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
        for nid in &self.order.clone() {
            if let Some(slot) = self.nodes.get_mut(nid) {
                let recompact_ok = match slot.mode {
                    KvSlotMode::Token => {
                        // Re-tokenize from stored text
                        if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                            let n_tok = tokens.len() as i32;
                            let positions: Vec<i32> = (pos..pos + n_tok).collect();
                            let _ = engine.encode(&tokens, &positions, 0);
                            slot.start = pos;
                            slot.end = pos + n_tok;
                            slot.n_tokens = n_tok;
                            pos += n_tok;
                            true
                        } else { false }
                    }
                    KvSlotMode::Embedding => {
                        // Re-inject embedding if available
                        if let (Some(embd_fn), Some(get_embd)) = (&encode_embd_fn, &get_embedding) {
                            if let Some(embd) = get_embd(*nid) {
                                let _ = embd_fn(&embd, &[pos], 0);
                                slot.start = pos;
                                slot.end = pos + 1;
                                slot.n_tokens = 1;
                                pos += 1;
                                true
                            } else { false }
                        } else { false }
                    }
                    KvSlotMode::EmbeddingWithMinimalTag { n_label_tokens } => {
                        if let (Some(embd_fn), Some(get_embd)) = (&encode_embd_fn, &get_embedding) {
                            if let Some(embd) = get_embd(*nid) {
                                // Re-inject embedding
                                let _ = embd_fn(&embd, &[pos], 0);
                                // Re-encode label tokens from text (first few tokens)
                                if n_label_tokens > 0 {
                                    if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                                        let n = (n_label_tokens as usize).min(tokens.len());
                                        let label_positions: Vec<i32> = ((pos + 1)..=(pos + n as i32)).collect();
                                        let _ = engine.encode(&tokens[..n], &label_positions, 0);
                                    }
                                }
                                let total = 1 + n_label_tokens;
                                slot.start = pos;
                                slot.end = pos + total;
                                slot.n_tokens = total;
                                pos += total;
                                true
                            } else { false }
                        } else { false }
                    }
                    KvSlotMode::EmbeddingWithTags { n_tag_tokens } => {
                        if let (Some(embd_fn), Some(get_embd)) = (&encode_embd_fn, &get_embedding) {
                            if let Some(embd) = get_embd(*nid) {
                                // Re-inject embedding
                                let _ = embd_fn(&embd, &[pos], 0);
                                // Re-encode tag tokens from text
                                if n_tag_tokens > 0 {
                                    if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                                        let n = (n_tag_tokens as usize).min(tokens.len());
                                        let tag_positions: Vec<i32> = ((pos + 1)..=(pos + n as i32)).collect();
                                        let _ = engine.encode(&tokens[..n], &tag_positions, 0);
                                    }
                                }
                                let total = 1 + n_tag_tokens;
                                slot.start = pos;
                                slot.end = pos + total;
                                slot.n_tokens = total;
                                pos += total;
                                true
                            } else { false }
                        } else { false }
                    }
                };

                // Fallback: if embedding re-injection failed, re-encode as text
                if !recompact_ok {
                    if let Ok(tokens) = engine.tokenize(&slot.text, false, true) {
                        let n_tok = tokens.len() as i32;
                        let positions: Vec<i32> = (pos..pos + n_tok).collect();
                        let _ = engine.encode(&tokens, &positions, 0);
                        slot.start = pos;
                        slot.end = pos + n_tok;
                        slot.n_tokens = n_tok;
                        slot.mode = KvSlotMode::Token;
                        slot.tier = KvTier::Alpha;
                        pos += n_tok;
                    }
                }
            }
        }
        self.next_pos = pos;
    }
}
