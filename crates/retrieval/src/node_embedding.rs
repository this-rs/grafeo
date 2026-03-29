//! NodeEmbeddingCache — pre-computed embeddings for graph nodes (Phase C).
//!
//! Each graph node gets a dense embedding ∈ R^n_embd computed by extracting
//! the LLM's last-layer hidden state from the node's text. When a FactGNN
//! is available, the text embedding is fused with the GNN's topology embedding
//! via ProjectionNet: concat(h, g) → ProjectionNet → p ∈ R^n_embd.
//!
//! The fused embedding carries both semantic content (from LLM) and structural
//! position (from GNN message-passing), then gets injected into the KV cache
//! via `batch.embd` (1 position per node instead of ~50 tokens).

use std::collections::HashMap;
use anyhow::{Result, bail};
use obrain_common::types::NodeId;
use crate::ProjectionNet;

/// Cache of pre-computed node embeddings.
///
/// Each entry maps a `NodeId` to a dense float vector of size `n_embd`.
/// The cache is invalidated when the model changes (different `model_name`).
pub struct NodeEmbeddingCache {
    /// NodeId → embedding vector (f32, length = n_embd).
    cache: HashMap<NodeId, Vec<f32>>,
    /// Model embedding dimension.
    n_embd: usize,
    /// Model identifier — cache is invalidated when this changes.
    model_name: String,
}

impl NodeEmbeddingCache {
    /// Create a new empty cache for a given model.
    pub fn new(n_embd: usize, model_name: &str) -> Self {
        Self {
            cache: HashMap::new(),
            n_embd,
            model_name: model_name.to_string(),
        }
    }

    /// Check if we have an embedding for this node.
    pub fn has(&self, node_id: NodeId) -> bool {
        self.cache.contains_key(&node_id)
    }

    /// Get the embedding for a node (None if not cached).
    pub fn get(&self, node_id: NodeId) -> Option<&[f32]> {
        self.cache.get(&node_id).map(|v| v.as_slice())
    }

    /// Insert a pre-computed embedding.
    ///
    /// Returns an error if the embedding dimension doesn't match `n_embd`.
    pub fn insert(&mut self, node_id: NodeId, embedding: Vec<f32>) -> Result<()> {
        if embedding.len() != self.n_embd {
            bail!(
                "NodeEmbeddingCache::insert: embedding.len()={} but n_embd={}",
                embedding.len(), self.n_embd
            );
        }
        self.cache.insert(node_id, embedding);
        Ok(())
    }

    /// Remove a node's embedding (e.g. when node text changes).
    pub fn invalidate(&mut self, node_id: NodeId) {
        self.cache.remove(&node_id);
    }

    /// Clear all cached embeddings.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached embeddings.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the model name this cache was built for.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the embedding dimension.
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Check if this cache is valid for the given model.
    /// Returns false if the model changed (cache should be rebuilt).
    pub fn is_valid_for(&self, model_name: &str) -> bool {
        self.model_name == model_name
    }

    /// Iterate over all cached (NodeId, embedding) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &[f32])> {
        self.cache.iter().map(|(nid, emb)| (*nid, emb.as_slice()))
    }
}

/// Compute the text embedding for a string using the LLM's last-layer hidden state.
///
/// This tokenizes `text`, encodes it on a temporary sequence, extracts the
/// hidden state of the last token, then cleans up. The result is a Vec<f32>
/// of size `n_embd`.
///
/// **Important**: This temporarily enables embedding output on the engine
/// and uses seq_id=2 to avoid interfering with persistent (0) or query (1) sequences.
pub fn compute_text_embedding(engine: &crate::Engine, text: &str) -> Result<Vec<f32>> {
    let n_embd = engine.n_embd();
    if n_embd == 0 {
        bail!("compute_text_embedding: n_embd is 0");
    }

    let tokens = engine.tokenize(text, false, true)?;
    if tokens.is_empty() {
        bail!("compute_text_embedding: tokenization produced 0 tokens for text: {:?}", &text[..text.len().min(50)]);
    }

    // Use seq_id=2 (ablation/temp) to avoid polluting seq 0/1
    let seq_id = 2i32;

    // Clear any previous data on this seq
    engine.clear_seq(seq_id);

    // Enable embedding extraction
    engine.set_embeddings(true);

    // Encode tokens at positions 0..n
    let positions: Vec<i32> = (0..tokens.len() as i32).collect();
    engine.encode(&tokens, &positions, seq_id)?;

    // Extract last-layer hidden state of the last token
    let hidden = engine.get_embedding(-1);
    if hidden.is_empty() {
        engine.set_embeddings(false);
        engine.clear_seq(seq_id);
        bail!("compute_text_embedding: get_embedding returned empty (embeddings not enabled?)");
    }

    let result = hidden.to_vec();

    // Cleanup
    engine.set_embeddings(false);
    engine.clear_seq(seq_id);

    Ok(result)
}

/// Compute a fused embedding for a single node: concat(text_h, gnn_g) → ProjectionNet → p.
///
/// If `gnn_embedding` is None, falls back to text-only embedding (no projection).
pub fn compute_fused_embedding(
    engine: &crate::Engine,
    text: &str,
    gnn_embedding: Option<&[f32]>,
    projection_net: Option<&ProjectionNet>,
) -> Result<Vec<f32>> {
    let text_h = compute_text_embedding(engine, text)?;

    match (gnn_embedding, projection_net) {
        (Some(g), Some(pnet)) => {
            // Fuse: concat(h, g) → ProjectionNet → p ∈ R^n_embd
            let mut fused_input = Vec::with_capacity(text_h.len() + g.len());
            fused_input.extend_from_slice(&text_h);
            fused_input.extend_from_slice(g);
            pnet.forward(&fused_input)
        }
        _ => {
            // No GNN or no ProjectionNet — use raw text embedding
            Ok(text_h)
        }
    }
}

/// Context for GNN-fused embedding computation.
pub struct FusionContext<'a> {
    /// GNN node embeddings: NodeId → [f32; 64] (post message-passing).
    pub gnn_embeddings: HashMap<NodeId, Vec<f32>>,
    /// ProjectionNet for concat(h, g) → p.
    pub projection_net: &'a ProjectionNet,
}

/// Compute embeddings for multiple nodes in batch.
///
/// For each `(NodeId, text)` pair, computes the embedding and inserts it
/// into the cache. If a `FusionContext` is provided, uses GNN-fused embeddings
/// via ProjectionNet. Otherwise falls back to text-only embeddings.
///
/// Existing entries are skipped unless `force` is true.
/// Returns the number of newly computed embeddings.
pub fn compute_node_embeddings(
    engine: &crate::Engine,
    cache: &mut NodeEmbeddingCache,
    nodes: &[(NodeId, String)],
    force: bool,
) -> Result<usize> {
    compute_node_embeddings_with_fusion(engine, cache, nodes, force, None)
}

/// Compute node embeddings with optional GNN fusion.
pub fn compute_node_embeddings_with_fusion(
    engine: &crate::Engine,
    cache: &mut NodeEmbeddingCache,
    nodes: &[(NodeId, String)],
    force: bool,
    fusion: Option<&FusionContext>,
) -> Result<usize> {
    let mut computed = 0usize;

    for (node_id, text) in nodes {
        if !force && cache.has(*node_id) {
            continue;
        }

        let gnn_g = fusion.and_then(|f| f.gnn_embeddings.get(node_id).map(|v| v.as_slice()));
        let pnet = fusion.map(|f| f.projection_net);

        match compute_fused_embedding(engine, text, gnn_g, pnet) {
            Ok(embedding) => {
                cache.insert(*node_id, embedding)?;
                computed += 1;
            }
            Err(e) => {
                eprintln!("  ⚠ compute_fused_embedding failed for node {:?}: {}", node_id, e);
            }
        }
    }

    Ok(computed)
}

// ── Phase D6: Online GNN re-scoring on tier transitions ─────────

/// Re-score a node's embedding after tier promotion.
///
/// When a node is promoted (Γ→Β→Α), its embedding is recomputed with
/// the latest fact_score (activation frequency) factored in:
///   embd_weighted[i] = embd_base[i] * (0.6 * fact_score + 0.4)
///
/// If a GNN embedding is available, it's fused via ProjectionNet.
/// The result is updated in-place in the embedding cache.
///
/// Returns true if the embedding was successfully updated.
pub fn rescore_on_promote(
    node_id: NodeId,
    fact_score: f32,
    cache: &mut NodeEmbeddingCache,
    gnn_embedding: Option<&[f32]>,
    projection_net: Option<&ProjectionNet>,
) -> bool {
    // Get current embedding as base
    let base = match cache.get(node_id) {
        Some(e) => e.to_vec(),
        None => return false,
    };

    let n_embd = cache.n_embd();

    // Weight by fact_score: blend between base (0.4 floor) and fully activated (1.0)
    let weight = 0.6 * fact_score + 0.4;
    let mut weighted: Vec<f32> = base.iter().map(|&x| x * weight).collect();

    // If GNN embedding + ProjectionNet available, fuse
    if let (Some(g), Some(pnet)) = (gnn_embedding, projection_net) {
        let mut fused_input = Vec::with_capacity(n_embd + g.len());
        fused_input.extend_from_slice(&weighted);
        fused_input.extend_from_slice(g);
        if let Ok(projected) = pnet.forward(&fused_input) {
            weighted = projected;
        }
    }

    // Update cache in-place
    cache.insert(node_id, weighted).is_ok()
}

/// Propagate re-scoring to neighbors of a promoted node (1-hop).
///
/// For each neighbor, applies `rescore_on_promote` with their own fact_score.
/// Limited to `max_neighbors` to bound latency.
///
/// Returns the number of neighbors re-scored.
pub fn propagate_neighbors(
    center_node: NodeId,
    adjacency: &std::collections::HashMap<NodeId, std::collections::HashSet<NodeId>>,
    fact_scores: &std::collections::HashMap<NodeId, f32>,
    cache: &mut NodeEmbeddingCache,
    gnn_embeddings: Option<&std::collections::HashMap<NodeId, Vec<f32>>>,
    projection_net: Option<&ProjectionNet>,
    max_neighbors: usize,
) -> usize {
    let neighbors = match adjacency.get(&center_node) {
        Some(n) => n,
        None => return 0,
    };

    let mut rescored = 0usize;
    for &neighbor in neighbors.iter().take(max_neighbors) {
        if !cache.has(neighbor) {
            continue;
        }
        let fact_score = fact_scores.get(&neighbor).copied().unwrap_or(0.0);
        let gnn_e = gnn_embeddings.and_then(|m| m.get(&neighbor).map(|v| v.as_slice()));

        if rescore_on_promote(neighbor, fact_score, cache, gnn_e, projection_net) {
            rescored += 1;
        }
    }

    rescored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_ops() {
        let mut cache = NodeEmbeddingCache::new(4, "test-model");
        let nid = NodeId::from(42u64);

        assert!(!cache.has(nid));
        assert!(cache.get(nid).is_none());
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Insert
        cache.insert(nid, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(cache.has(nid));
        assert_eq!(cache.get(nid).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.len(), 1);

        // Wrong dimension
        let nid2 = NodeId::from(43u64);
        assert!(cache.insert(nid2, vec![1.0, 2.0]).is_err());

        // Invalidate
        cache.invalidate(nid);
        assert!(!cache.has(nid));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_model_validation() {
        let cache = NodeEmbeddingCache::new(4, "qwen3-14b");
        assert!(cache.is_valid_for("qwen3-14b"));
        assert!(!cache.is_valid_for("llama-3.2-3b"));
    }

    #[test]
    fn test_rescore_on_promote() {
        let mut cache = NodeEmbeddingCache::new(4, "test");
        let nid = NodeId::from(1u64);
        cache.insert(nid, vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // fact_score = 1.0 → weight = 0.6*1.0 + 0.4 = 1.0 → no change
        let ok = rescore_on_promote(nid, 1.0, &mut cache, None, None);
        assert!(ok);
        assert_eq!(cache.get(nid).unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        // fact_score = 0.0 → weight = 0.4 → all values * 0.4
        cache.insert(nid, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        rescore_on_promote(nid, 0.0, &mut cache, None, None);
        let embd = cache.get(nid).unwrap();
        assert!((embd[0] - 0.4).abs() < 0.001);
        assert!((embd[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_rescore_missing_node() {
        let mut cache = NodeEmbeddingCache::new(4, "test");
        let ok = rescore_on_promote(NodeId::from(99u64), 1.0, &mut cache, None, None);
        assert!(!ok);
    }

    #[test]
    fn test_propagate_neighbors() {
        use std::collections::{HashMap, HashSet};

        let mut cache = NodeEmbeddingCache::new(4, "test");
        let center = NodeId::from(0u64);
        let n1 = NodeId::from(1u64);
        let n2 = NodeId::from(2u64);
        let n3 = NodeId::from(3u64);

        cache.insert(center, vec![1.0; 4]).unwrap();
        cache.insert(n1, vec![2.0; 4]).unwrap();
        cache.insert(n2, vec![3.0; 4]).unwrap();
        // n3 not in cache — should be skipped

        let mut adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        adjacency.entry(center).or_default().insert(n1);
        adjacency.entry(center).or_default().insert(n2);
        adjacency.entry(center).or_default().insert(n3);

        let mut fact_scores = HashMap::new();
        fact_scores.insert(n1, 0.5f32);
        fact_scores.insert(n2, 1.0f32);

        let rescored = propagate_neighbors(
            center, &adjacency, &fact_scores, &mut cache, None, None, 20,
        );
        assert_eq!(rescored, 2); // n1 + n2 rescored, n3 skipped (not in cache)

        // n1 was rescored with fact_score=0.5 → weight=0.7 → 2.0 * 0.7 = 1.4
        let e1 = cache.get(n1).unwrap();
        assert!((e1[0] - 1.4).abs() < 0.001);
    }
}
