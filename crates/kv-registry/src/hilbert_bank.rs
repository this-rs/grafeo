//! Hilbert Bank Manager — segments of the Hilbert curve as loadable/evictable units.
//!
//! Each bank corresponds to a contiguous range on the Hilbert curve,
//! representing a topologically coherent group of graph nodes.
//! Banks can be loaded (encode embeddings in tier Γ) and evicted (seq_rm) as blocks.

use crate::Tokenizer;
use crate::hilbert::HilbertLayout;
use crate::registry::KvNodeRegistry;
use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet, VecDeque};

/// A segment of the Hilbert curve containing topologically related nodes.
#[derive(Debug, Clone)]
pub struct HilbertBank {
    /// Unique bank ID (0-based).
    pub id: usize,
    /// Human-readable name (e.g., community label or "bank_0").
    pub name: String,
    /// Hilbert position range [start, end) — contiguous on the curve.
    pub hilbert_start: u32,
    pub hilbert_end: u32,
    /// Node IDs in this bank, sorted by Hilbert position.
    pub node_ids: Vec<NodeId>,
    /// Whether this bank is currently loaded in the KV cache.
    pub loaded: bool,
    /// Bank importance score (updated by ablation reward feedback).
    pub importance: f32,
    /// Cumulative activation count across rounds.
    pub activation_count: u32,
}

impl HilbertBank {
    /// Number of nodes in this bank.
    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    /// Width of the Hilbert range.
    pub fn range_width(&self) -> u32 {
        self.hilbert_end.saturating_sub(self.hilbert_start)
    }
}

/// Manager for Hilbert-segmented banks.
#[derive(Debug)]
pub struct BankManager {
    /// All banks, indexed by bank ID.
    pub banks: Vec<HilbertBank>,
    /// Reverse map: NodeId → bank index.
    pub node_to_bank: HashMap<NodeId, usize>,
    /// Maximum number of banks loaded simultaneously. 0 = unlimited.
    pub max_loaded: usize,
    /// LRU order: front = least recently used, back = most recently used.
    /// Contains bank IDs of currently loaded banks.
    pub lru_order: VecDeque<usize>,
}

impl BankManager {
    /// Segment a HilbertLayout into banks based on community assignments.
    ///
    /// `communities`: maps NodeId → community_id (e.g., from Louvain clustering).
    /// Nodes in the same community form a bank.
    /// Nodes without a community assignment go into a "misc" bank.
    pub fn from_communities(layout: &HilbertLayout, communities: &HashMap<NodeId, usize>) -> Self {
        // Group nodes by community
        let mut community_nodes: HashMap<usize, Vec<(NodeId, u32)>> = HashMap::new();
        let mut orphan_nodes: Vec<(NodeId, u32)> = Vec::new();

        for (&nid, &pos) in &layout.positions {
            if let Some(&comm) = communities.get(&nid) {
                community_nodes.entry(comm).or_default().push((nid, pos));
            } else {
                orphan_nodes.push((nid, pos));
            }
        }

        let mut banks = Vec::new();
        let mut node_to_bank = HashMap::new();

        // Sort communities by their minimum Hilbert position for deterministic ordering
        let mut comm_ids: Vec<usize> = community_nodes.keys().copied().collect();
        comm_ids.sort_by_key(|&c| {
            community_nodes[&c]
                .iter()
                .map(|&(_, p)| p)
                .min()
                .unwrap_or(0)
        });

        for comm_id in comm_ids {
            let mut nodes = community_nodes.remove(&comm_id).unwrap();
            nodes.sort_by_key(|&(_, pos)| pos);

            let hilbert_start = nodes.first().map(|&(_, p)| p).unwrap_or(0);
            let hilbert_end = nodes.last().map(|&(_, p)| p + 1).unwrap_or(0);

            let bank_idx = banks.len();
            let node_ids: Vec<NodeId> = nodes.iter().map(|&(nid, _)| nid).collect();
            for &nid in &node_ids {
                node_to_bank.insert(nid, bank_idx);
            }

            banks.push(HilbertBank {
                id: bank_idx,
                name: format!("community_{comm_id}"),
                hilbert_start,
                hilbert_end,
                node_ids,
                loaded: false,
                importance: 0.0,
                activation_count: 0,
            });
        }

        // Orphan bank (if any)
        if !orphan_nodes.is_empty() {
            orphan_nodes.sort_by_key(|&(_, pos)| pos);
            let hilbert_start = orphan_nodes.first().map(|&(_, p)| p).unwrap_or(0);
            let hilbert_end = orphan_nodes.last().map(|&(_, p)| p + 1).unwrap_or(0);

            let bank_idx = banks.len();
            let node_ids: Vec<NodeId> = orphan_nodes.iter().map(|&(nid, _)| nid).collect();
            for &nid in &node_ids {
                node_to_bank.insert(nid, bank_idx);
            }

            banks.push(HilbertBank {
                id: bank_idx,
                name: "misc".to_string(),
                hilbert_start,
                hilbert_end,
                node_ids,
                loaded: false,
                importance: 0.0,
                activation_count: 0,
            });
        }

        Self {
            banks,
            node_to_bank,
            max_loaded: 0,
            lru_order: VecDeque::new(),
        }
    }

    /// Set maximum loaded banks. 0 = unlimited.
    pub fn with_max_loaded(mut self, max_loaded: usize) -> Self {
        self.max_loaded = max_loaded;
        self
    }

    /// Segment a HilbertLayout into banks of fixed size (when no community info).
    ///
    /// `chunk_size`: max nodes per bank.
    pub fn from_chunks(layout: &HilbertLayout, chunk_size: usize) -> Self {
        let sorted = layout.nodes_by_position();
        let chunk_size = chunk_size.max(1);

        let mut banks = Vec::new();
        let mut node_to_bank = HashMap::new();

        for (idx, chunk) in sorted.chunks(chunk_size).enumerate() {
            let hilbert_start = chunk.first().map(|&(_, p)| p).unwrap_or(0);
            let hilbert_end = chunk.last().map(|&(_, p)| p + 1).unwrap_or(0);
            let node_ids: Vec<NodeId> = chunk.iter().map(|&(nid, _)| nid).collect();

            for &nid in &node_ids {
                node_to_bank.insert(nid, idx);
            }

            banks.push(HilbertBank {
                id: idx,
                name: format!("bank_{idx}"),
                hilbert_start,
                hilbert_end,
                node_ids,
                loaded: false,
                importance: 0.0,
                activation_count: 0,
            });
        }

        Self {
            banks,
            node_to_bank,
            max_loaded: 0,
            lru_order: VecDeque::new(),
        }
    }

    /// Reset all banks to unloaded state. Call after `clear_kv()` to keep
    /// BankManager in sync with the actual KV cache state.
    pub fn reset_loaded(&mut self) {
        for bank in &mut self.banks {
            bank.loaded = false;
        }
        self.lru_order.clear();
    }

    /// Load a bank: encode all its nodes as embeddings (tier Γ) into the KV cache.
    ///
    /// If `max_loaded > 0` and loading this bank would exceed the limit,
    /// the least-recently-used bank is auto-evicted first.
    ///
    /// `get_embedding`: provides the embedding vector for a NodeId.
    /// `get_text`: provides the text label for a NodeId (stored in KvSlot.text).
    /// `encode_embd_fn`: the FFI function to inject embeddings into KV.
    /// `engine`: needed for LRU auto-eviction (seq_rm + pos resync).
    pub fn load_bank<F>(
        &mut self,
        bank_id: usize,
        registry: &mut KvNodeRegistry,
        get_embedding: &dyn Fn(NodeId) -> Option<Vec<f32>>,
        get_text: &dyn Fn(NodeId) -> String,
        encode_embd_fn: &F,
        engine: Option<&dyn Tokenizer>,
    ) -> Result<usize>
    where
        F: Fn(&[f32], &[i32], i32) -> Result<usize>,
    {
        if bank_id >= self.banks.len() {
            anyhow::bail!("load_bank: bank {bank_id} not found");
        }

        if self.banks[bank_id].loaded {
            // Already loaded — just touch LRU
            self.lru_touch(bank_id);
            return Ok(0);
        }

        // Auto-evict LRU if at capacity
        if self.max_loaded > 0 && self.loaded_count() >= self.max_loaded {
            if let Some(engine) = engine {
                self.evict_lru(registry, engine)?;
            } else {
                anyhow::bail!(
                    "load_bank: max_loaded={} reached but no engine for auto-evict",
                    self.max_loaded
                );
            }
        }

        let mut loaded = 0usize;
        for &nid in &self.banks[bank_id].node_ids.clone() {
            if registry.get_slot(nid).is_some() {
                continue; // Already in KV
            }
            if let Some(embd) = get_embedding(nid) {
                let text = get_text(nid);
                let embd_clone = embd.clone();
                registry.register_embedding(nid, &text, &embd_clone, |e, p, s| {
                    encode_embd_fn(e, p, s)
                })?;
                loaded += 1;
            }
        }

        let bank = &mut self.banks[bank_id];
        bank.loaded = true;
        kv_debug!(
            "  [D5] loaded bank '{}': {} nodes at positions [{}, {}]",
            bank.name, loaded, bank.hilbert_start, bank.hilbert_end
        );

        // Add to LRU (most recently used = back)
        self.lru_order.retain(|&id| id != bank_id);
        self.lru_order.push_back(bank_id);

        Ok(loaded)
    }

    /// Touch a bank in the LRU: move it to the back (most recently used).
    fn lru_touch(&mut self, bank_id: usize) {
        self.lru_order.retain(|&id| id != bank_id);
        self.lru_order.push_back(bank_id);
    }

    /// Evict the least-recently-used bank (front of LRU deque).
    /// Returns the number of nodes evicted, or 0 if no loaded bank.
    pub fn evict_lru(
        &mut self,
        registry: &mut KvNodeRegistry,
        engine: &dyn Tokenizer,
    ) -> Result<usize> {
        let victim = match self.lru_order.pop_front() {
            Some(id) => id,
            None => return Ok(0),
        };
        self.evict_bank(victim, registry, engine)
    }

    /// Evict a bank: remove all its nodes from the KV cache.
    pub fn evict_bank(
        &mut self,
        bank_id: usize,
        registry: &mut KvNodeRegistry,
        engine: &dyn Tokenizer,
    ) -> Result<usize> {
        let bank = self
            .banks
            .get(bank_id)
            .ok_or_else(|| anyhow::anyhow!("evict_bank: bank {bank_id} not found"))?;

        if !bank.loaded {
            return Ok(0);
        }

        let mut evicted = 0usize;
        let node_ids = bank.node_ids.clone();
        for &nid in &node_ids {
            if let Some(slot) = registry.nodes.remove(&nid) {
                engine.evict(slot.start, slot.end);
                evicted += 1;
            }
        }
        registry.order.retain(|id| !node_ids.contains(id));

        // Update next_pos to reflect KV state
        if evicted > 0 {
            registry.next_pos = engine.seq_pos_max(0) + 1;
        }

        let bank = &mut self.banks[bank_id];
        bank.loaded = false;
        kv_debug!(
            "  [D5] evicted bank '{}': {} nodes freed",
            bank.name, evicted
        );

        // Remove from LRU
        self.lru_order.retain(|&id| id != bank_id);

        Ok(evicted)
    }

    /// Update bank importance from ablation rewards.
    ///
    /// `bank_rewards`: maps bank_index → Δ_logprob reward.
    /// `decay`: exponential moving average decay (e.g., 0.7).
    pub fn update_importance(&mut self, bank_rewards: &HashMap<usize, f32>, decay: f32) {
        for (bank_idx, &reward) in bank_rewards {
            if let Some(bank) = self.banks.get_mut(*bank_idx) {
                bank.importance = decay * bank.importance + (1.0 - decay) * reward;
                bank.activation_count += 1;
            }
        }
    }

    /// Get banks sorted by importance (lowest first — candidates for eviction).
    pub fn eviction_candidates(&self) -> Vec<usize> {
        let mut loaded: Vec<(usize, f32)> = self
            .banks
            .iter()
            .filter(|b| b.loaded)
            .map(|b| (b.id, b.importance))
            .collect();
        loaded.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        loaded.into_iter().map(|(id, _)| id).collect()
    }

    /// Get the bank containing a given node.
    pub fn bank_for_node(&self, node_id: NodeId) -> Option<usize> {
        self.node_to_bank.get(&node_id).copied()
    }

    /// Number of banks.
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// Number of loaded banks.
    pub fn loaded_count(&self) -> usize {
        self.banks.iter().filter(|b| b.loaded).count()
    }

    /// Total nodes across all banks.
    pub fn total_nodes(&self) -> usize {
        self.banks.iter().map(|b| b.len()).sum()
    }

    /// Get bank IDs relevant for a set of node IDs (for selective loading).
    pub fn banks_for_nodes(&self, node_ids: &[NodeId]) -> HashSet<usize> {
        node_ids
            .iter()
            .filter_map(|nid| self.node_to_bank.get(nid).copied())
            .collect()
    }

    // ── Bank selection by cosine retrieval (T4.2) ─────────────────

    /// Compute the centroid embedding for each bank.
    ///
    /// The centroid is the element-wise mean of all node embeddings in the bank.
    /// Banks with no embeddings available are skipped (no entry in the result).
    ///
    /// `get_embedding`: function that returns the embedding for a NodeId
    /// (typically from `NodeEmbeddingCache::get()`).
    pub fn compute_bank_centroids<F>(
        &self,
        get_embedding: F,
    ) -> HashMap<usize, Vec<f32>>
    where
        F: Fn(NodeId) -> Option<Vec<f32>>,
    {
        let mut centroids = HashMap::new();

        for bank in &self.banks {
            let mut sum: Option<Vec<f32>> = None;
            let mut count = 0usize;

            for &nid in &bank.node_ids {
                if let Some(embd) = get_embedding(nid) {
                    match &mut sum {
                        Some(s) => {
                            for (i, &v) in embd.iter().enumerate() {
                                if i < s.len() {
                                    s[i] += v;
                                }
                            }
                        }
                        None => {
                            sum = Some(embd);
                        }
                    }
                    count += 1;
                }
            }

            if let Some(mut s) = sum {
                if count > 0 {
                    let inv = 1.0 / count as f32;
                    for v in &mut s {
                        *v *= inv;
                    }
                }
                centroids.insert(bank.id, s);
            }
        }

        centroids
    }

    /// Select the top-K most relevant banks for a query, with 1-hop expansion.
    ///
    /// 1. Compute cosine similarity between `query_embd` and each bank centroid.
    /// 2. Take the top-K by similarity.
    /// 3. Expand with 1-hop neighbor banks (banks that share edges with selected banks).
    ///
    /// `centroids`: pre-computed from `compute_bank_centroids()`.
    /// `adjacency`: inter-bank adjacency (bank_id → set of neighbor bank_ids).
    /// `k`: number of top banks to select before expansion.
    ///
    /// Returns bank IDs ordered by relevance (top-K first, then neighbors).
    pub fn select_banks(
        &self,
        query_embd: &[f32],
        centroids: &HashMap<usize, Vec<f32>>,
        adjacency: &HashMap<usize, HashSet<usize>>,
        k: usize,
    ) -> Vec<usize> {
        // Score each bank by cosine similarity
        let mut scores: Vec<(usize, f32)> = centroids
            .iter()
            .map(|(&bank_id, centroid)| {
                let sim = cosine_similarity(query_embd, centroid);
                (bank_id, sim)
            })
            .collect();

        // Sort by similarity (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Top-K
        let top_k: Vec<usize> = scores.iter().take(k).map(|&(id, _)| id).collect();

        // 1-hop expansion: add neighbor banks not already in top-K
        let mut result: Vec<usize> = top_k.clone();
        let top_k_set: HashSet<usize> = top_k.iter().copied().collect();

        for &bank_id in &top_k {
            if let Some(neighbors) = adjacency.get(&bank_id) {
                for &neighbor in neighbors {
                    if !top_k_set.contains(&neighbor) && !result.contains(&neighbor) {
                        result.push(neighbor);
                    }
                }
            }
        }

        result
    }

    /// Build inter-bank adjacency from node-level edges.
    ///
    /// Two banks are adjacent if any node in bank A has an edge to any node in bank B.
    /// `edges`: iterator of (source_node, target_node) pairs.
    pub fn build_bank_adjacency<I>(&self, edges: I) -> HashMap<usize, HashSet<usize>>
    where
        I: IntoIterator<Item = (NodeId, NodeId)>,
    {
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (src, tgt) in edges {
            if let (Some(&bank_a), Some(&bank_b)) = (
                self.node_to_bank.get(&src),
                self.node_to_bank.get(&tgt),
            ) {
                if bank_a != bank_b {
                    adjacency.entry(bank_a).or_default().insert(bank_b);
                    adjacency.entry(bank_b).or_default().insert(bank_a);
                }
            }
        }

        adjacency
    }

    /// E5: Re-segment banks from new communities (e.g., after Hilbert re-layout).
    ///
    /// Nodes currently loaded in the KV cache (`loaded_nodes`) keep their bank
    /// assignment until they are evicted. Only unloaded nodes migrate to their
    /// new community-based bank.
    ///
    /// Returns the number of nodes that changed bank assignment.
    pub fn resegment(
        &mut self,
        new_layout: &HilbertLayout,
        new_communities: &HashMap<NodeId, usize>,
        loaded_nodes: &HashSet<NodeId>,
    ) -> usize {
        // Build the new bank structure from new communities
        let new_mgr = Self::from_communities(new_layout, new_communities);

        // Track which loaded nodes stay in their original bank
        let mut migrated = 0usize;

        // For each node, decide: keep old assignment (if loaded) or use new
        let mut final_node_to_bank = HashMap::new();

        for (&nid, &new_bank_idx) in &new_mgr.node_to_bank {
            if loaded_nodes.contains(&nid) {
                // Keep old assignment if it exists
                if let Some(&old_bank_idx) = self.node_to_bank.get(&nid) {
                    final_node_to_bank.insert(nid, old_bank_idx);
                    continue;
                }
            }
            // Use new assignment
            if self.node_to_bank.get(&nid) != Some(&new_bank_idx) {
                migrated += 1;
            }
            final_node_to_bank.insert(nid, new_bank_idx);
        }

        // Replace banks with new structure, preserving loaded status for banks
        // that still exist (same index, same name)
        let old_loaded: HashMap<String, (bool, f32, u32)> = self
            .banks
            .iter()
            .map(|b| (b.name.clone(), (b.loaded, b.importance, b.activation_count)))
            .collect();

        self.banks = new_mgr.banks;
        self.node_to_bank = final_node_to_bank;

        // Restore loaded status for banks that match by name
        for bank in &mut self.banks {
            if let Some(&(loaded, importance, activation_count)) = old_loaded.get(&bank.name) {
                bank.loaded = loaded;
                bank.importance = importance;
                bank.activation_count = activation_count;
            }
        }

        // Rebuild node_ids in each bank from the final assignment
        for bank in &mut self.banks {
            bank.node_ids.clear();
        }
        for (&nid, &bank_idx) in &self.node_to_bank {
            if let Some(bank) = self.banks.get_mut(bank_idx) {
                bank.node_ids.push(nid);
            }
        }
        // Sort node_ids within each bank for determinism
        for bank in &mut self.banks {
            bank.node_ids.sort_by_key(|n| n.0);
        }

        // Rebuild LRU from loaded banks (preserve relative order where possible)
        let old_lru: Vec<usize> = self.lru_order.iter().copied().collect();
        self.lru_order.clear();
        // Re-add banks that are still loaded and still exist
        for id in &old_lru {
            if self.banks.get(*id).map_or(false, |b| b.loaded) {
                self.lru_order.push_back(*id);
            }
        }
        // Add any loaded banks not in old LRU (newly restored from name match)
        for bank in &self.banks {
            if bank.loaded && !self.lru_order.contains(&bank.id) {
                self.lru_order.push_back(bank.id);
            }
        }

        if migrated > 0 {
            kv_debug!(
                "  [E5] bank resegment: {} nodes migrated, {} banks",
                migrated,
                self.banks.len()
            );
        }

        migrated
    }
}

/// Cosine similarity between two vectors. Returns 0.0 if either has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_layout() -> (HilbertLayout, HashMap<NodeId, usize>) {
        // 9 nodes, 3 communities of 3
        let mut positions = HashMap::new();
        let mut coords_2d = HashMap::new();
        let mut communities = HashMap::new();

        for i in 0u64..9 {
            let nid = NodeId(i);
            positions.insert(nid, i as u32 + 10); // base_pos=10
            coords_2d.insert(nid, (0.0, 0.0));
            communities.insert(nid, (i / 3) as usize); // 3 communities of 3
        }

        let layout = HilbertLayout {
            positions,
            order: 2,
            coords_2d,
        };
        (layout, communities)
    }

    #[test]
    fn test_from_communities() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        assert_eq!(mgr.len(), 3);
        assert_eq!(mgr.total_nodes(), 9);

        // Each bank has 3 nodes
        for bank in &mgr.banks {
            assert_eq!(bank.len(), 3);
            assert!(!bank.loaded);
            assert_eq!(bank.importance, 0.0);
        }

        // Reverse map works
        assert_eq!(mgr.bank_for_node(NodeId(0)), Some(0));
        assert_eq!(mgr.bank_for_node(NodeId(4)), Some(1));
        assert_eq!(mgr.bank_for_node(NodeId(8)), Some(2));
        assert_eq!(mgr.bank_for_node(NodeId(99)), None);
    }

    #[test]
    fn test_from_chunks() {
        let (layout, _) = make_test_layout();
        let mgr = BankManager::from_chunks(&layout, 4);

        // 9 nodes / 4 = 3 banks (4, 4, 1)
        assert_eq!(mgr.len(), 3);
        assert_eq!(mgr.banks[0].len(), 4);
        assert_eq!(mgr.banks[1].len(), 4);
        assert_eq!(mgr.banks[2].len(), 1);
    }

    #[test]
    fn test_eviction_candidates() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);

        // Mark all as loaded with different importance
        mgr.banks[0].loaded = true;
        mgr.banks[0].importance = 0.8;
        mgr.banks[1].loaded = true;
        mgr.banks[1].importance = 0.2;
        mgr.banks[2].loaded = true;
        mgr.banks[2].importance = 0.5;

        let candidates = mgr.eviction_candidates();
        // Lowest importance first
        assert_eq!(candidates, vec![1, 2, 0]);
    }

    #[test]
    fn test_update_importance() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);

        let mut rewards = HashMap::new();
        rewards.insert(0, 0.5f32);
        rewards.insert(1, -0.3f32);

        mgr.update_importance(&rewards, 0.7);

        // bank 0: 0.7 * 0.0 + 0.3 * 0.5 = 0.15
        assert!((mgr.banks[0].importance - 0.15).abs() < 0.001);
        // bank 1: 0.7 * 0.0 + 0.3 * (-0.3) = -0.09
        assert!((mgr.banks[1].importance - (-0.09)).abs() < 0.001);
        // bank 2: unchanged
        assert_eq!(mgr.banks[2].importance, 0.0);
    }

    #[test]
    fn test_banks_for_nodes() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        let relevant = mgr.banks_for_nodes(&[NodeId(0), NodeId(5), NodeId(8)]);
        assert_eq!(relevant.len(), 3); // All 3 communities
        assert!(relevant.contains(&0));
        assert!(relevant.contains(&1));
        assert!(relevant.contains(&2));
    }

    #[test]
    fn test_resegment_preserves_loaded() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);

        // Mark bank 0 as loaded
        mgr.banks[0].loaded = true;
        mgr.banks[0].importance = 0.9;

        // Loaded nodes = bank 0's nodes
        let loaded: HashSet<NodeId> = mgr.banks[0].node_ids.iter().copied().collect();

        // New communities: shift node 3 from community 1 → community 0
        let mut new_communities = communities.clone();
        new_communities.insert(NodeId(3), 0);

        // Resegment
        let migrated = mgr.resegment(&layout, &new_communities, &loaded);

        // Loaded nodes (bank 0) should keep their bank assignment
        for &nid in &loaded {
            assert_eq!(
                mgr.bank_for_node(nid),
                Some(0),
                "Loaded node {:?} should stay in bank 0",
                nid
            );
        }

        // Node 3 was unloaded → should have migrated
        assert!(migrated > 0, "At least one node should have migrated");
    }

    #[test]
    fn test_disjoint_hilbert_ranges() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        // Banks should have non-overlapping ranges (since communities are Hilbert-sorted)
        for i in 0..mgr.len() {
            for j in (i + 1)..mgr.len() {
                let a = &mgr.banks[i];
                let b = &mgr.banks[j];
                // They shouldn't overlap (start_a < end_b && start_b < end_a)
                let overlap = a.hilbert_start < b.hilbert_end && b.hilbert_start < a.hilbert_end;
                // In our test setup with sequential positions, communities are contiguous
                assert!(
                    !overlap,
                    "Banks {} and {} overlap: [{},{}) vs [{},{})",
                    i, j, a.hilbert_start, a.hilbert_end, b.hilbert_start, b.hilbert_end
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // LRU eviction tests
    // ─────────────────────────────────────────────────────────────────

    use std::cell::Cell;

    /// Mock Tokenizer that tracks evictions for testing.
    struct MockTokenizer {
        pos_max: Cell<i32>,
    }

    impl MockTokenizer {
        fn new(pos_max: i32) -> Self {
            Self {
                pos_max: Cell::new(pos_max),
            }
        }
    }

    impl crate::Tokenizer for MockTokenizer {
        fn tokenize(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<i32>> {
            Ok(vec![1]) // dummy
        }
        fn encode(&self, _tokens: &[i32], _positions: &[i32], _seq_id: i32) -> Result<()> {
            Ok(())
        }
        fn token_count(&self, _text: &str) -> Result<usize> {
            Ok(1)
        }
        fn evict(&self, _start: i32, _end: i32) {
            // Simulate eviction by decrementing pos_max
            let current = self.pos_max.get();
            self.pos_max.set((current - 1).max(-1));
        }
        fn clear_kv(&self) {
            self.pos_max.set(-1);
        }
        fn n_ctx(&self) -> u32 {
            8192
        }
        fn seq_pos_max(&self, _seq_id: i32) -> i32 {
            self.pos_max.get()
        }
    }

    /// Helper: make a registry for testing load/evict.
    fn make_test_registry() -> KvNodeRegistry {
        KvNodeRegistry::new("test header", 100) // header_end = 100
    }

    /// Dummy embedding provider: returns [1.0; 4] for any node.
    fn dummy_embedding(_nid: NodeId) -> Option<Vec<f32>> {
        Some(vec![1.0, 2.0, 3.0, 4.0])
    }

    fn dummy_text(nid: NodeId) -> String {
        format!("node_{}", nid.0)
    }

    /// Dummy encode function that just succeeds.
    fn dummy_encode(_embd: &[f32], _pos: &[i32], _seq_id: i32) -> Result<usize> {
        Ok(1)
    }

    #[test]
    fn test_lru_load_3_banks_max_2_evicts_oldest() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities)
            .with_max_loaded(2);

        let mut registry = make_test_registry();
        let engine = MockTokenizer::new(99); // starts at pos 99

        // Load bank 0 → OK (1/2)
        let loaded = mgr
            .load_bank(0, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine))
            .unwrap();
        assert!(loaded > 0);
        assert!(mgr.banks[0].loaded);
        assert_eq!(mgr.loaded_count(), 1);
        assert_eq!(mgr.lru_order.len(), 1);
        assert_eq!(mgr.lru_order[0], 0);

        // Load bank 1 → OK (2/2)
        let loaded = mgr
            .load_bank(1, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine))
            .unwrap();
        assert!(loaded > 0);
        assert!(mgr.banks[1].loaded);
        assert_eq!(mgr.loaded_count(), 2);
        assert_eq!(mgr.lru_order.len(), 2);

        // Load bank 2 → at capacity, should auto-evict bank 0 (LRU front)
        let loaded = mgr
            .load_bank(2, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine))
            .unwrap();
        assert!(loaded > 0);
        assert!(!mgr.banks[0].loaded, "Bank 0 should have been evicted (LRU)");
        assert!(mgr.banks[1].loaded, "Bank 1 should still be loaded");
        assert!(mgr.banks[2].loaded, "Bank 2 should be loaded");
        assert_eq!(mgr.loaded_count(), 2, "Should not exceed max_loaded=2");
        assert_eq!(mgr.lru_order.len(), 2);
        // LRU order: bank 1 (older), bank 2 (newest)
        assert_eq!(mgr.lru_order[0], 1);
        assert_eq!(mgr.lru_order[1], 2);
    }

    #[test]
    fn test_lru_touch_on_reload_prevents_eviction() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities)
            .with_max_loaded(2);

        let mut registry = make_test_registry();
        let engine = MockTokenizer::new(99);

        // Load bank 0, then bank 1
        mgr.load_bank(0, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();
        mgr.load_bank(1, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();

        // "Touch" bank 0 by re-loading it (already loaded → just LRU touch)
        let loaded = mgr
            .load_bank(0, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine))
            .unwrap();
        assert_eq!(loaded, 0, "Already loaded, should return 0");

        // Now LRU order should be: bank 1 (front/oldest), bank 0 (back/newest)
        assert_eq!(mgr.lru_order[0], 1);
        assert_eq!(mgr.lru_order[1], 0);

        // Load bank 2 → should evict bank 1 (LRU front), NOT bank 0
        mgr.load_bank(2, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();

        assert!(mgr.banks[0].loaded, "Bank 0 was touched, should survive");
        assert!(!mgr.banks[1].loaded, "Bank 1 should be evicted (LRU)");
        assert!(mgr.banks[2].loaded, "Bank 2 just loaded");
    }

    #[test]
    fn test_unlimited_max_loaded() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);
        // max_loaded = 0 (default) = unlimited

        let mut registry = make_test_registry();

        // Load all 3 banks without engine (no auto-evict needed)
        for bank_id in 0..3 {
            mgr.load_bank(bank_id, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, None).unwrap();
        }
        assert_eq!(mgr.loaded_count(), 3);
        assert_eq!(mgr.lru_order.len(), 3);
    }

    #[test]
    fn test_evict_lru_explicit() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);

        let mut registry = make_test_registry();
        let engine = MockTokenizer::new(99);

        // Load banks 0 and 1
        mgr.load_bank(0, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();
        mgr.load_bank(1, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();

        // Explicitly evict LRU
        let evicted = mgr.evict_lru(&mut registry, &engine).unwrap();
        assert!(evicted > 0, "Should evict nodes from bank 0");
        assert!(!mgr.banks[0].loaded, "Bank 0 (LRU) should be evicted");
        assert!(mgr.banks[1].loaded, "Bank 1 should remain");
        assert_eq!(mgr.lru_order.len(), 1);
        assert_eq!(mgr.lru_order[0], 1);
    }

    #[test]
    fn test_evict_bank_removes_from_lru() {
        let (layout, communities) = make_test_layout();
        let mut mgr = BankManager::from_communities(&layout, &communities);

        let mut registry = make_test_registry();
        let engine = MockTokenizer::new(99);

        mgr.load_bank(0, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();
        mgr.load_bank(1, &mut registry, &dummy_embedding, &dummy_text, &dummy_encode, Some(&engine)).unwrap();
        assert_eq!(mgr.lru_order.len(), 2);

        // Evict bank 1 directly (not through LRU)
        mgr.evict_bank(1, &mut registry, &engine).unwrap();
        assert!(!mgr.banks[1].loaded);
        assert_eq!(mgr.lru_order.len(), 1, "Bank 1 should be removed from LRU");
        assert_eq!(mgr.lru_order[0], 0, "Only bank 0 remains in LRU");
    }

    // ─────────────────────────────────────────────────────────────────
    // Bank selection / centroid tests (T4.2)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_compute_bank_centroids() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        // Embeddings: each node gets [node_id, 0, 0, 0]
        let centroids = mgr.compute_bank_centroids(|nid: NodeId| {
            Some(vec![nid.0 as f32, 0.0, 0.0, 0.0])
        });

        assert_eq!(centroids.len(), 3, "3 banks → 3 centroids");

        // Bank 0: nodes 0,1,2 → centroid = [1.0, 0, 0, 0]
        let c0 = &centroids[&0];
        assert_eq!(c0.len(), 4);
        assert!((c0[0] - 1.0).abs() < 0.01, "mean of 0,1,2 = 1.0, got {}", c0[0]);

        // Bank 1: nodes 3,4,5 → centroid = [4.0, 0, 0, 0]
        let c1 = &centroids[&1];
        assert!((c1[0] - 4.0).abs() < 0.01, "mean of 3,4,5 = 4.0, got {}", c1[0]);

        // Bank 2: nodes 6,7,8 → centroid = [7.0, 0, 0, 0]
        let c2 = &centroids[&2];
        assert!((c2[0] - 7.0).abs() < 0.01, "mean of 6,7,8 = 7.0, got {}", c2[0]);
    }

    #[test]
    fn test_compute_bank_centroids_partial_embeddings() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        // Only return embeddings for even nodes
        let centroids = mgr.compute_bank_centroids(|nid: NodeId| {
            if nid.0 % 2 == 0 {
                Some(vec![nid.0 as f32, 1.0])
            } else {
                None
            }
        });

        // Bank 0 has nodes 0,1,2 → only 0,2 have embeddings → centroid = [1.0, 1.0]
        let c0 = &centroids[&0];
        assert!((c0[0] - 1.0).abs() < 0.01, "mean of 0,2 = 1.0");

        // Bank 1 has nodes 3,4,5 → only 4 has embedding → centroid = [4.0, 1.0]
        let c1 = &centroids[&1];
        assert!((c1[0] - 4.0).abs() < 0.01, "only node 4");
    }

    #[test]
    fn test_select_banks_top_k() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        // Centroids: bank0=[1,0,0], bank1=[0,1,0], bank2=[0,0,1]
        let mut centroids = HashMap::new();
        centroids.insert(0, vec![1.0, 0.0, 0.0]);
        centroids.insert(1, vec![0.0, 1.0, 0.0]);
        centroids.insert(2, vec![0.0, 0.0, 1.0]);

        let no_adjacency = HashMap::new();

        // Query close to bank 0
        let selected = mgr.select_banks(&[0.9, 0.1, 0.0], &centroids, &no_adjacency, 1);
        assert_eq!(selected[0], 0, "Bank 0 is closest to [0.9, 0.1, 0.0]");
        assert_eq!(selected.len(), 1, "K=1, no adjacency → just 1 bank");

        // Query close to bank 2
        let selected = mgr.select_banks(&[0.0, 0.0, 1.0], &centroids, &no_adjacency, 1);
        assert_eq!(selected[0], 2, "Bank 2 is closest to [0,0,1]");

        // K=2
        let selected = mgr.select_banks(&[0.5, 0.5, 0.0], &centroids, &no_adjacency, 2);
        assert_eq!(selected.len(), 2);
        // Both bank 0 and bank 1 should be selected (equal cosine to [0.5, 0.5, 0])
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_select_banks_with_1hop_expansion() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        let mut centroids = HashMap::new();
        centroids.insert(0, vec![1.0, 0.0, 0.0]);
        centroids.insert(1, vec![0.0, 1.0, 0.0]);
        centroids.insert(2, vec![0.0, 0.0, 1.0]);

        // Bank 0 → Bank 1 are neighbors
        let mut adjacency = HashMap::new();
        adjacency.insert(0, HashSet::from([1]));
        adjacency.insert(1, HashSet::from([0]));

        // Query selects bank 0 (K=1), expansion adds bank 1
        let selected = mgr.select_banks(&[1.0, 0.0, 0.0], &centroids, &adjacency, 1);
        assert_eq!(selected.len(), 2, "K=1 + 1 neighbor");
        assert_eq!(selected[0], 0, "Top-K first");
        assert_eq!(selected[1], 1, "Neighbor second");
    }

    #[test]
    fn test_build_bank_adjacency() {
        let (layout, communities) = make_test_layout();
        let mgr = BankManager::from_communities(&layout, &communities);

        // Edges: node 2 (bank 0) → node 3 (bank 1), node 5 (bank 1) → node 6 (bank 2)
        let edges = vec![
            (NodeId(2), NodeId(3)),
            (NodeId(5), NodeId(6)),
        ];

        let adjacency = mgr.build_bank_adjacency(edges);

        // Bank 0 ↔ Bank 1
        assert!(adjacency[&0].contains(&1));
        assert!(adjacency[&1].contains(&0));
        // Bank 1 ↔ Bank 2
        assert!(adjacency[&1].contains(&2));
        assert!(adjacency[&2].contains(&1));
        // Bank 0 ↔ Bank 2: NOT adjacent (no direct edge)
        assert!(!adjacency.get(&0).map_or(false, |s| s.contains(&2)));
    }

    #[test]
    fn test_select_banks_performance_10_banks() {
        // Simulate 10 banks with random centroids
        let mut centroids = HashMap::new();
        for i in 0..10 {
            let mut c = vec![0.0f32; 64];
            c[i % 64] = 1.0; // Sparse orthogonal centroids
            centroids.insert(i, c);
        }

        // Build a simple BankManager with 10 banks
        let mut positions = HashMap::new();
        let mut coords_2d = HashMap::new();
        let mut communities = HashMap::new();
        for i in 0u64..100 {
            let nid = NodeId(i);
            positions.insert(nid, i as u32);
            coords_2d.insert(nid, (0.0, 0.0));
            communities.insert(nid, (i / 10) as usize);
        }
        let layout = HilbertLayout {
            positions,
            order: 4,
            coords_2d,
        };
        let mgr = BankManager::from_communities(&layout, &communities);
        let adjacency = HashMap::new();

        let query = vec![0.0f32; 64]; // zero query

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = mgr.select_banks(&query, &centroids, &adjacency, 3);
        }
        let elapsed = start.elapsed();

        // 1000 iterations of select_banks(10 banks) should be < 10ms total
        assert!(
            elapsed.as_millis() < 100, // generous margin for CI
            "1000× select_banks took {}ms (budget: <10ms total)",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_cosine_similarity_basic() {
        use super::cosine_similarity;

        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);
        // Orthogonal
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 0.001);
        // Opposite
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 0.001);
        // Zero vector
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
