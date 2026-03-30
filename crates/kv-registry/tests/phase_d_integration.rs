//! Phase D Integration Test — validates the full Hilbert + Tier + Bank pipeline.
//!
//! Tests the complete flow WITHOUT an LLM:
//! 1. Graph → Spectral Embedding → Hilbert Layout
//! 2. Banks from communities
//! 3. Tier promotion/demotion lifecycle
//! 4. Bank load/evict
//! 5. Importance-based eviction ordering

use anyhow::Result;
use kv_registry::Tokenizer;
use kv_registry::hilbert::{AdjacencyList, HilbertLayout, spectral_embedding_2d};
use kv_registry::hilbert_bank::BankManager;
use kv_registry::registry::{KvNodeRegistry, KvSlotMode, KvTier};
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};

// ── Mock Tokenizer (no LLM needed) ─────────────────────────────

struct MockTokenizer {
    evicted: std::cell::RefCell<Vec<(i32, i32)>>,
}

impl MockTokenizer {
    fn new() -> Self {
        Self {
            evicted: std::cell::RefCell::new(Vec::new()),
        }
    }
}

impl Tokenizer for MockTokenizer {
    fn tokenize(&self, text: &str, _add_bos: bool, _parse_special: bool) -> Result<Vec<i32>> {
        // Simple mock: 1 token per 4 chars
        let n = (text.len() / 4).max(1);
        Ok((0..n as i32).collect())
    }

    fn encode(&self, _tokens: &[i32], _positions: &[i32], _seq_id: i32) -> Result<()> {
        Ok(())
    }

    fn evict(&self, start: i32, end: i32) {
        self.evicted.borrow_mut().push((start, end));
    }

    fn n_ctx(&self) -> u32 {
        4096
    }

    fn clear_kv(&self) {}

    fn seq_pos_max(&self, _seq_id: i32) -> i32 {
        0
    }

    fn token_count(&self, text: &str) -> Result<usize> {
        Ok((text.len() / 4).max(1))
    }
}

// ── Helper: build a test graph ──────────────────────────────────

/// Build a graph with 3 communities of `size` nodes each, connected by bridges.
fn build_test_graph(
    size: usize,
) -> (
    HashMap<NodeId, HashSet<NodeId>>,
    HashMap<NodeId, usize>, // communities
) {
    let mut adjacency: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
    let mut communities: HashMap<NodeId, usize> = HashMap::new();

    let n = size * 3;
    for i in 0..n {
        let nid = NodeId(i as u64);
        let comm = i / size;
        communities.insert(nid, comm);
        adjacency.entry(nid).or_default();
    }

    // Intra-community: complete graph
    for comm in 0..3 {
        for i in (comm * size)..((comm + 1) * size) {
            for j in (comm * size)..((comm + 1) * size) {
                if i != j {
                    adjacency
                        .entry(NodeId(i as u64))
                        .or_default()
                        .insert(NodeId(j as u64));
                }
            }
        }
    }

    // Inter-community bridges
    adjacency
        .entry(NodeId((size - 1) as u64))
        .or_default()
        .insert(NodeId(size as u64));
    adjacency
        .entry(NodeId(size as u64))
        .or_default()
        .insert(NodeId((size - 1) as u64));
    adjacency
        .entry(NodeId((2 * size - 1) as u64))
        .or_default()
        .insert(NodeId((2 * size) as u64));
    adjacency
        .entry(NodeId((2 * size) as u64))
        .or_default()
        .insert(NodeId((2 * size - 1) as u64));

    (adjacency, communities)
}

// ── Test 1: Full pipeline graph → Hilbert → Banks ──────────────

#[test]
fn test_full_pipeline_graph_to_banks() {
    let (adjacency, communities) = build_test_graph(9); // 27 nodes, 3 communities of 9
    assert_eq!(adjacency.len(), 27);

    // Step 1: Hilbert layout
    let layout = HilbertLayout::compute(&adjacency, 10); // base_pos = 10
    assert_eq!(layout.len(), 27);

    // All positions should be >= 10 (base)
    for &pos in layout.positions.values() {
        assert!(pos >= 10, "Position {pos} should be >= base 10");
    }

    // Step 2: Banks from communities
    let mgr = BankManager::from_communities(&layout, &communities);
    assert_eq!(mgr.len(), 3);
    assert_eq!(mgr.total_nodes(), 27);

    // Each bank should have 9 nodes
    for bank in &mgr.banks {
        assert_eq!(bank.len(), 9);
    }

    // Every node should be in exactly one bank
    let mut all_nodes: HashSet<NodeId> = HashSet::new();
    for bank in &mgr.banks {
        for &nid in &bank.node_ids {
            assert!(all_nodes.insert(nid), "Node {:?} in multiple banks", nid);
        }
    }
    assert_eq!(all_nodes.len(), 27);
}

// ── Test 2: Tier lifecycle (Γ → Β → Α → Β → Γ) ────────────────

#[test]
fn test_tier_lifecycle() {
    let engine = MockTokenizer::new();
    let mut registry = KvNodeRegistry::new("header", 5);
    registry.begin_query();

    let nid = NodeId(42);

    // Register as embedding (Γ)
    registry
        .register_embedding(nid, ":Person Test", &vec![0.0f32; 64], |_e, _p, _s| Ok(1))
        .unwrap();
    assert_eq!(registry.get_tier(nid), Some(KvTier::Gamma));

    // Promote Γ → Β
    registry.promote_to_beta(nid, ":Person", &engine).unwrap();
    assert_eq!(registry.get_tier(nid), Some(KvTier::Beta));
    assert!(matches!(
        registry.get_slot(nid).unwrap().mode,
        KvSlotMode::EmbeddingWithMinimalTag { .. }
    ));

    // Promote Β → Α
    registry
        .promote_to_alpha(nid, ":Person Test [→knows Alice]", &engine)
        .unwrap();
    assert_eq!(registry.get_tier(nid), Some(KvTier::Alpha));
    assert!(matches!(
        registry.get_slot(nid).unwrap().mode,
        KvSlotMode::EmbeddingWithTags { .. }
    ));

    // Demote Α → Β
    registry.demote_to_beta(nid, &engine).unwrap();
    assert_eq!(registry.get_tier(nid), Some(KvTier::Beta));

    // Demote Β → Γ
    registry.demote_to_gamma(nid, &engine).unwrap();
    assert_eq!(registry.get_tier(nid), Some(KvTier::Gamma));
    assert_eq!(registry.get_slot(nid).unwrap().n_tokens, 1); // Back to 1 position
}

// ── Test 3: Tier budget enforcement ─────────────────────────────

#[test]
fn test_tier_budget() {
    let engine = MockTokenizer::new();
    let mut registry = KvNodeRegistry::new("header", 5);
    registry.tier_budget = kv_registry::TierBudget::new(3); // Max 3 promotions

    // Register 5 nodes as Γ
    for i in 0..5 {
        registry
            .register_embedding(NodeId(i), ":Test", &vec![0.0f32; 64], |_e, _p, _s| Ok(1))
            .unwrap();
    }

    // Promote 3 → should succeed
    for i in 0..3 {
        assert!(registry.promote_to_beta(NodeId(i), ":T", &engine).is_ok());
    }

    // 4th promotion → should fail (budget exhausted)
    assert!(registry.promote_to_beta(NodeId(3), ":T", &engine).is_err());

    // Reset budget
    registry.tier_budget.reset();

    // Now should work
    assert!(registry.promote_to_beta(NodeId(3), ":T", &engine).is_ok());
}

// ── Test 4: Bank importance-based eviction ──────────────────────

#[test]
fn test_bank_importance_eviction() {
    let (adjacency, communities) = build_test_graph(5); // 15 nodes

    let layout = HilbertLayout::compute(&adjacency, 10);
    let mut mgr = BankManager::from_communities(&layout, &communities);

    // Mark all banks as loaded
    for bank in &mut mgr.banks {
        bank.loaded = true;
    }

    // Set different importance levels
    let mut rewards = HashMap::new();
    rewards.insert(0, 0.9f32); // Bank 0: high importance
    rewards.insert(1, 0.1f32); // Bank 1: low importance
    rewards.insert(2, 0.5f32); // Bank 2: medium
    mgr.update_importance(&rewards, 0.0); // decay=0 → direct assignment

    // Eviction candidates: lowest importance first
    let candidates = mgr.eviction_candidates();
    assert_eq!(candidates[0], 1); // Bank 1 first (lowest)
    assert_eq!(candidates[1], 2); // Bank 2 second
    assert_eq!(candidates[2], 0); // Bank 0 last (highest)
}

// ── Test 5: Spectral embedding quality ──────────────────────────

#[test]
fn test_spectral_separates_communities() {
    // 3 communities of 8 nodes, connected by single bridges
    let mut adj: AdjacencyList = vec![Vec::new(); 24];

    // Community A: K8 (0-7)
    for i in 0..8 {
        for j in 0..8 {
            if i != j {
                adj[i].push(j);
            }
        }
    }
    // Community B: K8 (8-15)
    for i in 8..16 {
        for j in 8..16 {
            if i != j {
                adj[i].push(j);
            }
        }
    }
    // Community C: K8 (16-23)
    for i in 16..24 {
        for j in 16..24 {
            if i != j {
                adj[i].push(j);
            }
        }
    }
    // Bridges
    adj[7].push(8);
    adj[8].push(7);
    adj[15].push(16);
    adj[16].push(15);

    let coords = spectral_embedding_2d(&adj);
    assert_eq!(coords.len(), 24);

    // Compute centroid per community
    let centroid = |range: std::ops::Range<usize>| -> (f32, f32) {
        let n = range.len() as f32;
        let cx: f32 = range.clone().map(|i| coords[i].0).sum::<f32>() / n;
        let cy: f32 = range.map(|i| coords[i].1).sum::<f32>() / n;
        (cx, cy)
    };

    let ca = centroid(0..8);
    let cb = centroid(8..16);
    let cc = centroid(16..24);

    // All centroids should be distinct
    let dist_ab = ((ca.0 - cb.0).powi(2) + (ca.1 - cb.1).powi(2)).sqrt();
    let dist_bc = ((cb.0 - cc.0).powi(2) + (cb.1 - cc.1).powi(2)).sqrt();
    let dist_ac = ((ca.0 - cc.0).powi(2) + (ca.1 - cc.1).powi(2)).sqrt();

    assert!(dist_ab > 0.1, "Communities A-B not separated: {dist_ab}");
    assert!(dist_bc > 0.1, "Communities B-C not separated: {dist_bc}");
    assert!(dist_ac > 0.1, "Communities A-C not separated: {dist_ac}");
}

// ── Test 6: Hilbert preserves locality ──────────────────────────

#[test]
fn test_hilbert_locality_for_communities() {
    let (adjacency, communities) = build_test_graph(9); // 27 nodes

    let layout = HilbertLayout::compute(&adjacency, 0);

    // For each community, compute position spread (max - min)
    for comm_id in 0..3 {
        let mut positions: Vec<u32> = communities
            .iter()
            .filter(|&(_, c)| *c == comm_id)
            .map(|(nid, _)| layout.get_position(*nid).unwrap())
            .collect();

        // Community of 9 nodes: most should be clustered, but bridge nodes
        // may be pulled towards adjacent communities. Check median spread instead.
        // With 27 total positions, a community of 9 can span up to 27 in worst case
        // but the inner quartile (positions 2..7) should be tight.
        positions.sort();
        let iqr_spread = positions[6] - positions[2]; // inner 5 of 9
        assert!(
            iqr_spread <= 15,
            "Community {comm_id} IQR spread too large: {iqr_spread} (positions: {positions:?})"
        );
    }
}

// ── Test 7: Tier distribution metrics ───────────────────────────

#[test]
fn test_tier_distribution() {
    let engine = MockTokenizer::new();
    let mut registry = KvNodeRegistry::new("header", 5);
    registry.begin_query();

    // Register 10 nodes: 3 Α, 3 Β, 4 Γ
    for i in 0..10 {
        registry
            .register_embedding(NodeId(i), ":Test", &vec![0.0f32; 64], |_e, _p, _s| Ok(1))
            .unwrap();
    }

    // Promote 0-2 to Alpha
    for i in 0..3 {
        registry
            .promote_to_alpha(NodeId(i), ":full tag", &engine)
            .unwrap();
    }
    // Promote 3-5 to Beta
    for i in 3..6 {
        registry
            .promote_to_beta(NodeId(i), ":label", &engine)
            .unwrap();
    }

    let (alpha, beta, gamma) = registry.tier_distribution();
    assert_eq!(alpha, 3);
    assert_eq!(beta, 3);
    assert_eq!(gamma, 4);

    // KV compression: Alpha uses more positions, Gamma uses 1
    let total_positions: i32 = registry.nodes.values().map(|s| s.n_tokens).sum();
    let n_nodes = registry.nodes.len() as i32;
    // Compression ratio = n_nodes / total_positions
    let compression = n_nodes as f64 / total_positions as f64;
    assert!(
        compression < 1.0,
        "Tiers should use more than 1 pos on average"
    );
    assert!(compression > 0.3, "Not too much overhead: {compression}");
}

// ── Test 8: End-to-end pipeline integration ─────────────────────

#[test]
fn test_end_to_end_pipeline() {
    let engine = MockTokenizer::new();
    let (adjacency, communities) = build_test_graph(9); // 27 nodes

    // 1. Compute Hilbert layout
    let layout = HilbertLayout::compute(&adjacency, 10);
    assert_eq!(layout.len(), 27);

    // 2. Create registry with Hilbert layout
    let mut registry = KvNodeRegistry::new("header", 10);
    registry.set_hilbert_layout(layout.clone());
    registry.begin_query();

    // 3. Create bank manager
    let mut mgr = BankManager::from_communities(&layout, &communities);
    assert_eq!(mgr.len(), 3);

    // 4. Load bank 0 (simulated — can't call load_bank without real encode_embd_fn
    //    that satisfies FnOnce, but we test the structure)
    let bank0_nodes = mgr.banks[0].node_ids.clone();
    for &nid in &bank0_nodes {
        registry
            .register_embedding(nid, ":Test", &vec![0.0f32; 64], |_e, _p, _s| Ok(1))
            .unwrap();
    }
    mgr.banks[0].loaded = true;

    // 5. Verify nodes are at Hilbert positions
    for &nid in &bank0_nodes {
        let slot = registry.get_slot(nid).unwrap();
        let expected_pos = layout.get_position(nid).unwrap() as i32;
        assert_eq!(
            slot.start, expected_pos,
            "Node {:?} at pos {} but expected Hilbert pos {}",
            nid, slot.start, expected_pos
        );
    }

    // 6. Promote a node to Alpha
    let promoted_nid = bank0_nodes[0];
    registry
        .promote_to_alpha(promoted_nid, ":full tag", &engine)
        .unwrap();
    assert_eq!(registry.get_tier(promoted_nid), Some(KvTier::Alpha));

    // 7. Demote it back to Gamma
    registry.demote_to_gamma(promoted_nid, &engine).unwrap();
    assert_eq!(registry.get_tier(promoted_nid), Some(KvTier::Gamma));

    // 8. Bank importance
    let mut rewards = HashMap::new();
    rewards.insert(0, 0.8f32);
    rewards.insert(1, 0.2f32);
    rewards.insert(2, 0.5f32);
    mgr.update_importance(&rewards, 0.0);

    // 9. Eviction candidates: bank 1 first (lowest importance)
    let candidates = mgr.eviction_candidates();
    // Only bank 0 is loaded
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0], 0);
}
