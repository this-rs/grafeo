//! Ξ(t) PersistNet — Neural persistence classifier.
//!
//! Replaces pattern matching for fact detection. Uses the LLM's own embedding
//! (last-layer hidden state of the user message) to decide if a message contains
//! information worth persisting as a :Memory node.
//!
//! Architecture:
//!   LLM embedding (n_embd) → Random Projection (n_embd → 64) → MLP (64 → 64 → 1) → sigmoid
//!
//! Training:
//!   REINFORCE online — reward comes from GNN's USED_IN tracking.
//!   If a persisted memory gets used in future turns with positive reward → reinforce.
//!   If never used after K turns → punish.
//!
//! Cold start:
//!   First `COLD_START_TURNS` turns: persist everything (score = 1.0).
//!   After that, PersistNet gates persistence.

use obrain::ObrainDB;
use obrain_common::types::{PropertyKey, Value};
use obrain_core::graph::lpg::LpgStore;

/// Internal dimension for projection and MLP.
const DIM: usize = 64;

/// Number of turns before PersistNet starts gating (everything persisted before).
const COLD_START_TURNS: u32 = 20;

/// Persist threshold — messages with score above this are persisted.
const PERSIST_THRESHOLD: f32 = 0.5;

/// PersistNet — lightweight MLP that classifies "should this message be persisted?"
pub struct PersistNet {
    /// Random projection matrix: DIM × n_embd (row-major, sparse-ish).
    /// Maps LLM hidden state to DIM-dimensional space.
    /// Fixed after initialization (never trained).
    pub projection: Vec<f32>,
    /// MLP layer 1: DIM × DIM (row-major)
    pub w1: Vec<f32>,
    /// MLP bias 1: DIM
    pub b1: Vec<f32>,
    /// MLP layer 2 (output): DIM → 1
    pub w2: Vec<f32>,
    /// Output bias
    pub b2: f32,
    /// Number of gradient updates performed.
    pub n_updates: u64,
    /// Dimension of LLM embedding (set on first forward pass).
    pub n_embd: usize,
    /// Total turns seen (for cold start tracking).
    pub total_turns: u32,
}

impl PersistNet {
    /// Create a new PersistNet. `n_embd` is the LLM's embedding dimension.
    /// If n_embd=0, it will be set lazily on first forward().
    pub fn new(n_embd: usize) -> Self {
        let projection = if n_embd > 0 {
            init_projection(n_embd)
        } else {
            Vec::new()
        };

        PersistNet {
            projection,
            w1: xavier_init_rect(DIM, DIM),
            b1: vec![0.0; DIM],
            w2: xavier_init_rect(1, DIM),
            b2: 0.0,
            n_updates: 0,
            n_embd,
            total_turns: 0,
        }
    }

    /// Forward pass: LLM embedding → persist_score ∈ [0, 1].
    ///
    /// Returns (persist_score, projected_embedding) — the projected embedding is
    /// cached for the backward pass (REINFORCE gradient).
    pub fn forward(&mut self, llm_embedding: &[f32]) -> (f32, Vec<f32>) {
        // Lazy init projection if n_embd was unknown at construction
        if self.n_embd == 0 && !llm_embedding.is_empty() {
            self.n_embd = llm_embedding.len();
            self.projection = init_projection(self.n_embd);
        }

        // Cold start: always persist
        if self.total_turns < COLD_START_TURNS {
            let projected = self.project(llm_embedding);
            return (1.0, projected);
        }

        // 1. Random projection: n_embd → DIM
        let z = self.project(llm_embedding);

        // 2. MLP layer 1: h = ReLU(W1·z + b1)
        let mut h = vec![0.0f32; DIM];
        for i in 0..DIM {
            let mut sum = self.b1[i];
            for j in 0..DIM {
                sum += self.w1[i * DIM + j] * z[j];
            }
            h[i] = sum.max(0.0); // ReLU
        }

        // 3. MLP layer 2: score = σ(w2·h + b2)
        let mut logit = self.b2;
        for j in 0..DIM {
            logit += self.w2[j] * h[j];
        }
        let score = sigmoid(logit);

        (score, z)
    }

    /// Check if a message should be persisted based on its score.
    pub fn should_persist(&self, score: f32) -> bool {
        if self.total_turns < COLD_START_TURNS {
            return true;
        }
        score > PERSIST_THRESHOLD
    }

    /// Increment turn counter (call once per turn).
    pub fn tick(&mut self) {
        self.total_turns += 1;
    }

    /// REINFORCE update: adjust weights based on reward signal.
    ///
    /// - `projected`: the DIM-dim projected embedding from forward()
    /// - `score`: the persist_score from forward()
    /// - `reward`: positive if the memory was useful, negative if wasteful
    pub fn update(&mut self, projected: &[f32], score: f32, reward: f32) {
        if reward.abs() < 0.01 { return; }
        if projected.len() != DIM { return; }

        // Adaptive learning rate (same schedule as FactGNN)
        let lr = 0.01 / (1.0 + 0.001 * self.n_updates as f32);

        // Policy gradient: ∇log π(a|s) = (1 - score) for action=persist, -score for action=skip
        // Since we only update on persisted memories, gradient = reward × (1 - score)
        let grad_scale = lr * reward * (1.0 - score);

        // Recompute hidden layer for gradient
        let mut h = vec![0.0f32; DIM];
        for i in 0..DIM {
            let mut sum = self.b1[i];
            for j in 0..DIM {
                sum += self.w1[i * DIM + j] * projected[j];
            }
            h[i] = sum.max(0.0); // ReLU
        }

        // Update w2 (output layer): ∂L/∂w2 = grad_scale × h
        for j in 0..DIM {
            self.w2[j] += grad_scale * h[j];
        }
        self.b2 += grad_scale;

        // Update w1 (hidden layer): backprop through ReLU
        // ∂L/∂w1[i][j] = grad_scale × w2[i] × (h[i] > 0) × projected[j]
        for i in 0..DIM {
            if h[i] > 0.0 {
                let upstream = grad_scale * self.w2[i];
                for j in 0..DIM {
                    self.w1[i * DIM + j] += upstream * projected[j];
                }
                self.b1[i] += upstream;
            }
        }

        self.n_updates += 1;
    }

    /// Project LLM embedding to DIM dimensions.
    fn project(&self, embedding: &[f32]) -> Vec<f32> {
        if embedding.is_empty() || self.projection.is_empty() {
            return vec![0.0; DIM];
        }

        let n_embd = embedding.len();
        let mut z = vec![0.0f32; DIM];
        for i in 0..DIM {
            let row_start = i * n_embd;
            let mut sum = 0.0f32;
            for j in 0..n_embd {
                sum += self.projection[row_start + j] * embedding[j];
            }
            z[i] = sum;
        }

        // L2 normalize
        let norm = z.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in z.iter_mut() {
            *v /= norm;
        }
        z
    }

    // ── Persistence (save/load weights to PersonaDB) ──────────────────

    /// Save PersistNet weights to PersonaDB as :PersistNetWeights nodes.
    pub fn save_weights(&self, db: &ObrainDB) {
        let store = db.store();

        // Remove old weights
        for &nid in &store.nodes_by_label("PersistNetWeights") {
            db.delete_node(nid);
        }

        // Save MLP weights
        let w1_data = weights_to_base64(&self.w1);
        let b1_data = weights_to_base64(&self.b1);
        let w2_data = weights_to_base64(&self.w2);

        db.create_node_with_props(&["PersistNetWeights"], [
            ("layer", Value::String("w1".to_string().into())),
            ("data", Value::String(w1_data.into())),
            ("dim", Value::Int64(DIM as i64)),
            ("n_embd", Value::Int64(self.n_embd as i64)),
            ("n_updates", Value::Int64(self.n_updates as i64)),
            ("total_turns", Value::Int64(self.total_turns as i64)),
        ]);
        db.create_node_with_props(&["PersistNetWeights"], [
            ("layer", Value::String("b1".to_string().into())),
            ("data", Value::String(b1_data.into())),
        ]);
        db.create_node_with_props(&["PersistNetWeights"], [
            ("layer", Value::String("w2".to_string().into())),
            ("data", Value::String(w2_data.into())),
        ]);
        db.create_node_with_props(&["PersistNetWeights"], [
            ("layer", Value::String("b2".to_string().into())),
            ("data", Value::String(weights_to_base64(&[self.b2]).into())),
        ]);
    }

    /// Load PersistNet weights from PersonaDB. Returns true if weights were loaded.
    pub fn load_weights(&mut self, store: &LpgStore) -> bool {
        let weight_nodes = store.nodes_by_label("PersistNetWeights");
        if weight_nodes.is_empty() { return false; }

        let mut loaded = 0u32;
        for &nid in &weight_nodes {
            if let Some(node) = store.get_node(nid) {
                let layer = node.properties.get(&PropertyKey::from("layer"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let data = node.properties.get(&PropertyKey::from("data"))
                    .and_then(|v| v.as_str()).unwrap_or("");

                match layer {
                    "w1" => {
                        if let Some(w) = base64_to_weights(data) {
                            if w.len() == DIM * DIM {
                                self.w1 = w;
                                loaded += 1;
                            }
                        }
                        // Also load metadata
                        self.n_updates = node.properties.get(&PropertyKey::from("n_updates"))
                            .and_then(|v| if let Value::Int64(n) = v { Some(*n as u64) } else { None })
                            .unwrap_or(0);
                        self.total_turns = node.properties.get(&PropertyKey::from("total_turns"))
                            .and_then(|v| if let Value::Int64(n) = v { Some(*n as u32) } else { None })
                            .unwrap_or(0);
                        let saved_n_embd = node.properties.get(&PropertyKey::from("n_embd"))
                            .and_then(|v| if let Value::Int64(n) = v { Some(*n as usize) } else { None })
                            .unwrap_or(0);
                        if saved_n_embd > 0 && self.n_embd == 0 {
                            self.n_embd = saved_n_embd;
                            self.projection = init_projection(saved_n_embd);
                        }
                    }
                    "b1" => {
                        if let Some(w) = base64_to_weights(data) {
                            if w.len() == DIM { self.b1 = w; loaded += 1; }
                        }
                    }
                    "w2" => {
                        if let Some(w) = base64_to_weights(data) {
                            if w.len() == DIM { self.w2 = w; loaded += 1; }
                        }
                    }
                    "b2" => {
                        if let Some(w) = base64_to_weights(data) {
                            if w.len() == 1 { self.b2 = w[0]; loaded += 1; }
                        }
                    }
                    _ => {}
                }
            }
        }

        loaded > 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Initialize random projection matrix (DIM × n_embd).
/// Uses a fixed seed so the projection is deterministic across runs.
/// Sparse-ish: Gaussian random with scaling 1/√n_embd for variance preservation.
fn init_projection(n_embd: usize) -> Vec<f32> {
    let n = DIM * n_embd;
    let scale = 1.0 / (n_embd as f32).sqrt();
    let mut w = vec![0.0f32; n];

    // Deterministic PRNG seeded from n_embd
    let mut state: u64 = 0x9e3779b97f4a7c15 ^ (n_embd as u64).wrapping_mul(0x517cc1b727220a95);
    for v in w.iter_mut() {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58476d1ce4e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d049bb133111eb);
        state ^= state >> 31;
        let f = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        *v = f * scale;
    }
    w
}

/// Xavier initialization for a (rows × cols) matrix.
fn xavier_init_rect(rows: usize, cols: usize) -> Vec<f32> {
    let n = rows * cols;
    let scale = (2.0 / (rows + cols) as f64).sqrt() as f32;
    let mut w = vec![0.0f32; n];

    let mut state: u64 = 0x517cc1b727220a95 ^ (rows as u64).wrapping_mul(31).wrapping_add((cols as u64).wrapping_mul(37));
    for v in w.iter_mut() {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58476d1ce4e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d049bb133111eb);
        state ^= state >> 31;
        let f = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        *v = f * scale;
    }
    w
}

// ── base64 (same as fact_gnn.rs, duplicated to avoid cross-module dep) ──

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn weights_to_base64(w: &[f32]) -> String {
    let bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    base64_encode(&bytes)
}

fn base64_to_weights(s: &str) -> Option<Vec<f32>> {
    let bytes = base64_decode(s)?;
    if bytes.len() % 4 != 0 { return None; }
    let w: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(w)
}

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char); }
        else { result.push('='); }
        if chunk.len() > 2 { result.push(B64_CHARS[(triple & 0x3F) as usize] as char); }
        else { result.push('='); }
    }
    result
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_end_matches('=');
    let mut result = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for c in s.chars() {
        let val = match c {
            'A'..='Z' => c as u32 - 'A' as u32,
            'a'..='z' => c as u32 - 'a' as u32 + 26,
            '0'..='9' => c as u32 - '0' as u32 + 52,
            '+' => 62,
            '/' => 63,
            _ => continue,
        };
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            result.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_start_always_persists() {
        let mut net = PersistNet::new(128);
        // During cold start, everything should be persisted
        let fake_emb = vec![0.1f32; 128];
        let (score, _proj) = net.forward(&fake_emb);
        assert_eq!(score, 1.0, "Cold start should return score=1.0");
        assert!(net.should_persist(score));
    }

    #[test]
    fn test_after_cold_start_uses_mlp() {
        let mut net = PersistNet::new(128);
        net.total_turns = COLD_START_TURNS; // Skip cold start

        let fake_emb = vec![0.1f32; 128];
        let (score, _proj) = net.forward(&fake_emb);
        // Score should be between 0 and 1 (sigmoid output)
        assert!(score >= 0.0 && score <= 1.0, "score={score} not in [0,1]");
        // With random init, score should be roughly 0.5
        assert!((score - 0.5).abs() < 0.3, "score={score} too far from 0.5 with random init");
    }

    #[test]
    fn test_update_changes_weights() {
        let mut net = PersistNet::new(128);
        net.total_turns = COLD_START_TURNS;

        let fake_emb = vec![0.5f32; 128];
        let (score, proj) = net.forward(&fake_emb);
        let w2_before: Vec<f32> = net.w2.clone();

        // Positive reward → should reinforce persistence
        net.update(&proj, score, 1.0);

        let w2_diff: f32 = net.w2.iter().zip(w2_before.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(w2_diff > 0.0, "Weights should change after update");
        assert_eq!(net.n_updates, 1);
    }

    #[test]
    fn test_learning_direction() {
        let mut net = PersistNet::new(128);
        net.total_turns = COLD_START_TURNS;

        let fake_emb = vec![0.3f32; 128];
        let (score_before, _) = net.forward(&fake_emb);

        // Train with positive reward → score should increase
        for _ in 0..50 {
            let (s, proj) = net.forward(&fake_emb);
            net.update(&proj, s, 1.0);
        }

        let (score_after, _) = net.forward(&fake_emb);
        assert!(score_after > score_before,
            "Score should increase after positive reward: {score_before} → {score_after}");
    }

    #[test]
    fn test_negative_reward_decreases() {
        let mut net = PersistNet::new(128);
        net.total_turns = COLD_START_TURNS;

        let fake_emb = vec![0.7f32; 128];
        let (score_before, _) = net.forward(&fake_emb);

        // Train with negative reward → score should decrease
        for _ in 0..50 {
            let (s, proj) = net.forward(&fake_emb);
            net.update(&proj, s, -1.0);
        }

        let (score_after, _) = net.forward(&fake_emb);
        assert!(score_after < score_before,
            "Score should decrease after negative reward: {score_before} → {score_after}");
    }

    #[test]
    fn test_projection_deterministic() {
        let net1 = PersistNet::new(256);
        let net2 = PersistNet::new(256);
        assert_eq!(net1.projection, net2.projection,
            "Projection should be deterministic for same n_embd");
    }

    #[test]
    fn test_lazy_init() {
        let mut net = PersistNet::new(0); // Unknown n_embd
        assert!(net.projection.is_empty());

        let fake_emb = vec![0.1f32; 512];
        let (_score, proj) = net.forward(&fake_emb);

        assert_eq!(net.n_embd, 512);
        assert_eq!(net.projection.len(), DIM * 512);
        assert_eq!(proj.len(), DIM);
    }

    #[test]
    fn test_save_load_roundtrip() {
        // Create a net and train it a bit
        let mut net = PersistNet::new(128);
        net.total_turns = 30;
        let fake_emb = vec![0.5f32; 128];
        for _ in 0..10 {
            let (s, proj) = net.forward(&fake_emb);
            net.update(&proj, s, 0.5);
        }

        // Save to ObrainDB
        let db = ObrainDB::new_in_memory();
        net.save_weights(&db);

        // Load into fresh net
        let mut net2 = PersistNet::new(0);
        let loaded = net2.load_weights(&db.store());
        assert!(loaded, "Should load successfully");
        assert_eq!(net2.n_updates, net.n_updates);
        assert_eq!(net2.total_turns, net.total_turns);
        assert_eq!(net2.n_embd, 128);
        assert_eq!(net2.w1, net.w1);
        assert_eq!(net2.w2, net.w2);
        assert_eq!(net2.b1, net.b1);
        assert_eq!(net2.b2, net.b2);
    }
}
