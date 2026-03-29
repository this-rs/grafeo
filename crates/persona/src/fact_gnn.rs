//! Ξ(t) T4 — Heterogeneous Fact-GNN: 2-layer message passing on ObrainDB PersonaDB.
//!
//! Architecture:
//! - 64-dim embeddings, deterministic init from node properties
//! - Layer 1: aggregate neighbor messages weighted by edge type
//! - Layer 2: same, now each node sees its 2-hop neighborhood
//! - Readout: dot(query_embed, node_embed) → score per :Fact node
//! - Online REINFORCE training: gradient = reward × score × h_neighbor
//! - Weights persisted as :GNNWeights nodes in PersonaDB

use std::collections::{HashMap, HashSet, VecDeque};
use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::{Direction, lpg::LpgStore};

/// GNN dimension (embedding size per node).
const DIM: usize = 64;

/// Edge types recognized by the GNN, each with its own weight matrix.
const EDGE_TYPES: &[&str] = &[
    "REINFORCES", "CONTRADICTS", "USED_IN", "EXTRACTED_FROM",
    "TEMPORAL_NEXT", "EXTRACTS", "MENTIONS",
];

/// Node types, each with its own update matrix.
const NODE_TYPES: &[&str] = &["Fact", "ConvTurn", "Pattern"];

/// A lightweight GNN operating directly on PersonaDB graph.
pub struct FactGNN {
    pub dim: usize,
    /// W_message[edge_type] → DIM×DIM flattened (row-major)
    pub w_message: HashMap<String, Vec<f32>>,
    /// W_update[node_type] → DIM×DIM flattened (row-major)
    pub w_update: HashMap<String, Vec<f32>>,
    /// Number of gradient updates performed (for adaptive LR).
    pub n_updates: u64,
}

impl FactGNN {
    /// Create a new GNN with Xavier-initialized weights.
    pub fn new() -> Self {
        let mut w_message = HashMap::new();
        let mut w_update = HashMap::new();

        for &et in EDGE_TYPES {
            w_message.insert(et.to_string(), xavier_init(DIM));
        }
        for &nt in NODE_TYPES {
            w_update.insert(nt.to_string(), xavier_init(DIM));
        }

        FactGNN { dim: DIM, w_message, w_update, n_updates: 0 }
    }

    /// Compute deterministic embedding for a node based on its properties.
    fn init_embedding(store: &LpgStore, nid: NodeId) -> [f32; DIM] {
        let node = match store.get_node(nid) {
            Some(n) => n,
            None => return [0.0; DIM],
        };

        // Build a seed string from label + key properties
        let label = node.labels.first()
            .map(|l| { let s: &str = l.as_ref(); s.to_string() })
            .unwrap_or_default();

        let seed_str = match label.as_str() {
            "Fact" => {
                let key = node.properties.get(&PropertyKey::from("key"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let value = node.properties.get(&PropertyKey::from("value"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let ft = node.properties.get(&PropertyKey::from("fact_type"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                format!("Fact:{key}:{value}:{ft}")
            }
            "ConvTurn" => {
                let qh = node.properties.get(&PropertyKey::from("query_hash"))
                    .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
                    .unwrap_or(0);
                format!("ConvTurn:{qh}")
            }
            "Pattern" => {
                let trigger = node.properties.get(&PropertyKey::from("trigger"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                format!("Pattern:{trigger}")
            }
            _ => format!("{label}:{}", nid.0),
        };

        // Hash → deterministic PRNG → 64-dim normalized vector
        hash_to_embedding(&seed_str)
    }

    /// Collect the k-hop subgraph around center nodes via BFS.
    fn collect_subgraph(store: &LpgStore, center_nodes: &[NodeId], hops: usize) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for &nid in center_nodes {
            if visited.insert(nid) {
                queue.push_back((nid, 0));
            }
        }

        while let Some((nid, depth)) = queue.pop_front() {
            if depth >= hops { continue; }
            // Outgoing edges
            for (target, _eid) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                if visited.insert(target) {
                    queue.push_back((target, depth + 1));
                }
            }
            // Incoming edges
            for (source, _eid) in store.edges_from(nid, Direction::Incoming).collect::<Vec<_>>() {
                if visited.insert(source) {
                    queue.push_back((source, depth + 1));
                }
            }
        }

        visited
    }

    /// Forward pass: score :Fact nodes by relevance to query.
    ///
    /// Returns Vec<(NodeId, f32)> sorted by score descending.
    pub fn forward(
        &self,
        store: &LpgStore,
        query_embed: &[f32; DIM],
        center_nodes: &[NodeId],
        hops: usize,
    ) -> Vec<(NodeId, f32)> {
        // 1. Collect subgraph
        let subgraph = Self::collect_subgraph(store, center_nodes, hops);
        if subgraph.is_empty() { return Vec::new(); }

        // 2. Init embeddings
        let mut embeddings: HashMap<NodeId, [f32; DIM]> = HashMap::new();
        for &nid in &subgraph {
            embeddings.insert(nid, Self::init_embedding(store, nid));
        }

        // 3. Two rounds of message passing
        for _layer in 0..2 {
            let mut new_embeddings: HashMap<NodeId, [f32; DIM]> = HashMap::new();

            for &nid in &subgraph {
                let mut h = embeddings[&nid];

                // Aggregate messages from neighbors
                for (target, eid) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                    if !subgraph.contains(&target) { continue; }
                    let edge_type = Self::get_edge_label(store, nid, target, eid);
                    if let Some(w) = self.w_message.get(&edge_type) {
                        let h_neighbor = &embeddings[&target];
                        let msg = mat_vec_mul(w, h_neighbor);
                        // h_v += σ(W_edge · h_u)
                        for i in 0..DIM {
                            h[i] += relu(msg[i]);
                        }
                    }
                }

                // Incoming edges too (bidirectional message passing)
                for (source, eid) in store.edges_from(nid, Direction::Incoming).collect::<Vec<_>>() {
                    if !subgraph.contains(&source) { continue; }
                    let edge_type = Self::get_edge_label(store, source, nid, eid);
                    if let Some(w) = self.w_message.get(&edge_type) {
                        let h_neighbor = &embeddings[&source];
                        let msg = mat_vec_mul(w, h_neighbor);
                        for i in 0..DIM {
                            h[i] += relu(msg[i]);
                        }
                    }
                }

                // Apply node-type update
                let label = store.get_node(nid)
                    .and_then(|n| n.labels.first().map(|l| { let s: &str = l.as_ref(); s.to_string() }))
                    .unwrap_or_default();
                if let Some(w) = self.w_update.get(&label) {
                    let updated = mat_vec_mul(w, &h);
                    for i in 0..DIM {
                        h[i] = relu(updated[i]);
                    }
                }

                // Normalize to prevent explosion
                let norm = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for i in 0..DIM {
                    h[i] /= norm;
                }

                new_embeddings.insert(nid, h);
            }

            // Swap for next layer
            embeddings = new_embeddings;
        }

        // 4. Readout: score = dot(query_embed, h_v) for :Fact nodes only
        let mut scores: Vec<(NodeId, f32)> = Vec::new();
        for &nid in &subgraph {
            let is_fact = store.get_node(nid)
                .map(|n| n.labels.iter().any(|l| { let s: &str = l.as_ref(); s == "Fact" }))
                .unwrap_or(false);
            if !is_fact { continue; }

            // Only active facts
            let active = store.get_node(nid)
                .and_then(|n| n.properties.get(&PropertyKey::from("active"))
                    .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None }))
                .unwrap_or(true);
            if !active { continue; }

            let h = &embeddings[&nid];
            let score: f32 = (0..DIM).map(|i| query_embed[i] * h[i]).sum();
            scores.push((nid, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Online REINFORCE update: adjust weights based on reward signal.
    pub fn update(
        &mut self,
        store: &LpgStore,
        used_nodes: &[NodeId],
        scores: &[(NodeId, f32)],
        reward: f32,
    ) {
        if reward.abs() < 0.01 { return; }

        // Adaptive learning rate
        let lr = 0.01 / (1.0 + 0.001 * self.n_updates as f32);

        // For each used node, compute gradient and update
        let score_map: HashMap<NodeId, f32> = scores.iter().copied().collect();

        for &nid in used_nodes {
            let score = score_map.get(&nid).copied().unwrap_or(0.0);
            let h = Self::init_embedding(store, nid);

            // Update message weights for edges touching this node
            for (target, eid) in store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                let edge_type = Self::get_edge_label(store, nid, target, eid);
                if let Some(w) = self.w_message.get_mut(&edge_type) {
                    let h_neighbor = Self::init_embedding(store, target);
                    // gradient = reward × score × outer(h, h_neighbor)
                    let grad_scale = lr * reward * score;
                    for i in 0..DIM {
                        for j in 0..DIM {
                            w[i * DIM + j] += grad_scale * h[i] * h_neighbor[j];
                        }
                    }
                }
            }
        }

        self.n_updates += 1;
    }

    /// Get the edge label between two nodes.
    fn get_edge_label(store: &LpgStore, _from: NodeId, _to: NodeId, _eid: obrain_common::types::EdgeId) -> String {
        // ObrainDB edges have labels — try to read it
        // The edge iterator gives us (target, eid), we need the label
        // For now, check by scanning known edge types via structure
        // Note: LpgStore.get_edge(eid) returns the edge with its label
        if let Some(edge) = store.get_edge(_eid) {
            let s: &str = edge.edge_type.as_ref();
            return s.to_string();
        }
        "UNKNOWN".to_string()
    }

    /// Save GNN weights to PersonaDB as :GNNWeights nodes.
    pub fn save_weights(&self, db: &ObrainDB) {
        let store = db.store();

        // Remove old weights
        for &nid in &store.nodes_by_label("GNNWeights") {
            db.delete_node(nid);
        }

        // Save message weights
        for (edge_type, w) in &self.w_message {
            let data = weights_to_base64(w);
            db.create_node_with_props(&["GNNWeights"], [
                ("layer", Value::String("message".to_string().into())),
                ("edge_type", Value::String(edge_type.clone().into())),
                ("data", Value::String(data.into())),
                ("dim", Value::Int64(self.dim as i64)),
                ("n_updates", Value::Int64(self.n_updates as i64)),
            ]);
        }

        // Save update weights
        for (node_type, w) in &self.w_update {
            let data = weights_to_base64(w);
            db.create_node_with_props(&["GNNWeights"], [
                ("layer", Value::String("update".to_string().into())),
                ("edge_type", Value::String(node_type.clone().into())),
                ("data", Value::String(data.into())),
                ("dim", Value::Int64(self.dim as i64)),
                ("n_updates", Value::Int64(self.n_updates as i64)),
            ]);
        }
    }

    /// Load GNN weights from PersonaDB. Returns true if weights were loaded.
    pub fn load_weights(&mut self, store: &LpgStore) -> bool {
        let weight_nodes = store.nodes_by_label("GNNWeights");
        if weight_nodes.is_empty() { return false; }

        let mut loaded = 0u32;
        for &nid in &weight_nodes {
            if let Some(node) = store.get_node(nid) {
                let layer = node.properties.get(&PropertyKey::from("layer"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let key = node.properties.get(&PropertyKey::from("edge_type"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let data = node.properties.get(&PropertyKey::from("data"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                let n_upd = node.properties.get(&PropertyKey::from("n_updates"))
                    .and_then(|v| if let Value::Int64(n) = v { Some(*n as u64) } else { None })
                    .unwrap_or(0);

                if let Some(w) = base64_to_weights(data, DIM) {
                    match layer {
                        "message" => { self.w_message.insert(key, w); }
                        "update" => { self.w_update.insert(key, w); }
                        _ => {}
                    }
                    self.n_updates = self.n_updates.max(n_upd);
                    loaded += 1;
                }
            }
        }

        loaded > 0
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        0.01 / (1.0 + 0.001 * self.n_updates as f32)
    }

    /// Get number of updates.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Score a specific set of fact NodeIds by relevance to query.
    ///
    /// Wrapper around `forward()` that filters results to only include
    /// the given fact_ids. Facts not in the subgraph get score 0.0.
    /// Returns Vec<(NodeId, f32)> sorted by score descending.
    pub fn score_facts(
        &self,
        store: &LpgStore,
        query_embed: &[f32; DIM],
        fact_ids: &[NodeId],
        hops: usize,
    ) -> Vec<(NodeId, f32)> {
        if fact_ids.is_empty() { return Vec::new(); }

        // Run full forward pass centered on the fact nodes
        let all_scores = self.forward(store, query_embed, fact_ids, hops);

        // Filter to only requested fact_ids
        let requested: HashSet<NodeId> = fact_ids.iter().copied().collect();
        let mut result: Vec<(NodeId, f32)> = all_scores.into_iter()
            .filter(|(nid, _)| requested.contains(nid))
            .collect();

        // Add missing facts with score 0.0 (not reached by subgraph)
        let scored: HashSet<NodeId> = result.iter().map(|(nid, _)| *nid).collect();
        for &fid in fact_ids {
            if !scored.contains(&fid) {
                result.push((fid, 0.0));
            }
        }

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Score arbitrary data graph NodeIds via the MENTIONS bridge.
    ///
    /// Data graph nodes are not directly in the PersonaDB GNN subgraph.
    /// But ConvTurns have MENTIONS edges pointing to data graph nodes.
    /// A data node's score = max GNN score of ConvTurns that MENTION it.
    ///
    /// `persona_store`: the PersonaDB LpgStore (has ConvTurns, Facts, MENTIONS)
    /// `data_node_ids`: NodeIds from the data graph to score
    ///
    /// Returns HashMap<NodeId, f32> for all requested nodes (0.0 if no MENTIONS).
    pub fn score_node_ids(
        &self,
        persona_store: &LpgStore,
        query_embed: &[f32; DIM],
        data_node_ids: &[NodeId],
    ) -> HashMap<NodeId, f32> {
        if data_node_ids.is_empty() || self.n_updates < 20 {
            // Not enough training data — return uniform scores
            return data_node_ids.iter().map(|&nid| (nid, 1.0)).collect();
        }

        // Find ConvTurns that MENTION any of the data nodes
        let data_set: HashSet<NodeId> = data_node_ids.iter().copied().collect();
        let conv_turns = persona_store.nodes_by_label("ConvTurn");

        // Build reverse map: data_node_id → Vec<ConvTurn NodeId>
        let mut mentioned_by: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &ct_id in &conv_turns {
            for (target, eid) in persona_store.edges_from(ct_id, Direction::Outgoing).collect::<Vec<_>>() {
                if let Some(edge) = persona_store.get_edge(eid) {
                    let label: &str = edge.edge_type.as_ref();
                    if label == "MENTIONS" && data_set.contains(&target) {
                        mentioned_by.entry(target).or_default().push(ct_id);
                    }
                }
            }
        }

        // Run GNN forward pass centered on all relevant ConvTurns
        let all_ct_ids: Vec<NodeId> = mentioned_by.values().flatten().copied().collect();
        if all_ct_ids.is_empty() {
            return data_node_ids.iter().map(|&nid| (nid, 0.0)).collect();
        }

        // Get embeddings for ConvTurns via forward pass (2-hop sees Facts + Patterns)
        let subgraph = Self::collect_subgraph(persona_store, &all_ct_ids, 2);
        let mut embeddings: HashMap<NodeId, [f32; DIM]> = HashMap::new();
        for &nid in &subgraph {
            embeddings.insert(nid, Self::init_embedding(persona_store, nid));
        }
        // 2-layer message passing
        for _layer in 0..2 {
            let mut new_embeddings: HashMap<NodeId, [f32; DIM]> = HashMap::new();
            for &nid in &subgraph {
                let mut h = embeddings[&nid];
                for (target, eid) in persona_store.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>() {
                    if !subgraph.contains(&target) { continue; }
                    let edge_type = Self::get_edge_label(persona_store, nid, target, eid);
                    if let Some(w) = self.w_message.get(&edge_type) {
                        let h_neighbor = &embeddings[&target];
                        let msg = mat_vec_mul(w, h_neighbor);
                        for i in 0..DIM { h[i] += relu(msg[i]); }
                    }
                }
                for (source, eid) in persona_store.edges_from(nid, Direction::Incoming).collect::<Vec<_>>() {
                    if !subgraph.contains(&source) { continue; }
                    let edge_type = Self::get_edge_label(persona_store, source, nid, eid);
                    if let Some(w) = self.w_message.get(&edge_type) {
                        let h_neighbor = &embeddings[&source];
                        let msg = mat_vec_mul(w, h_neighbor);
                        for i in 0..DIM { h[i] += relu(msg[i]); }
                    }
                }
                let label = persona_store.get_node(nid)
                    .and_then(|n| n.labels.first().map(|l| { let s: &str = l.as_ref(); s.to_string() }))
                    .unwrap_or_default();
                if let Some(w) = self.w_update.get(&label) {
                    let updated = mat_vec_mul(w, &h);
                    for i in 0..DIM { h[i] = relu(updated[i]); }
                }
                let norm = h.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for i in 0..DIM { h[i] /= norm; }
                new_embeddings.insert(nid, h);
            }
            embeddings = new_embeddings;
        }

        // Score each data node = max dot(query, conv_turn_embedding) across its MENTIONS
        let mut scores: HashMap<NodeId, f32> = HashMap::new();
        for &data_nid in data_node_ids {
            if let Some(ct_ids) = mentioned_by.get(&data_nid) {
                let max_score = ct_ids.iter()
                    .filter_map(|ct| embeddings.get(ct))
                    .map(|h| (0..DIM).map(|i| query_embed[i] * h[i]).sum::<f32>())
                    .fold(f32::NEG_INFINITY, f32::max);
                scores.insert(data_nid, if max_score.is_finite() { max_score } else { 0.0 });
            } else {
                scores.insert(data_nid, 0.0);
            }
        }
        scores
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Linear algebra helpers (no external dependency)
// ═══════════════════════════════════════════════════════════════════════════════

/// Xavier initialization for DIM×DIM matrix.
fn xavier_init(dim: usize) -> Vec<f32> {
    let n = dim * dim;
    let scale = (2.0 / dim as f64).sqrt() as f32;
    let mut w = vec![0.0f32; n];

    // Simple PRNG (xorshift64) seeded from dim for reproducibility
    let mut state: u64 = 0x517cc1b727220a95 ^ (dim as u64 * 31);
    for v in w.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Convert to [-1, 1] then scale
        let f = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        *v = f * scale;
    }
    w
}

/// Hash a string to a DIM-dimensional normalized embedding.
fn hash_to_embedding(s: &str) -> [f32; DIM] {
    let mut state: u64 = 0xcbf29ce484222325; // FNV offset basis
    for b in s.bytes() {
        state ^= b as u64;
        state = state.wrapping_mul(0x100000001b3); // FNV prime
    }

    let mut embed = [0.0f32; DIM];
    for i in 0..DIM {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        embed[i] = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
    }

    // L2 normalize
    let norm = embed.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in embed.iter_mut() {
        *v /= norm;
    }
    embed
}

/// Matrix-vector multiply: W (dim×dim, row-major) × v (dim) → result (dim).
fn mat_vec_mul(w: &[f32], v: &[f32; DIM]) -> [f32; DIM] {
    let mut result = [0.0f32; DIM];
    for i in 0..DIM {
        let mut sum = 0.0f32;
        let row_start = i * DIM;
        for j in 0..DIM {
            sum += w[row_start + j] * v[j];
        }
        result[i] = sum;
    }
    result
}

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Serialize weights to base64 string.
fn weights_to_base64(w: &[f32]) -> String {
    let bytes: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    base64_encode(&bytes)
}

/// Deserialize weights from base64 string.
fn base64_to_weights(s: &str, dim: usize) -> Option<Vec<f32>> {
    let bytes = base64_decode(s)?;
    let expected = dim * dim * 4;
    if bytes.len() != expected { return None; }

    let w: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(w)
}

// Simple base64 encode/decode (no external dependency)

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
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

/// Create a query embedding from a text string (deterministic hash).
pub fn query_embedding(text: &str) -> [f32; DIM] {
    hash_to_embedding(&format!("query:{text}"))
}
