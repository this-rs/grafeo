//! HeadProfiler — Captures per-head attention patterns for topology-aware head routing.
//!
//! Phase B of the Cortex architecture: analyze which attention heads are naturally
//! sensitive to different graph topology banks (core, relations, 2-hop, background).
//!
//! ## Architecture
//! 1. An eval callback captures "kq_soft_max" tensors during inference (READ-ONLY)
//! 2. For each layer/head, we compute mean attention on positions from each bank
//! 3. After N queries, K-means clustering groups heads by topology affinity
//! 4. The resulting HeadProfile guides the HeadRouter (B2)
//!
//! ## Gotcha
//! The eval callback is READ-ONLY — we can observe tensors but NOT modify them.
//! The callback fires for every tensor in the compute graph, so we must filter
//! by name ("kq_soft_max") to avoid processing irrelevant tensors.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Number of banks in the topology masking system.
pub const N_BANKS: usize = 4;

/// A single snapshot of attention distribution for one head on one query.
#[derive(Debug, Clone)]
pub struct HeadSnapshot {
    /// Layer index in the model.
    pub layer: u32,
    /// Head index within the layer.
    pub head: u32,
    /// Mean attention weight on each bank's positions [core, relations, 2-hop, background].
    pub bank_attention: [f32; N_BANKS],
}

/// Profile result for a single head after clustering.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeadProfile {
    /// Layer index.
    pub layer: u32,
    /// Head index.
    pub head: u32,
    /// Cluster assignment (0..n_clusters).
    pub cluster_id: u32,
    /// Mean attention vector across all observed queries [core, relations, 2-hop, background].
    pub mean_attention: [f32; N_BANKS],
    /// Label for the cluster (e.g., "fact-sensitive", "context-sensitive").
    pub label: String,
}

/// Bank assignment: maps KV cache positions to bank IDs.
/// Set before each query so the eval callback knows which positions belong to which bank.
#[derive(Debug, Clone, Default)]
pub struct BankAssignment {
    /// position → bank_id (0..N_BANKS). Positions not in this map are ignored.
    pub pos_to_bank: HashMap<i32, u32>,
}

/// Shared state for the eval callback.
/// The callback captures attention patterns and stores them here.
#[derive(Debug)]
pub struct ProfilerState {
    /// Current bank assignments (set before each query).
    pub bank_assignment: BankAssignment,
    /// Snapshots collected during current query.
    pub current_snapshots: Vec<HeadSnapshot>,
    /// Whether profiling is active (callback will skip if false).
    pub active: bool,
    /// Total positions in the current mask (for normalization).
    pub n_total_positions: u32,
}

impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            bank_assignment: BankAssignment::default(),
            current_snapshots: Vec::new(),
            active: false,
            n_total_positions: 0,
        }
    }
}

/// Thread-safe handle to profiler state, shared with the eval callback.
pub type ProfilerHandle = Arc<Mutex<ProfilerState>>;

/// HeadProfiler — captures and analyzes per-head attention patterns.
pub struct HeadProfiler {
    /// Shared state accessed by the eval callback.
    pub state: ProfilerHandle,
    /// All snapshots across all queries (one entry per query = Vec<HeadSnapshot>).
    history: Vec<Vec<HeadSnapshot>>,
    /// Number of layers in the model.
    n_layers: u32,
    /// Number of heads per layer.
    n_heads: u32,
}

impl HeadProfiler {
    /// Create a new HeadProfiler.
    pub fn new(n_layers: u32, n_heads: u32) -> Self {
        Self {
            state: Arc::new(Mutex::new(ProfilerState::default())),
            history: Vec::new(),
            n_layers,
            n_heads,
        }
    }

    /// Get a clone of the shared state handle (for passing to the eval callback).
    pub fn handle(&self) -> ProfilerHandle {
        Arc::clone(&self.state)
    }

    /// Set bank assignments before a query.
    /// Call this before each `llama_decode` to tell the profiler which positions belong to which bank.
    pub fn set_bank_assignment(&self, pos_to_bank: HashMap<i32, u32>, n_total_positions: u32) {
        let mut state = self.state.lock().unwrap();
        state.bank_assignment.pos_to_bank = pos_to_bank;
        state.n_total_positions = n_total_positions;
    }

    /// Enable profiling. The eval callback will start capturing attention tensors.
    pub fn start(&self) {
        let mut state = self.state.lock().unwrap();
        state.active = true;
        state.current_snapshots.clear();
    }

    /// Disable profiling and collect the current query's snapshots.
    pub fn stop_and_collect(&mut self) {
        let snapshots = {
            let mut state = self.state.lock().unwrap();
            state.active = false;
            std::mem::take(&mut state.current_snapshots)
        };
        if !snapshots.is_empty() {
            self.history.push(snapshots);
        }
    }

    /// Number of queries profiled so far.
    pub fn n_queries(&self) -> usize {
        self.history.len()
    }

    /// Compute mean attention per head across all observed queries.
    /// Returns a map of (layer, head) → mean [bank_attention; N_BANKS].
    pub fn mean_attention_per_head(&self) -> HashMap<(u32, u32), [f32; N_BANKS]> {
        let mut accum: HashMap<(u32, u32), (Vec<[f32; N_BANKS]>,)> = HashMap::new();

        for query_snapshots in &self.history {
            for snap in query_snapshots {
                let key = (snap.layer, snap.head);
                accum
                    .entry(key)
                    .or_insert_with(|| (Vec::new(),))
                    .0
                    .push(snap.bank_attention);
            }
        }

        accum
            .into_iter()
            .map(|(key, (samples,))| {
                let n = samples.len() as f32;
                let mut mean = [0.0f32; N_BANKS];
                for s in &samples {
                    for b in 0..N_BANKS {
                        mean[b] += s[b];
                    }
                }
                for b in 0..N_BANKS {
                    mean[b] /= n;
                }
                (key, mean)
            })
            .collect()
    }

    /// Cluster heads by topology affinity using K-means.
    ///
    /// Groups heads into `n_clusters` based on their mean attention distribution
    /// across banks. Returns a HeadProfile for each head.
    pub fn cluster_heads(&self, n_clusters: usize) -> Vec<HeadProfile> {
        let means = self.mean_attention_per_head();
        if means.is_empty() {
            return Vec::new();
        }

        // Collect all (layer, head) keys sorted
        let mut keys: Vec<(u32, u32)> = means.keys().copied().collect();
        keys.sort();

        let vectors: Vec<[f32; N_BANKS]> = keys.iter().map(|k| means[k]).collect();
        let n = vectors.len();
        let k = n_clusters.min(n);

        // K-means initialization: spread initial centroids evenly
        let mut centroids: Vec<[f32; N_BANKS]> = (0..k).map(|i| vectors[i * n / k]).collect();

        let mut assignments = vec![0usize; n];

        // K-means iterations (max 50)
        for _ in 0..50 {
            let mut changed = false;

            // Assign each point to nearest centroid
            for (i, vec) in vectors.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da: f32 = (0..N_BANKS).map(|d| (a[d] - vec[d]).powi(2)).sum();
                        let db: f32 = (0..N_BANKS).map(|d| (b[d] - vec[d]).powi(2)).sum();
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap();

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Recompute centroids
            for c in 0..k {
                let mut sum = [0.0f32; N_BANKS];
                let mut count = 0;
                for (i, vec) in vectors.iter().enumerate() {
                    if assignments[i] == c {
                        for b in 0..N_BANKS {
                            sum[b] += vec[b];
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    for b in 0..N_BANKS {
                        centroids[c][b] = sum[b] / count as f32;
                    }
                }
            }
        }

        // Label clusters based on dominant bank attention
        let cluster_labels: Vec<String> = centroids
            .iter()
            .map(|c| {
                let max_bank = c
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                match max_bank {
                    0 => "fact-sensitive".to_string(),
                    1 => "relation-sensitive".to_string(),
                    2 => "context-sensitive".to_string(),
                    3 => "background-sensitive".to_string(),
                    _ => "general".to_string(),
                }
            })
            .collect();

        // Build profiles
        keys.iter()
            .enumerate()
            .map(|(i, &(layer, head))| HeadProfile {
                layer,
                head,
                cluster_id: assignments[i] as u32,
                mean_attention: vectors[i],
                label: cluster_labels[assignments[i]].clone(),
            })
            .collect()
    }

    /// Export profiles to JSON.
    pub fn export_json(&self, n_clusters: usize) -> String {
        let profiles = self.cluster_heads(n_clusters);
        serde_json::to_string_pretty(&profiles).unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Eval Callback — captures "kq_soft_max" tensors during inference
// ═══════════════════════════════════════════════════════════════════════

use crate::ffi;

/// The eval callback function passed to llama.cpp context params.
///
/// # Safety
/// Called from C code. `user_data` must be a valid `ProfilerHandle` (Arc<Mutex<ProfilerState>>).
///
/// When `ask == true`: return true if we want this tensor's data (we only want "kq_soft_max").
/// When `ask == false`: the tensor data is available for reading.
pub unsafe extern "C" fn profiler_eval_callback(
    t: *mut ffi::ggml_tensor,
    ask: bool,
    user_data: *mut std::os::raw::c_void,
) -> bool {
    if t.is_null() || user_data.is_null() {
        return true; // continue graph computation
    }

    // SAFETY: t is a valid tensor pointer from GGML, user_data is our ProfilerHandle
    let tensor = unsafe { &*t };
    let name = unsafe {
        std::ffi::CStr::from_ptr(tensor.name.as_ptr())
            .to_str()
            .unwrap_or("")
    };

    // We only care about post-softmax attention weights
    if !name.starts_with("kq_soft_max") {
        return true;
    }

    // Reconstruct the Arc without dropping it (we're borrowing)
    // SAFETY: user_data was created from Arc::into_raw in set_profiler
    let handle = unsafe { &*(user_data as *const Mutex<ProfilerState>) };

    if ask {
        // Ask phase: return true to request this tensor's data
        let state = handle.lock().unwrap();
        return state.active;
    }

    // Data phase: extract attention patterns per head
    let mut state = match handle.lock() {
        Ok(s) => s,
        Err(_) => return true,
    };

    if !state.active || state.bank_assignment.pos_to_bank.is_empty() {
        return true;
    }

    // Parse layer index from tensor name: "kq_soft_max-0", "kq_soft_max-1", etc.
    let layer: u32 = name
        .strip_prefix("kq_soft_max-")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Tensor shape: kq_soft_max has shape [n_kv, n_tokens, n_head, 1] typically
    // After softmax, each row sums to 1.0
    let ne = tensor.ne;
    let nb = tensor.nb;
    let n_kv = ne[0] as usize; // number of KV positions attended to
    let n_tokens = ne[1] as usize; // number of query tokens (batch)
    let n_head = ne[2] as usize; // number of attention heads

    if tensor.data.is_null() || n_kv == 0 || n_head == 0 {
        return true;
    }

    // For each head, compute mean attention weight per bank
    for h in 0..n_head {
        let mut bank_sum = [0.0f32; N_BANKS];
        let mut bank_count = [0u32; N_BANKS];
        let mut total_weight = 0.0f32;

        // Average across all tokens in the batch
        for tok in 0..n_tokens {
            // SAFETY: Pointer arithmetic within tensor data bounds (n_kv * n_tokens * n_head)
            let row_ptr =
                unsafe { (tensor.data as *const u8).add(h * nb[2] + tok * nb[1]) as *const f32 };

            for kv_pos in 0..n_kv {
                let attn_weight = unsafe { *row_ptr.add(kv_pos) };
                let pos = kv_pos as i32;

                if let Some(&bank) = state.bank_assignment.pos_to_bank.get(&pos) {
                    if (bank as usize) < N_BANKS {
                        bank_sum[bank as usize] += attn_weight;
                        bank_count[bank as usize] += 1;
                    }
                }
                total_weight += attn_weight;
            }
        }

        // Normalize: fraction of total attention going to each bank
        if total_weight > 1e-8 {
            let mut bank_attention = [0.0f32; N_BANKS];
            for b in 0..N_BANKS {
                bank_attention[b] = bank_sum[b] / total_weight;
            }

            state.current_snapshots.push(HeadSnapshot {
                layer,
                head: h as u32,
                bank_attention,
            });
        }
    }

    true // continue graph computation
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = HeadProfiler::new(32, 32);
        assert_eq!(profiler.n_queries(), 0);
        assert_eq!(profiler.n_layers, 32);
        assert_eq!(profiler.n_heads, 32);
    }

    #[test]
    fn test_clustering_synthetic_3_groups() {
        // Simulate 3 distinct head types:
        // Group A (heads 0-3): attend mostly to bank 0 (core)
        // Group B (heads 4-7): attend mostly to bank 2 (2-hop)
        // Group C (heads 8-11): balanced attention
        let mut profiler = HeadProfiler::new(1, 12);

        for _query in 0..20 {
            let mut snapshots = Vec::new();

            // Group A: fact-sensitive
            for h in 0..4 {
                snapshots.push(HeadSnapshot {
                    layer: 0,
                    head: h,
                    bank_attention: [0.7, 0.1, 0.1, 0.1],
                });
            }
            // Group B: context-sensitive
            for h in 4..8 {
                snapshots.push(HeadSnapshot {
                    layer: 0,
                    head: h,
                    bank_attention: [0.1, 0.1, 0.7, 0.1],
                });
            }
            // Group C: balanced
            for h in 8..12 {
                snapshots.push(HeadSnapshot {
                    layer: 0,
                    head: h,
                    bank_attention: [0.25, 0.25, 0.25, 0.25],
                });
            }

            profiler.history.push(snapshots);
        }

        let profiles = profiler.cluster_heads(3);
        assert_eq!(profiles.len(), 12);

        // Verify that heads in the same synthetic group got the same cluster
        let cluster_a = profiles[0].cluster_id;
        for p in &profiles[0..4] {
            assert_eq!(
                p.cluster_id, cluster_a,
                "heads 0-3 should be in same cluster"
            );
        }

        let cluster_b = profiles[4].cluster_id;
        for p in &profiles[4..8] {
            assert_eq!(
                p.cluster_id, cluster_b,
                "heads 4-7 should be in same cluster"
            );
        }

        let cluster_c = profiles[8].cluster_id;
        for p in &profiles[8..12] {
            assert_eq!(
                p.cluster_id, cluster_c,
                "heads 8-11 should be in same cluster"
            );
        }

        // All 3 clusters should be different
        assert_ne!(cluster_a, cluster_b);
        assert_ne!(cluster_a, cluster_c);
        assert_ne!(cluster_b, cluster_c);

        eprintln!("Cluster labels:");
        for p in &profiles {
            eprintln!(
                "  head {} → cluster {} ({}), mean={:?}",
                p.head, p.cluster_id, p.label, p.mean_attention
            );
        }
    }

    #[test]
    fn test_mean_attention_computation() {
        let mut profiler = HeadProfiler::new(1, 2);

        // Query 1
        profiler.history.push(vec![
            HeadSnapshot {
                layer: 0,
                head: 0,
                bank_attention: [0.8, 0.1, 0.05, 0.05],
            },
            HeadSnapshot {
                layer: 0,
                head: 1,
                bank_attention: [0.1, 0.1, 0.7, 0.1],
            },
        ]);

        // Query 2
        profiler.history.push(vec![
            HeadSnapshot {
                layer: 0,
                head: 0,
                bank_attention: [0.6, 0.2, 0.1, 0.1],
            },
            HeadSnapshot {
                layer: 0,
                head: 1,
                bank_attention: [0.2, 0.1, 0.5, 0.2],
            },
        ]);

        let means = profiler.mean_attention_per_head();
        let h0 = means[&(0, 0)];
        let h1 = means[&(0, 1)];

        // Head 0: mean = [(0.8+0.6)/2, (0.1+0.2)/2, (0.05+0.1)/2, (0.05+0.1)/2]
        //       = [0.7, 0.15, 0.075, 0.075]
        assert!((h0[0] - 0.7).abs() < 0.01);
        assert!((h0[1] - 0.15).abs() < 0.01);

        // Head 1: bank 2 dominant
        assert!(h1[2] > h1[0], "head 1 should be context-sensitive");
    }

    #[test]
    fn test_export_json() {
        let mut profiler = HeadProfiler::new(1, 2);
        profiler.history.push(vec![
            HeadSnapshot {
                layer: 0,
                head: 0,
                bank_attention: [0.8, 0.1, 0.05, 0.05],
            },
            HeadSnapshot {
                layer: 0,
                head: 1,
                bank_attention: [0.1, 0.1, 0.7, 0.1],
            },
        ]);

        let json = profiler.export_json(2);
        assert!(json.contains("fact-sensitive") || json.contains("context-sensitive"));
        let parsed: Vec<HeadProfile> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 2);
    }
}
