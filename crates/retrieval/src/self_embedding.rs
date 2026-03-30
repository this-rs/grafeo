//! SelfEmbeddingProjector — projects introspective metrics into the LLM's latent space.
//!
//! Architecture:
//!   Input:  ~20 self-metric features (reward, energy, head α, etc.)
//!   Linear: (N_SELF_FEATURES → n_embd) + L2 normalize
//!   Output: Vec<f32> of dimension n_embd, ready for KV cache injection.
//!
//! Training: REINFORCE with lr=1e-4, gradient clipped to [-1, 1].
//! Persistence: save/load weights from PersonaDB :SelfEmbedWeights node.

use persona::SelfMetrics;

/// Number of scalar features extracted from SelfMetrics.
pub const N_SELF_FEATURES: usize = 20;

/// Projects self-metrics into LLM embedding space.
pub struct SelfEmbeddingProjector {
    /// Weight matrix [n_embd × N_SELF_FEATURES], row-major.
    pub w: Vec<f32>,
    /// Bias [n_embd].
    pub b: Vec<f32>,
    /// LLM embedding dimension.
    pub n_embd: usize,
    /// Learning rate for REINFORCE updates.
    pub lr: f32,
    /// Number of training updates performed.
    pub n_updates: u64,
    /// Last projected output (cached for REINFORCE gradient).
    last_input: Vec<f32>,
    last_output: Vec<f32>,
}

impl SelfEmbeddingProjector {
    /// Create with Xavier initialization.
    pub fn new(n_embd: usize) -> Self {
        let limit = (6.0 / (N_SELF_FEATURES + n_embd) as f64).sqrt() as f32;
        let n_weights = n_embd * N_SELF_FEATURES;

        // Deterministic pseudo-random initialization (splitmix64)
        let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut w = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            seed = seed.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let r = (z as f64) / (u64::MAX as f64); // [0, 1)
            w.push((r as f32 * 2.0 - 1.0) * limit);
        }

        Self {
            w,
            b: vec![0.0; n_embd],
            n_embd,
            lr: 1e-4,
            n_updates: 0,
            last_input: Vec::new(),
            last_output: Vec::new(),
        }
    }

    /// Extract scalar features from SelfMetrics.
    pub fn featurize(metrics: &SelfMetrics) -> Vec<f32> {
        let mut features = Vec::with_capacity(N_SELF_FEATURES);

        // Core metrics (4)
        features.push(metrics.reward_avg as f32);
        features.push(metrics.mask_reward_avg as f32);
        features.push(metrics.gnn_facts_active as f32 / 20.0); // normalize to ~[0,1]

        // Learning trend encoded as float (3 values → one-hot-ish)
        let trend = match metrics.learning_trend.as_str() {
            "improving" => 1.0,
            "stable" => 0.0,
            "declining" => -1.0,
            "cold_start" => -0.5,
            _ => 0.0,
        };
        features.push(trend);

        // Head router top-5 α values (5 × 2 = 10 features)
        for i in 0..5 {
            if let Some((head_idx, alpha)) = metrics.head_router_top5.get(i) {
                features.push(*head_idx as f32 / 32.0); // normalize head index
                features.push(*alpha);
            } else {
                features.push(0.0);
                features.push(0.0);
            }
        }

        // Formula name hash (1 feature — cheap signal about which formula is active)
        let name_hash = metrics
            .formula_active_name
            .bytes()
            .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        features.push((name_hash as f32) / (u32::MAX as f32));

        // Pad to N_SELF_FEATURES
        while features.len() < N_SELF_FEATURES {
            features.push(0.0);
        }
        features.truncate(N_SELF_FEATURES);
        features
    }

    /// Project metrics → n_embd vector (L2-normalized).
    pub fn project(&mut self, metrics: &SelfMetrics) -> Vec<f32> {
        let input = Self::featurize(metrics);
        let mut output = vec![0.0f32; self.n_embd];

        // matmul: output = W × input + b
        for i in 0..self.n_embd {
            let mut sum = self.b[i];
            let row_start = i * N_SELF_FEATURES;
            for j in 0..N_SELF_FEATURES {
                sum += self.w[row_start + j] * input[j];
            }
            output[i] = sum;
        }

        // L2 normalize
        let norm = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut output {
                *x /= norm;
            }
        }

        // Cache for REINFORCE
        self.last_input = input;
        self.last_output = output.clone();

        output
    }

    /// REINFORCE update: reward > 0 → reinforce current projection direction.
    /// reward < 0 → push weights away from current projection.
    pub fn train_step(&mut self, reward: f32) {
        if self.last_input.is_empty() || self.last_output.is_empty() {
            return;
        }

        let grad_scale = reward * self.lr;

        // Gradient: d(output)/d(w_ij) ≈ input[j] (pre-normalization, simplified)
        // REINFORCE: Δw_ij = lr × reward × input[j] × sign(output[i])
        for i in 0..self.n_embd {
            let sign_out = if self.last_output[i] >= 0.0 { 1.0 } else { -1.0 };
            let row_start = i * N_SELF_FEATURES;
            for j in 0..N_SELF_FEATURES {
                let grad = grad_scale * self.last_input[j] * sign_out;
                // Gradient clipping
                let clipped = grad.clamp(-0.01, 0.01);
                self.w[row_start + j] += clipped;
            }
            // Bias update
            self.b[i] += (grad_scale * sign_out).clamp(-0.01, 0.01);
        }

        self.n_updates += 1;
    }

    /// Serialize weights to a flat f32 vec (for PersonaDB storage).
    pub fn save_weights(&self) -> Vec<f32> {
        let mut data = Vec::with_capacity(self.w.len() + self.b.len() + 3);
        // Header: [n_embd as f32, N_SELF_FEATURES as f32, n_updates as f32]
        data.push(self.n_embd as f32);
        data.push(N_SELF_FEATURES as f32);
        data.push(self.n_updates as f32);
        data.extend_from_slice(&self.w);
        data.extend_from_slice(&self.b);
        data
    }

    /// Load weights from a flat f32 vec. Returns true if successful.
    pub fn load_weights(&mut self, data: &[f32]) -> bool {
        if data.len() < 3 {
            return false;
        }
        let stored_n_embd = data[0] as usize;
        let stored_n_features = data[1] as usize;
        let stored_updates = data[2] as u64;

        if stored_n_embd != self.n_embd || stored_n_features != N_SELF_FEATURES {
            return false;
        }

        let expected = 3 + self.n_embd * N_SELF_FEATURES + self.n_embd;
        if data.len() < expected {
            return false;
        }

        let w_len = self.w.len();
        let b_len = self.b.len();
        self.w.copy_from_slice(&data[3..3 + w_len]);
        self.b
            .copy_from_slice(&data[3 + w_len..3 + w_len + b_len]);
        self.n_updates = stored_updates;
        true
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_metrics() -> SelfMetrics {
        SelfMetrics {
            reward_avg: 0.5,
            mask_reward_avg: 0.3,
            head_router_top5: vec![(0, 0.8), (5, 0.6), (12, 0.4)],
            formula_active_name: "F1-GravityLinear".to_string(),
            formula_explanation: "test".to_string(),
            gnn_facts_active: 10,
            learning_trend: "improving".to_string(),
        }
    }

    #[test]
    fn test_featurize_length() {
        let features = SelfEmbeddingProjector::featurize(&default_metrics());
        assert_eq!(features.len(), N_SELF_FEATURES);
    }

    #[test]
    fn test_project_dimension() {
        let n_embd = 128;
        let mut proj = SelfEmbeddingProjector::new(n_embd);
        let output = proj.project(&default_metrics());
        assert_eq!(output.len(), n_embd);
    }

    #[test]
    fn test_project_l2_normalized() {
        let mut proj = SelfEmbeddingProjector::new(64);
        let output = proj.project(&default_metrics());
        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "L2 norm should be ~1.0, got {norm}");
    }

    #[test]
    fn test_train_step_changes_weights() {
        let mut proj = SelfEmbeddingProjector::new(32);
        let w_before: Vec<f32> = proj.w.clone();
        proj.project(&default_metrics());
        proj.train_step(1.0);
        assert_ne!(proj.w, w_before, "weights should change after train step");
        assert_eq!(proj.n_updates, 1);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut proj = SelfEmbeddingProjector::new(32);
        proj.project(&default_metrics());
        proj.train_step(0.5);
        proj.train_step(-0.3);

        let saved = proj.save_weights();
        let output_before = proj.project(&default_metrics());

        let mut proj2 = SelfEmbeddingProjector::new(32);
        assert!(proj2.load_weights(&saved));
        let output_after = proj2.project(&default_metrics());

        assert_eq!(proj2.n_updates, 2);
        for (a, b) in output_before.iter().zip(output_after.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "outputs should match after load: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_different_metrics_different_output() {
        let mut proj = SelfEmbeddingProjector::new(32);
        let m1 = default_metrics();
        let mut m2 = default_metrics();
        m2.reward_avg = -0.8;
        m2.learning_trend = "declining".to_string();

        let o1 = proj.project(&m1);
        let o2 = proj.project(&m2);

        // Cosine distance should be > 0 (different inputs → different outputs)
        let dot: f32 = o1.iter().zip(o2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot < 0.99,
            "different metrics should produce different projections, cosine={dot}"
        );
    }

    #[test]
    fn test_load_wrong_dimensions_fails() {
        let mut proj = SelfEmbeddingProjector::new(32);
        let bad_data = vec![64.0, N_SELF_FEATURES as f32, 0.0]; // wrong n_embd
        assert!(!proj.load_weights(&bad_data));
    }
}
