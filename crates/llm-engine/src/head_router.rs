//! HeadRouter — Learnable per-head topology routing via REINFORCE.
//!
//! Phase B of the Cortex architecture: each attention head learns which
//! graph topology banks to attend to, based on reward signals.
//!
//! ## Architecture
//! - α-weights: `[n_head × n_bank]` logits, initialized to 0.0 (sigmoid=0.5 = neutral)
//! - Forward: `sigmoid(α[h][b])` → probability head h sees bank b
//! - Backward: REINFORCE update `α += lr × reward × indicator(bank visible)`
//! - The α-weights replace the fixed BankConfig ratios from Phase A
//!
//! ## Key insight (Decision e2ef9780)
//! GGML indexes masks with `(head_idx % mask->ne[2]) * mask->nb[2]`.
//! Per-head masking works without kernel changes — the mask encodes
//! topology via bank→head routing, and ne[2] broadcasting handles GQA.

use crate::profiler::N_BANKS;

/// HeadRouter — learnable routing weights for per-head topology masking.
#[derive(Debug, Clone)]
pub struct HeadRouter {
    /// Logit weights [n_head * n_bank]. sigmoid(α) = probability of visibility.
    pub alpha: Vec<f32>,
    /// Number of attention heads.
    pub n_head: usize,
    /// Number of topology banks.
    pub n_bank: usize,
    /// Learning rate for REINFORCE updates.
    pub lr: f32,
    /// Number of updates applied.
    pub n_updates: u64,
    /// Warmup: no learning for the first N queries (observation only).
    pub warmup_remaining: u32,
    /// Per-head α for self-embedding positions (proprioceptive channel).
    /// sigmoid(self_alpha[h]) = probability head h attends to self-embed positions.
    /// Separate from bank α because self-embeds are a different signal type.
    pub self_alpha: Vec<f32>,
}

/// Visibility threshold: sigmoid(α) > this means "head sees bank".
pub const VISIBILITY_THRESHOLD: f32 = 0.3;

/// Clamp range for α to prevent saturation.
const ALPHA_CLIP_MIN: f32 = -5.0;
const ALPHA_CLIP_MAX: f32 = 5.0;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl HeadRouter {
    /// Create a new HeadRouter with neutral α-weights (sigmoid(0)=0.5).
    pub fn new(n_head: usize, n_bank: usize, lr: f32, warmup: u32) -> Self {
        Self {
            alpha: vec![0.0; n_head * n_bank],
            n_head,
            n_bank,
            lr,
            n_updates: 0,
            warmup_remaining: warmup,
            self_alpha: vec![0.0; n_head], // neutral: sigmoid(0)=0.5
        }
    }

    /// Create with default parameters (4 banks, lr=0.01, warmup=50).
    pub fn default_for_model(n_head: usize) -> Self {
        Self::new(n_head, N_BANKS, 0.01, 50)
    }

    /// Forward pass: compute visibility probabilities [n_head × n_bank].
    ///
    /// Returns `sigmoid(α[h*n_bank+b])` for each (head, bank) pair.
    /// Values near 1.0 = head strongly prefers seeing this bank.
    /// Values near 0.0 = head prefers NOT seeing this bank.
    pub fn forward(&self) -> Vec<f32> {
        self.alpha.iter().map(|&a| sigmoid(a)).collect()
    }

    /// Get visibility probability for a specific (head, bank) pair.
    pub fn prob(&self, head: usize, bank: usize) -> f32 {
        if head >= self.n_head || bank >= self.n_bank {
            return 0.5; // neutral fallback
        }
        sigmoid(self.alpha[head * self.n_bank + bank])
    }

    /// Check if a head should see a bank (prob > threshold).
    pub fn is_visible(&self, head: usize, bank: usize) -> bool {
        self.prob(head, bank) > VISIBILITY_THRESHOLD
    }

    /// Backward pass: REINFORCE update of α-weights.
    ///
    /// For each head h and bank b:
    ///   `α[h][b] += lr × reward[h] × visibility_fraction[h][b]`
    ///
    /// where visibility_fraction = proportion of the bank's nodes that were
    /// actually visible to head h (from the forward pass).
    ///
    /// # Arguments
    /// - `rewards_per_head`: reward signal per head `[n_head]`
    /// - `visibility`: which banks were visible for each head `[n_head × n_bank]`
    ///   (1.0 = fully visible, 0.0 = fully masked)
    pub fn backward(&mut self, rewards_per_head: &[f32], visibility: &[f32]) {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return;
        }

        let n = self.n_head.min(rewards_per_head.len());
        for h in 0..n {
            let reward = rewards_per_head[h];
            for b in 0..self.n_bank {
                let idx = h * self.n_bank + b;
                if idx >= visibility.len() {
                    continue;
                }

                let vis = visibility[idx];
                let grad = reward * vis;
                self.alpha[idx] =
                    (self.alpha[idx] + self.lr * grad).clamp(ALPHA_CLIP_MIN, ALPHA_CLIP_MAX);
            }
        }

        self.n_updates += 1;
    }

    /// Simplified backward: uniform reward across all heads.
    ///
    /// Used when we don't have per-head reward (Phase B without B3 ablation).
    /// The reward is distributed based on which banks each head saw.
    pub fn backward_uniform(&mut self, reward: f32) {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return;
        }

        let probs = self.forward();
        for h in 0..self.n_head {
            for b in 0..self.n_bank {
                let idx = h * self.n_bank + b;
                let vis = probs[idx];
                let grad = reward * vis;
                self.alpha[idx] =
                    (self.alpha[idx] + self.lr * grad).clamp(ALPHA_CLIP_MIN, ALPHA_CLIP_MAX);
            }
        }

        self.n_updates += 1;
    }

    /// Get α-weights as a flat Vec (for serialization).
    pub fn weights(&self) -> &[f32] {
        &self.alpha
    }

    /// Restore α-weights from a flat Vec.
    pub fn load_weights(&mut self, weights: &[f32]) {
        if weights.len() == self.alpha.len() {
            self.alpha.copy_from_slice(weights);
        }
    }

    /// Get self-embedding visibility probability for a head.
    pub fn self_prob(&self, head: usize) -> f32 {
        if head >= self.n_head {
            return 0.5;
        }
        sigmoid(self.self_alpha[head])
    }

    /// Check if a head should attend to self-embedding positions.
    pub fn self_is_visible(&self, head: usize) -> bool {
        self.self_prob(head) > VISIBILITY_THRESHOLD
    }

    /// REINFORCE update for self-embedding α, given a uniform reward.
    ///
    /// Called when the model uses (or fails to use) self-metrics.
    /// Positive reward → heads that attended to self-embeds are reinforced.
    pub fn backward_self(&mut self, reward: f32) {
        if self.warmup_remaining > 0 {
            return; // warmup managed by main backward()
        }
        for h in 0..self.n_head {
            let p = sigmoid(self.self_alpha[h]);
            let grad = reward * p;
            self.self_alpha[h] =
                (self.self_alpha[h] + self.lr * grad).clamp(ALPHA_CLIP_MIN, ALPHA_CLIP_MAX);
        }
    }

    /// Get self_alpha weights for serialization.
    pub fn self_weights(&self) -> &[f32] {
        &self.self_alpha
    }

    /// Restore self_alpha weights.
    pub fn load_self_weights(&mut self, weights: &[f32]) {
        if weights.len() == self.self_alpha.len() {
            self.self_alpha.copy_from_slice(weights);
        }
    }

    /// Compute head specialization entropy (lower = more specialized).
    ///
    /// For each head, compute the entropy of its bank visibility distribution.
    /// Return the mean entropy across all heads.
    /// - Low entropy (~0) = head is specialized (strongly prefers specific banks)
    /// - High entropy (~ln(n_bank)) = head is general (equal attention to all banks)
    pub fn specialization_entropy(&self) -> f32 {
        let probs = self.forward();
        let mut total_entropy = 0.0f32;

        for h in 0..self.n_head {
            let start = h * self.n_bank;
            let head_probs = &probs[start..start + self.n_bank];

            // Normalize to distribution
            let sum: f32 = head_probs.iter().sum();
            if sum < 1e-8 {
                continue;
            }

            let mut entropy = 0.0f32;
            for &p in head_probs {
                let p_norm = p / sum;
                if p_norm > 1e-8 {
                    entropy -= p_norm * p_norm.ln();
                }
            }
            total_entropy += entropy;
        }

        total_entropy / self.n_head as f32
    }

    /// Debug summary: α-weights stats per bank.
    pub fn summary(&self) -> String {
        let probs = self.forward();
        let mut bank_means = vec![0.0f32; self.n_bank];
        for h in 0..self.n_head {
            for b in 0..self.n_bank {
                bank_means[b] += probs[h * self.n_bank + b];
            }
        }
        for b in 0..self.n_bank {
            bank_means[b] /= self.n_head as f32;
        }

        let labels = ["core", "relations", "2-hop", "background"];
        let mut s = format!(
            "HeadRouter: {} updates, lr={}, entropy={:.3}\n",
            self.n_updates,
            self.lr,
            self.specialization_entropy()
        );
        for b in 0..self.n_bank {
            let label = labels.get(b).unwrap_or(&"?");
            s += &format!("  bank {} ({}): mean_prob={:.3}\n", b, label, bank_means[b]);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_neutral() {
        let router = HeadRouter::new(4, 4, 0.01, 0);
        assert_eq!(router.alpha.len(), 16);
        assert!(router.alpha.iter().all(|&a| a == 0.0));

        // Forward should return 0.5 for all
        let probs = router.forward();
        assert_eq!(probs.len(), 16);
        for &p in &probs {
            assert!(
                (p - 0.5).abs() < 0.001,
                "neutral α=0 should give sigmoid=0.5, got {p}"
            );
        }
    }

    #[test]
    fn test_forward_extreme_values() {
        let mut router = HeadRouter::new(2, 2, 0.01, 0);
        router.alpha[0] = 5.0; // head 0, bank 0 → ~0.993
        router.alpha[1] = -5.0; // head 0, bank 1 → ~0.007

        let probs = router.forward();
        assert!(probs[0] > 0.99, "sigmoid(5.0) should be ~1.0");
        assert!(probs[1] < 0.01, "sigmoid(-5.0) should be ~0.0");

        assert!(router.is_visible(0, 0));
        assert!(!router.is_visible(0, 1));
    }

    #[test]
    fn test_backward_positive_reward() {
        let mut router = HeadRouter::new(2, 2, 0.1, 0);
        let rewards = vec![1.0, 1.0]; // positive reward for both heads
        let visibility = vec![1.0, 0.0, 0.0, 1.0]; // head0→bank0, head1→bank1 visible

        let alpha_before = router.alpha.clone();
        router.backward(&rewards, &visibility);

        // α[0] (head0, bank0) should increase (positive reward × visible)
        assert!(
            router.alpha[0] > alpha_before[0],
            "positive reward + visible → α should increase"
        );
        // α[1] (head0, bank1) should NOT change (not visible)
        assert_eq!(
            router.alpha[1], alpha_before[1],
            "not visible → α should not change"
        );
        // α[3] (head1, bank1) should increase
        assert!(router.alpha[3] > alpha_before[3]);
    }

    #[test]
    fn test_backward_negative_reward() {
        let mut router = HeadRouter::new(2, 2, 0.1, 0);
        let rewards = vec![-0.5, -0.5];
        let visibility = vec![1.0, 1.0, 1.0, 1.0]; // all visible

        let alpha_before = router.alpha.clone();
        router.backward(&rewards, &visibility);

        // All α should decrease (negative reward × visible)
        for i in 0..4 {
            assert!(
                router.alpha[i] < alpha_before[i],
                "negative reward + visible → α[{i}] should decrease"
            );
        }
    }

    #[test]
    fn test_warmup_skips_learning() {
        let mut router = HeadRouter::new(2, 2, 0.1, 3);
        let rewards = vec![1.0, 1.0];
        let visibility = vec![1.0, 1.0, 1.0, 1.0];

        // First 3 calls should be warmup (no learning)
        for _ in 0..3 {
            router.backward(&rewards, &visibility);
            assert!(
                router.alpha.iter().all(|&a| a == 0.0),
                "warmup: α should stay 0"
            );
        }

        // 4th call should learn
        router.backward(&rewards, &visibility);
        assert!(
            router.alpha.iter().any(|&a| a != 0.0),
            "after warmup: α should change"
        );
    }

    #[test]
    fn test_alpha_clipping() {
        let mut router = HeadRouter::new(1, 1, 10.0, 0); // very high lr
        let rewards = vec![1.0];
        let visibility = vec![1.0];

        // Many updates should clip at 5.0
        for _ in 0..100 {
            router.backward(&rewards, &visibility);
        }
        assert!((router.alpha[0] - 5.0).abs() < 0.001, "should clip at 5.0");

        // Negative rewards should clip at -5.0
        let neg_rewards = vec![-1.0];
        for _ in 0..200 {
            router.backward(&neg_rewards, &visibility);
        }
        assert!(
            (router.alpha[0] - (-5.0)).abs() < 0.001,
            "should clip at -5.0"
        );
    }

    #[test]
    fn test_convergence_200_updates() {
        // Simulate: bank 0 gets reward +1 per visible unit, bank 3 gets -1 per visible unit.
        // This should drive α[bank 0] up and α[bank 3] down.
        let mut router = HeadRouter::new(4, N_BANKS, 0.05, 0);

        for _ in 0..200 {
            let probs = router.forward();

            let mut rewards = vec![0.0f32; 4];
            let mut visibility = vec![0.0f32; 4 * N_BANKS];

            for h in 0..4 {
                // Reward: +1 for seeing bank 0, -1 for seeing bank 3
                // We want to reward bank 0 visibility and punish bank 3 visibility
                for b in 0..N_BANKS {
                    let p = probs[h * N_BANKS + b];
                    visibility[h * N_BANKS + b] = p;
                }
                // Only bank 0 and bank 3 contribute to reward (in opposite directions)
                rewards[h] = 1.0; // positive: reward bank 0

                // Apply selective gradient: positive for bank 0, negative for bank 3
                // We do two backward passes: one positive for bank 0, one negative for bank 3
            }

            // Positive reward with bank 0 visibility only
            let mut vis_core = vec![0.0f32; 4 * N_BANKS];
            let mut vis_bg = vec![0.0f32; 4 * N_BANKS];
            for h in 0..4 {
                vis_core[h * N_BANKS + 0] = probs[h * N_BANKS + 0];
                vis_bg[h * N_BANKS + 3] = probs[h * N_BANKS + 3];
            }

            router.backward(&vec![1.0; 4], &vis_core);
            router.backward(&vec![-1.0; 4], &vis_bg);
        }

        // After convergence: bank 0 should have high α, bank 3 should have low α
        for h in 0..4 {
            let p_core = router.prob(h, 0);
            let p_bg = router.prob(h, 3);

            assert!(
                p_core > 0.9,
                "head {h}: core visibility should converge to >0.9, got {p_core:.3}"
            );
            assert!(
                p_bg < 0.2,
                "head {h}: background visibility should converge to <0.2, got {p_bg:.3}"
            );
        }

        let entropy = router.specialization_entropy();
        assert!(
            entropy < 1.2,
            "after convergence, entropy should be low: {entropy:.3}"
        );

        eprintln!("  Convergence test: {} updates", router.n_updates);
        eprintln!("  Specialization entropy: {:.3}", entropy);
        eprintln!("{}", router.summary());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut router = HeadRouter::new(4, 4, 0.01, 0);
        router.alpha[0] = 3.14;
        router.alpha[5] = -2.71;
        router.n_updates = 42;

        let weights = router.weights().to_vec();

        let mut restored = HeadRouter::new(4, 4, 0.01, 0);
        restored.load_weights(&weights);
        restored.n_updates = 42;

        assert_eq!(restored.alpha, router.alpha);
        assert_eq!(restored.n_updates, router.n_updates);
    }

    #[test]
    fn test_specialization_entropy() {
        // All uniform (α=0 → sigmoid=0.5 for all) → high entropy
        let uniform = HeadRouter::new(4, 4, 0.01, 0);
        let entropy_uniform = uniform.specialization_entropy();

        // Specialized (one bank very high, others very low)
        let mut specialized = HeadRouter::new(4, 4, 0.01, 0);
        for h in 0..4 {
            specialized.alpha[h * 4 + 0] = 5.0; // core: high
            specialized.alpha[h * 4 + 1] = -5.0; // relations: low
            specialized.alpha[h * 4 + 2] = -5.0; // 2-hop: low
            specialized.alpha[h * 4 + 3] = -5.0; // background: low
        }
        let entropy_specialized = specialized.specialization_entropy();

        assert!(
            entropy_specialized < entropy_uniform,
            "specialized should have lower entropy: {entropy_specialized:.3} < {entropy_uniform:.3}"
        );
    }
}
