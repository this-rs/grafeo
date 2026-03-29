//! Ξ(t) T3 — Generation step signals: entropy, confidence metrics.
//!
//! Extracted from logits at each decode step for use by:
//! - Reward enrichment (T4): entropy as quality signal
//! - AFE: attention formula selection based on generation uncertainty
//! - IPTR: in-process tool triggering on high-entropy steps

/// Signals computed from logits at a single decode step.
#[derive(Debug, Clone)]
pub struct StepSignals {
    /// Shannon entropy H = -Σ p·log₂(p) over top-k softmax distribution.
    pub entropy: f32,
    /// Probability of the most likely token (confidence).
    pub top1_prob: f32,
    /// Cumulative probability mass in top-p (p=0.9) tokens.
    pub top_p_mass: f32,
    /// Position in the generation (0-indexed).
    pub token_position: u32,
}

/// Aggregated signals over a full generation.
#[derive(Debug, Clone)]
pub struct GenerationSignals {
    /// Per-step signals.
    pub steps: Vec<StepSignals>,
    /// Average entropy across all steps.
    pub avg_entropy: f32,
    /// Maximum entropy across all steps.
    pub max_entropy: f32,
    /// Number of steps with entropy > HIGH_ENTROPY_THRESHOLD.
    pub high_entropy_count: u32,
    /// Token ID of the first generated token (for ablation reward).
    pub first_token_id: Option<i32>,
    /// Full logits at step 0 (for ablation reward log-prob computation).
    /// Only populated when ablation is needed. Clone is expensive (~512KB for 128K vocab).
    pub first_step_logits: Option<Vec<f32>>,
}

/// Entropy above this threshold indicates high uncertainty.
pub const HIGH_ENTROPY_THRESHOLD: f32 = 3.0;

impl GenerationSignals {
    /// Build aggregated signals from per-step data.
    pub fn from_steps(steps: Vec<StepSignals>) -> Self {
        if steps.is_empty() {
            return Self {
                steps,
                avg_entropy: 0.0,
                max_entropy: 0.0,
                high_entropy_count: 0,
                first_token_id: None,
                first_step_logits: None,
            };
        }
        let sum: f32 = steps.iter().map(|s| s.entropy).sum();
        let max = steps.iter().map(|s| s.entropy).fold(f32::NEG_INFINITY, f32::max);
        let high = steps.iter().filter(|s| s.entropy > HIGH_ENTROPY_THRESHOLD).count() as u32;
        Self {
            avg_entropy: sum / steps.len() as f32,
            max_entropy: max,
            high_entropy_count: high,
            first_token_id: None,
            first_step_logits: None,
            steps,
        }
    }

    /// Empty signals (no generation happened).
    pub fn empty() -> Self {
        Self {
            steps: Vec::new(),
            avg_entropy: 0.0,
            max_entropy: 0.0,
            high_entropy_count: 0,
            first_token_id: None,
            first_step_logits: None,
        }
    }
}

/// Compute entropy metrics from raw logits using partial top-k sort + softmax.
///
/// This avoids softmaxing the full vocabulary (128K+). We take the top-k largest
/// logits, softmax over that subset, and compute H. The approximation error is
/// negligible since tokens outside top-256 have vanishingly small probability.
///
/// Returns (entropy, top1_prob, top_p_mass_at_0_9).
pub fn compute_entropy_top_k(logits: &[f32], k: usize) -> (f32, f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let k = k.min(logits.len());

    // Partial sort: find top-k logits via selection
    // For k=256 and vocab=128K, this is much faster than full sort
    let mut top_k: Vec<f32> = Vec::with_capacity(k);
    if logits.len() <= k {
        top_k.extend_from_slice(logits);
    } else {
        // Simple approach: use a min-heap of size k
        // For k=256 this is O(n·log(k)) ≈ O(n·8) — fast enough
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        let mut heap: BinaryHeap<Reverse<OrderedFloat>> = BinaryHeap::with_capacity(k + 1);
        for &val in logits {
            heap.push(Reverse(OrderedFloat(val)));
            if heap.len() > k {
                heap.pop();
            }
        }
        top_k = heap.into_iter().map(|Reverse(OrderedFloat(v))| v).collect();
    }

    // Softmax over top-k: subtract max for numerical stability
    let max_logit = top_k.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = top_k.iter().map(|&x| (x - max_logit).exp()).sum();
    let inv_sum = 1.0 / exp_sum;

    // Sort descending for top-p calculation
    top_k.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut entropy: f32 = 0.0;
    let mut cum_mass: f32 = 0.0;
    let mut top_p_mass: f32 = 1.0; // will be set when we cross 0.9
    let mut top_p_found = false;

    let top1_prob = (top_k[0] - max_logit).exp() * inv_sum;

    for &logit in &top_k {
        let p = (logit - max_logit).exp() * inv_sum;
        if p > 1e-10 {
            entropy -= p * p.log2();
        }
        cum_mass += p;
        if !top_p_found && cum_mass >= 0.9 {
            top_p_mass = cum_mass;
            top_p_found = true;
        }
    }
    if !top_p_found {
        top_p_mass = cum_mass;
    }

    (entropy, top1_prob, top_p_mass)
}

/// Wrapper for f32 that implements Ord (for BinaryHeap).
#[derive(Clone, Copy, PartialEq)]
struct OrderedFloat(f32);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        // Uniform logits → max entropy
        let logits: Vec<f32> = vec![0.0; 100];
        let (entropy, top1_prob, _top_p) = compute_entropy_top_k(&logits, 256);
        // H(uniform over 100) = log2(100) ≈ 6.64
        assert!((entropy - 6.644).abs() < 0.1, "entropy={entropy}");
        assert!((top1_prob - 0.01).abs() < 0.001, "top1={top1_prob}");
    }

    #[test]
    fn test_peaked_distribution() {
        // One very high logit → low entropy
        let mut logits: Vec<f32> = vec![0.0; 100];
        logits[0] = 20.0;
        let (entropy, top1_prob, _) = compute_entropy_top_k(&logits, 256);
        assert!(entropy < 0.5, "entropy should be low: {entropy}");
        assert!(top1_prob > 0.95, "top1 should be high: {top1_prob}");
    }

    #[test]
    fn test_empty_logits() {
        let (e, t, p) = compute_entropy_top_k(&[], 256);
        assert_eq!(e, 0.0);
        assert_eq!(t, 0.0);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_top_k_smaller_than_vocab() {
        // 1000 logits, top-256 → should still work
        let logits: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01).collect();
        let (entropy, top1_prob, top_p_mass) = compute_entropy_top_k(&logits, 256);
        assert!(entropy > 0.0);
        assert!(top1_prob > 0.0);
        assert!(top_p_mass >= 0.9);
    }

    #[test]
    fn test_generation_signals_from_steps() {
        let steps = vec![
            StepSignals { entropy: 2.0, top1_prob: 0.5, top_p_mass: 0.95, token_position: 0 },
            StepSignals { entropy: 4.0, top1_prob: 0.2, top_p_mass: 0.85, token_position: 1 },
            StepSignals { entropy: 1.0, top1_prob: 0.8, top_p_mass: 0.99, token_position: 2 },
        ];
        let gs = GenerationSignals::from_steps(steps);
        assert!((gs.avg_entropy - 2.333).abs() < 0.01);
        assert_eq!(gs.max_entropy, 4.0);
        assert_eq!(gs.high_entropy_count, 1); // only 4.0 > 3.0
    }
}
