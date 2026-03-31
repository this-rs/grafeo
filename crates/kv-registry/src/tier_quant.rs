//! Pre-quantization of embeddings per KV tier before injection via batch.embd.
//!
//! Simulates quantize→dequantize in pure Rust so the model sees a "noisy" embedding
//! matching the target quantization level. The KV cache still stores f16, but the
//! values are already rounded to the target precision — zero C++ changes needed.
//!
//! Tier assignments:
//! - **Alpha** (focus): highest precision (Q8_0 K, Q8_0 V)
//! - **Beta** (warm): medium precision (Q5_0 K, Q5_0 V)
//! - **Gamma** (background): lowest precision (Q4_0 K, Q4_0 V)
//!
//! Asymmetric mode: Keys can use lower precision than Values because the topology
//! mask already constrains which tokens to attend to (reducing K's role).

use super::registry::KvTier;

/// Quantization type to simulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    F16,  // no quantization noise
    Q8_0, // 8-bit symmetric: block_size=32, 256 levels
    Q5_0, // 5-bit symmetric: block_size=32, 32 levels
    Q4_0, // 4-bit symmetric: block_size=32, 16 levels
    Q3_0, // 3-bit symmetric: block_size=32, 8 levels
}

/// Whether we're quantizing a Key or Value embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOrValue {
    Key,
    Value,
}

/// Per-tier quantization configuration.
#[derive(Debug, Clone)]
pub struct TierQuantConfig {
    pub alpha_k: QuantType,
    pub alpha_v: QuantType,
    pub beta_k: QuantType,
    pub beta_v: QuantType,
    pub gamma_k: QuantType,
    pub gamma_v: QuantType,
}

impl Default for TierQuantConfig {
    /// Default: symmetric, moderate compression per tier.
    fn default() -> Self {
        Self {
            alpha_k: QuantType::Q8_0,
            alpha_v: QuantType::Q8_0,
            beta_k: QuantType::Q5_0,
            beta_v: QuantType::Q5_0,
            gamma_k: QuantType::Q4_0,
            gamma_v: QuantType::Q4_0,
        }
    }
}

impl TierQuantConfig {
    /// Asymmetric config: Keys get lower precision (topology mask compensates),
    /// Values keep higher precision (carry semantic content).
    pub fn asymmetric() -> Self {
        Self {
            alpha_k: QuantType::Q5_0,
            alpha_v: QuantType::Q8_0,
            beta_k: QuantType::Q4_0,
            beta_v: QuantType::Q5_0,
            gamma_k: QuantType::Q3_0,
            gamma_v: QuantType::Q4_0,
        }
    }

    /// Aggressive: lower precision across the board.
    pub fn aggressive() -> Self {
        Self {
            alpha_k: QuantType::Q5_0,
            alpha_v: QuantType::Q5_0,
            beta_k: QuantType::Q4_0,
            beta_v: QuantType::Q4_0,
            gamma_k: QuantType::Q3_0,
            gamma_v: QuantType::Q3_0,
        }
    }

    /// No quantization (passthrough).
    pub fn none() -> Self {
        Self {
            alpha_k: QuantType::F16,
            alpha_v: QuantType::F16,
            beta_k: QuantType::F16,
            beta_v: QuantType::F16,
            gamma_k: QuantType::F16,
            gamma_v: QuantType::F16,
        }
    }

    /// Lookup the QuantType for a (tier, role) pair.
    pub fn get(&self, tier: KvTier, role: KeyOrValue) -> QuantType {
        match (tier, role) {
            (KvTier::Alpha, KeyOrValue::Key) => self.alpha_k,
            (KvTier::Alpha, KeyOrValue::Value) => self.alpha_v,
            (KvTier::Beta, KeyOrValue::Key) => self.beta_k,
            (KvTier::Beta, KeyOrValue::Value) => self.beta_v,
            (KvTier::Gamma, KeyOrValue::Key) => self.gamma_k,
            (KvTier::Gamma, KeyOrValue::Value) => self.gamma_v,
        }
    }
}

/// Number of quantization levels for a QuantType.
fn n_levels(qt: QuantType) -> u32 {
    match qt {
        QuantType::F16 => 0,  // no quantization
        QuantType::Q8_0 => 256,
        QuantType::Q5_0 => 32,
        QuantType::Q4_0 => 16,
        QuantType::Q3_0 => 8,
    }
}

const BLOCK_SIZE: usize = 32;

/// Simulate quantize→dequantize for a single block of values.
///
/// ggml symmetric quantization: find max |x|, scale to [-levels/2, levels/2-1],
/// round, then dequantize back.
fn simulate_block(block: &mut [f32], levels: u32) {
    if levels == 0 || block.is_empty() {
        return; // F16 passthrough
    }

    let half = (levels / 2) as f32;

    // Find max absolute value in block (ggml: d = max / half)
    let max_abs = block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_abs == 0.0 {
        return; // all zeros, nothing to quantize
    }

    let d = max_abs / half; // scale factor
    let id = 1.0 / d; // inverse scale for quantization

    for x in block.iter_mut() {
        // Quantize: round to nearest integer in [-half, half-1]
        let q = (*x * id).round().clamp(-half, half - 1.0);
        // Dequantize: scale back
        *x = q * d;
    }
}

/// Simulate quantize→dequantize on a full embedding vector.
///
/// Processes in blocks of 32 (matching ggml block_q*_0 layout).
/// Returns a new vector with quantization noise applied.
pub fn simulate_quantize(embedding: &[f32], qtype: QuantType) -> Vec<f32> {
    if qtype == QuantType::F16 {
        return embedding.to_vec(); // passthrough
    }

    let levels = n_levels(qtype);
    let mut result = embedding.to_vec();

    for chunk in result.chunks_mut(BLOCK_SIZE) {
        simulate_block(chunk, levels);
    }

    result
}

/// Apply tier-based quantization to an embedding.
///
/// Looks up the appropriate QuantType for the (tier, role) pair
/// and applies simulate_quantize.
pub fn tier_quantize(
    embedding: &[f32],
    tier: KvTier,
    role: KeyOrValue,
    config: &TierQuantConfig,
) -> Vec<f32> {
    let qtype = config.get(tier, role);
    simulate_quantize(embedding, qtype)
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
        // Simple LCG PRNG for deterministic tests
        let mut state = seed;
        (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Map to [-1, 1] range with some outliers
                let u = (state >> 33) as f32 / (1u64 << 31) as f32;
                (u - 0.5) * 2.0 * if (state >> 60) > 12 { 3.0 } else { 1.0 }
            })
            .collect()
    }

    #[test]
    fn test_f16_passthrough() {
        let embd = vec![1.0, 2.0, 3.0, -1.5];
        let result = simulate_quantize(&embd, QuantType::F16);
        assert_eq!(embd, result);
    }

    #[test]
    fn test_q8_0_high_fidelity() {
        let embd = random_embedding(128, 42);
        let result = simulate_quantize(&embd, QuantType::Q8_0);
        let cos = cosine_similarity(&embd, &result);
        assert!(
            cos > 0.999,
            "Q8_0 cosine should be > 0.999, got {cos:.6}"
        );
    }

    #[test]
    fn test_q5_0_good_fidelity() {
        let embd = random_embedding(128, 42);
        let result = simulate_quantize(&embd, QuantType::Q5_0);
        let cos = cosine_similarity(&embd, &result);
        assert!(cos > 0.99, "Q5_0 cosine should be > 0.99, got {cos:.6}");
    }

    #[test]
    fn test_q4_0_acceptable_fidelity() {
        let embd = random_embedding(128, 42);
        let result = simulate_quantize(&embd, QuantType::Q4_0);
        let cos = cosine_similarity(&embd, &result);
        assert!(cos > 0.97, "Q4_0 cosine should be > 0.97, got {cos:.6}");
    }

    #[test]
    fn test_q3_0_lower_fidelity() {
        let embd = random_embedding(128, 42);
        let result = simulate_quantize(&embd, QuantType::Q3_0);
        let cos = cosine_similarity(&embd, &result);
        assert!(cos > 0.90, "Q3_0 cosine should be > 0.90, got {cos:.6}");
    }

    #[test]
    fn test_fidelity_ordering() {
        let embd = random_embedding(3072, 123); // LLM-size embedding
        let c8 = cosine_similarity(&embd, &simulate_quantize(&embd, QuantType::Q8_0));
        let c5 = cosine_similarity(&embd, &simulate_quantize(&embd, QuantType::Q5_0));
        let c4 = cosine_similarity(&embd, &simulate_quantize(&embd, QuantType::Q4_0));
        let c3 = cosine_similarity(&embd, &simulate_quantize(&embd, QuantType::Q3_0));
        assert!(
            c8 > c5 && c5 > c4 && c4 > c3,
            "Fidelity ordering broken: Q8={c8:.6} Q5={c5:.6} Q4={c4:.6} Q3={c3:.6}"
        );
    }

    #[test]
    fn test_tier_quantize_default() {
        let embd = random_embedding(128, 99);
        let config = TierQuantConfig::default();

        // Alpha gets Q8_0 (highest)
        let alpha_k = tier_quantize(&embd, KvTier::Alpha, KeyOrValue::Key, &config);
        let cos_alpha = cosine_similarity(&embd, &alpha_k);

        // Gamma gets Q4_0 (lowest)
        let gamma_k = tier_quantize(&embd, KvTier::Gamma, KeyOrValue::Key, &config);
        let cos_gamma = cosine_similarity(&embd, &gamma_k);

        assert!(
            cos_alpha > cos_gamma,
            "Alpha should have higher fidelity than Gamma: alpha={cos_alpha:.6} gamma={cos_gamma:.6}"
        );
    }

    #[test]
    fn test_asymmetric_kv() {
        let embd = random_embedding(128, 77);
        let config = TierQuantConfig::asymmetric();

        // Alpha: K=Q5_0, V=Q8_0 → V should be higher fidelity
        let k = tier_quantize(&embd, KvTier::Alpha, KeyOrValue::Key, &config);
        let v = tier_quantize(&embd, KvTier::Alpha, KeyOrValue::Value, &config);
        let cos_k = cosine_similarity(&embd, &k);
        let cos_v = cosine_similarity(&embd, &v);
        assert!(
            cos_v > cos_k,
            "Asymmetric: Value should have higher fidelity than Key: V={cos_v:.6} K={cos_k:.6}"
        );
    }

    #[test]
    fn test_zeros_no_panic() {
        let embd = vec![0.0; 64];
        let result = simulate_quantize(&embd, QuantType::Q4_0);
        assert_eq!(result, embd);
    }

    #[test]
    fn test_single_element() {
        let embd = vec![1.5];
        let result = simulate_quantize(&embd, QuantType::Q4_0);
        assert!(!result[0].is_nan());
        // Single element: max_abs=1.5, d=1.5/8=0.1875
        // q = round(1.5 / 0.1875) = round(8) = 8, clamped to 7
        // dequant = 7 * 0.1875 = 1.3125
        assert!((result[0] - 1.3125).abs() < 1e-6, "got {}", result[0]);
    }

    #[test]
    fn test_dimension_preserved() {
        let embd = random_embedding(3072, 1);
        for qt in [
            QuantType::Q8_0,
            QuantType::Q5_0,
            QuantType::Q4_0,
            QuantType::Q3_0,
        ] {
            let result = simulate_quantize(&embd, qt);
            assert_eq!(result.len(), 3072, "Dimension must be preserved for {:?}", qt);
        }
    }

    #[test]
    fn test_near_idempotent() {
        let embd = random_embedding(128, 55);
        let once = simulate_quantize(&embd, QuantType::Q4_0);
        let twice = simulate_quantize(&once, QuantType::Q4_0);
        // Not strictly idempotent (clamping changes max_abs on 2nd pass),
        // but cosine between once and twice should be ~1.0
        let cos = cosine_similarity(&once, &twice);
        assert!(
            cos > 0.995,
            "Double quantization should be near-idempotent, got cosine={cos:.6}"
        );
    }

    #[test]
    fn test_config_get() {
        let config = TierQuantConfig::default();
        assert_eq!(config.get(KvTier::Alpha, KeyOrValue::Key), QuantType::Q8_0);
        assert_eq!(config.get(KvTier::Gamma, KeyOrValue::Value), QuantType::Q4_0);

        let asym = TierQuantConfig::asymmetric();
        assert_eq!(asym.get(KvTier::Alpha, KeyOrValue::Key), QuantType::Q5_0);
        assert_eq!(asym.get(KvTier::Alpha, KeyOrValue::Value), QuantType::Q8_0);
    }
}
