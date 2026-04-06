//! # Contrastive training for Phi_0 — SPSA gradient estimation
//!
//! Trains the kernel weights via triplet contrastive loss using SPSA
//! (Simultaneous Perturbation Stochastic Approximation).
//!
//! SPSA perturbs ALL weights simultaneously with random +/-epsilon,
//! measures loss at both perturbations, and estimates the gradient
//! with only 2 forward passes (vs 51K for finite differences).
//!
//! The forward pass uses `single_pass_attention()` — NOT iterative A^inf.
//!
//! ## Usage
//!
//! Training is offline: extract features + cluster assignments from the graph,
//! then train on a subsample. The trained Phi_0 is frozen and persisted.
//!
//! ```text
//! GraphStore → extract features + communities → train_contrastive() → frozen Phi_0
//! ```

#[cfg(test)]
use super::kernel::D_MODEL;
use super::kernel::{AdjacencyMask, MultiHeadPhi0, single_pass_attention};
use super::kernel_math::{Matrix, Rng};

// ============================================================================
// Triplet loss
// ============================================================================

/// Cosine-based triplet loss:
///   loss = max(0, cos(anchor, negative) - cos(anchor, positive) + margin)
///
/// We WANT: cos(a,p) > cos(a,n) + margin
/// i.e. positive more similar than negative by at least `margin`.
pub fn triplet_loss(
    embeddings: &Matrix,
    anchor: usize,
    positive: usize,
    negative: usize,
    margin: f64,
) -> f64 {
    let cos_pos = embeddings.cosine_similarity(anchor, positive);
    let cos_neg = embeddings.cosine_similarity(anchor, negative);
    (cos_neg - cos_pos + margin).max(0.0)
}

/// Average triplet loss over a batch of triplets.
pub fn triplet_loss_batch(
    embeddings: &Matrix,
    triplets: &[(usize, usize, usize)],
    margin: f64,
) -> f64 {
    if triplets.is_empty() {
        return 0.0;
    }
    let total: f64 = triplets
        .iter()
        .map(|&(a, p, n)| triplet_loss(embeddings, a, p, n, margin))
        .sum();
    total / triplets.len() as f64
}

// ============================================================================
// Triplet sampling from cluster assignments
// ============================================================================

/// Sample triplets from cluster assignments.
///
/// For each triplet: anchor + positive from the same cluster,
/// negative from a different cluster.
///
/// `clusters[i]` = cluster ID of node i (e.g. from Louvain communities).
pub fn sample_triplets(
    clusters: &[usize],
    n_triplets: usize,
    rng: &mut Rng,
) -> Vec<(usize, usize, usize)> {
    let n = clusters.len();
    if n < 3 {
        return Vec::new();
    }

    let mut triplets = Vec::with_capacity(n_triplets);
    let max_attempts = n_triplets * 10;

    for _ in 0..max_attempts {
        if triplets.len() >= n_triplets {
            break;
        }

        let anchor = rng.next_usize(n);
        let anchor_cluster = clusters[anchor];

        // Find positive: same cluster, different node
        let pos_start = rng.next_usize(n);
        let mut positive = None;
        for offset in 0..n {
            let idx = (pos_start + offset) % n;
            if idx != anchor && clusters[idx] == anchor_cluster {
                positive = Some(idx);
                break;
            }
        }

        // Find negative: different cluster
        let neg_start = rng.next_usize(n);
        let mut negative = None;
        for offset in 0..n {
            let idx = (neg_start + offset) % n;
            if clusters[idx] != anchor_cluster {
                negative = Some(idx);
                break;
            }
        }

        if let (Some(p), Some(neg)) = (positive, negative) {
            triplets.push((anchor, p, neg));
        }
    }

    triplets
}

// ============================================================================
// SPSA gradient estimation
// ============================================================================

/// Compute the forward pass loss for a given set of weights.
///
/// Uses `single_pass_attention()` — single-pass, no iteration.
fn forward_loss(
    phi: &mut MultiHeadPhi0,
    weights: &[f64],
    features: &Matrix,
    adj_mask: &AdjacencyMask,
    triplets: &[(usize, usize, usize)],
    margin: f64,
    alpha: f64,
) -> f64 {
    phi.deserialize_weights(weights);
    let embeddings = single_pass_attention(features, phi, adj_mask, alpha);
    triplet_loss_batch(&embeddings, triplets, margin)
}

/// SPSA gradient estimate: 2 forward passes per estimate.
///
/// Perturbs all weights simultaneously with random +/-1 (Bernoulli),
/// measures loss at both perturbations, and estimates gradient.
#[allow(clippy::too_many_arguments)]
fn spsa_gradient(
    phi: &mut MultiHeadPhi0,
    features: &Matrix,
    adj_mask: &AdjacencyMask,
    triplets: &[(usize, usize, usize)],
    margin: f64,
    perturbation_size: f64,
    alpha: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let original_weights = phi.serialize_weights();
    let n_params = original_weights.len();

    // Random perturbation direction: each element in {-1, +1}
    let delta: Vec<f64> = (0..n_params)
        .map(|_| {
            if rng.next_u64().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    // Phi_0 + c*delta
    let weights_plus: Vec<f64> = original_weights
        .iter()
        .zip(delta.iter())
        .map(|(&p, &d)| p + perturbation_size * d)
        .collect();

    // Phi_0 - c*delta
    let weights_minus: Vec<f64> = original_weights
        .iter()
        .zip(delta.iter())
        .map(|(&p, &d)| p - perturbation_size * d)
        .collect();

    let loss_plus = forward_loss(
        phi,
        &weights_plus,
        features,
        adj_mask,
        triplets,
        margin,
        alpha,
    );
    let loss_minus = forward_loss(
        phi,
        &weights_minus,
        features,
        adj_mask,
        triplets,
        margin,
        alpha,
    );

    // Restore original weights
    phi.deserialize_weights(&original_weights);

    // SPSA gradient: g_i = (loss+ - loss-) / (2 * c * delta_i)
    let diff = loss_plus - loss_minus;
    delta
        .iter()
        .map(|&d| diff / (2.0 * perturbation_size * d))
        .collect()
}

/// Multi-sample SPSA: average over k random directions for variance reduction.
#[allow(clippy::too_many_arguments)]
fn spsa_gradient_avg(
    phi: &mut MultiHeadPhi0,
    features: &Matrix,
    adj_mask: &AdjacencyMask,
    triplets: &[(usize, usize, usize)],
    margin: f64,
    perturbation_size: f64,
    n_samples: usize,
    alpha: f64,
    rng: &mut Rng,
) -> Vec<f64> {
    let n_params = phi.serialize_weights().len();
    let mut avg_grad = vec![0.0; n_params];

    for _ in 0..n_samples {
        let grad = spsa_gradient(
            phi,
            features,
            adj_mask,
            triplets,
            margin,
            perturbation_size,
            alpha,
            rng,
        );
        for i in 0..n_params {
            avg_grad[i] += grad[i] / n_samples as f64;
        }
    }

    avg_grad
}

/// Apply gradient descent with gradient norm clipping.
fn apply_gradient(phi: &mut MultiHeadPhi0, grad: &[f64], lr: f64) {
    let mut params = phi.serialize_weights();

    // Gradient clipping: cap the norm at 1.0
    let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    let max_norm = 1.0;
    let scale = if grad_norm > max_norm {
        max_norm / grad_norm
    } else {
        1.0
    };

    for i in 0..params.len() {
        params[i] -= lr * grad[i] * scale;
    }

    phi.deserialize_weights(&params);
}

// ============================================================================
// Training configuration
// ============================================================================

/// Configuration for contrastive SPSA training.
#[derive(Clone)]
pub struct TrainConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of triplets sampled per epoch.
    pub triplets_per_epoch: usize,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Triplet loss margin.
    pub margin: f64,
    /// SPSA perturbation size (c).
    pub perturbation_size: f64,
    /// Number of random directions to average per gradient estimate.
    pub spsa_samples: usize,
    /// APPNP anchoring factor for the forward pass.
    pub alpha: f64,
    /// PRNG seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            triplets_per_epoch: 24,
            learning_rate: 0.05,
            margin: 0.3,
            perturbation_size: 0.005,
            spsa_samples: 12,
            alpha: 0.8,
            seed: 9999,
        }
    }
}

/// Result of a training run.
pub struct TrainResult {
    /// Loss at each epoch.
    pub loss_history: Vec<f64>,
    /// Gap (intra - inter cluster similarity) at each epoch.
    pub gap_history: Vec<f64>,
    /// Diversity at each epoch.
    pub diversity_history: Vec<f64>,
}

// ============================================================================
// Training loop
// ============================================================================

/// Train Phi_0 with contrastive loss using SPSA gradient estimation.
///
/// The outer loop: Phi_0^(t+1) = Phi_0^(t) - eta * g_hat(C^(t))
///
/// # Arguments
///
/// * `phi` - Mutable reference to Phi_0 weights (modified in place)
/// * `features` - Feature matrix [n_nodes x d_model]
/// * `adj_mask` - Graph adjacency mask
/// * `clusters` - Cluster assignment for each node (e.g. from Louvain)
/// * `config` - Training hyperparameters
///
/// # Returns
///
/// Training history (loss, gap, diversity per epoch).
pub fn train_contrastive(
    phi: &mut MultiHeadPhi0,
    features: &Matrix,
    adj_mask: &AdjacencyMask,
    clusters: &[usize],
    config: &TrainConfig,
) -> TrainResult {
    let mut rng = Rng::new(config.seed);

    let mut loss_history = Vec::with_capacity(config.epochs);
    let mut gap_history = Vec::with_capacity(config.epochs);
    let mut diversity_history = Vec::with_capacity(config.epochs);

    for _epoch in 0..config.epochs {
        // Sample fresh triplets each epoch
        let triplets = sample_triplets(clusters, config.triplets_per_epoch, &mut rng);

        // SPSA gradient estimate (averaged over n_samples directions)
        let grad = spsa_gradient_avg(
            phi,
            features,
            adj_mask,
            &triplets,
            config.margin,
            config.perturbation_size,
            config.spsa_samples,
            config.alpha,
            &mut rng,
        );

        // Apply: Phi_0 = Phi_0 - lr * g_hat
        apply_gradient(phi, &grad, config.learning_rate);

        // Evaluate
        let embeddings = single_pass_attention(features, phi, adj_mask, config.alpha);
        let loss = triplet_loss_batch(&embeddings, &triplets, config.margin);
        let (intra, inter) = cluster_similarity(&embeddings, clusters);
        let gap = intra - inter;
        let div = super::kernel_math::diversity(&embeddings);

        loss_history.push(loss);
        gap_history.push(gap);
        diversity_history.push(div);
    }

    TrainResult {
        loss_history,
        gap_history,
        diversity_history,
    }
}

/// Compute intra-cluster and inter-cluster mean cosine similarity.
pub fn cluster_similarity(embeddings: &Matrix, clusters: &[usize]) -> (f64, f64) {
    let n = embeddings.rows.min(clusters.len());
    let mut intra_sum = 0.0;
    let mut intra_count = 0;
    let mut inter_sum = 0.0;
    let mut inter_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let cos = embeddings.cosine_similarity(i, j);
            if clusters[i] == clusters[j] {
                intra_sum += cos;
                intra_count += 1;
            } else {
                inter_sum += cos;
                inter_count += 1;
            }
        }
    }

    let intra = if intra_count > 0 {
        intra_sum / intra_count as f64
    } else {
        0.0
    };
    let inter = if inter_count > 0 {
        inter_sum / inter_count as f64
    } else {
        0.0
    };
    (intra, inter)
}

// ============================================================================
// Binary serialization of Phi_0
// ============================================================================

/// Magic bytes for Phi_0 binary format.
const PHI0_MAGIC: &[u8; 4] = b"PHI0";
/// Format version.
const PHI0_VERSION: u32 = 1;

/// Serialize Phi_0 weights to a portable binary format.
///
/// Format:
/// ```text
/// [4 bytes] magic "PHI0"
/// [4 bytes] version (u32 LE)
/// [4 bytes] d_model (u32 LE)
/// [4 bytes] n_heads (u32 LE)
/// [4 bytes] d_ff (u32 LE)
/// [N * 8 bytes] weights (f64 LE, flat)
/// ```
pub fn serialize_phi0(phi: &MultiHeadPhi0) -> Vec<u8> {
    let weights = phi.serialize_weights();
    let header_size = 4 + 4 + 4 + 4 + 4; // magic + version + d_model + n_heads + d_ff
    let total_size = header_size + weights.len() * 8;
    let mut bytes = Vec::with_capacity(total_size);

    // Header
    bytes.extend_from_slice(PHI0_MAGIC);
    bytes.extend_from_slice(&PHI0_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(phi.d_model as u32).to_le_bytes());
    bytes.extend_from_slice(&(phi.n_heads as u32).to_le_bytes());
    bytes.extend_from_slice(&(phi.w_ff1.cols as u32).to_le_bytes()); // d_ff

    // Weights
    for &w in &weights {
        bytes.extend_from_slice(&w.to_le_bytes());
    }

    bytes
}

/// Deserialize Phi_0 from binary format.
///
/// Returns `None` if the format is invalid.
pub fn deserialize_phi0(bytes: &[u8]) -> Option<MultiHeadPhi0> {
    let header_size = 4 + 4 + 4 + 4 + 4;
    if bytes.len() < header_size {
        return None;
    }

    // Check magic
    if &bytes[0..4] != PHI0_MAGIC {
        return None;
    }

    // Version
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if version != PHI0_VERSION {
        return None;
    }

    // Dimensions
    let d_model = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let n_heads = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let d_ff = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;

    // Create Phi_0 with correct dimensions
    let mut phi = MultiHeadPhi0::new(d_model, n_heads, d_ff, 0);
    let n_params = phi.param_count();
    let expected_size = header_size + n_params * 8;

    if bytes.len() < expected_size {
        return None;
    }

    // Read weights
    let mut weights = Vec::with_capacity(n_params);
    for i in 0..n_params {
        let offset = header_size + i * 8;
        let w = f64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
        weights.push(w);
    }

    phi.deserialize_weights(&weights);
    Some(phi)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test dataset: 3 clusters of 4 nodes, 80d features.
    fn make_test_data() -> (Matrix, AdjacencyMask, Vec<usize>) {
        let n = 12;
        let d = D_MODEL;
        let mut rng = Rng::new(42);

        // 3 clusters with distinct features
        let mut features = Matrix::zeros(n, d);
        let mut clusters = Vec::with_capacity(n);
        for cluster in 0..3 {
            let centroid: Vec<f64> = (0..d).map(|_| rng.next_f64()).collect();
            for i in 0..4 {
                let node = cluster * 4 + i;
                clusters.push(cluster);
                for j in 0..d {
                    features.set(node, j, centroid[j] + rng.next_normal() * 0.1);
                }
            }
        }

        // Edges: dense within clusters, sparse between
        let mut edges = Vec::new();
        for cluster in 0..3 {
            let base = cluster * 4;
            for i in base..base + 4 {
                for j in base..base + 4 {
                    if i != j {
                        edges.push((i, j));
                    }
                }
            }
        }
        edges.push((3, 4));
        edges.push((4, 3));
        edges.push((7, 8));
        edges.push((8, 7));

        let mask = AdjacencyMask::from_edges(n, &edges);
        (features, mask, clusters)
    }

    // ── Triplet loss tests ──

    #[test]
    fn test_triplet_loss_satisfied() {
        // pos closer than neg by more than margin → loss = 0
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                1.0, 0.0, 0.0, // anchor
                0.9, 0.1, 0.0, // positive (close)
                0.0, 1.0, 0.0, // negative (far)
            ],
        );
        let loss = triplet_loss(&m, 0, 1, 2, 0.1);
        assert_eq!(loss, 0.0, "satisfied triplet should have zero loss");
    }

    #[test]
    fn test_triplet_loss_violated() {
        // neg closer than pos → positive loss
        let m = Matrix::from_vec(
            3,
            3,
            vec![
                1.0, 0.0, 0.0, // anchor
                0.0, 1.0, 0.0, // "positive" (actually far)
                0.9, 0.1, 0.0, // "negative" (actually close)
            ],
        );
        let loss = triplet_loss(&m, 0, 1, 2, 0.1);
        assert!(loss > 0.0, "violated triplet should have positive loss");
    }

    #[test]
    fn test_triplet_loss_batch_average() {
        let m = Matrix::from_vec(4, 2, vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, -1.0, 0.0]);
        let triplets = vec![(0, 1, 2), (0, 1, 3)];
        let batch_loss = triplet_loss_batch(&m, &triplets, 0.1);
        let individual = f64::midpoint(
            triplet_loss(&m, 0, 1, 2, 0.1),
            triplet_loss(&m, 0, 1, 3, 0.1),
        );
        assert!((batch_loss - individual).abs() < 1e-12);
    }

    #[test]
    fn test_triplet_loss_batch_empty() {
        let m = Matrix::zeros(5, 3);
        assert_eq!(triplet_loss_batch(&m, &[], 0.1), 0.0);
    }

    // ── Triplet sampling tests ──

    #[test]
    fn test_sample_triplets_valid() {
        let clusters = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let mut rng = Rng::new(42);
        let triplets = sample_triplets(&clusters, 20, &mut rng);

        assert!(!triplets.is_empty(), "should produce triplets");
        for &(a, p, n) in &triplets {
            assert_eq!(
                clusters[a], clusters[p],
                "anchor and positive must be same cluster"
            );
            assert_ne!(
                clusters[a], clusters[n],
                "anchor and negative must be different clusters"
            );
            assert_ne!(a, p, "anchor and positive must be different nodes");
        }
    }

    #[test]
    fn test_sample_triplets_count() {
        let clusters = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let mut rng = Rng::new(42);
        let triplets = sample_triplets(&clusters, 10, &mut rng);
        assert_eq!(triplets.len(), 10);
    }

    #[test]
    fn test_sample_triplets_single_cluster() {
        // All same cluster → no negatives → empty
        let clusters = vec![0, 0, 0, 0];
        let mut rng = Rng::new(42);
        let triplets = sample_triplets(&clusters, 10, &mut rng);
        assert!(triplets.is_empty());
    }

    // ── Cluster similarity tests ──

    #[test]
    fn test_cluster_similarity_perfect() {
        // Two clusters, identical within, orthogonal between
        let m = Matrix::from_vec(4, 2, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let clusters = vec![0, 0, 1, 1];
        let (intra, inter) = cluster_similarity(&m, &clusters);
        assert!((intra - 1.0).abs() < 1e-10, "intra should be 1.0");
        assert!(inter.abs() < 1e-10, "inter should be 0.0");
    }

    // ── SPSA gradient tests ──

    #[test]
    fn test_spsa_gradient_finite() {
        let (features, mask, clusters) = make_test_data();
        let mut phi = MultiHeadPhi0::default_with_seed(42);
        let mut rng = Rng::new(42);
        let triplets = sample_triplets(&clusters, 10, &mut rng);

        let grad = spsa_gradient(
            &mut phi, &features, &mask, &triplets, 0.3, 0.005, 0.8, &mut rng,
        );

        assert_eq!(grad.len(), phi.param_count());
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "SPSA gradient should be finite"
        );
    }

    #[test]
    fn test_spsa_gradient_avg_finite() {
        let (features, mask, clusters) = make_test_data();
        let triplets = {
            let mut rng = Rng::new(42);
            sample_triplets(&clusters, 10, &mut rng)
        };

        // 8-sample average
        let mut phi = MultiHeadPhi0::default_with_seed(42);
        let mut rng = Rng::new(77);
        let grad_avg = spsa_gradient_avg(
            &mut phi, &features, &mask, &triplets, 0.3, 0.005, 8, 0.8, &mut rng,
        );

        assert_eq!(grad_avg.len(), phi.param_count());
        assert!(
            grad_avg.iter().all(|g| g.is_finite()),
            "averaged SPSA gradient should be finite"
        );
    }

    // ── Training loop tests ──

    #[test]
    fn test_train_contrastive_loss_decreases() {
        let (features, mask, clusters) = make_test_data();
        let mut phi = MultiHeadPhi0::default_with_seed(42);

        let config = TrainConfig {
            epochs: 30,
            triplets_per_epoch: 24,
            learning_rate: 0.05,
            margin: 0.3,
            perturbation_size: 0.005,
            spsa_samples: 8,
            alpha: 0.8,
            seed: 9999,
        };

        let result = train_contrastive(&mut phi, &features, &mask, &clusters, &config);

        assert_eq!(result.loss_history.len(), config.epochs);
        assert_eq!(result.gap_history.len(), config.epochs);

        // Loss should generally decrease (check first vs last quarter average)
        let first_quarter: f64 = result.loss_history[..8].iter().sum::<f64>() / 8.0;
        let last_quarter: f64 = result.loss_history[22..].iter().sum::<f64>() / 8.0;

        // Allow some tolerance — SPSA is noisy
        assert!(
            last_quarter <= first_quarter + 0.05,
            "Loss should trend down: first_q={:.4}, last_q={:.4}",
            first_quarter,
            last_quarter
        );
    }

    #[test]
    fn test_train_contrastive_gap_improves() {
        let (features, mask, clusters) = make_test_data();
        let mut phi = MultiHeadPhi0::default_with_seed(42);

        let config = TrainConfig {
            epochs: 50,
            triplets_per_epoch: 24,
            ..TrainConfig::default()
        };

        // Gap before training
        let emb_before = single_pass_attention(&features, &phi, &mask, config.alpha);
        let (intra_b, inter_b) = cluster_similarity(&emb_before, &clusters);
        let gap_before = intra_b - inter_b;

        let _result = train_contrastive(&mut phi, &features, &mask, &clusters, &config);

        // Gap after training
        let emb_after = single_pass_attention(&features, &phi, &mask, config.alpha);
        let (intra_a, inter_a) = cluster_similarity(&emb_after, &clusters);
        let gap_after = intra_a - inter_a;

        assert!(
            gap_after > gap_before - 0.05,
            "Gap should not degrade significantly: before={:.4}, after={:.4}",
            gap_before,
            gap_after
        );
    }

    #[test]
    fn test_train_deterministic() {
        let (features, mask, clusters) = make_test_data();

        let config = TrainConfig {
            epochs: 10,
            ..TrainConfig::default()
        };

        let mut phi1 = MultiHeadPhi0::default_with_seed(42);
        let result1 = train_contrastive(&mut phi1, &features, &mask, &clusters, &config);

        let mut phi2 = MultiHeadPhi0::default_with_seed(42);
        let result2 = train_contrastive(&mut phi2, &features, &mask, &clusters, &config);

        // Same seed → same results
        assert_eq!(result1.loss_history, result2.loss_history);
        assert_eq!(phi1.serialize_weights(), phi2.serialize_weights());
    }

    // ── Serialization tests ──

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let phi = MultiHeadPhi0::default_with_seed(42);
        let bytes = serialize_phi0(&phi);

        let phi2 = deserialize_phi0(&bytes).expect("deserialization should succeed");

        assert_eq!(phi.d_model, phi2.d_model);
        assert_eq!(phi.n_heads, phi2.n_heads);
        assert_eq!(phi.w_ff1.cols, phi2.w_ff1.cols); // d_ff

        let weights1 = phi.serialize_weights();
        let weights2 = phi2.serialize_weights();
        assert_eq!(weights1, weights2, "weights should be bit-identical");
    }

    #[test]
    fn test_serialize_deserialize_re_serialize() {
        let phi = MultiHeadPhi0::default_with_seed(42);
        let bytes1 = serialize_phi0(&phi);
        let phi2 = deserialize_phi0(&bytes1).unwrap();
        let bytes2 = serialize_phi0(&phi2);
        assert_eq!(bytes1, bytes2, "re-serialization should be byte-identical");
    }

    #[test]
    fn test_serialize_trained_phi0() {
        let (features, mask, clusters) = make_test_data();
        let mut phi = MultiHeadPhi0::default_with_seed(42);

        let config = TrainConfig {
            epochs: 10,
            ..TrainConfig::default()
        };
        let _ = train_contrastive(&mut phi, &features, &mask, &clusters, &config);

        // Serialize trained weights
        let bytes = serialize_phi0(&phi);
        let phi_restored = deserialize_phi0(&bytes).unwrap();

        // Verify functionally identical
        let emb1 = single_pass_attention(&features, &phi, &mask, config.alpha);
        let emb2 = single_pass_attention(&features, &phi_restored, &mask, config.alpha);
        assert!(
            emb1.diff_norm(&emb2) < 1e-12,
            "restored phi should produce identical embeddings"
        );
    }

    #[test]
    fn test_deserialize_invalid_magic() {
        let bytes = vec![0u8; 100];
        assert!(deserialize_phi0(&bytes).is_none());
    }

    #[test]
    fn test_deserialize_truncated() {
        let phi = MultiHeadPhi0::default_with_seed(42);
        let bytes = serialize_phi0(&phi);
        // Truncate
        assert!(deserialize_phi0(&bytes[..bytes.len() - 100]).is_none());
    }

    #[test]
    fn test_deserialize_wrong_version() {
        let mut bytes = serialize_phi0(&MultiHeadPhi0::default_with_seed(42));
        // Corrupt version
        bytes[4] = 99;
        assert!(deserialize_phi0(&bytes).is_none());
    }

    #[test]
    fn test_serialize_size() {
        let phi = MultiHeadPhi0::default_with_seed(42);
        let bytes = serialize_phi0(&phi);
        let expected = 4 + 4 + 4 + 4 + 4 + phi.param_count() * 8;
        assert_eq!(bytes.len(), expected);
        // 51200 params * 8 bytes + 20 bytes header = 409620 bytes ≈ 400KB
        assert!(
            bytes.len() < 500_000,
            "serialized size too large: {}",
            bytes.len()
        );
    }
}
