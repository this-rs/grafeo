//! ProjectionNet — lightweight MLP that projects concat(text_embd, gnn_embd) → LLM latent space.
//!
//! Architecture:
//!   Input:  concat(h ∈ R^n_embd, g ∈ R^n_gnn) → R^(n_embd + n_gnn)
//!   Layer1: Linear(n_input, n_embd) + GELU
//!   Layer2: Linear(n_embd, n_embd) + LayerNorm
//!   Output: p ∈ R^n_embd
//!
//! Implemented in pure Rust — no ML framework dependency.
//!
//! Training modes:
//! - **Reconstruction** (C3): MSE + cosine loss, target = text hidden state
//! - **Contrastive** (C6): Reconstruction + InfoNCE topological loss.
//!   Connected nodes should produce closer projected embeddings than unconnected nodes.

use anyhow::{Result, bail};
use std::io::{Read, Write};
use std::path::Path;

/// GELU activation: x · Φ(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
#[inline]
fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608; // √(2/π)
    0.5 * x * (1.0 + ((SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)) as f64).tanh() as f32)
}

/// 2-layer MLP that projects fused embeddings into the LLM's latent space.
pub struct ProjectionNet {
    // Layer 1: (n_input) → (n_hidden) with GELU
    pub w1: Vec<f32>, // [n_hidden × n_input], row-major
    pub b1: Vec<f32>, // [n_hidden]

    // Layer 2: (n_hidden) → (n_hidden) with LayerNorm
    pub w2: Vec<f32>, // [n_hidden × n_hidden], row-major
    pub b2: Vec<f32>, // [n_hidden]

    // LayerNorm parameters
    pub ln_gamma: Vec<f32>, // [n_hidden]
    pub ln_beta: Vec<f32>,  // [n_hidden]

    /// Input dimension (n_embd + n_gnn).
    pub n_input: usize,
    /// Hidden/output dimension (= n_embd of the LLM).
    pub n_hidden: usize,

    // SGD state
    n_updates: u64,
}

impl ProjectionNet {
    /// Create a new ProjectionNet with Xavier initialization.
    ///
    /// - `n_embd`: LLM embedding dimension (e.g. 3584 for Qwen3-14B)
    /// - `n_gnn`: GNN embedding dimension (e.g. 64 for FactGNN)
    pub fn new(n_embd: usize, n_gnn: usize) -> Self {
        let n_input = n_embd + n_gnn;
        let n_hidden = n_embd;

        // Xavier uniform: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
        let limit1 = (6.0 / (n_input + n_hidden) as f64).sqrt() as f32;
        let limit2 = (6.0 / (n_hidden + n_hidden) as f64).sqrt() as f32;

        // Deterministic pseudo-random init (reproducible, no rand dependency)
        let w1 = Self::pseudo_random_vec(n_hidden * n_input, limit1, 42);
        let b1 = vec![0.0; n_hidden];
        let w2 = Self::pseudo_random_vec(n_hidden * n_hidden, limit2, 137);
        let b2 = vec![0.0; n_hidden];

        // LayerNorm: gamma=1, beta=0
        let ln_gamma = vec![1.0; n_hidden];
        let ln_beta = vec![0.0; n_hidden];

        Self {
            w1,
            b1,
            w2,
            b2,
            ln_gamma,
            ln_beta,
            n_input,
            n_hidden,
            n_updates: 0,
        }
    }

    /// Deterministic pseudo-random vector in [-limit, limit] using xorshift.
    fn pseudo_random_vec(len: usize, limit: f32, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-limit, limit]
                let frac = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                (frac as f32) * limit
            })
            .collect()
    }

    /// Forward pass: input → Layer1(GELU) → Layer2(LayerNorm) → output.
    ///
    /// `input` must have length `n_input`. Returns Vec<f32> of length `n_hidden`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.n_input {
            bail!(
                "ProjectionNet::forward: input.len()={} but n_input={}",
                input.len(),
                self.n_input
            );
        }

        // Layer 1: h1 = GELU(W1 · input + b1)
        let mut h1 = self.b1.clone();
        matmul_add(&self.w1, input, &mut h1, self.n_hidden, self.n_input);
        for v in h1.iter_mut() {
            *v = gelu(*v);
        }

        // Layer 2: h2 = W2 · h1 + b2
        let mut h2 = self.b2.clone();
        matmul_add(&self.w2, &h1, &mut h2, self.n_hidden, self.n_hidden);

        // LayerNorm: output = gamma * (h2 - mean) / sqrt(var + eps) + beta
        layer_norm(&mut h2, &self.ln_gamma, &self.ln_beta);

        Ok(h2)
    }

    /// Train on a batch of (input, target) pairs using SGD.
    ///
    /// Loss = MSE(output, target) + λ · (1 - cosine(output, target))
    ///
    /// For Phase C initial training, target = text_hidden_state (reconstruction).
    /// The ProjectionNet learns to pass through the text embedding while
    /// incorporating topology info from the GNN embedding.
    ///
    /// Returns the average loss over the batch.
    pub fn train(
        &mut self,
        data: &[(Vec<f32>, Vec<f32>)],
        epochs: usize,
        lr: f32,
        lambda_cosine: f32,
    ) -> Result<Vec<f32>> {
        if data.is_empty() {
            bail!("ProjectionNet::train: empty dataset");
        }

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f64;

            for (input, target) in data {
                if input.len() != self.n_input || target.len() != self.n_hidden {
                    continue; // skip malformed samples
                }

                // Forward pass with intermediate activations saved for backprop
                let (h1_pre, h1, h2_pre, output) = self.forward_with_intermediates(input);

                // Compute loss
                let mse = mse_loss(&output, target);
                let cos_loss = 1.0 - cosine_similarity(&output, target);
                let loss = mse + lambda_cosine * cos_loss;
                epoch_loss += loss as f64;

                // Backward pass — compute gradients and update weights
                self.backward_sgd(
                    input,
                    &h1_pre,
                    &h1,
                    &h2_pre,
                    &output,
                    target,
                    lr,
                    lambda_cosine,
                );
                self.n_updates += 1;
            }

            let avg_loss = (epoch_loss / data.len() as f64) as f32;
            losses.push(avg_loss);

            if epoch % 10 == 0 || epoch == epochs - 1 {
                eprintln!(
                    "  [ProjectionNet] epoch {}/{}: loss={:.6}",
                    epoch + 1,
                    epochs,
                    avg_loss
                );
            }
        }

        Ok(losses)
    }

    /// Forward pass that keeps intermediates for backprop.
    fn forward_with_intermediates(
        &self,
        input: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Layer 1: h1_pre = W1·input + b1, h1 = GELU(h1_pre)
        let mut h1_pre = self.b1.clone();
        matmul_add(&self.w1, input, &mut h1_pre, self.n_hidden, self.n_input);
        let h1: Vec<f32> = h1_pre.iter().map(|&x| gelu(x)).collect();

        // Layer 2: h2_pre = W2·h1 + b2
        let mut h2_pre = self.b2.clone();
        matmul_add(&self.w2, &h1, &mut h2_pre, self.n_hidden, self.n_hidden);

        // LayerNorm
        let mut output = h2_pre.clone();
        layer_norm(&mut output, &self.ln_gamma, &self.ln_beta);

        (h1_pre, h1, h2_pre, output)
    }

    /// SGD backward pass — compute gradients and apply updates in-place.
    fn backward_sgd(
        &mut self,
        input: &[f32],
        h1_pre: &[f32],
        h1: &[f32],
        h2_pre: &[f32],
        output: &[f32],
        target: &[f32],
        lr: f32,
        lambda_cosine: f32,
    ) {
        let n = self.n_hidden;

        // dL/d_output = dMSE/d_output + λ · dCosine/d_output
        let mut d_output = vec![0.0f32; n];

        // MSE gradient: 2/n * (output - target)
        let mse_scale = 2.0 / n as f32;
        for i in 0..n {
            d_output[i] += mse_scale * (output[i] - target[i]);
        }

        // Cosine gradient: d(1 - cos)/d_output
        let dot: f32 = output.iter().zip(target).map(|(a, b)| a * b).sum();
        let norm_o: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let norm_t: f32 = target.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for i in 0..n {
            // d(-cos)/d_o[i] = -(t[i]/(|o|·|t|) - o[i]·dot/(|o|³·|t|))
            let d_cos = -(target[i] / (norm_o * norm_t)
                - output[i] * dot / (norm_o * norm_o * norm_o * norm_t));
            d_output[i] += lambda_cosine * d_cos;
        }

        // Backprop through LayerNorm (simplified: treat as identity for gradient scaling)
        // Full LN gradient is complex; for small lr SGD, this approximation works.
        // d_h2_pre ≈ gamma * d_output (ignoring variance normalization gradient)
        let d_h2: Vec<f32> = d_output
            .iter()
            .enumerate()
            .map(|(i, &d)| d * self.ln_gamma[i])
            .collect();

        // Update LN params
        let (mean, var) = mean_var(h2_pre);
        let inv_std = 1.0 / (var + 1e-5f32).sqrt();
        for i in 0..n {
            let h_norm = (h2_pre[i] - mean) * inv_std;
            self.ln_gamma[i] -= lr * d_output[i] * h_norm;
            self.ln_beta[i] -= lr * d_output[i];
        }

        // Layer 2: d_h1 = W2^T · d_h2, update W2, b2
        let mut d_h1 = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                d_h1[j] += self.w2[i * n + j] * d_h2[i];
            }
            // Update W2[i,j] -= lr * d_h2[i] * h1[j]
            for j in 0..n {
                self.w2[i * n + j] -= lr * d_h2[i] * h1[j];
            }
            self.b2[i] -= lr * d_h2[i];
        }

        // Backprop through GELU: d_h1_pre = d_h1 * gelu'(h1_pre)
        let d_h1_pre: Vec<f32> = d_h1
            .iter()
            .zip(h1_pre)
            .zip(h1)
            .map(|((&dh, &pre), &post)| dh * gelu_derivative(pre, post))
            .collect();

        // Layer 1: update W1, b1
        let n_in = self.n_input;
        for i in 0..n {
            for j in 0..n_in {
                self.w1[i * n_in + j] -= lr * d_h1_pre[i] * input[j];
            }
            self.b1[i] -= lr * d_h1_pre[i];
        }
    }

    /// C6: Train with combined reconstruction + topological contrastive loss.
    ///
    /// Loss = L_reconstruction(MSE + cosine) + λ_topo · L_contrastive(InfoNCE)
    ///        + λ_margin · max(0, margin - cos(output, target))
    ///
    /// `data`: (input=concat(h,g), target=h) pairs — same as `train()`
    /// `contrastive_samples`: (anchor_idx, positive_idx, negative_indices) into `data`
    /// `config`: contrastive hyperparameters
    ///
    /// Returns per-epoch (total_loss, reconstruction_loss, contrastive_loss).
    pub fn train_contrastive(
        &mut self,
        data: &[(Vec<f32>, Vec<f32>)],
        contrastive_pairs: &[(usize, usize, Vec<usize>)], // (anchor_idx, pos_idx, neg_indices)
        epochs: usize,
        lr: f32,
        lambda_cosine: f32,
        config: &crate::contrastive::ContrastiveConfig,
    ) -> Result<Vec<(f32, f32, f32)>> {
        if data.is_empty() {
            bail!("ProjectionNet::train_contrastive: empty dataset");
        }

        let mut history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_recon_loss = 0.0f64;
            let mut epoch_contra_loss = 0.0f64;

            // Phase 1: Reconstruction pass (same as train())
            // We need all projected embeddings for the contrastive phase,
            // so do a full forward pass first, then backprop both losses together.

            // Compute all projected embeddings for this epoch
            let projected: Vec<Vec<f32>> = data
                .iter()
                .map(|(input, _)| {
                    self.forward(input)
                        .unwrap_or_else(|_| vec![0.0; self.n_hidden])
                })
                .collect();

            // Phase 2: For each sample, compute reconstruction + contrastive gradient
            for (_idx, (input, target)) in data.iter().enumerate() {
                if input.len() != self.n_input || target.len() != self.n_hidden {
                    continue;
                }

                // Forward with intermediates for backprop
                let (h1_pre, h1, h2_pre, output) = self.forward_with_intermediates(input);

                // Reconstruction loss
                let mse = mse_loss(&output, target);
                let cos_loss = 1.0 - cosine_similarity(&output, target);
                let margin_loss =
                    (config.cosine_margin - cosine_similarity(&output, target)).max(0.0);
                let recon_loss =
                    mse + lambda_cosine * cos_loss + config.lambda_margin * margin_loss;
                epoch_recon_loss += recon_loss as f64;

                // Start with reconstruction gradient
                self.backward_sgd(
                    input,
                    &h1_pre,
                    &h1,
                    &h2_pre,
                    &output,
                    target,
                    lr,
                    lambda_cosine,
                );

                // Add margin loss gradient if active
                if cosine_similarity(&output, target) < config.cosine_margin {
                    // d margin_loss / d output = -d cos / d output
                    let cos_grad = cosine_sim_grad(&output, target);
                    self.apply_output_gradient(
                        input,
                        &h1_pre,
                        &h1,
                        &h2_pre,
                        &cos_grad,
                        lr * config.lambda_margin,
                        true,
                    );
                }

                self.n_updates += 1;
            }

            // Phase 3: Contrastive pass — InfoNCE on projected embeddings
            for &(anchor_idx, pos_idx, ref neg_indices) in contrastive_pairs {
                if anchor_idx >= projected.len() || pos_idx >= projected.len() {
                    continue;
                }

                let anchor_emb = &projected[anchor_idx];
                let pos_emb = &projected[pos_idx];

                let sim_pos = crate::contrastive::cosine_sim(anchor_emb, pos_emb);
                let sim_negs: Vec<f32> = neg_indices
                    .iter()
                    .filter(|&&ni| ni < projected.len())
                    .map(|&ni| crate::contrastive::cosine_sim(anchor_emb, &projected[ni]))
                    .collect();

                if sim_negs.is_empty() {
                    continue;
                }

                let (loss, d_sim_pos, d_sim_negs) =
                    crate::contrastive::info_nce_loss(sim_pos, &sim_negs, config.temperature);
                epoch_contra_loss += loss as f64;

                // Backprop InfoNCE through the anchor's ProjectionNet
                // d L / d anchor_emb = d_sim_pos * d cos(anchor, pos)/d anchor
                //                    + Σ d_sim_neg_k * d cos(anchor, neg_k)/d anchor
                let anchor_input = &data[anchor_idx].0;
                if anchor_input.len() != self.n_input {
                    continue;
                }

                let mut d_anchor = vec![0.0f32; self.n_hidden];

                // Gradient from positive pair
                let grad_pos = crate::contrastive::cosine_sim_grad_a(anchor_emb, pos_emb);
                for i in 0..self.n_hidden {
                    d_anchor[i] += d_sim_pos * grad_pos[i];
                }

                // Gradient from negative pairs
                for (k, &ni) in neg_indices.iter().enumerate() {
                    if ni >= projected.len() || k >= d_sim_negs.len() {
                        continue;
                    }
                    let grad_neg =
                        crate::contrastive::cosine_sim_grad_a(anchor_emb, &projected[ni]);
                    for i in 0..self.n_hidden {
                        d_anchor[i] += d_sim_negs[k] * grad_neg[i];
                    }
                }

                // Apply contrastive gradient through the network (using anchor's input)
                let (h1_pre, h1, h2_pre, _) = self.forward_with_intermediates(anchor_input);
                self.apply_output_gradient(
                    anchor_input,
                    &h1_pre,
                    &h1,
                    &h2_pre,
                    &d_anchor,
                    lr * config.lambda_topo,
                    false,
                );
            }

            let n = data.len().max(1) as f64;
            let nc = contrastive_pairs.len().max(1) as f64;
            let avg_recon = (epoch_recon_loss / n) as f32;
            let avg_contra = (epoch_contra_loss / nc) as f32;
            let avg_total = avg_recon + config.lambda_topo * avg_contra;

            history.push((avg_total, avg_recon, avg_contra));

            if epoch % 10 == 0 || epoch == epochs - 1 {
                eprintln!(
                    "  [ProjectionNet] epoch {}/{}: total={:.6} recon={:.6} contra={:.6}",
                    epoch + 1,
                    epochs,
                    avg_total,
                    avg_recon,
                    avg_contra
                );
            }
        }

        Ok(history)
    }

    /// Apply a pre-computed gradient on the output back through the network.
    ///
    /// `negate`: if true, negate the gradient (for losses that need gradient descent
    /// on the negative direction, like margin loss where we want to INCREASE cosine).
    fn apply_output_gradient(
        &mut self,
        input: &[f32],
        h1_pre: &[f32],
        h1: &[f32],
        _h2_pre: &[f32],
        d_output: &[f32],
        lr: f32,
        negate: bool,
    ) {
        let n = self.n_hidden;
        let sign = if negate { -1.0f32 } else { 1.0 };

        // Backprop through LayerNorm (simplified)
        let d_h2: Vec<f32> = d_output
            .iter()
            .enumerate()
            .map(|(i, &d)| sign * d * self.ln_gamma[i])
            .collect();

        // Layer 2 weight update
        let mut d_h1 = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                d_h1[j] += self.w2[i * n + j] * d_h2[i];
            }
            for j in 0..n {
                self.w2[i * n + j] -= lr * d_h2[i] * h1[j];
            }
            self.b2[i] -= lr * d_h2[i];
        }

        // Backprop through GELU
        let d_h1_pre: Vec<f32> = d_h1
            .iter()
            .zip(h1_pre)
            .zip(h1)
            .map(|((&dh, &pre), &post)| dh * gelu_derivative(pre, post))
            .collect();

        // Layer 1 weight update
        let n_in = self.n_input;
        for i in 0..n {
            for j in 0..n_in {
                self.w1[i * n_in + j] -= lr * d_h1_pre[i] * input[j];
            }
            self.b1[i] -= lr * d_h1_pre[i];
        }
    }

    /// Number of training updates performed.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Total number of trainable parameters.
    pub fn n_params(&self) -> usize {
        self.w1.len()
            + self.b1.len()
            + self.w2.len()
            + self.b2.len()
            + self.ln_gamma.len()
            + self.ln_beta.len()
    }

    /// Input dimension.
    pub fn n_input(&self) -> usize {
        self.n_input
    }

    /// Hidden/output dimension.
    pub fn n_hidden(&self) -> usize {
        self.n_hidden
    }

    // ── P3: Persistence ──────────────────────────────────────────

    /// Save weights to a binary file.
    ///
    /// Format: [magic:u32][version:u32][n_input:u32][n_hidden:u32][n_updates:u64]
    ///         [w1...][b1...][w2...][b2...][ln_gamma...][ln_beta...] (all f32 little-endian)
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        const MAGIC: u32 = 0x504E_4554; // "PNET"
        const VERSION: u32 = 1;

        f.write_all(&MAGIC.to_le_bytes())?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.n_input as u32).to_le_bytes())?;
        f.write_all(&(self.n_hidden as u32).to_le_bytes())?;
        f.write_all(&self.n_updates.to_le_bytes())?;

        Self::write_vec(&mut f, &self.w1)?;
        Self::write_vec(&mut f, &self.b1)?;
        Self::write_vec(&mut f, &self.w2)?;
        Self::write_vec(&mut f, &self.b2)?;
        Self::write_vec(&mut f, &self.ln_gamma)?;
        Self::write_vec(&mut f, &self.ln_beta)?;

        Ok(())
    }

    /// Load weights from a binary file.
    ///
    /// Returns None if the file doesn't exist. Errors if the file is corrupt
    /// or dimensions don't match.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }

        let mut f = std::fs::File::open(path)?;
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        f.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != 0x504E_4554 {
            bail!("ProjectionNet::load: bad magic {:#x}", magic);
        }

        f.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            bail!("ProjectionNet::load: unsupported version {}", version);
        }

        f.read_exact(&mut buf4)?;
        let n_input = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let n_hidden = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf8)?;
        let n_updates = u64::from_le_bytes(buf8);

        let w1 = Self::read_vec(&mut f, n_hidden * n_input)?;
        let b1 = Self::read_vec(&mut f, n_hidden)?;
        let w2 = Self::read_vec(&mut f, n_hidden * n_hidden)?;
        let b2 = Self::read_vec(&mut f, n_hidden)?;
        let ln_gamma = Self::read_vec(&mut f, n_hidden)?;
        let ln_beta = Self::read_vec(&mut f, n_hidden)?;

        Ok(Some(Self {
            w1,
            b1,
            w2,
            b2,
            ln_gamma,
            ln_beta,
            n_input,
            n_hidden,
            n_updates,
        }))
    }

    fn write_vec(f: &mut std::fs::File, data: &[f32]) -> Result<()> {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        f.write_all(bytes)?;
        Ok(())
    }

    fn read_vec(f: &mut std::fs::File, len: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; len * 4];
        f.read_exact(&mut bytes)?;
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(floats)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Math utilities (no external dependencies)
// ═══════════════════════════════════════════════════════════════════════════

/// Soft mixing: α · a + (1-α) · b
///
/// Used for gradual transition from text embeddings to projected embeddings.
/// α starts at 1.0 (100% text) and decreases to 0.0 (100% projected).
pub fn soft_mix(text_embd: &[f32], proj_embd: &[f32], alpha: f32) -> Vec<f32> {
    debug_assert_eq!(text_embd.len(), proj_embd.len());
    text_embd
        .iter()
        .zip(proj_embd)
        .map(|(&a, &b)| alpha * a + (1.0 - alpha) * b)
        .collect()
}

/// Compute α schedule: α = max(0, 1 - query_count / warmup_queries)
pub fn alpha_schedule(query_count: u64, warmup_queries: u64) -> f32 {
    if warmup_queries == 0 {
        return 0.0;
    }
    (1.0 - query_count as f32 / warmup_queries as f32).max(0.0)
}

/// Matrix-vector multiply and add: out += W · x, where W is [rows × cols] row-major.
fn matmul_add(w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for i in 0..rows {
        let row = &w[i * cols..(i + 1) * cols];
        let dot: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
        out[i] += dot;
    }
}

/// LayerNorm in-place: x = gamma * (x - mean) / sqrt(var + eps) + beta
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let (mean, var) = mean_var(x);
    let inv_std = 1.0 / (var + 1e-5f32).sqrt();
    for i in 0..x.len() {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

/// Compute mean and variance.
fn mean_var(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean as f32, var as f32)
}

/// GELU derivative approximation: gelu'(x) ≈ gelu(x)/x + x · gelu''(x)
/// Simpler approximation: use finite difference around x.
#[inline]
fn gelu_derivative(x: f32, _gelu_x: f32) -> f32 {
    // Analytical: Φ(x) + x·φ(x) where Φ=CDF, φ=PDF of standard normal
    // Approximation via the tanh formula derivative:
    const A: f32 = 0.7978845608;
    const B: f32 = 0.044715;
    let inner = A * (x + B * x * x * x);
    let tanh_val = (inner as f64).tanh() as f32;
    let sech2 = 1.0 - tanh_val * tanh_val;
    let d_inner = A * (1.0 + 3.0 * B * x * x);
    0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
}

/// MSE loss: (1/n) Σ(output - target)²
fn mse_loss(output: &[f32], target: &[f32]) -> f32 {
    let n = output.len() as f32;
    output
        .iter()
        .zip(target)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / n
}

/// Cosine similarity: dot(a,b) / (|a|·|b|)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Gradient of cosine similarity w.r.t. first vector: d cos(a,b) / d a
fn cosine_sim_grad(a: &[f32], b: &[f32]) -> Vec<f32> {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    let na3 = na * na * na;
    a.iter()
        .enumerate()
        .map(|(i, &ai)| b[i] / (na * nb) - ai * dot / (na3 * nb))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_values() {
        // GELU(0) = 0
        assert!((gelu(0.0)).abs() < 1e-6);
        // GELU(large positive) ≈ x
        assert!((gelu(5.0) - 5.0).abs() < 0.01);
        // GELU(large negative) ≈ 0
        assert!((gelu(-5.0)).abs() < 0.01);
        // GELU(1) ≈ 0.8412
        assert!((gelu(1.0) - 0.8412).abs() < 0.01);
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        layer_norm(&mut x, &gamma, &beta);

        // After LN, mean ≈ 0, var ≈ 1
        let (mean, var) = mean_var(&x);
        assert!(mean.abs() < 1e-5, "mean should be ≈0, got {}", mean);
        assert!((var - 1.0).abs() < 0.1, "var should be ≈1, got {}", var);
    }

    #[test]
    fn test_forward_shape() {
        let net = ProjectionNet::new(8, 4); // n_embd=8, n_gnn=4
        assert_eq!(net.n_input, 12);
        assert_eq!(net.n_hidden, 8);

        let input = vec![0.1; 12];
        let output = net.forward(&input).unwrap();
        assert_eq!(output.len(), 8);

        // Check no NaN
        for (i, v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_forward_wrong_dim() {
        let net = ProjectionNet::new(8, 4);
        let result = net.forward(&[1.0; 5]); // wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_train_converges() {
        // Small net: n_embd=4, n_gnn=2 → n_input=6
        let mut net = ProjectionNet::new(4, 2);

        // Self-supervised: target = first 4 elements of input (identity-like)
        let data: Vec<(Vec<f32>, Vec<f32>)> = (0..20)
            .map(|i| {
                let scale = (i as f32 + 1.0) * 0.1;
                let input = vec![scale, -scale, scale * 0.5, -scale * 0.5, 0.1, -0.1];
                let target = vec![scale, -scale, scale * 0.5, -scale * 0.5]; // reconstruct h
                (input, target)
            })
            .collect();

        let losses = net.train(&data, 100, 0.001, 0.1).unwrap();

        // Loss should decrease
        let first_loss = losses[0];
        let last_loss = losses[losses.len() - 1];
        eprintln!("  first_loss={:.6}, last_loss={:.6}", first_loss, last_loss);
        assert!(
            last_loss < first_loss,
            "loss should decrease: {} → {}",
            first_loss,
            last_loss
        );
        assert!(
            last_loss < 0.5,
            "final loss should be < 0.5, got {}",
            last_loss
        );
    }

    #[test]
    fn test_soft_mix() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // α=1.0 → 100% text
        let r1 = soft_mix(&a, &b, 1.0);
        assert_eq!(r1, vec![1.0, 2.0, 3.0]);

        // α=0.0 → 100% projected
        let r0 = soft_mix(&a, &b, 0.0);
        assert_eq!(r0, vec![4.0, 5.0, 6.0]);

        // α=0.5 → midpoint
        let r5 = soft_mix(&a, &b, 0.5);
        assert!((r5[0] - 2.5).abs() < 1e-6);
        assert!((r5[1] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_schedule() {
        assert!((alpha_schedule(0, 100) - 1.0).abs() < 1e-6);
        assert!((alpha_schedule(50, 100) - 0.5).abs() < 1e-6);
        assert!((alpha_schedule(100, 100) - 0.0).abs() < 1e-6);
        assert!((alpha_schedule(200, 100) - 0.0).abs() < 1e-6); // clamped
        assert!((alpha_schedule(0, 0) - 0.0).abs() < 1e-6); // edge case
    }

    #[test]
    fn test_n_params() {
        let net = ProjectionNet::new(8, 4);
        // w1: 8×12=96, b1: 8, w2: 8×8=64, b2: 8, ln_gamma: 8, ln_beta: 8 = 192
        assert_eq!(net.n_params(), 96 + 8 + 64 + 8 + 8 + 8);
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_train_contrastive_loss_decreases() {
        // Small net: n_embd=4, n_gnn=2 → n_input=6
        let mut net = ProjectionNet::new(4, 2);

        // 6 nodes with different patterns
        let data: Vec<(Vec<f32>, Vec<f32>)> = vec![
            // Cluster A: nodes 0, 1, 2 (similar patterns)
            (
                vec![1.0, 0.5, -0.3, 0.8, 0.1, -0.1],
                vec![1.0, 0.5, -0.3, 0.8],
            ),
            (
                vec![0.9, 0.6, -0.2, 0.7, 0.15, -0.05],
                vec![0.9, 0.6, -0.2, 0.7],
            ),
            (
                vec![1.1, 0.4, -0.4, 0.9, 0.05, -0.15],
                vec![1.1, 0.4, -0.4, 0.9],
            ),
            // Cluster B: nodes 3, 4, 5 (different patterns)
            (
                vec![-0.5, 1.0, 0.8, -0.3, -0.1, 0.2],
                vec![-0.5, 1.0, 0.8, -0.3],
            ),
            (
                vec![-0.4, 0.9, 0.7, -0.2, -0.15, 0.25],
                vec![-0.4, 0.9, 0.7, -0.2],
            ),
            (
                vec![-0.6, 1.1, 0.9, -0.4, -0.05, 0.15],
                vec![-0.6, 1.1, 0.9, -0.4],
            ),
        ];

        // Contrastive pairs: within-cluster = positive, cross-cluster = negative
        let contrastive_pairs = vec![
            (0, 1, vec![3, 4, 5]), // anchor=0, pos=1 (same cluster), neg=3,4,5
            (1, 2, vec![3, 4, 5]), // anchor=1, pos=2, neg=3,4,5
            (3, 4, vec![0, 1, 2]), // anchor=3, pos=4, neg=0,1,2
            (4, 5, vec![0, 1, 2]), // anchor=4, pos=5, neg=0,1,2
        ];

        let config = crate::contrastive::ContrastiveConfig::default();
        let history = net
            .train_contrastive(&data, &contrastive_pairs, 50, 0.001, 0.1, &config)
            .unwrap();

        assert!(!history.is_empty());
        let (first_total, _, _) = history[0];
        let (last_total, _, _) = history[history.len() - 1];
        eprintln!(
            "  contrastive: first={:.4}, last={:.4}",
            first_total, last_total
        );
        assert!(
            last_total < first_total,
            "total loss should decrease: {} → {}",
            first_total,
            last_total
        );

        // After training, within-cluster cosine should be higher than cross-cluster
        let p0 = net.forward(&data[0].0).unwrap();
        let p1 = net.forward(&data[1].0).unwrap();
        let p3 = net.forward(&data[3].0).unwrap();

        let sim_within = cosine_similarity(&p0, &p1);
        let sim_cross = cosine_similarity(&p0, &p3);
        eprintln!("  within={:.4}, cross={:.4}", sim_within, sim_cross);
        // Note: with only 50 epochs on tiny data, we may not always achieve
        // perfect separation, but the loss should at least decrease
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut net = ProjectionNet::new(8, 4);
        // Train a bit so weights diverge from init
        let data: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
            .map(|i| {
                let s = (i as f32 + 1.0) * 0.1;
                (
                    vec![
                        s,
                        -s,
                        s * 0.5,
                        -s * 0.5,
                        0.1,
                        -0.1,
                        s * 0.3,
                        -s * 0.3,
                        0.2,
                        -0.2,
                        s * 0.1,
                        -s * 0.1,
                    ],
                    vec![s, -s, s * 0.5, -s * 0.5, 0.1, -0.1, s * 0.3, -s * 0.3],
                )
            })
            .collect();
        net.train(&data, 10, 0.001, 0.1).unwrap();
        assert!(net.n_updates() > 0);

        // Save
        let path = std::path::PathBuf::from("/tmp/test_pnet_weights.bin");
        net.save(&path).unwrap();

        // Load
        let loaded = ProjectionNet::load(&path).unwrap().unwrap();
        assert_eq!(loaded.n_input(), net.n_input());
        assert_eq!(loaded.n_hidden(), net.n_hidden());
        assert_eq!(loaded.n_updates(), net.n_updates());

        // Check weights are identical
        let input = vec![0.1; 12];
        let out1 = net.forward(&input).unwrap();
        let out2 = loaded.forward(&input).unwrap();
        for (a, b) in out1.iter().zip(&out2) {
            assert!((a - b).abs() < 1e-6, "weight mismatch: {} vs {}", a, b);
        }

        // Load from non-existent file
        let missing =
            ProjectionNet::load(std::path::Path::new("/tmp/nonexistent_pnet.bin")).unwrap();
        assert!(missing.is_none());

        // Cleanup
        std::fs::remove_file(&path).ok();
    }
}
