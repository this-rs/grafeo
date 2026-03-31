//! ProjectionNet training pipeline — auto-generates training data and manages
//! initial + online training for the Phase C embedding projection.
//!
//! P1: At startup, generates (concat(h, g), h) pairs from existing nodes and
//!     trains the ProjectionNet to reconstruct text embeddings from fused input.
//!
//! P2: During the session, collects new samples and periodically refines weights.
//!
//! C6: When graph edges are provided, uses contrastive loss (InfoNCE) in addition
//!     to reconstruction loss. Connected nodes should produce similar embeddings.

use anyhow::Result;
use obrain_common::types::NodeId;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::contrastive::{ContrastiveConfig, GraphTopology};
use crate::node_embedding::compute_text_embedding;
use crate::{Engine, ProjectionNet};

/// Configuration for ProjectionNet training.
pub struct TrainingConfig {
    /// Number of epochs for initial training (P1). Default: 50.
    pub initial_epochs: usize,
    /// Learning rate. Default: 0.001.
    pub lr: f32,
    /// Cosine loss weight. Default: 0.1.
    pub lambda_cosine: f32,
    /// Minimum samples needed to trigger training. Default: 5.
    pub min_samples: usize,
    /// Online training: epochs per update (P2). Default: 5.
    pub online_epochs: usize,
    /// Online training: retrain every N queries. Default: 10.
    pub online_interval: u64,
    /// Path to save/load weights.
    pub weights_path: Option<PathBuf>,
    /// Kurtosis regularization weight. Penalizes excess kurtosis (> 3.0) in output
    /// distributions, producing quantization-friendly embeddings. Default: 0.0 (disabled).
    pub lambda_kurtosis: f32,
    /// InnerQ regularization weight. Penalizes unequal per-channel variance across
    /// the batch, producing uniformly-distributed channels. Default: 0.0 (disabled).
    pub lambda_innerq: f32,
    /// C6: Contrastive training config. None = reconstruction only (pre-C6 behavior).
    pub contrastive: Option<ContrastiveConfig>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            initial_epochs: 50,
            lr: 0.001,
            lambda_cosine: 0.1,
            min_samples: 5,
            online_epochs: 5,
            online_interval: 10,
            weights_path: None,
            lambda_kurtosis: 0.0,
            lambda_innerq: 0.0,
            contrastive: Some(ContrastiveConfig::default()),
        }
    }
}

/// Manages ProjectionNet training lifecycle.
pub struct TrainingManager {
    config: TrainingConfig,
    /// Accumulated training data: (input=concat(h,g), target=h).
    training_data: Vec<(Vec<f32>, Vec<f32>)>,
    /// Node ID → index in training_data (for contrastive pair generation).
    node_to_idx: HashMap<NodeId, usize>,
    /// Contrastive pairs: (anchor_idx, pos_idx, neg_indices) into training_data.
    contrastive_pairs: Vec<(usize, usize, Vec<usize>)>,
    /// Query counter for online training triggers.
    query_count: u64,
}

impl TrainingManager {
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            training_data: Vec::new(),
            node_to_idx: HashMap::new(),
            contrastive_pairs: Vec::new(),
            query_count: 0,
        }
    }

    /// P1 + C6: Generate training data from existing nodes and train the ProjectionNet.
    ///
    /// For each node, computes:
    /// - text_h = LLM hidden state (target)
    /// - gnn_g = GNN embedding (64-dim)
    /// - input = concat(text_h, gnn_g)
    ///
    /// If `graph_edges` is provided and contrastive config is set, uses combined
    /// reconstruction + InfoNCE contrastive loss.
    ///
    /// Returns the number of training samples used.
    pub fn initial_training(
        &mut self,
        pnet: &mut ProjectionNet,
        engine: &Engine,
        node_texts: &[(NodeId, String)],
        gnn_embeddings: &HashMap<NodeId, Vec<f32>>,
        graph_edges: Option<&[(NodeId, NodeId)]>,
    ) -> Result<usize> {
        let t0 = std::time::Instant::now();

        // Try to load existing weights first
        if let Some(ref path) = self.config.weights_path {
            if let Ok(Some(loaded)) = ProjectionNet::load(path) {
                if loaded.n_input() == pnet.n_input() && loaded.n_hidden() == pnet.n_hidden() {
                    eprintln!(
                        "  [ProjectionNet] Loaded pre-trained weights ({} updates) from {:?}",
                        loaded.n_updates(),
                        path
                    );
                    *pnet = loaded;
                    return Ok(0); // Skip training, weights are loaded
                } else {
                    eprintln!(
                        "  [ProjectionNet] Saved weights have wrong dims ({}/{} vs {}/{}), retraining",
                        loaded.n_input(),
                        loaded.n_hidden(),
                        pnet.n_input(),
                        pnet.n_hidden()
                    );
                }
            }
        }

        // Generate training pairs
        let mut data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut node_to_idx: HashMap<NodeId, usize> = HashMap::new();
        let mut ordered_nids: Vec<NodeId> = Vec::new();

        for (nid, text) in node_texts {
            // Get text embedding (target)
            let text_h = match compute_text_embedding(engine, text) {
                Ok(h) => h,
                Err(_) => continue,
            };

            // Get GNN embedding (may be absent for some nodes)
            let gnn_g = gnn_embeddings.get(nid);

            if let Some(g) = gnn_g {
                // Build input: concat(h, g)
                let mut input = Vec::with_capacity(text_h.len() + g.len());
                input.extend_from_slice(&text_h);
                input.extend_from_slice(g);

                let idx = data.len();
                node_to_idx.insert(*nid, idx);
                ordered_nids.push(*nid);
                data.push((input, text_h));
            }
            // Skip nodes without GNN embeddings — they'll use text-only fallback
        }

        if data.len() < self.config.min_samples {
            eprintln!(
                "  [ProjectionNet] Only {} samples (need {}), skipping initial training",
                data.len(),
                self.config.min_samples
            );
            return Ok(0);
        }

        // C6: Generate contrastive pairs if graph edges are available
        let use_contrastive =
            self.config.contrastive.is_some() && graph_edges.is_some() && data.len() >= 3;

        if use_contrastive {
            let edges = graph_edges.unwrap();
            let config = self.config.contrastive.as_ref().unwrap();
            let available: HashSet<NodeId> = node_to_idx.keys().copied().collect();

            let topo = GraphTopology::from_edges(&ordered_nids, edges);
            let samples = topo.generate_samples(&available, config.n_negatives, 42);

            self.contrastive_pairs.clear();
            for sample in &samples {
                if let (Some(&a_idx), Some(&p_idx)) = (
                    node_to_idx.get(&sample.anchor),
                    node_to_idx.get(&sample.positive),
                ) {
                    let neg_indices: Vec<usize> = sample
                        .negatives
                        .iter()
                        .filter_map(|nid| node_to_idx.get(nid).copied())
                        .collect();
                    if !neg_indices.is_empty() {
                        self.contrastive_pairs.push((a_idx, p_idx, neg_indices));
                    }
                }
            }

            eprintln!(
                "  [ProjectionNet] C6 contrastive: {} pairs from {} edges, {} available nodes",
                self.contrastive_pairs.len(),
                edges.len(),
                available.len()
            );

            // Train with combined loss
            let history = pnet.train_contrastive(
                &data,
                &self.contrastive_pairs,
                self.config.initial_epochs,
                self.config.lr,
                self.config.lambda_cosine,
                self.config.lambda_kurtosis,
                self.config.lambda_innerq,
                config,
            )?;

            let (final_total, final_recon, final_contra) =
                history
                    .last()
                    .copied()
                    .unwrap_or((f32::MAX, f32::MAX, f32::MAX));

            eprintln!(
                "  [ProjectionNet] C6 training done in {:.1}s: total={:.6} (recon={:.6} contra={:.6}), {} updates",
                t0.elapsed().as_secs_f32(),
                final_total,
                final_recon,
                final_contra,
                pnet.n_updates()
            );
        } else {
            // Reconstruction-only training (pre-C6 fallback)
            eprintln!(
                "  [ProjectionNet] Training on {} node pairs ({} epochs, reconstruction only)...",
                data.len(),
                self.config.initial_epochs
            );

            let losses = pnet.train(
                &data,
                self.config.initial_epochs,
                self.config.lr,
                self.config.lambda_cosine,
                self.config.lambda_kurtosis,
                self.config.lambda_innerq,
            )?;
            let final_loss = losses.last().copied().unwrap_or(f32::MAX);

            eprintln!(
                "  [ProjectionNet] Training done in {:.1}s: final_loss={:.6}, {} updates",
                t0.elapsed().as_secs_f32(),
                final_loss,
                pnet.n_updates()
            );
        }

        // Save weights
        if let Some(ref path) = self.config.weights_path {
            if let Err(e) = pnet.save(path) {
                eprintln!("  ⚠ [ProjectionNet] Failed to save weights: {}", e);
            } else {
                eprintln!("  [ProjectionNet] Weights saved to {:?}", path);
            }
        }

        // Keep data for online refinement
        self.training_data = data.clone();
        self.node_to_idx = node_to_idx;

        Ok(data.len())
    }

    /// P2: Called after each query. Collects new samples and triggers
    /// online refinement when enough queries have accumulated.
    ///
    /// Returns true if training was performed.
    pub fn on_query(
        &mut self,
        pnet: &mut ProjectionNet,
        new_samples: &[(Vec<f32>, Vec<f32>)], // (concat(h,g), h) pairs from this query's nodes
    ) -> bool {
        self.query_count += 1;

        // Accumulate new samples
        for (input, target) in new_samples {
            if input.len() == pnet.n_input() && target.len() == pnet.n_hidden() {
                self.training_data.push((input.clone(), target.clone()));
            }
        }

        // Check if it's time for online training
        if self.query_count % self.config.online_interval != 0 {
            return false;
        }

        if self.training_data.len() < self.config.min_samples {
            return false;
        }

        // Online training: use contrastive if pairs are available, else reconstruction-only
        if !self.contrastive_pairs.is_empty() {
            if let Some(ref config) = self.config.contrastive {
                match pnet.train_contrastive(
                    &self.training_data,
                    &self.contrastive_pairs,
                    self.config.online_epochs,
                    self.config.lr * 0.5,
                    self.config.lambda_cosine,
                    self.config.lambda_kurtosis,
                    self.config.lambda_innerq,
                    config,
                ) {
                    Ok(history) => {
                        let (total, recon, contra) =
                            history
                                .last()
                                .copied()
                                .unwrap_or((f32::MAX, f32::MAX, f32::MAX));
                        eprintln!(
                            "  [ProjectionNet] C6 online update (query {}): total={:.6} (r={:.6} c={:.6}), {} samples",
                            self.query_count,
                            total,
                            recon,
                            contra,
                            self.training_data.len()
                        );
                        if let Some(ref path) = self.config.weights_path {
                            let _ = pnet.save(path);
                        }
                        return true;
                    }
                    Err(e) => {
                        eprintln!("  ⚠ [ProjectionNet] C6 online training failed: {}", e);
                    }
                }
            }
        }

        // Fallback: reconstruction-only
        match pnet.train(
            &self.training_data,
            self.config.online_epochs,
            self.config.lr * 0.5, // lower lr for online updates
            self.config.lambda_cosine,
            self.config.lambda_kurtosis,
            self.config.lambda_innerq,
        ) {
            Ok(losses) => {
                let final_loss = losses.last().copied().unwrap_or(f32::MAX);
                eprintln!(
                    "  [ProjectionNet] Online update (query {}): loss={:.6}, {} samples, {} total updates",
                    self.query_count,
                    final_loss,
                    self.training_data.len(),
                    pnet.n_updates()
                );

                // Save weights periodically
                if let Some(ref path) = self.config.weights_path {
                    let _ = pnet.save(path);
                }
                true
            }
            Err(e) => {
                eprintln!("  ⚠ [ProjectionNet] Online training failed: {}", e);
                false
            }
        }
    }

    /// Save weights at session end.
    pub fn save_weights(&self, pnet: &ProjectionNet) {
        if let Some(ref path) = self.config.weights_path {
            if let Err(e) = pnet.save(path) {
                eprintln!(
                    "  ⚠ [ProjectionNet] Failed to save weights at shutdown: {}",
                    e
                );
            }
        }
    }

    /// Get the accumulated training data count.
    pub fn n_samples(&self) -> usize {
        self.training_data.len()
    }

    /// Get the query count.
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Get the number of contrastive pairs.
    pub fn n_contrastive_pairs(&self) -> usize {
        self.contrastive_pairs.len()
    }
}

/// Convenience: compute the weights file path from persona directory.
pub fn weights_path_for_persona(persona_dir: &Path) -> PathBuf {
    persona_dir.join("projection_net.bin")
}
