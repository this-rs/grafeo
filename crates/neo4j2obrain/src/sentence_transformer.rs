//! # `SentenceTransformer` — WordPiece + ONNX embedder for stage 2
//!
//! Self-contained ONNX text embedder aligned on the implementation used
//! by `obrain-hub`'s `retrieval` crate (the one that actually runs at
//! query time). Same code pattern as the historic
//! `obrain/neo4j2obrain/src/enrich_kb.rs::SentenceTransformer`.
//!
//! ## Why a local copy rather than `obrain-engine::OnnxEmbeddingModel`?
//!
//! `obrain-engine` configures `ort` with `default-features = false,
//! features = ["load-dynamic", "api-24"]`. That path deadlocks on macOS
//! inside `ort::environment::current` → `ort::setup_api` →
//! `Error::new_internal` (triple-`OnceLock` reentrance) when the ONNX
//! Runtime ABI cannot be resolved, and the error creation path itself
//! never returns. See the `gotcha` note tagged `ort deadlock` for the
//! full diagnosis.
//!
//! The prod path — `obrain-chat/crates/retrieval::SentenceTransformer`
//! — uses `ort = "2.0.0-rc.12"` with **default features** (bundled ONNX
//! Runtime downloaded by the build script) and has been running daily
//! on obrain-hub without issue. We replicate that config here so the
//! importer produces embeddings in the **same vector space** as the
//! runtime embedder — nodes imported by `neo4j2obrain` are immediately
//! recall-compatible with queries issued by `obrain-hub`.
//!
//! ## Contract
//!
//! * Input: UTF-8 strings, any length (truncated to `max_seq_len`).
//! * Output: 384-dimensional f32 vectors, L2-normalised, mean-pooled
//!   over valid (non-pad) token positions.
//! * Tokenizer: WordPiece over a `vocab.txt` (one token per line). The
//!   `[CLS]` / `[SEP]` / `[UNK]` / `[PAD]` special tokens default to
//!   ids `101 / 102 / 100 / 0` when absent from the vocab (standard
//!   BERT-family layout).
//!
//! ## Thread safety
//!
//! `ort::session::Session` is not `Sync`; we wrap it in `UnsafeCell` and
//! implement `Send`/`Sync` manually. The pipeline calls `embed_batch`
//! from a single thread, matching the prod usage pattern.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};

/// Map `ort` errors to `anyhow`. `ort::Error` is not `Send + Sync + 'static`
/// on every platform, so we flatten it to a string before wrapping.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("ort: {}", e)
}

/// Lightweight sentence-transformer running entirely in ONNX Runtime.
///
/// Produces dense 384-d text embeddings without LLM involvement. The
/// session is wrapped in `UnsafeCell` to allow `&self` inference — same
/// as the runtime `retrieval::SentenceTransformer`.
pub struct SentenceTransformer {
    session: UnsafeCell<ort::session::Session>,
    /// Embedding dimensionality — 384 for MiniLM-L6-v2, detected from
    /// the output tensor shape on first inference if needed.
    pub embed_dim: usize,
    /// WordPiece vocabulary: token string → token id.
    vocab: HashMap<String, i64>,
    /// Maximum sequence length fed to ONNX. 128 is the stock MiniLM
    /// setting and is enough for ~200 words of English / French text.
    max_seq_len: usize,
}

// SAFETY: the pipeline calls `embed_batch` from a single thread. Multi-
// threaded use would require mutex-wrapping the session.
unsafe impl Send for SentenceTransformer {}
unsafe impl Sync for SentenceTransformer {}

impl SentenceTransformer {
    /// Load an ONNX model + WordPiece vocab from disk.
    ///
    /// * `model_path`: path to the `.onnx` file (MiniLM-L6-v2 or
    ///   compatible sentence-transformer). Must be 384-dim output.
    /// * `vocab_path`: path to a WordPiece `vocab.txt` (one token per
    ///   line; line index = token id).
    ///
    /// Uses `intra_threads = 2` — the obrain-hub prod default. The
    /// stage-2 parallel path bypasses this helper and calls
    /// [`Self::load_with_threads`] with `1` to avoid oversubscribing
    /// the rayon thread pool; this wrapper is retained for the `Arc`-
    /// sharing helper in `embedder.rs` and for any future single-
    /// threaded call site (tests, one-off inference).
    #[allow(dead_code)]
    pub fn load(model_path: &Path, vocab_path: &Path) -> Result<Self> {
        Self::load_with_threads(model_path, vocab_path, 2)
    }

    /// Variant with explicit intra-op thread count. `retrieval` uses 2
    /// threads in prod — a good balance between throughput and CPU
    /// contention with the rest of the import pipeline.
    pub fn load_with_threads(
        model_path: &Path,
        vocab_path: &Path,
        intra_threads: usize,
    ) -> Result<Self> {
        let session = ort::session::Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(intra_threads)
            .map_err(ort_err)?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!(
                    "ort: failed to load ONNX model at {}: {}",
                    model_path.display(),
                    e
                )
            })?;

        // MiniLM-L6-v2 fixed output dim. If we ever plug in a different
        // model we can probe the output shape lazily on first inference,
        // but for now we pin the contract to match L2_DIM = 384.
        let embed_dim = 384usize;

        let vocab_text = std::fs::read_to_string(vocab_path).map_err(|e| {
            anyhow::anyhow!(
                "SentenceTransformer: failed to read vocab at {}: {}",
                vocab_path.display(),
                e
            )
        })?;
        let vocab: HashMap<String, i64> = vocab_text
            .lines()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as i64))
            .collect();

        if vocab.is_empty() {
            bail!(
                "SentenceTransformer: vocabulary is empty at {}",
                vocab_path.display()
            );
        }

        tracing::info!(
            "SentenceTransformer loaded: {} vocab tokens, embed_dim={}, model={}",
            vocab.len(),
            embed_dim,
            model_path.display()
        );

        Ok(Self {
            session: UnsafeCell::new(session),
            embed_dim,
            vocab,
            max_seq_len: 128,
        })
    }

    /// Tokenise a single text with greedy WordPiece over the vocab.
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids)` of length
    /// `max_seq_len`, padded with `[PAD]` (id 0 by default).
    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let cls_id = self.vocab.get("[CLS]").copied().unwrap_or(101);
        let sep_id = self.vocab.get("[SEP]").copied().unwrap_or(102);
        let unk_id = self.vocab.get("[UNK]").copied().unwrap_or(100);
        let pad_id = self.vocab.get("[PAD]").copied().unwrap_or(0);

        let mut token_ids = vec![cls_id];
        let text_lower = text.to_lowercase();
        for word in text_lower.split_whitespace() {
            let mut start = 0;
            let chars: Vec<char> = word.chars().collect();
            let mut is_first = true;

            while start < chars.len() {
                let mut end = chars.len();
                let mut found = false;

                while start < end {
                    let substr: String = chars[start..end].iter().collect();
                    let candidate = if is_first {
                        substr.clone()
                    } else {
                        format!("##{}", substr)
                    };

                    if let Some(&id) = self.vocab.get(&candidate) {
                        token_ids.push(id);
                        found = true;
                        start = end;
                        is_first = false;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    token_ids.push(unk_id);
                    start += 1;
                    is_first = false;
                }
            }

            if token_ids.len() >= self.max_seq_len - 1 {
                break;
            }
        }

        token_ids.push(sep_id);
        token_ids.truncate(self.max_seq_len);

        let actual_len = token_ids.len();
        let mut attention_mask = vec![1i64; actual_len];
        let mut token_type_ids = vec![0i64; actual_len];

        token_ids.resize(self.max_seq_len, pad_id);
        attention_mask.resize(self.max_seq_len, 0);
        token_type_ids.resize(self.max_seq_len, 0);

        (token_ids, attention_mask, token_type_ids)
    }

    /// Embed a batch of texts into L2-normalised 384-d vectors.
    ///
    /// Mean-pools over valid (non-pad) tokens when the ONNX output is
    /// `[batch, seq, hidden]`, or passes through when it's already
    /// pooled as `[batch, hidden]`.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let n = texts.len();
        let seq_len = self.max_seq_len;

        let mut all_input_ids = Vec::with_capacity(n * seq_len);
        let mut all_attention_mask = Vec::with_capacity(n * seq_len);
        let mut all_token_type_ids = Vec::with_capacity(n * seq_len);

        for text in texts {
            let (ids, mask, types) = self.tokenize(text);
            all_input_ids.extend_from_slice(&ids);
            all_attention_mask.extend_from_slice(&mask);
            all_token_type_ids.extend_from_slice(&types);
        }

        let input_ids =
            ort::value::Tensor::from_array(([n, seq_len], all_input_ids.into_boxed_slice()))
                .map_err(ort_err)?;

        let attention_mask_tensor = ort::value::Tensor::from_array((
            [n, seq_len],
            all_attention_mask.clone().into_boxed_slice(),
        ))
        .map_err(ort_err)?;

        let token_type_ids =
            ort::value::Tensor::from_array(([n, seq_len], all_token_type_ids.into_boxed_slice()))
                .map_err(ort_err)?;

        // SAFETY: single-threaded access, the pipeline owns this session.
        let session = unsafe { &mut *self.session.get() };
        let outputs = session
            .run(ort::inputs![
                input_ids,
                attention_mask_tensor,
                token_type_ids
            ])
            .map_err(ort_err)?;

        let (shape, raw) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;

        let embeddings = if shape.len() == 3 {
            // [batch, seq, hidden] — mean-pool over non-masked tokens.
            let hidden_dim = shape[2] as usize;
            let seq_dim = shape[1] as usize;

            let mut results = Vec::with_capacity(n);
            for i in 0..n {
                let mut pooled = vec![0.0f32; hidden_dim];
                let mask_sum: f32 = all_attention_mask[i * seq_len..(i + 1) * seq_len]
                    .iter()
                    .map(|&m| m as f32)
                    .sum();
                if mask_sum > 0.0 {
                    for t in 0..seq_dim {
                        let m = all_attention_mask[i * seq_len + t] as f32;
                        if m > 0.0 {
                            let offset = (i * seq_dim + t) * hidden_dim;
                            for d in 0..hidden_dim {
                                pooled[d] += raw[offset + d];
                            }
                        }
                    }
                    for d in 0..hidden_dim {
                        pooled[d] /= mask_sum;
                    }
                }
                l2_normalise(&mut pooled);
                results.push(pooled);
            }
            results
        } else if shape.len() == 2 {
            // [batch, hidden] — already pooled.
            let hidden_dim = shape[1] as usize;
            let mut results = Vec::with_capacity(n);
            for i in 0..n {
                let mut emb = raw[i * hidden_dim..(i + 1) * hidden_dim].to_vec();
                l2_normalise(&mut emb);
                results.push(emb);
            }
            results
        } else {
            bail!("SentenceTransformer: unexpected output shape {:?}", shape);
        };

        Ok(embeddings)
    }
}

fn l2_normalise(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
