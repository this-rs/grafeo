//! # ONNX embedder wiring (feature = "embed")
//!
//! Thin wrapper over the local [`SentenceTransformer`] — the same
//! WordPiece + ONNX pattern that the obrain-hub prod runtime uses at
//! query time. Loading by path instead of by HuggingFace preset keeps
//! the importer aligned with the runtime's exact `.onnx` + `vocab.txt`
//! pair, so vectors land in the same space no download, no HF cache
//! lookups, no model-name resolution.
//!
//! This module is compiled only under `--features embed` — the
//! hermetic default workspace build does not pull the ~17 MB ort
//! transitive closure.
//!
//! ## Contract
//!
//! * Input: UTF-8 strings.
//! * Output: 384-dim f32 vectors, L2-normalised, mean-pooled.
//! * Same tokenizer, same ONNX session semantics as
//!   `obrain-chat/crates/retrieval::SentenceTransformer`.

#![cfg(feature = "embed")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::sentence_transformer::SentenceTransformer;

/// Load a [`SentenceTransformer`] from an explicit `.onnx` + `vocab.txt`
/// pair. Returns an `Arc` so the pipeline can share the session across
/// the batched stage-2 loop without recloning the vocab map.
///
/// * `model_path`: 384-dim sentence-transformer `.onnx` (MiniLM-L6-v2
///   or any compatible export). Must match the runtime model used by
///   obrain-hub for the vectors to be query-compatible.
/// * `vocab_path`: WordPiece `vocab.txt` (one token per line). Must be
///   the same vocab the `.onnx` was trained with.
///
/// **Note**: since stage 2 now spawns a thread-local
/// `SentenceTransformer` per rayon worker (`intra_threads = 1`), the
/// pipeline hits [`crate::sentence_transformer::SentenceTransformer::load_with_threads`]
/// directly. This helper is kept for callers that want a shareable
/// `Arc`-wrapped session (tests, one-off scripts) — hence the
/// `#[allow(dead_code)]`.
#[allow(dead_code)]
pub fn load(model_path: &Path, vocab_path: &Path) -> Result<Arc<SentenceTransformer>> {
    let st = SentenceTransformer::load(model_path, vocab_path).with_context(|| {
        format!(
            "load SentenceTransformer (model={}, vocab={})",
            model_path.display(),
            vocab_path.display()
        )
    })?;
    Ok(Arc::new(st))
}
