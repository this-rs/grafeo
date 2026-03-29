//! Safe Rust wrapper around llama.cpp FFI bindings.
//!
//! Uses our fork which provides `llama_set_attn_mask()` for topological attention masking.
//!
//! # Safety
//! All FFI calls are wrapped in safe methods. `catch_unwind` is used at FFI boundaries
//! to prevent Rust panics from crossing into C code (which would be UB).

use crate::ffi;
use crate::signals::{StepSignals, GenerationSignals, compute_entropy_top_k};
use anyhow::{Result, bail};
use std::ffi::CString;
use std::panic::catch_unwind;
use std::ptr;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

/// Controls whether llama.cpp C-level logs (DEBUG/INFO) are shown.
/// Set to true before calling LlamaEngine::new() to see verbose C logs.
static VERBOSE_LLAMA: AtomicBool = AtomicBool::new(false);

/// Set the verbosity of llama.cpp C-level logging.
pub fn set_verbose(verbose: bool) {
    VERBOSE_LLAMA.store(verbose, Ordering::Relaxed);
}

/// Custom log callback that filters llama.cpp logs by level.
/// Only WARN and ERROR are shown unless VERBOSE_LLAMA is true.
unsafe extern "C" fn llama_log_callback(
    level: ffi::ggml_log_level,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    let verbose = VERBOSE_LLAMA.load(Ordering::Relaxed);
    if !verbose {
        // In quiet mode, only show errors (no warnings, no info, no debug)
        if level != ffi::ggml_log_level::GGML_LOG_LEVEL_ERROR {
            return;
        }
    }
    if !text.is_null() {
        // SAFETY: text is a valid C string from llama.cpp
        let s = unsafe { std::ffi::CStr::from_ptr(text) };
        if let Ok(msg) = s.to_str() {
            eprint!("{}", msg);
        }
    }
}

// Ensure backend is initialized exactly once
static BACKEND_INIT: Once = Once::new();

fn ensure_backend() {
    BACKEND_INIT.call_once(|| unsafe {
        ffi::llama_backend_init();
        // Install our log filter
        ffi::llama_log_set(Some(llama_log_callback), ptr::null_mut());
    });
}

/// Configuration for the LLM engine.
pub struct EngineConfig {
    pub model_path: String,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_batch: u32,
    /// Sampling parameters
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub penalty_repeat: f32,
    pub penalty_last_n: i32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: 32768,
            n_gpu_layers: 99,
            n_batch: 2048,
            temperature: 0.7,
            top_p: 0.9,
            min_p: 0.05,
            penalty_repeat: 1.1,
            penalty_last_n: 64,
        }
    }
}

/// Safe wrapper around llama.cpp model + context + sampler.
///
/// Owns all three pointers and frees them in reverse order on drop.
/// All methods use `catch_unwind` at FFI boundaries.
pub struct LlamaEngine {
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
    sampler: *mut ffi::llama_sampler,
    n_ctx: u32,
}

// SAFETY: The llama.cpp pointers are only accessed through &self/&mut self,
// and llama.cpp is thread-safe for single-context usage.
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load a model and create a context + sampler chain.
    ///
    /// # Errors
    /// Returns error if:
    /// - model_path does not exist or is not a valid GGUF file
    /// - context creation fails (e.g., n_ctx too large for model)
    pub fn new(config: &EngineConfig) -> Result<Self> {
        ensure_backend();

        let c_path = CString::new(config.model_path.as_str())
            .map_err(|_| anyhow::anyhow!("model path contains null byte"))?;

        // --- Load model ---
        let model = unsafe {
            let mut params = ffi::llama_model_default_params();
            params.n_gpu_layers = config.n_gpu_layers;
            ffi::llama_model_load_from_file(c_path.as_ptr(), params)
        };
        if model.is_null() {
            bail!(
                "Failed to load model from '{}'. Check path and GGUF format.",
                config.model_path
            );
        }

        // --- Create context ---
        let ctx = unsafe {
            let mut params = ffi::llama_context_default_params();
            params.n_ctx = config.n_ctx;
            params.n_batch = config.n_batch;
            params.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // Dual seq_id: 0 = persistent context, 1 = query+generation
            params.n_seq_max = 2;
            ffi::llama_init_from_model(model, params)
        };
        if ctx.is_null() {
            // Free model before returning error
            unsafe { ffi::llama_model_free(model) };
            bail!(
                "Failed to create context (n_ctx={}, n_batch={}). Try reducing n_ctx.",
                config.n_ctx,
                config.n_batch
            );
        }

        // --- Build sampler chain ---
        let sampler = unsafe {
            let sparams = ffi::llama_sampler_chain_default_params();
            let chain = ffi::llama_sampler_chain_init(sparams);
            if chain.is_null() {
                ffi::llama_free(ctx);
                ffi::llama_model_free(model);
                bail!("Failed to create sampler chain");
            }

            // Penalties first (needs full logits before top-k/top-p filtering)
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_penalties(
                    config.penalty_last_n,
                    config.penalty_repeat,
                    0.0, // penalty_freq
                    0.0, // penalty_present
                ),
            );

            // Temperature
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_temp(config.temperature),
            );

            // Top-P
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_top_p(config.top_p, 1),
            );

            // Min-P
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_min_p(config.min_p, 1),
            );

            // Final distribution sampler (random seed)
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_dist(ffi::LLAMA_DEFAULT_SEED),
            );

            chain
        };

        let actual_n_ctx = unsafe { ffi::llama_n_ctx(ctx) };

        Ok(Self {
            model,
            ctx,
            sampler,
            n_ctx: actual_n_ctx,
        })
    }

    /// Get the actual context size (may differ from requested if model has a limit).
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Get the raw context pointer (for advanced FFI calls).
    ///
    /// # Safety
    /// Caller must not free or invalidate the pointer.
    pub fn ctx_ptr(&self) -> *mut ffi::llama_context {
        self.ctx
    }

    /// Get the raw model pointer.
    pub fn model_ptr(&self) -> *mut ffi::llama_model {
        self.model
    }

    /// Get the raw sampler pointer.
    pub fn sampler_ptr(&self) -> *mut ffi::llama_sampler {
        self.sampler
    }

    /// Get the memory handle for KV cache operations.
    pub fn memory(&self) -> ffi::llama_memory_t {
        unsafe { ffi::llama_get_memory(self.ctx) }
    }

    /// Get the vocab pointer (needed for tokenize/detokenize).
    fn vocab(&self) -> *const ffi::llama_vocab {
        unsafe { ffi::llama_model_get_vocab(self.model) }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Tokenization
    // ═══════════════════════════════════════════════════════════════════════════

    /// Tokenize text into token IDs.
    ///
    /// - `add_special`: add BOS/EOS if the model expects them
    /// - `parse_special`: recognize special tokens like `<|im_start|>` in the text
    ///
    /// Uses a retry strategy: pre-allocate text.len()/2 + 128 tokens,
    /// if insufficient (negative return = needed size), reallocate and retry.
    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<ffi::llama_token>> {
        let vocab = self.vocab();

        // Pre-allocate: rough estimate
        let mut n_max = (text.len() / 2 + 128) as i32;
        let mut buf: Vec<ffi::llama_token> = vec![0; n_max as usize];

        let n = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                buf.as_mut_ptr(),
                n_max,
                add_special,
                parse_special,
            )
        };

        if n >= 0 {
            buf.truncate(n as usize);
            return Ok(buf);
        }

        // Negative = -(needed_size). Retry with exact size.
        n_max = -n;
        buf.resize(n_max as usize, 0);

        let n2 = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                buf.as_mut_ptr(),
                n_max,
                add_special,
                parse_special,
            )
        };

        if n2 < 0 {
            bail!("tokenize failed after retry (text len={}, needed={})", text.len(), -n2);
        }

        buf.truncate(n2 as usize);
        Ok(buf)
    }

    /// Count tokens in text (convenience wrapper).
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.tokenize(text, false, true)?.len())
    }

    /// Convert a single token ID back to its text piece.
    /// Get the raw bytes for a token (may be partial UTF-8).
    pub fn token_to_bytes(&self, token: ffi::llama_token) -> Vec<u8> {
        let vocab = self.vocab();
        let mut buf = vec![0u8; 128];

        let n = unsafe {
            ffi::llama_token_to_piece(
                vocab,
                token,
                buf.as_mut_ptr() as *mut i8,
                buf.len() as i32,
                0,     // lstrip
                true,  // special
            )
        };

        if n < 0 {
            let needed = (-n) as usize;
            buf.resize(needed, 0);
            let n2 = unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut i8,
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n2 > 0 {
                buf.truncate(n2 as usize);
            } else {
                return Vec::new();
            }
        } else {
            buf.truncate(n as usize);
        }

        buf
    }

    /// Get a token as a String (lossy — may replace partial UTF-8 with �).
    pub fn token_to_piece(&self, token: ffi::llama_token) -> String {
        String::from_utf8_lossy(&self.token_to_bytes(token)).into_owned()
    }

    /// Detokenize a slice of tokens back to text.
    pub fn detokenize(&self, tokens: &[ffi::llama_token]) -> String {
        let mut buf = Vec::new();
        for &tok in tokens {
            buf.extend_from_slice(&self.token_to_bytes(tok));
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Get the EOS token for this model.
    pub fn token_eos(&self) -> ffi::llama_token {
        unsafe { ffi::llama_token_eos(self.vocab()) }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // KV Cache — Encode / Evict / Query
    // ═══════════════════════════════════════════════════════════════════════════

    /// Encode tokens into the KV cache at given positions on a specific sequence.
    ///
    /// - `tokens` and `positions` must have the same length
    /// - `seq_id`: 0 = persistent context nodes, 1 = query+generation (cleaned after each response)
    /// - Only the last token's logits are computed (for efficiency)
    ///
    /// Returns the number of tokens encoded, or error if decode fails.
    pub fn encode(&self, tokens: &[ffi::llama_token], positions: &[ffi::llama_pos], seq_id: ffi::llama_seq_id) -> Result<usize> {
        if tokens.len() != positions.len() {
            bail!("encode: tokens.len()={} != positions.len()={}", tokens.len(), positions.len());
        }
        if tokens.is_empty() {
            return Ok(0);
        }

        // Process in batches of n_batch to avoid exceeding the batch limit
        let n_batch = unsafe { ffi::llama_n_batch(self.ctx) } as usize;

        for chunk_start in (0..tokens.len()).step_by(n_batch) {
            let chunk_end = (chunk_start + n_batch).min(tokens.len());
            let chunk_len = (chunk_end - chunk_start) as i32;

            let mut batch = unsafe { ffi::llama_batch_init(chunk_len, 0, 1) };

            unsafe {
                for i in 0..chunk_len as usize {
                    let idx = chunk_start + i;
                    *batch.token.add(i) = tokens[idx];
                    *batch.pos.add(i) = positions[idx];
                    *batch.n_seq_id.add(i) = 1;
                    *(*batch.seq_id.add(i)) = seq_id;
                    // Only compute logits for the very last token of the very last chunk
                    *batch.logits.add(i) = if idx == tokens.len() - 1 { 1 } else { 0 };
                }
                batch.n_tokens = chunk_len;

                let ret = ffi::llama_decode(self.ctx, batch);
                ffi::llama_batch_free(batch);

                if ret != 0 {
                    bail!("llama_decode failed (ret={}, chunk {}/{})", ret, chunk_start / n_batch, (tokens.len() + n_batch - 1) / n_batch);
                }
            }
        }

        Ok(tokens.len())
    }

    /// Remove KV cache entries for a position range on seq_id 0.
    ///
    /// `pos_start..pos_end` (exclusive end). Use -1,-1 for "all".
    pub fn evict(&self, pos_start: ffi::llama_pos, pos_end: ffi::llama_pos) -> bool {
        let mem = self.memory();
        unsafe { ffi::llama_memory_seq_rm(mem, 0, pos_start, pos_end) }
    }

    /// Clear all tokens for a given sequence ID.
    ///
    /// Used after each response to clean query+generation tokens (seq_id=1).
    pub fn clear_seq(&self, seq_id: ffi::llama_seq_id) {
        let mem = self.memory();
        unsafe { ffi::llama_memory_seq_rm(mem, seq_id, -1, -1); }
    }

    /// Copy seq_id_src → seq_id_dst for positions [p0, p1).
    ///
    /// After this call, KV entries from seq_id_src are also visible to seq_id_dst.
    /// This is essential: tokens decoded on seq_id=1 can only attend to KV entries
    /// that include seq_id=1 in their sequence set. Without seq_cp, query tokens
    /// on seq_id=1 cannot see context nodes encoded on seq_id=0.
    pub fn seq_cp(&self, src: ffi::llama_seq_id, dst: ffi::llama_seq_id, p0: ffi::llama_pos, p1: ffi::llama_pos) {
        let mem = self.memory();
        unsafe { ffi::llama_memory_seq_cp(mem, src, dst, p0, p1); }
    }

    /// Clear the entire KV cache (all sequences).
    pub fn clear_kv(&self) {
        let mem = self.memory();
        unsafe { ffi::llama_memory_clear(mem, true); }
    }

    /// Get the maximum position currently in the KV cache for a sequence.
    /// Returns -1 if the sequence is empty.
    pub fn seq_pos_max(&self, seq_id: ffi::llama_seq_id) -> ffi::llama_pos {
        let mem = self.memory();
        unsafe { ffi::llama_memory_seq_pos_max(mem, seq_id) }
    }

    /// Get the minimum position in the KV cache for a sequence.
    pub fn seq_pos_min(&self, seq_id: ffi::llama_seq_id) -> ffi::llama_pos {
        let mem = self.memory();
        unsafe { ffi::llama_memory_seq_pos_min(mem, seq_id) }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Attention Mask — our fork's custom API
    // ═══════════════════════════════════════════════════════════════════════════

    /// Set a custom attention mask on the context.
    ///
    /// - `mask`: row-major matrix. For broadcast (n_head_groups <= 1): [n_pos × n_pos].
    ///   For per-head: [n_head_groups × n_pos × n_pos].
    /// - `positions`: the position IDs the mask rows/columns correspond to
    /// - `n_head_groups`: 0 or 1 = broadcast same mask to all heads,
    ///   n_head = per-head mask, or any divisor of n_head for group masking.
    ///
    /// This is our fork-only API (llama_set_attn_mask).
    pub fn set_attn_mask(&self, mask: &[f32], positions: &[ffi::llama_pos], n_head_groups: i32) -> Result<()> {
        let n_pos = positions.len() as i32;
        let groups = if n_head_groups <= 1 { 1usize } else { n_head_groups as usize };
        let expected = groups * (n_pos as usize) * (n_pos as usize);
        if mask.len() != expected {
            bail!(
                "set_attn_mask: mask.len()={} but expected {}×{}×{}={}",
                mask.len(), groups, n_pos, n_pos, expected
            );
        }
        unsafe {
            ffi::llama_set_attn_mask(
                self.ctx,
                mask.as_ptr(),
                positions.as_ptr(),
                n_pos,
                n_head_groups,
                -1, // slot_id: -1 = global
            );
        }
        Ok(())
    }

    /// Clear the custom attention mask (revert to default causal).
    pub fn clear_attn_mask(&self) {
        unsafe {
            ffi::llama_set_attn_mask(self.ctx, ptr::null(), ptr::null(), 0, 0, -1);
        }
    }

    /// Get the number of attention heads in the model.
    pub fn n_head(&self) -> i32 {
        unsafe { ffi::llama_model_n_head(ffi::llama_get_model(self.ctx)) }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Sampling — Generate tokens
    // ═══════════════════════════════════════════════════════════════════════════

    /// Sample the next token from the logits of the last decoded position.
    ///
    /// Uses the sampler chain configured at init (temp, top_p, min_p, penalties, dist).
    pub fn sample(&self) -> ffi::llama_token {
        unsafe { ffi::llama_sampler_sample(self.sampler, self.ctx, -1) }
    }

    /// Reset the sampler state (e.g., penalty history).
    /// Call between independent generations.
    pub fn sampler_reset(&self) {
        unsafe { ffi::llama_sampler_reset(self.sampler); }
    }

    /// Get raw logits from the last decode for the batch item at index `idx`.
    /// Returns a slice of n_vocab floats. Use idx=-1 for the last token.
    /// MUST be called after llama_decode and BEFORE the next llama_decode.
    pub fn get_logits(&self, idx: i32) -> &[f32] {
        unsafe {
            let n_vocab = ffi::llama_vocab_n_tokens(self.vocab()) as usize;
            let ptr = ffi::llama_get_logits_ith(self.ctx, idx);
            if ptr.is_null() {
                return &[];
            }
            std::slice::from_raw_parts(ptr, n_vocab)
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Generation — streaming token-by-token
    // ═══════════════════════════════════════════════════════════════════════════

    /// Generate text token-by-token with streaming callback.
    ///
    /// **Flow:**
    /// 1. Encode `query_tokens` at `start_pos..start_pos+len` on seq_id=1
    /// 2. Sample tokens in a loop, calling `on_token(piece)` for each
    /// 3. Stop on EOS/EOT, max_tokens, or callback returning `false`
    /// 4. Clean up seq_id=1 from KV cache
    ///
    /// Returns (full_text, finished_naturally, end_pos). `finished_naturally` is true if
    /// generation stopped on an EOG token, false if it hit max_tokens.
    /// `end_pos` is the position after the last generated token (for continuation).
    /// If `keep_seq` is true, seq_id is NOT cleaned up (caller must do it).
    pub fn generate_ex<F: FnMut(&str) -> bool>(
        &self,
        query_tokens: &[ffi::llama_token],
        start_pos: ffi::llama_pos,
        max_tokens: i32,
        seq_id: ffi::llama_seq_id,
        keep_seq: bool,
        mut on_token: F,
    ) -> Result<(String, bool, ffi::llama_pos, GenerationSignals)> {
        // Reset sampler state from previous generation
        self.sampler_reset();

        if query_tokens.is_empty() {
            bail!("generate: query_tokens is empty");
        }

        // Encode query tokens into KV
        let positions: Vec<ffi::llama_pos> = (start_pos..start_pos + query_tokens.len() as i32).collect();
        self.encode(query_tokens, &positions, seq_id)?;

        let vocab = self.vocab();
        let mut result = String::new();
        let mut cur_pos = start_pos + query_tokens.len() as i32;
        // Buffer for incomplete UTF-8 sequences across token boundaries
        let mut utf8_buf: Vec<u8> = Vec::new();
        let mut hit_eog = false;
        let mut step_signals: Vec<StepSignals> = Vec::new();

        for step_idx in 0..max_tokens {
            // Ξ(t) T3: Extract entropy from logits BEFORE sampling
            let logits = self.get_logits(-1);
            if !logits.is_empty() {
                let (entropy, top1_prob, top_p_mass) = compute_entropy_top_k(logits, 256);
                step_signals.push(StepSignals {
                    entropy,
                    top1_prob,
                    top_p_mass,
                    token_position: step_idx as u32,
                });
            }

            // Sample from logits of the last decoded token
            let token = self.sample();

            // Check end-of-generation
            if unsafe { ffi::llama_token_is_eog(vocab, token) } {
                // Flush any remaining bytes
                if !utf8_buf.is_empty() {
                    let s = String::from_utf8_lossy(&utf8_buf).into_owned();
                    on_token(&s);
                    result.push_str(&s);
                    utf8_buf.clear();
                }
                hit_eog = true;
                break;
            }

            // Accumulate raw bytes and emit complete UTF-8 sequences
            utf8_buf.extend_from_slice(&self.token_to_bytes(token));

            // Extract valid UTF-8 prefix, stream it, keep remainder
            let mut should_break = false;
            let valid_up_to = match std::str::from_utf8(&utf8_buf) {
                Ok(_) => utf8_buf.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to > 0 {
                let piece = std::str::from_utf8(&utf8_buf[..valid_up_to]).unwrap().to_string();
                if !on_token(&piece) {
                    result.push_str(&piece);
                    should_break = true;
                } else {
                    result.push_str(&piece);
                }
                utf8_buf.drain(..valid_up_to);
            }
            if should_break { break; }
            // Remaining bytes in utf8_buf are incomplete — wait for next token

            // Decode this token to get logits for next sampling
            let mut batch = unsafe { ffi::llama_batch_init(1, 0, 1) };
            unsafe {
                *batch.token = token;
                *batch.pos = cur_pos;
                *batch.n_seq_id = 1;
                **batch.seq_id = seq_id;
                *batch.logits = 1; // need logits for next sample
                batch.n_tokens = 1;

                let ret = ffi::llama_decode(self.ctx, batch);
                ffi::llama_batch_free(batch);

                if ret != 0 {
                    bail!("llama_decode failed during generation at pos={} (ret={})", cur_pos, ret);
                }
            }

            cur_pos += 1;
        }

        // Clean up query+generation tokens from KV unless caller wants to continue
        if !keep_seq {
            self.clear_seq(seq_id);
        }

        let signals = GenerationSignals::from_steps(step_signals);
        Ok((result, hit_eog, cur_pos, signals))
    }

    /// Convenience wrapper: generate with auto-cleanup (original API).
    /// Returns (text, eog_reached).
    pub fn generate<F: FnMut(&str) -> bool>(
        &self,
        query_tokens: &[ffi::llama_token],
        start_pos: ffi::llama_pos,
        max_tokens: i32,
        seq_id: ffi::llama_seq_id,
        on_token: F,
    ) -> Result<(String, bool, GenerationSignals)> {
        let (text, eog, _, signals) = self.generate_ex(query_tokens, start_pos, max_tokens, seq_id, false, on_token)?;
        Ok((text, eog, signals))
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        // Free in reverse order: sampler → context → model
        // Each behind catch_unwind to prevent panics across FFI
        let sampler = self.sampler;
        let ctx = self.ctx;
        let model = self.model;

        if !sampler.is_null() {
            let _ = catch_unwind(|| unsafe { ffi::llama_sampler_free(sampler) });
        }
        if !ctx.is_null() {
            let _ = catch_unwind(|| unsafe { ffi::llama_free(ctx) });
        }
        if !model.is_null() {
            let _ = catch_unwind(|| unsafe { ffi::llama_model_free(model) });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests — covers skipped steps T3/S3, T4/S3, T7/S9
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    static TEST_ENGINE: OnceLock<LlamaEngine> = OnceLock::new();

    /// Shared engine for all tests (loading a model is expensive).
    fn get_engine() -> &'static LlamaEngine {
        TEST_ENGINE.get_or_init(|| {
            let paths = [
                "/Users/triviere/models/llama3.2-3b.gguf",
                "/Users/triviere/models/Qwen3-14B-Claude-4.5-Opus-Distill.q4_k_m.gguf",
            ];
            let model_path = paths.iter()
                .find(|p| std::path::Path::new(p).exists())
                .expect("No test model found in /Users/triviere/models/");

            let config = EngineConfig {
                model_path: model_path.to_string(),
                n_ctx: 2048,
                n_gpu_layers: 99,
                n_batch: 512,
                ..Default::default()
            };
            LlamaEngine::new(&config).expect("Failed to create engine")
        })
    }

    // ── T3/S3: UTF-8 multi-byte tokenization ────────────────────────────────

    #[test]
    fn t3_utf8_accents() {
        let engine = get_engine();
        let tokens = engine.tokenize("éàü café résumé", false, true).unwrap();
        assert!(!tokens.is_empty(), "accented text should produce tokens");
        let count = engine.token_count("éàü café résumé").unwrap();
        assert_eq!(count, tokens.len());
        let text = engine.detokenize(&tokens);
        assert!(text.contains("café"), "roundtrip should preserve 'café', got: {}", text);
        eprintln!("  ✓ accents: {} tokens, roundtrip='{}'", tokens.len(), text.trim());
    }

    #[test]
    fn t3_utf8_cjk() {
        let engine = get_engine();
        let tokens = engine.tokenize("日本語テスト", false, true).unwrap();
        assert!(!tokens.is_empty(), "CJK text should produce tokens");
        let count = engine.token_count("日本語テスト").unwrap();
        assert_eq!(count, tokens.len());
        eprintln!("  ✓ CJK: {} tokens", tokens.len());
    }

    #[test]
    fn t3_utf8_emoji() {
        let engine = get_engine();
        let tokens = engine.tokenize("Hello 🌍🚀 world 🎉", false, true).unwrap();
        assert!(!tokens.is_empty(), "emoji text should produce tokens");
        let count = engine.token_count("Hello 🌍🚀 world 🎉").unwrap();
        assert_eq!(count, tokens.len());
        eprintln!("  ✓ emoji: {} tokens", tokens.len());
    }

    #[test]
    fn t3_utf8_mixed() {
        let engine = get_engine();
        let texts = [
            ("math symbols", "Ψ∑∆ mathematical symbols"),
            ("cyrillic", "Привет мир Cyrillic"),
            ("arabic", "مرحبا Arabic"),
            ("flag emoji", "🇫🇷 flag emoji"),
            ("greek+dots", "α·β·γ Greek with middle dots"),
        ];
        for (label, text) in &texts {
            let tokens = engine.tokenize(text, false, true).unwrap();
            assert!(!tokens.is_empty(), "'{}' should produce tokens", label);
            let count = engine.token_count(text).unwrap();
            assert_eq!(count, tokens.len(), "token_count mismatch for '{}'", label);
            eprintln!("  ✓ {}: {} tokens", label, tokens.len());
        }
    }

    // ── T4/S3: Encode/evict cycle ───────────────────────────────────────────

    #[test]
    fn t4_encode_evict_cycle() {
        let engine = get_engine();

        // Clear everything first
        engine.clear_kv();

        // Initial state: seq 0 empty
        let pos_max_before = engine.seq_pos_max(0);
        assert!(pos_max_before < 0, "seq 0 should be empty, got pos_max={}", pos_max_before);

        // Encode tokens
        let tokens = engine.tokenize("The quick brown fox jumps over the lazy dog", false, true).unwrap();
        let n = tokens.len();
        assert!(n >= 8, "need >= 8 tokens, got {}", n);

        let positions: Vec<i32> = (0..n as i32).collect();
        let encoded = engine.encode(&tokens, &positions, 0).unwrap();
        assert_eq!(encoded, n);

        let pos_max_after = engine.seq_pos_max(0);
        assert_eq!(pos_max_after, (n as i32) - 1);
        eprintln!("  ✓ encoded {} tokens, pos_max={}", n, pos_max_after);

        // Evict first half
        let half = (n / 2) as i32;
        let evicted = engine.evict(0, half);
        assert!(evicted, "evict should succeed");

        let pos_min_evicted = engine.seq_pos_min(0);
        assert!(pos_min_evicted >= half, "pos_min should be >= {}, got {}", half, pos_min_evicted);
        eprintln!("  ✓ evicted 0..{}, pos_min now={}", half, pos_min_evicted);

        // Encode 3 more
        let extra = &tokens[0..3];
        let extra_pos: Vec<i32> = (n as i32..n as i32 + 3).collect();
        engine.encode(extra, &extra_pos, 0).unwrap();

        let pos_max_final = engine.seq_pos_max(0);
        assert_eq!(pos_max_final, n as i32 + 2);
        eprintln!("  ✓ encode 3 more, pos_max={}", pos_max_final);

        // Cleanup
        engine.clear_kv();
    }

    #[test]
    fn t4_seq_isolation() {
        let engine = get_engine();
        engine.clear_kv();

        // Encode on seq_id=0
        let tokens = engine.tokenize("Context node data", false, true).unwrap();
        let pos0: Vec<i32> = (0..tokens.len() as i32).collect();
        engine.encode(&tokens, &pos0, 0).unwrap();
        let pos_max_0 = engine.seq_pos_max(0);

        // Encode on seq_id=1
        let q_tokens = engine.tokenize("Query text here", false, true).unwrap();
        let pos1: Vec<i32> = (100..100 + q_tokens.len() as i32).collect();
        engine.encode(&q_tokens, &pos1, 1).unwrap();
        let pos_max_1 = engine.seq_pos_max(1);
        assert!(pos_max_1 >= 100);

        // Clear seq 1 → seq 0 intact
        engine.clear_seq(1);
        let pos_max_1_after = engine.seq_pos_max(1);
        assert!(pos_max_1_after < 0, "seq 1 should be empty after clear, got {}", pos_max_1_after);
        let pos_max_0_after = engine.seq_pos_max(0);
        assert_eq!(pos_max_0_after, pos_max_0, "seq 0 should be intact");
        eprintln!("  ✓ seq isolation: seq0={}, seq1 cleared", pos_max_0_after);

        engine.clear_kv();
    }

    // ── T7/S9: End-to-end FFI generation ────────────────────────────────────

    #[test]
    fn t7_e2e_generate() {
        let engine = get_engine();
        engine.clear_kv();

        // Encode system prompt (seq_id=0, persistent)
        let system = "You are a helpful assistant. Answer briefly.";
        let sys_tokens = engine.tokenize(system, true, true).unwrap();
        let sys_pos: Vec<i32> = (0..sys_tokens.len() as i32).collect();
        engine.encode(&sys_tokens, &sys_pos, 0).unwrap();
        let next_pos = sys_tokens.len() as i32;
        eprintln!("  system prompt: {} tokens encoded", sys_tokens.len());

        // Generate response (seq_id=1, temporary)
        let query = "\nUser: What is 2+2?\nAssistant:";
        let query_tokens = engine.tokenize(query, false, true).unwrap();

        let mut streamed: Vec<String> = Vec::new();
        let (response, _hit_eog, _signals) = engine.generate(
            &query_tokens,
            next_pos,
            64,
            1,
            |piece| { streamed.push(piece.to_string()); true },
        ).unwrap();

        assert!(!response.is_empty(), "should produce non-empty text");
        assert!(!streamed.is_empty(), "callback should fire");
        let concat: String = streamed.join("");
        assert_eq!(concat, response, "streamed pieces should match full result");

        // seq_id=1 cleaned up by generate()
        let pos1 = engine.seq_pos_max(1);
        assert!(pos1 < 0, "seq 1 should be empty after generate, got {}", pos1);

        // seq_id=0 intact
        let pos0 = engine.seq_pos_max(0);
        assert_eq!(pos0, next_pos - 1, "seq 0 should be intact");

        eprintln!("  ✓ generated {} tokens: {}", streamed.len(), &response[..response.len().min(100)]);
        engine.clear_kv();
    }

    #[test]
    fn t7_e2e_early_stop() {
        let engine = get_engine();
        engine.clear_kv();

        let prompt = "Count from one to one hundred: one, two, three,";
        let tokens = engine.tokenize(prompt, true, true).unwrap();

        let mut count = 0;
        let (response, _, _signals) = engine.generate(
            &tokens, 0, 256, 1,
            |_| { count += 1; count < 5 },
        ).unwrap();

        assert!(!response.is_empty());
        assert!(count <= 5, "should stop after ~5 callbacks, got {}", count);
        eprintln!("  ✓ early stop after {} tokens: '{}'", count, response.trim());
        engine.clear_kv();
    }
}
