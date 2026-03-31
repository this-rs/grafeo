//! Safe Rust wrapper around llama.cpp FFI bindings.
//!
//! Uses our fork which provides `llama_set_attn_mask()` for topological attention masking.
//!
//! # Safety
//! All FFI calls are wrapped in safe methods. `catch_unwind` is used at FFI boundaries
//! to prevent Rust panics from crossing into C code (which would be UB).

use crate::ffi;
use crate::signals::{AdaptiveEntropyThreshold, GenerationSignals, StepSignals, compute_entropy_top_k};
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
    /// KV cache quantization type for keys (default: f16, use q8_0/q4_0 for TurboQuant)
    pub type_k: i32,
    /// KV cache quantization type for values (default: f16)
    pub type_v: i32,
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
            type_k: 1, // GGML_TYPE_F16
            type_v: 1, // GGML_TYPE_F16
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
    /// V cache is quantized (q8_0, q4_0, etc.) — requires flash attention.
    /// When true, features that force the non-flash path (state_bias) must be disabled.
    v_quantized: bool,
    /// If set, the profiler handle keeps the Arc alive for the eval callback.
    /// The raw pointer was passed to llama.cpp as cb_eval_user_data.
    _profiler_handle: Option<crate::profiler::ProfilerHandle>,
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
            // KV cache quantization (TurboQuant: rotation Hadamard activée quand quantized)
            params.type_k = std::mem::transmute(config.type_k);
            params.type_v = std::mem::transmute(config.type_v);
            // V quantized requires flash_attn at init (llama.cpp enforces this).
            // IMPORTANT: when V is quantized, features that force the non-flash path
            // (state_bias → ext_state_kq_b) must be disabled on the Rust side, because
            // the non-flash path cannot handle quantized V (wrong layout + no dequant → segfault).
            // build_attn_mha() has a safety net that forces flash even if bias is set.
            let v_quantized = config.type_v != 1; // 1 = GGML_TYPE_F16
            if v_quantized {
                params.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else {
                params.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }
            // Triple seq_id: 0 = persistent context, 1 = query+generation, 2 = ablation eval (B3)
            params.n_seq_max = 3;
            // Unified KV: all seq_ids share the full n_ctx pool.
            // Without this, kv_unified=false (default) splits n_ctx into n_seq_max streams
            // (e.g., 8192/3 = 2816 cells per stream), causing FAILED_PREPARE at pos=2816.
            params.kv_unified = true;
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
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_temp(config.temperature));

            // Top-P
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_p(config.top_p, 1));

            // Min-P
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_min_p(config.min_p, 1));

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
            v_quantized: config.type_v != 1, // 1 = GGML_TYPE_F16
            _profiler_handle: None,
        })
    }

    /// Load a model with an eval callback for head profiling.
    ///
    /// The profiler's eval callback captures "kq_soft_max" tensors during inference,
    /// allowing analysis of per-head attention distribution across topology banks.
    ///
    /// **Important**: Flash attention must be disabled (it's already disabled by default)
    /// because Flash Attention fuses the KQ computation and doesn't emit individual
    /// "kq_soft_max" tensors to the eval callback.
    pub fn new_with_profiler(
        config: &EngineConfig,
        profiler_handle: crate::profiler::ProfilerHandle,
    ) -> Result<Self> {
        ensure_backend();

        let c_path = CString::new(config.model_path.as_str())
            .map_err(|_| anyhow::anyhow!("model path contains null byte"))?;

        let model = unsafe {
            let mut params = ffi::llama_model_default_params();
            params.n_gpu_layers = config.n_gpu_layers;
            ffi::llama_model_load_from_file(c_path.as_ptr(), params)
        };
        if model.is_null() {
            bail!("Failed to load model from '{}'", config.model_path);
        }

        // Get raw pointer to the Mutex<ProfilerState> inside the Arc.
        // The Arc stays alive via _profiler_handle in the struct.
        let raw_handle = std::sync::Arc::as_ptr(&profiler_handle) as *mut std::os::raw::c_void;

        let ctx = unsafe {
            let mut params = ffi::llama_context_default_params();
            params.n_ctx = config.n_ctx;
            params.n_batch = config.n_batch;
            params.type_k = std::mem::transmute(config.type_k);
            params.type_v = std::mem::transmute(config.type_v);
            let v_quantized = config.type_v != 1;
            if v_quantized {
                params.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
            } else {
                params.flash_attn_type = ffi::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_DISABLED;
            }
            params.n_seq_max = 3;
            params.kv_unified = true;

            // Set eval callback for head profiling
            params.cb_eval = Some(crate::profiler::profiler_eval_callback);
            params.cb_eval_user_data = raw_handle;

            ffi::llama_init_from_model(model, params)
        };
        if ctx.is_null() {
            unsafe { ffi::llama_model_free(model) };
            bail!(
                "Failed to create context with profiler (n_ctx={})",
                config.n_ctx
            );
        }

        let sampler = unsafe {
            let sparams = ffi::llama_sampler_chain_default_params();
            let chain = ffi::llama_sampler_chain_init(sparams);
            if chain.is_null() {
                ffi::llama_free(ctx);
                ffi::llama_model_free(model);
                bail!("Failed to create sampler chain");
            }
            ffi::llama_sampler_chain_add(
                chain,
                ffi::llama_sampler_init_penalties(
                    config.penalty_last_n,
                    config.penalty_repeat,
                    0.0,
                    0.0,
                ),
            );
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_temp(config.temperature));
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_p(config.top_p, 1));
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_min_p(config.min_p, 1));
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
            v_quantized: config.type_v != 1,
            _profiler_handle: Some(profiler_handle),
        })
    }

    /// Get the actual context size (may differ from requested if model has a limit).
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Returns true if V cache is quantized (q8_0, q4_0, etc.).
    /// When true, flash attention is required — features that force the non-flash path
    /// (state_bias, per-head kq_b) will cause a segfault and must be disabled.
    pub fn v_is_quantized(&self) -> bool {
        self.v_quantized
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

    /// Get number of attention heads in the model.
    pub fn n_heads(&self) -> u32 {
        unsafe { ffi::llama_model_n_head(self.model) as u32 }
    }

    /// Get number of KV heads (GQA groups) in the model.
    pub fn n_heads_kv(&self) -> u32 {
        unsafe { ffi::llama_model_n_head_kv(self.model) as u32 }
    }

    /// Get number of layers in the model.
    pub fn n_layers(&self) -> u32 {
        unsafe { ffi::llama_model_n_layer(self.model) as u32 }
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
    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<ffi::llama_token>> {
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
            bail!(
                "tokenize failed after retry (text len={}, needed={})",
                text.len(),
                -n2
            );
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
                0,    // lstrip
                true, // special
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
    // Chat Template — Dynamic extraction from model GGUF metadata
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get the chat template embedded in the model's GGUF metadata.
    ///
    /// Returns `None` if the model has no chat template (rare for modern models).
    /// The template is a Jinja2 string, but `llama_chat_apply_template` handles
    /// the rendering internally for known template families.
    pub fn chat_template(&self) -> Option<String> {
        unsafe {
            let ptr = ffi::llama_model_chat_template(self.model, ptr::null());
            if ptr.is_null() {
                return None;
            }
            let cstr = std::ffi::CStr::from_ptr(ptr);
            cstr.to_str().ok().map(|s| s.to_owned())
        }
    }

    /// Format messages using the model's native chat template.
    ///
    /// Uses `llama_chat_apply_template` which supports a pre-defined list of
    /// template families (ChatML, Llama 3, Mistral, Gemma, etc.).
    ///
    /// - `messages`: slice of (role, content) pairs
    /// - `add_assistant`: if true, append the assistant turn prefix (for generation)
    ///
    /// Returns the formatted prompt string, or falls back to ChatML if the model
    /// template is not recognized.
    pub fn apply_chat_template(
        &self,
        messages: &[(&str, &str)],
        add_assistant: bool,
    ) -> Result<String> {
        // Build llama_chat_message array
        let c_roles: Vec<CString> = messages
            .iter()
            .map(|(role, _)| CString::new(*role).unwrap())
            .collect();
        let c_contents: Vec<CString> = messages
            .iter()
            .map(|(_, content)| CString::new(*content).unwrap())
            .collect();

        let chat_msgs: Vec<ffi::llama_chat_message> = c_roles
            .iter()
            .zip(c_contents.iter())
            .map(|(role, content)| ffi::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            })
            .collect();

        // Get model template (or NULL for auto-detect)
        let tmpl_str = self.chat_template();
        let tmpl_cstring = tmpl_str.as_ref().map(|s| CString::new(s.as_str()).unwrap());
        let tmpl_ptr = tmpl_cstring
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(ptr::null());

        // First call: get needed buffer size
        let needed = unsafe {
            ffi::llama_chat_apply_template(
                tmpl_ptr,
                chat_msgs.as_ptr(),
                chat_msgs.len(),
                add_assistant,
                ptr::null_mut(),
                0,
            )
        };

        if needed < 0 {
            // Template not recognized — fall back to ChatML
            return Ok(self.fallback_chatml(messages, add_assistant));
        }

        // Allocate buffer and render
        let buf_size = (needed + 1) as usize;
        let mut buf = vec![0u8; buf_size];

        let written = unsafe {
            ffi::llama_chat_apply_template(
                tmpl_ptr,
                chat_msgs.as_ptr(),
                chat_msgs.len(),
                add_assistant,
                buf.as_mut_ptr() as *mut i8,
                buf_size as i32,
            )
        };

        if written < 0 {
            bail!("llama_chat_apply_template failed on second call");
        }

        buf.truncate(written as usize);
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// Fallback to ChatML format if the model's template is not recognized.
    fn fallback_chatml(&self, messages: &[(&str, &str)], add_assistant: bool) -> String {
        let mut out = String::new();
        for (role, content) in messages {
            out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        if add_assistant {
            out.push_str("<|im_start|>assistant\n");
        }
        out
    }

    /// Format a single user query for generation, using the model's native template.
    ///
    /// Returns the "continuation" part to append after the system header in the KV cache:
    /// - system closing tag + user message + user closing tag + assistant opening tag
    ///
    /// For ChatML: `<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n`
    /// For Llama 3: `<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`
    pub fn format_user_turn(&self, query: &str) -> Result<String> {
        // Strategy: render [system=".", user=query] with add_assistant=true,
        // then strip the "open" system part (without closing tag) to get
        // the continuation that starts with the system closing tag.
        let messages_with_system = vec![
            ("system", "."),
            ("user", query),
        ];
        let full = self.apply_chat_template(&messages_with_system, true)?;

        // Get the "open" system part (without closing tag)
        let system_open = self.format_system_open(".")?;

        // The continuation starts right after the open system part,
        // which means it begins with the system closing tag — exactly what we need.
        if let Some(rest) = full.strip_prefix(&system_open) {
            Ok(rest.to_string())
        } else {
            // Fallback: return the full render
            Ok(full)
        }
    }

    /// Format the system header opening using the model's native template.
    ///
    /// Returns the system message **without** the closing tag, so that additional
    /// content (graph nodes) can be appended before the message is closed.
    ///
    /// For ChatML: `<|im_start|>system\n{content}`
    /// For Llama 3: `<|start_header_id|>system<|end_header_id|>\n\n{content}`
    pub fn format_system_header(&self, system_content: &str) -> Result<String> {
        let messages = vec![("system", system_content)];
        self.apply_chat_template(&messages, false)
    }

    /// Format the "open" system message (without closing tag).
    ///
    /// The closing tag is stripped so that graph node text can be appended
    /// inside the system message before the user turn closes it.
    pub fn format_system_open(&self, system_content: &str) -> Result<String> {
        // Render the full system message (with closing tag)
        let full = self.format_system_header(system_content)?;

        // Find the content in the rendered output and keep everything up to
        // (and including) the content — strip the closing tag after it.
        if let Some(pos) = full.rfind(system_content) {
            Ok(full[..pos + system_content.len()].to_string())
        } else {
            // Shouldn't happen — content should appear verbatim in the output
            Ok(full)
        }
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
    pub fn encode(
        &self,
        tokens: &[ffi::llama_token],
        positions: &[ffi::llama_pos],
        seq_id: ffi::llama_seq_id,
    ) -> Result<usize> {
        if tokens.len() != positions.len() {
            bail!(
                "encode: tokens.len()={} != positions.len()={}",
                tokens.len(),
                positions.len()
            );
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
                    bail!(
                        "llama_decode failed (ret={}, chunk {}/{})",
                        ret,
                        chunk_start / n_batch,
                        (tokens.len() + n_batch - 1) / n_batch
                    );
                }
            }
        }

        Ok(tokens.len())
    }

    /// Encode pre-computed embeddings directly into the KV cache (Phase C).
    ///
    /// Instead of passing token IDs through the model's embedding layer, this injects
    /// float vectors directly at layer 0 via `llama_batch.embd`.
    ///
    /// - `embeddings`: flat f32 slice of size `n_tokens × n_embd`, row-major.
    ///   Each consecutive `n_embd` floats represents one "virtual token".
    /// - `positions`: KV position for each virtual token (same length as n_tokens)
    /// - `seq_id`: sequence ID to assign
    ///
    /// Returns the number of virtual tokens encoded.
    ///
    /// # Panics / Errors
    /// - If `embeddings.len()` is not a multiple of `n_embd`
    /// - If `embeddings.len() / n_embd != positions.len()`
    /// - If `llama_decode` fails
    pub fn encode_embeddings(
        &self,
        embeddings: &[f32],
        positions: &[ffi::llama_pos],
        seq_id: ffi::llama_seq_id,
    ) -> Result<usize> {
        let n_embd = self.n_embd();
        if n_embd == 0 {
            bail!("encode_embeddings: n_embd is 0 — model not loaded?");
        }
        if embeddings.len() % n_embd != 0 {
            bail!(
                "encode_embeddings: embeddings.len()={} not divisible by n_embd={}",
                embeddings.len(),
                n_embd
            );
        }
        let n_tokens = embeddings.len() / n_embd;
        if n_tokens != positions.len() {
            bail!(
                "encode_embeddings: n_tokens={} (from embeddings) != positions.len()={}",
                n_tokens,
                positions.len()
            );
        }
        if n_tokens == 0 {
            return Ok(0);
        }

        let n_batch = unsafe { ffi::llama_n_batch(self.ctx) } as usize;

        for chunk_start in (0..n_tokens).step_by(n_batch) {
            let chunk_end = (chunk_start + n_batch).min(n_tokens);
            let chunk_len = (chunk_end - chunk_start) as i32;

            // llama_batch_init(n_tokens, embd, n_seq_max):
            // When embd > 0, batch.embd is allocated (n_tokens × embd × sizeof(float))
            // and batch.token is NULL.
            let mut batch = unsafe { ffi::llama_batch_init(chunk_len, n_embd as i32, 1) };

            unsafe {
                // Copy embedding vectors into batch.embd
                let embd_ptr = batch.embd;
                if embd_ptr.is_null() {
                    ffi::llama_batch_free(batch);
                    bail!(
                        "encode_embeddings: batch.embd is null after llama_batch_init — FFI error"
                    );
                }

                for i in 0..chunk_len as usize {
                    let idx = chunk_start + i;
                    // Copy n_embd floats for this virtual token
                    let src = &embeddings[idx * n_embd..(idx + 1) * n_embd];
                    std::ptr::copy_nonoverlapping(src.as_ptr(), embd_ptr.add(i * n_embd), n_embd);
                    *batch.pos.add(i) = positions[idx];
                    *batch.n_seq_id.add(i) = 1;
                    *(*batch.seq_id.add(i)) = seq_id;
                    // Only compute logits for the very last token of the very last chunk
                    *batch.logits.add(i) = if idx == n_tokens - 1 { 1 } else { 0 };
                }
                batch.n_tokens = chunk_len;

                let ret = ffi::llama_decode(self.ctx, batch);
                ffi::llama_batch_free(batch);

                if ret != 0 {
                    bail!(
                        "llama_decode (embd) failed (ret={}, chunk {}/{})",
                        ret,
                        chunk_start / n_batch,
                        (n_tokens + n_batch - 1) / n_batch
                    );
                }
            }
        }

        Ok(n_tokens)
    }

    /// Inject a single embedding vector at a specific KV cache position on seq_id 0.
    ///
    /// Convenience wrapper around `encode_embeddings` for self-embedding injection.
    /// The embedding must have exactly `n_embd` elements.
    ///
    /// # Safety notes
    /// - Uses `llama_batch_init(1, n_embd, 1)` with `batch.embd` — verifies not null
    /// - Position must be consecutive with existing KV entries (post llama_memory_* API)
    pub fn inject_embedding(&self, pos: ffi::llama_pos, embd: &[f32]) -> Result<()> {
        let n_embd = self.n_embd();
        if embd.len() != n_embd {
            bail!(
                "inject_embedding: embd.len()={} != n_embd={}",
                embd.len(),
                n_embd
            );
        }
        self.encode_embeddings(embd, &[pos], 0)?;
        Ok(())
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
        unsafe {
            ffi::llama_memory_seq_rm(mem, seq_id, -1, -1);
        }
    }

    /// Copy seq_id_src → seq_id_dst for positions [p0, p1).
    ///
    /// After this call, KV entries from seq_id_src are also visible to seq_id_dst.
    /// This is essential: tokens decoded on seq_id=1 can only attend to KV entries
    /// that include seq_id=1 in their sequence set. Without seq_cp, query tokens
    /// on seq_id=1 cannot see context nodes encoded on seq_id=0.
    pub fn seq_cp(
        &self,
        src: ffi::llama_seq_id,
        dst: ffi::llama_seq_id,
        p0: ffi::llama_pos,
        p1: ffi::llama_pos,
    ) {
        let mem = self.memory();
        unsafe {
            ffi::llama_memory_seq_cp(mem, src, dst, p0, p1);
        }
    }

    /// Clear the entire KV cache (all sequences).
    pub fn clear_kv(&self) {
        let mem = self.memory();
        unsafe {
            ffi::llama_memory_clear(mem, true);
        }
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
    pub fn set_attn_mask(
        &self,
        mask: &[f32],
        positions: &[ffi::llama_pos],
        n_head_groups: i32,
    ) -> Result<()> {
        let n_pos = positions.len() as i32;
        let groups = if n_head_groups <= 1 {
            1usize
        } else {
            n_head_groups as usize
        };
        let expected = groups * (n_pos as usize) * (n_pos as usize);
        if mask.len() != expected {
            bail!(
                "set_attn_mask: mask.len()={} but expected {}×{}×{}={}",
                mask.len(),
                groups,
                n_pos,
                n_pos,
                expected
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

    /// Set external state bias (additive kq_b) from SelfMetrics.
    ///
    /// The bias tensor is added to attention scores (kq) before softmax in
    /// `build_attn_mha()`. Layout: `[n_head × n_kv]` flattened row-major.
    /// Broadcasts across query tokens (all queries see the same bias).
    ///
    /// This is our fork-only API (llama_set_state_bias, Phase 5 Axe B).
    pub fn set_state_bias(&self, bias: &[f32], n_head: i32, n_kv: i32) {
        let expected = (n_head as usize) * (n_kv as usize);
        if bias.len() != expected {
            eprintln!(
                "  [StateBias] WARNING: bias.len()={} but expected n_head×n_kv={}×{}={}",
                bias.len(),
                n_head,
                n_kv,
                expected
            );
            return;
        }
        unsafe {
            ffi::llama_set_state_bias(self.ctx, bias.as_ptr(), n_head, n_kv);
        }
    }

    /// Clear the external state bias (revert to no bias).
    pub fn clear_state_bias(&self) {
        unsafe {
            ffi::llama_set_state_bias(self.ctx, ptr::null(), 0, 0);
        }
    }

    /// Get the number of attention heads in the model.
    pub fn n_head(&self) -> i32 {
        unsafe { ffi::llama_model_n_head(ffi::llama_get_model(self.ctx)) }
    }

    /// Get the number of KV heads (GQA groups) in the model.
    /// For GQA models (e.g. Qwen3-14B: 40 heads, 8 KV heads), this is much smaller
    /// than n_head. The attention mask only needs n_head_kv groups because heads in
    /// the same GQA group share the same KV cache.
    pub fn n_head_kv(&self) -> i32 {
        unsafe { ffi::llama_model_n_head_kv(ffi::llama_get_model(self.ctx)) }
    }

    /// Decode tokens on a given seq_id and return logits for the last token.
    ///
    /// Used for ablation reward: re-evaluate query under different masks to
    /// measure bank contribution via log-prob differential.
    /// Returns a Vec<f32> of n_vocab logits. Caller must manage seq_id cleanup.
    pub fn decode_for_logits(
        &self,
        tokens: &[ffi::llama_token],
        positions: &[ffi::llama_pos],
        seq_id: ffi::llama_seq_id,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            bail!("decode_for_logits: empty tokens");
        }
        if tokens.len() != positions.len() {
            bail!(
                "decode_for_logits: tokens.len()={} != positions.len()={}",
                tokens.len(),
                positions.len()
            );
        }

        let n_batch = unsafe { ffi::llama_n_batch(self.ctx) } as usize;
        let total = tokens.len();

        for chunk_start in (0..total).step_by(n_batch) {
            let chunk_end = (chunk_start + n_batch).min(total);
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
                    *batch.logits.add(i) = if idx == total - 1 { 1 } else { 0 };
                }
                batch.n_tokens = chunk_len;

                let ret = ffi::llama_decode(self.ctx, batch);
                ffi::llama_batch_free(batch);

                if ret != 0 {
                    bail!("decode_for_logits: llama_decode failed (ret={})", ret);
                }
            }
        }

        // Copy logits (they're invalidated by next decode)
        let logits = self.get_logits(-1);
        Ok(logits.to_vec())
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
        unsafe {
            ffi::llama_sampler_reset(self.sampler);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Embeddings — for PersistNet and semantic representation
    // ═══════════════════════════════════════════════════════════════════════════

    /// Enable/disable embedding output from decode calls.
    /// When enabled, `get_embedding()` returns the last-layer hidden state.
    /// Must be called BEFORE the decode/encode that should produce embeddings.
    pub fn set_embeddings(&self, enable: bool) {
        unsafe {
            ffi::llama_set_embeddings(self.ctx, enable);
        }
    }

    /// Get the model's embedding dimension (n_embd).
    pub fn n_embd(&self) -> usize {
        unsafe { ffi::llama_model_n_embd(self.model) as usize }
    }

    /// Get the embedding (last-layer hidden state) for the ith token in the last batch.
    /// Use idx=-1 for the last token. Returns empty slice if embeddings not enabled.
    /// MUST be called after decode/encode and BEFORE the next one.
    pub fn get_embedding(&self, idx: i32) -> &[f32] {
        unsafe {
            let ptr = ffi::llama_get_embeddings_ith(self.ctx, idx);
            if ptr.is_null() {
                return &[];
            }
            let n_embd = self.n_embd();
            std::slice::from_raw_parts(ptr, n_embd)
        }
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

    /// Apply additive logit biases to the last decode output.
    ///
    /// Modifies the logit buffer in place. Call AFTER `llama_decode` and BEFORE
    /// `sample()`. The biases map token_id → additive bias (positive = more likely).
    ///
    /// Used by IPTR to steer generation toward graph-retrieved concepts.
    pub fn apply_logit_biases(&self, biases: &std::collections::HashMap<u32, f32>) {
        if biases.is_empty() {
            return;
        }
        unsafe {
            let n_vocab = ffi::llama_vocab_n_tokens(self.vocab()) as usize;
            let ptr = ffi::llama_get_logits_ith(self.ctx, -1);
            if ptr.is_null() {
                return;
            }
            for (&token_id, &bias) in biases {
                let tid = token_id as usize;
                if tid < n_vocab {
                    *ptr.add(tid) += bias;
                }
            }
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
    ///
    /// `on_needs_tool`: optional IPTR callback. Called when entropy exceeds the adaptive
    /// threshold. Receives (entropy, recent_text, token_position) and returns logit biases
    /// to apply before the next sampling step.
    pub fn generate_ex<F: FnMut(&str) -> bool>(
        &self,
        query_tokens: &[ffi::llama_token],
        start_pos: ffi::llama_pos,
        max_tokens: i32,
        seq_id: ffi::llama_seq_id,
        keep_seq: bool,
        mut on_token: F,
        mut on_needs_tool: Option<&mut dyn FnMut(f32, &str, u32) -> Option<std::collections::HashMap<u32, f32>>>,
    ) -> Result<(String, bool, ffi::llama_pos, GenerationSignals)> {
        // Reset sampler state from previous generation
        self.sampler_reset();

        if query_tokens.is_empty() {
            bail!("generate: query_tokens is empty");
        }

        // Encode query tokens into KV
        let positions: Vec<ffi::llama_pos> =
            (start_pos..start_pos + query_tokens.len() as i32).collect();
        self.encode(query_tokens, &positions, seq_id)?;

        let vocab = self.vocab();
        let mut result = String::new();
        let mut cur_pos = start_pos + query_tokens.len() as i32;
        // Buffer for incomplete UTF-8 sequences across token boundaries
        let mut utf8_buf: Vec<u8> = Vec::new();
        let mut hit_eog = false;
        let mut step_signals: Vec<StepSignals> = Vec::new();
        let mut first_token_id: Option<i32> = None;
        let mut first_step_logits: Option<Vec<f32>> = None;
        let mut entropy_threshold = AdaptiveEntropyThreshold::new();

        for step_idx in 0..max_tokens {
            // Ξ(t) T3: Extract entropy from logits BEFORE sampling
            let logits = self.get_logits(-1);
            if !logits.is_empty() {
                let (entropy, top1_prob, top_p_mass) = compute_entropy_top_k(logits, 256);

                // IPTR: check adaptive threshold then update window
                let needs_tool = entropy_threshold.is_high(entropy);
                entropy_threshold.update(entropy);

                step_signals.push(StepSignals {
                    entropy,
                    top1_prob,
                    top_p_mass,
                    token_position: step_idx as u32,
                    needs_tool,
                });

                // B3: Capture full logits at step 0 for ablation reward
                if step_idx == 0 {
                    first_step_logits = Some(logits.to_vec());
                }

                // IPTR: dispatch tools on high entropy, apply logit biases
                if needs_tool {
                    if let Some(ref mut cb) = on_needs_tool {
                        // Build recent_text from last ~200 bytes of result (UTF-8 safe)
                        let recent = if result.len() > 200 {
                            let mut start = result.len() - 200;
                            // Walk forward to find a valid UTF-8 char boundary
                            while !result.is_char_boundary(start) && start < result.len() {
                                start += 1;
                            }
                            &result[start..]
                        } else {
                            &result
                        };
                        if let Some(biases) = cb(entropy, recent, step_idx as u32) {
                            if !biases.is_empty() {
                                kv_registry::kv_debug!(
                                    "  [IPTR] logit biases applied: {} tokens, entropy={:.2}",
                                    biases.len(),
                                    entropy
                                );
                                self.apply_logit_biases(&biases);
                            }
                        }
                    }
                }
            }

            // Sample from logits of the last decoded token
            let token = self.sample();

            // Capture first generated token for ablation reward (B3)
            if step_idx == 0 {
                first_token_id = Some(token);
            }

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
                let piece = std::str::from_utf8(&utf8_buf[..valid_up_to])
                    .unwrap()
                    .to_string();
                if !on_token(&piece) {
                    result.push_str(&piece);
                    should_break = true;
                } else {
                    result.push_str(&piece);
                }
                utf8_buf.drain(..valid_up_to);
            }
            if should_break {
                break;
            }
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
                    bail!(
                        "llama_decode failed during generation at pos={} (ret={})",
                        cur_pos,
                        ret
                    );
                }
            }

            cur_pos += 1;
        }

        // Clean up query+generation tokens from KV unless caller wants to continue
        if !keep_seq {
            self.clear_seq(seq_id);
        }

        let mut signals = GenerationSignals::from_steps(step_signals);
        signals.first_token_id = first_token_id;
        signals.first_step_logits = first_step_logits;
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
        let (text, eog, _, signals) =
            self.generate_ex(query_tokens, start_pos, max_tokens, seq_id, false, on_token, None)?;
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
            let model_path = paths
                .iter()
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
        assert!(
            text.contains("café"),
            "roundtrip should preserve 'café', got: {}",
            text
        );
        eprintln!(
            "  ✓ accents: {} tokens, roundtrip='{}'",
            tokens.len(),
            text.trim()
        );
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
        assert!(
            pos_max_before < 0,
            "seq 0 should be empty, got pos_max={}",
            pos_max_before
        );

        // Encode tokens
        let tokens = engine
            .tokenize("The quick brown fox jumps over the lazy dog", false, true)
            .unwrap();
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
        assert!(
            pos_min_evicted >= half,
            "pos_min should be >= {}, got {}",
            half,
            pos_min_evicted
        );
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
        assert!(
            pos_max_1_after < 0,
            "seq 1 should be empty after clear, got {}",
            pos_max_1_after
        );
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
        let (response, _hit_eog, _signals) = engine
            .generate(&query_tokens, next_pos, 64, 1, |piece| {
                streamed.push(piece.to_string());
                true
            })
            .unwrap();

        assert!(!response.is_empty(), "should produce non-empty text");
        assert!(!streamed.is_empty(), "callback should fire");
        let concat: String = streamed.join("");
        assert_eq!(concat, response, "streamed pieces should match full result");

        // seq_id=1 cleaned up by generate()
        let pos1 = engine.seq_pos_max(1);
        assert!(
            pos1 < 0,
            "seq 1 should be empty after generate, got {}",
            pos1
        );

        // seq_id=0 intact
        let pos0 = engine.seq_pos_max(0);
        assert_eq!(pos0, next_pos - 1, "seq 0 should be intact");

        eprintln!(
            "  ✓ generated {} tokens: {}",
            streamed.len(),
            &response[..response.len().min(100)]
        );
        engine.clear_kv();
    }

    #[test]
    fn t7_e2e_early_stop() {
        let engine = get_engine();
        engine.clear_kv();

        let prompt = "Count from one to one hundred: one, two, three,";
        let tokens = engine.tokenize(prompt, true, true).unwrap();

        let mut count = 0;
        let (response, _, _signals) = engine
            .generate(&tokens, 0, 256, 1, |_| {
                count += 1;
                count < 5
            })
            .unwrap();

        assert!(!response.is_empty());
        assert!(count <= 5, "should stop after ~5 callbacks, got {}", count);
        eprintln!(
            "  ✓ early stop after {} tokens: '{}'",
            count,
            response.trim()
        );
        engine.clear_kv();
    }

    // ── C1/SC1.3: Embedding injection round-trip ────────────────────────────

    #[test]
    fn c1_embed_roundtrip() {
        let engine = get_engine();
        engine.clear_kv();

        // Step 1: Encode "Hello world" as tokens → extract hidden state
        let text = "Hello world";
        let tokens = engine.tokenize(text, false, true).unwrap();
        let n = tokens.len();
        let positions_a: Vec<i32> = (0..n as i32).collect();

        engine.set_embeddings(true);
        engine.encode(&tokens, &positions_a, 0).unwrap();

        // Extract hidden state of the last token
        let hidden = engine.get_embedding(-1).to_vec();
        let n_embd = engine.n_embd();
        assert_eq!(
            hidden.len(),
            n_embd,
            "hidden state should be n_embd={}",
            n_embd
        );
        assert!(
            hidden.iter().any(|&v| v != 0.0),
            "hidden state should be non-zero"
        );

        // Get logits from token-encoded path
        let logits_token = engine.get_logits(-1).to_vec();
        assert!(!logits_token.is_empty());
        engine.set_embeddings(false);

        // Step 2: Clear KV, re-encode the hidden state directly via encode_embeddings
        engine.clear_kv();

        // We encode the hidden state as a single virtual token at position 0
        // Note: this is a simplified test — in production, each token would have its own embedding.
        // Here we inject the last hidden state and check that logits are coherent.
        let embd_positions: Vec<i32> = vec![0];
        engine
            .encode_embeddings(&hidden, &embd_positions, 0)
            .unwrap();

        let logits_embd = engine.get_logits(-1).to_vec();
        assert!(!logits_embd.is_empty());
        assert_eq!(logits_embd.len(), logits_token.len());

        // Check logits are valid (no NaN/Inf)
        for (i, &v) in logits_embd.iter().enumerate() {
            assert!(v.is_finite(), "logits_embd[{}] is not finite: {}", i, v);
        }

        // Cosine similarity between logits from token path vs embedding path
        // Note: because we're injecting the last hidden state (not all tokens),
        // the logits won't be identical but should show correlation.
        let dot: f64 = logits_token
            .iter()
            .zip(logits_embd.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm_a: f64 = logits_token
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm_b: f64 = logits_embd
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };

        eprintln!(
            "  ✓ embed roundtrip: cosine(logits_token, logits_embd) = {:.4}",
            cosine
        );
        // Relaxed threshold: injecting just the last hidden state is NOT equivalent
        // to the full token sequence, but logits should be in a similar space.
        assert!(
            cosine > 0.3,
            "cosine similarity too low: {:.4} (expected > 0.3)",
            cosine
        );

        engine.clear_kv();
    }

    // ── C1/SC1.4: Token+embedding coexistence ───────────────────────────────

    #[test]
    fn c1_token_embd_coexistence() {
        let engine = get_engine();
        engine.clear_kv();

        let n_embd = engine.n_embd();

        // Step 1: Encode 3 tokens via encode() at positions 0,1,2
        let tokens = engine.tokenize("The quick brown", false, true).unwrap();
        let n_tok = tokens.len().min(3);
        let tok_slice = &tokens[..n_tok];
        let tok_pos: Vec<i32> = (0..n_tok as i32).collect();
        engine.encode(tok_slice, &tok_pos, 0).unwrap();
        let pos_after_tokens = engine.seq_pos_max(0);
        eprintln!("  encoded {} tokens, pos_max={}", n_tok, pos_after_tokens);

        // Step 2: Inject 1 embedding at position (n_tok)
        // Use a random-ish but valid embedding (small random values)
        let mut fake_embd = vec![0.0f32; n_embd];
        for (i, v) in fake_embd.iter_mut().enumerate() {
            *v = ((i as f32 * 0.01).sin()) * 0.1; // deterministic, small values
        }
        let embd_pos = vec![n_tok as i32];
        engine.encode_embeddings(&fake_embd, &embd_pos, 0).unwrap();
        let pos_after_embd = engine.seq_pos_max(0);
        assert_eq!(
            pos_after_embd, n_tok as i32,
            "pos_max should include embd position"
        );
        eprintln!(
            "  injected embedding at pos={}, pos_max={}",
            n_tok, pos_after_embd
        );

        // Step 3: Encode 2 more tokens after the embedding
        let more_tokens = engine.tokenize("fox jumps", false, true).unwrap();
        let n_more = more_tokens.len().min(2);
        let more_slice = &more_tokens[..n_more];
        let more_pos: Vec<i32> = (n_tok as i32 + 1..n_tok as i32 + 1 + n_more as i32).collect();
        engine.encode(more_slice, &more_pos, 0).unwrap();

        // Step 4: Verify KV cache is coherent
        let final_pos = engine.seq_pos_max(0);
        assert_eq!(
            final_pos,
            n_tok as i32 + n_more as i32,
            "final pos_max should be correct"
        );

        // Step 5: Get logits — should be valid (not NaN/Inf)
        let logits = engine.get_logits(-1);
        assert!(!logits.is_empty(), "logits should be non-empty");
        let all_finite = logits.iter().all(|v| v.is_finite());
        assert!(
            all_finite,
            "all logits should be finite after mixed token+embd encoding"
        );

        eprintln!(
            "  ✓ coexistence: {} tokens + 1 embd + {} tokens = pos_max {}, logits OK",
            n_tok, n_more, final_pos
        );

        engine.clear_kv();
    }
}
