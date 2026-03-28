//! Tokenizer trait — abstracts LlamaEngine for testability.

use anyhow::Result;

/// Abstraction over the LLM engine's tokenization and KV cache operations.
pub trait Tokenizer {
    fn tokenize(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<i32>>;
    fn encode(&self, tokens: &[i32], positions: &[i32], seq_id: i32) -> Result<()>;
    fn token_count(&self, text: &str) -> Result<usize>;
    fn evict(&self, start: i32, end: i32);
    fn clear_kv(&self);
    fn n_ctx(&self) -> u32;
    fn seq_pos_max(&self, seq_id: i32) -> i32;
}
