use anyhow::Result;
use llm_engine::LlamaEngine;
use kv_registry::Tokenizer;

/// Newtype wrapper to implement Tokenizer for LlamaEngine (orphan rule).
pub struct Engine(pub LlamaEngine);

impl std::ops::Deref for Engine {
    type Target = LlamaEngine;
    fn deref(&self) -> &LlamaEngine { &self.0 }
}

impl Tokenizer for Engine {
    fn tokenize(&self, text: &str, add_bos: bool, add_special: bool) -> Result<Vec<i32>> {
        self.0.tokenize(text, add_bos, add_special)
    }
    fn encode(&self, tokens: &[i32], positions: &[i32], seq_id: i32) -> Result<()> {
        self.0.encode(tokens, positions, seq_id).map(|_| ())
    }
    fn token_count(&self, text: &str) -> Result<usize> {
        self.0.token_count(text)
    }
    fn evict(&self, start: i32, end: i32) {
        self.0.evict(start, end);
    }
    fn clear_kv(&self) {
        self.0.clear_kv();
    }
    fn n_ctx(&self) -> u32 {
        self.0.n_ctx()
    }
    fn seq_pos_max(&self, seq_id: i32) -> i32 {
        self.0.seq_pos_max(seq_id)
    }
}
