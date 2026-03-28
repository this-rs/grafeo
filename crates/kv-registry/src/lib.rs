//! KV Registry — manages graph node ↔ KV cache slot mappings,
//! semantic banks, and conversation fragments.

pub mod tokenizer;
pub mod registry;
pub mod banks;
pub mod context;
pub mod conv;

// Re-exports for convenience
pub use tokenizer::Tokenizer;
pub use registry::{KvNodeRegistry, KvSlot, KvMetrics};
pub use banks::{KvBank, load_bank_cache, save_bank_cache, discover_banks};
pub use context::{ContextNode, QueryContext};
pub use conv::{ConvFragments, ConvFragment, CONV_NODE_BASE};
