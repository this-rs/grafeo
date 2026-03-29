//! KV Registry — manages graph node ↔ KV cache slot mappings,
//! semantic banks, and conversation fragments.

pub mod tokenizer;
pub mod registry;
pub mod banks;
pub mod context;
pub mod conv;
pub mod hilbert;
pub mod hilbert_bank;

// Re-exports for convenience
pub use tokenizer::Tokenizer;
pub use registry::{KvNodeRegistry, KvSlot, KvSlotMode, KvMetrics, KvTier, TierBudget};
pub use hilbert::{HilbertLayout, WeightedAdjacencyList, build_fused_adjacency, spectral_embedding_2d_weighted};
pub use hilbert_bank::{HilbertBank, BankManager};
pub use banks::{KvBank, load_bank_cache, save_bank_cache, discover_banks};
pub use context::{ContextNode, QueryContext};
pub use conv::{ConvFragments, ConvFragment, CONV_NODE_BASE};
