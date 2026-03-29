//! KV Registry — manages graph node ↔ KV cache slot mappings,
//! semantic banks, and conversation fragments.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global debug flag — set from main via `kv_registry::set_debug(true)`.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Enable or disable debug logging for kv-registry and downstream crates.
pub fn set_debug(enabled: bool) {
    DEBUG.store(enabled, Ordering::Relaxed);
}

/// Check if debug logging is enabled.
#[inline]
pub fn is_debug() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

/// Print to stderr only if debug mode is enabled.
#[macro_export]
macro_rules! kv_debug {
    ($($arg:tt)*) => {
        if $crate::is_debug() {
            eprintln!($($arg)*);
        }
    };
}

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
pub use conv::{ConvFragments, ConvFragment, CONV_NODE_BASE, ColdSearch, ColdHit};
