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

pub mod banks;
pub mod context;
pub mod conv;
pub mod hilbert;
pub mod hilbert_bank;
pub mod registry;
pub mod tokenizer;

// Re-exports for convenience
pub use banks::{KvBank, discover_banks, load_bank_cache, save_bank_cache};
pub use context::{ContextNode, QueryContext};
pub use conv::{CONV_NODE_BASE, ColdHit, ColdSearch, ConvFragment, ConvFragments};
pub use hilbert::{
    HilbertLayout, WeightedAdjacencyList, build_fused_adjacency, spectral_embedding_2d_weighted,
};
pub use hilbert_bank::{BankManager, HilbertBank};
pub use registry::{KvMetrics, KvNodeRegistry, KvSlot, KvSlotMode, KvTier, TierBudget};
pub use tokenizer::Tokenizer;
