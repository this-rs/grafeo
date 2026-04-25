//! Error types for the substrate crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SubstrateError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid magic — not a substrate file")]
    BadMagic,

    #[error("unsupported format version {0} (expected {1})")]
    UnsupportedVersion(u32, u32),

    #[error("schema CRC mismatch: claimed {claimed:#x}, actual {actual:#x}")]
    SchemaCrcMismatch { claimed: u32, actual: u32 },

    #[error("WAL encode failed: {0}")]
    WalEncode(String),

    #[error("WAL decode failed: {0}")]
    WalDecode(String),

    #[error("WAL short read: needed {needed} bytes, got {got}")]
    WalShortRead { needed: usize, got: usize },

    #[error("WAL frame invalid: {0}")]
    WalBadFrame(String),

    #[error("WAL CRC mismatch: claimed {claimed:#x}, actual {actual:#x}")]
    WalCrcMismatch { claimed: u32, actual: u32 },

    #[error("substrate file out of space (array capacity exhausted)")]
    OutOfSpace,

    #[error("node id out of bounds: {0}")]
    NodeOutOfBounds(u32),

    #[error("edge id out of bounds: {0}")]
    EdgeOutOfBounds(u64),

    /// Catch-all for in-crate invariants that don't deserve a bespoke
    /// variant. Used by higher-level components (e.g. the label registry)
    /// when they need to surface a structured error without widening the
    /// public error API.
    #[error("substrate invariant violated: {0}")]
    Internal(String),
}

pub type SubstrateResult<T> = Result<T, SubstrateError>;
