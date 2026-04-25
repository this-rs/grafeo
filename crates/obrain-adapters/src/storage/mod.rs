//! Storage backends - how your data gets persisted.
//!
//! | Backend | Speed | Durability | Use when |
//! | ------- | ----- | ---------- | -------- |
//! | `wal` (feature-gated) | Fast | Survives crashes | Production workloads |
//! | `file` (feature-gated) | Fast | Crash-safe `.obrain` single-file | Shipping data snapshots |
//!
//! The WAL (Write-Ahead Log) writes changes to disk before applying them,
//! so you can recover after crashes without losing committed transactions.
//! The WAL module requires filesystem I/O and is gated behind the `wal` feature.
//!
//! The `file` module (requires `obrain-file` feature) implements a single-file
//! `.obrain` format with dual-header crash safety and sidecar WAL.
//!
//! > **Note (T17 W4.p4)**: the historical in-memory `MemoryBackend` (a thin
//! > `Arc<LpgStore>` wrapper) was removed as part of the substrate cutover.
//! > For in-memory / test usage, construct a `SubstrateStore` directly via
//! > `SubstrateStore::open_tempfile()` from the `obrain-substrate` crate.

#[cfg(feature = "obrain-file")]
pub mod file;
#[cfg(feature = "wal")]
pub mod wal;

#[cfg(feature = "obrain-file")]
pub use file::ObrainFileManager;
#[cfg(feature = "wal")]
pub use wal::WalManager;
