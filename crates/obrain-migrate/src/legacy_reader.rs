//! # Legacy `.obrain` reader — directory and single-file dispatch
//!
//! Opens a legacy `.obrain` input in read-only mode and returns an
//! `Arc<dyn GraphStore>` consumable by [`crate::converter`]. Two layouts
//! are supported:
//!
//! * **Directory layout** — epoch files persisted by `LpgStore` via the
//!   `tiered-storage` feature. Hydrated via
//!   [`LpgStore::restore_from_epoch_files`].
//! * **Single-file `.obrain`** (`GRAF` magic) — the on-disk format used by
//!   `ObrainDB::open` and shipped with e.g. `wikipedia.obrain`. Opened
//!   through `obrain-engine::ObrainDB::open_read_only`.
//!
//! The dispatch is done on filesystem type: `path.is_dir()` routes to the
//! LpgStore path; `path.is_file()` routes to the single-file path (gated
//! on the `single-file-read` feature).
//!
//! ## Why `ObrainDB` rather than re-implementing the format here
//!
//! The single-file format has two on-disk variants:
//! - **v1 (legacy bincode)** — the snapshot is bincode-decoded into an
//!   `LpgStore`. `apply_snapshot_data` handles this in obrain-engine.
//! - **v2 (native mmap)** — `MmapStore::from_mmap` exposes the mmap'd
//!   arrays directly as a zero-copy `GraphStore`. Open is instant;
//!   iteration reads through the page cache instead of allocating.
//!
//! Both paths are already baked into `ObrainDB::open_read_only`. Mirroring
//! them here would duplicate version-sensitive code for no benefit. The
//! converter only needs the trait surface (`all_node_ids`, `get_node`,
//! `edges_from`, `get_node_property`, `edge_type`), all of which
//! `MmapStore` and `LpgStore` implement.
//!
//! ## Keep-alive
//!
//! When we open a single-file DB we hold on to the `ObrainDB` via
//! `std::mem::forget` for the process lifetime. This keeps the mmap /
//! exclusive file lock alive until the migration exits. Safe because
//! `obrain-migrate` is a one-shot CLI — the leak is bounded by process
//! termination.
//!
//! # Contract
//!
//! - This module exists **only** inside `obrain-migrate`. Nothing else in
//!   the obrain ecosystem should depend on the legacy reader after the
//!   T17 cut-over; the whole dependency chain retires together with the
//!   binary.
//! - The returned store is treated as read-only by the migrator. Writes
//!   are technically still possible on the `LpgStore` variant but
//!   `converter.rs` never calls any mutation on it. The `MmapStore`
//!   variant is genuinely immutable by construction.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::traits::GraphStore;

/// Opens a legacy `.obrain` input and returns its graph store.
///
/// Dispatches on the filesystem type:
/// * directory → `LpgStore` hydrated from epoch files (`tiered-storage`).
/// * regular file → single-file `.obrain` (requires the
///   `single-file-read` feature).
pub fn open_legacy(path: &Path) -> Result<Arc<dyn GraphStore>> {
    if !path.exists() {
        anyhow::bail!("legacy input does not exist: {}", path.display());
    }

    if path.is_dir() {
        open_legacy_dir(path)
    } else if path.is_file() {
        open_legacy_file(path)
    } else {
        anyhow::bail!(
            "legacy input is neither a regular file nor a directory: {}",
            path.display()
        );
    }
}

/// Opens a directory-layout `.obrain` (LpgStore + epoch files).
fn open_legacy_dir(path: &Path) -> Result<Arc<dyn GraphStore>> {
    let store =
        LpgStore::new().context("failed to construct LpgStore for legacy directory read")?;

    // Wire the persist dir so the epoch store can mmap the files.
    store
        .set_persist_dir(path.to_path_buf())
        .with_context(|| format!("set_persist_dir failed for {}", path.display()))?;

    // Rebuild the in-memory indexes from the mmap'd epoch files.
    store
        .restore_from_epoch_files()
        .with_context(|| format!("restore_from_epoch_files failed for {}", path.display()))?;

    tracing::info!(
        path = %path.display(),
        "opened legacy `.obrain` directory (LpgStore hydrated from epoch files)",
    );
    Ok(Arc::new(store) as Arc<dyn GraphStore>)
}

/// T17 final cutover (2026-04-23): single-file `.obrain` (`GRAF`-magic)
/// format is no longer readable. The pre-substrate v1/v2 code paths relied
/// on obrain-engine's `obrain-file` feature + the LpgStore module, both of
/// which are being retired. Users with a pre-substrate single-file
/// database must use obrain-migrate ≤ v0.0.1 to convert to the
/// directory-based LpgStore format first, then run this version to
/// convert to substrate. (The intermediate directory format is still
/// readable via the `legacy-read` feature.)
fn open_legacy_file(path: &Path) -> Result<Arc<dyn GraphStore>> {
    anyhow::bail!(
        "legacy input is a single-file `.obrain` ({}) — this format is \
         no longer supported since T17 final cutover. Use \
         obrain-migrate ≤ v0.0.1 to convert to the directory-based \
         LpgStore format first, then re-run this version to convert to \
         substrate.",
        path.display()
    )
}
