//! # `strip_skipped_props` — one-shot cleanup of `substrate.props`
//!
//! Removes every `(key, value)` pair whose key appears in
//! [`SKIP_ON_LOAD_PROP_KEYS`](obrain_substrate::SKIP_ON_LOAD_PROP_KEYS)
//! from the `substrate.props` sidecar, then atomically re-persists the
//! snapshot. Zones (`substrate.nodes`, `.edges`, `.tier*`, WAL, dict,
//! meta) are **not touched**.
//!
//! ## Why
//!
//! T16 post-mortem identified that `SubstrateStore::from_substrate`
//! inflates anon-RSS by `file_size × ~7` when `substrate.props` contains
//! large `Vector` values — on wiki-scale corpora, `_st_embedding`
//! (3.94 M vectors × 1.5 KB) alone pushes open-time anon-RSS from ~1 GiB
//! to ~57 GiB. The hub runtime never reads `_st_embedding` via
//! `get_node_property` — the tier zones are the sole runtime accessor —
//! so keeping the vectors in `substrate.props` is pure duplication.
//!
//! Filtering at load time only partially helps (the bincode decode
//! already allocates the `Arc<[f32]>` values), so we strip them from
//! the sidecar itself. One-shot: future migrations (post-T16.x) won't
//! write these keys in the first place.
//!
//! ## Safety
//!
//! * Writes atomically via `persist()` (tmp file + rename). A crash
//!   mid-write leaves either the old or the new file, never a torn one.
//! * A `.pre-strip` APFS clonefile is created before rewriting so a
//!   manual rollback is `mv substrate.props.pre-strip substrate.props`.
//! * Zones and WAL are untouched — reopening an unstripped base still
//!   works, just with the old RSS profile.
//!
//! ## Usage
//!
//! ```text
//! # Strip the default SKIP_ON_LOAD_PROP_KEYS set:
//! cargo run --release -p obrain-substrate --example strip_skipped_props -- \
//!     /Users/me/.obrain/db/po
//!
//! # Strip a custom set of keys (comma-separated, overrides default):
//! cargo run --release -p obrain-substrate --example strip_skipped_props -- \
//!     /Users/me/.obrain/db/po \
//!     --keys _st_embedding,_kernel_embedding,_hilbert_features
//! ```
//!
//! Optional flag `--dry-run` reports what would be stripped without
//! touching the file. Position-independent: `--dry-run` and
//! `--keys <csv>` can appear in any order after the path.

use std::path::PathBuf;

use obrain_substrate::{PropertiesSnapshotV1, SKIP_ON_LOAD_PROP_KEYS};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

fn resolve_substrate_dir(base: &std::path::Path) -> PathBuf {
    let nested = base.join("substrate.obrain");
    if nested.join("substrate.meta").exists() {
        return nested;
    }
    base.to_path_buf()
}

fn main() -> Result<(), BoxErr> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: strip_skipped_props <substrate-dir> [--dry-run] [--keys k1,k2,...]".into());
    }
    let base: PathBuf = PathBuf::from(args.remove(0));

    // Parse flags in any order.
    let mut dry_run = false;
    let mut custom_keys: Option<Vec<String>> = None;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--keys" => {
                let csv = iter.next().ok_or_else(|| -> BoxErr {
                    "--keys requires a comma-separated value".into()
                })?;
                custom_keys = Some(
                    csv.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                );
            }
            other => {
                return Err(format!("unknown arg: {other}").into());
            }
        }
    }

    // Build the effective skip set.
    let default_keys: Vec<String> =
        SKIP_ON_LOAD_PROP_KEYS.iter().map(|s| s.to_string()).collect();
    let keys: Vec<String> = custom_keys.unwrap_or(default_keys);
    if keys.is_empty() {
        return Err("no keys to strip (empty --keys)".into());
    }

    let dir = resolve_substrate_dir(&base);
    let props_path = dir.join("substrate.props");

    if !props_path.exists() {
        return Err(
            format!("no substrate.props at {}", props_path.display()).into(),
        );
    }

    let size_before = std::fs::metadata(&props_path)?.len();
    eprintln!(
        "strip: opening {} ({:.2} GiB)",
        props_path.display(),
        size_before as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Load (one full decode — peak anon tracks the file size).
    let t_load = std::time::Instant::now();
    let mut snap = PropertiesSnapshotV1::load(&props_path)?;
    eprintln!(
        "strip: loaded snapshot in {:.2}s ({} node entries, {} edge entries)",
        t_load.elapsed().as_secs_f64(),
        snap.nodes.len(),
        snap.edges.len()
    );

    // Strip. Linear scan over a small SKIP set is the fastest approach
    // for <10 short strings; any hash overhead would dominate.
    let skip_fn =
        |k: &str| keys.iter().any(|s| s.as_str() == k);

    let mut stripped_node_props = 0usize;
    let mut stripped_node_entries = 0usize;
    for e in snap.nodes.iter_mut() {
        let before = e.props.len();
        e.props.retain(|(k, _)| !skip_fn(k));
        let delta = before - e.props.len();
        if delta > 0 {
            stripped_node_props += delta;
            stripped_node_entries += 1;
        }
    }
    let mut stripped_edge_props = 0usize;
    let mut stripped_edge_entries = 0usize;
    for e in snap.edges.iter_mut() {
        let before = e.props.len();
        e.props.retain(|(k, _)| !skip_fn(k));
        let delta = before - e.props.len();
        if delta > 0 {
            stripped_edge_props += delta;
            stripped_edge_entries += 1;
        }
    }

    // Prune entries that became empty after stripping — no point in
    // keeping zero-prop PropEntrys in the snapshot (the loader filters
    // empties too but we skip the serialisation cost).
    let nodes_before_prune = snap.nodes.len();
    snap.nodes.retain(|e| !e.props.is_empty());
    let nodes_pruned = nodes_before_prune - snap.nodes.len();

    let edges_before_prune = snap.edges.len();
    snap.edges.retain(|e| !e.props.is_empty());
    let edges_pruned = edges_before_prune - snap.edges.len();

    eprintln!("strip: summary");
    eprintln!("  skip keys       : {:?}", keys);
    eprintln!(
        "  node-props stripped : {} across {} node entries",
        stripped_node_props, stripped_node_entries
    );
    eprintln!(
        "  edge-props stripped : {} across {} edge entries",
        stripped_edge_props, stripped_edge_entries
    );
    eprintln!(
        "  empty entries pruned: {} nodes, {} edges",
        nodes_pruned, edges_pruned
    );

    if dry_run {
        eprintln!("strip: --dry-run specified, not rewriting the file");
        return Ok(());
    }

    if stripped_node_props == 0 && stripped_edge_props == 0 {
        eprintln!("strip: nothing to strip, exiting without rewrite");
        return Ok(());
    }

    // Clone the pre-state via APFS clonefile (constant time) for safety.
    // We use `std::fs::copy` here rather than the clonefile syscall —
    // APFS promotes to clonefile automatically for intra-volume copies
    // on macOS. On other filesystems this is a real copy but still a
    // bounded one-off cost.
    let backup = props_path.with_extension("props.pre-strip");
    if !backup.exists() {
        std::fs::copy(&props_path, &backup)?;
        eprintln!(
            "strip: backup written to {} ({} B)",
            backup.display(),
            std::fs::metadata(&backup)?.len()
        );
    } else {
        eprintln!(
            "strip: backup already exists at {}, keeping existing",
            backup.display()
        );
    }

    // Re-persist atomically. `PropertiesSnapshotV1::persist` uses tmp +
    // rename, so a crash leaves the file in either pre-or-post state.
    let t_write = std::time::Instant::now();
    snap.persist(&props_path)?;
    let size_after = std::fs::metadata(&props_path)?.len();
    eprintln!(
        "strip: rewrote {} in {:.2}s ({:.2} GiB -> {:.2} GiB, -{:.1}%)",
        props_path.display(),
        t_write.elapsed().as_secs_f64(),
        size_before as f64 / (1024.0 * 1024.0 * 1024.0),
        size_after as f64 / (1024.0 * 1024.0 * 1024.0),
        (1.0 - size_after as f64 / size_before as f64) * 100.0,
    );
    Ok(())
}
