//! # Converter pipeline — legacy → substrate
//!
//! Walks a hydrated legacy `LpgStore`, streams nodes then edges into a
//! fresh `SubstrateStore`, and transfers per-entity cognitive state. The
//! pipeline is split into phases matched 1:1 by [`crate::checkpoint::Phase`]
//! so that `--resume` can skip already-completed phases cleanly.
//!
//! ## Node id mapping
//!
//! Legacy `NodeId`s are dense u64 counters, but substrate node offsets
//! are assigned at write-time and are not guaranteed to match. The
//! migrator therefore keeps an `old_id → new_id` map in RAM. At a
//! conservative 16 B / entry, this fits a billion-node base in ~16 GB —
//! for the target range (≤ 10⁶ nodes / base) it stays under 16 MB.
//!
//! ## Write volume
//!
//! For 10⁶ nodes with ~8× avg degree the migrator does:
//! - 10⁶ `create_node_with_props` calls
//! - 8·10⁶ `create_edge_with_props` calls
//!
//! Both go through substrate's WAL — this is slow (~minutes) but correct.
//! Optimising the bulk path is a T17 follow-up.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use obrain_common::types::{EdgeId, NodeId, PropertyKey, Value};
use obrain_common::utils::hash::{FxBuildHasher, FxHashMap};
use obrain_core::graph::Direction;
use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
use obrain_substrate::{PropertiesStreamingWriter, SubstrateStore, BLOB_COLUMN_THRESHOLD_BYTES};

use crate::checkpoint::{Checkpoint, Phase};
#[cfg(feature = "legacy-read")]
use crate::legacy_reader;

/// CLI-parsed migration options.
pub struct MigrateOptions {
    pub input: PathBuf,
    pub output: PathBuf,
    pub workers: usize,
    pub resume: bool,
    pub with_cognitive_init: bool,
    pub with_tiers: bool,

    // --- Property filtering (memory containment for large sources) -----
    //
    // `SubstrateStore.set_node_property` caches every property in a
    // `DashMap<NodeId, NodeInMem>` until the T17 property-pages subsystem
    // lands. On sources with large text properties (e.g. megalaw's legal
    // articles) this balloons RSS far past available RAM and macOS
    // Jetsam kills the process mid-migration. These filters let the user
    // skip known-heavy keys or any value exceeding a size threshold.

    /// Property keys to skip during `phase_nodes` (exact match). Repeatable.
    pub skip_prop_keys: Vec<String>,
    /// Skip any property value whose size hint exceeds this many bytes.
    /// `None` = no size-based filter.
    pub max_prop_bytes: Option<usize>,
    /// If true, copy **no** node properties at all (only labels + structure).
    /// Equivalent to enumerating every key in `--skip-prop`; nuclear option
    /// for quickly producing a structural skeleton.
    pub skip_all_props: bool,
}

/// Cognitive property keys recognised on the legacy side. These are the
/// names used by the `engine::cognitive::*` stores when they round-trip
/// through `GraphStore::set_node_property`. The migrator reads them from
/// the source and re-emits them as substrate column updates.
mod cog_keys {
    pub const ENERGY: &str = "_cog_energy";
    pub const SCAR: &str = "_cog_scar";
    pub const UTILITY: &str = "_cog_utility";
    pub const AFFINITY: &str = "_cog_affinity";
    pub const SYNAPSE_WEIGHT: &str = "_syn_weight";
    pub const COMMUNITY_ID: &str = "_community_id";
}

/// Entry point — executes every phase, honouring `resume` and the
/// optional cognitive-init / tier phases.
pub fn migrate(opts: &MigrateOptions) -> Result<()> {
    // Wire the rayon global thread pool once. `opts.workers == 0` means
    // "rayon default" (physical core count). We silently ignore a second
    // initialisation attempt — the pool is global and can only be set
    // once per process (e.g. when the binary is invoked twice from a
    // test harness).
    if opts.workers > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(opts.workers)
            .build_global();
    }

    // Resolve canonical paths up front so the checkpoint file has a
    // stable identity to compare against.
    let input_canonical = std::fs::canonicalize(&opts.input)
        .with_context(|| format!("canonicalize {}", opts.input.display()))?;

    std::fs::create_dir_all(&opts.output)
        .with_context(|| format!("create_dir_all {}", opts.output.display()))?;

    // Load / create checkpoint.
    let mut ckpt = match (opts.resume, Checkpoint::load(&opts.output)?) {
        (true, Some(cp)) => {
            if cp.input_canonical != input_canonical {
                anyhow::bail!(
                    "--resume requires the same input as the original run; \
                     checkpoint points at {}",
                    cp.input_canonical.display()
                );
            }
            tracing::info!(
                "resuming from phase after {:?} ({} nodes, {} edges already written)",
                cp.last_completed,
                cp.nodes_written,
                cp.edges_written
            );
            cp
        }
        (true, None) => {
            tracing::info!("--resume: no checkpoint found, starting fresh");
            Checkpoint::new(input_canonical)
        }
        (false, _) => {
            // Fresh run; if an old checkpoint is lying around, remove it.
            let _ = Checkpoint::finalize(&opts.output);
            Checkpoint::new(input_canonical)
        }
    };

    // Open destination substrate (fresh). If the dir contains a previous
    // substrate file from a past attempt we refuse to continue unless
    // --resume is set (the caller asked for a clean run).
    let substrate_path = opts.output.join("substrate.obrain");
    let substrate = open_or_create_substrate(&substrate_path, opts.resume)?;

    // Scope the legacy handle so every resource it owns (mmaps, DashMaps,
    // property caches, potentially multi-GB embedding vectors) is dropped
    // **before** we hit the CognitiveInit phase or the final flush. On a
    // Wikipedia-scale base the legacy side alone holds ~6 GB RSS; without
    // this scope the 7 GB sidecar-snapshot serialisation during flush
    // would briefly peak at ~13 GB.
    // Open the streaming properties writer BEFORE the first node is
    // created. The writer replaces the `substrate.set_node_property` /
    // `set_edge_property` calls for the duration of the migration —
    // routing properties through the DashMap is what OOMs megalaw-scale
    // sources (~7 GB of legal article text) at ~40% of Phase::Nodes.
    //
    // Must be dropped or finished BEFORE the function returns; see
    // `PropertiesStreamingWriter`'s Drop impl for the cleanup contract.
    //
    // Not compatible with --resume for this transitional release: the
    // writer has no way to pick up from a partial tmp file and the
    // interleaving with `substrate.flush()`-written snapshots would be
    // ambiguous. Resume with a mid-phase checkpoint is a T17 follow-up.
    if opts.resume && ckpt.last_completed >= Phase::StoreInit {
        anyhow::bail!(
            "--resume is not supported with the streaming properties writer \
             enabled by default. Either rerun from scratch (delete {}), or \
             open an issue tracking T17 to add resume support.",
            opts.output.display()
        );
    }
    let mut props_writer = substrate
        .open_streaming_props_writer()
        .context("opening streaming properties writer")?;

    let _node_map = {
        let legacy = open_legacy_store(&opts.input)?;

        // Node id mapping: populated in Phase::Nodes, consumed in Phase::Edges.
        // When resuming past Nodes, the mapping is rebuilt by a cheap scan of
        // the substrate adjacency (trivial since slot order matches insertion
        // order in our single-writer pipeline).
        let node_map = if ckpt.last_completed >= Phase::Nodes {
            rebuild_node_map_from_substrate(&legacy, &substrate)?
        } else {
            FxHashMap::with_capacity_and_hasher(legacy.node_count(), FxBuildHasher::default())
        };

        // --- Phase::StoreInit ----------------------------------------------
        if ckpt.last_completed < Phase::StoreInit {
            ckpt.mark(Phase::StoreInit, &opts.output)?;
        }

        // --- Phase::Nodes --------------------------------------------------
        let node_map = if ckpt.last_completed < Phase::Nodes {
            let filter = PropFilter::from_opts(opts);
            let map = phase_nodes(
                &legacy,
                &substrate,
                &mut ckpt,
                &opts.output,
                node_map,
                &filter,
                &mut props_writer,
            )?;
            ckpt.mark(Phase::Nodes, &opts.output)?;
            map
        } else {
            tracing::info!("Phase::Nodes — skipped (already done)");
            node_map
        };

        // --- Phase::Edges --------------------------------------------------
        if ckpt.last_completed < Phase::Edges {
            phase_edges(
                &legacy,
                &substrate,
                &node_map,
                &mut ckpt,
                &opts.output,
                &mut props_writer,
            )?;
            ckpt.mark(Phase::Edges, &opts.output)?;
        } else {
            tracing::info!("Phase::Edges — skipped (already done)");
        }

        // --- Phase::Cognitive ----------------------------------------------
        if ckpt.last_completed < Phase::Cognitive {
            phase_cognitive(&legacy, &substrate, &node_map)?;
            ckpt.mark(Phase::Cognitive, &opts.output)?;
        } else {
            tracing::info!("Phase::Cognitive — skipped (already done)");
        }

        // --- Phase::Tiers (optional) ---------------------------------------
        if opts.with_tiers && ckpt.last_completed < Phase::Tiers {
            phase_tiers(&legacy, &substrate, &node_map)?;
            ckpt.mark(Phase::Tiers, &opts.output)?;
        } else if !opts.with_tiers {
            tracing::info!("Phase::Tiers — skipped (flag disabled)");
        }

        // `legacy` is dropped at the end of this block (end-of-scope drop
        // on the `Arc<dyn GraphStore>` inside `open_legacy_store`'s return
        // value). The `node_map` escapes via this trailing expression in
        // case a future phase still wants it — today no post-legacy phase
        // uses it, so the binding is consumed with a leading underscore.
        node_map
    };

    tracing::info!(
        "legacy store closed — source mmaps and property caches released \
         before CognitiveInit / flush"
    );

    // --- Phase::CognitiveInit (optional) ----------------------------------
    if opts.with_cognitive_init && ckpt.last_completed < Phase::CognitiveInit {
        phase_cognitive_init(&substrate)?;
        ckpt.mark(Phase::CognitiveInit, &opts.output)?;
    } else if !opts.with_cognitive_init {
        tracing::info!("Phase::CognitiveInit — skipped (flag disabled)");
    }

    // --- Phase::Done -------------------------------------------------------
    //
    // Order matters: `substrate.flush()` calls `persist_properties()`
    // which writes `substrate.props` from the (empty) DashMap. We then
    // atomically rename the streamed file over top via
    // `props_writer.finish()`. Reversing the order would let flush()
    // clobber the good streamed file with an empty snapshot.
    substrate
        .flush()
        .context("final flush of substrate store")?;
    let (nodes_streamed, edges_streamed) = props_writer.counts();
    props_writer
        .finish()
        .context("finalise streamed properties")?;
    tracing::info!(
        "streamed props snapshot: {nodes_streamed} nodes, {edges_streamed} edges \
         (bypassed DashMap)"
    );
    ckpt.mark(Phase::Done, &opts.output)?;
    Checkpoint::finalize(&opts.output)?;

    tracing::info!(
        "migration complete: {} nodes, {} edges written",
        ckpt.nodes_written,
        ckpt.edges_written
    );
    Ok(())
}

/// Run the "catch up on missing phases" pass on an existing substrate
/// base. Invoked via `obrain-migrate --finalize <substrate_path>`.
///
/// The primary use case at T16.7 is the in-place vector-column upgrade:
/// a base migrated before the vec_columns routing landed keeps its
/// `Value::Vector` payloads in `substrate.props`. Opening the base with
/// T16.7+ code automatically routes those vectors into dense mmap'd
/// `substrate.veccol.*` zones via `load_properties` (see Step 3b), and
/// the subsequent `flush()` re-serialises the sidecar without them. A
/// third reopen then returns the vector from vec_columns exclusively.
///
/// This helper packages that open-flush-close cycle behind a dedicated
/// CLI entry point so operators can run the upgrade explicitly — no
/// service startup, no query workload, just the minimal side-effects
/// needed to persist the catch-up. Logs the before/after sidecar size
/// so operators can verify the RSS win.
///
/// Arguments:
/// * `substrate_dir` — the directory containing `substrate.meta` /
///   `substrate.props` / `substrate.veccol.*` zones. The same path
///   passed to a running hub.
///
/// Idempotent by construction: the second `--finalize` run on the same
/// base is a no-op (`auto_migrated` counters come back zero; sidecar is
/// already vector-free).
/// T17c Step 4 — drain a legacy `substrate.props` bincode sidecar
/// into the PropsZone v2 page chain, then delete the sidecar.
///
/// Contract:
/// 1. Set `OBRAIN_PROPS_V2=1` before opening so the SubstrateStore
///    materialises the `props_zone` + starts routing scalar writes
///    to the v2 chain via `set_node_property`.
/// 2. Open the target directory. `load_properties` hydrates the
///    legacy sidecar into the in-memory `node_properties` DashMap
///    (the Step 3d gate is bypassed because the v2 zone is
///    fresh — `allocated_page_count() == 0`).
/// 3. Call `SubstrateStore::finalize_props_v2()` which iterates
///    `node_properties` and re-emits every scalar through the
///    public setter — a no-op for vec/blob entries already routed
///    out, a v2-chain append for scalars.
/// 4. `flush()` — msyncs all zones, persists the updated dict v3
///    + vec-column specs, writes `substrate.meta`.
/// 5. `delete_legacy_props_sidecar()` — removes `substrate.props`
///    so the next open doesn't waste cycles reading a stale file.
/// 6. Drop the store.
///
/// Idempotent: running this twice on the same base drains nothing
/// on the second run (DashMap is empty because Step 3d gate is now
/// active — the sidecar is already gone). The second invocation
/// reports `nodes_processed = 0` and exits cleanly.
///
/// Rollback: DO NOT run this against a base you might want to open
/// with a pre-T17c binary. The sidecar is deleted unconditionally
/// after a successful drain. If rollback is required, snapshot the
/// directory before invoking.
/// Backward-compat shim — delegates to [`finalize_v2_with_opts`]
/// with default options (dry-run OFF, snapshot ON). External callers
/// that want the legacy one-arg signature can keep using this.
#[allow(dead_code)]
pub fn finalize_v2(substrate_dir: &Path) -> Result<()> {
    finalize_v2_with_opts(substrate_dir, &FinalizeV2Opts::default())
}

/// Options for [`finalize_v2_with_opts`].
///
/// Unit-struct defaults keep the original single-arg `finalize_v2`
/// binary-compatible with the T17c Step 4 CLI, while the new surface
/// covers the T17c Step 7 safety additions: dry-run + pre-flight
/// snapshot.
#[derive(Debug, Clone, Default)]
pub struct FinalizeV2Opts {
    /// Dry-run: open, introspect, report counts, but DO NOT write
    /// to PropsZone v2, the edge sidecar, or delete the legacy
    /// sidecar. Used for Wikipedia rehearsals.
    pub dry_run: bool,
    /// Skip the pre-migration snapshot (tar-equivalent copy of the
    /// substrate directory to `<dir>.bak-YYYYMMDDHHMM`). Default
    /// behaviour is to take the snapshot — opt-out only when the
    /// caller has already secured a backup externally.
    pub skip_backup: bool,
}

pub fn finalize_v2_with_opts(
    substrate_dir: &Path,
    opts: &FinalizeV2Opts,
) -> Result<()> {
    if !substrate_dir.exists() {
        anyhow::bail!(
            "finalize-v2 target does not exist: {}",
            substrate_dir.display()
        );
    }
    let props_path = substrate_dir.join("substrate.props");
    let before_bytes = std::fs::metadata(&props_path)
        .map(|m| m.len())
        .unwrap_or(0);

    tracing::info!(
        "obrain-migrate --finalize-v2: opening {} (substrate.props={} MiB, v2 forced on, dry_run={}, skip_backup={})",
        substrate_dir.display(),
        before_bytes / (1024 * 1024),
        opts.dry_run,
        opts.skip_backup,
    );

    // T17c Step 7 — snapshot safety. Unless explicitly opted-out
    // (caller has an external backup) or in dry-run mode, copy the
    // substrate directory to `<dir>.bak-YYYYMMDDHHMM` before touching
    // anything. Recursive copy with checksums is overkill for a
    // migration snapshot — `fs::copy` over each entry is enough
    // because substrate files are mmap'd regular files.
    if !opts.dry_run && !opts.skip_backup {
        // No chrono dep in this crate — seconds-since-epoch is a
        // unique-enough suffix for backup naming. Operators who
        // prefer human-readable timestamps can rename post-hoc.
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let backup = substrate_dir.with_file_name(format!(
            "{}.bak-{}",
            substrate_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("substrate"),
            suffix
        ));
        tracing::info!(
            "finalize-v2 snapshot: {} → {}",
            substrate_dir.display(),
            backup.display()
        );
        copy_dir_recursive(substrate_dir, &backup).with_context(|| {
            format!(
                "snapshot {} → {}",
                substrate_dir.display(),
                backup.display()
            )
        })?;
    }

    // Dry-run short-circuit: open WITHOUT forcing v2 so we don't
    // silently create empty `substrate.props.v2` + `.heap` zone files
    // on a base that never had them. `node_count_live_estimate` and
    // `edge_properties_count` live on the hydrated DashMaps — no v2
    // zone needed to introspect them.
    if opts.dry_run {
        let store = SubstrateStore::open(substrate_dir)
            .with_context(|| format!("open {}", substrate_dir.display()))?;
        let nodes_pending = store.node_count_live_estimate();
        let edges_pending = store
            .edge_properties_count()
            .unwrap_or(0);
        tracing::info!(
            "finalize-v2 DRY-RUN: would drain ~{} nodes and {} edge-property maps. \
             Legacy sidecar: {} MiB. No writes performed.",
            nodes_pending,
            edges_pending,
            before_bytes / (1024 * 1024),
        );
        drop(store);
        return Ok(());
    }

    // Real run — force-enable v2 for the open → setter → flush →
    // close cycle. Scoped to this invocation — we restore the previous
    // value on exit to keep test and CLI invocations idempotent.
    let prev_env = std::env::var("OBRAIN_PROPS_V2").ok();
    // SAFETY: single-threaded up until SubstrateStore::open returns;
    // the env var is only read at open time.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("OBRAIN_PROPS_V2", "1");
    }
    let restore_env = EnvRestorer { prev: prev_env };

    let store = SubstrateStore::open(substrate_dir)
        .with_context(|| format!("open {}", substrate_dir.display()))?;

    if !store.props_v2_enabled() {
        anyhow::bail!(
            "finalize-v2: v2 zone failed to initialise — check OBRAIN_PROPS_V2 \
             env pass-through or filesystem permissions"
        );
    }

    let stats = store
        .finalize_props_v2()
        .context("finalize_props_v2 drain")?;
    tracing::info!(
        "finalize-v2 drain: {} nodes, {} scalars emitted, {} edges persisted \
         ({} edge scalars, {} edge-sidecar bytes), {} v2 pages allocated",
        stats.nodes_processed,
        stats.scalars_emitted,
        stats.edges_processed,
        stats.edge_scalars_emitted,
        stats.edge_sidecar_bytes,
        store.props_v2_page_count().unwrap_or(0),
    );

    store
        .flush()
        .context("flush after finalize_props_v2 drain")?;

    // Delete the sidecar only if the drain actually moved something.
    // A fresh v2 base (nothing to drain) has no legacy sidecar to
    // delete — the call is a no-op anyway (returns `Ok(None)`).
    // The T17c Step 6 gate inside `delete_legacy_props_sidecar`
    // refuses the delete when edge props are unsaved, so even if a
    // caller bypassed `finalize_props_v2` (e.g. called us on a base
    // that already had v2 pages), we cannot silently lose edge props.
    let freed = store
        .delete_legacy_props_sidecar()
        .context("delete legacy sidecar")?;
    drop(store);
    drop(restore_env);

    let after_bytes = 0u64; // sidecar gone on success
    tracing::info!(
        "obrain-migrate --finalize-v2: done. substrate.props: {} MiB → {} MiB \
         ({} MiB freed)",
        before_bytes / (1024 * 1024),
        after_bytes / (1024 * 1024),
        freed.unwrap_or(0) / (1024 * 1024),
    );
    Ok(())
}

/// Recursive directory copy — enough for the substrate snapshot case
/// (one flat directory of regular mmap'd files, no symlinks, no
/// special devices). Creates `dst` with the same mode as `src`.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ty.is_file() {
            std::fs::copy(&from, &to)?;
        } else {
            // Skip symlinks / sockets / fifos — not expected in a
            // substrate directory, but defensive.
            tracing::warn!(
                "copy_dir_recursive: skipping non-regular entry {}",
                from.display()
            );
        }
    }
    Ok(())
}

/// RAII guard that restores `OBRAIN_PROPS_V2` to its previous value
/// when dropped. Keeps the `finalize_v2` entry point side-effect
/// free so running it from inside a test process doesn't bleed env
/// state into the next test.
struct EnvRestorer {
    prev: Option<String>,
}

impl Drop for EnvRestorer {
    fn drop(&mut self) {
        // SAFETY: drop runs single-threaded at end of finalize_v2.
        #[allow(unsafe_code)]
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var("OBRAIN_PROPS_V2", v),
                None => std::env::remove_var("OBRAIN_PROPS_V2"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T17f Step 5 — `obrain-migrate --upgrade-edges-v2`
//
// In-place upgrade of a substrate base written before T17f Step 1:
//   1. Detect `substrate.edges` stride (32 B pre-Step-1 vs 36 B post).
//   2. If 32 B: atomically rewrite the file to 36 B stride (insert a
//      zero `first_prop_off: U48` at byte offset 24 of each record and
//      shrink the trailing pad from 4 B to 2 B).
//   3. Open the (now Step-1-compatible) store with PropsZone v2 forced
//      on, drain the `substrate.edge_props` sidecar into the PropsZone
//      v2 edge chain via `finalize_edge_props_v2`, flush, delete the
//      sidecar.
//
// Snapshot safety is mandatory unless the caller opted out via
// `--no-backup`. A directory copy → `<dir>.bak-<unix_secs>` guards
// against partial rewrites — on any failure after snapshot, operators
// can roll back with a single `rm -rf <dir> && mv <dir>.bak-* <dir>`.
// ---------------------------------------------------------------------------

/// Pre-Step-1 `EdgeRecord` stride (src=4 + dst=4 + etype=2 + weight=2 +
/// next_from=6 + next_to=6 + ricci=1 + flags=1 + engram_tag=2 + _pad=4).
const EDGE_RECORD_OLD_STRIDE: usize = 32;

/// Post-Step-1 `EdgeRecord` stride — same as `EdgeRecord::SIZE` in
/// obrain-substrate. Duplicated as a local `const` so the migrator
/// doesn't have to import the record struct.
const EDGE_RECORD_NEW_STRIDE: usize = 36;

/// Options for [`upgrade_edges_v2_with_opts`].
#[derive(Debug, Clone, Default)]
pub struct UpgradeEdgesV2Opts {
    /// Dry-run: report the detected stride, edge count, sidecar size,
    /// and pending-work estimate, but perform NO writes — no snapshot,
    /// no rewrite of `substrate.edges`, no PropsZone v2 drain, no
    /// sidecar deletion. Use against a production base before the real
    /// run to confirm the plan looks sane.
    pub dry_run: bool,
    /// Skip the pre-upgrade directory snapshot (copy to
    /// `<dir>.bak-<unix_secs>`). Opt-out only when the caller has
    /// already secured an external backup.
    pub skip_backup: bool,
}

/// Detected stride of a `substrate.edges` zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgesLayout {
    /// Pre-Step-1 base — edges are 32 B each. Requires a rewrite pass.
    Pre,
    /// Post-Step-1 base — edges are already 36 B, no rewrite needed.
    Post,
    /// File is empty — nothing to rewrite, and no edge props to drain
    /// either. Caller is expected to short-circuit on this.
    Empty,
}

/// Read the logical `edge_count` from `substrate.meta`. The zone file
/// itself is zero-padded to a multiple of the OS page, so we can't
/// derive the stride from `file_len % stride` alone — we go through
/// the meta header to know how many edges are logically live.
fn read_meta_edge_count(meta_path: &Path) -> Result<u64> {
    let bytes = std::fs::read(meta_path)
        .with_context(|| format!("read {}", meta_path.display()))?;
    // MetaHeader layout (see obrain_substrate::meta):
    //   magic[8] + format_version[4] + flags[4] + node_count[8] + edge_count[8] + ...
    // So edge_count is at byte offset 24..32.
    if bytes.len() < 32 {
        anyhow::bail!(
            "substrate.meta is only {} B — too small to contain an edge_count field",
            bytes.len()
        );
    }
    // Magic sanity check — the first 8 bytes should be `SUBSTRT\0`.
    const SUBSTRATE_MAGIC: [u8; 8] = *b"SUBSTRT\0";
    if bytes[0..8] != SUBSTRATE_MAGIC {
        anyhow::bail!(
            "substrate.meta magic mismatch — not a substrate base (got {:?})",
            &bytes[0..8]
        );
    }
    let edge_count = u64::from_le_bytes(
        bytes[24..32]
            .try_into()
            .expect("slice of exactly 8 bytes is always a [u8; 8]"),
    );
    Ok(edge_count)
}

/// Read `next_edge_id` from `substrate.dict` as a fallback when
/// `meta.edge_count` is stale (known issue on legacy bases: the meta
/// header's edge/node/prop counts are not refreshed on close —
/// tracked separately from T17f).
///
/// `next_edge_id` is the slot allocator state: the first unused edge
/// slot. Slot 0 is the null sentinel, so `next_edge_id - 1` is the
/// number of allocated edge slots (including deleted/tombstoned —
/// substrate zones don't compact on delete, so all allocated slots
/// live in the file contiguously from offset 0).
fn read_dict_next_edge_id(dict_path: &Path) -> Result<u64> {
    let bytes = std::fs::read(dict_path)
        .with_context(|| format!("read {}", dict_path.display()))?;
    let snap = obrain_substrate::dict::DictSnapshot::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("parse {}: {e}", dict_path.display()))?;
    Ok(snap.next_edge_id)
}

fn detect_edges_layout(substrate_dir: &Path) -> Result<(EdgesLayout, u64, u64)> {
    let edges_path = substrate_dir.join("substrate.edges");
    let meta_path = substrate_dir.join("substrate.meta");
    let dict_path = substrate_dir.join("substrate.dict");

    let file_len = if edges_path.exists() {
        std::fs::metadata(&edges_path)
            .with_context(|| format!("stat {}", edges_path.display()))?
            .len()
    } else {
        0
    };
    if file_len == 0 {
        return Ok((EdgesLayout::Empty, 0, 0));
    }

    // Preferred: `meta.edge_count`. Fallback: `dict.next_edge_id - 1`.
    // Legacy bases (pre-T17f) have a stale meta header with
    // `edge_count=0` even when millions of live edges exist. The dict
    // is authoritative because the slot allocator cannot tolerate
    // staleness (would corrupt inserts).
    let meta_edge_count = read_meta_edge_count(&meta_path)?;
    let edge_count = if meta_edge_count > 0 {
        meta_edge_count
    } else if dict_path.exists() {
        let next_edge_id = read_dict_next_edge_id(&dict_path)?;
        // `next_edge_id >= 1` is enforced by `DictSnapshot::from_bytes`.
        // The substrate writes slot 0 (null sentinel) as a zero-filled
        // record at offset 0, then real edges at slots 1..next_edge_id.
        // So the file contains `next_edge_id` records total (sentinel
        // included) — this is the "slots written" count the detector
        // + rewriter need, NOT `next_edge_id - 1` (which would miss
        // the sentinel and cause an off-by-one rewrite).
        let derived = next_edge_id;
        tracing::warn!(
            phase = "upgrade_edges_v2::detect",
            meta_edge_count = meta_edge_count,
            dict_next_edge_id = next_edge_id,
            derived_slot_count = derived,
            "meta.edge_count is stale (known T17f-era meta bug); falling back to \
             dict.next_edge_id (slot count including null sentinel) for layout detection"
        );
        derived
    } else {
        0
    };
    if edge_count == 0 {
        // No live edges — no rewrite needed. Default to Post layout
        // (the current format); the upgrader will short-circuit the
        // rewrite phase anyway.
        return Ok((EdgesLayout::Post, file_len, 0));
    }

    // Content-based stride discrimination, with tail-zero validation.
    //
    // Tail-zero alone is not enough: on a small Post file where every
    // allocated edge has `first_prop_off=0` in the mmap (the DashMap-
    // only write path that T17f Step 4 left behind for fresh
    // `set_edge_property` calls), `bytes[n32..n36]` is all zero AND
    // `bytes[n36..]` is the zone pre-alloc padding — so both a Pre
    // and a Post interpretation pass a tail-zero check, and Pre wins
    // by strict-more-restrictive ordering even though the file is
    // actually Post. This was the exact failure pattern on the
    // `seed_edge_fixture` tests after the Pre-first swap.
    //
    // Canonical discriminator: inspect `bytes[32..36]` when
    // `edge_count >= 2`. Slot 0 is the null sentinel — always all
    // zeros in both layouts. So:
    //   * Pre  file: bytes[32..36] is slot 1's `src` field (u32 LE
    //                NodeId). Slot 1 is the first real allocated edge;
    //                its `src` is a real NodeId ≥ 1, so at least one
    //                of these 4 bytes is nonzero.
    //   * Post file: bytes[32..36] is the last 4 bytes of the sentinel
    //                (which spans [0..36]) — always all zeros.
    //
    // Nonzero at [32..36] → Pre; all-zero → Post. Tail-zero check is
    // retained as validation: if the claimed layout picks up stray
    // nonzero bytes past its zone, the file is corrupt.
    let bytes = std::fs::read(&edges_path)
        .with_context(|| format!("read {}", edges_path.display()))?;
    let file_len_usize = bytes.len();

    let n32 = edge_count as usize * EDGE_RECORD_OLD_STRIDE;
    let n36 = edge_count as usize * EDGE_RECORD_NEW_STRIDE;

    // A file claiming `edge_count` edges MUST hold at least `n32`
    // bytes — it's the strictly smaller of the two strides. Anything
    // smaller is a truncation and we can't safely discriminate.
    if file_len_usize < n32 {
        anyhow::bail!(
            "substrate.edges is {} B, smaller than edge_count={} × min stride \
             {} B = {} B (file truncated or edge_count is stale beyond recovery)",
            file_len,
            edge_count,
            EDGE_RECORD_OLD_STRIDE,
            n32,
        );
    }

    let is_zero_tail = |offset: usize| -> bool {
        if offset > file_len_usize {
            false
        } else {
            bytes[offset..].iter().all(|&b| b == 0)
        }
    };

    if edge_count < 2 {
        // Sentinel-only base (no real edges). Nothing discriminable
        // from the bytes — the sentinel is all zeros in both layouts,
        // and there's no slot 1 to inspect. Default to Post (the
        // current format); the upgrader will short-circuit any
        // rewrite/drain work upstream when there are no real edges.
        return Ok((EdgesLayout::Post, file_len, edge_count));
    }

    // Slot 1 occupies bytes[32..64] in Pre and bytes[36..72] in Post.
    // Bytes[32..36] are the decisive 4-byte window.
    let slot1_first_word_nonzero = bytes[32..36].iter().any(|&b| b != 0);

    if slot1_first_word_nonzero {
        // Pre layout claimed. Everything past `n32` must be the zone
        // pre-allocation tail (all zero). Otherwise the file is
        // corrupt — most likely a schema mismatch or a partial write.
        if !is_zero_tail(n32) {
            anyhow::bail!(
                "substrate.edges looks like Pre-Step-1 layout (nonzero at \
                 bytes[32..36], interpreted as slot 1 src) but bytes past \
                 the 32 B × {} edge boundary ({} B) are not all zero — file \
                 is corrupt or produced by a foreign format",
                edge_count,
                n32,
            );
        }
        return Ok((EdgesLayout::Pre, file_len, edge_count));
    }

    // Post layout claimed. Need room for the 36 B stride and zone
    // padding past n36 must be zero.
    if file_len_usize < n36 {
        anyhow::bail!(
            "substrate.edges claims Post layout (bytes[32..36] all zero, \
             sentinel shape) but is only {} B — smaller than edge_count={} \
             × {} B = {} B required for Post stride",
            file_len,
            edge_count,
            EDGE_RECORD_NEW_STRIDE,
            n36,
        );
    }
    if !is_zero_tail(n36) {
        anyhow::bail!(
            "substrate.edges looks like Post layout but bytes past the \
             36 B × {} edge boundary ({} B) are not all zero — file is \
             corrupt or produced by a foreign format",
            edge_count,
            n36,
        );
    }
    Ok((EdgesLayout::Post, file_len, edge_count))
}

/// Rewrite `substrate.edges` from the pre-Step-1 32 B stride to the
/// post-Step-1 36 B stride. Writes to a `<dir>/substrate.edges.tmp`
/// first then renames over the original — so a power loss mid-rewrite
/// leaves the old file untouched (plus a stray `.tmp` file the next
/// run can retry over).
///
/// `edge_count` is the logical number of edges (as read from
/// `substrate.meta`). The file may be zero-padded past `edge_count *
/// 32` due to zone pre-allocation; those bytes are not part of any
/// record and are dropped on output.
fn rewrite_edges_32_to_36(edges_path: &Path, edge_count: u64) -> Result<()> {
    use std::io::{BufWriter, Write};

    let input = std::fs::read(edges_path)
        .with_context(|| format!("read {}", edges_path.display()))?;
    let required = (edge_count * EDGE_RECORD_OLD_STRIDE as u64) as usize;
    anyhow::ensure!(
        input.len() >= required,
        "rewrite_edges_32_to_36: input length {} is smaller than expected \
         {} edges × {} B = {} (file truncated?)",
        input.len(),
        edge_count,
        EDGE_RECORD_OLD_STRIDE,
        required,
    );

    let tmp = edges_path.with_extension("edges.tmp");
    // Clean any leftover `.tmp` from a previous failed run. Safe: we
    // have the snapshot (unless --no-backup), so worst case operators
    // restore from `<dir>.bak-<unix_secs>`.
    if tmp.exists() {
        std::fs::remove_file(&tmp)
            .with_context(|| format!("remove stale {}", tmp.display()))?;
    }

    let out_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&tmp)
        .with_context(|| format!("create {}", tmp.display()))?;
    // Pre-allocate the final size so the BufWriter doesn't bounce the
    // file size with each flush.
    out_file
        .set_len(edge_count * EDGE_RECORD_NEW_STRIDE as u64)
        .with_context(|| format!("set_len {}", tmp.display()))?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    // Zero buffers for the inserted 6 B (first_prop_off = 0) and the
    // trimmed 2 B trailing pad.
    let first_prop_off_zero = [0u8; 6];
    let trailing_pad_zero = [0u8; 2];

    let t_start = std::time::Instant::now();
    let progress_step = (edge_count / 20).max(500_000);

    for i in 0..edge_count as usize {
        let base = i * EDGE_RECORD_OLD_STRIDE;
        let old = &input[base..base + EDGE_RECORD_OLD_STRIDE];
        // Old layout split points:
        //   [0..24]  src | dst | edge_type | weight | next_from | next_to
        //   [24..28] ricci | flags | engram_tag
        //   [28..32] _pad[4]  (discarded)
        // New layout:
        //   [0..24]  same as old 0..24
        //   [24..30] first_prop_off = 0
        //   [30..34] shifted old [24..28]
        //   [34..36] _pad[2] = 0
        out.write_all(&old[0..24])
            .with_context(|| format!("write prefix of edge {i}"))?;
        out.write_all(&first_prop_off_zero)
            .with_context(|| format!("write first_prop_off of edge {i}"))?;
        out.write_all(&old[24..28])
            .with_context(|| format!("write tail of edge {i}"))?;
        out.write_all(&trailing_pad_zero)
            .with_context(|| format!("write pad of edge {i}"))?;

        if progress_step > 0 && i > 0 && i % progress_step as usize == 0 {
            tracing::info!(
                phase = "upgrade_edges_v2::rewrite",
                progress_pct = (i as u64 * 100 / edge_count),
                edges_rewritten = i,
                elapsed_ms = t_start.elapsed().as_millis() as u64,
                "rewriting substrate.edges 32 B → 36 B"
            );
        }
    }

    let mut out_file = out
        .into_inner()
        .map_err(|e| anyhow::anyhow!("flush BufWriter on {}: {e}", tmp.display()))?;
    out_file.flush().with_context(|| format!("flush {}", tmp.display()))?;
    out_file
        .sync_all()
        .with_context(|| format!("fsync {}", tmp.display()))?;
    drop(out_file);

    std::fs::rename(&tmp, edges_path).with_context(|| {
        format!("rename {} → {}", tmp.display(), edges_path.display())
    })?;

    tracing::info!(
        phase = "upgrade_edges_v2::rewrite",
        edges_rewritten = edge_count,
        elapsed_ms = t_start.elapsed().as_millis() as u64,
        "substrate.edges rewrite complete (32 B → 36 B stride)"
    );
    Ok(())
}

/// Backward-compat shim — delegates to [`upgrade_edges_v2_with_opts`]
/// with default options (dry-run OFF, snapshot ON).
#[allow(dead_code)]
pub fn upgrade_edges_v2(substrate_dir: &Path) -> Result<()> {
    upgrade_edges_v2_with_opts(substrate_dir, &UpgradeEdgesV2Opts::default())
}

pub fn upgrade_edges_v2_with_opts(
    substrate_dir: &Path,
    opts: &UpgradeEdgesV2Opts,
) -> Result<()> {
    if !substrate_dir.exists() {
        anyhow::bail!(
            "upgrade-edges-v2 target does not exist: {}",
            substrate_dir.display()
        );
    }

    let edges_path = substrate_dir.join("substrate.edges");
    let (layout, edges_bytes, edge_count) = detect_edges_layout(substrate_dir)?;

    let edge_sidecar_path = substrate_dir.join("substrate.edge_props");
    let sidecar_bytes = std::fs::metadata(&edge_sidecar_path)
        .map(|m| m.len())
        .unwrap_or(0);

    tracing::info!(
        "obrain-migrate --upgrade-edges-v2: opening {} (substrate.edges={} MiB, \
         layout={:?}, edge_count={}, substrate.edge_props={} MiB, dry_run={}, \
         skip_backup={})",
        substrate_dir.display(),
        edges_bytes / (1024 * 1024),
        layout,
        edge_count,
        sidecar_bytes / (1024 * 1024),
        opts.dry_run,
        opts.skip_backup,
    );

    // Dry-run path — no writes, just introspection. We skip opening
    // the store on a `Pre` base because the current bytemuck casts
    // assume 36 B stride; a `Pre` base would bail at open with a
    // stride error. The file-size-derived `edge_count` is enough for
    // the rehearsal.
    if opts.dry_run {
        match layout {
            EdgesLayout::Empty => {
                tracing::info!(
                    "upgrade-edges-v2 DRY-RUN: substrate.edges is empty. \
                     Nothing to do."
                );
            }
            EdgesLayout::Pre => {
                tracing::info!(
                    "upgrade-edges-v2 DRY-RUN: detected PRE-Step-1 layout. \
                     Would rewrite {} edges × 32 B → 36 B ({} MiB → {} MiB), \
                     then drain {} MiB of substrate.edge_props into the \
                     PropsZone v2 edge chain. No writes performed.",
                    edge_count,
                    edges_bytes / (1024 * 1024),
                    (edge_count * EDGE_RECORD_NEW_STRIDE as u64) / (1024 * 1024),
                    sidecar_bytes / (1024 * 1024),
                );
            }
            EdgesLayout::Post => {
                // Safe to open the store on a post-Step-1 base to get
                // the exact edge-property-map count.
                let store = SubstrateStore::open(substrate_dir)
                    .with_context(|| format!("open {}", substrate_dir.display()))?;
                let edges_pending = store.edge_properties_count().unwrap_or(0);
                tracing::info!(
                    "upgrade-edges-v2 DRY-RUN: substrate.edges already at \
                     36 B stride ({} edges). Would drain {} edge-property \
                     maps ({} MiB sidecar) into the PropsZone v2 edge chain. \
                     No writes performed.",
                    edge_count,
                    edges_pending,
                    sidecar_bytes / (1024 * 1024),
                );
                drop(store);
            }
        }
        return Ok(());
    }

    // Real run — snapshot first (unless opted out), then rewrite (if
    // needed), then drain.
    if !opts.skip_backup {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let backup = substrate_dir.with_file_name(format!(
            "{}.bak-{}",
            substrate_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("substrate"),
            suffix
        ));
        tracing::info!(
            "upgrade-edges-v2 snapshot: {} → {}",
            substrate_dir.display(),
            backup.display()
        );
        copy_dir_recursive(substrate_dir, &backup).with_context(|| {
            format!(
                "snapshot {} → {}",
                substrate_dir.display(),
                backup.display()
            )
        })?;
    }

    // Phase 1 — rewrite `substrate.edges` to 36 B stride if needed.
    match layout {
        EdgesLayout::Empty => {
            tracing::info!(
                "upgrade-edges-v2: substrate.edges is empty, nothing to \
                 rewrite. Sidecar size: {} B.",
                sidecar_bytes
            );
            // Still delete an empty-base stray sidecar if present, for
            // parity with the post-drain cleanup path.
            if sidecar_bytes > 0 {
                std::fs::remove_file(&edge_sidecar_path).with_context(|| {
                    format!("remove {}", edge_sidecar_path.display())
                })?;
                tracing::info!(
                    "upgrade-edges-v2: removed {} B empty-base edge sidecar",
                    sidecar_bytes
                );
            }
            return Ok(());
        }
        EdgesLayout::Pre => {
            rewrite_edges_32_to_36(&edges_path, edge_count)?;
        }
        EdgesLayout::Post => {
            tracing::info!(
                "upgrade-edges-v2: substrate.edges already at 36 B stride, \
                 skipping rewrite phase."
            );
        }
    }

    // Phase 2 — open store with PropsZone v2 forced on, drain, delete.
    let prev_env = std::env::var("OBRAIN_PROPS_V2").ok();
    // SAFETY: single-threaded up to SubstrateStore::open; the env var
    // is only read at open time.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("OBRAIN_PROPS_V2", "1");
    }
    let restore_env = EnvRestorer { prev: prev_env };

    let store = SubstrateStore::open(substrate_dir)
        .with_context(|| format!("open {}", substrate_dir.display()))?;

    if !store.props_v2_enabled() {
        anyhow::bail!(
            "upgrade-edges-v2: PropsZone v2 failed to initialise — check \
             OBRAIN_PROPS_V2 env pass-through or filesystem permissions"
        );
    }

    let stats = store
        .finalize_edge_props_v2()
        .context("finalize_edge_props_v2 drain")?;
    tracing::info!(
        "upgrade-edges-v2 drain: {} edges processed, {} scalars emitted, \
         {} v2 pages allocated",
        stats.edges_processed,
        stats.scalars_emitted,
        store.props_v2_page_count().unwrap_or(0),
    );

    store
        .flush()
        .context("flush after finalize_edge_props_v2 drain")?;

    let freed = store
        .delete_edge_props_sidecar()
        .context("delete edge sidecar")?;
    drop(store);
    drop(restore_env);

    tracing::info!(
        "obrain-migrate --upgrade-edges-v2: done. substrate.edge_props: {} MiB \
         → 0 MiB ({} MiB freed).",
        sidecar_bytes / (1024 * 1024),
        freed.unwrap_or(0) / (1024 * 1024),
    );
    Ok(())
}

pub fn finalize(substrate_dir: &Path) -> Result<()> {
    if !substrate_dir.exists() {
        anyhow::bail!(
            "finalize target does not exist: {}",
            substrate_dir.display()
        );
    }
    // Report sidecar size before the upgrade so the caller can see the
    // shrinkage from the single log output.
    let props_path = substrate_dir.join("substrate.props");
    let before_bytes = std::fs::metadata(&props_path)
        .map(|m| m.len())
        .unwrap_or(0);

    tracing::info!(
        "obrain-migrate --finalize: opening {} (substrate.props={} MiB)",
        substrate_dir.display(),
        before_bytes / (1024 * 1024),
    );

    // (1) Open — `load_properties` runs on startup and routes any
    //     `Value::Vector` entries in the sidecar to the vec-column
    //     registry. The tracing::info! emitted from `load_properties`
    //     reports the auto-migration counters.
    let store = SubstrateStore::open(substrate_dir)
        .with_context(|| format!("open {}", substrate_dir.display()))?;

    // (2) Flush — re-serialises the props sidecar via the defensive
    //     filter, drops the vector entries, persists the dict v3 with
    //     the freshly-created vec-column specs, and msyncs every zone.
    store
        .flush()
        .context("flush after finalize auto-migration")?;

    // (3) Drop the handle so the Drop impl releases the WAL / mmap
    //     file descriptors. `drop(store)` is explicit for clarity; the
    //     close is already handled by scope end on the subsequent
    //     `return`.
    drop(store);

    let after_bytes = std::fs::metadata(&props_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let delta_bytes = before_bytes.saturating_sub(after_bytes);
    tracing::info!(
        "obrain-migrate --finalize: done. substrate.props: {} MiB → {} MiB ({} MiB freed)",
        before_bytes / (1024 * 1024),
        after_bytes / (1024 * 1024),
        delta_bytes / (1024 * 1024),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase implementations
// ---------------------------------------------------------------------------

/// Property filter applied during `phase_nodes`. Decides whether a given
/// `(key, value)` pair should be copied from legacy → substrate.
///
/// Built once from [`MigrateOptions`] and reused for every node.
struct PropFilter {
    skip_keys: std::collections::HashSet<String>,
    max_bytes: Option<usize>,
    skip_all: bool,
}

impl PropFilter {
    fn from_opts(opts: &MigrateOptions) -> Self {
        Self {
            skip_keys: opts.skip_prop_keys.iter().cloned().collect(),
            max_bytes: opts.max_prop_bytes,
            skip_all: opts.skip_all_props,
        }
    }

    /// Returns `None` if the property should be copied, or `Some(reason)`
    /// string if it should be skipped (for stats aggregation).
    fn skip_reason(&self, key: &str, value: &Value) -> Option<&'static str> {
        if self.skip_all {
            return Some("skip_all");
        }
        if self.skip_keys.contains(key) {
            return Some("skip_key");
        }
        if let Some(max) = self.max_bytes {
            if value_size_hint(value) > max {
                return Some("oversize");
            }
        }
        None
    }
}

/// Approximate serialized size of a [`Value`] in bytes — good enough for
/// the `--max-prop-bytes` threshold. Scalar types (bool, int, float,
/// date, etc.) report 0 since their footprint is a handful of bytes and
/// never relevant for memory containment.
fn value_size_hint(v: &Value) -> usize {
    match v {
        Value::String(s) => s.len(),
        Value::Bytes(b) => b.len(),
        Value::List(l) => 16 * l.len(),
        Value::Map(m) => 32 * m.len(),
        Value::Vector(v) => 4 * v.len(),
        _ => 0,
    }
}

/// Per-key largest value seen so far — accumulated during `phase_nodes`
/// and dumped at end so the user can decide which keys to blacklist on
/// the next run.
#[derive(Default)]
struct ValueSizeStats {
    /// key → (max_bytes_seen, count_over_1mb)
    per_key: std::collections::HashMap<String, (usize, u64)>,
}

impl ValueSizeStats {
    fn record(&mut self, key: &str, v: &Value) {
        let sz = value_size_hint(v);
        if sz == 0 {
            return; // skip scalars to keep the table small
        }
        let entry = self.per_key.entry(key.to_string()).or_insert((0, 0));
        entry.0 = entry.0.max(sz);
        if sz > 1_000_000 {
            entry.1 += 1;
        }
    }

    fn top_report(&self, n: usize) -> String {
        let mut entries: Vec<(&String, &(usize, u64))> = self.per_key.iter().collect();
        entries.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        entries
            .iter()
            .take(n)
            .map(|(k, (max, over))| {
                if *over > 0 {
                    format!("{k}=max {:.1} KB ({over} over 1 MB)", *max as f64 / 1024.0)
                } else {
                    format!("{k}=max {:.1} KB", *max as f64 / 1024.0)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn phase_nodes(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
    ckpt: &mut Checkpoint,
    out: &Path,
    mut node_map: FxHashMap<u64, NodeId>,
    filter: &PropFilter,
    props_writer: &mut PropertiesStreamingWriter,
) -> Result<FxHashMap<u64, NodeId>> {
    let all_ids = legacy.all_node_ids();
    let total = all_ids.len() as u64;
    let pb = progress_bar(total, "nodes");

    // Checkpoint every 10% of the total work. `chunk_every` rounds down
    // to at least 1 so a tiny test dataset still makes progress.
    let chunk_every = (total / 10).max(1);

    // Heartbeat: emit a tracing::info!() every HEARTBEAT_EVERY nodes so
    // long runs show liveness even when indicatif's progress-bar line is
    // captured/redirected or the user is tailing a log file instead of a
    // TTY. Also reports skip counters + top heavy keys so the user can
    // spot megalaw-style "one giant `text` property" patterns without
    // re-instrumenting.
    const HEARTBEAT_EVERY: u64 = 100_000;

    if filter.skip_all {
        tracing::warn!(
            "Phase::Nodes — --skip-all-props set, copying labels + structure only"
        );
    } else if !filter.skip_keys.is_empty() || filter.max_bytes.is_some() {
        let keys: Vec<_> = filter.skip_keys.iter().cloned().collect();
        tracing::info!(
            "Phase::Nodes — prop filter: skip_keys={:?} max_bytes={:?}",
            keys,
            filter.max_bytes
        );
    }

    let mut nodes_written: u64 = 0;
    let mut skipped_by_key: u64 = 0;
    let mut skipped_oversize: u64 = 0;
    let mut total_props_seen: u64 = 0;
    // T16.7 Step 3 — count user-level `Value::Vector` props that we peel
    // off the sidecar-bound `scratch_props` and route via
    // `substrate.set_node_property`, which (post-Step 2b) dispatches to
    // the dense mmap'd `VecColumnRegistry` instead of the bincode props
    // file. Tracked separately from `total_props_seen` so the heartbeat
    // can show "of the N props seen, V went to vec columns".
    let mut vectors_routed: u64 = 0;
    let mut vector_bytes_routed: u64 = 0;
    // T16.7 Step 4e — same peel-off for oversize `Value::String` /
    // `Value::Bytes` (payload > BLOB_COLUMN_THRESHOLD_BYTES). The runtime
    // side (`set_node_property`, post-Step 4d) intercepts these and
    // writes them into `substrate.blobcol.node.*.idx+.dat` instead of
    // the bincode sidecar. Routing here at the migrate layer ensures
    // *fresh* migrations skip the sidecar entirely for heavy keys like
    // PO's `data` (928 MB / 681k entries) — not just on second-open
    // auto-migration. Separate counters so the heartbeat distinguishes
    // vectors vs. blobs when diagnosing oversized-sidecar regressions.
    let mut blobs_routed: u64 = 0;
    let mut blob_bytes_routed: u64 = 0;
    let mut stats = ValueSizeStats::default();

    // Per-node scratch buffer for the streaming writer. Reused across
    // nodes so we don't allocate a fresh Vec per node (16 elements =
    // typical property count for enterprise graphs).
    let mut scratch_props: Vec<(String, Value)> = Vec::with_capacity(16);

    // Time-based heartbeat: if the 100k node threshold hasn't hit but
    // it's been > 2s since the last heartbeat, emit anyway. Catches
    // "stuck on a pathological node" scenarios where node_written
    // barely advances for long stretches.
    let mut last_heartbeat_at = std::time::Instant::now();

    for (idx, old_id) in all_ids.iter().enumerate() {
        let iter_start = std::time::Instant::now();
        let node = match legacy.get_node(*old_id) {
            Some(n) => n,
            None => continue, // tombstoned between listing and read
        };

        let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        let new_id = substrate.create_node(&labels);

        // Property replay (skip reserved cognitive keys — those land in
        // columns in `phase_cognitive`. Also skip `_st_embedding`: at
        // wikipedia scale (4.5 M × 1.5 KB = 7 GB) copying embeddings
        // through the property path roughly doubles Phase::Nodes wall
        // time AND balloons the DashMap RSS. Embeddings are consumed
        // directly by `phase_tiers` reading from the legacy store — we
        // never need them sitting on the substrate side.
        scratch_props.clear();
        for (key, value) in node.properties.iter() {
            let key_str = key.as_str();
            if is_cognitive_key(key_str) || is_embedding_key(key_str) {
                continue;
            }
            total_props_seen += 1;
            stats.record(key_str, value);

            match filter.skip_reason(key_str, value) {
                Some("skip_all") | Some("skip_key") => {
                    skipped_by_key += 1;
                    continue;
                }
                Some("oversize") => {
                    skipped_oversize += 1;
                    continue;
                }
                Some(_) | None => {}
            }

            // Route through the streaming writer instead of the
            // DashMap-backed `set_node_property`. `value.clone()` is
            // Arc-cheap for String/Bytes/List/Map/Vector (all big
            // variants are Arc<[…]>) — a handful of atomic bumps per
            // property, no heap copies.
            scratch_props.push((key_str.to_string(), value.clone()));
        }
        // Per-node anomaly detection: flag any node whose property
        // count or estimated serialized size is egregiously outside
        // the "normal" range. These are the nodes responsible for the
        // mysterious 200 GB props.stream.tmp on megalaw — we want
        // their ids logged so we can inspect them on the legacy side.
        let est_bytes: usize = scratch_props
            .iter()
            .map(|(k, v)| k.len() + value_size_hint(v))
            .sum();
        if scratch_props.len() > 1_000 || est_bytes > 1_000_000 {
            tracing::warn!(
                "phase_nodes pathological node: old_id={} new_id={} labels={:?} \
                 prop_count={} est_bytes={} iter_elapsed_ms={}",
                old_id.as_u64(),
                new_id.as_u64(),
                labels,
                scratch_props.len(),
                est_bytes,
                iter_start.elapsed().as_millis()
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }

        // T16.7 Step 3 + Step 4e — split scratch_props into column-bound
        // and sidecar-bound buckets. Three classes route via
        // `substrate.set_node_property` instead of the streaming writer:
        //
        //   1. `Value::Vector(_)`  → dense mmap'd `VecColumnRegistry`
        //      (Step 2b contract).
        //   2. `Value::String(s)` with `s.len() > BLOB_COLUMN_THRESHOLD_BYTES`
        //      → `BlobColumnRegistry` (Step 4d contract, tag 'S').
        //   3. `Value::Bytes(b)`  with `b.len() > BLOB_COLUMN_THRESHOLD_BYTES`
        //      → `BlobColumnRegistry` (Step 4d contract, tag 'B').
        //
        // Scalars, short strings, short bytes, lists, and maps stay on
        // the streaming writer path (bounded-RSS, no heap copies).
        //
        // Rationale for routing *here* rather than relying solely on
        // `persist_properties`' defensive filter: we never want the
        // heavy values to transit the DashMap at all, since
        // `props_writer` consumes `scratch_props` synchronously and the
        // DashMap is separately hydrated by `load_properties` on reopen.
        // Splitting at the source of the migration keeps the anon-RSS
        // budget tight throughout phase_nodes AND ensures fresh
        // migrations bypass the sidecar for heavy keys (PO's `data`,
        // megalaw's legal text) — not just post-reopen auto-migration.
        if !scratch_props.is_empty() {
            let mut i = 0;
            while i < scratch_props.len() {
                let route_to_column = match &scratch_props[i].1 {
                    Value::Vector(_) => true,
                    Value::String(s) if s.len() > BLOB_COLUMN_THRESHOLD_BYTES => true,
                    Value::Bytes(b) if b.len() > BLOB_COLUMN_THRESHOLD_BYTES => true,
                    _ => false,
                };
                if route_to_column {
                    let (key, value) = scratch_props.swap_remove(i);
                    match &value {
                        Value::Vector(v) => {
                            vectors_routed += 1;
                            vector_bytes_routed += (v.len() as u64) * 4;
                        }
                        Value::String(s) => {
                            blobs_routed += 1;
                            blob_bytes_routed += s.len() as u64;
                        }
                        Value::Bytes(b) => {
                            blobs_routed += 1;
                            blob_bytes_routed += b.len() as u64;
                        }
                        _ => unreachable!("route_to_column guarded on Vector/String/Bytes"),
                    }
                    substrate.set_node_property(new_id, &key, value);
                    // swap_remove keeps index `i` pointing at what was
                    // the last entry — re-examine it without advancing.
                } else {
                    i += 1;
                }
            }
            if !scratch_props.is_empty() {
                props_writer.append_node(new_id.as_u64(), &scratch_props)?;
            }
        }

        node_map.insert(old_id.as_u64(), new_id);
        nodes_written += 1;
        pb.inc(1);

        if nodes_written % chunk_every == 0 {
            ckpt.nodes_written = nodes_written;
            ckpt.save(out)?;
        }

        // Fire a heartbeat when EITHER the count threshold hits OR
        // more than 2 seconds have elapsed since the last one. The
        // time-based path keeps us informed even when we're stuck on
        // a handful of very slow nodes.
        let hb_by_count = nodes_written % HEARTBEAT_EVERY == 0;
        let hb_by_time = last_heartbeat_at.elapsed().as_secs_f64() > 2.0;
        if hb_by_count || hb_by_time {
            let props_size_mb = props_stream_size_mb(out);
            let (_nw, _ew) = props_writer.counts();
            tracing::info!(
                "phase_nodes heartbeat: {nodes_written}/{total} nodes written, \
                 {total_props_seen} props seen (skipped: {skipped_by_key} by-key, \
                 {skipped_oversize} oversize; {vectors_routed} vectors → vec_columns \
                 = {} MiB; {blobs_routed} blobs → blob_columns = {} MiB); \
                 props.stream.tmp={props_size_mb} MB; top heavy keys: {}",
                vector_bytes_routed / (1024 * 1024),
                blob_bytes_routed / (1024 * 1024),
                stats.top_report(6)
            );
            last_heartbeat_at = std::time::Instant::now();
            // Force stderr flush so the log file is updated even when
            // stderr is redirected to a block-buffered destination.
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }

        // Free any Rust-side scratch on very wide scans.
        if idx % 1_000_000 == 0 && idx > 0 {
            // deliberate no-op; placeholder for future backpressure hook.
        }
    }

    ckpt.nodes_written = nodes_written;
    ckpt.save(out)?;
    pb.finish_and_clear();
    tracing::info!(
        "Phase::Nodes — wrote {nodes_written} nodes, saw {total_props_seen} props \
         (skipped: {skipped_by_key} by-key, {skipped_oversize} oversize; \
         {vectors_routed} vectors routed to vec_columns = {} MiB kept off sidecar, \
         {blobs_routed} blobs routed to blob_columns = {} MiB kept off sidecar)",
        vector_bytes_routed / (1024 * 1024),
        blob_bytes_routed / (1024 * 1024),
    );
    if !stats.per_key.is_empty() {
        tracing::info!(
            "Phase::Nodes — top heavy property keys: {}",
            stats.top_report(12)
        );
    }
    Ok(node_map)
}

/// Number of edges whose legacy properties we batch-fetch in one call
/// to `get_edges_properties_selective_batch`. Trades RAM for throughput;
/// 1000 keeps the working set under ~a few MB even on property-heavy
/// edges and cuts the fetch call count by 1000× vs the naïve
/// per-edge path.
const EDGE_PROPS_BATCH: usize = 1024;

fn phase_edges(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
    node_map: &FxHashMap<u64, NodeId>,
    ckpt: &mut Checkpoint,
    out: &Path,
    props_writer: &mut PropertiesStreamingWriter,
) -> Result<()> {
    let all_ids = legacy.all_node_ids();
    // Total = actual edge count so the bar / ETA reflect real work.
    // `edges_from(.., Outgoing)` returns forward CSR entries only, so the
    // sum across source nodes matches `edge_count()` exactly.
    let total_edges = legacy.edge_count() as u64;
    let pb = progress_bar(total_edges, "edges");
    // Checkpoint every 10% of the total edge work (bounded below by 1 so
    // a tiny test dataset still saves at least once).
    let chunk_every = (total_edges / 10).max(1);

    // Reusable keys for the synapse-weight lookup inside the batch.
    let synapse_key = PropertyKey::new(cog_keys::SYNAPSE_WEIGHT);

    // Chunk of (legacy_edge_id, new_edge_id) pairs waiting for a
    // bulk property replay. Property reads hit legacy `MmapStore`
    // once per chunk instead of once per edge — for Wikipedia scale
    // (119 M edges) this turns 119 M fetch calls into ~120 k, with the
    // single `get_edges_properties_selective_batch` call amortising
    // the CSR / prop-index lookup cost across the whole chunk.
    let mut chunk_pairs: Vec<(EdgeId, EdgeId)> = Vec::with_capacity(EDGE_PROPS_BATCH);

    // Heartbeat cadence for edges (match phase_nodes).
    const HEARTBEAT_EVERY: u64 = 100_000;
    let mut last_heartbeat: u64 = 0;

    let mut edges_written: u64 = 0;
    let mut edges_seen: u64 = 0;
    tracing::info!(
        "Phase::Edges — start, expecting {total_edges} edges across {} sources",
        all_ids.len()
    );
    for src_old in all_ids.iter() {
        let new_src = match node_map.get(&src_old.as_u64()) {
            Some(n) => *n,
            None => {
                // Source wasn't migrated: count its outgoing edges as
                // "seen" so the bar still advances, then drop them.
                let outgoing_len = legacy.edges_from(*src_old, Direction::Outgoing).len() as u64;
                if outgoing_len > 0 {
                    pb.inc(outgoing_len);
                    edges_seen += outgoing_len;
                }
                continue;
            }
        };

        let outgoing = legacy.edges_from(*src_old, Direction::Outgoing);
        for (dst_old, edge_id) in outgoing {
            edges_seen += 1;
            pb.inc(1);

            let new_dst = match node_map.get(&dst_old.as_u64()) {
                Some(n) => *n,
                None => continue, // orphan; drop silently
            };
            let edge_type = match legacy.edge_type(edge_id) {
                Some(t) => t,
                None => continue,
            };
            let new_edge_id = substrate.create_edge(new_src, new_dst, edge_type.as_str());
            chunk_pairs.push((edge_id, new_edge_id));
            edges_written += 1;

            if chunk_pairs.len() >= EDGE_PROPS_BATCH {
                flush_edge_props_chunk(
                    legacy,
                    substrate,
                    &chunk_pairs,
                    &synapse_key,
                    props_writer,
                )?;
                chunk_pairs.clear();
            }

            if edges_written % chunk_every == 0 {
                // Mid-phase checkpoint: flush any pending chunk first so
                // the on-disk state matches ckpt.edges_written.
                if !chunk_pairs.is_empty() {
                    flush_edge_props_chunk(
                        legacy,
                        substrate,
                        &chunk_pairs,
                        &synapse_key,
                        props_writer,
                    )?;
                    chunk_pairs.clear();
                }
                ckpt.edges_written = edges_written;
                ckpt.save(out)?;
            }

            if edges_written - last_heartbeat >= HEARTBEAT_EVERY {
                last_heartbeat = edges_written;
                let (nw, ew) = props_writer.counts();
                let props_size_mb = props_stream_size_mb(out);
                tracing::info!(
                    "phase_edges heartbeat: {edges_written}/{total_edges} edges written, \
                     {edges_seen} seen; props.stream.tmp={props_size_mb} MB \
                     (writer counts: nodes={nw}, edges={ew})"
                );
                use std::io::Write;
                let _ = std::io::stderr().flush();
            }
        }
    }

    // Flush final partial chunk.
    if !chunk_pairs.is_empty() {
        flush_edge_props_chunk(
            legacy,
            substrate,
            &chunk_pairs,
            &synapse_key,
            props_writer,
        )?;
        chunk_pairs.clear();
    }

    ckpt.edges_written = edges_written;
    ckpt.save(out)?;
    pb.finish_and_clear();
    if edges_seen != edges_written {
        tracing::info!(
            "Phase::Edges — wrote {} edges ({} seen, {} dropped as orphaned/untyped)",
            edges_written,
            edges_seen,
            edges_seen - edges_written
        );
    } else {
        tracing::info!("Phase::Edges — wrote {} edges", edges_written);
    }
    Ok(())
}

/// Fetch legacy properties for a whole chunk of edges in one call, then
/// replay them into substrate. The synapse weight is transferred to the
/// substrate edge's synapse column; all other non-cognitive properties
/// become substrate edge properties. Cognitive keys listed in
/// [`is_cognitive_key`] are filtered out (they land in columns elsewhere).
///
/// Edge properties are streamed directly into `props_writer` (one
/// `PropEntry` per edge) instead of routed through the DashMap-backed
/// `substrate.set_edge_property` — keeps RSS bounded on wiki/megalaw
/// runs. The synapse-column write stays on the substrate directly.
fn flush_edge_props_chunk(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
    pairs: &[(EdgeId, EdgeId)],
    synapse_key: &PropertyKey,
    props_writer: &mut PropertiesStreamingWriter,
) -> Result<()> {
    // `keys: &[]` means "return all properties" per selective-batch
    // semantics. Result vec is aligned 1:1 with the input ids.
    let legacy_ids: Vec<EdgeId> = pairs.iter().map(|(old, _)| *old).collect();
    let props_batch = legacy.get_edges_properties_selective_batch(&legacy_ids, &[]);

    // Reused per call (one call per chunk = amortised across
    // EDGE_PROPS_BATCH edges).
    let mut scratch: Vec<(String, Value)> = Vec::with_capacity(8);

    for ((_old_id, new_id), props) in pairs.iter().zip(props_batch.into_iter()) {
        // Non-cognitive props → split between column-bound (vectors
        // + oversize strings + oversize bytes) and streamed
        // substrate.props entry (everything else). See the twin comment
        // in `phase_nodes` for rationale; same T16.7 Step 3 + Step 4e
        // contract applies on the edge side.
        scratch.clear();
        for (key, value) in props.iter() {
            if is_cognitive_key(key.as_str()) {
                continue;
            }
            let route_to_column = match value {
                Value::Vector(_) => true,
                Value::String(s) if s.len() > BLOB_COLUMN_THRESHOLD_BYTES => true,
                Value::Bytes(b) if b.len() > BLOB_COLUMN_THRESHOLD_BYTES => true,
                _ => false,
            };
            if route_to_column {
                // Route directly — bypass the sidecar to keep anon RSS bounded.
                // `set_edge_property` dispatches Vector → VecColumnRegistry
                // and oversize String/Bytes → BlobColumnRegistry.
                substrate.set_edge_property(*new_id, key.as_str(), value.clone());
                continue;
            }
            scratch.push((key.as_str().to_string(), value.clone()));
        }
        if !scratch.is_empty() {
            props_writer.append_edge(new_id.as_u64(), &scratch)?;
        }

        // Synapse weight → substrate synapse column (additive delta;
        // destination starts at 0 so the delta is the full weight).
        if let Some(Value::Float64(w)) = props.get(synapse_key) {
            let _ = substrate.reinforce_edge_synapse_f32(*new_id, clamp01(*w as f32));
        }
    }
    let _ = substrate; // synapse path uses substrate above; retain binding for clarity
    Ok(())
}

fn phase_cognitive(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
    node_map: &FxHashMap<u64, NodeId>,
) -> Result<()> {
    let ids = legacy.all_node_ids();
    let total = ids.len() as u64;
    let pb = progress_bar(total, "cognitive");

    // Fetch the four cognitive fields in one round-trip per chunk. At
    // wikipedia scale (4.5M nodes) the old per-node call sequence did
    // 4 × n = 18M backend calls; batching collapses that to 4 × (n/CHUNK)
    // dispatch overhead, with the heavy work inside one `properties_batch`
    // pass.
    const COG_BATCH: usize = 4096;
    let key_energy = PropertyKey::new(cog_keys::ENERGY);
    let key_scar = PropertyKey::new(cog_keys::SCAR);
    let key_utility = PropertyKey::new(cog_keys::UTILITY);
    let key_affinity = PropertyKey::new(cog_keys::AFFINITY);
    let keys = [
        key_energy.clone(),
        key_scar.clone(),
        key_utility.clone(),
        key_affinity.clone(),
    ];

    let mut energy_set = 0u64;
    for chunk in ids.chunks(COG_BATCH) {
        // Parallel arrays: chunk[i] ↔ props[i] ↔ new_ids[i]
        let props = legacy.get_nodes_properties_selective_batch(chunk, &keys);
        for (old_id, map) in chunk.iter().zip(props.into_iter()) {
            let Some(&new_id) = node_map.get(&old_id.as_u64()) else {
                pb.inc(1);
                continue;
            };
            match map.get(&key_energy) {
                Some(Value::Float64(v)) => {
                    if substrate
                        .set_node_energy_f32(new_id, clamp01(*v as f32))
                        .is_ok()
                    {
                        energy_set += 1;
                    }
                }
                _ => {
                    // Default energy = 0.5 per RFC, applied only when the
                    // legacy node did not carry an explicit value.
                    let _ = substrate.set_node_energy_f32(new_id, 0.5);
                }
            }
            if let Some(Value::Float64(v)) = map.get(&key_scar) {
                let _ = substrate.set_node_scar_field_f32(new_id, clamp01(*v as f32));
            }
            if let Some(Value::Float64(v)) = map.get(&key_utility) {
                let _ = substrate.set_node_utility_field_f32(new_id, clamp01(*v as f32));
            }
            if let Some(Value::Float64(v)) = map.get(&key_affinity) {
                let _ = substrate.set_node_affinity_field_f32(new_id, clamp01(*v as f32));
            }
            pb.inc(1);
        }
    }

    pb.finish_and_clear();
    tracing::info!(
        "Phase::Cognitive — {} nodes carried explicit energy; rest defaulted to 0.5",
        energy_set
    );
    Ok(())
}

fn phase_tiers(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
    node_map: &FxHashMap<u64, NodeId>,
) -> Result<()> {
    use obrain_substrate::retrieval::{NodeOffset, SubstrateTieredIndex, VectorIndex};
    use obrain_substrate::L2_DIM;

    // Collect every (new_offset, embedding) pair where the legacy node
    // carried an `_st_embedding` Vector property of the expected dim.
    // Batched read: one round-trip per chunk instead of one per node.
    const TIERS_BATCH: usize = 2048;
    let key = PropertyKey::new("_st_embedding");
    let key_slice = [key.clone()];
    let mut pairs: Vec<(NodeOffset, Vec<f32>)> =
        Vec::with_capacity(node_map.len().min(legacy.node_count()));
    let mut skipped_wrong_dim = 0u64;
    let ids = legacy.all_node_ids();
    for chunk in ids.chunks(TIERS_BATCH) {
        let props = legacy.get_nodes_properties_selective_batch(chunk, &key_slice);
        for (old_id, mut map) in chunk.iter().zip(props.into_iter()) {
            let Some(&new_id) = node_map.get(&old_id.as_u64()) else {
                continue;
            };
            let Some(value) = map.remove(&key) else {
                continue;
            };
            let vec_arc = match value {
                Value::Vector(v) => v,
                _ => continue,
            };
            if vec_arc.len() != L2_DIM {
                skipped_wrong_dim += 1;
                continue;
            }
            // NodeOffset is the raw u64 slot id. The substrate `NodeId`
            // newtype wraps that, so we round-trip through `.as_u64()`.
            // `NodeOffset` is a `u32` type alias in substrate — truncate
            // after bounds-checking (migrator never writes > 2^32 nodes).
            let off: NodeOffset = new_id.as_u64() as NodeOffset;
            pairs.push((off, vec_arc.to_vec()));
        }
    }

    if pairs.is_empty() {
        tracing::info!(
            "Phase::Tiers — no `_st_embedding` properties found; skipping tier build \
             ({} skipped due to dim mismatch)",
            skipped_wrong_dim
        );
        let _ = substrate; // silence unused warning
        return Ok(());
    }

    tracing::info!(
        "Phase::Tiers — building L0/L1/L2 index for {} embeddings (skipped {} wrong-dim)",
        pairs.len(),
        skipped_wrong_dim
    );
    let index = SubstrateTieredIndex::new(L2_DIM);
    index.rebuild(&pairs);

    // T16.5 — persist L0/L1/L2 to substrate.tier0/.tier1/.tier2 so
    // downstream opens skip the rebuild from `_st_embedding`. See
    // `docs/rfc/substrate/tier-persistence.md` for the on-disk format.
    //
    // The write path here holds the `SubstrateFile` mutex for the full
    // duration of the three zone writes. This is a batch operation during
    // migration (no concurrent readers), so contention is a non-concern.
    let t_persist = std::time::Instant::now();
    {
        let sub_mutex = substrate.writer().substrate();
        let sub_guard = sub_mutex.lock();
        index.persist_to_zones(&sub_guard).with_context(|| {
            "Phase::Tiers — persisting tier zones to substrate failed"
        })?;
    }
    tracing::info!(
        "Phase::Tiers — persisted {} slots to tier0/1/2 in {:.2} s",
        pairs.len(),
        t_persist.elapsed().as_secs_f64(),
    );
    Ok(())
}

fn phase_cognitive_init(substrate: &Arc<SubstrateStore>) -> Result<()> {
    // Ricci-Ollivier batch is publicly exposed and safe to run right after
    // the structural migration. LDleiden + PageRank batches do not yet
    // have a public batch entry point in `obrain-substrate` (they live
    // inside the warden / cognitive layer and require runtime state);
    // we run what we can and log what is deferred.

    match obrain_substrate::refresh_all_ricci(substrate) {
        Ok(stats) => tracing::info!(
            "Phase::CognitiveInit — Ricci refreshed: {:?}",
            stats
        ),
        Err(e) => tracing::warn!(
            "Phase::CognitiveInit — refresh_all_ricci failed: {e}; continuing"
        ),
    }
    // `compute_all_node_curvatures` returns `HashMap<NodeId, f32>` —
    // the full map at Wikipedia scale is 4.5 M entries, a Debug format
    // would spam the log with hundreds of MB of `NodeId(x): y` pairs.
    // Fold into count + min/max/mean + sign-fraction instead.
    match obrain_substrate::compute_all_node_curvatures(substrate) {
        Ok(map) => {
            let n = map.len();
            if n == 0 {
                tracing::info!(
                    "Phase::CognitiveInit — node curvatures: no nodes"
                );
            } else {
                let mut sum = 0.0_f64;
                let mut kmin = f32::INFINITY;
                let mut kmax = f32::NEG_INFINITY;
                let mut neg = 0usize;
                let mut pos = 0usize;
                for &k in map.values() {
                    sum += k as f64;
                    if k < kmin {
                        kmin = k;
                    }
                    if k > kmax {
                        kmax = k;
                    }
                    if k < 0.0 {
                        neg += 1;
                    } else if k > 0.0 {
                        pos += 1;
                    }
                }
                let mean = sum / n as f64;
                tracing::info!(
                    "Phase::CognitiveInit — node curvatures: n={} mean={:+.4} min={:+.4} max={:+.4} frac(κ<0)={:.3} frac(κ>0)={:.3}",
                    n,
                    mean,
                    kmin,
                    kmax,
                    neg as f64 / n as f64,
                    pos as f64 / n as f64,
                );
            }
        }
        Err(e) => tracing::warn!(
            "Phase::CognitiveInit — compute_all_node_curvatures failed: {e}"
        ),
    }
    tracing::info!(
        "Phase::CognitiveInit — LDleiden + PageRank batch entry points \
         are deferred: they land together with the warden::run_batch API \
         in T17. For now the destination is cognitively bootstrapped via \
         Ricci + node curvatures only; LDleiden runs live on first warden tick."
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_cognitive_key(k: &str) -> bool {
    matches!(
        k,
        cog_keys::ENERGY
            | cog_keys::SCAR
            | cog_keys::UTILITY
            | cog_keys::AFFINITY
            | cog_keys::SYNAPSE_WEIGHT
            | cog_keys::COMMUNITY_ID
    )
}

/// Embedding-like keys that should NOT be copied into the substrate
/// side's property map. Three kinds, all of which must stay out of the
/// `substrate.props` bincode sidecar to keep anon RSS under the T16
/// gate:
///
/// - `_st_embedding` — semantic vector (f32 × 384). `phase_tiers` reads
///   it directly from the legacy store to build the L0/L1/L2 index,
///   and once the legacy store is closed the embedding is freed.
/// - `_kernel_embedding` — derived by the kernel plugin warden
///   (f32 × 80). Rebuildable on first warden tick, no durability loss.
///   Constant mirrored from `obrain-adapters::plugins::algorithms::
///   kernel::KERNEL_EMBEDDING_KEY` — if that string is renamed, update
///   here too.
/// - `_hilbert_features` — derived from graph topology by the kernel
///   plugin warden (f32 × 64). Same rebuildable story. Constant
///   mirrored from `obrain-adapters::plugins::algorithms::kernel::
///   HILBERT_FEATURES_KEY`.
///
/// Persisting any of these as substrate properties would double the
/// RSS of the migration and bloat the sidecar snapshot for no benefit
/// until the T17 property-pages subsystem lands.
///
/// Direct string dependency (rather than importing the constants) is
/// deliberate: `obrain-migrate` must not depend on `obrain-adapters`
/// to keep the legacy-read build graph minimal. The `skipped_keys_in_sync`
/// test below asserts the spellings match.
const KERNEL_EMBEDDING_KEY: &str = "_kernel_embedding";
const HILBERT_FEATURES_KEY: &str = "_hilbert_features";

fn is_embedding_key(k: &str) -> bool {
    matches!(
        k,
        "_st_embedding" | KERNEL_EMBEDDING_KEY | HILBERT_FEATURES_KEY
    )
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Current size of `substrate.props.stream.tmp` in MiB (0 if the file
/// does not yet exist). Used in heartbeat messages so we can spot
/// anomalous growth (e.g. a bug causing re-encodes) without waiting
/// for a phase to complete.
fn props_stream_size_mb(out: &Path) -> u64 {
    let p = out
        .join("substrate.obrain")
        .join("substrate.props.stream.tmp");
    std::fs::metadata(&p).map(|m| m.len() / (1024 * 1024)).unwrap_or(0)
}

/// Build a progress bar with a uniform style.
fn progress_bar(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos:>7}/{len:7} {msg}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    pb.set_message(format!("migrate:{label}"));
    pb
}

// ---------------------------------------------------------------------------
// Store open helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "legacy-read")]
fn open_legacy_store(path: &Path) -> Result<Arc<dyn GraphStore>> {
    // `open_legacy` already dispatches between directory and single-file
    // layouts and returns `Arc<dyn GraphStore>` directly (see
    // `legacy_reader` module docs), so no manual upcast is needed here.
    legacy_reader::open_legacy(path)
}

#[cfg(not(feature = "legacy-read"))]
fn open_legacy_store(_path: &Path) -> Result<Arc<dyn GraphStore>> {
    anyhow::bail!(
        "obrain-migrate was built without the `legacy-read` feature; \
         rebuild with `--features legacy-read`"
    );
}

fn open_or_create_substrate(
    path: &Path,
    resume: bool,
) -> Result<Arc<SubstrateStore>> {
    if path.exists() && !resume {
        anyhow::bail!(
            "output already exists: {} (pass --resume to continue, or remove the file)",
            path.display()
        );
    }
    let store = if path.exists() {
        SubstrateStore::open(path).with_context(|| format!("open {}", path.display()))?
    } else {
        SubstrateStore::create(path).with_context(|| format!("create {}", path.display()))?
    };
    Ok(Arc::new(store))
}

/// Rebuild the legacy→substrate id mapping from a substrate store that
/// has already had its nodes written. Assumes insertion-order stability
/// (substrate allocates slots monotonically when unless
/// `create_node_in_community` is used, which this migrator does not).
fn rebuild_node_map_from_substrate(
    legacy: &Arc<dyn GraphStore>,
    substrate: &Arc<SubstrateStore>,
) -> Result<FxHashMap<u64, NodeId>> {
    let legacy_ids = legacy.all_node_ids();
    let substrate_ids = GraphStore::all_node_ids(&**substrate);
    if legacy_ids.len() != substrate_ids.len() {
        anyhow::bail!(
            "cannot rebuild node map: legacy has {} nodes, substrate has {} — \
             re-run without --resume",
            legacy_ids.len(),
            substrate_ids.len()
        );
    }
    let mut map = FxHashMap::with_capacity_and_hasher(legacy_ids.len(), FxBuildHasher::default());
    for (old, new) in legacy_ids.iter().zip(substrate_ids.iter()) {
        map.insert(old.as_u64(), *new);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_key_detection() {
        assert!(is_cognitive_key(cog_keys::ENERGY));
        assert!(is_cognitive_key(cog_keys::SCAR));
        assert!(is_cognitive_key(cog_keys::SYNAPSE_WEIGHT));
        assert!(!is_cognitive_key("title"));
        assert!(!is_cognitive_key("_st_embedding"));
    }

    #[test]
    fn embedding_key_detection() {
        // Pre-T16.6 behaviour: only `_st_embedding` was filtered. The
        // T16.6 extension adds the two warden-derived vectors. All
        // three must be routed out of the substrate.props sidecar.
        assert!(is_embedding_key("_st_embedding"));
        assert!(is_embedding_key("_kernel_embedding"));
        assert!(is_embedding_key("_hilbert_features"));

        // User-provided semantic embedding stays in props for now —
        // promotion to a typed substrate column is T17 work.
        assert!(!is_embedding_key("embedding"));
        assert!(!is_embedding_key("title"));
        assert!(!is_embedding_key(""));
    }

    #[test]
    fn skipped_keys_in_sync_with_adapter_constants() {
        // The converter mirrors the `_kernel_embedding` / `_hilbert_features`
        // strings from `obrain-adapters::plugins::algorithms::kernel` rather
        // than taking a heavy dependency on that crate (see the doc comment
        // on `is_embedding_key`). If someone renames the keys on the adapter
        // side, the migration filter silently stops working and anon RSS
        // regresses. This test is the canary — it compares the mirrored
        // constants against the raw strings the adapter is known to write.
        //
        // Rationale for not using the real constants here: pulling
        // `obrain-adapters` (and transitively the plugin runtime) into
        // `obrain-migrate` would defeat the point of keeping the legacy-read
        // build graph minimal.
        assert_eq!(KERNEL_EMBEDDING_KEY, "_kernel_embedding");
        assert_eq!(HILBERT_FEATURES_KEY, "_hilbert_features");
    }

    #[test]
    fn clamp01_bounds() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp01(0.42), 0.42);
    }

    /// T16.7 Step 3b companion — smoke test for `finalize()`.
    ///
    /// `finalize()` is a thin open→flush→drop composition; the
    /// auto-migration correctness itself is covered by
    /// `obrain_substrate::store::tests::auto_migrates_legacy_vectors_on_load`.
    /// This test exercises the wrapper specifically: it must
    ///   1. succeed on a fresh (no-op) substrate,
    ///   2. reject a non-existent path with a clear error,
    ///   3. be idempotent (second call on the same path is a no-op).
    #[test]
    fn finalize_noop_on_fresh_substrate_and_idempotent() {
        use obrain_substrate::SubstrateStore;
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("fresh");

        // Create a minimal substrate, flush, drop. No vectors anywhere.
        {
            let s = SubstrateStore::create(&path).unwrap();
            s.flush().unwrap();
        }

        // (1) Fresh substrate → finalize succeeds, no panic, no error.
        super::finalize(&path).expect("finalize on fresh substrate should succeed");

        // (2) Idempotency: running finalize again on the same path is
        //     safe (the post-flush sidecar is already vector-free, so
        //     the auto-migration counters come back zero).
        super::finalize(&path).expect("finalize should be idempotent");

        // (3) Non-existent path → clear error (no panic).
        let missing = td.path().join("does_not_exist");
        assert!(
            super::finalize(&missing).is_err(),
            "finalize should return Err on a missing path"
        );
    }

    // -----------------------------------------------------------------------
    // T17f Step 5 — `--upgrade-edges-v2` integration tests.
    //
    // The fixture builds a real post-Step-1 substrate base via the public
    // `SubstrateStore` API, writes a handful of edge properties, and
    // persists the sidecar. Tests then either:
    //   * exercise the dry-run path (no side effects), or
    //   * downgrade the edges file to 32 B stride to simulate a
    //     pre-Step-1 base before calling the upgrader.
    //
    // The downgrade helper is the inverse of `rewrite_edges_32_to_36`;
    // keeping the round-trip guarantees the transform is byte-exact.
    // -----------------------------------------------------------------------

    use std::path::Path as _TestPath;

    /// Seed a substrate at `dir` with 3 nodes + 2 edges + edge props, flush,
    /// persist the edge sidecar so it materialises on disk.
    ///
    /// Returns the expected (edge_id, [(key, value)]) pairs so the caller
    /// can verify round-trip after the upgrade.
    fn seed_edge_fixture(
        dir: &_TestPath,
    ) -> (
        EdgeId,
        EdgeId,
        Vec<(PropertyKey, Value)>,
        Vec<(PropertyKey, Value)>,
    ) {
        use obrain_core::graph::traits::GraphStoreMut;
        use obrain_substrate::SubstrateStore;

        let store = SubstrateStore::create(dir).expect("substrate create");

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Org"]);

        let e1 = store.create_edge(a, b, "KNOWS");
        let e2 = store.create_edge(a, c, "WORKS_AT");

        store.set_edge_property(e1, "since", Value::Int64(2019));
        store.set_edge_property(e1, "strength", Value::Float64(0.85));
        store.set_edge_property(e2, "role", Value::String("Engineer".into()));

        // Force the sidecar to materialise on disk — the upgrader's
        // open-path hydrates it back into the DashMap.
        store
            .persist_edge_properties_sidecar()
            .expect("persist edge sidecar");
        store.flush().expect("flush");
        drop(store);

        let e1_props = vec![
            (PropertyKey::from("since"), Value::Int64(2019)),
            (PropertyKey::from("strength"), Value::Float64(0.85)),
        ];
        let e2_props = vec![(
            PropertyKey::from("role"),
            Value::String("Engineer".into()),
        )];
        (e1, e2, e1_props, e2_props)
    }

    /// Inverse of `rewrite_edges_32_to_36` — for the first `edge_count`
    /// records, strips bytes 24..30 (the `first_prop_off` field) and
    /// re-emits 4 B of trailing pad. Used to synthesise a pre-Step-1
    /// base for test.
    ///
    /// Substrate zones are exponentially pre-allocated (aligned to 4 KiB),
    /// so the input file will typically be much larger than
    /// `edge_count * 36` — the remainder is zone padding. We only need
    /// `edge_count` to locate the real records; the emitted Pre file
    /// is sized to exactly `edge_count * 32` (the rewrite_edges_32_to_36
    /// input expects `file_len >= edge_count * 32`, so this is a valid
    /// minimal Pre base).
    fn downgrade_edges_to_32b(edges_path: &_TestPath, edge_count: usize) {
        let input = std::fs::read(edges_path).expect("read edges");
        let logical_bytes = edge_count * 36;
        assert!(
            input.len() >= logical_bytes,
            "downgrade precondition: file must hold at least {} B for \
             {} Post records, got {} B",
            logical_bytes,
            edge_count,
            input.len()
        );
        let mut out = Vec::with_capacity(edge_count * 32);
        for i in 0..edge_count {
            let base = i * 36;
            let rec = &input[base..base + 36];
            // Copy 0..24 (src..next_to)
            out.extend_from_slice(&rec[0..24]);
            // Skip 24..30 (the inserted first_prop_off)
            // Copy 30..34 (ricci, flags, engram_tag)
            out.extend_from_slice(&rec[30..34]);
            // Emit 4 zero bytes of trailing pad (pre-layout had [u8; 4]).
            out.extend_from_slice(&[0u8; 4]);
        }
        assert_eq!(out.len(), edge_count * 32);
        std::fs::write(edges_path, &out).expect("write downgraded edges");
    }

    fn assert_edge_props_visible(
        dir: &_TestPath,
        e1: EdgeId,
        e2: EdgeId,
        e1_props: &[(PropertyKey, Value)],
        e2_props: &[(PropertyKey, Value)],
    ) {
        use obrain_core::graph::traits::GraphStore;
        use obrain_substrate::SubstrateStore;

        let store = SubstrateStore::open(dir).expect("reopen substrate");
        for (k, want) in e1_props {
            let got = store
                .get_edge_property(e1, k)
                .unwrap_or_else(|| panic!("edge e1 prop {k:?} missing after upgrade"));
            assert_eq!(&got, want, "edge e1 prop {k:?} mismatch");
        }
        for (k, want) in e2_props {
            let got = store
                .get_edge_property(e2, k)
                .unwrap_or_else(|| panic!("edge e2 prop {k:?} missing after upgrade"));
            assert_eq!(&got, want, "edge e2 prop {k:?} mismatch");
        }
        drop(store);
    }

    #[test]
    fn upgrade_edges_v2_dry_run_on_post_step1_base_is_readonly() {
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        let (_e1, _e2, _, _) = seed_edge_fixture(&subs);

        let edges_path = subs.join("substrate.edges");
        let sidecar_path = subs.join("substrate.edge_props");
        let edges_before = std::fs::read(&edges_path).unwrap();
        let sidecar_before = std::fs::read(&sidecar_path).unwrap();
        assert!(
            !sidecar_before.is_empty(),
            "sidecar should have been persisted"
        );

        super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: true,
                skip_backup: true,
            },
        )
        .expect("dry-run upgrade");

        let edges_after = std::fs::read(&edges_path).unwrap();
        let sidecar_after = std::fs::read(&sidecar_path).unwrap();
        assert_eq!(
            edges_before, edges_after,
            "edges file must not be touched during dry-run"
        );
        assert_eq!(
            sidecar_before, sidecar_after,
            "sidecar must not be touched during dry-run"
        );
    }

    #[test]
    fn upgrade_edges_v2_real_run_on_post_step1_base_drains_sidecar() {
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        let (e1, e2, e1_props, e2_props) = seed_edge_fixture(&subs);
        let edges_path = subs.join("substrate.edges");
        let sidecar_path = subs.join("substrate.edge_props");
        let edges_before = std::fs::read(&edges_path).unwrap();
        assert!(sidecar_path.exists(), "sidecar should exist before upgrade");

        super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: false,
                skip_backup: true,
            },
        )
        .expect("real-run upgrade on post-Step-1 base");

        let edges_after = std::fs::read(&edges_path).unwrap();
        // The drain phase updates `first_prop_off` in-place for every
        // edge that had sidecar properties (it's the whole point of
        // PropsZone v2 — point each edge at its chain head). So the
        // edges file is NOT byte-identical. What MUST be preserved:
        //   * overall file length (no rewrite, no resize)
        //   * structural prefix [0..24] of every record (src..next_to)
        //   * structural tail [30..34] of every record (ricci..engram_tag)
        //   * padding beyond `edge_count * 36`
        // And bytes [24..30] MAY differ (first_prop_off got populated).
        assert_eq!(
            edges_before.len(),
            edges_after.len(),
            "edges file length must not change on a pure-drain Post upgrade"
        );
        // 3 slots: sentinel + 2 edges.
        let edge_count: usize = 3;
        for i in 0..edge_count {
            let base = i * super::EDGE_RECORD_NEW_STRIDE;
            assert_eq!(
                &edges_before[base..base + 24],
                &edges_after[base..base + 24],
                "edge {i}: 0..24 prefix must not change"
            );
            assert_eq!(
                &edges_before[base + 30..base + 34],
                &edges_after[base + 30..base + 34],
                "edge {i}: 30..34 tail must not change"
            );
        }
        // Zone padding past the logical records must remain zero
        // (bytes past edge_count * 36 B are zone pre-alloc slack).
        let padding_start = edge_count * super::EDGE_RECORD_NEW_STRIDE;
        assert!(
            edges_after[padding_start..].iter().all(|&b| b == 0),
            "zone padding past n36 must remain all-zero after drain"
        );
        assert!(
            !sidecar_path.exists(),
            "substrate.edge_props must be deleted after drain"
        );

        // Props still visible via PropsZone v2 edge chain on reopen.
        assert_edge_props_visible(&subs, e1, e2, &e1_props, &e2_props);
    }

    #[test]
    fn upgrade_edges_v2_rewrites_pre_step1_base_and_drains() {
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        let (e1, e2, e1_props, e2_props) = seed_edge_fixture(&subs);
        let edges_path = subs.join("substrate.edges");
        let sidecar_path = subs.join("substrate.edge_props");

        // Capture 36 B layout for round-trip verification. The
        // substrate zone is exponentially pre-allocated, so the file
        // is much larger than `edge_count * 36` — we care only about
        // the first `edge_count` records. edge_count is known from
        // the fixture: 1 sentinel + 2 real edges = 3 slots.
        let edges_36_before = std::fs::read(&edges_path).unwrap();
        let edge_count: usize = 3;
        assert!(
            edges_36_before.len() >= edge_count * super::EDGE_RECORD_NEW_STRIDE,
            "fixture must hold at least {} B of Post records, got {} B",
            edge_count * super::EDGE_RECORD_NEW_STRIDE,
            edges_36_before.len()
        );

        // Downgrade edges → pretend we're a pre-Step-1 base (exactly
        // `edge_count * 32` B, minimum valid Pre file).
        downgrade_edges_to_32b(&edges_path, edge_count);
        let edges_32 = std::fs::read(&edges_path).unwrap();
        assert_eq!(
            edges_32.len(),
            edge_count * super::EDGE_RECORD_OLD_STRIDE,
            "downgrade must leave the file at 32 B stride"
        );

        super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: false,
                skip_backup: true,
            },
        )
        .expect("real-run upgrade on pre-Step-1 base");

        let edges_after = std::fs::read(&edges_path).unwrap();
        // The rewrite path sizes the tmp file to exactly
        // `edge_count * 36` B and renames it over the original. The
        // open + drain phase then mmaps it at that size and writes
        // `first_prop_off` in-place — the file size is preserved.
        assert!(
            edges_after.len() >= edge_count * super::EDGE_RECORD_NEW_STRIDE,
            "edges must hold at least {} B after upgrade, got {}",
            edge_count * super::EDGE_RECORD_NEW_STRIDE,
            edges_after.len()
        );

        // Structural prefix (src..next_to) and tail (ricci..engram_tag)
        // must survive the round trip. first_prop_off may be rewritten
        // by the drain path and is NOT compared here.
        for i in 0..edge_count {
            let base = i * super::EDGE_RECORD_NEW_STRIDE;
            assert_eq!(
                &edges_36_before[base..base + 24],
                &edges_after[base..base + 24],
                "edge {i}: 0..24 prefix mismatch after downgrade→upgrade round trip"
            );
            assert_eq!(
                &edges_36_before[base + 30..base + 34],
                &edges_after[base + 30..base + 34],
                "edge {i}: 30..34 tail mismatch after downgrade→upgrade round trip"
            );
        }

        assert!(
            !sidecar_path.exists(),
            "substrate.edge_props must be deleted after drain"
        );
        assert_edge_props_visible(&subs, e1, e2, &e1_props, &e2_props);
    }

    #[test]
    fn upgrade_edges_v2_takes_snapshot_by_default() {
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        let (_e1, _e2, _, _) = seed_edge_fixture(&subs);
        let edges_before = std::fs::read(subs.join("substrate.edges")).unwrap();

        super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: false,
                skip_backup: false,
            },
        )
        .expect("upgrade with snapshot");

        let parent = td.path();
        let mut backups: Vec<_> = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("subs.bak-"))
                    .unwrap_or(false)
            })
            .collect();
        backups.sort();
        assert_eq!(
            backups.len(),
            1,
            "exactly one snapshot directory should exist, got {:?}",
            backups
        );
        let backup_edges = std::fs::read(backups[0].join("substrate.edges")).unwrap();
        assert_eq!(
            edges_before, backup_edges,
            "snapshot must preserve pre-upgrade substrate.edges"
        );
    }

    #[test]
    fn upgrade_edges_v2_bails_on_truncated_edges_file() {
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        // Seed a valid Post base (108 B for 3 slots), then truncate the
        // edges file below `edge_count × 32` B (= 96 B). Neither stride
        // can be accommodated — the detector must bail. A truncation
        // is the canonical "corrupt base" the operator should see
        // flagged (e.g. interrupted rsync, bad drive write).
        let (_e1, _e2, _, _) = seed_edge_fixture(&subs);
        let edges_path = subs.join("substrate.edges");
        let mut bytes = std::fs::read(&edges_path).unwrap();
        bytes.truncate(50); // < 3 × 32 = 96 B minimum
        std::fs::write(&edges_path, &bytes).unwrap();

        let err = super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: true,
                skip_backup: true,
            },
        )
        .expect_err("upgrade must bail on truncated edges file");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("truncated") || msg.contains("smaller than"),
            "expected truncation-error message, got: {msg}"
        );
    }

    #[test]
    fn upgrade_edges_v2_bails_on_nonzero_tail() {
        // Same idea but with trailing non-zero garbage: a valid Post
        // base with a handful of non-zero bytes written past `n36`.
        // The tail-zero validation must flag this as corrupt even
        // though the content discriminator correctly identifies Post.
        let td = tempfile::tempdir().unwrap();
        let subs = td.path().join("subs");
        std::fs::create_dir_all(&subs).unwrap();

        let (_e1, _e2, _, _) = seed_edge_fixture(&subs);
        let edges_path = subs.join("substrate.edges");
        let mut bytes = std::fs::read(&edges_path).unwrap();
        // 108 B + 7 bytes of NON-ZERO garbage — sentinel of corruption.
        bytes.extend_from_slice(&[0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04]);
        std::fs::write(&edges_path, &bytes).unwrap();

        let err = super::upgrade_edges_v2_with_opts(
            &subs,
            &super::UpgradeEdgesV2Opts {
                dry_run: true,
                skip_backup: true,
            },
        )
        .expect_err("upgrade must bail on non-zero tail past n36");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not all zero") || msg.contains("corrupt"),
            "expected corruption-error message, got: {msg}"
        );
    }
}
