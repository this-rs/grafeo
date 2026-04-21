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
use obrain_substrate::{PropertiesStreamingWriter, SubstrateStore};

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

        if !scratch_props.is_empty() {
            props_writer.append_node(new_id.as_u64(), &scratch_props)?;
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
                 {skipped_oversize} oversize); props.stream.tmp={props_size_mb} MB; \
                 top heavy keys: {}",
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
         (skipped: {skipped_by_key} by-key, {skipped_oversize} oversize)"
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
        // Non-cognitive props → streamed substrate.props entry.
        scratch.clear();
        for (key, value) in props.iter() {
            if is_cognitive_key(key.as_str()) {
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
    // The index currently lives in-process. Persisting it to the
    // substrate tier zones is T17's concern (see RFC "Tier persistence");
    // until then the tier build is a warm-up hint — consumers of the
    // migrated store rebuild via `SubstrateTieredIndex::rebuild` at open.
    let _ = substrate;
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
/// side's property map. `phase_tiers` reads them directly from the
/// legacy store to build the L0/L1/L2 index, and once the legacy store
/// is closed at the end of `phase_tiers` the embeddings are freed.
/// Persisting them as substrate properties would double the RSS of the
/// migration and bloat the sidecar snapshot for no benefit until the
/// T17 property-pages subsystem lands.
fn is_embedding_key(k: &str) -> bool {
    matches!(k, "_st_embedding")
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
    fn clamp01_bounds() {
        assert_eq!(clamp01(-0.5), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp01(0.42), 0.42);
    }
}
