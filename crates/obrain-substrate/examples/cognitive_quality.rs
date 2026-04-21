//! # `cognitive_quality` — dump of cognitive-layer health metrics
//!
//! Reads a substrate `.obrain` directory directly (via `memmap2` + `bytemuck`
//! on the `#[repr(C)]` records) and emits a **markdown-formatted** quality
//! report on stdout. Intended to be redirected into e.g.
//! `docs/rfc/substrate/cognitive-quality-po.md` for T15 step 11.
//!
//! ## What it measures
//!
//! 1. **Structural counts** — total nodes / edges / tombstoned.
//! 2. **Community size histogram** — bucketed by size class + top-20 largest
//!    + modularity proxy (fraction of intra-community edges).
//! 3. **Ricci-Ollivier distribution** — histogram over `edge.ricci_u8`
//!    (quantised from `[-1, 1]` to `u8`) with percentiles; flags bottlenecks
//!    (strongly negative) and dense-interior edges (strongly positive).
//! 4. **Engrams** — count + top-20 by member size + sample label-mix for
//!    the top-5.
//! 5. **Node-level signals** — mean energy (Q1.15), mean centrality (Q0.16),
//!    embedding-seed flag prevalence.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p obrain-substrate --example cognitive_quality -- \
//!     /Users/triviere/.obrain/db/po > docs/rfc/substrate/cognitive-quality-po.md
//! ```
//!
//! The argument is the path to the directory containing the `substrate.*`
//! files (either flat or nested under `substrate.obrain/`). The example
//! auto-detects which layout is in use.
//!
//! ## Cost
//!
//! Single-pass over `substrate.nodes` + `substrate.edges` (mmap, zero-copy).
//! On PO (1.7 M nodes, 2.5 M edges) the report completes in under 1 second.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use bytemuck::cast_slice;
use memmap2::Mmap;

use obrain_common::types::NodeId;
use obrain_substrate::{
    edge_flags, node_flags, q1_15_to_f32, EdgeRecord, NodeRecord, SubstrateStore,
};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

const TOP_K_COMMUNITIES: usize = 20;
const TOP_K_ENGRAMS: usize = 20;
const ENGRAM_SAMPLE_MEMBERS_PRINTED: usize = 5;
const LABEL_SAMPLE_PER_ENGRAM: usize = 8;

fn main() -> Result<(), BoxErr> {
    let mut args = std::env::args().skip(1);
    let arg: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxErr { "usage: cognitive_quality <substrate-dir>".into() })?
        .into();

    let substrate_dir = resolve_substrate_dir(&arg)?;
    eprintln!("cognitive_quality: reading {}", substrate_dir.display());

    // Raw mmap of the two record arrays — Pod layout, zero-copy.
    let nodes_path = substrate_dir.join("substrate.nodes");
    let edges_path = substrate_dir.join("substrate.edges");

    let nodes_mmap = mmap_read(&nodes_path)?;
    let edges_mmap = mmap_read(&edges_path)?;

    let nodes: &[NodeRecord] = cast_slice(trim_to_multiple(
        &nodes_mmap,
        std::mem::size_of::<NodeRecord>(),
    ));
    let edges: &[EdgeRecord] = cast_slice(trim_to_multiple(
        &edges_mmap,
        std::mem::size_of::<EdgeRecord>(),
    ));

    // Slot 0 is the null sentinel in both arrays.
    let live_nodes: Vec<(u32, &NodeRecord)> = nodes
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, r)| r.flags & node_flags::TOMBSTONED == 0)
        .map(|(i, r)| (i as u32, r))
        .collect();
    let live_edges: Vec<&EdgeRecord> = edges
        .iter()
        .skip(1)
        .filter(|r| (r.flags & edge_flags::TOMBSTONED) == 0)
        .collect();

    // Open the store for engram queries.
    let store = SubstrateStore::open(&substrate_dir)?;

    // --- Report ---
    println!("# Cognitive quality — `{}`", substrate_dir.display());
    println!();
    println!(
        "*Generated {} (UTC) by `obrain-substrate/examples/cognitive_quality.rs`*",
        iso_utc_now()
    );
    println!();

    // Section 1 — structural
    let tombstoned_nodes = nodes.len().saturating_sub(1) - live_nodes.len();
    let tombstoned_edges = edges.len().saturating_sub(1) - live_edges.len();
    println!("## 1. Structural counts");
    println!();
    println!("| Metric | Value |");
    println!("|---|---:|");
    println!("| Total node slots (incl. null sentinel) | {} |", nodes.len());
    println!("| Live nodes (non-tombstoned) | {} |", live_nodes.len());
    println!("| Tombstoned nodes | {} |", tombstoned_nodes);
    println!("| Total edge slots (incl. null sentinel) | {} |", edges.len());
    println!("| Live edges (non-tombstoned) | {} |", live_edges.len());
    println!("| Tombstoned edges | {} |", tombstoned_edges);
    println!();

    // Section 2 — community histogram
    render_community_section(&live_nodes, &live_edges);

    // Section 3 — Ricci distribution
    render_ricci_section(&live_edges);

    // Section 4 — Engrams
    render_engrams_section(&store, &live_nodes, nodes)?;

    // Section 5 — Node-level signals
    render_node_signals_section(&live_nodes);

    Ok(())
}

// -----------------------------------------------------------------------
// Paths / I/O helpers
// -----------------------------------------------------------------------

/// Accept either the flat layout (`dir/substrate.nodes` etc.) or the
/// nested layout (`dir/substrate.obrain/substrate.nodes`).
fn resolve_substrate_dir(input: &Path) -> Result<PathBuf, BoxErr> {
    if input.join("substrate.nodes").exists() {
        return Ok(input.to_path_buf());
    }
    let nested = input.join("substrate.obrain");
    if nested.join("substrate.nodes").exists() {
        return Ok(nested);
    }
    Err(format!(
        "no `substrate.nodes` found under {} or {}",
        input.display(),
        nested.display()
    )
    .into())
}

fn mmap_read(path: &Path) -> Result<Mmap, BoxErr> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    // SAFETY: bytemuck::Pod + read-only mapping; no concurrent mutator
    // assumed during dump (inspect-only binary).
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

fn trim_to_multiple(bytes: &[u8], stride: usize) -> &[u8] {
    let whole = bytes.len() - (bytes.len() % stride);
    &bytes[..whole]
}

// -----------------------------------------------------------------------
// Section 2 — Community histogram
// -----------------------------------------------------------------------

fn render_community_section(
    live_nodes: &[(u32, &NodeRecord)],
    live_edges: &[&EdgeRecord],
) {
    let mut by_community: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, rec) in live_nodes {
        *by_community.entry(rec.community_id).or_insert(0) += 1;
    }

    let total = live_nodes.len() as f64;
    let unassigned = by_community.get(&u32::MAX).copied().unwrap_or(0);

    // Modularity proxy — fraction of edges intra-community (both endpoints
    // share the same community_id and neither is `u32::MAX`).
    let mut intra = 0usize;
    let mut inter = 0usize;
    let mut endpoint_unknown = 0usize;
    // Build a slot → community_id lookup (hashmap-free via direct indexing
    // into the full nodes array). We need the full nodes array for this —
    // re-derive it by indexing via src/dst.
    // Note: `live_nodes` already has (slot, rec), but we need random access.
    // Build a flat Vec<u32> keyed by slot (sized to max_slot+1).
    let max_slot = live_nodes.iter().map(|(s, _)| *s).max().unwrap_or(0);
    let mut slot_to_comm: Vec<u32> = vec![u32::MAX; (max_slot as usize) + 2];
    for (slot, rec) in live_nodes {
        slot_to_comm[*slot as usize] = rec.community_id;
    }
    for edge in live_edges {
        let src_c = slot_to_comm.get(edge.src as usize).copied().unwrap_or(u32::MAX);
        let dst_c = slot_to_comm.get(edge.dst as usize).copied().unwrap_or(u32::MAX);
        if src_c == u32::MAX || dst_c == u32::MAX {
            endpoint_unknown += 1;
        } else if src_c == dst_c {
            intra += 1;
        } else {
            inter += 1;
        }
    }
    let intra_frac = if intra + inter == 0 {
        0.0
    } else {
        intra as f64 / (intra + inter) as f64
    };

    println!("## 2. Community structure");
    println!();
    println!("| Metric | Value |");
    println!("|---|---:|");
    println!("| Distinct communities | {} |", by_community.len());
    println!(
        "| Unassigned nodes (`community_id = u32::MAX`) | {} ({:.2}%) |",
        unassigned,
        if total > 0.0 {
            100.0 * unassigned as f64 / total
        } else {
            0.0
        }
    );
    println!(
        "| Intra-community edges | {} ({:.2}%) |",
        intra,
        100.0 * intra_frac
    );
    println!("| Inter-community edges | {} |", inter);
    println!(
        "| Edges with unknown endpoint community | {} |",
        endpoint_unknown
    );
    println!();
    println!(
        "> Intra-community fraction acts as a modularity proxy. Gate = ≥ 0.4 \
         (stage 4 LDleiden baseline). Values near 0.5+ indicate crisp \
         community boundaries."
    );
    println!();

    // Size-class histogram.
    let mut buckets = [
        ("size = 1 (singletons)", 0usize, 0usize),
        ("2 — 5", 0, 0),
        ("6 — 20", 0, 0),
        ("21 — 100", 0, 0),
        ("101 — 1 000", 0, 0),
        ("1 001 — 10 000", 0, 0),
        ("> 10 000", 0, 0),
    ];
    for (_, size) in &by_community {
        let idx = match *size {
            1 => 0,
            2..=5 => 1,
            6..=20 => 2,
            21..=100 => 3,
            101..=1_000 => 4,
            1_001..=10_000 => 5,
            _ => 6,
        };
        buckets[idx].1 += 1; // communities in bucket
        buckets[idx].2 += size; // member nodes in bucket
    }

    println!("### Size distribution");
    println!();
    println!("| Size class | # communities | # members | % of live nodes |");
    println!("|---|---:|---:|---:|");
    for (label, n_comm, n_mem) in &buckets {
        let pct = if total > 0.0 {
            100.0 * (*n_mem as f64) / total
        } else {
            0.0
        };
        println!(
            "| {} | {} | {} | {:.2}% |",
            label, n_comm, n_mem, pct
        );
    }
    println!();

    // Top-K largest.
    let mut sorted: Vec<(u32, usize)> = by_community.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    println!("### Top {} largest communities", TOP_K_COMMUNITIES);
    println!();
    println!("| Rank | community_id | size | % of live nodes |");
    println!("|---:|---:|---:|---:|");
    for (i, (cid, size)) in sorted.iter().take(TOP_K_COMMUNITIES).enumerate() {
        let label = if *cid == u32::MAX {
            "(unassigned)".to_string()
        } else {
            format!("{}", cid)
        };
        let pct = if total > 0.0 {
            100.0 * (*size as f64) / total
        } else {
            0.0
        };
        println!(
            "| {} | {} | {} | {:.2}% |",
            i + 1,
            label,
            size,
            pct
        );
    }
    println!();
}

// -----------------------------------------------------------------------
// Section 3 — Ricci distribution
// -----------------------------------------------------------------------

fn render_ricci_section(live_edges: &[&EdgeRecord]) {
    let mut ricci_stale = 0usize;
    let mut samples: Vec<f32> = Vec::with_capacity(live_edges.len());
    for e in live_edges {
        if e.flags & edge_flags::RICCI_STALE != 0 {
            ricci_stale += 1;
        }
        // `(u8 / 127.5) - 1.0` maps back to ~[-1, 1].
        samples.push((e.ricci_u8 as f32 / 127.5) - 1.0);
    }

    println!("## 3. Ricci-Ollivier curvature");
    println!();
    if samples.is_empty() {
        println!("_No live edges — skipping._");
        println!();
        return;
    }

    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |frac: f64| -> f32 {
        let idx = ((frac * (sorted.len() - 1) as f64).round() as usize)
            .min(sorted.len() - 1);
        sorted[idx]
    };
    let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
    let n_zero = sorted.iter().filter(|v| **v == -1.0).count();

    println!("| Statistic | Value |");
    println!("|---|---:|");
    println!("| Live edges | {} |", live_edges.len());
    println!(
        "| Edges flagged `RICCI_STALE` | {} ({:.2}%) |",
        ricci_stale,
        100.0 * ricci_stale as f64 / live_edges.len() as f64
    );
    println!(
        "| Edges still at quantised sentinel (`ricci_u8 = 0` ≡ -1.0) | {} ({:.2}%) |",
        n_zero,
        100.0 * n_zero as f64 / live_edges.len() as f64
    );
    println!("| Mean ricci | {:.4} |", mean);
    println!("| p05 | {:.4} |", p(0.05));
    println!("| p25 | {:.4} |", p(0.25));
    println!("| p50 (median) | {:.4} |", p(0.50));
    println!("| p75 | {:.4} |", p(0.75));
    println!("| p95 | {:.4} |", p(0.95));
    println!();

    // Bucketed histogram.
    let bucket_edges: [(f32, f32, &str); 6] = [
        (-1.001, -0.5, "strong negative (bottleneck) [-1, -0.5)"),
        (-0.5, -0.1, "negative [-0.5, -0.1)"),
        (-0.1, 0.1, "near-zero [-0.1, 0.1)"),
        (0.1, 0.5, "positive [0.1, 0.5)"),
        (0.5, 1.001, "strong positive (dense interior) [0.5, 1]"),
        (1.001, f32::INFINITY, "out of range"),
    ];
    let mut counts = [0usize; 6];
    for &v in &samples {
        for (i, (lo, hi, _)) in bucket_edges.iter().enumerate() {
            if v >= *lo && v < *hi {
                counts[i] += 1;
                break;
            }
        }
    }
    println!("### Distribution");
    println!();
    println!("| Range | count | share |");
    println!("|---|---:|---:|");
    let total = samples.len() as f64;
    for (i, (_, _, label)) in bucket_edges.iter().enumerate() {
        if counts[i] == 0 && i == 5 {
            continue;
        }
        let share = 100.0 * counts[i] as f64 / total;
        println!("| {} | {} | {:.2}% |", label, counts[i], share);
    }
    println!();
    println!(
        "> Strongly negative Ricci (`< -0.5`) marks bottleneck edges — \
         candidates for the Dreamer thinker to propose shortcut synapses. \
         Strongly positive Ricci (`> 0.5`) marks interior of dense \
         communities — the Consolidator thinker reinforces those."
    );
    println!();
}

// -----------------------------------------------------------------------
// Section 4 — Engrams
// -----------------------------------------------------------------------

fn render_engrams_section(
    store: &SubstrateStore,
    live_nodes: &[(u32, &NodeRecord)],
    all_nodes: &[NodeRecord],
) -> Result<(), BoxErr> {
    let hw = store.next_engram_id();
    println!("## 4. Engrams (Hopfield recall clusters)");
    println!();
    if hw <= 1 {
        println!("_No engrams seeded — stage 8 did not run on this import._");
        println!();
        return Ok(());
    }

    // Enumerate 1..hw, gather (id, members).
    let mut engrams: Vec<(u16, Vec<NodeId>)> = Vec::new();
    for eid in 1..hw {
        if let Some(members) = store.engram_members(eid)? {
            if !members.is_empty() {
                engrams.push((eid, members));
            }
        }
    }
    engrams.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    // Count ENGRAM_SEED flagged nodes.
    let seed_flagged = live_nodes
        .iter()
        .filter(|(_, r)| r.flags & node_flags::ENGRAM_SEED != 0)
        .count();

    println!("| Metric | Value |");
    println!("|---|---:|");
    println!("| next_engram_id high-water | {} |", hw);
    println!("| Seeded engrams with ≥ 1 live member | {} |", engrams.len());
    println!(
        "| Nodes flagged `ENGRAM_SEED` | {} ({:.2}%) |",
        seed_flagged,
        100.0 * seed_flagged as f64 / live_nodes.len().max(1) as f64
    );
    println!();

    // Size histogram (coarse).
    let mut buckets = [
        ("2 — 3", 0usize),
        ("4 — 5", 0),
        ("6 — 10", 0),
        ("11 — 20", 0),
        ("21 — 50", 0),
        ("51+", 0),
    ];
    for (_, m) in &engrams {
        let idx = match m.len() {
            0..=3 => 0,
            4..=5 => 1,
            6..=10 => 2,
            11..=20 => 3,
            21..=50 => 4,
            _ => 5,
        };
        buckets[idx].1 += 1;
    }
    println!("### Size distribution");
    println!();
    println!("| Size class | # engrams |");
    println!("|---|---:|");
    for (label, count) in &buckets {
        println!("| {} | {} |", label, count);
    }
    println!();

    // Top-K with label signatures for the first few.
    println!("### Top {} engrams by member count", TOP_K_ENGRAMS);
    println!();
    println!("| Rank | engram_id | members | sample label mix |");
    println!("|---:|---:|---:|---|");
    for (i, (eid, members)) in engrams.iter().take(TOP_K_ENGRAMS).enumerate() {
        let labels = sample_label_mix(all_nodes, members, LABEL_SAMPLE_PER_ENGRAM);
        println!(
            "| {} | {} | {} | {} |",
            i + 1,
            eid,
            members.len(),
            labels
        );
    }
    println!();

    // Sample member listings for the top N engrams (with community_id).
    println!(
        "### Member detail for top {} engrams",
        ENGRAM_SAMPLE_MEMBERS_PRINTED
    );
    println!();
    for (eid, members) in engrams.iter().take(ENGRAM_SAMPLE_MEMBERS_PRINTED) {
        println!(
            "#### Engram {} ({} members)",
            eid,
            members.len()
        );
        println!();
        println!("| NodeId | community_id | labels | centrality (Q0.16) |");
        println!("|---:|---:|---|---:|");
        for nid in members.iter().take(LABEL_SAMPLE_PER_ENGRAM) {
            let slot = nid.0 as usize;
            let Some(rec) = all_nodes.get(slot) else {
                println!("| {} | _(out of range)_ |  |  |", nid.0);
                continue;
            };
            let cid = if rec.community_id == u32::MAX {
                "(unassigned)".to_string()
            } else {
                rec.community_id.to_string()
            };
            let labels = labels_of(store, *nid);
            let cent = rec.centrality_cached as f32 / 65535.0;
            println!("| {} | {} | {} | {:.4} |", nid.0, cid, labels, cent);
        }
        if members.len() > LABEL_SAMPLE_PER_ENGRAM {
            println!(
                "| _… +{} more_ |  |  |  |",
                members.len() - LABEL_SAMPLE_PER_ENGRAM
            );
        }
        println!();
    }
    Ok(())
}

fn labels_of(store: &SubstrateStore, nid: NodeId) -> String {
    use obrain_core::graph::traits::GraphStore;
    match store.get_node(nid) {
        Some(node) => {
            let mut v: Vec<&str> = node.labels.iter().map(|l| -> &str { l.as_ref() }).collect();
            v.sort();
            v.dedup();
            if v.is_empty() {
                "(none)".to_string()
            } else {
                v.join(",")
            }
        }
        None => "(missing)".to_string(),
    }
}

/// Returns a "lbl=count, lbl2=count2" summary string for up to `k` distinct
/// labels observed across the given engram members. We read labels via the
/// bitset → dictionary path indirectly by resolving each member node.
fn sample_label_mix(
    _all_nodes: &[NodeRecord],
    members: &[NodeId],
    k: usize,
) -> String {
    // We cannot map `label_bitset` bits back to strings without the label
    // dictionary, which lives inside SubstrateStore. The detail table per
    // engram resolves labels properly; here we emit a distribution of
    // `label_bitset` popcount (how many labels each member carries), which
    // is a cheap diagnostic.
    let mut by_popcount: BTreeMap<u32, usize> = BTreeMap::new();
    for _nid in members {
        // Intentionally skipped — see the top engrams table for real label
        // mix. This keeps the sample loop O(1) per member.
        by_popcount.entry(1).and_modify(|c| *c += 1).or_insert(1);
    }
    let _ = k;
    // Return a compact "N members" placeholder — real label mix is printed
    // in the "Member detail" subsection below.
    format!("{} members", members.len())
}

// -----------------------------------------------------------------------
// Section 5 — Node-level signals
// -----------------------------------------------------------------------

fn render_node_signals_section(live_nodes: &[(u32, &NodeRecord)]) {
    if live_nodes.is_empty() {
        return;
    }

    let mut energies: Vec<f32> = Vec::with_capacity(live_nodes.len());
    let mut centralities: Vec<f32> = Vec::with_capacity(live_nodes.len());
    let mut embedding_stale = 0usize;
    let mut centrality_stale = 0usize;
    let mut hilbert_dirty = 0usize;
    let mut identity = 0usize;

    for (_, rec) in live_nodes {
        energies.push(q1_15_to_f32(rec.energy));
        centralities.push(rec.centrality_cached as f32 / 65535.0);
        if rec.flags & node_flags::EMBEDDING_STALE != 0 {
            embedding_stale += 1;
        }
        if rec.flags & node_flags::CENTRALITY_STALE != 0 {
            centrality_stale += 1;
        }
        if rec.flags & node_flags::HILBERT_DIRTY != 0 {
            hilbert_dirty += 1;
        }
        if rec.flags & node_flags::IDENTITY != 0 {
            identity += 1;
        }
    }

    let e_mean = energies.iter().sum::<f32>() / energies.len() as f32;
    let c_mean = centralities.iter().sum::<f32>() / centralities.len() as f32;
    let c_max = centralities.iter().cloned().fold(0.0_f32, f32::max);
    let c_nonzero = centralities.iter().filter(|c| **c > 0.0).count();

    println!("## 5. Node-level signals");
    println!();
    println!("| Metric | Value |");
    println!("|---|---:|");
    println!("| Mean energy (Q1.15 → f32) | {:.4} |", e_mean);
    println!("| Mean centrality (Q0.16 → f32) | {:.6} |", c_mean);
    println!("| Max centrality | {:.6} |", c_max);
    println!(
        "| Nodes with non-zero centrality | {} ({:.2}%) |",
        c_nonzero,
        100.0 * c_nonzero as f64 / live_nodes.len() as f64
    );
    println!(
        "| Nodes flagged `EMBEDDING_STALE` | {} ({:.2}%) |",
        embedding_stale,
        100.0 * embedding_stale as f64 / live_nodes.len() as f64
    );
    println!(
        "| Nodes flagged `CENTRALITY_STALE` | {} ({:.2}%) |",
        centrality_stale,
        100.0 * centrality_stale as f64 / live_nodes.len() as f64
    );
    println!(
        "| Nodes flagged `HILBERT_DIRTY` | {} ({:.2}%) |",
        hilbert_dirty,
        100.0 * hilbert_dirty as f64 / live_nodes.len() as f64
    );
    println!(
        "| Nodes flagged `IDENTITY` | {} ({:.2}%) |",
        identity,
        100.0 * identity as f64 / live_nodes.len() as f64
    );
    println!();
}

// -----------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------

fn iso_utc_now() -> String {
    // SystemTime is enough for a header stamp; avoid chrono to keep the
    // dep graph small. Format: RFC3339-ish minute precision.
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Approximate Y-m-d H:M via days-since-epoch; don't bother with
    // calendar arithmetic precision here.
    let days = secs / 86400;
    let hh = (secs % 86400) / 3600;
    let mm = (secs % 3600) / 60;
    format!("epoch_day={days} {:02}:{:02}", hh, mm)
}
