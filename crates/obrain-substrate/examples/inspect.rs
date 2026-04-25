//! # `inspect` — quick structural QA of a migrated `.obrain` substrate
//!
//! Opens a `SubstrateStore` directory and reports:
//! * node / edge counts
//! * label distribution (top-K)
//! * edge-type distribution (top-K, sampled)
//! * embedding coverage at `L2_DIM = 384` (`_st_embedding`), sampled
//! * a handful of sample nodes with their label set and first few properties
//!
//! Intended as a migration-verification tool — run it right after
//! `obrain-migrate` or `neo4j2obrain` finishes, before trusting the output
//! for downstream cognitive workloads.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p obrain-substrate --example inspect -- \
//!     /tmp/wikipedia-substrate.obrain/substrate.obrain
//! ```
//!
//! A second optional argument tunes the sample size used for per-node
//! inspection (default: 20000). Counts (`node_count`, `edge_count`,
//! label distribution) are always full-scan.

use std::collections::BTreeMap;
use std::path::PathBuf;

use obrain_common::{PropertyKey, Value};
use obrain_core::graph::Direction;
use obrain_core::graph::traits::GraphStore;
use obrain_substrate::SubstrateStore;

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

const L2_DIM: usize = 384;
const DEFAULT_SAMPLE: usize = 20_000;
const SAMPLE_NODES_PRINTED: usize = 5;
const TOP_K: usize = 20;
const PROP_VAL_TRUNC: usize = 120;

fn main() -> Result<(), BoxErr> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxErr { "usage: inspect <substrate-dir> [sample-size]".into() })?
        .into();
    let sample: usize = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_SAMPLE);

    eprintln!("inspect: opening {}", path.display());
    let t0 = std::time::Instant::now();
    let store = SubstrateStore::open(&path)?;
    eprintln!("inspect: opened in {:.2}s", t0.elapsed().as_secs_f64());

    // ------- Structural counts (full-scan) -------
    let t = std::time::Instant::now();
    let n_nodes = store.node_count();
    let n_edges = store.edge_count();
    eprintln!(
        "inspect: counts computed in {:.2}s",
        t.elapsed().as_secs_f64()
    );

    println!();
    println!("=== Substrate: {} ===", path.display());
    println!("nodes : {n_nodes}");
    println!("edges : {n_edges}");

    // ------- Full node_ids scan for label distribution -------
    let t = std::time::Instant::now();
    let node_ids = store.node_ids();
    eprintln!(
        "inspect: node_ids() in {:.2}s ({} ids)",
        t.elapsed().as_secs_f64(),
        node_ids.len()
    );

    // Sample IDs (deterministic stride — no rand dep).
    let sample_ids: Vec<_> = if node_ids.len() <= sample {
        node_ids.clone()
    } else {
        let step = node_ids.len() / sample;
        node_ids
            .iter()
            .copied()
            .step_by(step.max(1))
            .take(sample)
            .collect()
    };

    println!("sample: {} nodes (full scan over labels)", sample_ids.len());

    // ------- Label distribution (over sample) -------
    let mut label_hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut nodes_with_any_label = 0usize;
    for &nid in &sample_ids {
        if let Some(node) = store.get_node(nid) {
            if !node.labels.is_empty() {
                nodes_with_any_label += 1;
            }
            for lab in node.labels.iter() {
                let s: &str = lab.as_ref();
                *label_hist.entry(s.to_string()).or_insert(0) += 1;
            }
        }
    }

    println!();
    println!(
        "=== labels (top {TOP_K}, over sample={}) ===",
        sample_ids.len()
    );
    println!("nodes_with_any_label : {nodes_with_any_label}");
    print_top_k(&label_hist, TOP_K);

    // ------- Embedding coverage (L2_DIM = 384, over sample) -------
    let emb_key = PropertyKey::new("_st_embedding");
    let mut emb_present = 0usize;
    let mut emb_wrong_dim = 0usize;
    for &nid in &sample_ids {
        if let Some(v) = store.get_node_property(nid, &emb_key) {
            match v {
                Value::Vector(vec) => {
                    if vec.len() == L2_DIM {
                        emb_present += 1;
                    } else {
                        emb_wrong_dim += 1;
                    }
                }
                _ => emb_wrong_dim += 1,
            }
        }
    }
    let coverage = if sample_ids.is_empty() {
        0.0
    } else {
        emb_present as f64 / sample_ids.len() as f64
    };

    println!();
    println!(
        "=== _st_embedding coverage (sample={}) ===",
        sample_ids.len()
    );
    println!(
        "present@L2_DIM={L2_DIM} : {emb_present}  ({:.2}%)",
        coverage * 100.0
    );
    println!("wrong_shape/type      : {emb_wrong_dim}");
    println!(
        "absent                : {}",
        sample_ids.len().saturating_sub(emb_present + emb_wrong_dim)
    );

    // ------- Edge-type distribution (over sampled edges via sampled src nodes) -------
    // Substrate exposes `edge_type(EdgeId)` but not a full `edge_ids()` scan
    // in this example scope. We sample edges by walking outgoing chains from
    // a capped number of sampled nodes — good enough to get a
    // representative edge_type distribution at very low cost.
    let et_hist = edge_type_histogram(&store, &sample_ids);
    println!();
    println!(
        "=== edge_type (top {TOP_K}, {} edges sampled) ===",
        et_hist.values().sum::<usize>()
    );
    print_top_k(&et_hist, TOP_K);

    // ------- Sample nodes -------
    println!();
    println!("=== sample nodes ({SAMPLE_NODES_PRINTED}) ===");
    let step = sample_ids.len() / SAMPLE_NODES_PRINTED.max(1);
    let picks: Vec<_> = (0..SAMPLE_NODES_PRINTED)
        .filter_map(|i| sample_ids.get(i * step.max(1)).copied())
        .collect();
    for nid in picks {
        let Some(node) = store.get_node(nid) else {
            println!("  [{:?}] <missing>", nid);
            continue;
        };
        let labels: Vec<&str> = node.labels.iter().map(|l| -> &str { l.as_ref() }).collect();
        println!(
            "  [{:?}] labels={:?}  props={}",
            nid,
            labels,
            node.properties.len()
        );
        let mut shown = 0usize;
        for (k, v) in node.properties.iter() {
            if shown >= 6 {
                println!("    ... (+{} more)", node.properties.len() - shown);
                break;
            }
            // Skip the big embedding vector — summarise instead.
            if k.as_str() == "_st_embedding" {
                match v {
                    Value::Vector(vec) => println!(
                        "    _st_embedding : Vector(dim={}, [0..3]={:.3?})",
                        vec.len(),
                        &vec[..vec.len().min(3)]
                    ),
                    other => println!("    _st_embedding : {other:?}"),
                }
                shown += 1;
                continue;
            }
            println!("    {} : {}", k.as_str(), display_value(v, PROP_VAL_TRUNC));
            shown += 1;
        }
    }

    println!();
    println!("inspect: done in {:.2}s total", t0.elapsed().as_secs_f64());
    Ok(())
}

fn edge_type_histogram(
    store: &SubstrateStore,
    sample_ids: &[obrain_common::types::NodeId],
) -> BTreeMap<String, usize> {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    // Walk outgoing edges for the first 1000 sample ids to keep the probe
    // cheap. Uses the GraphStore::out_edges API if present; we route via
    // `get_node` adjacency otherwise. Substrate exposes out_edges on its
    // GraphStore impl.
    let probe_n = sample_ids.len().min(1000);
    for &nid in &sample_ids[..probe_n] {
        for (_, eid) in store.edges_from(nid, Direction::Outgoing) {
            if let Some(t) = store.edge_type(eid) {
                let s: &str = t.as_ref();
                *hist.entry(s.to_string()).or_insert(0) += 1;
            }
        }
    }
    hist
}

fn print_top_k(hist: &BTreeMap<String, usize>, k: usize) {
    if hist.is_empty() {
        println!("  (empty)");
        return;
    }
    let mut items: Vec<(&String, &usize)> = hist.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1));
    let total: usize = hist.values().sum();
    for (name, count) in items.iter().take(k) {
        let pct = **count as f64 / total.max(1) as f64 * 100.0;
        println!("  {:>10}  ({:5.2}%)  {}", count, pct, name);
    }
    if items.len() > k {
        println!("  ... (+{} more)", items.len() - k);
    }
}

fn display_value(v: &Value, max: usize) -> String {
    let raw = match v {
        Value::String(s) => {
            let s_ref: &str = s.as_ref();
            format!("{:?}", s_ref)
        }
        Value::Bytes(b) => format!("Bytes(len={})", b.len()),
        Value::Vector(vec) => format!("Vector(dim={})", vec.len()),
        other => format!("{other:?}"),
    };
    if raw.len() > max {
        let mut end = max;
        while !raw.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}… ({} chars)", &raw[..end], raw.len())
    } else {
        raw
    }
}
