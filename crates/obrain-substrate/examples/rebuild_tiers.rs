//! # `rebuild_tiers` — rebuild tier zones (tier0/1/2) on an existing substrate
//!
//! Use case: you have a `SubstrateStore` whose `_st_embedding` properties
//! are already populated (e.g. a migration was done before T16.5 landed,
//! or a base was imported without `--with-tiers`), but its tier zones
//! on disk are empty / stale. This binary scans `_st_embedding` props
//! over the whole store, rebuilds a fresh `SubstrateTieredIndex`, and
//! persists it via `persist_to_zones`.
//!
//! The operation is **in-place** and does not touch nodes, edges, props,
//! or the WAL. Only `substrate.tier0`, `substrate.tier1`, and
//! `substrate.tier2` are (re)written.
//!
//! ## Usage
//! ```text
//! cargo run --release -p obrain-substrate --example rebuild_tiers -- \
//!     /Users/me/.obrain/db/po
//! ```
//!
//! Optional second arg: `--dry-run` — only count eligible embeddings,
//! do not persist.

use std::path::PathBuf;

use obrain_common::{PropertyKey, Value};
use obrain_core::graph::traits::GraphStore;
use obrain_substrate::SubstrateStore;
use obrain_substrate::retrieval::{NodeOffset, SubstrateTieredIndex, VectorIndex};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

const L2_DIM: usize = 384;

fn main() -> Result<(), BoxErr> {
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args
        .next()
        .ok_or_else(|| -> BoxErr { "usage: rebuild_tiers <substrate-dir> [--dry-run]".into() })?
        .into();
    let dry_run = args.next().map(|s| s == "--dry-run").unwrap_or(false);

    eprintln!("rebuild_tiers: opening {}", path.display());
    let t_open = std::time::Instant::now();
    let store = SubstrateStore::open(&path)?;
    eprintln!(
        "rebuild_tiers: opened in {:.2}s",
        t_open.elapsed().as_secs_f64()
    );

    let n_nodes = store.node_count();
    eprintln!("rebuild_tiers: store has {n_nodes} nodes");

    // --- Phase 1: scan all _st_embedding props -------------------------
    let emb_key = PropertyKey::new("_st_embedding");
    let node_ids = store.node_ids();
    eprintln!("rebuild_tiers: {} node_ids enumerated", node_ids.len());

    let t_scan = std::time::Instant::now();
    let mut pairs: Vec<(NodeOffset, Vec<f32>)> = Vec::with_capacity(node_ids.len());
    let mut scanned = 0usize;
    let mut wrong_shape = 0usize;
    let mut absent = 0usize;

    for nid in &node_ids {
        scanned += 1;
        if scanned % 250_000 == 0 {
            eprintln!(
                "rebuild_tiers: scanned {}/{} (found {} embeddings, {:.1}s elapsed)",
                scanned,
                node_ids.len(),
                pairs.len(),
                t_scan.elapsed().as_secs_f64()
            );
        }

        let Some(v) = store.get_node_property(*nid, &emb_key) else {
            absent += 1;
            continue;
        };
        match v {
            Value::Vector(vec) => {
                if vec.len() == L2_DIM {
                    // NodeId is u64 but substrate uses the low 32 bits as
                    // the slot/offset (see store.rs: `id.0 as u32`).
                    // Value::Vector wraps Arc<[f32]>; deref + to_vec materialises an owned Vec<f32>.
                    pairs.push((nid.0 as NodeOffset, vec.to_vec()));
                } else {
                    wrong_shape += 1;
                }
            }
            _ => wrong_shape += 1,
        }
    }

    eprintln!(
        "rebuild_tiers: scan done in {:.2}s — {} eligible, {} wrong_shape, {} absent",
        t_scan.elapsed().as_secs_f64(),
        pairs.len(),
        wrong_shape,
        absent
    );

    if pairs.is_empty() {
        eprintln!("rebuild_tiers: nothing to persist, aborting");
        return Err("no _st_embedding @ L2_DIM=384 found in the store".into());
    }

    let coverage = pairs.len() as f64 / node_ids.len() as f64;
    eprintln!(
        "rebuild_tiers: coverage = {:.2}% ({} / {})",
        coverage * 100.0,
        pairs.len(),
        node_ids.len()
    );

    if dry_run {
        eprintln!("rebuild_tiers: --dry-run specified, not persisting");
        return Ok(());
    }

    // --- Phase 2: rebuild the index ------------------------------------
    let t_build = std::time::Instant::now();
    let index = SubstrateTieredIndex::new(L2_DIM);
    index.rebuild(&pairs);
    eprintln!(
        "rebuild_tiers: rebuild() done in {:.2}s (len={})",
        t_build.elapsed().as_secs_f64(),
        index.len()
    );

    // --- Phase 3: persist to tier0/1/2 zones ---------------------------
    let t_persist = std::time::Instant::now();
    {
        let sub_mutex = store.writer().substrate();
        let sub_guard = sub_mutex.lock();
        index.persist_to_zones(&sub_guard)?;
    }
    eprintln!(
        "rebuild_tiers: persist_to_zones done in {:.2}s",
        t_persist.elapsed().as_secs_f64()
    );

    // --- Phase 4: summary ----------------------------------------------
    println!();
    println!("=== rebuild_tiers: DONE ===");
    println!("store         : {}", path.display());
    println!("nodes         : {}", n_nodes);
    println!("embeddings    : {} ({:.2}%)", pairs.len(), coverage * 100.0);
    println!("wrong_shape   : {wrong_shape}");
    println!("absent        : {absent}");
    println!("index.len()   : {}", index.len());
    println!("total elapsed : {:.2}s", t_open.elapsed().as_secs_f64());
    Ok(())
}
