//! # `checkpoint_cli` — flush the mmap zones and truncate the WAL.
//!
//! Opens an existing substrate directory (which replays the WAL into the
//! in-memory zone mmaps), then calls [`checkpoint`] to msync/fsync every
//! zone and rewrite `substrate.meta` atomically. The WAL is truncated to
//! a single Checkpoint marker, so the next open is O(1) instead of
//! O(wal-size).
//!
//! Used by T16 to amortise the one-off WAL-replay cost on production bases
//! (PO, Wikipedia, megalaw) before measuring startup / RSS gates.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p obrain-substrate --example checkpoint_cli -- \
//!     /Users/triviere/.obrain/db/po
//! ```
//!
//! The argument is the path to the directory containing the `substrate.*`
//! files (either flat or nested under `substrate.obrain/`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use obrain_substrate::{SubstrateFile, checkpoint, wal_io::SyncMode, writer::Writer};

fn resolve_substrate_dir(base: &Path) -> PathBuf {
    let nested = base.join("substrate.obrain");
    if nested.join("substrate.meta").exists() {
        return nested;
    }
    base.to_path_buf()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: checkpoint_cli <substrate-dir>");
        std::process::exit(2);
    }

    let base = PathBuf::from(&args[1]);
    let dir = resolve_substrate_dir(&base);
    println!("substrate : {}", dir.display());

    // Measure WAL + zone file sizes before, for reporting.
    let wal_path = dir.join("substrate.wal");
    let wal_before = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    let nodes_before = std::fs::metadata(dir.join("substrate.nodes"))
        .map(|m| m.len())
        .unwrap_or(0);
    let edges_before = std::fs::metadata(dir.join("substrate.edges"))
        .map(|m| m.len())
        .unwrap_or(0);
    let props_before = std::fs::metadata(dir.join("substrate.props"))
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "before    : wal={:.2} GiB  nodes={:.2} MiB  edges={:.2} MiB  props={:.2} GiB",
        wal_before as f64 / (1024.0 * 1024.0 * 1024.0),
        nodes_before as f64 / (1024.0 * 1024.0),
        edges_before as f64 / (1024.0 * 1024.0),
        props_before as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // --- open writer to materialise a WalWriter pointing at substrate.wal ----
    // The Writer opens the WAL with next_lsn = meta.last_wal_offset + 1, which
    // is fine for checkpoint (checkpoint writes a fresh marker and truncates).
    let t_open_w = Instant::now();
    let sub_for_writer = SubstrateFile::open(&dir)?;
    let w = Writer::new(sub_for_writer, SyncMode::EveryCommit)?;
    let wal_writer = w.wal();
    drop(w);
    println!(
        "open+wal  : {:.2} ms",
        t_open_w.elapsed().as_secs_f64() * 1000.0
    );

    // --- reopen substrate fresh → replays WAL into mmap zones ----------------
    // This is the expensive part — same replay cost that T16 measures on open.
    let t_replay = Instant::now();
    let mut sub2 = SubstrateFile::open(&dir)?;
    println!("replay    : {:.2} s", t_replay.elapsed().as_secs_f64());

    // --- checkpoint: flush zones, rewrite meta atomically, truncate WAL ------
    let t_ckpt = Instant::now();
    let stats = checkpoint(&mut sub2, &wal_writer)?;
    println!(
        "checkpoint: {:.2} s  (wal_offset_before={} at_lsn={} ts={})",
        t_ckpt.elapsed().as_secs_f64(),
        stats.wal_offset_before,
        stats.at_lsn,
        stats.checkpoint_timestamp
    );

    drop(sub2);
    drop(wal_writer);

    // --- after sizes ---------------------------------------------------------
    let wal_after = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    let nodes_after = std::fs::metadata(dir.join("substrate.nodes"))
        .map(|m| m.len())
        .unwrap_or(0);
    let edges_after = std::fs::metadata(dir.join("substrate.edges"))
        .map(|m| m.len())
        .unwrap_or(0);
    let props_after = std::fs::metadata(dir.join("substrate.props"))
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "after     : wal={:.2} MiB  nodes={:.2} MiB  edges={:.2} MiB  props={:.2} GiB",
        wal_after as f64 / (1024.0 * 1024.0),
        nodes_after as f64 / (1024.0 * 1024.0),
        edges_after as f64 / (1024.0 * 1024.0),
        props_after as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "wal delta : {} -> {} bytes ({:.2}x reduction)",
        wal_before,
        wal_after,
        wal_before as f64 / wal_after.max(1) as f64,
    );

    Ok(())
}
