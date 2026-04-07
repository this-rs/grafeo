#!/usr/bin/env -S cargo +nightly -Zscript
//! Benchmark database boot time: WAL replay vs mmap epoch files.
//!
//! Usage: cargo run --release --features "tiered-storage,wal,gql" --example bench_boot -- /path/to/db

use std::time::Instant;

fn main() {
    let db_path = std::env::args().nth(1).expect("Usage: bench_boot <db_path>");
    let path = std::path::Path::new(&db_path);

    println!("=== Obrain Boot Benchmark ===");
    println!("Database: {}", path.display());
    println!();

    // Check if epochs/ directory exists
    let epochs_dir = path.join("epochs");
    let has_epochs = epochs_dir.exists()
        && std::fs::read_dir(&epochs_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);

    if has_epochs {
        println!("Epoch files found — measuring mmap + partial WAL replay...");
    } else {
        println!("No epoch files — measuring full WAL replay...");
    }

    // Open database and measure time
    let start = Instant::now();
    let db = obrain_engine::ObrainDB::open(path).expect("failed to open database");
    let elapsed = start.elapsed();

    let stats = db.compute_statistics();
    println!("Boot time: {:.3}s", elapsed.as_secs_f64());
    println!("Nodes: {}", stats.node_count);
    println!("Edges: {}", stats.edge_count);
    println!();

    // If no epochs exist, offer to compact
    if !has_epochs {
        println!("Running compact to create epoch files...");
        #[cfg(feature = "tiered-storage")]
        {
            let compact_start = Instant::now();
            match db.compact() {
                Ok(epoch_path) => {
                    let compact_elapsed = compact_start.elapsed();
                    println!(
                        "Compact done in {:.1}s — epoch file: {}",
                        compact_elapsed.as_secs_f64(),
                        epoch_path.display()
                    );
                    let file_size = std::fs::metadata(&epoch_path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    println!("Epoch file size: {:.1} MB", file_size as f64 / 1_048_576.0);
                }
                Err(e) => {
                    eprintln!("Compact failed: {e}");
                }
            }
        }
        #[cfg(not(feature = "tiered-storage"))]
        {
            eprintln!("tiered-storage feature not enabled, cannot compact");
        }
    }

    db.close().expect("close");
}
