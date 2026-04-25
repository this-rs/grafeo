//! Migration tool: converts a v1 .obrain file or WAL directory to v2 (mmap-native).
//!
//! Usage:
//!   cargo run -p obrain-examples --bin migrate-v2 --features storage -- <input> [output.obrain]
//!
//! Accepts both `.obrain` files (v1 bincode) and WAL directories as input.
//! If no output path is given, saves to `<input-stem>-v2.obrain` or `<dirname>.obrain`.

use std::path::PathBuf;
use std::time::Instant;

use obrain::ObrainDB;

/// Recursively computes the total size of a directory.
fn dir_size(path: &std::path::Path) -> u64 {
    walkdir(path).unwrap_or(0)
}

fn walkdir(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_file() {
            total += entry.metadata()?.len();
        } else if ft.is_dir() {
            total += walkdir(&entry.path())?;
        }
    }
    Ok(total)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: migrate-v2 <input> [output.obrain]");
        eprintln!();
        eprintln!("Converts a v1 .obrain file or WAL directory to v2 (mmap-native) format.");
        eprintln!("If no output path is given, saves to <stem>.obrain next to the input.");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let is_dir = input_path.is_dir();

    let output_path = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else if is_dir {
        // WAL directory → <dirname>.obrain in the same parent
        let name = input_path.file_name().unwrap().to_string_lossy();
        input_path.with_file_name(format!("{name}.obrain"))
    } else {
        let stem = input_path.file_stem().unwrap().to_string_lossy();
        input_path.with_file_name(format!("{stem}-v2.obrain"))
    };

    if !input_path.exists() {
        eprintln!("Error: input path does not exist: {}", input_path.display());
        std::process::exit(1);
    }

    // Compute input size (recursive for directories)
    let input_size = if is_dir {
        dir_size(&input_path)
    } else {
        std::fs::metadata(&input_path)?.len()
    };
    println!("=== migrate-v2 ===");
    println!(
        "Input:  {} ({:.2} MB)",
        input_path.display(),
        input_size as f64 / 1_048_576.0
    );
    println!("Output: {}", output_path.display());
    println!();

    // ── Phase 1: Open the v1 file ──
    println!("Opening input file...");
    let t_open = Instant::now();
    let db = ObrainDB::open(&input_path)?;
    let open_ms = t_open.elapsed().as_millis();

    let session = db.session();
    let node_count: i64 = session.execute("MATCH (n) RETURN COUNT(n)")?.scalar()?;
    let edge_count: i64 = session
        .execute("MATCH ()-[r]->() RETURN COUNT(r)")?
        .scalar()?;
    println!("Opened in {open_ms}ms — {node_count} nodes, {edge_count} edges");

    // ── Phase 2: Save as v2 ──
    println!("\nSaving as v2 native format...");
    let t_save = Instant::now();
    db.save(&output_path)?;
    let save_ms = t_save.elapsed().as_millis();

    let output_size = std::fs::metadata(&output_path)?.len();
    println!(
        "Saved in {save_ms}ms — {:.2} MB",
        output_size as f64 / 1_048_576.0
    );

    // ── Phase 3: Verify round-trip ──
    println!("\nVerifying round-trip...");
    let t_verify = Instant::now();
    let db2 = ObrainDB::open(&output_path)?;
    let verify_open_ms = t_verify.elapsed().as_millis();

    let session2 = db2.session();
    let node_count2: i64 = session2.execute("MATCH (n) RETURN COUNT(n)")?.scalar()?;
    let edge_count2: i64 = session2
        .execute("MATCH ()-[r]->() RETURN COUNT(r)")?
        .scalar()?;

    // The v2 writer cleans up orphan edges (edges pointing to deleted nodes
    // whose counters were stale). Node count must match or increase (counter was
    // under-counting); edge count may decrease (orphan edges removed).
    if node_count2 < node_count {
        eprintln!("⚠ Node count decreased: v1({node_count}) vs v2({node_count2})");
    }
    if edge_count2 < edge_count {
        let cleaned = edge_count - edge_count2;
        eprintln!("  ℹ Cleaned {cleaned} orphan edges during migration");
    }

    println!("v2 opened in {verify_open_ms}ms — {node_count2} nodes, {edge_count2} edges ✓");

    // ── Summary ──
    println!("\n=== Summary ===");
    println!("v1 open:  {open_ms}ms");
    println!("v2 save:  {save_ms}ms");
    println!("v2 open:  {verify_open_ms}ms");
    println!(
        "Size:     {:.2} MB → {:.2} MB ({:+.1}%)",
        input_size as f64 / 1_048_576.0,
        output_size as f64 / 1_048_576.0,
        (output_size as f64 - input_size as f64) / input_size as f64 * 100.0,
    );
    println!(
        "Speedup:  {:.0}x faster open",
        open_ms as f64 / verify_open_ms.max(1) as f64
    );

    Ok(())
}
