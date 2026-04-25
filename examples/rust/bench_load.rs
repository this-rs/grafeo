//! Benchmark tool: measures .obrain file open time and runs basic queries.
//!
//! Usage:
//!   cargo run -p obrain-examples --bin bench-load --features storage -- <file.obrain> [--queries]
//!
//! Options:
//!   --queries   Run sample queries after opening (default: just measure open time)

use std::path::PathBuf;
use std::time::Instant;

use obrain::ObrainDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench-load <file.obrain> [--queries]");
        eprintln!();
        eprintln!("Measures the open time of a .obrain file and optionally runs queries.");
        std::process::exit(1);
    }

    let db_path = PathBuf::from(&args[1]);
    let run_queries = args.iter().any(|a| a == "--queries");

    if !db_path.exists() {
        eprintln!("Error: file does not exist: {}", db_path.display());
        std::process::exit(1);
    }

    let file_size = std::fs::metadata(&db_path)?.len();
    println!("=== bench-load ===");
    println!(
        "File: {} ({:.2} MB)",
        db_path.display(),
        file_size as f64 / 1_048_576.0
    );
    println!();

    // ── Measure open time ──
    let t_open = Instant::now();
    let db = ObrainDB::open(&db_path)?;
    let open_us = t_open.elapsed().as_micros();
    let open_ms = open_us as f64 / 1000.0;

    let session = db.session();

    // Count nodes and edges
    let t_count = Instant::now();
    let node_count: i64 = session.execute("MATCH (n) RETURN COUNT(n)")?.scalar()?;
    let edge_count: i64 = session
        .execute("MATCH ()-[r]->() RETURN COUNT(r)")?
        .scalar()?;
    let count_ms = t_count.elapsed().as_millis();

    println!("Open time:   {open_ms:.1}ms");
    println!("Nodes:       {node_count}");
    println!("Edges:       {edge_count}");
    println!("Count query: {count_ms}ms");

    if run_queries {
        println!("\n--- Sample queries ---");

        // Query 1: Label distribution
        let t = Instant::now();
        let result = session.execute(
            "MATCH (n) RETURN labels(n)[0] AS label, COUNT(n) AS cnt ORDER BY cnt DESC LIMIT 10",
        )?;
        let ms = t.elapsed().as_millis();
        println!("\nLabel distribution ({ms}ms):");
        for row in &result.rows {
            println!("  {}: {}", row[0], row[1]);
        }

        // Query 2: Edge type distribution
        let t = Instant::now();
        let result = session.execute(
            "MATCH ()-[r]->() RETURN type(r) AS t, COUNT(r) AS cnt ORDER BY cnt DESC LIMIT 10",
        )?;
        let ms = t.elapsed().as_millis();
        println!("\nEdge type distribution ({ms}ms):");
        for row in &result.rows {
            println!("  {}: {}", row[0], row[1]);
        }

        // Query 3: Sample traversal
        let t = Instant::now();
        let result =
            session.execute("MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 5")?;
        let ms = t.elapsed().as_millis();
        println!("\nSample traversal ({ms}ms):");
        for row in &result.rows {
            println!("  {} --[{}]--> {}", row[0], row[1], row[2]);
        }
    }

    println!("\n=== Summary ===");
    println!("File size:  {:.2} MB", file_size as f64 / 1_048_576.0);
    println!("Open time:  {open_ms:.1}ms");
    if open_ms < 100.0 {
        println!("Status:     ✓ FAST (< 100ms target)");
    } else if open_ms < 1000.0 {
        println!("Status:     ~ MODERATE (< 1s)");
    } else {
        println!("Status:     ✗ SLOW ({:.1}s)", open_ms / 1000.0);
    }

    Ok(())
}
