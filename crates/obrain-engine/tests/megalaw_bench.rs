//! Benchmark the bottlenecks in obrain-chat "searching graph..." step
//! on the megalaw database (~21GB epoch file, 8.1M nodes, 14M edges).
//!
//! Run with:
//!   cargo test -p obrain-engine --features "cypher,wal,tiered-storage" --test megalaw_bench -- --nocapture

use obrain_core::graph::Direction;
use obrain_engine::ObrainDB;
use std::time::Instant;

#[test]
fn bench_megalaw_retrieval_steps() {
    let home = std::env::var("HOME").unwrap_or_default();
    let db_path = std::path::PathBuf::from(&home).join(".obrain/db/megalaw");

    if !db_path.exists() {
        println!("  Skipping — ~/.obrain/db/megalaw not found");
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("MEGALAW DATABASE — RETRIEVAL BOTTLENECK ANALYSIS");
    println!("{}\n", "=".repeat(70));

    // ── Phase 0: DB Open ──
    let start = Instant::now();
    let db = ObrainDB::open(&db_path).expect("open megalaw db");
    let open_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("  DB open: {:.1}ms", open_ms);

    let store = db.store();

    // ── Phase 1: Basic stats via direct API (not Cypher) ──
    println!("\n--- Database Stats (direct API) ---");

    let start = Instant::now();
    let nc = store.node_count();
    println!("  node_count():  {} ({:.3}ms)", nc, start.elapsed().as_secs_f64() * 1000.0);

    let start = Instant::now();
    let ec = store.edge_count();
    println!("  edge_count():  {} ({:.3}ms)", ec, start.elapsed().as_secs_f64() * 1000.0);

    // ── Phase 2: Label index performance ──
    println!("\n--- Label Index (nodes_by_label) ---");

    let labels = store.all_labels();
    println!("  {} distinct labels", labels.len());

    for label in &labels {
        // O(1) count
        let start = Instant::now();
        let count = store.node_count_by_label(label);
        let count_ms = start.elapsed().as_secs_f64() * 1000.0;

        // O(n) allocation
        let start = Instant::now();
        let nodes = store.nodes_by_label(label);
        let alloc_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "    {:<20} count={:>10} ({:.3}ms)  alloc={:>10.3}ms",
            label, count, count_ms, alloc_ms
        );
        assert_eq!(count, nodes.len(), "count mismatch for {}", label);
    }

    // ── Phase 3: Property access on Document nodes ──
    println!("\n--- Property Access Performance ---");
    let doc_nodes = store.nodes_by_label("Document");
    let sample_sizes = [100, 1000, 10000];

    for &size in &sample_sizes {
        if doc_nodes.len() < size {
            continue;
        }
        let sample = &doc_nodes[..size];

        let start = Instant::now();
        let mut with_name = 0usize;
        for &nid in sample {
            if let Some(node) = store.get_node(nid) {
                for (_, v) in node.properties.iter() {
                    if v.as_str().is_some() {
                        with_name += 1;
                        break;
                    }
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "    get_node() + props scan x{}: {:.3}ms ({:.3}us/node, {} with strings)",
            size,
            elapsed,
            elapsed * 1000.0 / size as f64,
            with_name
        );
    }

    // ── Phase 4: Neighbor expansion ──
    println!("\n--- Neighbor Expansion ---");

    // Pick 10 Document nodes
    let seeds: Vec<_> = doc_nodes.iter().take(10).copied().collect();

    let mut total_neighbors = 0usize;
    let start = Instant::now();
    for &seed in &seeds {
        let count = store.neighbors(seed, Direction::Both).count();
        total_neighbors += count;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  neighbors() x10 seeds: {} total neighbors, {:.3}ms ({:.1}us/call)",
        total_neighbors,
        elapsed,
        elapsed * 1000.0 / 10.0
    );

    // Hub detection: find high-degree nodes
    println!("\n  Hub detection (first 1000 nodes):");
    let mut degree_samples: Vec<(u64, usize)> = Vec::new();
    for &nid in doc_nodes.iter().take(1000) {
        let deg = store.neighbors(nid, Direction::Both).count();
        degree_samples.push((nid.as_u64(), deg));
    }
    degree_samples.sort_by(|a, b| b.1.cmp(&a.1));
    for &(nid, deg) in degree_samples.iter().take(5) {
        println!("    node {} : degree {}", nid, deg);
    }

    // BFS 2-hop from a high-degree node (the real bottleneck scenario)
    if let Some(&(hub_id, hub_deg)) = degree_samples.first() {
        use obrain_common::types::NodeId;
        let hub = NodeId::from(hub_id);
        println!(
            "\n  2-hop BFS from hub node {} (degree {}):",
            hub_id, hub_deg
        );

        let start = Instant::now();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        visited.insert(hub);
        queue.push_back((hub, 0u32));
        let mut hop_counts = [0usize; 3];

        while let Some((nid, depth)) = queue.pop_front() {
            hop_counts[depth as usize] += 1;
            if depth < 2 {
                for neighbor in store.neighbors(nid, Direction::Both) {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                        if visited.len() > 50_000 {
                            break;
                        }
                    }
                }
                if visited.len() > 50_000 {
                    break;
                }
            }
        }
        let bfs_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "    hop-0: {}, hop-1: {}, hop-2: {}, total visited: {}, {:.3}ms",
            hop_counts[0], hop_counts[1], hop_counts[2], visited.len(), bfs_ms
        );
    }

    // ── Phase 5: Cypher queries (what retrieve_nodes + context builder actually do) ──
    println!("\n--- Cypher Query Performance ---");
    let session = db.session();

    let queries: Vec<(&str, usize)> = vec![
        ("MATCH (n) RETURN count(n)", 3),
        ("MATCH (n:Document) RETURN count(n)", 3),
        ("MATCH (n) WHERE id(n) = 0 RETURN n", 5),
        ("MATCH (n)-[r]->(m) WHERE id(n) = 0 RETURN type(r), id(m) LIMIT 10", 5),
    ];

    // Isolated timing: direct API vs Cypher
    println!("\n  Direct API vs Cypher comparison:");

    // Direct API: O(1)
    let start = Instant::now();
    let direct_count = store.node_count_by_label("Document");
    let direct_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("    store.node_count_by_label(\"Document\"): {} ({:.3}ms)", direct_count, direct_ms);

    // Cypher: includes parse + translate + plan + execute
    let start = Instant::now();
    let cypher_count = session.execute_cypher("MATCH (n:Document) RETURN count(n)").unwrap().scalar::<i64>().unwrap();
    let cypher_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("    Cypher count(n:Document):              {} ({:.3}ms)", cypher_count, cypher_ms);
    println!("    Overhead: {:.1}ms (parse + translate + plan + execute)", cypher_ms - direct_ms);

    assert_eq!(direct_count as i64, cypher_count, "Counts must match");

    for (q, iters) in &queries {
        let mut times = Vec::new();
        let mut rc = 0;
        for _ in 0..*iters {
            let start = Instant::now();
            let r = session.execute_cypher(q).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            rc = r.row_count();
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "    {:<60} mean: {:>8.1}ms  min: {:>8.1}ms  [{} rows]",
            q, mean, min, rc
        );
    }

    println!("\n--- SUMMARY ---");
    println!("  DB: {} nodes, {} edges (~21GB epoch)", nc, ec);
    println!("  DB open (tiered restore): {:.1}ms", open_ms);
    println!("  Potential bottlenecks:");
    println!("    1. DB open (20s) — cold start, amortized across session");
    println!("    2. nodes_by_label on Document (8M nodes) — allocation time");
    println!("    3. BFS expansion from hub nodes — exponential blowup");
    println!("    4. Per-node property access on cold nodes — epoch mmap latency");
}
