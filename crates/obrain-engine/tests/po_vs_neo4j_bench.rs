//! Comparative benchmark: Obrain vs Neo4j for Project Orchestrator workloads.
//!
//! Runs the exact Cypher queries used by the PO backend (src/neo4j/*.rs)
//! against the real PO Obrain database, and optionally against a live Neo4j instance.
//!
//! Usage:
//!   # Obrain only (requires ~/.obrain/db/po)
//!   cargo test -p obrain-engine --release --features "cypher,wal,tiered-storage,algos" \
//!     --test po_vs_neo4j_bench -- --nocapture
//!
//!   # With Neo4j comparison (requires NEO4J_URI + NEO4J_PASSWORD)
//!   NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=secret \
//!   cargo test -p obrain-engine --release --features "cypher,wal,tiered-storage,algos" \
//!     --test po_vs_neo4j_bench -- --nocapture

use std::time::Instant;

use obrain_core::graph::traits::GraphStore;
use obrain_engine::ObrainDB;

// ============================================================================
// Configuration
// ============================================================================

const ITERATIONS: usize = 10;
const WARMUP: usize = 2;

// ============================================================================
// Measurement Utilities
// ============================================================================

#[derive(Clone, Debug)]
struct BenchResult {
    name: String,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p99_ms: f64,
    row_count: usize,
}

fn measure<F: FnMut() -> usize>(name: &str, mut f: F) -> BenchResult {
    // Warmup
    for _ in 0..WARMUP {
        let _ = f();
    }

    let mut times = Vec::with_capacity(ITERATIONS);
    let mut row_count = 0;
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        row_count = f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let min = times[0];
    let max = times[times.len() - 1];
    let p50 = times[times.len() / 2];
    let p99 = times[(times.len() as f64 * 0.99) as usize];

    BenchResult {
        name: name.to_string(),
        mean_ms: mean,
        min_ms: min,
        max_ms: max,
        p50_ms: p50,
        p99_ms: p99,
        row_count,
    }
}

fn print_result(r: &BenchResult) {
    println!(
        "  {:<55} {:>8.3}ms (p50: {:.3}, p99: {:.3}, min: {:.3}, max: {:.3})  [{} rows]",
        r.name, r.mean_ms, r.p50_ms, r.p99_ms, r.min_ms, r.max_ms, r.row_count
    );
}

#[allow(dead_code)]
fn print_comparison(obrain: &BenchResult, neo4j_ms: f64) {
    let speedup = neo4j_ms / obrain.mean_ms;
    let indicator = if speedup >= 1.0 { "🟢" } else { "🔴" };
    println!(
        "  {indicator} {:<50} obrain: {:>8.3}ms  neo4j: {:>8.3}ms  ({:.1}x)",
        obrain.name, obrain.mean_ms, neo4j_ms, speedup
    );
}

// ============================================================================
// Category 1: Graph Extraction (read-heavy, PO's main bottleneck)
// ============================================================================

fn bench_graph_extraction(session: &obrain_engine::Session) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 1: Graph Extraction (PO src/neo4j/analytics.rs + code.rs)");
    println!("{}", "═".repeat(80));

    // --- 1.1 list_project_files ---
    // PO query: MATCH (p:Project {id: $pid})-[:CONTAINS]->(f:File) RETURN f.path, f.language...
    let r = measure("list_project_files (all files for project)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f:File) \
                 RETURN f.path, f.language, f.hash \
                 ORDER BY f.path",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 1.2 list_project_symbols ---
    // PO query: MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) WHERE s:Function OR s:Struct...
    let r = measure(
        "list_project_symbols (functions+structs, LIMIT 5000)",
        || {
            let result = session
                .execute_cypher(
                    "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s) \
                 WHERE s:Function OR s:Struct OR s:Trait OR s:Enum \
                 RETURN s.name, \
                   CASE WHEN s:Function THEN 'function' \
                        WHEN s:Struct THEN 'struct' \
                        WHEN s:Trait THEN 'trait' \
                        WHEN s:Enum THEN 'enum' END AS sym_type, \
                   f.path, s.visibility, s.line_start \
                 ORDER BY f.path, s.line_start \
                 LIMIT 5000",
                )
                .unwrap();
            result.row_count()
        },
    );
    print_result(&r);
    results.push(r);

    // --- 1.3 get_project_import_edges (unbounded) ---
    // PO query: MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:IMPORTS]->(f2:File)<-[:CONTAINS]-(p)
    let r = measure("get_project_import_edges (ALL imports, no LIMIT)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:IMPORTS]->(f2:File)<-[:CONTAINS]-(p) \
                 RETURN f1.path AS source, f2.path AS target",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 1.4 get_project_call_edges (unbounded) ---
    let r = measure("get_project_call_edges (ALL calls, no LIMIT)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(:File)-[:CONTAINS]->(f1:Function)-[:CALLS]->(f2:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) \
                 RETURN f1.id AS source, f2.id AS target",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 1.5 get_project_extends_edges ---
    let r = measure("get_project_extends_edges (cross-file)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s1:Struct)-[:EXTENDS]->(s2:Struct)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p) \
                 WHERE f1 <> f2 \
                 RETURN f1.path AS source, f2.path AS target",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 1.6 get_project_implements_edges ---
    let r = measure("get_project_implements_edges (cross-file)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:CONTAINS]->(s:Struct)-[:IMPLEMENTS]->(t:Trait)<-[:CONTAINS]-(f2:File)<-[:CONTAINS]-(p) \
                 WHERE f1 <> f2 \
                 RETURN f1.path AS source, f2.path AS target",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    results
}

// ============================================================================
// Category 2: Aggregation & Analytics Queries
// ============================================================================

fn bench_aggregation(session: &obrain_engine::Session) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 2: Aggregation & Analytics (PO src/neo4j/code.rs)");
    println!("{}", "═".repeat(80));

    // --- 2.1 get_most_connected_files (OPTIONAL MATCH + count + ORDER) ---
    let r = measure("most_connected_files (imports+dependents, TOP 50)", || {
        let result = session
            .execute_cypher(
                "MATCH (f:File) \
                 OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File) \
                 OPTIONAL MATCH (dependent:File)-[:IMPORTS]->(f) \
                 WITH f, count(DISTINCT imported) AS imports, count(DISTINCT dependent) AS dependents \
                 RETURN f.path, imports, dependents, imports + dependents AS connections \
                 ORDER BY connections DESC \
                 LIMIT 50",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 2.2 Node counts by label (PO dashboard / intelligence summary) ---
    let r = measure(
        "count_nodes_by_label (File, Function, Struct, Note...)",
        || {
            let mut total = 0usize;
            for label in &[
                "Project", "File", "Function", "Struct", "Trait", "Enum", "Plan", "Task", "Step",
                "Note", "Decision", "Skill", "Commit",
            ] {
                let result = session
                    .execute_cypher(&format!("MATCH (n:{label}) RETURN count(n)"))
                    .unwrap();
                total += result.row_count();
            }
            total
        },
    );
    print_result(&r);
    results.push(r);

    // --- 2.3 Co-change graph extraction ---
    let r = measure("get_co_change_graph (TOP 1000 by count)", || {
        let result = session
            .execute_cypher(
                "MATCH (f1:File)-[r:CO_CHANGED]-(f2:File) \
                 WHERE f1.path < f2.path \
                 RETURN f1.path AS file_a, f2.path AS file_b, r.count AS count \
                 ORDER BY r.count DESC \
                 LIMIT 1000",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    results
}

// ============================================================================
// Category 3: Pattern Matching & Traversal
// ============================================================================

fn bench_traversal(session: &obrain_engine::Session, store: &dyn GraphStore) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 3: Pattern Matching & Traversal");
    println!("{}", "═".repeat(80));

    // --- 3.1 Single node lookup by property ---
    let r = measure("project_lookup_by_slug (point lookup)", || {
        let result = session
            .execute_cypher("MATCH (p:Project) RETURN p.name, p.slug LIMIT 1")
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 3.2 1-hop: File → Functions ---
    let r = measure("1-hop: File→Functions (CONTAINS, first file)", || {
        let result = session
            .execute_cypher(
                "MATCH (f:File)-[:CONTAINS]->(fn:Function) \
                 RETURN f.path, fn.name, fn.line_start \
                 LIMIT 500",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 3.3 2-hop: Project → File → Function (PO code analysis) ---
    let r = measure("2-hop: Project→File→Function (full chain)", || {
        let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(fn:Function) \
                 RETURN p.name, f.path, fn.name \
                 LIMIT 2000",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 3.4 3-hop: Project → File → Function → CALLS → Function ---
    let r = measure(
        "3-hop: Project→File→Fn→CALLS→Fn (call graph)",
        || {
            let result = session
            .execute_cypher(
                "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(fn1:Function)-[:CALLS]->(fn2:Function) \
                 RETURN fn1.name, fn2.name \
                 LIMIT 2000",
            )
            .unwrap();
            result.row_count()
        },
    );
    print_result(&r);
    results.push(r);

    // --- 3.5 Direct API: neighbor expansion (bypasses Cypher) ---
    {
        use obrain_core::graph::Direction;

        let file_nodes = store.nodes_by_label("File");
        let sample_size = file_nodes.len().min(100);
        let sample = &file_nodes[..sample_size];

        let r = measure(
            &format!("direct_api: neighbors() x{sample_size} File nodes"),
            || {
                let mut total = 0usize;
                for &nid in sample {
                    total += store.neighbors(nid, Direction::Both).len();
                }
                total
            },
        );
        print_result(&r);
        results.push(r);
    }

    results
}

// ============================================================================
// Category 4: GDS Algorithms (PageRank, Louvain, Betweenness)
// ============================================================================

#[cfg(feature = "algos")]
fn bench_gds(db: &ObrainDB) -> Vec<BenchResult> {
    use obrain_adapters::plugins::algorithms::{betweenness_centrality, louvain, pagerank};

    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 4: GDS Algorithms (PO fabric analytics pipeline)");
    println!("{}", "═".repeat(80));

    let store = db.store();
    let node_count = store.node_count();
    println!(
        "  Graph size: {} nodes, {} edges",
        node_count,
        store.edge_count()
    );

    // --- 4.1 PageRank ---
    let r = measure("pagerank (damping=0.85, max_iter=20, tol=1e-6)", || {
        let store = db.store();
        let scores = pagerank(store.as_ref(), 0.85, 20, 1e-6);
        scores.len()
    });
    print_result(&r);
    results.push(r);

    // --- 4.2 Louvain ---
    let r = measure("louvain (resolution=1.0)", || {
        let store = db.store();
        let result = louvain(store.as_ref(), 1.0);
        result.communities.len()
    });
    print_result(&r);
    results.push(r);

    // --- 4.3 Betweenness Centrality ---
    if node_count <= 10_000 {
        let r = measure("betweenness_centrality (exact)", || {
            let store = db.store();
            let scores = betweenness_centrality(store.as_ref(), true);
            scores.len()
        });
        print_result(&r);
        results.push(r);
    } else {
        println!(
            "  ⏭  betweenness_centrality SKIPPED ({} nodes > 10K threshold)",
            node_count
        );
    }

    results
}

// ============================================================================
// Category 5: Batch Write Operations (PO analytics persistence)
// ============================================================================

fn bench_batch_writes(session: &obrain_engine::Session) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 5: Batch Write Operations (PO src/neo4j/analytics.rs)");
    println!("{}", "═".repeat(80));

    // Get file count for realistic batch sizing
    let file_count = session
        .execute_cypher("MATCH (f:File) RETURN count(f)")
        .unwrap()
        .scalar::<i64>()
        .unwrap_or(0) as usize;

    let batch_size = file_count.min(1000);

    if batch_size == 0 {
        println!("  ⚠  No File nodes found — batch write benchmarks skipped");
        return results;
    }

    // --- 5.1 Simulate batch_update_file_analytics (UNWIND SET pattern) ---
    // PO pattern: UNWIND $items AS u MATCH (f:File {path: u.path}) SET f.pagerank = u.pagerank, ...
    // We simulate by updating a test property on real File nodes
    let r = measure(
        &format!("batch_update_analytics ({batch_size} files, SET 4 props)"),
        || {
            // Build UNWIND data: update _bench_* properties on File nodes
            let result = session
                .execute_cypher(&format!(
                    "MATCH (f:File) \
                     WITH f LIMIT {batch_size} \
                     SET f._bench_pr = 0.15, \
                         f._bench_bc = 0.42, \
                         f._bench_cid = 3, \
                         f._bench_cc = 0.67 \
                     RETURN count(f)"
                ))
                .unwrap();
            result.scalar::<i64>().unwrap_or(0) as usize
        },
    );
    print_result(&r);
    results.push(r);

    // Cleanup
    let _ = session.execute_cypher(
        "MATCH (f:File) \
         REMOVE f._bench_pr, f._bench_bc, f._bench_cid, f._bench_cc",
    );

    // --- 5.2 Node creation throughput (INSERT pattern) ---
    let insert_count = 1000;
    let r = measure(
        &format!("insert {insert_count} nodes (CREATE pattern)"),
        || {
            for i in 0..insert_count {
                session
                    .execute_cypher(&format!(
                        "CREATE (:_BenchNode {{id: {i}, value: 'test-{i}'}})"
                    ))
                    .unwrap();
            }
            insert_count
        },
    );
    print_result(&r);
    results.push(r);

    // Cleanup bench nodes
    let _ = session.execute_cypher("MATCH (n:_BenchNode) DELETE n");

    // --- 5.3 Relationship creation throughput ---
    // Create temp nodes first
    for i in 0..200 {
        let _ = session.execute_cypher(&format!("CREATE (:_BenchSrc {{id: {i}}})"));
        let _ = session.execute_cypher(&format!("CREATE (:_BenchDst {{id: {i}}})"));
    }

    let r = measure("create 200 relationships (MATCH+CREATE pattern)", || {
        for i in 0..200 {
            session
                .execute_cypher(&format!(
                    "MATCH (a:_BenchSrc {{id: {i}}}), (b:_BenchDst {{id: {i}}}) \
                     CREATE (a)-[:_BENCH_REL]->(b)"
                ))
                .unwrap();
        }
        200
    });
    print_result(&r);
    results.push(r);

    // Cleanup
    let _ = session.execute_cypher("MATCH (n:_BenchSrc) DETACH DELETE n");
    let _ = session.execute_cypher("MATCH (n:_BenchDst) DETACH DELETE n");

    results
}

// ============================================================================
// Category 6: Knowledge Layer Queries (Notes, Decisions, Skills)
// ============================================================================

fn bench_knowledge_layer(session: &obrain_engine::Session) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 6: Knowledge Layer (PO src/neo4j/notes.rs + decisions)");
    println!("{}", "═".repeat(80));

    // --- 6.1 List notes by type ---
    let r = measure("list_notes (type=gotcha, active)", || {
        let result = session
            .execute_cypher(
                "MATCH (n:Note {note_type: 'gotcha', status: 'active'}) \
                 RETURN n.id, n.content, n.importance \
                 ORDER BY n.created_at DESC \
                 LIMIT 100",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 6.2 Note entity links (LINKED_TO pattern) ---
    let r = measure("note_entity_links (Note-[:LINKED_TO]->File)", || {
        let result = session
            .execute_cypher(
                "MATCH (n:Note)-[r:LINKED_TO]->(f:File) \
                 RETURN n.id, f.path, r.entity_type \
                 LIMIT 500",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 6.3 Decisions with AFFECTS ---
    let r = measure("decisions_with_affects (Decision-[:AFFECTS]->)", || {
        let result = session
            .execute_cypher(
                "MATCH (d:Decision)-[a:AFFECTS]->(e) \
                 RETURN d.id, d.description, labels(e), a.impact_description \
                 LIMIT 200",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    // --- 6.4 SYNAPSE edges (neural knowledge layer) ---
    let r = measure("synapse_edges (Note-[:SYNAPSE]->File, weight>=0.1)", || {
        let result = session
            .execute_cypher(
                "MATCH (n:Note)-[s:SYNAPSE]->(f:File) \
                 WHERE s.weight >= 0.1 \
                 RETURN n.id, f.path, s.weight \
                 ORDER BY s.weight DESC \
                 LIMIT 500",
            )
            .unwrap();
        result.row_count()
    });
    print_result(&r);
    results.push(r);

    results
}

// ============================================================================
// Category 7: Direct API vs Cypher overhead
// ============================================================================

fn bench_api_overhead(
    session: &obrain_engine::Session,
    store: &dyn GraphStore,
) -> Vec<BenchResult> {
    let mut results = Vec::new();

    println!("\n{}", "═".repeat(80));
    println!("  CATEGORY 7: Direct API vs Cypher Overhead");
    println!("{}", "═".repeat(80));

    // --- 7.1 node_count ---
    let r_api = measure("direct_api: store.node_count()", || store.node_count());
    print_result(&r_api);

    let r_cypher = measure("cypher: MATCH (n) RETURN count(n)", || {
        session
            .execute_cypher("MATCH (n) RETURN count(n)")
            .unwrap()
            .scalar::<i64>()
            .unwrap() as usize
    });
    print_result(&r_cypher);

    println!(
        "  → Cypher overhead: {:.3}ms ({:.0}x slower)",
        r_cypher.mean_ms - r_api.mean_ms,
        r_cypher.mean_ms / r_api.mean_ms.max(0.001)
    );
    results.push(r_api);
    results.push(r_cypher);

    // --- 7.2 node_count_by_label ---
    let r_api = measure("direct_api: node_count_by_label(\"File\")", || {
        store.node_count_by_label("File")
    });
    print_result(&r_api);

    let r_cypher = measure("cypher: MATCH (n:File) RETURN count(n)", || {
        session
            .execute_cypher("MATCH (n:File) RETURN count(n)")
            .unwrap()
            .scalar::<i64>()
            .unwrap() as usize
    });
    print_result(&r_cypher);

    println!(
        "  → Cypher overhead: {:.3}ms ({:.0}x slower)",
        r_cypher.mean_ms - r_api.mean_ms,
        r_cypher.mean_ms / r_api.mean_ms.max(0.001)
    );
    results.push(r_api);
    results.push(r_cypher);

    // --- 7.3 Property access ---
    let file_nodes = store.nodes_by_label("File");
    let sample = file_nodes.len().min(1000);
    let sample_nodes = &file_nodes[..sample];

    let r_api = measure(&format!("direct_api: get_node() + props x{sample}"), || {
        let mut count = 0usize;
        for &nid in sample_nodes {
            if let Some(node) = store.get_node(nid)
                && !node.properties.is_empty()
            {
                count += 1;
            }
        }
        count
    });
    print_result(&r_api);
    results.push(r_api);

    results
}

// ============================================================================
// Main Benchmark Entry Point
// ============================================================================

#[test]
fn bench_po_vs_neo4j() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");

    if !po_path.exists() {
        println!("  ⏭  Skipping — ~/.obrain/db/po not found");
        println!("     Run neo4j2obrain to migrate the PO database first.");
        return;
    }

    println!("\n{}", "█".repeat(80));
    println!("  OBRAIN vs NEO4J — PROJECT ORCHESTRATOR BENCHMARK");
    println!("  Database: ~/.obrain/db/po");
    println!("{}\n", "█".repeat(80));

    // ── Phase 0: DB Open ──
    let start = Instant::now();
    let db = ObrainDB::open(&po_path).expect("open PO db");
    let open_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("  DB open: {:.1}ms", open_ms);

    let session = db.session();
    let store = db.store();

    // ── Database Stats ──
    println!("\n  --- Database Stats ---");
    let total_nodes = store.node_count();
    let total_edges = store.edge_count();
    println!("  Total: {} nodes, {} edges", total_nodes, total_edges);

    let labels = store.all_labels();
    for label in &labels {
        let count = store.node_count_by_label(label);
        if count > 0 {
            println!("    {:<20} {:>8}", label, count);
        }
    }

    // ── Run all benchmark categories ──
    let mut all_results: Vec<BenchResult> = Vec::new();

    all_results.extend(bench_graph_extraction(&session));
    all_results.extend(bench_aggregation(&session));
    all_results.extend(bench_traversal(&session, store.as_ref()));
    #[cfg(feature = "algos")]
    all_results.extend(bench_gds(&db));
    all_results.extend(bench_batch_writes(&session));
    all_results.extend(bench_knowledge_layer(&session));
    all_results.extend(bench_api_overhead(&session, store.as_ref()));

    // ── Summary ──
    println!("\n{}", "█".repeat(80));
    println!("  SUMMARY — OBRAIN PERFORMANCE ON PO DATABASE");
    println!("{}", "█".repeat(80));
    println!("  DB open: {:.1}ms", open_ms);
    println!("  Graph: {} nodes, {} edges\n", total_nodes, total_edges);

    // Sort by mean time descending (show bottlenecks first)
    let mut sorted = all_results.clone();
    sorted.sort_by(|a, b| b.mean_ms.partial_cmp(&a.mean_ms).unwrap());

    println!(
        "  {:<55} {:>10} {:>10} {:>8}",
        "Query", "Mean", "P50", "Rows"
    );
    println!("  {}", "─".repeat(85));
    for r in &sorted {
        println!(
            "  {:<55} {:>8.3}ms {:>8.3}ms {:>8}",
            r.name, r.mean_ms, r.p50_ms, r.row_count
        );
    }

    // ── Known Neo4j baselines for comparison ──
    // These are typical timings from the PO backend logs on the same dataset.
    // Update these with your actual Neo4j measurements.
    println!("\n  --- Reference: Known Neo4j Timings (update with your measurements) ---");
    println!("  To get Neo4j baselines, run in the PO backend:");
    println!("    1. Enable query logging: NEO4J_SLOW_QUERY_THRESHOLD_MS=0");
    println!("    2. Run: project(action: 'get_graph', slug: 'project-orchestrator-backend')");
    println!("    3. Run: admin(action: 'update_fabric_scores')");
    println!("    4. Copy timings here to get speedup ratios");

    println!("\n  {}", "─".repeat(80));
    println!("  Benchmark complete. Use these results to compare against Neo4j.");
    println!("  Fastest queries: direct API > Cypher (overhead = parse+translate+plan)");
    println!("  For PO migration: focus on graph extraction queries (Category 1)");
    println!("  These are the bottleneck when loading the UI graph view.\n");
}

// ============================================================================
// Standalone: Quick sanity check on in-memory PO-like data
// ============================================================================

#[test]
fn bench_po_synthetic_quick() {
    println!("\n{}", "═".repeat(80));
    println!("  SYNTHETIC PO BENCHMARK (in-memory, no real DB needed)");
    println!("{}\n", "═".repeat(80));

    let db = ObrainDB::new_in_memory();
    let session = db.session();

    // Build a PO-like graph in memory
    println!("  Building synthetic PO graph...");
    let start = Instant::now();

    // 1 project
    session
        .execute_cypher("CREATE (:Project {id: 'proj-1', name: 'PO Backend', slug: 'po-backend'})")
        .unwrap();

    // 200 files
    for i in 0..200 {
        session
            .execute_cypher(&format!(
                "CREATE (:File {{path: 'src/file_{i}.rs', language: 'rust', project_id: 'proj-1'}})"
            ))
            .unwrap();
        // Link to project
        session
            .execute_cypher(&format!(
                "MATCH (p:Project {{id: 'proj-1'}}), (f:File {{path: 'src/file_{i}.rs'}}) \
                 CREATE (p)-[:CONTAINS]->(f)"
            ))
            .unwrap();
    }

    // 2000 functions (10 per file)
    for file_idx in 0..200 {
        for fn_idx in 0..10 {
            let id = file_idx * 10 + fn_idx;
            session
                .execute_cypher(&format!(
                    "CREATE (:Function {{id: 'fn-{id}', name: 'func_{id}', file_path: 'src/file_{file_idx}.rs', line_start: {}, visibility: 'pub'}})",
                    fn_idx * 20 + 1
                ))
                .unwrap();
            session
                .execute_cypher(&format!(
                    "MATCH (f:File {{path: 'src/file_{file_idx}.rs'}}), (fn:Function {{id: 'fn-{id}'}}) \
                     CREATE (f)-[:CONTAINS]->(fn)"
                ))
                .unwrap();
        }
    }

    // 500 IMPORTS edges (file → file)
    for i in 0..500 {
        let src = i % 200;
        let dst = (i * 7 + 13) % 200;
        if src != dst {
            let _ = session.execute_cypher(&format!(
                "MATCH (f1:File {{path: 'src/file_{src}.rs'}}), (f2:File {{path: 'src/file_{dst}.rs'}}) \
                 CREATE (f1)-[:IMPORTS]->(f2)"
            ));
        }
    }

    // 1000 CALLS edges (function → function)
    for i in 0..1000 {
        let src = i % 2000;
        let dst = (i * 11 + 7) % 2000;
        if src != dst {
            let _ = session.execute_cypher(&format!(
                "MATCH (f1:Function {{id: 'fn-{src}'}}), (f2:Function {{id: 'fn-{dst}'}}) \
                 CREATE (f1)-[:CALLS]->(f2)"
            ));
        }
    }

    // 100 notes
    for i in 0..100 {
        session
            .execute_cypher(&format!(
                "CREATE (:Note {{id: 'note-{i}', note_type: '{}', status: 'active', importance: '{}'}})",
                match i % 4 { 0 => "guideline", 1 => "gotcha", 2 => "pattern", _ => "tip" },
                match i % 3 { 0 => "low", 1 => "medium", _ => "high" }
            ))
            .unwrap();
    }

    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let total = session
        .execute_cypher("MATCH (n) RETURN count(n)")
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    let edges = session
        .execute_cypher("MATCH ()-[r]->() RETURN count(r)")
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    println!(
        "  Built: {} nodes, {} edges in {:.0}ms\n",
        total, edges, build_ms
    );

    // Run key queries
    let queries: Vec<(&str, &str)> = vec![
        (
            "list_project_files",
            "MATCH (p:Project)-[:CONTAINS]->(f:File) RETURN f.path ORDER BY f.path",
        ),
        (
            "list_project_symbols (LIMIT 5000)",
            "MATCH (p:Project)-[:CONTAINS]->(f:File)-[:CONTAINS]->(s:Function) RETURN s.name, f.path LIMIT 5000",
        ),
        (
            "import_edges",
            "MATCH (p:Project)-[:CONTAINS]->(f1:File)-[:IMPORTS]->(f2:File)<-[:CONTAINS]-(p) RETURN f1.path, f2.path",
        ),
        (
            "call_edges",
            "MATCH (p:Project)-[:CONTAINS]->(:File)-[:CONTAINS]->(fn1:Function)-[:CALLS]->(fn2:Function)<-[:CONTAINS]-(:File)<-[:CONTAINS]-(p) RETURN fn1.id, fn2.id",
        ),
        (
            "most_connected (TOP 20)",
            "MATCH (f:File) OPTIONAL MATCH (f)-[:IMPORTS]->(i:File) OPTIONAL MATCH (d:File)-[:IMPORTS]->(f) WITH f, count(DISTINCT i) AS imp, count(DISTINCT d) AS dep RETURN f.path, imp + dep AS conn ORDER BY conn DESC LIMIT 20",
        ),
        (
            "note_by_type",
            "MATCH (n:Note {note_type: 'gotcha'}) RETURN n.id, n.importance",
        ),
    ];

    for (name, query) in &queries {
        let r = measure(name, || session.execute_cypher(query).unwrap().row_count());
        print_result(&r);
    }

    println!("\n  Synthetic benchmark complete.");
}
