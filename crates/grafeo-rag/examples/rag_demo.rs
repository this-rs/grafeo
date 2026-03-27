//! RAG demo — queries a GrafeoDB and produces augmented context.
//!
//! Usage:
//!   cargo run -p grafeo-rag --example rag_demo -- --db /path/to/db --query "your question"
//!   cargo run -p grafeo-rag --example rag_demo -- --db /path/to/db   (interactive mode)
//!
//! Without --query, enters interactive REPL mode.
//! Without --db, creates an in-memory demo database.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

use grafeo::GrafeoDB;
use grafeo_cognitive::engram::EngramStore;
use grafeo_rag::{EngramRetriever, GraphContextBuilder, RagConfig, RagPipeline};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut db_path: Option<String> = None;
    let mut query: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => {
                i += 1;
                if i < args.len() {
                    db_path = Some(args[i].clone());
                }
            }
            "--query" | "-q" => {
                i += 1;
                if i < args.len() {
                    query = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: rag_demo [--db <path>] [--query <text>]");
                eprintln!("  --db <path>    Path to a GrafeoDB directory");
                eprintln!("  --query <text>  Single query mode");
                eprintln!("  (no --query)    Interactive REPL mode");
                return;
            }
            _ => {
                query = Some(args[i..].join(" "));
                break;
            }
        }
        i += 1;
    }

    // Open or create database
    let db = match &db_path {
        Some(path) => {
            eprintln!("[rag] Opening database: {}", path);

            // Workaround for checkpoint.meta bug in directory-format DBs
            let ckpt = format!("{}/wal/checkpoint.meta", path);
            let _ = std::fs::remove_file(&ckpt);

            match GrafeoDB::open(path) {
                Ok(db) => {
                    eprintln!("[rag] Database opened");
                    db
                }
                Err(e) => {
                    eprintln!("[rag] Failed to open database: {}", e);
                    eprintln!("[rag] Falling back to in-memory demo");
                    create_demo_db()
                }
            }
        }
        None => {
            eprintln!("[rag] No --db specified, using in-memory demo database");
            create_demo_db()
        }
    };

    let store = Arc::clone(db.store());
    let node_count = store.node_count();
    eprintln!("[rag] Graph: {} nodes", node_count);

    let engram_store = Arc::new(EngramStore::new(None));

    let t0 = Instant::now();
    let retriever = Arc::new(EngramRetriever::with_defaults(
        Arc::clone(&store),
        Arc::clone(&engram_store),
        None,
    ));
    let index_time = t0.elapsed();

    let (total_nodes, distinct_terms, label_distribution) = retriever.index_stats();

    eprintln!(
        "[rag] Index: {} distinct terms across {} nodes ({:.0}ms)",
        distinct_terms,
        total_nodes,
        index_time.as_secs_f64() * 1000.0
    );
    for (label, count, frac) in label_distribution.iter().take(5) {
        eprintln!("[rag]   {:>6.1}%  {} ({})", frac, label, count);
    }

    let context_builder = GraphContextBuilder::new();

    let config = RagConfig {
        token_budget: 3000,
        max_context_nodes: 20,
        include_relations: true,
        include_labels: true,
        ..RagConfig::default()
    };

    let mut pipeline = RagPipeline::new(
        Arc::clone(&retriever) as Arc<dyn grafeo_rag::traits::Retriever>,
        context_builder,
        None,
        config,
    );

    match query {
        Some(q) => {
            run_query(&pipeline, &q, false);
        }
        None => {
            interactive_loop(&mut pipeline, &retriever, node_count);
        }
    }
}

/// Execute a single query and print results.
fn run_query(pipeline: &RagPipeline, query: &str, trace: bool) {
    eprintln!("[rag] Query: \"{}\"", query);

    if trace {
        match pipeline.retrieve(query) {
            Ok(result) => {
                eprintln!("[trace] Engrams matched: {}", result.engrams_matched);
                eprintln!("[trace] Nodes activated: {}", result.nodes_activated);
                eprintln!("[trace] Nodes retrieved: {}", result.nodes.len());
                for (i, node) in result.nodes.iter().take(10).enumerate() {
                    let labels = node.labels.join(",");
                    let name = node
                        .properties
                        .get("name")
                        .or_else(|| node.properties.get("title"))
                        .map(|s| s.as_str())
                        .unwrap_or("?");
                    eprintln!(
                        "[trace]   #{}: [{labels}] \"{name}\" score={:.3} src={:?}",
                        i + 1,
                        node.score,
                        node.source
                    );
                }
                eprintln!();
            }
            Err(e) => {
                eprintln!("[trace] Retrieval error: {}", e);
            }
        }
    }

    eprintln!("[rag] Retrieving...\n");

    let t0 = Instant::now();
    match pipeline.query(query) {
        Ok(context) => {
            let elapsed = t0.elapsed();
            if context.text.is_empty() {
                eprintln!("[rag] No relevant content found for this query.");
            } else {
                println!("{}", context.text);
                eprintln!(
                    "[rag] Context: {} nodes, ~{} tokens ({:.1}ms)",
                    context.nodes_included,
                    context.estimated_tokens,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
        Err(e) => {
            eprintln!("[rag] Retrieval error: {}", e);
        }
    }
}

/// Interactive REPL — read queries from stdin, display RAG context.
fn interactive_loop(
    pipeline: &mut RagPipeline,
    retriever: &Arc<EngramRetriever>,
    node_count: usize,
) {
    let mut trace_mode = false;

    eprintln!("[rag] Interactive mode — type a query, press Enter.");
    eprintln!("[rag] Commands: /help for all commands.\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        let preset = pipeline.config().preset_name();
        eprint!("rag({preset})> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                eprintln!("\n[rag] Bye!");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[rag] Read error: {}", e);
                break;
            }
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Handle commands
        match trimmed {
            "/quit" | "/exit" | "/q" => {
                eprintln!("[rag] Bye!");
                break;
            }
            "/help" | "/h" => {
                eprintln!("  Type any question to search the graph.\n");
                eprintln!("  /stats            — graph & pipeline stats");
                eprintln!("  /config            — show current configuration");
                eprintln!("  /config fast       — switch to fast preset");
                eprintln!("  /config balanced   — switch to balanced preset (default)");
                eprintln!("  /config thorough   — switch to thorough preset");
                eprintln!("  /trace             — toggle trace mode (show cues & scores)");
                eprintln!("  /reindex           — rebuild index from current graph state");
                eprintln!("  /quit or /exit     — leave\n");
                continue;
            }
            "/stats" => {
                let cfg = pipeline.config();
                let (total, terms, labels) = retriever.index_stats();
                eprintln!("  Graph: {} nodes", node_count);
                eprintln!("  Index: {} distinct terms, {} indexed nodes", terms, total);
                eprintln!("  Preset: {}", cfg.preset_name());
                eprintln!("  Token budget: {}", cfg.token_budget);
                eprintln!("  Max context nodes: {}", cfg.max_context_nodes);
                eprintln!("  Trace mode: {}", if trace_mode { "ON" } else { "OFF" });
                eprintln!("  --- Label distribution (dampening) ---");
                for (label, count, frac) in labels.iter().take(10) {
                    let dampening = (1.0 + frac / 100.0 * 10.0).ln().max(1.0);
                    eprintln!(
                        "    {:>6.1}%  {} ({}) → ÷{:.2}",
                        frac, label, count, dampening
                    );
                }
                eprintln!();
                continue;
            }
            "/reindex" => {
                eprintln!("[rag] Rebuilding index...");
                let t0 = Instant::now();
                retriever.reindex();
                let elapsed = t0.elapsed();
                let (total, terms, _) = retriever.index_stats();
                eprintln!(
                    "[rag] Reindexed: {} terms, {} nodes ({:.0}ms)\n",
                    terms,
                    total,
                    elapsed.as_secs_f64() * 1000.0
                );
                continue;
            }
            "/config" => {
                let cfg = pipeline.config();
                eprintln!("  preset: {}", cfg.preset_name());
                eprintln!("  --- Recall ---");
                eprintln!("  max_engrams: {}", cfg.max_engrams);
                eprintln!("  min_recall_confidence: {}", cfg.min_recall_confidence);
                eprintln!("  --- Activation ---");
                eprintln!("  activation_depth: {}", cfg.activation_depth);
                eprintln!("  activation_decay: {}", cfg.activation_decay);
                eprintln!("  max_activated_nodes: {}", cfg.max_activated_nodes);
                eprintln!("  --- Context ---");
                eprintln!("  token_budget: {}", cfg.token_budget);
                eprintln!("  max_context_nodes: {}", cfg.max_context_nodes);
                eprintln!("  include_relations: {}", cfg.include_relations);
                eprintln!("  noise_properties: {:?}\n", cfg.noise_properties);
                continue;
            }
            "/config fast" => {
                pipeline.set_config(RagConfig::fast());
                eprintln!("[rag] Switched to fast preset (budget=1000, depth=1)\n");
                continue;
            }
            "/config balanced" => {
                pipeline.set_config(RagConfig::balanced());
                eprintln!("[rag] Switched to balanced preset (budget=2000, depth=2)\n");
                continue;
            }
            "/config thorough" => {
                pipeline.set_config(RagConfig::thorough());
                eprintln!("[rag] Switched to thorough preset (budget=4000, depth=3)\n");
                continue;
            }
            "/trace" => {
                trace_mode = !trace_mode;
                eprintln!(
                    "[rag] Trace mode: {}\n",
                    if trace_mode { "ON" } else { "OFF" }
                );
                continue;
            }
            _ if trimmed.starts_with('/') => {
                eprintln!("[rag] Unknown command: {}. Type /help for help.\n", trimmed);
                continue;
            }
            _ => {}
        }

        run_query(pipeline, trimmed, trace_mode);
        println!();
    }
}

/// Create a demo in-memory database with sample data.
fn create_demo_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let queries = [
        "INSERT (:Project {name: 'Grafeo', description: 'High-performance embeddable graph database'})",
        "INSERT (:Project {name: 'Project Orchestrator', description: 'MCP-based project management with cognitive features'})",
        "INSERT (:Note {title: 'WAL Recovery Bug', content: 'checkpoint.meta causes data loss in directory format', type: 'gotcha', importance: 'high'})",
        "INSERT (:Note {title: 'Engram Architecture', content: 'Engrams are consolidated memory traces built from node ensembles via Hopfield spectral matching', type: 'pattern'})",
        "INSERT (:Plan {title: 'grafeo-rag implementation', description: 'Schema-agnostic retrieval via engrams', status: 'in_progress'})",
        "INSERT (:Task {title: 'Scaffolding crate', status: 'completed', tags: 'setup,architecture'})",
        "INSERT (:Task {title: 'Engram-based Retriever', status: 'completed', tags: 'core,retrieval'})",
        "INSERT (:Function {name: 'hopfield_retrieve', file: 'engram/hopfield.rs', description: 'Modern Hopfield content-addressable memory retrieval with per-engram precision'})",
        "INSERT (:Function {name: 'spread', file: 'activation.rs', description: 'BFS spreading activation through synapses with energy decay'})",
    ];

    for q in &queries {
        if let Err(e) = session.execute(q) {
            eprintln!("[demo] Failed to insert: {} — {}", q, e);
        }
    }

    let rel_queries = [
        "MATCH (p:Project {name: 'Grafeo'}), (n:Note {title: 'WAL Recovery Bug'}) INSERT (p)-[:HAS_NOTE]->(n)",
        "MATCH (p:Project {name: 'Project Orchestrator'}), (pl:Plan) INSERT (p)-[:HAS_PLAN]->(pl)",
        "MATCH (pl:Plan), (t:Task {title: 'Scaffolding crate'}) INSERT (pl)-[:HAS_TASK]->(t)",
        "MATCH (pl:Plan), (t:Task {title: 'Engram-based Retriever'}) INSERT (pl)-[:HAS_TASK]->(t)",
    ];

    for q in &rel_queries {
        if let Err(e) = session.execute(q) {
            eprintln!("[demo] Failed to create relation: {}", e);
        }
    }

    db
}
