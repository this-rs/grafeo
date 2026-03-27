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

use grafeo::GrafeoDB;
use grafeo_cognitive::engram::EngramStore;
use grafeo_rag::{
    EngramRetriever, GraphContextBuilder, RagConfig, RagPipeline,
};

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
                // Treat unknown args as query text
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

    // Print some stats
    let node_count = store.node_count();
    eprintln!("[rag] Graph: {} nodes", node_count);

    // Create cognitive components
    let engram_store = Arc::new(EngramStore::new(None));

    // Create the RAG pipeline
    let retriever = EngramRetriever::with_defaults(
        Arc::clone(&store),
        Arc::clone(&engram_store),
        None, // No synapse store for demo
    );

    let context_builder = GraphContextBuilder::new();

    let config = RagConfig {
        token_budget: 3000,
        max_context_nodes: 20,
        include_relations: true,
        include_labels: true,
        ..RagConfig::default()
    };

    let pipeline = RagPipeline::new(retriever, context_builder, None, config);

    match query {
        Some(q) => {
            // Single query mode
            run_query(&pipeline, &q);
        }
        None => {
            // Interactive REPL mode
            interactive_loop(&pipeline);
        }
    }
}

/// Execute a single query and print results.
fn run_query(pipeline: &RagPipeline, query: &str) {
    eprintln!("[rag] Query: \"{}\"", query);
    eprintln!("[rag] Retrieving...\n");

    match pipeline.query(query) {
        Ok(context) => {
            if context.text.is_empty() {
                eprintln!("[rag] No relevant content found for this query.");
            } else {
                println!("{}", context.text);
                eprintln!(
                    "[rag] Context: {} nodes, ~{} tokens",
                    context.nodes_included, context.estimated_tokens
                );
            }
        }
        Err(e) => {
            eprintln!("[rag] Retrieval error: {}", e);
        }
    }
}

/// Interactive REPL — read queries from stdin, display RAG context.
fn interactive_loop(pipeline: &RagPipeline) {
    eprintln!("[rag] Interactive mode — type a query, press Enter.");
    eprintln!("[rag] Commands: /quit or /exit to leave, /help for help.\n");

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Prompt
        eprint!("rag> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl-D)
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
                eprintln!("  Type any question to search the graph.");
                eprintln!("  /quit or /exit  — leave");
                eprintln!("  /help           — this message\n");
                continue;
            }
            _ => {}
        }

        run_query(pipeline, trimmed);
        println!(); // blank line between results
    }
}

/// Create a demo in-memory database with sample data.
fn create_demo_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Insert sample data
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

    // Add some relationships
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

