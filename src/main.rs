//! obrain-chat — Generic Graph-Augmented LLM with Attention Masking
//!
//! Architecture:
//! 1. Startup: open graph DB + auto-discover schema (labels, hierarchy, properties)
//! 2. Per query: structured retrieval driven by schema (fuzzy label match + name search)
//! 3. BFS expand + topological attention mask
//! 4. Send to llama.cpp server with mask
//!
//! Schema-agnostic: works on ANY graph, not just the PO schema.

#[cfg(feature = "http")]
#[allow(dead_code)]
mod http;

use anyhow::{Context, Result};
use llm_engine::{EngineConfig, LlamaEngine, set_verbose};
use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::lpg::LpgStore;
use rustyline::error::ReadlineError;
use rustyline::history::{DefaultHistory, History};
use rustyline::{Config, EditMode, Editor};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use think_filter::strip_think_tags;

/// Global debug flag — set via --debug CLI arg.
static DEBUG: AtomicBool = AtomicBool::new(false);

/// Print to stderr only if --debug is enabled.
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG.load(Ordering::Relaxed) {
            eprintln!($($arg)*);
        }
    };
}

use graph_schema::{GraphSchema, discover_schema, get_node_name_generic};
use kv_registry::{
    ConvFragments, KvBank, KvNodeRegistry, discover_banks, load_bank_cache, save_bank_cache,
};
use retrieval::{
    Engine, GenerationControl, GnnContext, OutputMode, is_meta_query, query_with_registry,
};
use retrieval::iptr_graph::FactSnapshot;
use retrieval::state_bias::StateMetrics;

use persona::{PersonaDB, RewardDetector, detect_facts, detect_facts_from_graph};
use retrieval::attn_dsl::AttnOp;
use retrieval::formula_selector::{
    FormulaCandidate, FormulaSelector, SelectedFormula, dsl_to_natural_language,
};

/// Build dynamic system header based on graph presence and persistent facts.
/// Score persona facts via GNN. Returns (key, value, score) with score=1.0 as fallback.
fn score_persona_facts(
    facts: &[(String, String)],
    fact_gnn: &Option<persona::fact_gnn::FactGNN>,
    persona_db: &Option<persona::PersonaDB>,
) -> Vec<(String, String, f32)> {
    if let (Some(gnn), Some(pdb)) = (fact_gnn, persona_db) {
        let store = pdb.db.store();
        let fact_ids = pdb.active_fact_ids();
        if fact_ids.is_empty() {
            return facts
                .iter()
                .map(|(k, v)| (k.clone(), v.clone(), 1.0))
                .collect();
        }
        let query_embed = persona::fact_gnn::query_embedding("context");
        let scores = gnn.score_facts(&store, &query_embed, &fact_ids, 2);
        let score_map: std::collections::HashMap<_, _> = scores.into_iter().collect();

        // Build a map from fact key → NodeId for matching
        let key_to_nid: std::collections::HashMap<String, obrain_common::types::NodeId> = fact_ids
            .iter()
            .filter_map(|&fid| {
                store.get_node(fid).and_then(|n| {
                    n.properties
                        .get(&obrain_common::types::PropertyKey::from("key"))
                        .and_then(|v| v.as_str())
                        .map(|k| (k.to_string(), fid))
                })
            })
            .collect();

        facts
            .iter()
            .map(|(k, v)| {
                let s = key_to_nid
                    .get(k)
                    .and_then(|nid| score_map.get(nid))
                    .copied()
                    .unwrap_or(1.0);
                (k.clone(), v.clone(), s)
            })
            .collect()
    } else {
        facts
            .iter()
            .map(|(k, v)| (k.clone(), v.clone(), 1.0))
            .collect()
    }
}

/// Build the system header with GNN-scored facts.
///
/// `scored_facts`: Vec of (key, value, gnn_score). Score=1.0 means unscored (no GNN).
/// Facts are injected sorted by score descending, with a ~500 token budget (~2000 chars).
/// Facts with score < 0.1 are omitted when there are more than 10 active facts.
fn build_system_header(has_graph: bool, scored_facts: &[(String, String, f32)]) -> String {
    let mut header = String::from("<|im_start|>system\n");

    // Identity from facts
    let name = scored_facts
        .iter()
        .find(|(k, _, _)| k == "name")
        .map(|(_, v, _)| v.as_str());
    if let Some(name) = name {
        header.push_str(&format!("You are {name}. "));
    } else {
        header.push_str("You are a helpful assistant. ");
    }

    if has_graph {
        header.push_str("Below is structured data from a graph database. Each entry shows [Type] Name and its relations. Use ONLY this data to answer. ");
    }

    header.push_str("Answer in the same language as the question. /no_think\n");

    // ── Memory system instructions ──
    header.push_str(r#"
## Mémoire
Tu as une mémoire persistante entre les sessions. Ce que tu sais est listé sous "Faits connus".
Quand l'utilisateur te dit quelque chose sur lui ou te demande de retenir une information, confirme naturellement.
Si un fait contredit un ancien, le nouveau le remplace. N'invente rien — utilise uniquement les faits listés.
"#);

    // Inject persistent facts — sorted by GNN score, budget-capped
    let mut other_facts: Vec<&(String, String, f32)> = scored_facts
        .iter()
        .filter(|(k, _, _)| k != "name")
        .collect();
    // Already sorted by caller, but ensure desc order
    other_facts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Apply threshold: omit low-score facts when we have plenty
    let threshold = if other_facts.len() > 10 { 0.1 } else { 0.0 };
    let other_facts: Vec<&&(String, String, f32)> = other_facts
        .iter()
        .filter(|(_, _, s)| *s >= threshold)
        .collect();

    if !other_facts.is_empty() {
        header.push_str("\nFaits connus :\n");
        let mut budget = 2000usize; // ~500 tokens at ~4 chars/token
        for (key, value, _score) in other_facts.iter().map(|f| (&f.0, &f.1, f.2)) {
            let line = format!("- {key} : {value}\n");
            if line.len() > budget {
                break;
            }
            budget -= line.len();
            header.push_str(&line);
        }
    } else if name.is_none() {
        header.push_str(
            "\nFaits connus : (aucun pour l'instant — l'utilisateur ne t'a encore rien dit)\n",
        );
    }

    header.push('\n');
    header
}

/// Build the system header using :Memory nodes (PersistNet mode).
/// Memories are raw user messages, scored by GNN energy.
fn build_memory_header(has_graph: bool, memories: &[(NodeId, String, f64)]) -> String {
    let mut header = String::from("<|im_start|>system\n");

    header.push_str("You are a helpful assistant. ");

    if has_graph {
        header.push_str("Below is structured data from a graph database. Each entry shows [Type] Name and its relations. Use ONLY this data to answer. ");
    }

    header.push_str("Answer in the same language as the question. /no_think\n");

    header.push_str(r#"
## Mémoire
Tu as une mémoire persistante entre les sessions. Ce que tu sais sur l'utilisateur est listé ci-dessous.
Utilise ces souvenirs pour personnaliser tes réponses. Si un souvenir contredit un autre, le plus récent prévaut.
"#);

    if !memories.is_empty() {
        header.push_str("\nSouvenirs :\n");
        let mut budget = 2000usize;
        for (_nid, text, _energy) in memories {
            let line = format!("- {text}\n");
            if line.len() > budget {
                break;
            }
            budget -= line.len();
            header.push_str(&line);
        }
    } else {
        header.push_str("\nSouvenirs : (aucun pour l'instant)\n");
    }

    header.push('\n');
    header
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn parse_arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}

fn main() -> Result<()> {
    // ── Generation control (shared with retrieval crate via Arc) ────
    let gen_ctl = GenerationControl {
        generating: Arc::new(AtomicBool::new(false)),
        sigint_received: Arc::new(AtomicBool::new(false)),
        gen_interrupted: Arc::new(AtomicBool::new(false)),
    };

    // Ξ(t) T3.6: Capture Ctrl+C for graceful shutdown (same path as /quit)
    let ctl_generating = gen_ctl.generating.clone();
    let ctl_sigint = gen_ctl.sigint_received.clone();
    let _ = ctrlc::set_handler(move || {
        if ctl_generating.load(Ordering::SeqCst) {
            ctl_sigint.store(true, Ordering::SeqCst);
            return;
        }
        if ctl_sigint.load(Ordering::SeqCst) {
            eprintln!("\n  [Ctrl+C] Force exit.");
            unsafe {
                unsafe extern "C" {
                    fn _exit(status: i32) -> !;
                }
                _exit(0);
            }
        }
        ctl_sigint.store(true, Ordering::SeqCst);
        eprintln!("\n  [Ctrl+C] Press Enter or Ctrl+C again to exit.");
    });

    // --debug flag (no value, just presence)
    let is_debug = std::env::args().any(|a| a == "--debug");
    if is_debug {
        DEBUG.store(true, Ordering::Relaxed);
        kv_registry::set_debug(true);
    }
    // Control llama.cpp C-level log verbosity
    set_verbose(is_debug);

    let model_path = parse_arg("--model")
        .unwrap_or_else(|| "/Users/triviere/models/qwen3-8b-q8.gguf".to_string());

    let db_path: Option<String> = parse_arg("--db");

    let max_nodes: usize = parse_arg("--max-nodes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let token_budget: i32 = parse_arg("--budget")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1400);

    let kv_capacity: i32 = parse_arg("--kv-capacity")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let n_ctx: u32 = parse_arg("--n-ctx")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32768);

    let n_gpu: i32 = parse_arg("--n-gpu")
        .and_then(|s| s.parse().ok())
        .unwrap_or(99);

    let persona_path: Option<PathBuf> = parse_arg("--persona").map(PathBuf::from);

    let http_addr: Option<std::net::SocketAddr> = parse_arg("--http").map(|s| {
        s.parse()
            .expect("Invalid --http address (expected host:port, e.g. 127.0.0.1:8080)")
    });

    // --profile-heads N: run N queries in profiling mode, export head_profile.json, then exit
    let profile_heads: Option<usize> = parse_arg("--profile-heads").and_then(|s| s.parse().ok());

    // --head-router: enable Phase B per-head α-routing via REINFORCE
    let use_head_router = std::env::args().any(|a| a == "--head-router");
    let head_router_lr: f32 = parse_arg("--head-router-lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.01);
    let head_router_warmup: u32 = parse_arg("--head-router-warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    // --head-router-granularity: "full" (n_head) or "gqa" (n_head_kv, default)
    // "gqa" (default): groups by KV heads — natural for GQA models, best recall-per-MB
    // "full": each query head routes independently — max specialization potential,
    //         more memory (~5× for GQA models), may benefit from larger GPU/Apple Silicon
    let head_router_granularity: String =
        parse_arg("--head-router-granularity").unwrap_or_else(|| "gqa".to_string());

    // Phase C: embedding injection
    let embd_injection_ratio: f32 = parse_arg("--embd-injection-ratio")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0); // 0.0 = disabled, 1.0 = inject all nodes via embeddings
    let _embd_warmup: u32 = parse_arg("--embd-warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100); // queries before full injection (for soft-mixing schedule)

    // Phase D: Hilbert layout for topology-aware KV positioning
    let hilbert_enabled: bool = std::env::args().any(|a| a == "--hilbert");

    eprintln!("=== obrain-chat — Generic Graph LLM + Topological Mask (FFI) ===\n");

    // ── HTTP server mode (--http host:port) ──────────────────────
    if let Some(http_addr) = http_addr {
        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "obrain".to_string());

        // Load all resources (same as REPL path but hands them to the actor)
        eprintln!("Loading model: {model_path}");
        let engine = Engine(LlamaEngine::new(&EngineConfig {
            model_path: model_path.clone(),
            n_ctx,
            n_gpu_layers: n_gpu,
            ..EngineConfig::default()
        })?);
        eprintln!("  Model loaded: n_ctx={}", engine.n_ctx());

        // Open database (optional)
        let (_db_holder, store, schema, banks): (
            Option<ObrainDB>,
            Option<Arc<LpgStore>>,
            Option<GraphSchema>,
            Vec<kv_registry::KvBank>,
        ) = if let Some(ref db_path) = db_path {
            eprintln!("Opening database: {db_path}");
            let ckpt = format!("{db_path}/wal/checkpoint.meta");
            let _ = std::fs::remove_file(&ckpt);
            let db = ObrainDB::open(db_path).context(format!("Failed to open DB at {db_path}"))?;
            let st = Arc::clone(db.store());
            eprintln!("  {} nodes, {} edges", st.node_count(), st.edge_count());
            let sch = graph_schema::discover_schema(&st);
            let bank_cache_path = std::path::PathBuf::from(format!("{db_path}.banks"));
            let nc = st.node_count();
            let ec = st.edge_count();
            let bnks = match kv_registry::load_bank_cache(&bank_cache_path, nc, ec) {
                Some(cached) => cached,
                None => {
                    let discovered = kv_registry::discover_banks(&st, &sch, 50);
                    kv_registry::save_bank_cache(&bank_cache_path, &discovered, nc, ec);
                    discovered
                }
            };
            (Some(db), Some(st), Some(sch), bnks)
        } else {
            eprintln!("No database specified (use --db <path> for graph-augmented mode)");
            (None, None, None, Vec::new())
        };

        // Persona DB
        let persona_resolved_path: String = if let Some(ref cp) = persona_path {
            cp.to_str().unwrap_or("conv.db").to_string()
        } else if let Some(ref db_p) = db_path {
            format!("{db_p}.persona")
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            let dir = format!("{home}/.obrain-chat");
            let _ = std::fs::create_dir_all(&dir);
            format!("{dir}/default.persona")
        };
        let ckpt_conv = format!("{persona_resolved_path}/wal/checkpoint.meta");
        let _ = std::fs::remove_file(&ckpt_conv);
        let persona_db = match persona::PersonaDB::open(&persona_resolved_path) {
            Ok(cdb) => {
                cdb.migrate_facts();
                cdb.seed_default_patterns();
                persona::RewardDetector::seed_default_reward_tokens(&cdb);
                Some(cdb)
            }
            Err(e) => {
                eprintln!("  Warning: could not open persona DB: {e}");
                None
            }
        };

        // Build system header + init registry
        let persona_facts: Vec<(String, String, f32)> = persona_db
            .as_ref()
            .map(|pdb| {
                pdb.active_facts()
                    .into_iter()
                    .map(|(k, v)| (k, v, 1.0))
                    .collect()
            })
            .unwrap_or_default();
        let system_header = build_system_header(store.is_some(), &persona_facts);
        let header_tokens_vec = engine.tokenize(&system_header, false, true)?;
        let header_n = header_tokens_vec.len() as i32;
        let header_positions: Vec<i32> = (0..header_n).collect();
        engine.encode(&header_tokens_vec, &header_positions, 0)?;
        let mut registry = KvNodeRegistry::new(&system_header, header_n);
        let mut conv_frags = ConvFragments::new();

        // Warmup banks
        let warmup_count = 3.min(banks.len());
        for bank in banks.iter().take(warmup_count) {
            let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
            registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected, &engine);
            for nid in &bank.node_ids {
                if registry.get_slot(*nid).is_some() {
                    continue;
                }
                if let Some(text) = bank.texts.get(nid) {
                    registry.register(*nid, text, &engine)?;
                }
            }
        }

        // Restore conversation fragments (3-tier: WARM archive + HOT in KV)
        if let Some(ref cdb) = persona_db {
            let all_recent = cdb.recent_messages(400);
            let mut pairs: Vec<(&str, &str)> = Vec::new();
            let mut i = 0;
            while i + 1 < all_recent.len() {
                let (ref role_q, ref content_q) = all_recent[i];
                let (ref role_a, ref content_a) = all_recent[i + 1];
                if role_q == "user" && role_a == "assistant" {
                    pairs.push((content_q.as_str(), content_a.as_str()));
                    i += 2;
                } else {
                    i += 1;
                }
            }
            let hot_count = 10.min(pairs.len());
            for (q, a) in &pairs[..pairs.len() - hot_count] {
                conv_frags.seed_warm(q, a, &[]);
            }
            for (q, a) in &pairs[pairs.len() - hot_count..] {
                let _ = conv_frags.add_turn(q, a, &[], &mut registry, &engine, kv_capacity);
            }
        }

        eprintln!(
            "  Registry: {} tokens encoded, {} banks warmed up",
            registry.next_pos, warmup_count
        );

        // Spawn actor with all resources
        let actor = server::actor::ActorHandle::spawn(server::actor::ActorConfig {
            engine,
            store,
            schema,
            banks,
            registry,
            conv_frags,
            persona_db,
            fact_gnn: None, // TODO: initialize GNN for server path too
            max_nodes,
            token_budget,
            kv_capacity,
        });

        let state = Arc::new(server::state::AppState {
            actor,
            model_name,
            model_created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            response_cache: server::routes_responses::new_response_cache(),
        });

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(server::start_server(http_addr, state))?;
        return Ok(());
    }

    // ── Load LLM via FFI ────────────────────────────────────────
    eprintln!("Loading model: {model_path}");

    // If profiling mode, create a HeadProfiler and attach its eval callback
    let mut head_profiler: Option<llm_engine::HeadProfiler> = None;

    let engine_config = EngineConfig {
        model_path: model_path.clone(),
        n_ctx,
        n_gpu_layers: n_gpu,
        ..EngineConfig::default()
    };

    let engine = if profile_heads.is_some() {
        let profiler = llm_engine::HeadProfiler::new(0, 0);
        let handle = profiler.handle();
        let eng = Engine(LlamaEngine::new_with_profiler(&engine_config, handle)?);
        eprintln!(
            "  Profiling mode: capturing attention patterns (n_heads={}, n_layers={})",
            eng.n_heads(),
            eng.n_layers()
        );
        head_profiler = Some(profiler);
        eng
    } else {
        Engine(LlamaEngine::new(&engine_config)?)
    };
    eprintln!("  Model loaded: n_ctx={}", engine.n_ctx());

    // ── Open database (optional) ──────────────────────────────────────
    let (db_holder, store, schema, banks): (
        Option<ObrainDB>,
        Option<Arc<LpgStore>>,
        Option<GraphSchema>,
        Vec<KvBank>,
    ) = if let Some(ref db_path) = db_path {
        eprintln!("Opening database: {db_path}");
        let ckpt = format!("{db_path}/wal/checkpoint.meta");
        let _ = std::fs::remove_file(&ckpt);

        let db = ObrainDB::open(db_path).context(format!("Failed to open DB at {db_path}"))?;
        let st = Arc::clone(db.store());
        eprintln!("  {} nodes, {} edges\n", st.node_count(), st.edge_count());

        debug!("Discovering schema...");
        let t0_schema = Instant::now();
        let sch = discover_schema(&st);
        debug!(
            "  Schema discovered in {:.0}ms:",
            t0_schema.elapsed().as_millis()
        );
        debug!(
            "    {} labels ({} structural, {} noise)",
            sch.labels.len(),
            sch.structural_labels.len(),
            sch.noise_labels.len()
        );
        for info in sch.labels.iter().take(10) {
            let marker = if info.is_noise { " [noise]" } else { "" };
            debug!(
                "      {:>6} {:20} imp={:.3} deg={:.1}{}",
                info.count, info.label, info.importance, info.avg_degree, marker
            );
        }
        if let Some(top_parent) = sch.parent_child.iter().next() {
            debug!(
                "    Hierarchy sample: {} → {:?}",
                top_parent.0,
                top_parent
                    .1
                    .iter()
                    .map(|(e, c)| format!("--{}--> {}", e, c))
                    .collect::<Vec<_>>()
            );
        }
        debug!("\n{}", "=".repeat(60));
        eprintln!(
            "Ready. {} structural labels, {} hierarchy rules.",
            sch.structural_labels.len(),
            sch.parent_child.len()
        );

        // Discover or load KV Banks
        let bank_cache_path = std::path::PathBuf::from(format!("{db_path}.banks"));
        let nc = st.node_count();
        let ec = st.edge_count();
        let bnks = match load_bank_cache(&bank_cache_path, nc, ec) {
            Some(cached) => {
                debug!(
                    "Loaded {} banks from cache ({})",
                    cached.len(),
                    bank_cache_path.display()
                );
                cached
            }
            None => {
                debug!("Discovering banks...");
                let discovered = discover_banks(&st, &sch, 50);
                save_bank_cache(&bank_cache_path, &discovered, nc, ec);
                debug!("  Saved bank cache to {}", bank_cache_path.display());
                discovered
            }
        };

        (Some(db), Some(st), Some(sch), bnks)
    } else {
        eprintln!("No database specified (use --db <path> for graph-augmented mode)\n");
        (None, None, None, Vec::new())
    };
    let _ = &db_holder; // keep DB alive for the store Arc

    // ── Conversation DB (auto-created if not specified) ──────────────
    let persona_resolved_path: String = if let Some(ref cp) = persona_path {
        cp.to_str().unwrap_or("conv.db").to_string()
    } else if let Some(ref db_p) = db_path {
        // Auto-create alongside the graph DB
        format!("{db_p}.persona")
    } else {
        // No graph DB: use ~/.obrain-chat/conv.db
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let dir = format!("{home}/.obrain-chat");
        let _ = std::fs::create_dir_all(&dir);
        format!("{dir}/default.persona")
    };
    debug!("Persona DB path: {persona_resolved_path}");
    // Remove stale checkpoint if exists
    let ckpt_conv = format!("{persona_resolved_path}/wal/checkpoint.meta");
    let _ = std::fs::remove_file(&ckpt_conv);
    let mut persona_db = match PersonaDB::open(&persona_resolved_path) {
        Ok(cdb) => {
            let convs = cdb.list_conversations();
            eprintln!(
                "Persona: {} ({} conversations, current: \"{}\")",
                persona_resolved_path,
                convs.len(),
                cdb.current_title()
            );
            let recent = cdb.recent_messages(4);
            if !recent.is_empty() {
                debug!("  Recent context ({} messages):", recent.len());
                for (role, content) in &recent {
                    let snippet: String = content.chars().take(80).collect();
                    debug!(
                        "    {}: {}{}",
                        role,
                        snippet,
                        if content.len() > 80 { "..." } else { "" }
                    );
                }
            }
            // Ξ(t) T1: migrate old facts + seed patterns + reward tokens at startup
            cdb.migrate_facts();
            cdb.seed_default_patterns();
            RewardDetector::seed_default_reward_tokens(&cdb);
            Some(cdb)
        }
        Err(e) => {
            eprintln!("  Warning: could not open persona DB: {e}");
            None
        }
    };

    if store.is_some() {
        eprintln!(
            "\nCommands: /quit, /schema, /kv, /banks, /history, /conversations, /new <title>"
        );
        eprintln!("  Ξ(t): /facts, /patterns, /stats, /addpattern <trigger>|<key>|<type>\n");
    } else {
        eprintln!("\nCommands: /quit, /history, /conversations, /new <title>");
        eprintln!("  Ξ(t): /facts, /patterns, /stats, /addpattern <trigger>|<key>|<type>\n");
    }

    // ── Load persistent facts and build dynamic system header ──
    let persona_facts: Vec<(String, String)> = persona_db
        .as_ref()
        .map(|pdb| pdb.active_facts())
        .unwrap_or_default();
    if !persona_facts.is_empty() {
        eprintln!(
            "  Facts: {}",
            persona_facts
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // At startup, GNN not yet initialized — use unscored facts (score=1.0)
    let scored_facts: Vec<(String, String, f32)> = persona_facts
        .iter()
        .map(|(k, v)| (k.clone(), v.clone(), 1.0))
        .collect();
    let system_header = build_system_header(store.is_some(), &scored_facts);
    debug!("System header:\n{system_header}");

    // ── Initialize KV Node Registry ─────────────────────────────
    let header_tokens_vec = engine.tokenize(&system_header, false, true)?;
    let header_n = header_tokens_vec.len() as i32;
    let header_positions: Vec<i32> = (0..header_n).collect();
    engine.encode(&header_tokens_vec, &header_positions, 0)?;
    let mut registry = KvNodeRegistry::new(&system_header, header_n);
    let mut conv_frags = ConvFragments::new();
    let mut round_tracker: Option<retrieval::RoundTracker> = if embd_injection_ratio > 0.0 {
        Some(retrieval::RoundTracker::new())
    } else {
        None // Phase D round tracking only relevant with embedding injection
    };
    debug!(
        "KV Registry initialized: header={} tokens (encoded in KV)",
        header_n
    );

    // ── Phase D: compute Hilbert layout if enabled ──────────────
    if hilbert_enabled {
        if let Some(ref st) = store {
            let hilbert_t0 = Instant::now();
            let node_ids = st.node_ids();
            let mut adjacency: std::collections::HashMap<
                NodeId,
                std::collections::HashSet<NodeId>,
            > = std::collections::HashMap::new();
            for &nid in &node_ids {
                use obrain_core::graph::Direction;
                let neighbors: std::collections::HashSet<NodeId> = st
                    .edges_from(nid, Direction::Outgoing)
                    .map(|(target, _)| target)
                    .collect();
                adjacency.insert(nid, neighbors);
            }
            // Also add reverse edges for undirected spectral embedding
            let adj_snapshot: Vec<(NodeId, Vec<NodeId>)> = adjacency
                .iter()
                .map(|(&k, v)| (k, v.iter().copied().collect::<Vec<_>>()))
                .collect();
            for (src, targets) in adj_snapshot {
                for tgt in targets {
                    adjacency.entry(tgt).or_default().insert(src);
                }
            }
            let layout = kv_registry::HilbertLayout::compute(&adjacency, header_n as u32);
            eprintln!(
                "  Hilbert layout: {} nodes, order={}, computed in {:.0}ms",
                layout.len(),
                layout.order,
                hilbert_t0.elapsed().as_millis()
            );
            registry.set_hilbert_layout(layout);
        } else {
            eprintln!("  --hilbert ignored: no database loaded");
        }
    }

    // ── Warmup: pre-load top banks into KV ──────────────────────
    let warmup_count = 3.min(banks.len());
    if warmup_count > 0 {
        let warmup_t0 = Instant::now();
        let mut warmup_loaded = 0;
        for bank in banks.iter().take(warmup_count) {
            let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
            registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected, &engine);
            for nid in &bank.node_ids {
                if registry.get_slot(*nid).is_some() {
                    continue;
                }
                if let Some(text) = bank.texts.get(nid) {
                    registry.register(*nid, text, &engine)?;
                    warmup_loaded += 1;
                }
            }
        }
        debug!(
            "  Warmup: pre-loaded {} banks ({} nodes) in {:.0}ms",
            warmup_count,
            warmup_loaded,
            warmup_t0.elapsed().as_millis()
        );
    }

    // ── Restore conversation fragments from persona_db (T3) ──────────
    // Load up to 400 messages (200 Q/A pairs) into the 3-tier system:
    //   - Oldest go into WARM tier (in-memory, searchable, promotable)
    //   - Most recent 20 go into HOT tier (KV-cache resident)
    if let Some(ref cdb) = persona_db {
        let all_recent = cdb.recent_messages(400); // 400 messages = up to 200 Q/A pairs
        if !all_recent.is_empty() {
            let restore_t0 = Instant::now();
            let mut hot_restored = 0u32;
            let mut warm_seeded = 0u32;

            // Pair up Q/A messages
            let mut pairs: Vec<(&str, &str)> = Vec::new();
            let mut i = 0;
            while i + 1 < all_recent.len() {
                let (ref role_q, ref content_q) = all_recent[i];
                let (ref role_a, ref content_a) = all_recent[i + 1];
                if role_q == "user" && role_a == "assistant" {
                    pairs.push((content_q.as_str(), content_a.as_str()));
                    i += 2;
                } else {
                    i += 1;
                }
            }

            // Split: older pairs → WARM, newest 10 → HOT
            let hot_count = 10.min(pairs.len());
            let warm_pairs = &pairs[..pairs.len() - hot_count];
            let hot_pairs = &pairs[pairs.len() - hot_count..];

            // Seed WARM tier (no KV encoding, just in-memory index)
            for (q, a) in warm_pairs {
                conv_frags.seed_warm(q, a, &[]);
                warm_seeded += 1;
            }

            // Restore HOT tier (KV-encoded)
            for (q, a) in hot_pairs {
                if let Err(e) = conv_frags.add_turn(q, a, &[], &mut registry, &engine, kv_capacity)
                {
                    debug!("  Warning: could not restore conv fragment: {e}");
                } else {
                    hot_restored += 1;
                }
            }

            if hot_restored > 0 || warm_seeded > 0 {
                debug!(
                    "  Restored {} HOT + {} WARM conversation fragments in {:.0}ms",
                    hot_restored,
                    warm_seeded,
                    restore_t0.elapsed().as_millis()
                );
            }
        }
    }

    // ── Interactive loop ─────────────────────────────────────────────
    // Rustyline: arrow keys, history (up/down), cursor movement, Ctrl+A/E, etc.
    let rl_config = Config::builder()
        .edit_mode(EditMode::Emacs)
        .auto_add_history(true)
        .max_history_size(1000)
        .expect("valid history size")
        .build();
    let mut rl: Editor<(), DefaultHistory> = Editor::with_config(rl_config)?;

    // Load readline history from PersonaDB (all past user prompts)
    if let Some(ref pdb) = persona_db {
        for prompt in pdb.user_history() {
            let _ = rl.add_history_entry(&prompt);
        }
        debug!(
            "  Loaded {} history entries from PersonaDB",
            rl.history().len()
        );
    }

    let mut turn_count: u32 = 0;
    #[allow(unused_assignments)]
    let mut current_facts = persona_facts; // live-updated when facts change
    let mut last_conv_turn_id: Option<NodeId> = None; // Ξ(t) T1: for TEMPORAL_NEXT chain
    let mut last_used_fact_ids: Vec<NodeId> = Vec::new(); // Ξ(t) T3: for reward propagation
    let mut prev_header_top5: Vec<String> = Vec::new(); // Ξ(t) T6: track top-5 fact keys for diff

    // ── Profile-heads mode: capture attention patterns and export ───
    if let (Some(n_profile_queries), Some(profiler)) = (profile_heads, &mut head_profiler) {
        eprintln!(
            "\n  [profile-heads] Running {} profiling queries...",
            n_profile_queries
        );

        // Profile queries — cover diverse topics to get varied attention patterns
        let profile_queries = vec![
            "Comment je m'appelle ?",
            "Où est-ce que j'habite ?",
            "Qu'est-ce que tu sais sur moi ?",
            "Quelle est ma couleur préférée ?",
            "Comment s'appelle mon chat ?",
            "Quel est mon métier ?",
            "Depuis quand je travaille là ?",
            "Raconte-moi une histoire courte",
            "Quelle est la capitale de la France ?",
            "Explique-moi la relativité en une phrase",
        ];

        // Set up bank assignments from the registry if available
        let mut bank_map: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
        for (bank_idx, bank) in banks.iter().enumerate() {
            for nid in &bank.node_ids {
                if let Some(slot) = registry.get_slot(*nid) {
                    for pos in slot.start..slot.end {
                        bank_map.insert(pos as i32, bank_idx as u32);
                    }
                }
            }
        }
        let n_positions = registry.next_pos as u32;

        // If no banks (no DB), assign all KV positions to bank 0 so profiler still captures data
        if bank_map.is_empty() && n_positions > 0 {
            for pos in 0..n_positions as i32 {
                bank_map.insert(pos, 0);
            }
            eprintln!(
                "  [profile-heads] No graph banks, all {} positions assigned to bank 0",
                n_positions
            );
        }

        for (i, query) in profile_queries.iter().take(n_profile_queries).enumerate() {
            eprintln!("  [{}/{}] {}", i + 1, n_profile_queries, query);

            profiler.set_bank_assignment(bank_map.clone(), n_positions);
            profiler.start();

            // Copy persistent context to ephemeral seq_id=1, encode query there
            // This triggers the eval callback which captures kq_soft_max tensors
            engine.clear_seq(1);
            engine.seq_cp(0, 1, 0, -1);
            let next_pos = engine.seq_pos_max(1) + 1;
            let tokens = engine.tokenize(query, false, true)?;
            let positions: Vec<i32> = (next_pos..next_pos + tokens.len() as i32).collect();
            let _ = engine.encode(&tokens, &positions, 1);
            engine.clear_seq(1); // cleanup ephemeral

            profiler.stop_and_collect();
        }

        let n_clusters = 4.min(engine.n_heads() as usize);
        let json = profiler.export_json(n_clusters);
        let out_path = "/tmp/head_profile.json";
        std::fs::write(out_path, &json)?;

        eprintln!(
            "\n  [profile-heads] Profiled {} queries, {} snapshots collected",
            profiler.n_queries(),
            profiler.n_queries()
        ); // simplified
        eprintln!("  [profile-heads] Exported to {}", out_path);
        eprintln!("  [profile-heads] {} clusters identified", n_clusters);

        return Ok(());
    }

    // Ξ(t) T3: Initialize RewardDetector
    let mut reward_detector: Option<RewardDetector> = persona_db
        .as_ref()
        .map(|pdb| RewardDetector::new(pdb, &engine));

    // Ξ(t) T4: Initialize FactGNN
    let mut fact_gnn: Option<persona::fact_gnn::FactGNN> = persona_db.as_ref().map(|pdb| {
        let mut gnn = persona::fact_gnn::FactGNN::new();
        let store = pdb.db.store();
        if gnn.load_weights(&store) {
            debug!(
                "  [GNN] Loaded persisted weights ({} updates, lr={:.4})",
                gnn.n_updates(),
                gnn.learning_rate()
            );
        } else {
            debug!(
                "  [GNN] Fresh Xavier-initialized weights (dim={})",
                gnn.dim()
            );
        }
        gnn
    });

    // Ξ(t) PersistNet: Initialize neural persistence classifier
    let mut persist_net: Option<persona::PersistNet> = persona_db.as_ref().map(|pdb| {
        let n_embd = engine.n_embd();
        let mut pnet = persona::PersistNet::new(n_embd);
        let store = pdb.db.store();
        if pnet.load_weights(&store) {
            debug!(
                "  [PersistNet] Loaded weights ({} updates, {} turns, n_embd={})",
                pnet.n_updates, pnet.total_turns, pnet.n_embd
            );
        } else {
            debug!(
                "  [PersistNet] Fresh init (n_embd={}, cold start for {} turns)",
                n_embd, 20
            );
        }
        pnet
    });

    // Enable embedding output for PersistNet (reads last-layer hidden state)
    if persist_net.is_some() {
        engine.set_embeddings(true);
    }

    // Cache for PersistNet: (projected_embedding, persist_score) from last forward
    let mut last_persist_result: Option<(Vec<f32>, f32)> = None;
    #[allow(unused_assignments)]
    let mut last_avg_entropy: Option<f32> = None;
    // B3: Per-head ablation reward from previous turn (for deferred HeadRouter update)
    let mut last_ablation_reward: Option<retrieval::AblationReward> = None;

    // Phase 4 AFE: Formula selector + tracking
    let formula_selector = FormulaSelector::new();
    let mut last_formula_node_id: Option<NodeId> = None;
    // Seed formulas in PersonaDB if empty
    if let Some(ref pdb) = persona_db {
        pdb.seed_formulas_if_empty();
        debug!("  [AFE] Formula selector initialized, seeds ensured");
    }

    // Phase 4 TP.1+TP.2: Self-embedding projector with KV injection
    // Disabled by default (Phase 5) — superseded by state_kq_b attention bias.
    // Enable with: cargo build --features self_embed_proprio
    #[cfg(feature = "self_embed_proprio")]
    let mut self_projector: Option<retrieval::self_embedding::SelfEmbeddingProjector> =
        persona_db.as_ref().map(|pdb| {
            let n_embd = engine.n_embd();
            let mut proj = retrieval::self_embedding::SelfEmbeddingProjector::new(n_embd);
            // Load persisted weights if available
            if let Some(weights) = pdb.load_self_embed_weights() {
                if proj.load_weights(&weights) {
                    debug!(
                        "  [Proprio] Loaded SelfEmbeddingProjector weights ({} floats)",
                        weights.len()
                    );
                }
            }
            debug!(
                "  [Proprio] SelfEmbeddingProjector initialized (n_embd={}, features={}, updates={})",
                n_embd,
                retrieval::self_embedding::N_SELF_FEATURES,
                proj.n_updates
            );
            proj
        });
    #[cfg(not(feature = "self_embed_proprio"))]
    let _self_projector: Option<()> = None;

    // Track previous assistant response for persisting corrections on positive reward
    #[allow(unused_assignments)]
    let mut last_assistant_response = String::new();
    // Track :Message NodeIds for reward propagation to PersonaDB messages
    #[allow(unused_assignments)]
    let mut last_user_msg_id: Option<NodeId> = None;
    #[allow(unused_assignments)]
    let mut last_asst_msg_id: Option<NodeId> = None;

    // Ξ(t) Phase B: HeadRouter for per-head topology routing
    // Granularity: "full" = n_head (each query head routes independently, max specialization)
    //              "gqa" = n_head_kv (GQA groups share routing, less memory)
    let mut head_router: Option<llm_engine::HeadRouter> = if use_head_router {
        let n_head = engine.n_heads() as usize;
        let n_head_kv = engine.n_heads_kv() as usize;
        let router_heads = if head_router_granularity == "gqa" {
            n_head_kv
        } else {
            n_head
        };
        let _mask_mem_per_pos2 = router_heads * 4; // bytes per position² element
        eprintln!(
            "  [Phase B] HeadRouter enabled: granularity={}, router_heads={} (n_head={}, n_kv={}), lr={}, warmup={}",
            head_router_granularity,
            router_heads,
            n_head,
            n_head_kv,
            head_router_lr,
            head_router_warmup
        );
        eprintln!(
            "  [Phase B] Mask memory: {:.0}MB per 1000 positions (budget allows ~{} pos)",
            (router_heads as f64 * 1000.0 * 1000.0 * 4.0) / 1e6,
            ((100.0 * 1024.0 * 1024.0 / (4.0 * router_heads as f64)) as f64).sqrt() as i32
        );
        let mut router = llm_engine::HeadRouter::new(
            router_heads,
            llm_engine::N_BANKS,
            head_router_lr,
            head_router_warmup,
        );
        // Load persisted self_alpha weights (TP.2)
        if let Some(ref pdb) = persona_db {
            if let Some(self_weights) = pdb.load_head_router_self_alpha() {
                router.load_self_weights(&self_weights);
                debug!(
                    "  [Proprio] Loaded HeadRouter self_alpha ({} heads)",
                    self_weights.len()
                );
            }
        }
        Some(router)
    } else {
        None
    };

    // Phase C: NodeEmbeddingCache + ProjectionNet training pipeline
    let (embd_cache, mut projection_net, mut training_manager): (
        Option<retrieval::NodeEmbeddingCache>,
        Option<retrieval::ProjectionNet>,
        Option<retrieval::TrainingManager>,
    ) = if embd_injection_ratio > 0.0 {
        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        eprintln!(
            "  [Phase C] Embedding injection enabled: ratio={:.0}%, model={}",
            embd_injection_ratio * 100.0,
            model_name
        );
        let mut cache = retrieval::NodeEmbeddingCache::new(engine.n_embd(), &model_name);

        // Determine weights path from persona dir
        let weights_path = parse_arg("--persona")
            .map(|p| retrieval::weights_path_for_persona(std::path::Path::new(&p)));

        // Collect node texts from banks
        let mut node_texts: Vec<(obrain_common::types::NodeId, String)> = Vec::new();
        for bank in &banks {
            for (nid, text) in &bank.texts {
                if !cache.has(*nid) {
                    node_texts.push((*nid, text.clone()));
                }
            }
        }

        // Get GNN embeddings if available
        let gnn_embeds: std::collections::HashMap<obrain_common::types::NodeId, Vec<f32>> =
            if let (Some(gnn), Some(pdb)) = (&fact_gnn, &persona_db) {
                let store = pdb.db.store();
                let node_ids: Vec<obrain_common::types::NodeId> =
                    node_texts.iter().map(|(nid, _)| *nid).collect();
                let embeds = gnn.get_node_embeddings(&store, &node_ids, 2);
                eprintln!(
                    "  [Phase C] GNN embeddings: {} nodes (dim={})",
                    embeds.len(),
                    gnn.dim()
                );
                embeds.into_iter().map(|(k, v)| (k, v.to_vec())).collect()
            } else {
                std::collections::HashMap::new()
            };

        let has_gnn = !gnn_embeds.is_empty();
        let gnn_dim = fact_gnn.as_ref().map(|g| g.dim()).unwrap_or(64);

        // P1: Create and train ProjectionNet (or load from disk)
        let mut pnet = retrieval::ProjectionNet::new(engine.n_embd(), gnn_dim);
        let mut tmgr = retrieval::TrainingManager::new(retrieval::TrainingConfig {
            initial_epochs: 30, // C6: contrastive plateaus ~epoch 20, 30 is sufficient
            lr: 0.001,
            lambda_cosine: 0.1,
            min_samples: 5,
            online_epochs: 5,
            online_interval: 10,
            weights_path: weights_path.clone(),
            contrastive: Some(retrieval::ContrastiveConfig::default()),
        });

        // C6: Extract graph edges for contrastive training
        let graph_edges: Option<Vec<(obrain_common::types::NodeId, obrain_common::types::NodeId)>> =
            store.as_ref().map(|st| {
                use obrain_core::graph::Direction;
                let mut edges = Vec::new();
                let node_set: std::collections::HashSet<obrain_common::types::NodeId> =
                    node_texts.iter().map(|(nid, _)| *nid).collect();
                for &(nid, _) in &node_texts {
                    for (target, _eid) in
                        st.edges_from(nid, Direction::Outgoing).collect::<Vec<_>>()
                    {
                        if node_set.contains(&target) {
                            edges.push((nid, target));
                        }
                    }
                }
                edges
            });
        if let Some(ref edges) = graph_edges {
            if !edges.is_empty() {
                eprintln!(
                    "  [Phase C] C6 contrastive: {} edges extracted from graph",
                    edges.len()
                );
            }
        }

        if has_gnn && !node_texts.is_empty() {
            // P1 + C6: Initial training with GNN fusion data + contrastive loss
            match tmgr.initial_training(
                &mut pnet,
                &engine,
                &node_texts,
                &gnn_embeds,
                graph_edges.as_deref(),
            ) {
                Ok(n) if n > 0 => {
                    eprintln!("  [Phase C] ProjectionNet trained on {} samples", n);
                }
                Ok(_) => {} // loaded from disk or skipped
                Err(e) => eprintln!("  ⚠ [Phase C] ProjectionNet training failed: {}", e),
            }

            // Now compute fused embeddings using the trained ProjectionNet
            let t0 = std::time::Instant::now();
            let fctx = retrieval::FusionContext {
                gnn_embeddings: gnn_embeds,
                projection_net: &pnet,
            };
            match retrieval::compute_node_embeddings_with_fusion(
                &engine,
                &mut cache,
                &node_texts,
                false,
                Some(&fctx),
            ) {
                Ok(n) => eprintln!(
                    "  [Phase C] Pre-computed {} fused node embeddings in {:.1}s",
                    n,
                    t0.elapsed().as_secs_f32()
                ),
                Err(e) => eprintln!("  ⚠ [Phase C] Fused embedding computation failed: {}", e),
            }
        } else if !node_texts.is_empty() {
            // No GNN — text-only embeddings (fallback)
            let t0 = std::time::Instant::now();
            match retrieval::compute_node_embeddings(&engine, &mut cache, &node_texts, false) {
                Ok(n) => eprintln!(
                    "  [Phase C] Pre-computed {} text-only node embeddings in {:.1}s",
                    n,
                    t0.elapsed().as_secs_f32()
                ),
                Err(e) => eprintln!("  ⚠ [Phase C] Text embedding computation failed: {}", e),
            }
        }

        (
            Some(cache),
            if has_gnn { Some(pnet) } else { None },
            Some(tmgr),
        )
    } else {
        (None, None, None)
    };

    // Ξ(t) T1: Now that GNN is loaded, rescore facts and update header
    if fact_gnn.is_some() && !current_facts.is_empty() {
        let scored = score_persona_facts(&current_facts, &fact_gnn, &persona_db);
        let new_header = build_system_header(store.is_some(), &scored);
        registry.update_header(&new_header);
        debug!(
            "  [GNN] Header rescored with {} facts after GNN init",
            scored.len()
        );
    }

    loop {
        // Ξ(t) T3.6: Check SIGINT before blocking on input
        if gen_ctl.sigint_received.load(Ordering::Relaxed) {
            break;
        }

        let line = match rl.readline("you> ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => {
                // Ctrl+C: same as SIGINT handler
                if gen_ctl.sigint_received.load(Ordering::SeqCst) {
                    eprintln!("\n  [Ctrl+C] Force exit.");
                    unsafe {
                        unsafe extern "C" {
                            fn _exit(status: i32) -> !;
                        }
                        _exit(0);
                    }
                }
                gen_ctl.sigint_received.store(true, Ordering::SeqCst);
                eprintln!("  [Ctrl+C] Press Ctrl+C again to exit.");
                continue;
            }
            Err(ReadlineError::Eof) => break, // Ctrl+D
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Multi-line mode: start with """ to paste multi-line text.
        // End with """ on its own line, or an empty line (double-Enter).
        let line = if line == "\"\"\"" || line.starts_with("\"\"\"") {
            let first_content = line.strip_prefix("\"\"\"").unwrap_or("").trim().to_string();
            let mut buf = if first_content.is_empty() {
                eprintln!("  (multi-line mode: paste your text, end with \"\"\" or empty line)");
                Vec::new()
            } else {
                vec![first_content]
            };
            loop {
                match rl.readline("...> ") {
                    Ok(l) => {
                        let trimmed = l.trim();
                        if trimmed == "\"\"\"" || trimmed.is_empty() {
                            break;
                        }
                        buf.push(l);
                    }
                    _ => break,
                }
            }
            buf.join("\n")
        } else {
            line
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        if line == "/quit"
            || line == "/exit"
            || line == "quit"
            || gen_ctl.sigint_received.load(Ordering::SeqCst)
        {
            break;
        }
        if line == "/schema" {
            if let Some(ref sch) = schema {
                eprintln!("  Structural: {:?}", sch.structural_labels);
                eprintln!("  Noise: {:?}", sch.noise_labels);
                for (parent, children) in &sch.parent_child {
                    eprintln!("  {} → {:?}", parent, children);
                }
            } else {
                eprintln!("  No graph loaded (use --db <path>)");
            }
            continue;
        }
        if line == "/banks" {
            for (i, bank) in banks.iter().enumerate() {
                let loaded = bank
                    .node_ids
                    .iter()
                    .filter(|nid| registry.get_slot(**nid).is_some())
                    .count();
                eprintln!(
                    "  [{}] {} ({} nodes, ~{} tok, {}/{} in KV)",
                    i,
                    bank.name,
                    bank.node_ids.len(),
                    bank.est_tokens,
                    loaded,
                    bank.node_ids.len()
                );
            }
            continue;
        }
        if line == "/history" {
            if let Some(ref cdb) = persona_db {
                let msgs = cdb.recent_messages(20);
                if msgs.is_empty() {
                    eprintln!("  (no messages in current conversation)");
                } else {
                    eprintln!("  History ({} messages):", msgs.len());
                    for (role, content) in &msgs {
                        let snippet: String = content.chars().take(120).collect();
                        eprintln!(
                            "  {}: {}{}",
                            role,
                            snippet,
                            if content.len() > 120 { "..." } else { "" }
                        );
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        // Ξ(t) T2: /patterns — list extraction patterns
        if line == "/patterns" {
            if let Some(ref pdb) = persona_db {
                let store = pdb.db.store();
                let patterns = store.nodes_by_label("Pattern");
                if patterns.is_empty() {
                    eprintln!("  (no patterns — run seed_default_patterns)");
                } else {
                    let mut items: Vec<(String, i64, f64, bool, bool)> = Vec::new();
                    for &nid in &patterns {
                        if let Some(node) = store.get_node(nid) {
                            let trigger = node
                                .properties
                                .get(&PropertyKey::from("trigger"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("?")
                                .to_string();
                            let hits = node
                                .properties
                                .get(&PropertyKey::from("hit_count"))
                                .and_then(|v| {
                                    if let Value::Int64(n) = v {
                                        Some(*n)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(0);
                            let avg_r = node
                                .properties
                                .get(&PropertyKey::from("avg_reward"))
                                .and_then(|v| {
                                    if let Value::Float64(f) = v {
                                        Some(*f)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(0.0);
                            let auto = node
                                .properties
                                .get(&PropertyKey::from("auto_generated"))
                                .and_then(|v| {
                                    if let Value::Bool(b) = v {
                                        Some(*b)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false);
                            let active = node
                                .properties
                                .get(&PropertyKey::from("active"))
                                .and_then(|v| {
                                    if let Value::Bool(b) = v {
                                        Some(*b)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(true);
                            items.push((trigger, hits, avg_r, auto, active));
                        }
                    }
                    items.sort_by(|a, b| b.1.cmp(&a.1)); // sort by hit_count desc
                    eprintln!("  Extraction patterns ({}):", items.len());
                    for (trigger, hits, avg_r, auto, active) in &items {
                        let tag = if *auto { " [auto]" } else { "" };
                        let status = if *active { "" } else { " [OFF]" };
                        eprintln!(
                            "    \"{trigger}\" → hits={hits}, avg_reward={avg_r:.2}{tag}{status}"
                        );
                    }
                }
            } else {
                eprintln!("  No persona DB");
            }
            continue;
        }
        // Ξ(t) T2: /addpattern <trigger> <key_template> <fact_type>
        if line.starts_with("/addpattern ") {
            if let Some(ref pdb) = persona_db {
                let rest = line.strip_prefix("/addpattern ").unwrap().trim();
                // Parse: "trigger text" key_template fact_type
                // Or simpler: trigger|key|type separated by |
                let parts: Vec<&str> = rest.splitn(3, '|').collect();
                if parts.len() == 3 {
                    let trigger = parts[0].trim();
                    let key_tmpl = parts[1].trim();
                    let fact_type = parts[2].trim();
                    pdb.db.create_node_with_props(
                        &["Pattern"],
                        [
                            ("trigger", Value::String(trigger.to_string().into())),
                            ("key_template", Value::String(key_tmpl.to_string().into())),
                            ("fact_type", Value::String(fact_type.to_string().into())),
                            ("hit_count", Value::Int64(0)),
                            ("avg_reward", Value::Float64(0.0)),
                            ("auto_generated", Value::Bool(false)),
                            ("active", Value::Bool(true)),
                        ],
                    );
                    eprintln!("  ✅ Pattern added: \"{trigger}\" → {key_tmpl} ({fact_type})");
                } else {
                    eprintln!("  Usage: /addpattern trigger text|key_template|fact_type");
                    eprintln!("  Example: /addpattern mon surnom est |nickname|identity");
                }
            } else {
                eprintln!("  No persona DB");
            }
            continue;
        }
        // Ξ(t) T5: /stats — system metrics
        if line == "/stats" {
            if let Some(ref pdb) = persona_db {
                let s = pdb.xi_stats();
                eprintln!("  ╔══════════════════════════════════════╗");
                eprintln!("  ║        Ξ(t) System Metrics           ║");
                eprintln!("  ╠══════════════════════════════════════╣");
                eprintln!(
                    "  ║ Facts:    {}/{} active (avg_e={:.2}, avg_c={:.2})",
                    s.facts_active, s.facts_total, s.avg_energy, s.avg_confidence
                );
                eprintln!(
                    "  ║ Patterns: {}/{} active ({} auto-generated)",
                    s.patterns_active, s.patterns_total, s.patterns_auto
                );
                eprintln!("  ║ ConvTurns: {}", s.conv_turns);
                eprintln!("  ║ Reward:   {:.3} (last 20 avg)", s.avg_reward_recent);
                eprintln!("  ║ MaskQual: {:.3} (last 20 avg)", s.avg_mask_reward);
                eprintln!("  ║ RewardTokens: {} loaded", s.reward_tokens);
                if let Some(ref gnn) = fact_gnn {
                    eprintln!(
                        "  ║ GNN:     dim={}, updates={}, lr={:.4}",
                        gnn.dim(),
                        gnn.n_updates(),
                        gnn.learning_rate()
                    );
                }
                eprintln!("  ╚══════════════════════════════════════╝");
            } else {
                eprintln!("  No persona DB");
            }
            continue;
        }
        if line == "/conversations" || line == "/convs" {
            if let Some(ref cdb) = persona_db {
                let convs = cdb.list_conversations();
                if convs.is_empty() {
                    eprintln!("  (no conversations)");
                } else {
                    for (nid, title, created, msg_count) in &convs {
                        let marker = if *nid == cdb.current_conv_id {
                            " ←"
                        } else {
                            ""
                        };
                        eprintln!(
                            "  [{}] {} ({} msgs, {}){}",
                            nid.0,
                            title,
                            msg_count,
                            &created[..10.min(created.len())],
                            marker
                        );
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/new ") {
            if let Some(ref mut cdb) = persona_db {
                let title = line.strip_prefix("/new ").unwrap().trim();
                cdb.new_conversation(title);
                eprintln!("  Created new conversation: \"{}\"", title);
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/switch ") {
            if let Some(ref mut cdb) = persona_db {
                let id_str = line.strip_prefix("/switch ").unwrap().trim();
                if let Ok(id_num) = id_str.parse::<u64>() {
                    if cdb.switch_to(NodeId(id_num)) {
                        eprintln!("  Switched to conversation: \"{}\"", cdb.current_title());
                    } else {
                        eprintln!("  Conversation {} not found", id_num);
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line == "/facts" {
            if let Some(ref pdb) = persona_db {
                let facts = pdb.list_facts();
                let active: Vec<_> = facts.iter().filter(|f| f.4).collect();
                if active.is_empty() {
                    eprintln!("  (no facts stored)");
                } else {
                    eprintln!("  Persistent facts ({}):", active.len());
                    for (nid, key, value, turn, _, confidence, energy, fact_type) in &active {
                        eprintln!(
                            "    [{fact_type}] {key} = {value} (turn {turn}, conf={confidence:.2}, energy={energy:.2}, id={})",
                            nid.0
                        );
                    }
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line.starts_with("/forget ") {
            if let Some(ref pdb) = persona_db {
                let key = line.strip_prefix("/forget ").unwrap().trim();
                if pdb.forget_fact(key) {
                    eprintln!("  ✓ Forgot fact: {}", key);
                } else {
                    eprintln!("  No active fact with key '{}'", key);
                }
            } else {
                eprintln!("  No persona DB (use --persona <path>)");
            }
            continue;
        }
        if line == "/kv" {
            debug!(
                "  KV: {} nodes, {} tokens, hit_rate={:.0}% (hits={}, misses={}, queries={})",
                registry.nodes.len(),
                registry.next_pos,
                registry.metrics.hit_rate() * 100.0,
                registry.metrics.cache_hits,
                registry.metrics.cache_misses,
                registry.metrics.total_queries
            );
            eprintln!("  Loaded nodes ({}):", registry.order.len());
            for nid in registry.order.iter().take(20) {
                let (label, name) = if let (Some(st), Some(sch)) = (&store, &schema) {
                    let label: Option<String> = st.get_node(*nid).and_then(|n| {
                        n.labels.first().map(|l| {
                            let s: &str = l.as_ref();
                            s.to_string()
                        })
                    });
                    let dp = label.as_deref().and_then(|l| sch.display_props.get(l));
                    let name = get_node_name_generic(st, *nid, dp);
                    (label, name)
                } else {
                    (None, String::new())
                };
                let label_str = label.as_deref().unwrap_or("?");
                let slot = registry.get_slot(*nid);
                if let Some(s) = slot {
                    eprintln!(
                        "    {} [{}-{}] last_q={} [{}] {}",
                        nid.0,
                        s.start,
                        s.end,
                        s.last_used,
                        label_str,
                        if name.is_empty() { "(unnamed)" } else { &name }
                    );
                }
            }
            if registry.order.len() > 20 {
                eprintln!("    ... and {} more", registry.order.len() - 20);
            }
            continue;
        }

        // Store user message in conversation DB
        let user_msg_id = persona_db
            .as_ref()
            .map(|cdb| cdb.add_message("user", &line));

        // T6: Meta queries (identity, memory) bypass graph retrieval
        let meta = is_meta_query(&line);
        let (q_store, q_schema) = if meta {
            debug!("  [path] meta query → fallback (no graph)");
            (None, None)
        } else {
            if store.is_some() {
                debug!("  [path] graph retrieval (store loaded)");
            } else {
                debug!("  [path] no graph loaded → fallback");
            }
            (store.as_ref(), schema.as_ref())
        };

        // Ξ(t) T6: Dynamic header — rebuild if memories/facts changed
        if persist_net.is_some() {
            // PersistNet mode: use :Memory nodes
            if let Some(ref pdb) = persona_db {
                let memories = pdb.active_memories();
                let new_top5: Vec<String> = memories
                    .iter()
                    .take(5)
                    .map(|(nid, _, _)| format!("{}", nid.0))
                    .collect();
                if new_top5 != prev_header_top5 && !new_top5.is_empty() {
                    let new_header = build_memory_header(store.is_some(), &memories);
                    registry.update_header(&new_header);
                    registry.full_recompact::<fn(&[f32], &[i32], i32) -> anyhow::Result<usize>>(
                        &engine, None, None,
                    );
                    debug!(
                        "  [T6] Header rebuilt: memories changed {:?} → {:?}",
                        prev_header_top5, new_top5
                    );
                    prev_header_top5 = new_top5;
                }
            }
        } else {
            // Legacy mode: use :Fact nodes with GNN scoring
            let scored = score_persona_facts(&current_facts, &fact_gnn, &persona_db);
            let new_top5: Vec<String> = scored
                .iter()
                .filter(|(k, _, _)| k != "name")
                .take(5)
                .map(|(k, _, _)| k.clone())
                .collect();
            if new_top5 != prev_header_top5 && !new_top5.is_empty() {
                let new_header = build_system_header(store.is_some(), &scored);
                registry.update_header(&new_header);
                registry.full_recompact::<fn(&[f32], &[i32], i32) -> anyhow::Result<usize>>(
                    &engine, None, None,
                );
                debug!(
                    "  [T6] Header rebuilt: top-5 changed {:?} → {:?}",
                    prev_header_top5, new_top5
                );
                prev_header_top5 = new_top5;
            }
        }

        // Ξ(t) T2: Build GNN context for composite E(t) scoring
        let gnn_ctx = if let (Some(gnn), Some(pdb)) = (&fact_gnn, &persona_db) {
            Some(GnnContext {
                gnn,
                persona_store: &*pdb.db.store(),
            })
        } else {
            None
        };

        // E3: Get coactivation map ref before passing round_tracker as &mut
        let coactivation_snapshot: Option<retrieval::CoactivationMap> =
            round_tracker.as_ref().map(|rt| rt.coactivation().clone());
        let coact_ref = coactivation_snapshot.as_ref();

        let cold_ref: Option<&dyn kv_registry::ColdSearch> = persona_db
            .as_ref()
            .map(|p| p as &dyn kv_registry::ColdSearch);

        // Phase 4 AFE: Select attention formula (compiled inside query_with_registry)
        // OBRAIN_AFE_DISABLE=1 forces Identity (for A/B benchmarking)
        let afe_disabled = std::env::var("OBRAIN_AFE_DISABLE")
            .map(|v| v == "1")
            .unwrap_or(false);
        let selected_formula: Option<SelectedFormula> = if afe_disabled {
            None
        } else if let Some(ref pdb) = persona_db {
            let formula_nodes = pdb.list_formulas();
            if !formula_nodes.is_empty() {
                let candidates: Vec<FormulaCandidate> = formula_nodes
                    .iter()
                    .map(|f| FormulaCandidate {
                        name: f.name.clone(),
                        dsl_json: f.dsl_json.clone(),
                        avg_reward: f.avg_reward,
                        energy: f.energy,
                        activation_count: f.activation_count,
                        generation: f.generation,
                        context_affinity: f.context_affinity.clone(),
                        active: f.active,
                    })
                    .collect();
                // Use turn_count as seed for deterministic but varying selection
                let selected = formula_selector.select(&candidates, &[], turn_count as u64);
                // Track the formula's NodeId for reward propagation
                if let Some(fnode) = formula_nodes.get(selected.candidate_index) {
                    last_formula_node_id = Some(fnode.id);
                }
                debug!(
                    "  [AFE] selected formula: {} (score={:.3})",
                    selected.name, selected.score
                );
                Some(selected)
            } else {
                None
            }
        } else {
            None
        };

        // Phase 4 TP.2: Inject self-embeddings into KV cache (proprioceptive channel)
        // Disabled by default (Phase 5) — superseded by state_kq_b.
        #[allow(unused_mut)]
        let mut self_embed_positions: Vec<i32> = Vec::new();
        #[cfg(feature = "self_embed_proprio")]
        if let (Some(proj), Some(pdb)) = (&mut self_projector, &persona_db) {
            let metrics = pdb.current_self_metrics();
            let has_metrics = metrics.reward_avg != 0.0
                || metrics.mask_reward_avg != 0.0
                || metrics.gnn_facts_active > 0
                || !metrics.formula_active_name.is_empty();
            if has_metrics {
                let embedding = proj.project(&metrics);
                let inject_pos = registry.next_pos;
                let kv_max = engine.seq_pos_max(0);
                if kv_max >= 0 && inject_pos <= kv_max {
                    let safe_pos = kv_max + 1;
                    eprintln!(
                        "  [Proprio] ⚠ position desync: next_pos={} <= kv_max={}, resyncing to {}",
                        inject_pos, kv_max, safe_pos
                    );
                    registry.next_pos = safe_pos;
                }
                let inject_pos = registry.next_pos;
                match engine.inject_embedding(inject_pos, &embedding) {
                    Ok(()) => {
                        self_embed_positions.push(inject_pos);
                        registry.next_pos = inject_pos + 1;
                        debug!(
                            "  [Proprio] self-embedding injected at pos={}, n_embd={}, kv_max={}",
                            inject_pos,
                            embedding.len(),
                            engine.seq_pos_max(0)
                        );
                    }
                    Err(e) => {
                        eprintln!("  [Proprio] injection failed: {e} (inject_pos={}, kv_max={})",
                            inject_pos, engine.seq_pos_max(0));
                    }
                }
            }
        }

        // Phase 5: Build IPTR FactSnapshot from PersonaDB (if available)
        let iptr_snapshot = if std::env::var("OBRAIN_IPTR_DISABLE").is_ok() {
            None
        } else {
            persona_db.as_ref().map(|pdb| {
                let tokenize_fn = |text: &str| -> Vec<(u32, f32)> {
                    match engine.tokenize(text, false, false) {
                        Ok(tokens) => tokens.iter().map(|&t| (t as u32, 1.0f32)).collect(),
                        Err(_) => Vec::new(),
                    }
                };
                let snap = FactSnapshot::from_persona_db(pdb, Some(&tokenize_fn));
                debug!(
                    "  [IPTR] FactSnapshot: {} facts, {} edges",
                    snap.facts.len(),
                    snap.adjacency.values().map(|v| v.len()).sum::<usize>() / 2
                );
                snap
            })
        };

        // Phase 5: Extract StateMetrics from PersonaDB SelfMetrics (if available)
        let state_metrics = if std::env::var("OBRAIN_STATE_BIAS_DISABLE").is_ok() {
            None
        } else {
            persona_db.as_ref().map(|pdb| {
                let sm = pdb.current_self_metrics();
                // confidence: mask_reward_avg is in [-1,1], map to [0,1] via (x+1)/2
                let confidence = ((sm.mask_reward_avg + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
                debug!(
                    "  [StateBias] reward_avg={:.3}, confidence={:.3}",
                    sm.reward_avg, confidence
                );
                StateMetrics {
                    reward_avg: sm.reward_avg as f32,
                    confidence,
                }
            })
        };

        match query_with_registry(
            &engine,
            q_store,
            q_schema,
            &mut registry,
            &mut conv_frags,
            &banks,
            &line,
            max_nodes,
            token_budget,
            kv_capacity,
            &gen_ctl,
            &OutputMode::Stdout,
            gnn_ctx.as_ref(),
            head_router.as_ref(),
            embd_cache.as_ref(),
            embd_injection_ratio,
            round_tracker.as_mut(),
            coact_ref,
            cold_ref,
            selected_formula.as_ref(),
            &self_embed_positions,
            iptr_snapshot.as_ref(),
            state_metrics.as_ref(),
        ) {
            Ok((response, relevant_graph_nodes, avg_entropy, ablation_reward)) => {
                // Ξ(t) T5: Store entropy signal for next turn's reward computation
                last_avg_entropy = avg_entropy;

                // Ξ(t) Phase B/B3: Log ablation reward + backward into HeadRouter
                if let Some(ref abl) = ablation_reward {
                    eprintln!(
                        "  [B3] head_contribution_entropy={:.3}, bank_contributions={:?}",
                        abl.head_contribution_entropy,
                        abl.bank_contributions
                            .iter()
                            .map(|c| format!("{c:.4}"))
                            .collect::<Vec<_>>()
                    );
                    // Backward: update HeadRouter α-weights with per-head reward signal
                    if let Some(ref mut router) = head_router {
                        let visibility = router.forward();
                        router.backward(&abl.per_head_rewards, &visibility);
                        eprintln!("  [B3] HeadRouter updated (n_updates={})", router.n_updates);
                    }
                }
                // P2: Online ProjectionNet refinement
                if let (Some(tmgr), Some(pnet)) = (&mut training_manager, &mut projection_net) {
                    // No new samples to add here (would need text_h computation per query node,
                    // which is expensive). Just trigger periodic retraining on accumulated data.
                    tmgr.on_query(pnet, &[]);
                }

                // Response already streamed to stdout by engine.generate
                let clean = strip_think_tags(&response);
                let mut trimmed = clean.trim().to_string();

                // If the model wrapped the ENTIRE response in <think>...</think>,
                // the ThinkFilter ate everything. Extract the think content as the
                // actual response — Qwen3 sometimes ignores /no_think.
                if trimmed.is_empty() && !response.is_empty() {
                    let think_content = response
                        .strip_prefix("<think>")
                        .unwrap_or(&response)
                        .strip_suffix("</think>")
                        .unwrap_or(&response)
                        .trim();
                    if !think_content.is_empty() {
                        // Show the response that was hidden by ThinkFilter
                        print!("assistant> {}\n\n", think_content);
                        trimmed = think_content.to_string();
                        debug!(
                            "  [think-rescue] Recovered {} chars from think-only response",
                            trimmed.len()
                        );
                    } else {
                        eprintln!("  ⚠ Response was completely empty (0 tokens generated)");
                    }
                }

                // Register Q&A as a concise fragment in the KV cache
                if let Err(e) = conv_frags.add_turn(
                    &line,
                    &trimmed,
                    &relevant_graph_nodes,
                    &mut registry,
                    &engine,
                    kv_capacity,
                ) {
                    eprintln!("  Warning: could not register conv fragment: {e}");
                }

                // Store full messages in conversation DB (for persistence across sessions)
                if let Some(ref cdb) = persona_db {
                    let asst_id = cdb.add_message("assistant", &trimmed);
                    if let Some(uid) = user_msg_id {
                        cdb.link_reply(asst_id, uid);
                    }
                    last_asst_msg_id = Some(asst_id);
                }
                last_user_msg_id = user_msg_id;

                // Track assistant response for potential correction persistence (Fix #1)
                last_assistant_response = trimmed.clone();

                // Ξ(t) T3: Compute reward for PREVIOUS turn based on current user input
                turn_count += 1;

                // Ctrl+C interruption → negative reward for THIS turn (user rejected output)
                let gen_was_interrupted = gen_ctl.gen_interrupted.swap(false, Ordering::SeqCst);
                if gen_was_interrupted {
                    if let (Some(rd), Some(pdb)) = (&mut reward_detector, &persona_db) {
                        let interrupt_penalty = -0.5;
                        if let Some(ct_id) = conv_frags.last_node_id() {
                            rd.propagate_reward(pdb, ct_id, interrupt_penalty, &last_used_fact_ids);
                        }
                        debug!(
                            "  [Reward] turn #{}: {:.2} (Ctrl+C interruption — negative signal)",
                            turn_count, interrupt_penalty
                        );
                        if let Some(ref mut gnn) = fact_gnn {
                            let store = pdb.db.store();
                            let scores = gnn.forward(
                                &store,
                                &persona::fact_gnn::query_embedding(&line),
                                &last_used_fact_ids,
                                2,
                            );
                            gnn.update(&store, &last_used_fact_ids, &scores, interrupt_penalty);
                        }
                        // PersistNet REINFORCE: interrupt = negative signal for previous persist decision
                        if let (Some(pnet), Some((proj, prev_score))) =
                            (&mut persist_net, &last_persist_result)
                        {
                            pnet.update(proj, *prev_score, interrupt_penalty);
                            debug!(
                                "  [PersistNet] REINFORCE (interrupt): reward={:.2}, score={:.2}, updates={}",
                                interrupt_penalty, prev_score, pnet.n_updates
                            );
                        }
                        // HeadRouter REINFORCE: interrupt = negative signal
                        if let Some(ref mut router) = head_router {
                            router.backward_uniform(interrupt_penalty);
                        }
                    }
                }

                if let (Some(rd), Some(pdb), Some(prev_ct)) =
                    (&mut reward_detector, &persona_db, last_conv_turn_id)
                {
                    let user_tokens = engine.tokenize(&line, false, false).unwrap_or_default();
                    // Ξ(t) T4: Enriched reward with factual success + entropy signal
                    let facts_for_reward: Vec<(String, String)> = current_facts.clone();
                    let signals = rd.compute_reward(
                        &user_tokens,
                        turn_count,
                        if facts_for_reward.is_empty() {
                            None
                        } else {
                            Some(&facts_for_reward)
                        },
                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(&trimmed)
                        },
                        last_avg_entropy, // Ξ(t) T5: entropy signal from previous turn's generation
                    );
                    let reward = signals.reward;
                    if reward.abs() > 0.01 {
                        rd.propagate_reward_ex(
                            pdb,
                            prev_ct,
                            reward,
                            &last_used_fact_ids,
                            Some(signals.factual_signal),
                        );
                        debug!(
                            "  [Reward] turn #{}: {:.2} (mask_quality={:.2})",
                            turn_count - 1,
                            reward,
                            signals.factual_signal
                        );

                        // Ξ(t) T4+T5: GNN online update with mask-quality-weighted reward.
                        // The factual_signal measures whether the topology mask guided
                        // attention to facts the model actually used. Blend it with
                        // the composite reward to give the GNN a stronger gradient
                        // toward node selection that produces useful attention patterns.
                        if let Some(ref mut gnn) = fact_gnn {
                            let store = pdb.db.store();
                            let scores = gnn.forward(
                                &store,
                                &persona::fact_gnn::query_embedding(&line),
                                &last_used_fact_ids,
                                2,
                            );
                            // 60% mask quality + 40% composite reward
                            let gnn_reward = 0.6 * signals.factual_signal + 0.4 * reward;
                            gnn.update(&store, &last_used_fact_ids, &scores, gnn_reward);
                            debug!(
                                "  [GNN] update #{}, lr={:.4}, gnn_reward={:.2} (mask={:.2}, composite={:.2})",
                                gnn.n_updates(),
                                gnn.learning_rate(),
                                gnn_reward,
                                signals.factual_signal,
                                reward
                            );
                        }

                        // Ξ(t) PersistNet REINFORCE: propagate reward to previous persist decision
                        if let (Some(pnet), Some((proj, prev_score))) =
                            (&mut persist_net, &last_persist_result)
                        {
                            pnet.update(proj, *prev_score, reward);
                            debug!(
                                "  [PersistNet] REINFORCE: reward={:.2}, prev_score={:.2}, updates={}",
                                reward, prev_score, pnet.n_updates
                            );
                        }

                        // Ξ(t) Phase B/B3: HeadRouter REINFORCE update (deferred)
                        // Use per-head ablation reward from PREVIOUS turn (when reward is known)
                        if let Some(ref mut router) = head_router {
                            if let Some(ref abl) = last_ablation_reward {
                                // B3: per-head reward scaled by user feedback reward
                                let scaled: Vec<f32> = abl
                                    .per_head_rewards
                                    .iter()
                                    .map(|&r| r * reward.signum()) // scale direction by user reward
                                    .collect();
                                let visibility = router.forward();
                                router.backward(&scaled, &visibility);
                                debug!(
                                    "  [HeadRouter B3] deferred update #{}, user_reward={:.2}, head_entropy={:.3}",
                                    router.n_updates, reward, abl.head_contribution_entropy
                                );
                            } else {
                                router.backward_uniform(reward);
                            }
                            if router.n_updates % 10 == 0 && router.n_updates > 0 {
                                debug!(
                                    "  [HeadRouter] update #{}, specialization_entropy={:.3}",
                                    router.n_updates,
                                    router.specialization_entropy()
                                );
                            }
                        }

                        // Phase 4 TP.2: Train self-embedding projector + HeadRouter self_alpha
                        // Disabled by default (Phase 5) — superseded by state_kq_b.
                        #[cfg(feature = "self_embed_proprio")]
                        if !self_embed_positions.is_empty() {
                            if let Some(ref mut proj) = self_projector {
                                proj.train_step(reward);
                                debug!(
                                    "  [Proprio] projector trained: reward={:.2}, updates={}",
                                    reward, proj.n_updates
                                );
                            }
                            if let Some(ref mut router) = head_router {
                                router.backward_self(reward);
                                debug!(
                                    "  [Proprio] HeadRouter self_alpha updated: reward={:.2}",
                                    reward
                                );
                            }
                        }

                        // Phase 4 AFE: Propagate reward to the formula that was active
                        if let (Some(fid), Some(pdb2)) =
                            (last_formula_node_id, &persona_db)
                        {
                            pdb2.update_formula_reward(fid, reward as f64);
                            debug!(
                                "  [AFE] formula reward: id={:?}, reward={:.2}",
                                fid, reward
                            );
                            // Homeostasis: check if mutation should be triggered
                            let formulas = pdb2.list_formulas();
                            let rewards: Vec<f64> = formulas
                                .iter()
                                .map(|f| f.avg_reward)
                                .collect();
                            if retrieval::formula_evolution::should_mutate_burst(&rewards, 0.05) {
                                // Find the active formula and mutate it
                                if let Some(active) = formulas.iter().find(|f| f.id == fid) {
                                    if let Ok(op) =
                                        serde_json::from_str::<AttnOp>(&active.dsl_json)
                                    {
                                        let mut rng = retrieval::formula_evolution::Rng::new(
                                            turn_count as u64,
                                        );
                                        let mutated =
                                            retrieval::formula_evolution::mutate(&op, &mut rng);
                                        if let Ok(json) = serde_json::to_string(&mutated.op) {
                                            let affinity_refs: Vec<&str> = active
                                                .context_affinity
                                                .iter()
                                                .map(|s| s.as_str())
                                                .collect();
                                            let child_id = pdb2.add_formula(
                                                &format!("{}-m{}", active.name, active.generation + 1),
                                                &json,
                                                &affinity_refs,
                                                active.generation + 1,
                                                Some(fid),
                                            );
                                            debug!(
                                                "  [AFE] mutation triggered: {} -> {:?} ({:?})",
                                                active.name, child_id, mutated.mutation_type
                                            );
                                        }
                                    }
                                }
                            }
                            // GC: cull dead formulas if population exceeded
                            let gc_candidates: Vec<retrieval::formula_evolution::GcCandidate> =
                                formulas
                                    .iter()
                                    .enumerate()
                                    .map(|(i, f)| retrieval::formula_evolution::GcCandidate {
                                        index: i,
                                        energy: f.energy,
                                        activation_count: f.activation_count,
                                        generation: f.generation,
                                    })
                                    .collect();
                            let gc_targets =
                                retrieval::formula_evolution::gc_candidates(
                                    &gc_candidates, 0.05, 50, 50,
                                );
                            for idx in gc_targets {
                                if let Some(f) = formulas.get(idx) {
                                    pdb2.deactivate_formula(f.id);
                                    debug!("  [AFE] GC deactivated: {}", f.name);
                                }
                            }
                        }

                        // Mark memories as "useful" when reward is positive
                        if reward > 0.1 {
                            if let Some(pdb2) = &persona_db {
                                for mem_id in pdb2.active_memory_ids() {
                                    pdb2.mark_memory_useful(mem_id, turn_count);
                                }
                            }
                        }

                        // Persist the CORRECTED assistant response when user gives positive feedback.
                        // This closes the main learning gap: PersistNet only persists user messages,
                        // but the correction (assistant's improved answer) is what matters for recall.
                        if reward > 0.3
                            && !last_assistant_response.is_empty()
                            && last_assistant_response.len() > 20
                        {
                            if let Some(pdb2) = &persona_db {
                                let mem_id = pdb2.add_memory(
                                    &format!("[correction] {}", last_assistant_response),
                                    reward as f32,
                                    turn_count.saturating_sub(1),
                                    last_conv_turn_id,
                                );
                                debug!(
                                    "  🧠 Assistant correction persisted (reward={:.2}, id={}, len={})",
                                    reward,
                                    mem_id.0,
                                    last_assistant_response.len()
                                );
                            }
                        }

                        // Propagate reward to :Message nodes in PersonaDB (Fix #3)
                        // This allows BM25 COLD search to boost validated corrections
                        if let Some(pdb2) = &persona_db {
                            if let Some(uid) = last_user_msg_id {
                                pdb2.set_message_reward(uid, reward as f64);
                            }
                            if let Some(aid) = last_asst_msg_id {
                                pdb2.set_message_reward(aid, reward as f64);
                            }
                            debug!(
                                "  [Reward→Message] propagated {:.2} to user={:?}, asst={:?}",
                                reward,
                                last_user_msg_id.map(|n| n.0),
                                last_asst_msg_id.map(|n| n.0)
                            );
                        }
                    }
                }

                // Ξ(t) T4: Save GNN weights every 10 turns
                if turn_count % 10 == 0 {
                    if let (Some(gnn), Some(pdb)) = (&fact_gnn, &persona_db) {
                        gnn.save_weights(&pdb.db);
                        debug!("  [GNN] Weights saved (turn {})", turn_count);
                    }
                    // Save PersistNet weights periodically too
                    if let (Some(pnet), Some(pdb)) = (&persist_net, &persona_db) {
                        pnet.save_weights(&pdb.db);
                        debug!("  [PersistNet] Weights saved (turn {})", turn_count);
                    }
                }

                // Ξ(t) PersistNet: Audit stale memories every 10 turns (after cold start)
                if turn_count % 10 == 0 && turn_count > 20 {
                    if let (Some(pnet), Some(pdb)) = (&mut persist_net, &persona_db) {
                        let stale = pdb.stale_memories(turn_count, 15); // 15 turns without being useful
                        for (mem_id, text, _src_turn, _last_useful) in &stale {
                            // Send negative reward to teach PersistNet not to persist similar messages
                            // We need to re-forward the memory text through the net to get the gradient
                            let hash_emb: Vec<f32> = {
                                let n = pnet.n_embd.max(1);
                                let mut emb = vec![0.0f32; n];
                                // Simple text hash → embedding for gradient signal
                                for (i, ch) in text.chars().enumerate() {
                                    emb[i % n] += (ch as u32 as f32) / 128.0;
                                }
                                emb
                            };
                            let (score, proj) = pnet.forward(&hash_emb);
                            pnet.update(&proj, score, -0.3); // moderate negative signal
                            pdb.deactivate_memory(*mem_id);
                            debug!(
                                "  [PersistNet] Stale memory deactivated: id={}, text='{}'",
                                mem_id.0,
                                text.chars().take(50).collect::<String>()
                            );
                        }
                        if !stale.is_empty() {
                            debug!(
                                "  [PersistNet] Audited: {} stale memories deactivated",
                                stale.len()
                            );
                        }
                    }
                }

                // Ξ(t) T1: Create :ConvTurn and wire edges
                let conv_turn_id = if let Some(ref pdb) = persona_db {
                    // Track which facts/memories were USED_IN this turn (injected in header)
                    let mut used_fact_ids = pdb.active_fact_ids();
                    // Also include :Memory nodes in USED_IN tracking
                    used_fact_ids.extend(pdb.active_memory_ids());

                    // Create the ConvTurn node
                    let ct_id = pdb.create_conv_turn(&line, &trimmed, turn_count);

                    // TEMPORAL_NEXT chain
                    if let Some(prev_ct) = last_conv_turn_id {
                        pdb.link_temporal(prev_ct, ct_id);
                    }

                    // USED_IN edges: facts that were in the context for this turn
                    pdb.mark_facts_used_in(&used_fact_ids, ct_id);

                    // MENTIONS edges: link to graph nodes retrieved for this query
                    for &gn in &relevant_graph_nodes {
                        pdb.link_mentions(ct_id, gn);
                    }

                    // Ξ(t) Phase B/B3: Persist per-head ablation reward on ConvTurn
                    if let Some(ref abl) = ablation_reward {
                        // Store per_head_rewards as JSON string (JSONB-like property)
                        let rewards_json = format!(
                            "[{}]",
                            abl.per_head_rewards
                                .iter()
                                .map(|r| format!("{r:.6}"))
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                        pdb.db.set_node_property(
                            ct_id,
                            "mask_reward_per_head",
                            Value::String(rewards_json.into()),
                        );
                        pdb.db.set_node_property(
                            ct_id,
                            "head_contribution_entropy",
                            Value::Float64(abl.head_contribution_entropy as f64),
                        );
                        // Also store bank_contributions for retrospective analysis
                        let banks_json = format!(
                            "[{}]",
                            abl.bank_contributions
                                .iter()
                                .map(|c| format!("{c:.6}"))
                                .collect::<Vec<_>>()
                                .join(",")
                        );
                        pdb.db.set_node_property(
                            ct_id,
                            "bank_contributions",
                            Value::String(banks_json.into()),
                        );
                    }

                    debug!(
                        "  [Ξ] ConvTurn #{} created, {} facts USED_IN, {} graph MENTIONS",
                        turn_count,
                        used_fact_ids.len(),
                        relevant_graph_nodes.len()
                    );

                    last_used_fact_ids = used_fact_ids;
                    Some(ct_id)
                } else {
                    last_used_fact_ids.clear();
                    None
                };
                last_conv_turn_id = conv_turn_id;

                // Ξ(t) PersistNet: Neural persistence decision
                // Extract LLM embedding of user message and decide if it should be persisted
                if let (Some(pnet), Some(pdb)) = (&mut persist_net, &persona_db) {
                    pnet.tick();

                    // Get last-layer hidden state from the encode that just happened
                    let embedding = engine.get_embedding(-1);
                    let (persist_score, projected) = if !embedding.is_empty() {
                        pnet.forward(embedding)
                    } else {
                        // Fallback: use hash embedding if LLM embedding unavailable
                        let fake = vec![0.0f32; pnet.n_embd.max(1)];
                        pnet.forward(&fake)
                    };

                    if pnet.should_persist(persist_score) {
                        let mem_id = pdb.add_memory(&line, persist_score, turn_count, conv_turn_id);
                        eprintln!(
                            "  🧠 Memory persisted (score={:.2}, id={})",
                            persist_score, mem_id.0
                        );

                        // Update header with memories
                        let memories = pdb.active_memories();
                        let memory_header = build_memory_header(store.is_some(), &memories);
                        registry.update_header(&memory_header);
                        debug!("  Header updated with {} memories", memories.len());
                    } else {
                        debug!(
                            "  [PersistNet] skip (score={:.2} < threshold)",
                            persist_score
                        );
                    }

                    last_persist_result = Some((projected, persist_score));
                } else if let Some(ref pdb) = persona_db {
                    // Fallback: pattern-based detection when PersistNet unavailable
                    let matches = detect_facts_from_graph(pdb, &line);
                    if !matches.is_empty() {
                        for m in &matches {
                            let fact_id = pdb.add_fact(&m.key, &m.value, turn_count, conv_turn_id);
                            pdb.db.create_edge(m.pattern_nid, fact_id, "EXTRACTS");
                            eprintln!(
                                "  💾 Fact stored: {} = {} (pattern={})",
                                m.key, m.value, m.pattern_nid.0
                            );
                        }
                        current_facts = pdb.active_facts();
                        let scored = score_persona_facts(&current_facts, &fact_gnn, &persona_db);
                        let new_header = build_system_header(store.is_some(), &scored);
                        registry.update_header(&new_header);
                        debug!(
                            "  Header updated with {} facts ({} scored)",
                            current_facts.len(),
                            scored.len()
                        );
                    }
                } else {
                    // No PersonaDB — use legacy hardcoded detection
                    let detected = detect_facts(&line);
                    if !detected.is_empty() {
                        for (key, value) in &detected {
                            eprintln!("  💾 Fact detected (no DB): {} = {}", key, value);
                        }
                    }
                }

                // Phase 4: Update :Self introspective metrics
                if let Some(ref pdb) = persona_db {
                    let (formula_name, formula_expl) = selected_formula
                        .as_ref()
                        .map(|f| {
                            (f.name.clone(), dsl_to_natural_language(&f.op))
                        })
                        .unwrap_or_else(|| ("none".to_string(), String::new()));
                    let head_top5: Vec<(usize, f32)> = head_router
                        .as_ref()
                        .map(|r| {
                            let vis = r.forward();
                            let n_bank = r.n_bank;
                            let mut head_scores: Vec<(usize, f32)> = (0..r.n_head)
                                .map(|h| {
                                    let max_vis = (0..n_bank)
                                        .map(|b| vis[h * n_bank + b])
                                        .fold(0.0f32, f32::max);
                                    (h, max_vis)
                                })
                                .collect();
                            head_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            head_scores.truncate(5);
                            head_scores
                        })
                        .unwrap_or_default();
                    let metrics = persona::SelfMetrics {
                        reward_avg: last_avg_entropy.unwrap_or(0.0) as f64,
                        mask_reward_avg: 0.0, // filled from reward signals when available
                        head_router_top5: head_top5,
                        formula_active_name: formula_name,
                        formula_explanation: formula_expl,
                        gnn_facts_active: last_used_fact_ids.len(),
                        learning_trend: if turn_count < 5 {
                            "cold_start".to_string()
                        } else {
                            "stable".to_string()
                        },
                    };
                    pdb.upsert_self_metrics(&metrics);
                    debug!(
                        "  [Self] metrics updated: reward_avg={:.2}, formula={}",
                        metrics.reward_avg, metrics.formula_active_name
                    );
                }

                // B3: Store ablation reward for deferred backward at next turn
                last_ablation_reward = ablation_reward;

                // Ξ(t) T5: Periodic pattern auto-generation and garbage collection
                if let Some(ref pdb) = persona_db {
                    if turn_count % 5 == 0 {
                        let generated = pdb.try_generate_patterns();
                        if generated > 0 {
                            debug!("  [AutoGen] Generated {} new patterns", generated);
                        }
                    }
                    if turn_count % 20 == 0 {
                        pdb.gc_persona_graph(turn_count);
                    }
                }
            }
            Err(e) => eprintln!("  Error: {e}\n"),
        }
    }

    // Ξ(t) T4: Save GNN weights at session end
    if let (Some(gnn), Some(pdb)) = (&fact_gnn, &persona_db) {
        gnn.save_weights(&pdb.db);
        debug!(
            "  [GNN] Weights saved at session end ({} updates)",
            gnn.n_updates()
        );
    }

    // Save ProjectionNet weights at session end (Phase C)
    if let (Some(tmgr), Some(pnet)) = (&training_manager, &projection_net) {
        tmgr.save_weights(pnet);
        debug!(
            "  [ProjectionNet] Weights saved at session end ({} updates, {} samples)",
            pnet.n_updates(),
            tmgr.n_samples()
        );
    }

    // Save SelfEmbeddingProjector weights at session end (Phase 4 TP.2)
    // Disabled by default (Phase 5) — superseded by state_kq_b.
    #[cfg(feature = "self_embed_proprio")]
    if let (Some(proj), Some(pdb)) = (&self_projector, &persona_db) {
        let weights = proj.save_weights();
        pdb.save_self_embed_weights(&weights);
        debug!(
            "  [Proprio] SelfEmbeddingProjector weights saved ({} updates)",
            proj.n_updates
        );
    }

    // Save HeadRouter self_alpha at session end (Phase 4 TP.2)
    if let (Some(router), Some(pdb)) = (&head_router, &persona_db) {
        pdb.save_head_router_self_alpha(router.self_weights());
        debug!(
            "  [Proprio] HeadRouter self_alpha saved ({} heads)",
            router.n_head
        );
    }

    // Save PersistNet weights at session end
    if let (Some(pnet), Some(pdb)) = (&persist_net, &persona_db) {
        pnet.save_weights(&pdb.db);
        debug!(
            "  [PersistNet] Weights saved ({} updates, {} turns)",
            pnet.n_updates, pnet.total_turns
        );
    }

    // Ξ(t) T3.6: Contextual end-of-session signal
    if let (Some(pdb), Some(prev_ct)) = (&persona_db, last_conv_turn_id) {
        let store = pdb.db.store();
        let last_reward = store
            .get_node(prev_ct)
            .and_then(|n| {
                n.properties
                    .get(&PropertyKey::from("reward"))
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(*f as f32)
                        } else {
                            None
                        }
                    })
            })
            .unwrap_or(0.0);

        if last_reward > 0.3 {
            // Session ended on a satisfied note — bonus to recent facts
            debug!(
                "  [Session] ended: satisfied (last_reward={:.2}, turns={})",
                last_reward, turn_count
            );
            for &fid in &last_used_fact_ids {
                if let Some(node) = store.get_node(fid) {
                    let energy = node
                        .properties
                        .get(&PropertyKey::from("energy"))
                        .and_then(|v| {
                            if let Value::Float64(f) = v {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1.0);
                    let new_energy = (energy + 0.1).min(2.0);
                    pdb.db
                        .set_node_property(fid, "energy", Value::Float64(new_energy));
                }
            }
        } else if last_reward < -0.2 {
            // Session ended frustrated — penalty on last turn's facts
            debug!(
                "  [Session] ended: frustrated (last_reward={:.2}, turns={})",
                last_reward, turn_count
            );
            for &fid in &last_used_fact_ids {
                if let Some(node) = store.get_node(fid) {
                    let energy = node
                        .properties
                        .get(&PropertyKey::from("energy"))
                        .and_then(|v| {
                            if let Value::Float64(f) = v {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(1.0);
                    let new_energy = (energy - 0.2).max(0.0);
                    pdb.db
                        .set_node_property(fid, "energy", Value::Float64(new_energy));
                }
            }
        } else {
            debug!(
                "  [Session] ended: neutral (last_reward={:.2}, turns={})",
                last_reward, turn_count
            );
        }
    }

    // History is persisted via PersonaDB (add_message) — no file needed.
    eprintln!("Bye!");
    Ok(())
}
