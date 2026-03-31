use anyhow::Result;
use graph_schema::GraphSchema;
use kv_registry::{
    BankManager, ColdSearch, ContextNode, ConvFragments, KvBank, KvNodeRegistry, QueryContext,
    Tokenizer,
};
use obrain_common::types::NodeId;
use obrain_core::graph::lpg::LpgStore;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use think_filter::ThinkFilter;

use crate::control::{GenerationControl, OutputMode, Spinner};
use crate::engine::Engine;
use crate::attn_compiler::{CompileContext, CompiledFormula, compile};
use crate::attn_dsl::AttnOp;
use crate::formula_selector::SelectedFormula;
use crate::generation::{AblationReward, compute_ablation_reward, generate_with_mask};
use crate::node_embedding::{NodeEmbeddingCache, compute_text_embedding};
use crate::round_tracker::{CoactivationMap, DemotionType, RoundTracker};
use crate::scoring::{get_micro_tag, retrieve_nodes};
use kv_registry::{HilbertLayout, build_fused_adjacency};

/// Optional GNN scoring context for composite E(t) decision.
pub struct GnnContext<'a> {
    pub gnn: &'a persona::fact_gnn::FactGNN,
    pub persona_store: &'a LpgStore,
}

/// Context for Hilbert bank-based structural elimination (T4.2).
///
/// When provided, `query_with_registry` selects top-K relevant banks via cosine
/// similarity between query embedding and bank centroids, loads them into KV,
/// evicts non-relevant banks, and filters retrieval to loaded bank nodes only.
pub struct BankContext<'a> {
    /// The bank manager (mutated: loads/evicts banks, updates LRU).
    pub manager: &'a mut BankManager,
    /// Pre-computed bank centroids (bank_id → mean embedding).
    /// Computed via `BankManager::compute_bank_centroids()`.
    pub centroids: HashMap<usize, Vec<f32>>,
    /// Inter-bank adjacency (bank_id → neighbor bank_ids).
    /// Computed via `BankManager::build_bank_adjacency()`.
    pub adjacency: HashMap<usize, HashSet<usize>>,
    /// Number of top banks to select before 1-hop expansion.
    pub top_k: usize,
}

pub fn query_with_registry(
    engine: &Engine,
    store: Option<&Arc<LpgStore>>,
    schema: Option<&GraphSchema>,
    registry: &mut KvNodeRegistry,
    conv_frags: &mut ConvFragments,
    banks: &[KvBank],
    query: &str,
    max_nodes: usize,
    token_budget: i32,
    kv_capacity: i32,
    ctl: &GenerationControl,
    output: &OutputMode,
    gnn_ctx: Option<&GnnContext>,
    head_router: Option<&llm_engine::HeadRouter>,
    embd_cache: Option<&NodeEmbeddingCache>,
    embd_injection_ratio: f32,
    mut round_tracker: Option<&mut RoundTracker>,
    coactivation: Option<&CoactivationMap>,
    cold_search: Option<&dyn ColdSearch>,
    selected_formula: Option<&SelectedFormula>,
    self_embed_positions: &[i32],
    iptr_snapshot: Option<&crate::iptr_graph::FactSnapshot>,
    state_metrics: Option<&crate::state_bias::StateMetrics>,
    bank_ctx: Option<BankContext>,
) -> Result<(String, Vec<NodeId>, Option<f32>, Option<AblationReward>)> {
    registry.begin_query();
    let t_start = Instant::now();

    // ── Check if query matches a bank name (load entire bank) ────
    let query_lower = query.to_lowercase();
    let mut _bank_loaded = false;
    for bank in banks {
        let bank_name_lower = bank.name.to_lowercase();
        let terms: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
            .filter(|s| s.len() > 2)
            .collect();
        let matches_bank = terms.iter().any(|term| {
            bank_name_lower.contains(term) && *term != "project" && *term != "les" && *term != "des"
        });
        if matches_bank {
            let missing: Vec<NodeId> = bank
                .node_ids
                .iter()
                .filter(|nid| !registry.nodes.contains_key(nid))
                .copied()
                .collect();
            if !missing.is_empty() {
                let protected: HashSet<NodeId> = bank.node_ids.iter().copied().collect();
                registry.ensure_capacity(bank.est_tokens, kv_capacity, &protected, engine);

                for nid in &missing {
                    if let Some(text) = bank.texts.get(nid) {
                        registry.register(*nid, text, engine)?;
                    }
                }
                _bank_loaded = true;
            } else {
                registry.touch(&bank.node_ids);
                _bank_loaded = true;
            }
        }
    }

    // ── T4.2: Hilbert bank selection by cosine retrieval ──────────
    // Before retrieve_nodes, select and load relevant banks so that only
    // nodes from loaded banks are considered. This is the structural
    // elimination step — most of the graph is never decoded.
    let loaded_bank_nodes: Option<HashSet<NodeId>> = if let Some(bctx) = bank_ctx {
        // Compute query embedding for bank cosine selection
        let query_embd = compute_text_embedding(engine, query).unwrap_or_default();

        if !query_embd.is_empty() && !bctx.centroids.is_empty() {
            // Select top-K banks + 1-hop neighbors
            let selected = bctx.manager.select_banks(
                &query_embd,
                &bctx.centroids,
                &bctx.adjacency,
                bctx.top_k,
            );

            // Load selected banks that aren't loaded yet
            let mut n_loaded = 0usize;
            let mut n_evicted = 0usize;

            // Evict loaded banks that are NOT in the selected set
            let selected_set: HashSet<usize> = selected.iter().copied().collect();
            let to_evict: Vec<usize> = bctx.manager.banks.iter()
                .filter(|b| b.loaded && !selected_set.contains(&b.id))
                .map(|b| b.id)
                .collect();
            for bank_id in to_evict {
                if let Ok(n) = bctx.manager.evict_bank(bank_id, registry, engine) {
                    n_evicted += n;
                }
            }

            // Load selected banks
            for &bank_id in &selected {
                if let Some(cache) = embd_cache {
                    let get_embd = |nid: NodeId| -> Option<Vec<f32>> {
                        cache.get(nid).map(|s| s.to_vec())
                    };
                    let get_text = |nid: NodeId| -> String {
                        // Try to get primary label from the store, fallback to node ID
                        if let Some(st) = store {
                            if let Some(node) = st.get_node(nid) {
                                if let Some(label) = node.labels.first() {
                                    return format!("[{}:{}]", label, nid.as_u64());
                                }
                            }
                        }
                        format!("node_{}", nid.as_u64())
                    };
                    let encode_fn = |e: &[f32], p: &[i32], s: i32| -> Result<usize> {
                        engine.encode_embeddings(e, p, s)
                    };
                    match bctx.manager.load_bank(
                        bank_id,
                        registry,
                        &get_embd,
                        &get_text,
                        &encode_fn,
                        Some(engine as &dyn Tokenizer),
                    ) {
                        Ok(n) => n_loaded += n,
                        Err(e) => {
                            kv_registry::kv_debug!(
                                "  [T4.2] load_bank {} failed: {}",
                                bank_id, e
                            );
                        }
                    }
                }
            }

            if n_loaded > 0 || n_evicted > 0 {
                kv_registry::kv_debug!(
                    "  [T4.2] bank selection: {} selected ({} loaded, {} nodes evicted), {} total loaded",
                    selected.len(), n_loaded, n_evicted, bctx.manager.loaded_count()
                );
            }

            // Collect all node IDs from loaded banks for filtering
            let loaded_nids: HashSet<NodeId> = bctx.manager.banks.iter()
                .filter(|b| b.loaded)
                .flat_map(|b| b.node_ids.iter().copied())
                .collect();
            Some(loaded_nids)
        } else {
            None
        }
    } else {
        None
    };

    // Run the generic retrieval to get scored nodes (only if graph is loaded)
    let (mut scored_nodes_for_query, adjacency, node_texts) =
        if let (Some(st), Some(sch)) = (store, schema) {
            retrieve_nodes(
                engine,
                st,
                sch,
                query,
                max_nodes,
                token_budget,
                embd_cache,
                coactivation,
            )?
        } else {
            (Vec::new(), HashMap::new(), HashMap::new())
        };

    // T4.2: Filter scored nodes to only those in loaded banks (structural elimination).
    // This is the core elimination: nodes from non-loaded banks are never considered,
    // reducing the working set from N_total to N_loaded_banks.
    if let Some(ref loaded_nids) = loaded_bank_nodes {
        let before = scored_nodes_for_query.len();
        scored_nodes_for_query.retain(|n| loaded_nids.contains(&n.id));
        let after = scored_nodes_for_query.len();
        if before != after {
            kv_registry::kv_debug!(
                "  [T4.2] structural elimination: {}/{} nodes retained ({}% eliminated)",
                after, before,
                if before > 0 { (before - after) * 100 / before } else { 0 }
            );
        }
    }

    // Ξ(t) T2: Enrich with GNN composite score E(t) = α·graph + β·gnn - γ·cost
    if let Some(ctx) = gnn_ctx {
        if ctx.gnn.n_updates() >= 20 && !scored_nodes_for_query.is_empty() {
            let data_nids: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();
            let query_embed = persona::fact_gnn::query_embedding(query);
            let gnn_scores = ctx
                .gnn
                .score_node_ids(ctx.persona_store, &query_embed, &data_nids);

            // Normalize graph scores to [0, 1] range
            let max_graph = scored_nodes_for_query
                .iter()
                .map(|n| n._score)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(1e-8);

            for node in &mut scored_nodes_for_query {
                let gnn_s = gnn_scores.get(&node.id).copied().unwrap_or(0.0);
                node.gnn_score = Some(gnn_s);

                // E(t) = 0.5·graph_norm + 0.35·gnn + 0.15·baseline
                let graph_norm = node._score / max_graph;
                let composite = 0.5 * graph_norm + 0.35 * gnn_s as f64 + 0.15;
                node._score = composite;
            }

            // Re-sort by composite score
            scored_nodes_for_query.sort_by(|a, b| {
                b._score
                    .partial_cmp(&a._score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    if scored_nodes_for_query.is_empty() {
        // No graph data — generate from conversation context + system prompt
        // Format using model's native chat template (extracted from GGUF metadata).
        let fallback_text = engine.format_user_turn(&format!("{query} /no_think"))?;
        let tokens = engine.tokenize(&fallback_text, false, true)?;
        // Clean seq_id=1 from any previous generation residue, then copy context
        engine.clear_seq(1);
        engine.seq_cp(0, 1, 0, -1);
        let mut filter = ThinkFilter::new();
        let mut first_visible = true;
        let spinner = match output {
            OutputMode::Stdout => Some(Spinner::start()),
            OutputMode::Channel(_) => None,
        };
        let spinner_alive = spinner.as_ref().map(|s| s.alive.clone());
        ctl.generating.store(true, Ordering::SeqCst);
        ctl.sigint_received.store(false, Ordering::SeqCst);
        // Use 4096 tokens max — thinking models (Qwen3) consume 500-800 tokens
        // in <think>...</think> for complex questions, leaving too little for the
        // visible response at 1024. The previous 256 limit was catastrophic.
        let output_ref = &output;
        let (resp, _, _signals) =
            engine.generate(&tokens, registry.next_pos, 4096, 1, |piece| {
                // Ctrl+C during generation → stop
                if ctl.sigint_received.load(Ordering::Relaxed) {
                    return false;
                }
                let visible = filter.feed(piece);
                if !visible.is_empty() {
                    match output_ref {
                        OutputMode::Stdout => {
                            if first_visible {
                                if let Some(ref a) = spinner_alive {
                                    a.store(false, Ordering::Relaxed);
                                }
                                print!("assistant> ");
                                let _ = io::stdout().flush();
                                first_visible = false;
                            }
                            print!("{}", visible);
                            let _ = io::stdout().flush();
                        }
                        OutputMode::Channel(tx) => {
                            first_visible = false;
                            let _ = tx.send(visible);
                        }
                    }
                }
                true
            })?;
        drop(spinner);
        ctl.generating.store(false, Ordering::SeqCst);
        let interrupted = ctl.sigint_received.swap(false, Ordering::SeqCst);
        if interrupted {
            ctl.gen_interrupted.store(true, Ordering::SeqCst);
            if matches!(output, OutputMode::Stdout) {
                eprint!("\n\x1b[90m  (interrompu)\x1b[0m");
            }
        }
        let remaining = filter.flush();
        if !remaining.is_empty() {
            match output {
                OutputMode::Stdout => {
                    print!("{}", remaining);
                }
                OutputMode::Channel(tx) => {
                    let _ = tx.send(remaining);
                }
            }
        }
        if matches!(output, OutputMode::Stdout) {
            println!("\n");
        }
        // CRITICAL: clean seq_id=1 after fallback generation.
        engine.clear_seq(1);
        // Diagnostic: if the visible response was empty, the model produced only <think> content
        if resp.trim().is_empty() {
            eprintln!("  [debug] Fallback generation returned empty response");
        } else if resp.contains("<think>") && !resp.contains("</think>") {
            eprintln!(
                "  [debug] Fallback response has unclosed <think> tag ({} chars)",
                resp.len()
            );
        } else if resp.starts_with("<think>") {
            let visible = think_filter::strip_think_tags(&resp);
            if visible.trim().is_empty() {
                eprintln!(
                    "  [debug] Fallback response was ALL think content ({} chars raw)",
                    resp.len()
                );
            }
        }
        return Ok((resp, Vec::new(), None, None));
    }

    // Identify which nodes need to be loaded into KV
    let needed_ids: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();
    let missing = registry.find_missing(&needed_ids);
    registry.touch(&needed_ids);


    // Estimate tokens needed for missing nodes
    let est_missing_tokens: i32 = missing
        .iter()
        .filter_map(|nid| node_texts.get(nid))
        .map(|text| (text.len() as f64 / 3.5) as i32 + 10)
        .sum();

    // Evict LRU if needed (protect nodes used in current query)
    let protected: HashSet<NodeId> = needed_ids.iter().copied().collect();
    registry.ensure_capacity(est_missing_tokens, kv_capacity, &protected, engine);

    // Encode missing nodes into KV via FFI.
    // Phase C: if embedding cache is available and injection_ratio > 0,
    // inject some nodes via batch.embd (1 KV position) instead of text tokens (N positions).
    let use_embd = embd_injection_ratio > 0.0 && embd_cache.is_some();
    let mut embd_injected = 0u32;
    let mut text_encoded = 0u32;

    for &nid in &missing {
        if let Some(text) = node_texts.get(&nid) {
            // Decide: embedding injection or text encoding?
            let should_inject = use_embd && {
                // Deterministic per-node decision based on node ID hash
                let hash = (nid.as_u64().wrapping_mul(2654435761) as u32) as f32 / u32::MAX as f32;
                hash < embd_injection_ratio
            };

            if should_inject {
                if let Some(cache) = embd_cache {
                    if let Some(embd) = cache.get(nid) {
                        let embd_vec = embd.to_vec();

                        // Phase C7: embedding + micro-tags (if graph is available)
                        let tag = match (store, schema) {
                            (Some(st), Some(sch)) => Some(get_micro_tag(st, sch, nid, 80)),
                            _ => None,
                        };

                        let inject_result = if let Some(ref tag_text) = tag {
                            // C7 path: 1 embedding + tag tokens (~3 positions)
                            registry.register_embedding_with_tags(
                                nid,
                                text,
                                &embd_vec,
                                tag_text,
                                |e, p, s| engine.encode_embeddings(e, p, s),
                                engine,
                            )
                        } else {
                            // C4 fallback: pure embedding (1 position)
                            registry.register_embedding(nid, text, &embd_vec, |e, p, s| {
                                engine.encode_embeddings(e, p, s)
                            })
                        };

                        match inject_result {
                            Ok(()) => {
                                embd_injected += 1;
                                continue;
                            }
                            Err(e) => {
                                kv_registry::kv_debug!(
                                    "  ⚠ embed injection failed for node {:?}: {}, fallback to text",
                                    nid, e
                                );
                            }
                        }
                    }
                }
            }

            // Fallback: standard text encoding
            registry.register(nid, text, engine)?;
            text_encoded += 1;
        }
    }

    if use_embd && (embd_injected > 0 || text_encoded > 0) {
        kv_registry::kv_debug!(
            "  [C4] nodes: {} via embedding, {} via text (ratio={:.0}%)",
            embd_injected,
            text_encoded,
            embd_injection_ratio * 100.0
        );
    }

    let _encode_time = t_start.elapsed().as_millis();

    // ── Phase D: Round-based tier promotion/demotion ──────────────
    if let Some(tracker) = round_tracker.as_deref_mut() {
        // D3.4: Apply reward adjustments from previous round to current scores
        let adjustments = tracker.get_reward_adjustments();
        if !adjustments.is_empty() {
            for node in &mut scored_nodes_for_query {
                if let Some(&reward) = adjustments.get(&node.id) {
                    node._score *= 1.0 + 0.3 * reward as f64;
                }
            }
            // Re-sort after adjustment
            scored_nodes_for_query.sort_by(|a, b| {
                b._score
                    .partial_cmp(&a._score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Compute query embedding for domain shift detection
        let query_embd = compute_text_embedding(engine, query).unwrap_or_default();

        // Record this round
        let scored_pairs: Vec<(NodeId, f64)> = scored_nodes_for_query
            .iter()
            .map(|n| (n.id, n._score))
            .collect();
        tracker.record_round(&scored_pairs, query_embd, registry);

        // Get demotions (decay + domain shift)
        let demotions = tracker.get_demotions(registry);
        let mut n_demoted = 0u32;
        for (nid, dtype) in &demotions {
            let result = match dtype {
                DemotionType::AlphaToBeta => registry.demote_to_beta(*nid, engine),
                DemotionType::BetaToGamma | DemotionType::AlphaToGamma => {
                    registry.demote_to_gamma(*nid, engine)
                }
            };
            if result.is_ok() {
                n_demoted += 1;
            }
        }

        // Get promotions (candidates → Alpha)
        let to_promote = tracker.get_promotions(&scored_pairs, registry);
        let mut n_promoted = 0u32;
        let mut promoted_nids: Vec<NodeId> = Vec::new();
        for nid in to_promote {
            if !registry.tier_budget.can_promote() {
                break;
            }
            // Get micro-tag for promotion
            let tag = match (store, schema) {
                (Some(st), Some(sch)) => get_micro_tag(st, sch, nid, 80),
                _ => continue,
            };
            if registry.promote_to_alpha(nid, &tag, engine).is_ok() {
                n_promoted += 1;
                promoted_nids.push(nid);
            }
        }

        // D6: log rescoring candidates
        if n_promoted > 0 {
            if gnn_ctx.is_some() || embd_cache.is_some() {
                let fact_scores = tracker.get_fact_scores();
                let n_with_facts = promoted_nids
                    .iter()
                    .filter(|nid| fact_scores.contains_key(nid))
                    .count();
                kv_registry::kv_debug!(
                    "  [D6] {} promoted nodes pending rescore ({} with fact_scores)",
                    n_promoted, n_with_facts
                );
            }
        }

        if n_demoted > 0 || n_promoted > 0 {
            let (a, b, g) = registry.tier_distribution();
            let shift = if tracker.detect_domain_shift(0.3) {
                " [DOMAIN SHIFT]"
            } else {
                ""
            };
            kv_registry::kv_debug!(
                "  [D2] round {}: +{} promoted, -{} demoted (Α={} Β={} Γ={}){}",
                tracker.current_round(),
                n_promoted,
                n_demoted,
                a,
                b,
                g,
                shift
            );
        }
    }

    // ── Find relevant conversation fragments ──────────────────────
    let (conv_node_ids, conv_adjacency) =
        conv_frags.find_relevant_with_cold(query, registry, engine, kv_capacity, cold_search);

    // Pull in graph nodes referenced by relevant conv fragments
    // (enables "donne plus de details" to re-surface Elun nodes from the last turn)
    const MAX_CONV_PULL: usize = 10;
    let mut conv_pulled_ids: Vec<NodeId> = Vec::new();
    for frag in &conv_frags.fragments {
        if conv_node_ids.contains(&frag.node_id) {
            let mut pulled_from_frag = 0;
            for &gn in &frag.related_graph_nodes {
                if pulled_from_frag >= MAX_CONV_PULL {
                    break;
                }
                let already_in_query = scored_nodes_for_query.iter().any(|n| n.id == gn);
                if !already_in_query && registry.get_slot(gn).is_some() {
                    conv_pulled_ids.push(gn);
                    pulled_from_frag += 1;
                }
            }
        }
    }
    if !conv_pulled_ids.is_empty() {
        registry.touch(&conv_pulled_ids);
    }

    // If the query is vague (all scores equal = no strong match) and conv fragments
    // provide context, prefer fragment-referenced nodes over generic retrieval.
    let vague_query = {
        let scores: HashSet<u64> = scored_nodes_for_query
            .iter()
            .map(|n| (n._score * 100.0) as u64)
            .collect();
        scores.len() <= 2 // all nodes have same score = nothing specific matched
    };
    let conv_provides_context = !conv_pulled_ids.is_empty();

    // Build QueryContext from registry positions (not from re-tokenization)
    // Bank assignment: top 25% → bank 0 (core), next 25% → bank 1, etc.
    let total_scored = scored_nodes_for_query.len();
    let bank_for_idx = |idx: usize| -> u32 {
        if total_scored == 0 {
            return 0;
        }
        let ratio = idx as f32 / total_scored as f32;
        if ratio < 0.25 {
            0
        } else if ratio < 0.50 {
            1
        } else if ratio < 0.75 {
            2
        } else {
            3
        }
    };
    let mut ctx_nodes: Vec<ContextNode> = Vec::new();
    if vague_query && conv_provides_context {
        // Vague query + conv context: only include graph nodes that are also
        // referenced by fragments (or already in fragment pull list)
        let pulled_set: HashSet<NodeId> = conv_pulled_ids.iter().copied().collect();
        for (idx, cn) in scored_nodes_for_query.iter().enumerate() {
            if pulled_set.contains(&cn.id) {
                if let Some(slot) = registry.get_slot(cn.id) {
                    ctx_nodes.push(ContextNode {
                        id: cn.id,
                        token_start: slot.start,
                        token_end: slot.end,
                        bank: bank_for_idx(idx),
                    });
                }
            }
        }
    } else {
        for (idx, cn) in scored_nodes_for_query.iter().enumerate() {
            if let Some(slot) = registry.get_slot(cn.id) {
                ctx_nodes.push(ContextNode {
                    id: cn.id,
                    token_start: slot.start,
                    token_end: slot.end,
                    bank: bank_for_idx(idx),
                });
            }
        }
    }

    // Add graph nodes pulled from conv fragment references
    for &gn in &conv_pulled_ids {
        if let Some(slot) = registry.get_slot(gn) {
            // Avoid duplicates
            if !ctx_nodes.iter().any(|n| n.id == gn) {
                ctx_nodes.push(ContextNode {
                    id: gn,
                    token_start: slot.start,
                    token_end: slot.end,
                    bank: 1, // conv-pulled graph nodes = relations tier
                });
            }
        }
    }

    // Add conversation fragment nodes to context
    for &conv_nid in &conv_node_ids {
        if let Some(slot) = registry.get_slot(conv_nid) {
            ctx_nodes.push(ContextNode {
                id: conv_nid,
                token_start: slot.start,
                token_end: slot.end,
                bank: 2, // conversation fragments = 2-hop context tier
            });
        }
    }

    // Merge adjacency: graph adjacency + conv fragment adjacency
    let mut merged_adjacency = adjacency;
    for (nid, neighbors) in conv_adjacency {
        merged_adjacency.entry(nid).or_default().extend(neighbors);
    }
    // Conv-pulled nodes should see each other (they were part of the same context)
    if conv_pulled_ids.len() > 1 {
        for i in 0..conv_pulled_ids.len() {
            for j in 0..conv_pulled_ids.len() {
                if i != j {
                    merged_adjacency
                        .entry(conv_pulled_ids[i])
                        .or_default()
                        .insert(conv_pulled_ids[j]);
                }
            }
        }
    }

    let ctx = QueryContext {
        total_tokens: registry.next_pos,
        header_tokens: registry.header_end,
        nodes: ctx_nodes,
        adjacency: merged_adjacency,
        self_embed_positions: self_embed_positions.to_vec(),
    };

    // Collect relevant graph node IDs for the caller (to link conv fragment)
    let relevant_graph_nodes: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();

    // Phase 4 AFE: Compile selected formula with actual node context
    let compiled_formula: Option<CompiledFormula> = selected_formula.and_then(|sel| {
        if matches!(sel.op, AttnOp::Identity) {
            return None; // Identity = no overlay
        }
        let n_nodes = ctx.nodes.len();
        if n_nodes == 0 {
            return None;
        }

        // Build graph_distances from adjacency via BFS (short-path between context nodes)
        let mut graph_distances = vec![u8::MAX; n_nodes * n_nodes];
        // Set self-distance = 0
        for i in 0..n_nodes {
            graph_distances[i * n_nodes + i] = 0;
        }
        // Build adjacency index: node_id → ctx_node index
        let id_to_idx: HashMap<NodeId, usize> = ctx.nodes.iter().enumerate()
            .map(|(i, n)| (n.id, i))
            .collect();
        // Set direct neighbors (distance = 1)
        for (i, node) in ctx.nodes.iter().enumerate() {
            if let Some(neighbors) = ctx.adjacency.get(&node.id) {
                for &nbr in neighbors {
                    if let Some(&j) = id_to_idx.get(&nbr) {
                        graph_distances[i * n_nodes + j] = 1;
                        graph_distances[j * n_nodes + i] = 1;
                    }
                }
            }
        }
        // BFS-extend: Floyd-Warshall for short distances (n_nodes typically < 50)
        for k in 0..n_nodes {
            for i in 0..n_nodes {
                for j in 0..n_nodes {
                    let d_ik = graph_distances[i * n_nodes + k];
                    let d_kj = graph_distances[k * n_nodes + j];
                    if d_ik != u8::MAX && d_kj != u8::MAX {
                        let d_ikj = d_ik.saturating_add(d_kj);
                        if d_ikj < graph_distances[i * n_nodes + j] {
                            graph_distances[i * n_nodes + j] = d_ikj;
                        }
                    }
                }
            }
        }

        // Count edges for graph_density
        let n_edges: usize = graph_distances.iter()
            .filter(|&&d| d == 1)
            .count() / 2; // undirected
        let max_edges = if n_nodes > 1 { n_nodes * (n_nodes - 1) / 2 } else { 1 };
        let graph_density = n_edges as f32 / max_edges as f32;

        // Coactivation data from the coactivation map (if available)
        let mut coactivation_data = vec![0.0f32; n_nodes * n_nodes];
        if let Some(coact) = coactivation {
            for i in 0..n_nodes {
                for j in (i + 1)..n_nodes {
                    let a = ctx.nodes[i].id;
                    let b = ctx.nodes[j].id;
                    let key = if a < b { (a, b) } else { (b, a) };
                    if let Some(entry) = coact.get(&key) {
                        coactivation_data[i * n_nodes + j] = entry.decay_score;
                        coactivation_data[j * n_nodes + i] = entry.decay_score;
                    }
                }
            }
        }

        let compile_ctx = CompileContext {
            n_nodes,
            n_head: engine.n_heads() as usize,
            n_pos: 0,
            header_tokens: ctx.header_tokens as usize,
            graph_distances,
            synapse_energies: vec![0.0; n_nodes * n_nodes], // TODO: from GNN edge scores
            coactivations: coactivation_data,
            node_ages: vec![0.0; n_nodes], // TODO: from registry query_counter
            current_entropy: 0.0,
            current_token_count: 0,
            context_type: String::new(),
            graph_density,
        };
        match compile(&sel.op, &compile_ctx) {
            Ok(compiled) if !compiled.is_noop() => {
                kv_registry::kv_debug!(
                    "  [AFE] compiled formula '{}': density={:.2}, n_nodes={}",
                    sel.name, graph_density, n_nodes
                );
                Some(compiled)
            }
            Ok(_) => None,
            Err(e) => {
                kv_registry::kv_debug!("  [AFE] compile error: {:?}", e);
                None
            }
        }
    });

    let (response, avg_entropy, first_token_id, first_step_logits) =
        generate_with_mask(engine, &ctx, query, ctl, output, head_router, compiled_formula.as_ref(), iptr_snapshot, state_metrics)?;

    // Ξ(t) Phase B/B3: Compute per-head ablation reward if HeadRouter is active
    let ablation = if let (Some(router), Some(ftid)) = (head_router, first_token_id) {
        let query_text = engine.format_user_turn(&format!("{query} /no_think"))
            .unwrap_or_else(|_| format!("<|im_end|>\n<|im_start|>user\n{query} /no_think<|im_end|>\n<|im_start|>assistant\n"));
        match engine.tokenize(&query_text, false, true) {
            Ok(qtokens) => {
                match compute_ablation_reward(
                    engine,
                    &ctx,
                    &qtokens,
                    ftid,
                    first_step_logits.as_deref(),
                    router,
                ) {
                    Ok(abl) => Some(abl),
                    Err(e) => {
                        kv_registry::kv_debug!("  [B3] Ablation reward failed: {e}");
                        None
                    }
                }
            }
            Err(_) => None,
        }
    } else {
        None
    };

    // Phase D3: Feed ablation reward back to round tracker for tier adjustments
    if let (Some(abl), Some(tracker)) = (&ablation, round_tracker.as_deref_mut()) {
        if !abl.node_rewards.is_empty() {
            let (n_accel_demote, n_preempt_promote) =
                tracker.reward_feedback(&abl.node_rewards, &ctx.adjacency, registry, engine);
            if n_accel_demote > 0 || n_preempt_promote > 0 {
                kv_registry::kv_debug!(
                    "  [D3] reward feedback: {} accelerated demotes, {} preemptive promotes",
                    n_accel_demote, n_preempt_promote
                );
            }
        }
    }

    Ok((response, relevant_graph_nodes, avg_entropy, ablation))
}

/// E4: Periodic Hilbert re-layout trigger.
///
/// Should be called after each round. Checks if `round_count % relayout_interval == 0`
/// and if there are co-activations to fuse. If so, recomputes the Hilbert layout
/// by fusing static graph edges with learned co-activations.
///
/// - `frozen_nodes`: nodes currently in the KV cache (keep their positions)
/// - Returns the number of nodes that got new positions, or 0 if no re-layout needed.
pub fn maybe_relayout(
    round_count: u64,
    relayout_interval: u64,
    layout: &mut HilbertLayout,
    db_adjacency: &HashMap<NodeId, std::collections::HashSet<NodeId>>,
    coactivation: &CoactivationMap,
    frozen_nodes: &std::collections::HashSet<NodeId>,
    base_position: u32,
    beta: f32,
) -> usize {
    if relayout_interval == 0 || round_count == 0 || round_count % relayout_interval != 0 {
        return 0;
    }
    if coactivation.is_empty() {
        return 0;
    }

    // Convert CoactivationMap to the format expected by build_fused_adjacency
    let coact_pairs: Vec<((NodeId, NodeId), f32)> = coactivation
        .iter()
        .map(|(&pair, entry)| (pair, entry.decay_score))
        .collect();

    let (weighted_adj, node_ids) = build_fused_adjacency(db_adjacency, &coact_pairs, beta);

    let updated = layout.update_from_fused(&weighted_adj, &node_ids, base_position, frozen_nodes);

    if updated > 0 {
        kv_registry::kv_debug!(
            "  [E4] Hilbert re-layout: {} nodes, {} co-activation edges fused, {} positions updated",
            node_ids.len(),
            coact_pairs.len(),
            updated
        );
    }

    updated
}
