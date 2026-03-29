use anyhow::Result;
use graph_schema::GraphSchema;
use kv_registry::{Tokenizer, KvNodeRegistry, KvBank, ConvFragments, ContextNode, QueryContext};
use think_filter::ThinkFilter;
use obrain_common::types::NodeId;
use obrain_core::graph::lpg::LpgStore;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::engine::Engine;
use crate::control::{GenerationControl, OutputMode, Spinner};
use crate::node_embedding::{NodeEmbeddingCache, compute_text_embedding};
use crate::scoring::{retrieve_nodes, get_micro_tag};
use crate::generation::{generate_with_mask, compute_ablation_reward, AblationReward};
use crate::round_tracker::{RoundTracker, DemotionType};

/// Optional GNN scoring context for composite E(t) decision.
pub struct GnnContext<'a> {
    pub gnn: &'a persona::fact_gnn::FactGNN,
    pub persona_store: &'a LpgStore,
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
            let missing: Vec<NodeId> = bank.node_ids.iter()
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

    // Run the generic retrieval to get scored nodes (only if graph is loaded)
    let (mut scored_nodes_for_query, adjacency, node_texts) =
        if let (Some(st), Some(sch)) = (store, schema) {
            retrieve_nodes(engine, st, sch, query, max_nodes, token_budget, embd_cache)?
        } else {
            (Vec::new(), HashMap::new(), HashMap::new())
        };

    // Ξ(t) T2: Enrich with GNN composite score E(t) = α·graph + β·gnn - γ·cost
    if let Some(ctx) = gnn_ctx {
        if ctx.gnn.n_updates() >= 20 && !scored_nodes_for_query.is_empty() {
            let data_nids: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();
            let query_embed = persona::fact_gnn::query_embedding(query);
            let gnn_scores = ctx.gnn.score_node_ids(ctx.persona_store, &query_embed, &data_nids);

            // Normalize graph scores to [0, 1] range
            let max_graph = scored_nodes_for_query.iter()
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
            scored_nodes_for_query.sort_by(|a, b|
                b._score.partial_cmp(&a._score).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    if scored_nodes_for_query.is_empty() {
        // No graph data — generate from conversation context + system prompt
        // /no_think MUST be in the user message for Qwen3 to reliably suppress thinking.
        // Placing it only in the system header gets "diluted" by graph context tokens.
        let fallback_text = format!(
            "<|im_end|>\n<|im_start|>user\n{query} /no_think<|im_end|>\n<|im_start|>assistant\n"
        );
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
        let (resp, _, _signals) = engine.generate(&tokens, registry.next_pos, 4096, 1, |piece| {
            // Ctrl+C during generation → stop
            if ctl.sigint_received.load(Ordering::Relaxed) { return false; }
            let visible = filter.feed(piece);
            if !visible.is_empty() {
                match output_ref {
                    OutputMode::Stdout => {
                        if first_visible {
                            if let Some(ref a) = spinner_alive { a.store(false, Ordering::Relaxed); }
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
                OutputMode::Stdout => { print!("{}", remaining); }
                OutputMode::Channel(tx) => { let _ = tx.send(remaining); }
            }
        }
        if matches!(output, OutputMode::Stdout) { println!("\n"); }
        // CRITICAL: clean seq_id=1 after fallback generation.
        engine.clear_seq(1);
        // Diagnostic: if the visible response was empty, the model produced only <think> content
        if resp.trim().is_empty() {
            eprintln!("  [debug] Fallback generation returned empty response");
        } else if resp.contains("<think>") && !resp.contains("</think>") {
            eprintln!("  [debug] Fallback response has unclosed <think> tag ({} chars)", resp.len());
        } else if resp.starts_with("<think>") {
            let visible = think_filter::strip_think_tags(&resp);
            if visible.trim().is_empty() {
                eprintln!("  [debug] Fallback response was ALL think content ({} chars raw)", resp.len());
            }
        }
        return Ok((resp, Vec::new(), None, None));
    }

    // Identify which nodes need to be loaded into KV
    let needed_ids: Vec<NodeId> = scored_nodes_for_query.iter().map(|n| n.id).collect();
    let missing = registry.find_missing(&needed_ids);
    registry.touch(&needed_ids);

    // Estimate tokens needed for missing nodes
    let est_missing_tokens: i32 = missing.iter()
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
                                nid, text, &embd_vec, tag_text,
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
                                eprintln!("  ⚠ embed injection failed for node {:?}: {}, fallback to text", nid, e);
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
        eprintln!("  [C4] nodes: {} via embedding, {} via text (ratio={:.0}%)",
            embd_injected, text_encoded, embd_injection_ratio * 100.0);
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
            scored_nodes_for_query.sort_by(|a, b|
                b._score.partial_cmp(&a._score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Compute query embedding for domain shift detection
        let query_embd = compute_text_embedding(engine, query)
            .unwrap_or_default();

        // Record this round
        let scored_pairs: Vec<(NodeId, f64)> = scored_nodes_for_query.iter()
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
            if result.is_ok() { n_demoted += 1; }
        }

        // Get promotions (candidates → Alpha)
        let to_promote = tracker.get_promotions(&scored_pairs, registry);
        let mut n_promoted = 0u32;
        let mut promoted_nids: Vec<NodeId> = Vec::new();
        for nid in to_promote {
            if !registry.tier_budget.can_promote() { break; }
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
                let n_with_facts = promoted_nids.iter()
                    .filter(|nid| fact_scores.contains_key(nid))
                    .count();
                eprintln!("  [D6] {} promoted nodes pending rescore ({} with fact_scores)",
                    n_promoted, n_with_facts);
            }
        }

        if n_demoted > 0 || n_promoted > 0 {
            let (a, b, g) = registry.tier_distribution();
            let shift = if tracker.detect_domain_shift(0.3) { " [DOMAIN SHIFT]" } else { "" };
            eprintln!("  [D2] round {}: +{} promoted, -{} demoted (Α={} Β={} Γ={}){}",
                tracker.current_round(), n_promoted, n_demoted, a, b, g, shift);
        }
    }

    // ── Find relevant conversation fragments ──────────────────────
    let (conv_node_ids, conv_adjacency) = conv_frags.find_relevant(query, registry);

    // Pull in graph nodes referenced by relevant conv fragments
    // (enables "donne plus de details" to re-surface Elun nodes from the last turn)
    const MAX_CONV_PULL: usize = 10;
    let mut conv_pulled_ids: Vec<NodeId> = Vec::new();
    for frag in &conv_frags.fragments {
        if conv_node_ids.contains(&frag.node_id) {
            let mut pulled_from_frag = 0;
            for &gn in &frag.related_graph_nodes {
                if pulled_from_frag >= MAX_CONV_PULL { break; }
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
        let scores: HashSet<u64> = scored_nodes_for_query.iter()
            .map(|n| (n._score * 100.0) as u64)
            .collect();
        scores.len() <= 2 // all nodes have same score = nothing specific matched
    };
    let conv_provides_context = !conv_pulled_ids.is_empty();

    // Build QueryContext from registry positions (not from re-tokenization)
    // Bank assignment: top 25% → bank 0 (core), next 25% → bank 1, etc.
    let total_scored = scored_nodes_for_query.len();
    let bank_for_idx = |idx: usize| -> u32 {
        if total_scored == 0 { return 0; }
        let ratio = idx as f32 / total_scored as f32;
        if ratio < 0.25 { 0 } else if ratio < 0.50 { 1 } else if ratio < 0.75 { 2 } else { 3 }
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
                    merged_adjacency.entry(conv_pulled_ids[i]).or_default().insert(conv_pulled_ids[j]);
                }
            }
        }
    }

    let ctx = QueryContext {
        total_tokens: registry.next_pos,
        header_tokens: registry.header_end,
        nodes: ctx_nodes,
        adjacency: merged_adjacency,
    };

    // Collect relevant graph node IDs for the caller (to link conv fragment)
    let relevant_graph_nodes: Vec<NodeId> = scored_nodes_for_query.iter()
        .map(|n| n.id)
        .collect();

    let (response, avg_entropy, first_token_id, first_step_logits) =
        generate_with_mask(engine, &ctx, query, ctl, output, head_router)?;

    // Ξ(t) Phase B/B3: Compute per-head ablation reward if HeadRouter is active
    let ablation = if let (Some(router), Some(ftid)) = (head_router, first_token_id) {
        let query_text = format!(
            "<|im_end|>\n<|im_start|>user\n{query} /no_think<|im_end|>\n<|im_start|>assistant\n"
        );
        match engine.tokenize(&query_text, false, true) {
            Ok(qtokens) => {
                match compute_ablation_reward(
                    engine, &ctx, &qtokens, ftid,
                    first_step_logits.as_deref(),
                    router,
                ) {
                    Ok(abl) => Some(abl),
                    Err(e) => {
                        eprintln!("  [B3] Ablation reward failed: {e}");
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
            let (n_accel_demote, n_preempt_promote) = tracker.reward_feedback(
                &abl.node_rewards, &ctx.adjacency, registry, engine,
            );
            if n_accel_demote > 0 || n_preempt_promote > 0 {
                eprintln!("  [D3] reward feedback: {} accelerated demotes, {} preemptive promotes",
                    n_accel_demote, n_preempt_promote);
            }
        }
    }

    Ok((response, relevant_graph_nodes, avg_entropy, ablation))
}
