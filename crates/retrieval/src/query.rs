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
use crate::scoring::retrieve_nodes;
use crate::generation::generate_with_mask;

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
) -> Result<(String, Vec<NodeId>)> {
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
    let (scored_nodes_for_query, adjacency, node_texts) =
        if let (Some(st), Some(sch)) = (store, schema) {
            retrieve_nodes(engine, st, sch, query, max_nodes, token_budget)?
        } else {
            (Vec::new(), HashMap::new(), HashMap::new())
        };

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
        // Use 1024 tokens max (not 256) to avoid truncated/garbage responses.
        // The low max_tokens was causing the model to output short garbage in some cases.
        let output_ref = &output;
        let (resp, _) = engine.generate(&tokens, registry.next_pos, 1024, 1, |piece| {
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
        return Ok((resp, Vec::new()));
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

    // Encode missing nodes into KV via FFI
    for &nid in &missing {
        if let Some(text) = node_texts.get(&nid) {
            registry.register(nid, text, engine)?;
        }
    }

    let _encode_time = t_start.elapsed().as_millis();

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

    let response = generate_with_mask(engine, &ctx, query, ctl, output)?;
    Ok((response, relevant_graph_nodes))
}
