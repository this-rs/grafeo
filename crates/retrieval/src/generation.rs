use anyhow::Result;
use kv_registry::QueryContext;
use llm_engine::mask_builder;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::Ordering;
use think_filter::ThinkFilter;

use crate::attn_compiler::CompiledFormula;
use crate::control::{GenerationControl, OutputMode, Spinner};
use crate::engine::Engine;

/// Result of per-head ablation reward computation.
#[derive(Debug, Clone)]
pub struct AblationReward {
    /// Per-bank contribution: delta log-prob when bank b is ablated.
    /// Positive = bank contributed positively (removing it hurts).
    pub bank_contributions: Vec<f32>,
    /// Per-head reward derived from bank contributions × visibility.
    /// Length = n_head.
    pub per_head_rewards: Vec<f32>,
    /// Head contribution entropy: low = few heads dominate, high = uniform.
    pub head_contribution_entropy: f32,
    /// Per-node reward: each node gets its bank's contribution.
    /// Phase D: used by RoundTracker for tier promotion/demotion.
    pub node_rewards: std::collections::HashMap<obrain_common::types::NodeId, f32>,
}

/// Returns (response_text, avg_entropy) where avg_entropy is from GenerationSignals.
pub fn generate_with_mask(
    engine: &Engine,
    ctx: &QueryContext,
    query: &str,
    ctl: &GenerationControl,
    output: &OutputMode,
    head_router: Option<&llm_engine::HeadRouter>,
    compiled_formula: Option<&CompiledFormula>,
    iptr_snapshot: Option<&crate::iptr_graph::FactSnapshot>,
    state_metrics: Option<&crate::state_bias::StateMetrics>,
) -> Result<(String, Option<f32>, Option<i32>, Option<Vec<f32>>)> {
    // /no_think MUST be in the user message for Qwen3 to reliably suppress thinking.
    // Placing it only in the system header gets "diluted" by graph context tokens.
    let query_text = format!(
        "<|im_end|>\n<|im_start|>user\n{query} /no_think<|im_end|>\n<|im_start|>assistant\n"
    );
    let query_tokens = engine.tokenize(&query_text, false, true)?;
    let query_n = query_tokens.len() as i32;

    let n_predict: i32 = 4096; // thinking models need 500-800 tokens for <think>, 1024 was truncating

    // ── Compute mask budget (dynamic) ───────────────────────────────
    // Per-head mask memory = n_groups × n_pos² × 4 bytes (f32).
    // Budget is a fraction of system RAM (default 512MB, env-overridable).
    // Cap = min(memory-derived, n_ctx) — never allocate more than the model can use.
    let use_perhead_mask = std::env::var("OBRAIN_PERHEAD")
        .map(|v| v == "1")
        .unwrap_or(false)
        || head_router.is_some();
    let mask_groups = if use_perhead_mask {
        head_router
            .map(|r| r.n_head)
            .unwrap_or(engine.n_head() as usize)
    } else {
        1
    };

    // Dynamic memory budget: default 512MB, overridable via OBRAIN_MASK_BUDGET_MB
    let max_mask_bytes: usize = std::env::var("OBRAIN_MASK_BUDGET_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512)
        * 1024
        * 1024;

    // max_positions = sqrt(budget / (4 × n_groups)), capped at n_ctx
    let n_ctx = engine.n_ctx() as i32;
    let max_mask_positions: i32 = ((max_mask_bytes / (4 * mask_groups)) as f64).sqrt() as i32;
    let max_mask_positions = max_mask_positions.min(n_ctx);
    let mask_mem_mb =
        (mask_groups as f64 * max_mask_positions as f64 * max_mask_positions as f64 * 4.0) / 1e6;

    // Count context positions (header + self-embed + relevant nodes + query)
    let mut n_context_positions: i32 = ctx.header_tokens;
    n_context_positions += ctx.self_embed_positions.len() as i32;
    for node in &ctx.nodes {
        n_context_positions += node.token_end - node.token_start;
    }
    n_context_positions += query_n;

    // n_gen_mask = remaining budget after context, capped at n_predict
    // With micro-tags (~3 pos/node) this typically gives 3500+ gen positions.
    // With full text (~50 pos/node) and many nodes it could be lower.
    let n_gen_mask: i32 = (max_mask_positions - n_context_positions)
        .max(512)
        .min(n_predict);
    kv_registry::kv_debug!(
        "  [mask] budget: {} context + {} gen = {} / {} max ({}× groups, {:.0}MB, n_ctx={})",
        n_context_positions,
        n_gen_mask,
        n_context_positions + n_gen_mask,
        max_mask_positions,
        mask_groups,
        mask_mem_mb,
        n_ctx
    );

    // ── Build sparse position map ──────────────────────────────────
    // Only include positions that matter: header + relevant nodes + query + gen.
    // Positions from non-relevant nodes in the KV are excluded (Rule 3b).
    // n_gen_mask now covers as many gen tokens as the memory budget allows,
    // eliminating the "topological leak" where continuation tokens bypass the mask.
    let mut positions: Vec<i32> = Vec::new();
    let mut pos_remap: HashMap<i32, usize> = HashMap::new();

    // Header positions
    for p in 0..ctx.header_tokens {
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
    }

    // Self-embedding positions (proprioceptive channel, between header and graph nodes)
    for &p in &ctx.self_embed_positions {
        if !pos_remap.contains_key(&p) {
            let idx = positions.len();
            pos_remap.insert(p, idx);
            positions.push(p);
        }
    }

    // Relevant node positions (from KV registry slots)
    for node in &ctx.nodes {
        for p in node.token_start..node.token_end {
            if !pos_remap.contains_key(&p) {
                let idx = positions.len();
                pos_remap.insert(p, idx);
                positions.push(p);
            }
        }
    }

    // Query + gen positions (appended AFTER all KV content)
    let q_start_real = ctx.total_tokens;
    for offset in 0..(query_n + n_gen_mask) {
        let p = q_start_real + offset;
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
    }

    let sz = positions.len();

    if sz as i32 > max_mask_positions {
        eprintln!(
            "  Warning: mask_size={} > {} ({}groups × {}²×4 = {:.0}MB > budget) — generating without mask. \
            Tip: set OBRAIN_MASK_BUDGET_MB={} to allow this.",
            sz,
            max_mask_positions,
            mask_groups,
            max_mask_positions,
            mask_mem_mb,
            // Suggest the minimum budget needed
            ((mask_groups * sz * sz * 4) as f64 / 1e6).ceil() as usize + 1
        );
        // Fallback: generate without topological mask (but still need seq_cp!)
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
        let output_ref = &output;
        let (resp, _, fallback_signals) =
            engine.generate(&query_tokens, q_start_real, n_predict, 1, |piece| {
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
        return Ok((
            resp,
            Some(fallback_signals.avg_entropy),
            fallback_signals.first_token_id,
            None,
        ));
    }

    // ── Compile attention mask (sparse) ──────────────────────────────
    // FFI: use f32::NEG_INFINITY (exact), not -1e30 (JSON workaround eliminated)
    let mut mask = vec![f32::NEG_INFINITY; sz * sz];

    let allow =
        |m: &mut [f32], real_i: i32, real_j: i32, remap: &HashMap<i32, usize>, sz: usize| {
            if let (Some(&mi), Some(&mj)) = (remap.get(&real_i), remap.get(&real_j)) {
                if mi < sz && mj < sz && real_j <= real_i {
                    m[mi * sz + mj] = 0.0;
                }
            }
        };

    let h = ctx.header_tokens;

    // Rule 0: Header positions — causal among themselves
    for i in 0..h {
        for j in 0..=i {
            allow(&mut mask, i, j, &pos_remap, sz);
        }
    }

    // Rule 1: Query + gen tokens see header + ALL relevant nodes + causal self
    // Non-relevant KV positions are NOT in pos_remap → invisible (sparse exclusion)
    for offset in 0..(query_n + n_gen_mask) {
        let i = q_start_real + offset;
        // See header
        for j in 0..h {
            allow(&mut mask, i, j, &pos_remap, sz);
        }
        // See all relevant node tokens
        for node in &ctx.nodes {
            for j in node.token_start..node.token_end {
                allow(&mut mask, i, j, &pos_remap, sz);
            }
        }
        // Causal self among query+gen tokens
        for prev in 0..=offset {
            allow(&mut mask, i, q_start_real + prev, &pos_remap, sz);
        }
    }

    // ── Set mask via FFI ──────────────────────────────────────────
    // OBRAIN_NO_MASK=1: skip topology mask entirely (A/B baseline for benchmarking).
    // OBRAIN_PERHEAD=1: per-head masking where different heads see different banks.
    // Default: broadcast topology mask (all heads see the same sparse positions).
    // ── Apply formula overlay (Phase 4 AFE) ────────────────────────
    // The compiled formula produces an overlay [n_nodes × n_nodes] that modulates
    // attention between graph nodes. We expand it to token-level positions.
    if let Some(formula) = compiled_formula {
        if let Some(ref overlay) = formula.kq_mask_overlay {
            let n_formula_nodes = ctx.nodes.len();
            if overlay.len() == n_formula_nodes * n_formula_nodes {
                let mut n_applied = 0u32;
                for (ni, node_i) in ctx.nodes.iter().enumerate() {
                    for (nj, node_j) in ctx.nodes.iter().enumerate() {
                        let val = overlay[ni * n_formula_nodes + nj];
                        if val == 0.0 {
                            continue;
                        }
                        // Expand node-pair overlay to all token-position pairs
                        for pi in node_i.token_start..node_i.token_end {
                            for pj in node_j.token_start..node_j.token_end {
                                if let (Some(&mi), Some(&mj)) =
                                    (pos_remap.get(&pi), pos_remap.get(&pj))
                                {
                                    if mi < sz && mj < sz && pj <= pi {
                                        let idx = mi * sz + mj;
                                        if val == f32::NEG_INFINITY
                                            || mask[idx] == f32::NEG_INFINITY
                                        {
                                            mask[idx] = f32::NEG_INFINITY;
                                        } else {
                                            mask[idx] += val;
                                        }
                                        n_applied += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                if n_applied > 0 {
                    kv_registry::kv_debug!(
                        "  [AFE] formula overlay applied: {} cells modified",
                        n_applied
                    );
                }
            }
        }
        // kq_b (positive bias) not applied yet — requires Phase 2 fork with kq_b support
        if formula.kq_b.is_some() {
            kv_registry::kv_debug!(
                "  [AFE] kq_b bias present but not applied (Phase 2 fork required)"
            );
        }
    }

    let no_mask = std::env::var("OBRAIN_NO_MASK")
        .map(|v| v == "1")
        .unwrap_or(false);
    let use_perhead = std::env::var("OBRAIN_PERHEAD")
        .map(|v| v == "1")
        .unwrap_or(false);
    if no_mask {
        eprintln!("  [A/B] Topology mask DISABLED (baseline mode)");
    } else if use_perhead || head_router.is_some() {
        // Phase B: per-head masking with HeadRouter (or Phase A per-head with OBRAIN_PERHEAD=1)
        let mb_nodes: Vec<mask_builder::NodePosition> = ctx
            .nodes
            .iter()
            .map(|n| mask_builder::NodePosition {
                pos_start: n.token_start,
                pos_end: n.token_end,
                bank: n.bank,
            })
            .collect();
        let config = mask_builder::default_bank_config();
        // n_head for the mask: use router's head count (reflects granularity setting),
        // or fall back to full n_head when using OBRAIN_PERHEAD env without a router.
        let mask_n_head = head_router
            .map(|r| r.n_head as i32)
            .unwrap_or_else(|| engine.n_head());
        let perhead = mask_builder::build_perhead_mask(
            &mb_nodes,
            ctx.header_tokens,
            sz as i32,
            mask_n_head,
            &config,
            head_router, // Phase B: learned α-weights, or None for Phase A fixed ratios
        );
        engine.set_attn_mask(&perhead.mask, &perhead.positions, perhead.n_head_groups)?;
    } else {
        engine.set_attn_mask(&mask, &positions, 0)?;
    }

    // ── State kq_b bias (Phase 5 — Axe B) ─────────────────────────
    // Compute and inject attention bias from SelfMetrics BEFORE generation.
    // This biases attention toward graph nodes (modulated by reward) and
    // conversation context (modulated by confidence). Static for the entire generation.
    let state_bias_disabled = std::env::var("OBRAIN_STATE_BIAS_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false);

    if !state_bias_disabled {
        if let Some(metrics) = state_metrics {
            let computer = crate::state_bias::StateBiasComputer::new(
                crate::state_bias::StateBiasConfig::default(),
            );

            // Build position map from QueryContext — separate graph nodes from conv fragments
            let graph_ranges: Vec<(i32, i32)> = ctx
                .nodes
                .iter()
                .filter(|n| n.bank != 2) // bank 2 = conversation fragments
                .map(|n| (n.token_start, n.token_end))
                .collect();
            let conv_ranges: Vec<(i32, i32)> = ctx
                .nodes
                .iter()
                .filter(|n| n.bank == 2) // bank 2 = conversation fragments
                .map(|n| (n.token_start, n.token_end))
                .collect();
            let n_kv = ctx.total_tokens;
            let position_map = crate::state_bias::StateBiasComputer::build_position_map(
                ctx.header_tokens,
                &graph_ranges,
                &conv_ranges,
                n_kv,
            );

            let n_head = engine.n_head() as usize;
            let bias = computer.compute(metrics, &position_map, n_head);
            let magnitude = crate::state_bias::StateBiasComputer::bias_magnitude(&bias);

            kv_registry::kv_debug!(
                "  [StateBias] reward={:.2}, confidence={:.2}, magnitude={:.4}, n_head={}, n_kv={}",
                metrics.reward_avg, metrics.confidence, magnitude, n_head, n_kv
            );

            if magnitude > 0.0 {
                engine.set_state_bias(&bias, n_head as i32, n_kv);
            }
        }
    }

    // ── CRITICAL: Make seq_id=0 KV entries visible to seq_id=1 ──
    // Without this, query tokens on seq_id=1 cannot attend to context
    // nodes encoded on seq_id=0. This was the root cause of the model
    // ignoring all graph context (hallucinating instead of using data).
    // Clear first to remove any stale tokens from a previous generation.
    engine.clear_seq(1);
    engine.seq_cp(0, 1, 0, -1);

    // ── IPTR dispatcher setup (Phase 5 — Axe A) ───────────────────
    // When a FactSnapshot is provided, create the dispatcher with GraphLookupTool.
    // The dispatcher runs inline during generation when entropy exceeds the adaptive
    // threshold, returning logit biases that steer the model toward graph concepts.
    let iptr_disabled = std::env::var("OBRAIN_IPTR_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let iptr_cooldown: usize = std::env::var("OBRAIN_IPTR_COOLDOWN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let mut iptr_dispatcher = if !iptr_disabled {
        iptr_snapshot.and_then(|snap| {
            if snap.is_empty() {
                return None;
            }
            let mut dispatcher = crate::iptr::IptrDispatcher::new(iptr_cooldown);
            let tool = crate::iptr_graph::GraphLookupTool::with_defaults(snap.clone());
            dispatcher.register_tool(Box::new(tool));
            kv_registry::kv_debug!(
                "  [IPTR] dispatcher ready: {} facts, cooldown={}",
                snap.len(),
                iptr_cooldown
            );
            Some(dispatcher)
        })
    } else {
        None
    };

    // Build the IPTR callback closure (captures &mut iptr_dispatcher)
    let mut iptr_callback = |entropy: f32, recent_text: &str, token_pos: u32| -> Option<HashMap<u32, f32>> {
        let dispatcher = iptr_dispatcher.as_mut()?;
        let ctx = crate::iptr::ToolContext {
            recent_text: recent_text.to_string(),
            entropy,
            already_retrieved: Vec::new(),
            token_position: token_pos,
        };
        let result = dispatcher.dispatch(&ctx)?;
        let (n_dispatch, total_us) = dispatcher.stats();
        eprintln!(
            "  [IPTR] graph_lookup triggered: {} biases, {} concepts, latency={}µs (dispatch #{}, total={}µs)",
            result.biases.len(),
            result.concepts_found.len(),
            result.latency_us,
            n_dispatch,
            total_us
        );
        Some(result.biases)
    };

    // ── Generate with streaming + auto-continuation ────────────────
    // Round 0: generate with custom mask (graph-guided attention).
    // If the model hits max_tokens without EOG, clear the mask and continue
    // with default causal attention. The model already has the graph context
    // from the first round. seq_id=1 is kept alive between rounds.
    let mut first_visible = true;
    let spinner = match output {
        OutputMode::Stdout => Some(Spinner::start()),
        OutputMode::Channel(_) => None,
    };
    let spinner_alive = spinner.as_ref().map(|s| s.alive.clone());
    ctl.generating.store(true, Ordering::SeqCst);
    ctl.sigint_received.store(false, Ordering::SeqCst);

    let mut filter = ThinkFilter::new();
    let mut full_response = String::new();
    let max_continuations = 3;
    let output_ref = &output;

    // Round 0: with mask
    let (chunk, mut hit_eog, mut next_pos, gen_signals) = engine.generate_ex(
        &query_tokens,
        q_start_real,
        n_predict,
        1,
        true, // keep_seq=true in case we need continuation
        |piece| {
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
        },
        Some(&mut iptr_callback),
    )?;
    drop(spinner);
    full_response.push_str(&chunk);

    // NOTE: Do NOT clear_attn_mask() here. The mask stays active during continuations
    // so that tokens beyond n_gen_mask still respect the topology (they'll be outside
    // the mask's position range and fall through to causal, but non-relevant KV entries
    // remain excluded from the sparse position map). The mask is cleared in cleanup below.

    let was_interrupted = ctl.sigint_received.load(Ordering::Relaxed);

    // Continuations: mask still set but gen tokens beyond n_gen_mask use causal fallback.
    // The sparse position map already excludes non-relevant nodes, so even without
    // explicit mask coverage, those KV entries are not in the position set.
    if !hit_eog && !was_interrupted {
        let cont_token = engine.tokenize(" ", false, false)?;

        for cont in 1..=max_continuations {
            if ctl.sigint_received.load(Ordering::Relaxed) {
                break;
            }

            let is_last = cont == max_continuations;

            let (chunk, eog, end_pos, _cont_signals) = engine.generate_ex(
                &cont_token,
                next_pos,
                n_predict,
                1,
                !is_last, // keep_seq alive unless last round
                |piece| {
                    if ctl.sigint_received.load(Ordering::Relaxed) {
                        return false;
                    }
                    let visible = filter.feed(piece);
                    if !visible.is_empty() {
                        match output_ref {
                            OutputMode::Stdout => {
                                if first_visible {
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
                },
                None, // IPTR: continuations don't need tool dispatch
            )?;

            next_pos = end_pos;
            full_response.push_str(&chunk);
            hit_eog = eog;

            if eog {
                // Natural end — clean up if seq was kept alive
                if !is_last {
                    engine.clear_seq(1);
                }
                break;
            }
        }

        // If we exhausted continuations without EOG, final cleanup
        if !hit_eog {
            // seq already cleaned by last generate_ex (keep_seq=false)
        }
    } else if was_interrupted {
        // Interrupted on round 0 — clean up seq_id=1
        engine.clear_seq(1);
    } else {
        // Natural end on round 0 — clean up seq_id=1
        engine.clear_seq(1);
    }

    // Clear the topology mask now that generation is complete.
    // This must happen AFTER all continuations so they benefit from
    // the sparse position exclusion (non-relevant nodes stay invisible).
    engine.clear_attn_mask();

    // Clear state bias after generation (it's specific to this query's metrics).
    if !state_bias_disabled && state_metrics.is_some() {
        engine.clear_state_bias();
    }

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

    // Diagnostic: detect empty visible responses
    if full_response.trim().is_empty() {
        eprintln!("  [debug] generate_with_mask returned empty response");
    } else if full_response.starts_with("<think>") {
        let visible = think_filter::strip_think_tags(&full_response);
        if visible.trim().is_empty() {
            eprintln!(
                "  [debug] Masked response was ALL think content ({} chars raw)",
                full_response.len()
            );
        }
    }

    Ok((
        full_response,
        Some(gen_signals.avg_entropy),
        gen_signals.first_token_id,
        gen_signals.first_step_logits,
    ))
}

/// Compute per-head ablation reward by measuring log-prob differential.
///
/// For each bank b, we decode the query tokens with bank b ablated (blocked for all heads)
/// and compare P(first_token | full_mask) vs P(first_token | ablated_b).
/// Delta = log P(full) - log P(ablated) = contribution of bank b.
///
/// The per-head reward is then: for head h, reward_h = Σ_b contribution_b × visibility_h_b
///
/// This uses seq_id=2 for the ablation decodes (cleaned up after).
pub fn compute_ablation_reward(
    engine: &Engine,
    ctx: &QueryContext,
    query_tokens: &[i32],
    first_token_id: i32,
    full_logits: Option<&[f32]>,
    router: &llm_engine::HeadRouter,
) -> Result<AblationReward> {
    let n_head = router.n_head;
    let n_bank = router.n_bank;
    let q_start = ctx.total_tokens;
    let query_n = query_tokens.len() as i32;

    // Build positions for query tokens
    let positions: Vec<i32> = (0..query_n).map(|i| q_start + i).collect();

    // Get log-prob of first_token under full mask (already decoded on seq_id=1)
    // If full_logits provided, use them; otherwise skip (return uniform)
    let full_log_prob = if let Some(logits) = full_logits {
        if (first_token_id as usize) < logits.len() {
            log_softmax_at(logits, first_token_id as usize)
        } else {
            0.0
        }
    } else {
        return Ok(AblationReward {
            bank_contributions: vec![0.0; n_bank],
            per_head_rewards: vec![0.0; n_head],
            head_contribution_entropy: (n_head as f32).ln(),
            node_rewards: std::collections::HashMap::new(),
        });
    };

    let mut bank_contributions = vec![0.0f32; n_bank];

    // For each bank, ablate it and measure log-prob drop
    for b in 0..n_bank {
        // Build ablated mask: same as broadcast mask but bank b blocked for all heads
        let ablated_nodes: Vec<mask_builder::NodePosition> = ctx
            .nodes
            .iter()
            .filter(|n| n.bank != b as u32) // exclude bank b
            .map(|n| mask_builder::NodePosition {
                pos_start: n.token_start,
                pos_end: n.token_end,
                bank: n.bank,
            })
            .collect();

        // Check if any nodes were in bank b (if not, ablation has no effect)
        let had_bank_b = ctx.nodes.iter().any(|n| n.bank == b as u32);
        if !had_bank_b {
            bank_contributions[b] = 0.0;
            continue;
        }

        // Build a simple broadcast mask (not per-head) for ablation eval
        // We only need the logits, not per-head differentiation
        let mut abl_positions: Vec<i32> = Vec::new();
        let mut abl_remap: HashMap<i32, usize> = HashMap::new();

        // Header
        for p in 0..ctx.header_tokens {
            let idx = abl_positions.len();
            abl_remap.insert(p, idx);
            abl_positions.push(p);
        }
        // Remaining nodes (bank b excluded)
        for node in &ablated_nodes {
            for p in node.pos_start..node.pos_end {
                if !abl_remap.contains_key(&p) {
                    let idx = abl_positions.len();
                    abl_remap.insert(p, idx);
                    abl_positions.push(p);
                }
            }
        }
        // Query positions
        for offset in 0..query_n {
            let p = q_start + offset;
            let idx = abl_positions.len();
            abl_remap.insert(p, idx);
            abl_positions.push(p);
        }

        let sz = abl_positions.len();
        let mut mask = vec![f32::NEG_INFINITY; sz * sz];

        let allow = |m: &mut [f32], ri: i32, rj: i32, remap: &HashMap<i32, usize>, sz: usize| {
            if let (Some(&mi), Some(&mj)) = (remap.get(&ri), remap.get(&rj)) {
                if mi < sz && mj < sz && rj <= ri {
                    m[mi * sz + mj] = 0.0;
                }
            }
        };

        // Header causal
        for i in 0..ctx.header_tokens {
            for j in 0..=i {
                allow(&mut mask, i, j, &abl_remap, sz);
            }
        }
        // Query sees header + remaining nodes + causal self
        for offset in 0..query_n {
            let i = q_start + offset;
            for j in 0..ctx.header_tokens {
                allow(&mut mask, i, j, &abl_remap, sz);
            }
            for node in &ablated_nodes {
                for j in node.pos_start..node.pos_end {
                    allow(&mut mask, i, j, &abl_remap, sz);
                }
            }
            for prev in 0..=offset {
                allow(&mut mask, i, q_start + prev, &abl_remap, sz);
            }
        }

        // Set ablated mask, decode on seq_id=2, get logits
        engine.set_attn_mask(&mask, &abl_positions, 0)?;
        engine.clear_seq(2);
        engine.seq_cp(0, 2, 0, -1);

        let ablated_logits = engine.decode_for_logits(query_tokens, &positions, 2)?;
        engine.clear_seq(2);
        engine.clear_attn_mask();

        let ablated_log_prob = if (first_token_id as usize) < ablated_logits.len() {
            log_softmax_at(&ablated_logits, first_token_id as usize)
        } else {
            full_log_prob // no change if token out of range
        };

        // Contribution = how much removing bank b hurts (positive = bank was helpful)
        bank_contributions[b] = full_log_prob - ablated_log_prob;
    }

    // Distribute bank contributions to heads via visibility weights
    let visibility = router.forward(); // [n_head × n_bank]
    let mut per_head_rewards = vec![0.0f32; n_head];

    for h in 0..n_head {
        let mut reward = 0.0f32;
        let mut vis_sum = 0.0f32;
        for b in 0..n_bank {
            let vis = visibility[h * n_bank + b];
            reward += bank_contributions[b] * vis;
            vis_sum += vis;
        }
        // Normalize by total visibility to avoid scale bias
        if vis_sum > 1e-8 {
            per_head_rewards[h] = reward / vis_sum;
        }
    }

    // Compute head contribution entropy
    let head_contribution_entropy = {
        let abs_rewards: Vec<f32> = per_head_rewards.iter().map(|r| r.abs()).collect();
        let sum: f32 = abs_rewards.iter().sum();
        if sum < 1e-8 {
            (n_head as f32).ln() // max entropy = uniform
        } else {
            let mut entropy = 0.0f32;
            for &r in &abs_rewards {
                let p = r / sum;
                if p > 1e-8 {
                    entropy -= p * p.ln();
                }
            }
            entropy
        }
    };

    // Phase D: map bank contributions to individual nodes
    let mut node_rewards = std::collections::HashMap::new();
    for node in &ctx.nodes {
        let b = node.bank as usize;
        if b < bank_contributions.len() {
            node_rewards.insert(node.id, bank_contributions[b]);
        }
    }

    Ok(AblationReward {
        bank_contributions,
        per_head_rewards,
        head_contribution_entropy,
        node_rewards,
    })
}

/// Compute log P(token) from raw logits using log-softmax.
fn log_softmax_at(logits: &[f32], token_idx: usize) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits
        .iter()
        .map(|&l| (l - max_logit).exp())
        .sum::<f32>()
        .ln();
    logits[token_idx] - max_logit - log_sum_exp
}
