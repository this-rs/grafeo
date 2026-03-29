use anyhow::Result;
use llm_engine::mask_builder;
use kv_registry::QueryContext;
use think_filter::ThinkFilter;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::Ordering;

use crate::engine::Engine;
use crate::control::{GenerationControl, OutputMode, Spinner};

pub fn generate_with_mask(
    engine: &Engine,
    ctx: &QueryContext,
    query: &str,
    ctl: &GenerationControl,
    output: &OutputMode,
) -> Result<String> {
    // /no_think MUST be in the user message for Qwen3 to reliably suppress thinking.
    // Placing it only in the system header gets "diluted" by graph context tokens.
    let query_text = format!(
        "<|im_end|>\n<|im_start|>user\n{query} /no_think<|im_end|>\n<|im_start|>assistant\n"
    );
    let query_tokens = engine.tokenize(&query_text, false, true)?;
    let query_n = query_tokens.len() as i32;

    let n_predict: i32 = 1024;

    // ── Build sparse position map ──────────────────────────────────
    // Only include positions that matter: header + relevant nodes + query + gen.
    // Positions from non-relevant nodes in the KV are excluded (Rule 3b).
    // n_gen_mask MUST cover the full generation to prevent fallback to default
    // causal attention (which would let the model see ALL KV including irrelevant nodes).
    let n_gen_mask: i32 = n_predict;

    let mut positions: Vec<i32> = Vec::new();
    let mut pos_remap: HashMap<i32, usize> = HashMap::new();

    // Header positions
    for p in 0..ctx.header_tokens {
        let idx = positions.len();
        pos_remap.insert(p, idx);
        positions.push(p);
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
    let max_mask_positions: i32 = 4000;

    if sz as i32 > max_mask_positions {
        eprintln!("  Warning: mask_size={} > {} — generating without mask.", sz, max_mask_positions);
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
        let (resp, _) = engine.generate(&query_tokens, q_start_real, n_predict, 1, |piece| {
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
        return Ok(resp);
    }

    // ── Compile attention mask (sparse) ──────────────────────────────
    // FFI: use f32::NEG_INFINITY (exact), not -1e30 (JSON workaround eliminated)
    let mut mask = vec![f32::NEG_INFINITY; sz * sz];

    let allow = |m: &mut [f32], real_i: i32, real_j: i32, remap: &HashMap<i32, usize>, sz: usize| {
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
        for j in 0..h { allow(&mut mask, i, j, &pos_remap, sz); }
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
    // Per-head masking: when enabled, different heads see different banks.
    // Enable via OBRAIN_PERHEAD=1 env var.
    let use_perhead = std::env::var("OBRAIN_PERHEAD").map(|v| v == "1").unwrap_or(false);
    if use_perhead {
        // Convert ContextNodes to mask_builder::NodePosition with bank from retrieval ranking.
        let mb_nodes: Vec<mask_builder::NodePosition> = ctx.nodes.iter().map(|n| {
            mask_builder::NodePosition {
                pos_start: n.token_start,
                pos_end: n.token_end,
                bank: n.bank,
            }
        }).collect();
        let config = mask_builder::default_bank_config();
        let perhead = mask_builder::build_perhead_mask(
            &mb_nodes,
            ctx.header_tokens,
            sz as i32,
            engine.n_head(),
            &config,
        );
        engine.set_attn_mask(&perhead.mask, &perhead.positions, perhead.n_head_groups)?;
    } else {
        engine.set_attn_mask(&mask, &positions, 0)?;
    }

    // ── CRITICAL: Make seq_id=0 KV entries visible to seq_id=1 ──
    // Without this, query tokens on seq_id=1 cannot attend to context
    // nodes encoded on seq_id=0. This was the root cause of the model
    // ignoring all graph context (hallucinating instead of using data).
    // Clear first to remove any stale tokens from a previous generation.
    engine.clear_seq(1);
    engine.seq_cp(0, 1, 0, -1);

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
    let (chunk, mut hit_eog, mut next_pos) = engine.generate_ex(
        &query_tokens, q_start_real, n_predict, 1,
        true, // keep_seq=true in case we need continuation
        |piece| {
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
        }
    )?;
    drop(spinner);
    full_response.push_str(&chunk);

    // Clear mask — either done or switching to causal for continuations
    engine.clear_attn_mask();

    let was_interrupted = ctl.sigint_received.load(Ordering::Relaxed);

    // Continuations: no mask, causal attention only
    if !hit_eog && !was_interrupted {
        let cont_token = engine.tokenize(" ", false, false)?;

        for cont in 1..=max_continuations {
            if ctl.sigint_received.load(Ordering::Relaxed) { break; }

            let is_last = cont == max_continuations;

            let (chunk, eog, end_pos) = engine.generate_ex(
                &cont_token, next_pos, n_predict, 1,
                !is_last, // keep_seq alive unless last round
                |piece| {
                    if ctl.sigint_received.load(Ordering::Relaxed) { return false; }
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
                }
            )?;

            next_pos = end_pos;
            full_response.push_str(&chunk);
            hit_eog = eog;

            if eog {
                // Natural end — clean up if seq was kept alive
                if !is_last { engine.clear_seq(1); }
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

    // Diagnostic: detect empty visible responses
    if full_response.trim().is_empty() {
        eprintln!("  [debug] generate_with_mask returned empty response");
    } else if full_response.starts_with("<think>") {
        let visible = think_filter::strip_think_tags(&full_response);
        if visible.trim().is_empty() {
            eprintln!("  [debug] Masked response was ALL think content ({} chars raw)", full_response.len());
        }
    }

    Ok(full_response)
}
