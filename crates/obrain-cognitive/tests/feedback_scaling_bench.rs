//! Plan 69e59065 — parametric scaling bench for the feedback() fallback
//! reinforce loop. The current `FALLBACK_CAP = 8` (rag/feedback.rs:38)
//! was set conservatively after a kernel watchdog incident at the
//! uncapped `C(N, 2)` worst case (gotcha 49938809). Now that:
//!
//!   - T0 fixed the synapse rehydration (no more "0 synapses post-reload")
//!   - T2 capped the fallback (no more 4 950-pair stampede)
//!   - T4 gave normalize_outgoing an inverted index (O(degree) not O(|S|))
//!   - the reward path baseline showed × 3937 SLA margin on 100k nodes
//!
//! the question is: how high can `FALLBACK_CAP` go while keeping a
//! **safety margin of 4×** under the 50 ms p99 gate?  Reinforces grow as
//! `C(N, 2) = N(N-1)/2`, so each step up costs a quadratic in operations
//! — but the inverted index keeps each individual reinforce O(degree).
//!
//! Per-cycle inner loop is identical to `feedback.rs:167-176`:
//! ```ignore
//! for i in 0..N { for j in i+1..N { synapse_store.reinforce(a, b, amt); } }
//! ```
//!
//! Reports `/tmp/feedback-scaling.json` with p50/p95/p99 + scans counter
//! delta + reinforces-per-cycle, and prints a verdict picking the largest
//! N that stays under `GATE / SAFETY_FACTOR`.
//!
//! Run: `cargo test -p obrain-cognitive --test feedback_scaling_bench \
//!        --features synapse,substrate --release -- --nocapture`

#![cfg(all(feature = "synapse", feature = "substrate"))]

use std::sync::Arc;
use std::time::Instant;

use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use serde_json::json;

/// 100 cycles of feedback() per N — same as feedback_baseline.
const CYCLES: usize = 100;

/// 50 ms is the plan-level p99 gate (constraint 40943e4e).
const GATE_P99_MS: f64 = 50.0;

/// Safety margin: only N values whose p99 is < GATE / SAFETY are
/// declared "safe to raise FALLBACK_CAP to". 4× leaves headroom for the
/// rest of feedback() (energy boosts, reward writes, mention scan,
/// chain walks) which this bench does NOT exercise.
const SAFETY_FACTOR: f64 = 4.0;

/// Pre-seed size: how many distinct synapse pairs the store starts with
/// before each scenario. 10k = realistic for a warm user brain.
const SEED_SYNAPSES: usize = 10_000;

/// N values swept. C(200, 2) = 19 900 reinforces / cycle is the upper
/// edge — it tells us where the cliff is, even if we won't ship there.
const N_SWEEP: &[usize] = &[8, 16, 24, 32, 48, 64, 100, 200];

fn percentile(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64 - 1.0) * p).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

/// One scaling scenario: pre-seed `seed_synapses` distinct pairs in a
/// fresh `SynapseStore`, then run `CYCLES` cycles of "C(N,2) reinforce
/// calls" exactly like `feedback.rs:167-176`. Returns per-cycle elapsed
/// in ns and counter deltas.
fn run_for_n(n: usize, seed_synapses: usize) -> (Vec<u128>, u64, u64) {
    // Pool large enough that the pre-seed plus the rotating context fit
    // without index collisions.
    let node_pool =
        ((1.0 + (1.0 + 8.0 * seed_synapses as f64).sqrt()) / 2.0).ceil() as u64 + n as u64 + 1;
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));

    // Pre-seed: walk pairs (i, j) with i<j until we hit seed_synapses.
    let mut seeded = 0usize;
    'outer: for i in 1..=node_pool {
        for j in (i + 1)..=node_pool {
            if seeded >= seed_synapses {
                break 'outer;
            }
            store.reinforce(NodeId(i), NodeId(j), 0.05);
            seeded += 1;
        }
    }

    // Rotating context of N evenly-spread node ids drawn from the pool.
    // Same construction as feedback_baseline so results are comparable.
    let stride = (node_pool / n as u64).max(1);
    let context: Vec<NodeId> = (0..n).map(|k| NodeId(1 + (k as u64) * stride)).collect();
    debug_assert_eq!(context.len(), n);

    let scans_before = store.dashmap_full_scans_total();
    let reinforces_before = store.reinforces_total();

    let mut elapsed_ns: Vec<u128> = Vec::with_capacity(CYCLES);
    for _ in 0..CYCLES {
        let t = Instant::now();
        // Inner loop, byte-for-byte identical to feedback.rs:167-176.
        for i in 0..context.len() {
            for j in (i + 1)..context.len() {
                store.reinforce(context[i], context[j], 0.01);
            }
        }
        elapsed_ns.push(t.elapsed().as_nanos());
    }

    let scans_delta = store.dashmap_full_scans_total() - scans_before;
    let reinforces_delta = store.reinforces_total() - reinforces_before;
    (elapsed_ns, scans_delta, reinforces_delta)
}

#[test]
fn feedback_scaling_finds_safe_fallback_cap() {
    let mut report = serde_json::Map::new();
    let mut largest_safe_n: Option<usize> = None;
    let safety_threshold_ms = GATE_P99_MS / SAFETY_FACTOR;

    eprintln!(
        "[scaling] sweep N ∈ {:?}, gate p99 ≤ {:.0} ms, safety threshold {:.2} ms (= gate / {:.0}×)",
        N_SWEEP, GATE_P99_MS, safety_threshold_ms, SAFETY_FACTOR,
    );
    eprintln!(
        "[scaling] each scenario: {} cycles on a SynapseStore pre-seeded with {} synapses",
        CYCLES, SEED_SYNAPSES
    );

    for &n in N_SWEEP {
        let (mut elapsed, scans_delta, reinforces_delta) = run_for_n(n, SEED_SYNAPSES);
        elapsed.sort_unstable();
        let p50 = percentile(&elapsed, 0.50);
        let p95 = percentile(&elapsed, 0.95);
        let p99 = percentile(&elapsed, 0.99);
        let p99_ms = p99 as f64 / 1_000_000.0;

        let pairs_per_cycle = n * (n - 1) / 2;
        let safe = p99_ms < safety_threshold_ms;
        if safe {
            largest_safe_n = Some(n);
        }

        report.insert(
            n.to_string(),
            json!({
                "n_context_nodes": n,
                "pairs_per_cycle": pairs_per_cycle,
                "cycles": CYCLES,
                "p50_ns": p50,
                "p95_ns": p95,
                "p99_ns": p99,
                "p99_ms": p99_ms,
                "scans_total_delta": scans_delta,
                "reinforces_total_delta": reinforces_delta,
                "scans_per_reinforce": scans_delta as f64 / reinforces_delta.max(1) as f64,
                "safe_under_gate_div_safety": safe,
            }),
        );

        eprintln!(
            "[scaling] N={:>3} ({:>5} pairs/cycle): p50={:>9} ns p95={:>9} ns p99={:>9} ns ({:>7.3} ms)  → {}",
            n,
            pairs_per_cycle,
            p50,
            p95,
            p99,
            p99_ms,
            if safe {
                format!("✓ safe (< {:.2} ms)", safety_threshold_ms)
            } else {
                format!("✗ over safety ({:.2} ms)", safety_threshold_ms)
            },
        );
    }

    let final_report = json!({
        "plan": "69e59065",
        "title": "FALLBACK_CAP scaling sweep — find the largest safe N",
        "current_fallback_cap": 8,
        "gate_p99_ms": GATE_P99_MS,
        "safety_factor": SAFETY_FACTOR,
        "safety_threshold_ms": safety_threshold_ms,
        "seed_synapses": SEED_SYNAPSES,
        "largest_safe_n": largest_safe_n,
        "recommendation": match largest_safe_n {
            Some(n) if n > 8 => format!(
                "FALLBACK_CAP can be safely raised from 8 to {n} (p99 stays under safety threshold {:.2} ms = gate / {:.0}×).",
                safety_threshold_ms, SAFETY_FACTOR
            ),
            Some(n) if n == 8 => "FALLBACK_CAP=8 is the largest safe value — keep as is.".to_string(),
            _ => "Even N=8 exceeds the safety threshold — investigate before changing FALLBACK_CAP.".to_string(),
        },
        "scenarios": report,
    });

    let path = "/tmp/feedback-scaling.json";
    std::fs::write(path, serde_json::to_string_pretty(&final_report).unwrap())
        .expect("write feedback-scaling.json");

    eprintln!("\n[scaling] report → {}", path);
    if let Some(n) = largest_safe_n {
        eprintln!(
            "[scaling] VERDICT: largest safe N = {n} (current FALLBACK_CAP = 8). \
             {} pairs/cycle at this N.",
            n * (n - 1) / 2
        );
    } else {
        eprintln!("[scaling] VERDICT: no safe N found — keep FALLBACK_CAP = 8.");
    }

    // We do NOT assert here — the bench is purely informational. The
    // assertion is the verdict above; modifying FALLBACK_CAP is a
    // separate, deliberate change driven by this report.
}
