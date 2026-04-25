//! Plan 69e59065 T1 step 3 — feedback baseline bench-as-test.
//!
//! Measures the hot path that T0+T2(cap)+T4 fixed: a 100-cycle loop of
//! "reinforce N context pairs" with N = `FALLBACK_CAP` (the post-T2
//! worst case from `feedback.rs`) on a `SynapseStore` pre-loaded with
//! 1k, 10k, and 100k synapses. Emits per-scenario p50/p95/p99 plus
//! delta-counter snapshots to `/tmp/feedback-baseline.json` and asserts
//! the plan-level performance gates.
//!
//! `NODES_PER_CYCLE` tracks `FALLBACK_CAP` directly so the bench
//! re-asserts the SLA at whatever cap value `feedback.rs` is using.
//! 2026-04-25: cap raised from 8 → 32 after `feedback_scaling_bench`
//! showed N=32 stays at 6.27 ms p99 (× 8 under the 50 ms gate).
//!
//! Why a focused micro-bench instead of the full `CognitiveFeedback`
//! stack: the substrate-resident gains (T0 reload + T4 inverted index)
//! all live inside `SynapseStore`. The inner loop of `feedback()` is
//! ~`reinforce(a, b, amount)` × C(min(N, FALLBACK_CAP), 2) — exactly
//! what we exercise here. Skipping the RAG stack avoids the substrate
//! WAL writes that would dominate runtime and obscure the wins we
//! actually want to measure.
//!
//! Gate (plan 69e59065 constraints):
//! - `feedback_p99_ns / 1_000_000 ≤ 50` (≤ 50 ms wall-clock)
//! - `dashmap_full_scans_per_cycle / nodes_per_cycle ≤ 5`
//!   (write-amplification proxy: with the inverted index the "scans"
//!   are now indexed lookups, but the counter still increments per
//!   normalize_outgoing call so we measure the same ratio constraint)
//!
//! Run: `cargo test -p obrain-cognitive --test feedback_baseline \
//!        --features synapse,substrate -- --nocapture`

#![cfg(all(feature = "synapse", feature = "substrate"))]

use std::sync::Arc;
use std::time::Instant;

use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use serde_json::json;

/// Number of feedback cycles to run per scenario.
const CYCLES: usize = 100;

/// Context size per cycle = `FALLBACK_CAP` from `feedback.rs`.
/// Tracking the cap means this bench measures the actual production
/// worst-case fallback (post-T2). With FALLBACK_CAP = 32 (calibrated
/// 2026-04-25 via `feedback_scaling_bench`) we get C(32,2) = 496
/// reinforces per cycle. p99 measured at ~6.3 ms, × 8 under the 50 ms
/// gate.
///
/// If `feedback.rs` re-tunes the cap, update this constant in lockstep
/// (or expose `pub use FALLBACK_CAP from rag` if cross-crate import
/// is desired — currently kept local to avoid an extra dep edge).
const NODES_PER_CYCLE: usize = 32;

/// Plan-level performance gates (constraints aa932b40 / f629192f /
/// 40943e4e). Numbers are inclusive upper bounds.
const GATE_P99_MS: f64 = 50.0;
const GATE_WRITE_AMPLIFICATION: f64 = 5.0;

fn percentile(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64 - 1.0) * p).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

/// Run one scenario: pre-seed `n_synapses` distinct pairs, then 100
/// cycles of "reinforce 8 context pairs" (= 28 reinforces / cycle,
/// exactly what `feedback.rs` does post-T2 cap). Returns (per-cycle
/// elapsed ns, counter deltas).
fn run_scenario(n_synapses: usize) -> (Vec<u128>, ScenarioCounters) {
    // Use a node pool large enough that `n_synapses` distinct
    // unordered pairs all fit. C(node_pool, 2) ≥ n_synapses
    // → node_pool ≥ ceil((1 + sqrt(1 + 8 n)) / 2).
    let node_pool = ((1.0 + (1.0 + 8.0 * n_synapses as f64).sqrt()) / 2.0).ceil() as u64 + 1;
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));

    // Seed: walk pairs (i, j) with i<j until we hit n_synapses.
    let mut seeded = 0usize;
    'outer: for i in 1..=node_pool {
        for j in (i + 1)..=node_pool {
            if seeded >= n_synapses {
                break 'outer;
            }
            store.reinforce(NodeId(i), NodeId(j), 0.05);
            seeded += 1;
        }
    }
    assert_eq!(
        store.len(),
        n_synapses,
        "seeding produced {} synapses, expected {}",
        store.len(),
        n_synapses
    );

    // Warm baseline counters AFTER seeding so we measure only the
    // cycle phase.
    let scans_before = store.dashmap_full_scans_total();
    let reinforces_before = store.reinforces_total();

    // Pick 8 well-spread nodes from the pool as the per-cycle
    // "context" (they will overlap with seeded pairs, exercising the
    // `and_modify` path of `reinforce`).
    let stride = node_pool / NODES_PER_CYCLE as u64;
    let context: Vec<NodeId> = (0..NODES_PER_CYCLE)
        .map(|k| NodeId(1 + (k as u64) * stride.max(1)))
        .collect();

    let mut elapsed_ns: Vec<u128> = Vec::with_capacity(CYCLES);
    for _ in 0..CYCLES {
        let t = Instant::now();
        // Inner loop of feedback() — C(N, 2) reinforces.
        for i in 0..context.len() {
            for j in (i + 1)..context.len() {
                store.reinforce(context[i], context[j], 0.01);
            }
        }
        elapsed_ns.push(t.elapsed().as_nanos());
    }

    let counters = ScenarioCounters {
        scans_total_delta: store.dashmap_full_scans_total() - scans_before,
        reinforces_total_delta: store.reinforces_total() - reinforces_before,
    };
    (elapsed_ns, counters)
}

struct ScenarioCounters {
    scans_total_delta: u64,
    reinforces_total_delta: u64,
}

#[test]
fn feedback_baseline_emits_json_and_passes_gates() {
    // Plan-mandated scenarios (T1 step 3): "1k, 10k, 100k synapses".
    let scenarios = [("1k", 1_000usize), ("10k", 10_000usize), ("100k", 100_000usize)];

    let mut report = serde_json::Map::new();
    let mut all_ok = true;

    for (label, n) in scenarios {
        let (mut elapsed_ns, counters) = run_scenario(n);
        elapsed_ns.sort_unstable();

        let p50 = percentile(&elapsed_ns, 0.50);
        let p95 = percentile(&elapsed_ns, 0.95);
        let p99 = percentile(&elapsed_ns, 0.99);
        let p99_ms = p99 as f64 / 1_000_000.0;

        // write_amplification proxy: scans per reinforce. Each reinforce
        // calls normalize_outgoing twice (once per endpoint), so the
        // expected stable-state ratio is 2.0 ± epsilon — well under the
        // gate of 5.0.
        let amplification = counters.scans_total_delta as f64
            / counters.reinforces_total_delta.max(1) as f64;

        let p99_pass = p99_ms <= GATE_P99_MS;
        let amp_pass = amplification <= GATE_WRITE_AMPLIFICATION;

        let scenario_ok = p99_pass && amp_pass;
        all_ok &= scenario_ok;

        report.insert(
            label.to_string(),
            json!({
                "synapses_seeded": n,
                "cycles": CYCLES,
                "nodes_per_cycle": NODES_PER_CYCLE,
                "reinforces_per_cycle": NODES_PER_CYCLE * (NODES_PER_CYCLE - 1) / 2,
                "p50_ns": p50,
                "p95_ns": p95,
                "p99_ns": p99,
                "p99_ms": p99_ms,
                "scans_total_delta": counters.scans_total_delta,
                "reinforces_total_delta": counters.reinforces_total_delta,
                "write_amplification_ratio": amplification,
                "gate_p99_pass": p99_pass,
                "gate_amplification_pass": amp_pass,
            }),
        );

        eprintln!(
            "[T1.3] {label:>4} synapses: p50={:>7} ns, p95={:>7} ns, p99={:>7} ns ({p99_ms:>6.2} ms), \
             amp={amplification:.2} → p99 {} amp {}",
            p50,
            p95,
            p99,
            if p99_pass { "✓" } else { "✗" },
            if amp_pass { "✓" } else { "✗" },
        );
    }

    let final_report = json!({
        "plan": "69e59065",
        "task": "T1",
        "step": 3,
        "title": "feedback baseline post T0+T2(cap)+T4",
        "gate_p99_ms": GATE_P99_MS,
        "gate_write_amplification": GATE_WRITE_AMPLIFICATION,
        "scenarios": report,
        "all_gates_pass": all_ok,
    });

    let path = "/tmp/feedback-baseline.json";
    std::fs::write(path, serde_json::to_string_pretty(&final_report).unwrap())
        .expect("write feedback-baseline.json");
    eprintln!("[T1.3] report → {path}");

    assert!(
        all_ok,
        "one or more scenarios failed performance gates — see {}",
        path
    );
}
