//! Plan 69e59065 T5 baseline bench-as-test — measures the reward path
//! that `cognitive_brain.rs::feedback` (lines 2710–2729) currently
//! drives synchronously on each cycle.
//!
//! Hot loop modelled here, per cycle, for `NODES_PER_CYCLE` context
//! nodes:
//!   - `get_node_property(nid, PROP_REWARD)`        — substrate chain walk
//!   - `set_node_property(nid, PROP_REWARD, f64)`   — chain append
//!   - utility boost + scar partial heal — column 5-bit, fast (NOT measured
//!     in this micro-bench: those are already column-resident; the question
//!     is whether the substrate property chain ops dominate p99)
//!
//! T5 (in the plan) proposes moving the reward to a RAM-only RewardStore
//! with a background consolidator. That refactor is justified only if
//! the `get_node_property + set_node_property` pair already pushes p99
//! over the 50 ms gate at realistic context sizes.
//!
//! The bench:
//!   - Pre-creates 1k / 10k / 100k substrate nodes
//!   - Pre-seeds PROP_REWARD on a fraction of them so subsequent reads
//!     hit a non-empty chain (closer to steady state)
//!   - Runs 100 cycles of (8 reads + 8 writes) on rotating context
//!   - Measures per-cycle elapsed + emits JSON
//!   - Asserts the same p99 ≤ 50 ms gate
//!
//! Run: `cargo test -p obrain-cognitive --test reward_path_baseline \
//!        --features synapse,substrate -- --nocapture`

#![cfg(all(feature = "synapse", feature = "substrate"))]

use std::time::Instant;

use obrain_common::types::{NodeId, Value};
use obrain_core::graph::GraphStore;
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use serde_json::json;

const PROP_REWARD: &str = "_reward";
const CYCLES: usize = 100;
const NODES_PER_CYCLE: usize = 8;
const GATE_P99_MS: f64 = 50.0;

fn percentile(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64 - 1.0) * p).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

fn run_scenario(n_nodes: usize) -> (Vec<u128>, /*pre_seeded*/ usize) {
    // FD-leak hardening — see gotcha note 8484a4b7-d87b-47d1-8cc3-9f4f5392312b
    // (project 95a5fc9d, 2026-04-25). On macOS each `SubstrateStore::create`
    // opens ~15-20 fd (nodes/edges/props.v2/wal/engrams/sidecars). With 3
    // scenarios `1k/10k/100k` running in the same test fn, drops only fire
    // at function return → ~60 fd accumulate AND `munmap` is async on macOS.
    // Three cumulative levers below: explicit drop scope, explicit `td`
    // unlink, and a 50ms breather so the kernel file table catches up
    // before the next scenario opens its own substrate.
    let td = tempfile::tempdir().expect("tempdir");
    let result = {
        let store = SubstrateStore::create(td.path().join("kb")).expect("create substrate");

        // Pre-create n_nodes nodes (single label).
        let mut ids = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            ids.push(store.create_node(&["n"]));
        }

        // Pre-seed PROP_REWARD on every 4th node so subsequent reads hit a
        // populated chain (steady-state realism). The rest stay empty —
        // also realistic, since fresh context nodes have no reward history.
        let mut pre_seeded = 0;
        for (k, &nid) in ids.iter().enumerate() {
            if k % 4 == 0 {
                store.set_node_property(nid, PROP_REWARD, Value::Float64(0.5));
                pre_seeded += 1;
            }
        }
        store.flush().expect("flush after seed");

        // Rotating context: each cycle picks 8 evenly-spread nodes, shifting
        // by 1 each cycle so we hit different shards / pages.
        let prop_key: obrain_common::types::PropertyKey = PROP_REWARD.into();
        let stride = (n_nodes / NODES_PER_CYCLE).max(1);

        let mut elapsed_ns: Vec<u128> = Vec::with_capacity(CYCLES);
        for cycle in 0..CYCLES {
            let context: Vec<NodeId> = (0..NODES_PER_CYCLE)
                .map(|k| ids[(cycle + k * stride) % n_nodes])
                .collect();

            let t = Instant::now();
            for &nid in &context {
                let old = store
                    .get_node_property(nid, &prop_key)
                    .and_then(|v| {
                        if let Value::Float64(f) = v {
                            Some(f)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0.0);
                let new = old * 0.8 + 0.5 * 0.2; // EMA, identical to feedback() formula
                store.set_node_property(nid, PROP_REWARD, Value::Float64(new));
            }
            elapsed_ns.push(t.elapsed().as_nanos());
        }

        // Final flush, then explicit drop — closes mmap fd before scope exits.
        store.flush().expect("flush before drop");
        drop(store);
        (elapsed_ns, pre_seeded)
    };
    drop(td); // unlink tempdir immediately
    // macOS munmap is async (TLB shootdown deferral). Give the kernel a
    // brief window to reclaim the substrate fd before the next scenario
    // opens its own. 50 ms is well below the test's per-scenario runtime
    // and avoids ENFILE on the 100k variant.
    std::thread::sleep(std::time::Duration::from_millis(50));
    result
}

#[test]
fn reward_path_baseline_emits_json_and_passes_gate() {
    let scenarios = [
        ("1k", 1_000usize),
        ("10k", 10_000usize),
        ("100k", 100_000usize),
    ];

    let mut report = serde_json::Map::new();
    let mut all_ok = true;

    for (label, n) in scenarios {
        let (mut elapsed_ns, pre_seeded) = run_scenario(n);
        elapsed_ns.sort_unstable();

        let p50 = percentile(&elapsed_ns, 0.50);
        let p95 = percentile(&elapsed_ns, 0.95);
        let p99 = percentile(&elapsed_ns, 0.99);
        let p99_ms = p99 as f64 / 1_000_000.0;
        let p99_pass = p99_ms <= GATE_P99_MS;
        all_ok &= p99_pass;

        report.insert(
            label.to_string(),
            json!({
                "nodes": n,
                "pre_seeded_with_reward": pre_seeded,
                "cycles": CYCLES,
                "ops_per_cycle": NODES_PER_CYCLE * 2, // get + set
                "p50_ns": p50,
                "p95_ns": p95,
                "p99_ns": p99,
                "p99_ms": p99_ms,
                "gate_p99_pass": p99_pass,
            }),
        );

        eprintln!(
            "[T5.bench] {label:>4} nodes: p50={:>9} ns, p95={:>9} ns, p99={:>9} ns ({p99_ms:>7.3} ms), \
             pre-seeded={pre_seeded} → p99 {}",
            p50,
            p95,
            p99,
            if p99_pass { "✓" } else { "✗" },
        );
    }

    let final_report = json!({
        "plan": "69e59065",
        "task": "T5",
        "step": "baseline-pre-implementation",
        "title": "reward path (substrate get+set) baseline before T5 RewardStore refactor",
        "gate_p99_ms": GATE_P99_MS,
        "scenarios": report,
        "all_gates_pass": all_ok,
    });

    let path = "/tmp/reward-path-baseline.json";
    std::fs::write(path, serde_json::to_string_pretty(&final_report).unwrap())
        .expect("write reward-path-baseline.json");
    eprintln!("[T5.bench] report → {path}");

    // Note: this assert tells us whether T5's RAM-only refactor is
    // necessary for the SLA. PASS → T5 becomes a backlog optimisation
    // and we can jump to T8. FAIL → implement T5 properly before T8.
    assert!(
        all_ok,
        "reward path p99 over 50 ms gate — T5 RewardStore refactor required (see {})",
        path
    );
}
