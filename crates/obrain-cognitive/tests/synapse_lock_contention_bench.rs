//! Plan 69e59065 T1 step 4 — empirical lock contention probe.
//!
//! The watchdog crash that motivated this plan (gotcha 49938809) was
//! initially diagnosed as memory pressure, then revised to "lock
//! contention + CPU spin" after RSS data refuted the mmap hypothesis.
//! That revision was a *guess*. This bench is the empirical check: how
//! does `SynapseStore::reinforce` throughput / tail latency scale with
//! thread count, and where is the contention if any?
//!
//! Per `reinforce()` call, the synapse store touches up to 10 DashMap
//! shard locks:
//!  - `synapses.entry(key)`              — 1 shard
//!  - `node_to_keys.entry(src)`          — 1 shard (via index_insert)
//!  - `node_to_keys.entry(dst)`          — 1 shard
//!  - `normalize_outgoing(src)`          — 1 read + N mutations
//!  - `normalize_outgoing(dst)`          — 1 read + N mutations
//!  - `access_order.insert(key)`         — 1 shard (touch)
//!  - `edge_ids.get/insert`              — 1 shard (when graph_store)
//!  - `synapses.get(&key)` for persist   — 1 shard
//!  - 3× `persist_edge_f64`              — graph_store (LPG)
//!
//! Default DashMap shard count is `4 × num_cpus`. On an 8-core machine
//! that's 32 shards. With 21 concurrent reinforces, the chance two
//! threads hit the same shard on any of 8+ shard operations is high.
//!
//! ## Bench design
//!
//! Two workload modes:
//! 1. **shared**  — all threads draw from a pool of 100 nodes. Worst
//!    case: every reinforce maps to the same ~10 hot keys, all shards
//!    are contended. Mirrors the "feedback() on a 100-node context
//!    with all 21 threads hammering the same hub" crash scenario.
//! 2. **disjoint** — each thread reinforces a private range of 100
//!    nodes (offset by `thread_id × 1000`). No shared keys. Best case
//!    for DashMap sharding — collisions only via unlucky hashing.
//!
//! For each (workload, thread_count) ∈ workloads × {1, 2, 4, 8, 16, 21}:
//!   - Spawn N threads, each does `OPS_PER_THREAD` reinforces
//!   - Measure wall-clock time per call (push to a per-thread Vec)
//!   - Aggregate: total throughput (ops/sec), p50/p95/p99/p99.9 (ns)
//!   - Compute contention factor: `p99(N) / p99(1)`
//!
//! ## Verdict logic
//!
//! Linear scaling (throughput ∝ thread_count, p99 stable) → not
//! lock-bound; the original "lock contention" hypothesis is wrong.
//!
//! Sub-linear scaling with bounded p99 increase → moderate contention,
//! probably acceptable; the watchdog crash was likely amplified by
//! lots-of-CPU-doing-real-work + 8 cores offline.
//!
//! Sub-linear scaling with **explosive p99 tail** → confirms the lock
//! contention hypothesis; T3 (move LPG metadata writes off hot path)
//! becomes critical.
//!
//! Run: `cargo test -p obrain-cognitive --test synapse_lock_contention_bench \
//!        --features synapse,substrate --release -- --nocapture`

#![cfg(all(feature = "synapse", feature = "substrate"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use serde_json::json;

/// Operations per thread. Picked so 1 thread runs ~50ms (dwarfs spawn
/// jitter) and N=21 still completes in seconds.
const OPS_PER_THREAD: usize = 5_000;

/// Thread counts swept. 21 mirrors the crash scenario from gotcha
/// 49938809; 32 stresses past the typical DashMap shard count on
/// 8-core machines (4 × num_cpus = 32 shards default).
const THREAD_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 21, 32];

/// In "shared" mode, pool of nodes that ALL threads draw pairs from.
/// 100 = upper end of the historical 100-node fallback all-pairs
/// behaviour (pre-T2 cap), so this represents the real worst case.
const SHARED_POOL_NODES: u64 = 100;

/// In "disjoint" mode, each thread has its own range of `DISJOINT_RANGE`
/// node ids starting at `thread_id × DISJOINT_OFFSET`.
const DISJOINT_RANGE: u64 = 100;
const DISJOINT_OFFSET: u64 = 1_000;

fn percentile(sorted_ns: &[u128], p: f64) -> u128 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((sorted_ns.len() as f64 - 1.0) * p).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

#[derive(Debug, Clone, Copy)]
enum Workload {
    Shared,
    Disjoint,
}

impl Workload {
    fn label(&self) -> &'static str {
        match self {
            Workload::Shared => "shared",
            Workload::Disjoint => "disjoint",
        }
    }

    /// Pick a (src, dst) pair for thread `tid` on iteration `i`.
    fn pick_pair(&self, tid: usize, i: usize) -> (NodeId, NodeId) {
        match self {
            Workload::Shared => {
                // Round-robin across the shared pool. tid offsets the
                // starting point so threads don't all begin on (1,2).
                let a = (i + tid) as u64 % SHARED_POOL_NODES;
                let b = (i + tid + 1 + (i / 7) as usize) as u64 % SHARED_POOL_NODES;
                if a == b {
                    (NodeId(a + 1), NodeId(((a + 2) % SHARED_POOL_NODES) + 1))
                } else {
                    (NodeId(a + 1), NodeId(b + 1))
                }
            }
            Workload::Disjoint => {
                let base = tid as u64 * DISJOINT_OFFSET;
                let a = base + (i as u64 % DISJOINT_RANGE);
                let b = base + ((i as u64 + 1) % DISJOINT_RANGE);
                if a == b {
                    (NodeId(a + 1), NodeId(base + ((a + 2) % DISJOINT_RANGE) + 1))
                } else {
                    (NodeId(a + 1), NodeId(b + 1))
                }
            }
        }
    }
}

/// Run a fresh SynapseStore with `n_threads` threads, each doing
/// `OPS_PER_THREAD` reinforces under `workload`. Returns:
///   - aggregate ops/sec (total ops / wall-clock total)
///   - per-call latency distribution (sorted ns) merged across threads
///   - delta `dashmap_full_scans_total`
fn run_workload(workload: Workload, n_threads: usize) -> (f64, Vec<u128>, u64) {
    let store = Arc::new(SynapseStore::new(SynapseConfig::default()));

    // Pre-seed the shared pool a bit so the first reinforces don't all
    // hit the cold-create path (which is artificially slow on the very
    // first insert per shard). Disjoint mode doesn't pre-seed; threads
    // begin on a clean private range.
    if matches!(workload, Workload::Shared) {
        for i in 1..SHARED_POOL_NODES {
            store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
        }
    }

    let scans_before = store.dashmap_full_scans_total();

    // Each thread records its own latencies; we merge after.
    let started = Arc::new(AtomicUsize::new(0));
    let total = Instant::now();

    let mut handles = Vec::with_capacity(n_threads);
    for tid in 0..n_threads {
        let store = Arc::clone(&store);
        let started = Arc::clone(&started);
        handles.push(std::thread::spawn(move || {
            // Cheap thread barrier so we measure actual peak contention,
            // not a thundering-herd start.
            started.fetch_add(1, Ordering::Relaxed);
            while started.load(Ordering::Relaxed) < n_threads {
                std::hint::spin_loop();
            }

            let mut my_latencies: Vec<u128> = Vec::with_capacity(OPS_PER_THREAD);
            for i in 0..OPS_PER_THREAD {
                let (src, dst) = workload.pick_pair(tid, i);
                let t = Instant::now();
                store.reinforce(src, dst, 0.01);
                my_latencies.push(t.elapsed().as_nanos());
            }
            my_latencies
        }));
    }

    let mut all_latencies: Vec<u128> =
        Vec::with_capacity(n_threads * OPS_PER_THREAD);
    for h in handles {
        let v = h.join().expect("thread panicked");
        all_latencies.extend(v);
    }
    let elapsed_total = total.elapsed();
    all_latencies.sort_unstable();

    let total_ops = (n_threads * OPS_PER_THREAD) as f64;
    let ops_per_sec = total_ops / elapsed_total.as_secs_f64();
    let scans_delta = store.dashmap_full_scans_total() - scans_before;

    (ops_per_sec, all_latencies, scans_delta)
}

#[test]
fn synapse_lock_contention_sweep() {
    let workloads = [Workload::Shared, Workload::Disjoint];

    let mut report = serde_json::Map::new();
    let mut headline = serde_json::Map::new();

    for w in workloads {
        eprintln!(
            "\n=== workload = {} (ops/thread = {}) ===",
            w.label(),
            OPS_PER_THREAD
        );
        eprintln!(
            "{:>6} {:>14} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "threads",
            "throughput/s",
            "p50_ns",
            "p95_ns",
            "p99_ns",
            "p999_ns",
            "p99_ms",
            "scaling"
        );

        let mut wl_report = serde_json::Map::new();
        let mut p99_at_1: Option<u128> = None;
        let mut throughput_at_1: Option<f64> = None;

        for &n in THREAD_SWEEP {
            let (ops_per_sec, lat, scans_delta) = run_workload(w, n);
            let p50 = percentile(&lat, 0.50);
            let p95 = percentile(&lat, 0.95);
            let p99 = percentile(&lat, 0.99);
            let p999 = percentile(&lat, 0.999);
            let p99_ms = p99 as f64 / 1_000_000.0;

            if n == 1 {
                p99_at_1 = Some(p99);
                throughput_at_1 = Some(ops_per_sec);
            }
            // Throughput scaling: ideal = N (perfect parallel scaling).
            // Actual / ideal = scaling efficiency, < 1 indicates contention.
            let scaling_eff = throughput_at_1
                .map(|t1| ops_per_sec / (t1 * n as f64))
                .unwrap_or(1.0);
            let p99_amp = p99_at_1
                .map(|p1| p99 as f64 / p1 as f64)
                .unwrap_or(1.0);

            eprintln!(
                "{:>6} {:>14.0} {:>11} {:>11} {:>11} {:>11} {:>11.3} {:>5.2}× eff={:.2}",
                n, ops_per_sec, p50, p95, p99, p999, p99_ms, p99_amp, scaling_eff
            );

            wl_report.insert(
                n.to_string(),
                json!({
                    "threads": n,
                    "ops_per_thread": OPS_PER_THREAD,
                    "throughput_ops_per_sec": ops_per_sec,
                    "scaling_efficiency": scaling_eff,
                    "p50_ns": p50,
                    "p95_ns": p95,
                    "p99_ns": p99,
                    "p999_ns": p999,
                    "p99_ms": p99_ms,
                    "p99_amplification_vs_1thread": p99_amp,
                    "dashmap_full_scans_delta": scans_delta,
                }),
            );
        }

        // Headline: 21-thread vs 1-thread amplification + scaling.
        if let (Some(p21), Some(t21)) = (
            wl_report.get("21").and_then(|v| v.get("p99_ns")).and_then(|v| v.as_u64()),
            wl_report
                .get("21")
                .and_then(|v| v.get("throughput_ops_per_sec"))
                .and_then(|v| v.as_f64()),
        ) {
            let p1 = p99_at_1.unwrap_or(1) as f64;
            let t1 = throughput_at_1.unwrap_or(1.0);
            headline.insert(
                w.label().to_string(),
                json!({
                    "threads_at_crash_scenario": 21,
                    "p99_amplification_21t_vs_1t": p21 as f64 / p1,
                    "scaling_efficiency_21t": t21 / (t1 * 21.0),
                    "throughput_21t_ops_per_sec": t21,
                    "p99_ms_21t": p21 as f64 / 1_000_000.0,
                }),
            );
        }

        report.insert(w.label().to_string(), serde_json::Value::Object(wl_report));
    }

    let final_report = json!({
        "plan": "69e59065",
        "task": "T1",
        "step": 4,
        "title": "synapse reinforce lock contention probe",
        "thread_sweep": THREAD_SWEEP,
        "ops_per_thread": OPS_PER_THREAD,
        "workloads": report,
        "headline_21_threads": headline,
    });

    let path = "/tmp/synapse-lock-contention.json";
    std::fs::write(path, serde_json::to_string_pretty(&final_report).unwrap())
        .expect("write");
    eprintln!("\n[contention] report → {}", path);

    // Don't assert. The bench is purely informational — its job is to
    // produce the data needed to confirm or refute the lock contention
    // hypothesis from gotcha 49938809.
}
