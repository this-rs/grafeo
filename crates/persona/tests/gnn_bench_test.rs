//! Ξ(t) T7 — GNN Performance Benchmarks (as tests with timing assertions)
//!
//! Validates latency targets without requiring criterion.
//! Each test measures wallclock time and asserts < target.

use persona::PersonaDB;
use persona::fact_gnn::{FactGNN, query_embedding};
use std::time::Instant;

fn temp_persona_db(suffix: &str) -> PersonaDB {
    let path = format!("/tmp/gnn_bench_{}_{}", std::process::id(), suffix);
    let _ = std::fs::remove_dir_all(&path);
    PersonaDB::open(&path).expect("Failed to create temp PersonaDB")
}

fn seed_n_facts(pdb: &PersonaDB, n: usize) {
    for i in 0..n {
        pdb.add_fact(
            &format!("fact_key_{i}"),
            &format!("This is the value for fact number {i} which is quite descriptive"),
            0,
            None,
        );
    }
}

#[test]
fn bench_gnn_forward_50_facts() {
    let pdb = temp_persona_db("forward");
    seed_n_facts(&pdb, 50);
    let gnn = FactGNN::new();
    let store = pdb.db.store();
    let fact_ids = pdb.active_fact_ids();
    let embed = query_embedding("benchmark query");

    // Warmup
    for _ in 0..3 {
        let _ = gnn.score_facts(&store, &embed, &fact_ids, 2);
    }

    // Measure
    let n_iters = 100;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = gnn.score_facts(&store, &embed, &fact_ids, 2);
    }
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_call_ms = per_call_us / 1000.0;

    eprintln!("  gnn_forward (50 facts): {:.2}ms/call ({} iters)", per_call_ms, n_iters);
    // In debug mode, allow 10ms (release target is < 1ms)
    assert!(per_call_ms < 10.0, "GNN forward should be < 10ms (debug), got {:.2}ms", per_call_ms);

    let _ = std::fs::remove_dir_all(format!("/tmp/gnn_bench_{}_forward", std::process::id()));
}

// Note: entropy computation bench is in llm-engine's own tests (signals.rs)
// Can't test from persona crate without adding llm-engine as dev-dependency

#[test]
fn bench_reward_propagation() {
    use persona::RewardDetector;

    let pdb = temp_persona_db("reward");
    seed_n_facts(&pdb, 10);
    let fact_ids = pdb.active_fact_ids();

    // Create a conv turn to propagate reward to
    let ct_id = pdb.create_conv_turn("test query", "test response", 0);
    pdb.mark_facts_used_in(&fact_ids, ct_id);

    // Build a mock RewardDetector (we need a tokenizer mock)
    // Since we can't mock the tokenizer here, we test propagate_reward directly
    let _store = pdb.db.store();
    let rd = RewardDetector {
        token_polarity: std::collections::HashMap::new(),
        prev_query_tokens: Vec::new(),
    };

    // Warmup
    for _ in 0..3 {
        rd.propagate_reward(&pdb, ct_id, 0.5, &fact_ids);
    }

    let n_iters = 100;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        rd.propagate_reward(&pdb, ct_id, 0.5, &fact_ids);
    }
    let elapsed = t0.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_call_ms = per_call_us / 1000.0;

    eprintln!("  reward_propagation (10 facts): {:.2}ms/call ({} iters)", per_call_ms, n_iters);
    assert!(per_call_ms < 5.0, "Reward propagation should be < 5ms, got {:.2}ms", per_call_ms);

    let _ = std::fs::remove_dir_all(format!("/tmp/gnn_bench_{}_reward", std::process::id()));
}
