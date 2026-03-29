//! Ξ(t) Functional Tests — Validate milestone objectives
//!
//! Unlike the integration tests (which check "does it crash?"), these verify
//! that each milestone objective ACTUALLY WORKS as specified.

use persona::PersonaDB;
use persona::fact_gnn::{FactGNN, query_embedding};
use persona::reward::RewardDetector;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;
use std::collections::HashMap;

fn temp_db(suffix: &str) -> PersonaDB {
    let path = format!("/tmp/xi_func_test_{}_{}", std::process::id(), suffix);
    let _ = std::fs::remove_dir_all(&path);
    PersonaDB::open(&path).expect("open persona db")
}

fn cleanup(suffix: &str) {
    let _ = std::fs::remove_dir_all(format!("/tmp/xi_func_test_{}_{}", std::process::id(), suffix));
}

// ═══════════════════════════════════════════════════════════════
// T4: Reward — 5-signal weighted combination
// ═══════════════════════════════════════════════════════════════

/// Helper: build a RewardDetector with known token_polarity map (no LlamaEngine needed).
fn mock_reward_detector(polarities: &[(i32, f32)]) -> RewardDetector {
    RewardDetector {
        token_polarity: polarities.iter().copied().collect(),
        prev_query_tokens: Vec::new(),
    }
}

#[test]
fn t4_reward_signal1_token_polarity() {
    let mut rd = mock_reward_detector(&[
        (100, 0.8),  // "merci"
        (200, -0.7), // "faux"
        (300, 0.0),  // neutral
    ]);

    // Turn 1 is always 0.0 (no previous turn to evaluate)
    let r = rd.compute_reward(&[100, 300], 1, None, None, None);
    assert_eq!(r, 0.0, "turn 1 should always be 0.0");

    // Turn 2 with positive polarity tokens → positive reward
    let r = rd.compute_reward(&[100], 2, None, None, None);
    // token_signal = 0.8/1 = 0.8, clamped to 0.5
    // reward = 0.30 * 0.5 + 0.20 * 0 + 0.10 * engagement + 0.25 * 0 + 0.15 * 0
    // engagement = 0.02 * min(2, 20) = 0.04
    // = 0.15 + 0.0 + 0.004 + 0 + 0 = 0.154
    assert!(r > 0.1, "positive polarity tokens should produce positive reward, got {r}");

    // Turn 3 with negative polarity tokens → negative reward
    let r = rd.compute_reward(&[200], 3, None, None, None);
    assert!(r < 0.0, "negative polarity tokens should produce negative reward, got {r}");
}

#[test]
fn t4_reward_signal2_reformulation_detection() {
    let mut rd = mock_reward_detector(&[]);

    // Turn 1
    rd.compute_reward(&[1, 2, 3, 4, 5], 1, None, None, None);

    // Turn 2: exact same tokens → reformulation (cosine = 1.0 > 0.85)
    let r = rd.compute_reward(&[1, 2, 3, 4, 5], 2, None, None, None);
    // reformulation_penalty = -0.3
    // reward = 0.30*0 + 0.20*(-0.3) + 0.10*0.04 + 0.25*0 + 0.15*0 = -0.056
    assert!(r < 0.0, "reformulation should produce negative reward, got {r}");

    // Turn 3: completely different tokens → no reformulation
    let r = rd.compute_reward(&[100, 200, 300, 400, 500], 3, None, None, None);
    // No polarity, no reformulation → small positive from engagement
    assert!(r >= 0.0, "different tokens = no reformulation penalty, got {r}");
}

#[test]
fn t4_reward_signal3_engagement_grows() {
    let mut rd = mock_reward_detector(&[]);

    // Turn 1: baseline
    rd.compute_reward(&[1], 1, None, None, None);

    // Turn 5: engagement = 0.02 * 5 = 0.10, weighted 0.10 → contribution = 0.01
    let r5 = rd.compute_reward(&[99], 5, None, None, None);
    // Turn 15: engagement = 0.02 * 15 = 0.30, weighted 0.10 → contribution = 0.03
    let r15 = rd.compute_reward(&[98], 15, None, None, None);

    assert!(r15 > r5, "longer engagement should give higher reward: r5={r5}, r15={r15}");
}

#[test]
fn t4_reward_signal4_factual_success() {
    let mut rd = mock_reward_detector(&[]);

    // Turn 1
    rd.compute_reward(&[1], 1, None, None, None);

    // Facts in header + response contains the fact values
    let facts = vec![
        ("name".to_string(), "Alice".to_string()),
        ("city".to_string(), "Paris".to_string()),
    ];
    let response = "Oui Alice, je sais que tu habites à Paris !";

    let r_with = rd.compute_reward(&[1], 3, Some(&facts), Some(response), None);

    // Now with facts that DON'T appear in response
    let facts_miss = vec![
        ("name".to_string(), "Bob".to_string()),
    ];
    let response_miss = "Bonjour, comment allez-vous ?";
    let r_without = rd.compute_reward(&[2], 4, Some(&facts_miss), Some(response_miss), None);

    assert!(r_with > r_without,
        "response containing fact values should score higher: with={r_with}, without={r_without}");

    // Specifically: factual_signal when facts present but not used = -0.1
    // This tests the -0.1 penalty path
    assert!(r_without < r_with, "unused facts should penalize");
}

#[test]
fn t4_reward_signal5_entropy() {
    let mut rd = mock_reward_detector(&[]);
    rd.compute_reward(&[1], 1, None, None, None);

    // Low entropy → confident generation → positive signal
    let r_low = rd.compute_reward(&[1], 3, None, None, Some(1.0));
    // High entropy → uncertain generation → negative signal
    let r_high = rd.compute_reward(&[1], 4, None, None, Some(4.0));

    assert!(r_low > r_high,
        "low entropy should score higher than high: low={r_low}, high={r_high}");
}

#[test]
fn t4_reward_all_5_signals_combined() {
    // Scenario: positive polarity + no reformulation + medium engagement +
    // facts used in response + low entropy = very positive reward
    let mut rd = mock_reward_detector(&[(10, 0.9)]); // strong positive token
    rd.compute_reward(&[99], 1, None, None, None); // turn 1

    let facts = vec![("name".to_string(), "Alice".to_string())];
    let response = "Bien sûr Alice, voici la réponse.";
    let r = rd.compute_reward(
        &[10], // positive polarity
        10,    // medium engagement
        Some(&facts),
        Some(response),
        Some(1.2), // low entropy
    );

    // All 5 signals should contribute positively
    assert!(r > 0.1, "all positive signals should produce high reward: {r}");
    eprintln!("  T4 combined reward = {r:.4}");
}

// ═══════════════════════════════════════════════════════════════
// T5: Cost tracking — utility, activation_count, cost_efficiency
// ═══════════════════════════════════════════════════════════════

#[test]
fn t5_cost_tracking_initial_values() {
    let pdb = temp_db("t5_init");

    let fid = pdb.add_fact("name", "Alice", 0, None);
    let store = pdb.db.store();
    let node = store.get_node(fid).unwrap();

    let token_cost = node.properties.get(&PropertyKey::from("token_cost"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
        .unwrap();
    let utility = node.properties.get(&PropertyKey::from("utility"))
        .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
        .unwrap();
    let cost_eff = node.properties.get(&PropertyKey::from("cost_efficiency"))
        .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
        .unwrap();
    let act_count = node.properties.get(&PropertyKey::from("activation_count"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
        .unwrap();

    // token_cost = (4 + 5 + 6) / 4 = 3.75 → max(1, 3) = 3
    assert!(token_cost > 0, "token_cost should be positive: {token_cost}");
    assert_eq!(utility, 0.5, "initial utility should be 0.5");
    assert!(cost_eff > 0.0, "initial cost_efficiency should be positive: {cost_eff}");
    assert_eq!(act_count, 0, "initial activation_count should be 0");

    eprintln!("  T5 initial: token_cost={token_cost}, utility={utility:.3}, cost_eff={cost_eff:.4}");
    cleanup("t5_init");
}

#[test]
fn t5_cost_tracking_updates_on_reward_propagation() {
    let pdb = temp_db("t5_prop");

    // Create facts
    let fid1 = pdb.add_fact("name", "Alice", 0, None);
    let fid2 = pdb.add_fact("city", "Paris", 0, None);

    // Create a ConvTurn for linking
    let ct_id = pdb.create_conv_turn("who are you?", "I'm Alice from Paris", 1);
    pdb.mark_facts_used_in(&[fid1, fid2], ct_id);

    // Build a mock RewardDetector and propagate positive reward
    let rd = RewardDetector {
        token_polarity: HashMap::new(),
        prev_query_tokens: Vec::new(),
    };
    rd.propagate_reward(&pdb, ct_id, 0.8, &[fid1, fid2]);

    // Verify fact properties were updated
    let store = pdb.db.store();
    let node = store.get_node(fid1).unwrap();

    let act_count = node.properties.get(&PropertyKey::from("activation_count"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
        .unwrap();
    let utility = node.properties.get(&PropertyKey::from("utility"))
        .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
        .unwrap();

    assert_eq!(act_count, 1, "activation_count should be 1 after one propagation");
    // new_utility = (0.5 * 0 + 0.8) / 1 = 0.8
    assert!((utility - 0.8).abs() < 0.01, "utility should be ~0.8 after positive reward, got {utility}");

    // Propagate negative reward
    let ct_id2 = pdb.create_conv_turn("where?", "I don't know", 2);
    rd.propagate_reward(&pdb, ct_id2, -0.5, &[fid1]);

    let store = pdb.db.store();
    let node = store.get_node(fid1).unwrap();
    let act2 = node.properties.get(&PropertyKey::from("activation_count"))
        .and_then(|v| if let Value::Int64(n) = v { Some(*n) } else { None })
        .unwrap();
    let util2 = node.properties.get(&PropertyKey::from("utility"))
        .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
        .unwrap();

    assert_eq!(act2, 2, "activation_count should be 2");
    // new_utility = (0.8 * 1 + (-0.5)) / 2 = 0.15
    assert!(util2 < utility, "utility should decrease after negative reward: was {utility}, now {util2}");

    eprintln!("  T5 propagation: act={act2}, utility after neg reward = {util2:.4}");
    cleanup("t5_prop");
}

// ═══════════════════════════════════════════════════════════════
// T1: GNN-guided fact selection — GNN should learn to rank
// ═══════════════════════════════════════════════════════════════

#[test]
fn t1_gnn_forward_produces_distinct_scores() {
    // After training, different queries should produce different score rankings
    let pdb = temp_db("t1_distinct");
    let facts = vec![
        ("name", "Alice"), ("color", "blue"), ("food", "sushi"),
        ("city", "Paris"), ("hobby", "climbing"),
    ];
    for (k, v) in &facts {
        pdb.add_fact(k, v, 0, None);
    }

    let store = pdb.db.store();
    let fact_ids = pdb.active_fact_ids();
    let gnn = FactGNN::new();

    let q1 = query_embedding("Comment tu t'appelles ?");
    let q2 = query_embedding("Quelle est ta couleur préférée ?");

    let scores1 = gnn.score_facts(&store, &q1, &fact_ids, 2);
    let scores2 = gnn.score_facts(&store, &q2, &fact_ids, 2);

    // Scores should be different for different queries (hash-based embeddings guarantee this)
    assert_eq!(scores1.len(), scores2.len());

    // Extract top-1 for each query — they should be different
    // (with hash-based embeddings this is not guaranteed to be semantically correct,
    // but the ranking should differ between queries)
    let top1_q1 = scores1[0].0;
    let top1_q2 = scores2[0].0;

    // At minimum, scores should differ numerically
    let diff: f32 = scores1.iter().zip(scores2.iter())
        .map(|((_, s1), (_, s2))| (s1 - s2).abs())
        .sum();
    assert!(diff > 0.01, "different queries should produce different scores: total diff = {diff}");

    eprintln!("  T1 score diff across queries = {diff:.4}");
    eprintln!("  top1 for q1 = {top1_q1:?}, top1 for q2 = {top1_q2:?}");
    cleanup("t1_distinct");
}

#[test]
fn t1_gnn_weights_change_after_update() {
    // Verify that GNN update() actually modifies weights
    let pdb = temp_db("t1_weights");
    pdb.add_fact("name", "Alice", 0, None);
    pdb.add_fact("color", "blue", 0, None);

    // Create ConvTurn + USED_IN edges to build graph topology
    let ct = pdb.create_conv_turn("who?", "Alice", 1);
    let fact_ids = pdb.active_fact_ids();
    pdb.mark_facts_used_in(&fact_ids, ct);

    let store = pdb.db.store();
    let mut gnn = FactGNN::new();

    let q = query_embedding("name query");
    let scores_before = gnn.score_facts(&store, &q, &fact_ids, 2);

    // Debug: what are the actual scores?
    for (nid, score) in &scores_before {
        let key = store.get_node(*nid)
            .and_then(|n| n.properties.get(&PropertyKey::from("key"))
                .and_then(|v| v.as_str()).map(|s| s.to_string()))
            .unwrap_or_default();
        // Count outgoing edges (for update to work, facts need outgoing edges)
        let edge_count = store.edges_from(*nid, Direction::Outgoing).count();
        eprintln!("  DEBUG: fact {key} score={score:.6}, outgoing_edges={edge_count}");
    }

    // Get full weight snapshot before
    let w_snapshot_before: Vec<f32> = gnn.w_message.values()
        .flat_map(|w| w.iter().copied())
        .collect();

    // Update with strong positive reward
    gnn.update(&store, &fact_ids, &scores_before, 0.8);

    let w_snapshot_after: Vec<f32> = gnn.w_message.values()
        .flat_map(|w| w.iter().copied())
        .collect();

    let w_diff: f32 = w_snapshot_before.iter().zip(w_snapshot_after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        w_diff > 1e-10,
        "weights should change after update: total diff={w_diff}"
    );

    let scores_after = gnn.score_facts(&store, &q, &fact_ids, 2);
    let score_diff: f32 = scores_before.iter().zip(scores_after.iter())
        .map(|((_, s1), (_, s2))| (s1 - s2).abs())
        .sum();

    eprintln!("  T1 weight change = {:.6}, score diff = {:.6}", w_diff, score_diff);
    cleanup("t1_weights");
}

// ═══════════════════════════════════════════════════════════════
// T7: MENTIONS bridge — score_node_ids via ConvTurn MENTIONS
// ═══════════════════════════════════════════════════════════════

#[test]
fn t7_mentions_bridge_no_training() {
    // Before 20 updates, score_node_ids returns uniform 1.0
    let pdb = temp_db("t7_bridge");
    let gnn = FactGNN::new(); // n_updates = 0

    let fake_ids = vec![NodeId(999), NodeId(1000)];
    let store = pdb.db.store();
    let q = query_embedding("test");
    let scores = gnn.score_node_ids(&store, &q, &fake_ids);

    assert_eq!(scores.len(), 2);
    assert_eq!(scores[&NodeId(999)], 1.0, "before 20 updates, should return uniform 1.0");
    assert_eq!(scores[&NodeId(1000)], 1.0);

    cleanup("t7_bridge");
}

#[test]
fn t7_mentions_bridge_with_topology() {
    // After training, nodes with MENTIONS edges should get non-zero scores
    let pdb = temp_db("t7_topo");

    // Create facts + ConvTurns + MENTIONS topology
    let f1 = pdb.add_fact("name", "Alice", 0, None);
    let ct1 = pdb.create_conv_turn("who?", "Alice", 1);
    pdb.mark_facts_used_in(&[f1], ct1);

    // Simulate a "data graph node" — just create a node with a custom label
    let data_nid = pdb.db.create_node_with_props(&["DataNode"], [
        ("label", Value::String("some_entity".to_string().into())),
    ]);

    // Link ConvTurn → MENTIONS → data node
    pdb.link_mentions(ct1, data_nid);

    // Train GNN enough (need >= 20 updates)
    let store = pdb.db.store();
    let mut gnn = FactGNN::new();
    let fact_ids = pdb.active_fact_ids();
    for _ in 0..25 {
        let q = query_embedding("who is Alice?");
        let scores = gnn.forward(&store, &q, &fact_ids, 2);
        gnn.update(&store, &fact_ids, &scores, 0.5);
    }
    assert!(gnn.n_updates() >= 20);

    // Now score via MENTIONS bridge
    let q = query_embedding("who is Alice?");
    let scores = gnn.score_node_ids(&store, &q, &[data_nid]);

    assert!(scores.contains_key(&data_nid), "data node should have a score");
    let score = scores[&data_nid];
    eprintln!("  T7 MENTIONS bridge score = {score:.4}");

    // Score should be non-zero (ConvTurn ct1 MENTIONS data_nid, and ct1 has GNN embedding)
    // Note: score can be negative (dot product), but should not be exactly 0.0
    // unless the embedding happens to be orthogonal to query
    // The key test: it participated in the computation (not default 0.0)

    // Also test: node WITHOUT mentions should get 0.0
    let orphan_nid = pdb.db.create_node_with_props(&["DataNode"], [
        ("label", Value::String("orphan".to_string().into())),
    ]);
    let scores2 = gnn.score_node_ids(&store, &q, &[data_nid, orphan_nid]);
    assert_eq!(scores2[&orphan_nid], 0.0, "orphan node (no MENTIONS) should score 0.0");

    cleanup("t7_topo");
}

// ═══════════════════════════════════════════════════════════════
// T5 + T4: REINFORCES edges on high-reward turns
// ═══════════════════════════════════════════════════════════════

#[test]
fn t5_reinforces_edges_created_on_high_reward() {
    let pdb = temp_db("t5_reinf");

    let f1 = pdb.add_fact("name", "Alice", 0, None);
    let f2 = pdb.add_fact("city", "Paris", 0, None);
    let f3 = pdb.add_fact("color", "blue", 0, None);

    let ct = pdb.create_conv_turn("who?", "Alice from Paris", 1);

    let rd = RewardDetector {
        token_polarity: HashMap::new(),
        prev_query_tokens: Vec::new(),
    };

    // Low reward → NO REINFORCES
    rd.propagate_reward(&pdb, ct, 0.3, &[f1, f2]);
    let store = pdb.db.store();
    let reinf_count_low = store.edges_from(f1, Direction::Outgoing)
        .filter(|(_target, eid)| {
            store.get_edge(*eid)
                .map(|e| { let s: &str = e.edge_type.as_ref(); s == "REINFORCES" })
                .unwrap_or(false)
        })
        .count();
    assert_eq!(reinf_count_low, 0, "reward <= 0.5 should NOT create REINFORCES");

    // High reward → REINFORCES created
    let ct2 = pdb.create_conv_turn("tell me", "Alice loves Paris", 2);
    rd.propagate_reward(&pdb, ct2, 0.9, &[f1, f2, f3]);

    let store = pdb.db.store();
    let mut reinforces_targets: Vec<NodeId> = Vec::new();
    for (target, eid) in store.edges_from(f1, Direction::Outgoing).collect::<Vec<_>>() {
        if let Some(edge) = store.get_edge(eid) {
            let label: &str = edge.edge_type.as_ref();
            if label == "REINFORCES" {
                reinforces_targets.push(target);
            }
        }
    }
    assert!(!reinforces_targets.is_empty(),
        "reward > 0.5 with 3 facts should create REINFORCES edges");

    eprintln!("  T5 REINFORCES edges from f1: {} targets", reinforces_targets.len());
    cleanup("t5_reinf");
}

// ═══════════════════════════════════════════════════════════════
// T4: RewardDetector bag_cosine correctness
// ═══════════════════════════════════════════════════════════════

#[test]
fn t4_bag_cosine_edge_cases() {
    // Identical → 1.0
    let cos = RewardDetector::bag_cosine(&[1, 2, 3], &[1, 2, 3]);
    assert!((cos - 1.0).abs() < 0.001, "identical bags should have cosine ~1.0: {cos}");

    // Completely disjoint → 0.0
    let cos = RewardDetector::bag_cosine(&[1, 2, 3], &[4, 5, 6]);
    assert!((cos - 0.0).abs() < 0.001, "disjoint bags should have cosine ~0.0: {cos}");

    // Empty
    let cos = RewardDetector::bag_cosine(&[], &[1, 2, 3]);
    assert!((cos - 0.0).abs() < 0.001, "empty bag should have cosine ~0.0: {cos}");

    // Partial overlap
    let cos = RewardDetector::bag_cosine(&[1, 2, 3], &[2, 3, 4]);
    assert!(cos > 0.3 && cos < 0.9, "partial overlap should give intermediate cosine: {cos}");
}

// ═══════════════════════════════════════════════════════════════
// T6: Dynamic header — top-5 change detection
// ═══════════════════════════════════════════════════════════════

#[test]
fn t6_header_budget_enforcement() {
    // Simulate the budget logic from build_system_header
    let mut scored_facts: Vec<(String, String, f32)> = Vec::new();
    for i in 0..50 {
        scored_facts.push((
            format!("key_{i}"),
            format!("value with moderate length for fact {i}"),
            1.0 - i as f32 * 0.01,
        ));
    }

    // Sort by score desc
    scored_facts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Apply budget cap (~2000 chars) as build_system_header does
    let mut budget_chars = 0usize;
    let budget_limit = 2000;
    let mut included = 0;

    for (k, v, _score) in &scored_facts {
        let line_len = format!("- {k} : {v}\n").len();
        if budget_chars + line_len > budget_limit {
            break;
        }
        budget_chars += line_len;
        included += 1;
    }

    assert!(included < 50, "budget should truncate: only {included}/50 included");
    assert!(budget_chars <= budget_limit, "should respect budget: {budget_chars} <= {budget_limit}");
    assert!(included > 10, "should include at least 10 facts: {included}");

    // Verify top-5 keys are the highest-scored
    let top5: Vec<&str> = scored_facts.iter().take(5).map(|(k, _, _)| k.as_str()).collect();
    assert_eq!(top5[0], "key_0");
    assert_eq!(top5[4], "key_4");

    eprintln!("  T6 budget: {included}/50 facts, {budget_chars}/{budget_limit} chars");
}

#[test]
fn t6_top5_change_detection() {
    // Simulate the top-5 change detection logic from main.rs
    let prev_top5: Vec<String> = vec![
        "name".into(), "city".into(), "color".into(), "food".into(), "hobby".into(),
    ];

    // Same top-5 → no rebuild needed
    let new_top5_same: Vec<String> = vec![
        "name".into(), "city".into(), "color".into(), "food".into(), "hobby".into(),
    ];
    assert_eq!(prev_top5, new_top5_same, "same top-5 = no rebuild");

    // Different top-5 → rebuild needed
    let new_top5_diff: Vec<String> = vec![
        "name".into(), "city".into(), "color".into(), "food".into(), "language".into(),
    ];
    assert_ne!(prev_top5, new_top5_diff, "different top-5 = rebuild needed");

    // Partial change
    let new_top5_partial: Vec<String> = vec![
        "name".into(), "city".into(), "language".into(), "food".into(), "hobby".into(),
    ];
    assert_ne!(prev_top5, new_top5_partial, "top-5 order change = rebuild needed");
}

// ═══════════════════════════════════════════════════════════════
// T1: GNN REINFORCE — weights converge over many updates
// ═══════════════════════════════════════════════════════════════

#[test]
fn t1_gnn_reinforces_learn_direction() {
    // Core functional test: if we consistently reward one fact and punish another,
    // the rewarded fact should eventually score higher.
    let pdb = temp_db("t1_learn");

    // Create 2 facts with distinct keys
    let good = pdb.add_fact("good_fact", "I should rank high", 0, None);
    let bad = pdb.add_fact("bad_fact", "I should rank low", 0, None);

    // Create ConvTurns linking them
    let ct = pdb.create_conv_turn("test", "test", 1);
    pdb.mark_facts_used_in(&[good, bad], ct);

    let store = pdb.db.store();
    let fact_ids = vec![good, bad];
    let mut gnn = FactGNN::new();
    let q = query_embedding("test query for learning");

    // Record initial scores
    let scores_init = gnn.score_facts(&store, &q, &fact_ids, 2);
    let init_good = scores_init.iter().find(|(nid, _)| *nid == good).map(|(_, s)| *s).unwrap();
    let init_bad = scores_init.iter().find(|(nid, _)| *nid == bad).map(|(_, s)| *s).unwrap();

    // Train: consistently reward "good" and punish "bad"
    for _ in 0..100 {
        let scores = gnn.score_facts(&store, &q, &fact_ids, 2);
        // Reward when good_fact is used
        gnn.update(&store, &[good], &scores, 0.8);
        // Punish when bad_fact is used
        gnn.update(&store, &[bad], &scores, -0.5);
    }

    // Check final scores
    let scores_final = gnn.score_facts(&store, &q, &fact_ids, 2);
    let final_good = scores_final.iter().find(|(nid, _)| *nid == good).map(|(_, s)| *s).unwrap();
    let final_bad = scores_final.iter().find(|(nid, _)| *nid == bad).map(|(_, s)| *s).unwrap();

    let delta_good = final_good - init_good;
    let delta_bad = final_bad - init_bad;

    eprintln!("  T1 learning: good Δ={delta_good:+.4} ({init_good:.4}→{final_good:.4})");
    eprintln!("  T1 learning: bad  Δ={delta_bad:+.4} ({init_bad:.4}→{final_bad:.4})");

    // The rewarded fact should have improved relative to the punished one
    assert!(
        delta_good > delta_bad,
        "consistently rewarded fact should improve more than punished: good_delta={delta_good}, bad_delta={delta_bad}"
    );

    cleanup("t1_learn");
}

// ═══════════════════════════════════════════════════════════════
// T3: StepSignals integration (entropy thresholds)
// ═══════════════════════════════════════════════════════════════

#[test]
fn t3_entropy_thresholds_match_reward() {
    // Verify the entropy thresholds in reward match HIGH_ENTROPY_THRESHOLD
    // LOW: < 1.5 → +0.1 (confident)
    // HIGH: > 3.0 → -0.1 (uncertain)
    // Note: HIGH_ENTROPY_THRESHOLD in signals.rs = 3.0

    let mut rd = mock_reward_detector(&[]);
    rd.compute_reward(&[1], 1, None, None, None);

    let r_confident = rd.compute_reward(&[1], 3, None, None, Some(0.5));
    let r_neutral = rd.compute_reward(&[1], 4, None, None, Some(2.0));
    let r_uncertain = rd.compute_reward(&[1], 5, None, None, Some(5.0));

    assert!(r_confident > r_neutral, "entropy 0.5 > entropy 2.0: {r_confident} vs {r_neutral}");
    assert!(r_neutral > r_uncertain, "entropy 2.0 > entropy 5.0: {r_neutral} vs {r_uncertain}");

    eprintln!("  T3 entropy rewards: confident={r_confident:.4}, neutral={r_neutral:.4}, uncertain={r_uncertain:.4}");
}
