//! Integration test: multi-agent cognitive personas with shared learning.
//!
//! Demonstrates:
//! 1. Two personas ("Traffic Law", "Labor Law") on the SAME synapse/energy graph
//! 2. Agent A queries through "Traffic Law" → rewarded → synapses reinforced
//! 3. Agent B queries through "Traffic Law" → benefits from A's learning
//! 4. Agent C queries through "Labor Law" → different activation pattern
//! 5. Auto-routing: the system picks the best persona for a set of cues

#![allow(unexpected_cfgs)]
#![cfg(all(feature = "synapse", feature = "energy", feature = "persona"))]

use obrain_cognitive::energy::{EnergyConfig, EnergyStore};
use obrain_cognitive::persona::{PersonaRecallEngine, PersonaStore};
use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use std::sync::Arc;

// ── Helpers ──────────────────────────────────────────────────────────────

/// Build a legal knowledge graph with synapses between related concepts.
fn build_legal_graph(synapse_store: &SynapseStore, energy_store: &EnergyStore) {
    // Traffic cluster
    let vitesse = NodeId(1);
    let exces = NodeId(2);
    let amende = NodeId(3);
    let permis = NodeId(4);
    let points = NodeId(5);
    let radar = NodeId(6);
    let retrait = NodeId(7);

    // Labor cluster
    let contrat = NodeId(10);
    let licenciement = NodeId(11);
    let prudhommes = NodeId(12);
    let salaire = NodeId(13);
    let cdi = NodeId(14);

    // Admin cluster (bridge between both)
    let administration = NodeId(20);
    let fonctionnaire = NodeId(21);

    // ── Traffic synapses ──
    synapse_store.reinforce(exces, vitesse, 0.9);
    synapse_store.reinforce(vitesse, amende, 0.8);
    synapse_store.reinforce(amende, retrait, 0.7);
    synapse_store.reinforce(retrait, points, 0.85);
    synapse_store.reinforce(points, permis, 0.9);
    synapse_store.reinforce(radar, vitesse, 0.75);
    synapse_store.reinforce(radar, amende, 0.6);

    // ── Labor synapses ──
    synapse_store.reinforce(contrat, salaire, 0.8);
    synapse_store.reinforce(contrat, cdi, 0.7);
    synapse_store.reinforce(licenciement, prudhommes, 0.85);
    synapse_store.reinforce(licenciement, contrat, 0.75);
    synapse_store.reinforce(salaire, licenciement, 0.3);

    // ── Bridge (weak link) ──
    synapse_store.reinforce(administration, fonctionnaire, 0.8);
    synapse_store.reinforce(fonctionnaire, contrat, 0.15); // weak cross-link
    synapse_store.reinforce(administration, amende, 0.1); // weak cross-link

    // ── Initial energy for all nodes ──
    for nid in [
        vitesse,
        exces,
        amende,
        permis,
        points,
        radar,
        retrait,
        contrat,
        licenciement,
        prudhommes,
        salaire,
        cdi,
        administration,
        fonctionnaire,
    ] {
        energy_store.boost(nid, 1.0);
    }
}

fn node_name(id: NodeId) -> &'static str {
    match id.0 {
        1 => "vitesse",
        2 => "excès",
        3 => "amende",
        4 => "permis",
        5 => "points",
        6 => "radar",
        7 => "retrait",
        10 => "contrat",
        11 => "licenciement",
        12 => "prud'hommes",
        13 => "salaire",
        14 => "CDI",
        20 => "administration",
        21 => "fonctionnaire",
        _ => "?",
    }
}

fn print_activation(activation: &[(NodeId, f64)], label: &str) {
    println!("\n  📡 Activation [{label}]:");
    for (i, (nid, energy)) in activation.iter().enumerate().take(10) {
        println!(
            "    [{i}] {:<18} energy={:.4}  (id={})",
            node_name(*nid),
            energy,
            nid.0
        );
    }
    if activation.len() > 10 {
        println!("    ... and {} more", activation.len() - 10);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_agent_shared_learning() {
    println!("\n{}", "═".repeat(60));
    println!("  TEST: Multi-agent shared learning through personas");
    println!("{}\n", "═".repeat(60));

    // ── Setup shared cognitive stores ──
    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    build_legal_graph(&synapse_store, &energy_store);

    // ── Create persona store ──
    let persona_store = Arc::new(PersonaStore::new());

    let traffic_id = persona_store.create("Traffic Law", "Expert en droit routier");
    let labor_id = persona_store.create("Labor Law", "Expert en droit du travail");

    // Seed personas with KNOWS
    {
        let mut traffic = persona_store.get_mut(traffic_id).unwrap();
        traffic.add_knows_batch(&[
            (NodeId(1), 0.9),  // vitesse
            (NodeId(2), 0.85), // excès
            (NodeId(3), 0.8),  // amende
            (NodeId(4), 0.7),  // permis
            (NodeId(5), 0.7),  // points
            (NodeId(6), 0.6),  // radar
            (NodeId(7), 0.75), // retrait
        ]);
    }
    {
        let mut labor = persona_store.get_mut(labor_id).unwrap();
        labor.add_knows_batch(&[
            (NodeId(10), 0.9),  // contrat
            (NodeId(11), 0.85), // licenciement
            (NodeId(12), 0.8),  // prud'hommes
            (NodeId(13), 0.7),  // salaire
            (NodeId(14), 0.6),  // CDI
        ]);
    }

    // ── Create the shared recall engine ──
    let engine = PersonaRecallEngine::new(
        Arc::clone(&persona_store),
        Arc::clone(&synapse_store),
        Arc::clone(&energy_store),
    );

    // ═══════════════════════════════════════════════════════════════════
    // Agent A: "excès de vitesse" through Traffic persona
    // ═══════════════════════════════════════════════════════════════════
    println!("  ── Agent A: 'excès de vitesse' via Traffic Law ──");

    let cues_traffic = vec![NodeId(2), NodeId(1)]; // excès, vitesse
    let result_a = engine.recall(&cues_traffic, Some(traffic_id));

    print_activation(&result_a.top_nodes, "Agent A (Traffic)");

    // Traffic cluster should dominate
    let traffic_energy: f64 = result_a
        .top_nodes
        .iter()
        .filter(|(n, _)| n.0 <= 7)
        .map(|(_, e)| e)
        .sum();
    let labor_energy: f64 = result_a
        .top_nodes
        .iter()
        .filter(|(n, _)| (10..=14).contains(&n.0))
        .map(|(_, e)| e)
        .sum();

    println!(
        "\n  Traffic energy: {:.4} vs Labor energy: {:.4}",
        traffic_energy, labor_energy
    );
    assert!(
        traffic_energy > labor_energy,
        "Traffic should dominate for traffic cues"
    );

    // Agent A gives positive feedback
    let feedback_a = engine.feedback(result_a.trail_id, true).unwrap();
    println!(
        "\n  ✅ Agent A feedback: synapses_reinforced={}, nodes_boosted={}, knows_discovered={}",
        feedback_a.synapses_reinforced, feedback_a.nodes_boosted, feedback_a.knows_discovered
    );

    // ═══════════════════════════════════════════════════════════════════
    // Agent B: SAME query, SAME persona — benefits from A's learning
    // ═══════════════════════════════════════════════════════════════════
    println!("\n  ── Agent B: same query, same persona (after A's reward) ──");

    let result_b = engine.recall(&cues_traffic, Some(traffic_id));
    print_activation(&result_b.top_nodes, "Agent B (Traffic, post-learning)");

    let traffic_energy_b: f64 = result_b
        .top_nodes
        .iter()
        .filter(|(n, _)| n.0 <= 7)
        .map(|(_, e)| e)
        .sum();

    println!(
        "\n  Agent A traffic energy: {:.4} → Agent B traffic energy: {:.4} (should be ≥)",
        traffic_energy, traffic_energy_b
    );
    // B should benefit from A's synapse reinforcement
    assert!(
        traffic_energy_b >= traffic_energy * 0.95,
        "B should benefit from A's learning"
    );

    engine.feedback(result_b.trail_id, true).unwrap();

    // ═══════════════════════════════════════════════════════════════════
    // Agent C: "licenciement" through Labor persona — different world
    // ═══════════════════════════════════════════════════════════════════
    println!("\n  ── Agent C: 'licenciement' via Labor Law ──");

    let cues_labor = vec![NodeId(11)]; // licenciement
    let result_c = engine.recall(&cues_labor, Some(labor_id));
    print_activation(&result_c.top_nodes, "Agent C (Labor)");

    let labor_energy_c: f64 = result_c
        .top_nodes
        .iter()
        .filter(|(n, _)| (10..=14).contains(&n.0))
        .map(|(_, e)| e)
        .sum();
    let traffic_energy_c: f64 = result_c
        .top_nodes
        .iter()
        .filter(|(n, _)| n.0 <= 7)
        .map(|(_, e)| e)
        .sum();

    println!(
        "\n  Labor energy: {:.4} vs Traffic energy: {:.4}",
        labor_energy_c, traffic_energy_c
    );
    assert!(
        labor_energy_c > traffic_energy_c,
        "Labor should dominate for labor cues through labor persona"
    );

    engine.feedback(result_c.trail_id, true).unwrap();

    // ═══════════════════════════════════════════════════════════════════
    // Auto-routing: which persona for "amende"?
    // ═══════════════════════════════════════════════════════════════════
    println!("\n  ── Auto-routing: best persona for 'amende' ──");

    let best = persona_store.best_for_cues(&[NodeId(3)]); // amende
    println!("  Best persona for 'amende': {:?}", best);
    assert_eq!(
        best,
        Some(traffic_id),
        "Traffic persona should match amende"
    );

    let best_labor = persona_store.best_for_cues(&[NodeId(12)]); // prud'hommes
    println!("  Best persona for 'prud'hommes': {:?}", best_labor);
    assert_eq!(best_labor, Some(labor_id));

    // ═══════════════════════════════════════════════════════════════════
    // Stats
    // ═══════════════════════════════════════════════════════════════════
    println!("\n  ── Final Persona Stats ──");
    for stat in persona_store.list() {
        println!(
            "  {} | knows={} avg_w={:.2} | queries={} rewards={} rate={:.0}%",
            stat.name,
            stat.knows_count,
            stat.avg_knows_weight,
            stat.query_count,
            stat.reward_count,
            stat.reward_rate * 100.0,
        );
    }

    println!("\n  ✅ Multi-agent shared learning: SUCCESS\n");
}

#[test]
fn persona_penalty_forgets_bad_regions() {
    println!("\n{}", "═".repeat(60));
    println!("  TEST: Penalty causes persona to forget bad regions");
    println!("{}\n", "═".repeat(60));

    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));

    // Simple graph: A → B → C
    synapse_store.reinforce(NodeId(1), NodeId(2), 0.8);
    synapse_store.reinforce(NodeId(2), NodeId(3), 0.7);
    energy_store.boost(NodeId(1), 1.0);
    energy_store.boost(NodeId(2), 1.0);
    energy_store.boost(NodeId(3), 1.0);

    let persona_store = Arc::new(PersonaStore::new());
    let pid = persona_store.create("test", "");
    {
        let mut p = persona_store.get_mut(pid).unwrap();
        p.add_knows(NodeId(1), 0.5);
        p.add_knows(NodeId(2), 0.08); // just above prune threshold
        p.add_knows(NodeId(3), 0.03); // barely above prune threshold
    }

    let engine = PersonaRecallEngine::new(
        Arc::clone(&persona_store),
        Arc::clone(&synapse_store),
        Arc::clone(&energy_store),
    );

    let result = engine.recall(&[NodeId(1)], Some(pid));
    let fb = engine.feedback(result.trail_id, false).unwrap();

    println!(
        "  Penalty feedback: weakened={}, pruned={}",
        fb.knows_weakened, fb.knows_pruned
    );

    let p = persona_store.get(pid).unwrap();
    println!("  KNOWS after penalty:");
    for (&nid, &w) in p.knows() {
        println!("    {} → {:.4}", node_name(nid), w);
    }

    // Node 3 (0.03 - 0.05 = -0.02) should be pruned
    assert_eq!(p.knows_weight(NodeId(3)), 0.0, "Node 3 should be pruned");
    // Node 1 should remain (0.5 - 0.05 = 0.45)
    assert!(
        p.knows_weight(NodeId(1)) > 0.4,
        "Node 1 should survive penalty"
    );

    println!("\n  ✅ Penalty forgets bad regions: SUCCESS\n");
}

#[test]
fn raw_recall_without_persona() {
    println!("\n{}", "═".repeat(60));
    println!("  TEST: Raw recall without persona (no bias)");
    println!("{}\n", "═".repeat(60));

    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));

    synapse_store.reinforce(NodeId(1), NodeId(2), 0.8);
    synapse_store.reinforce(NodeId(2), NodeId(3), 0.6);

    let persona_store = Arc::new(PersonaStore::new());
    let engine = PersonaRecallEngine::new(
        Arc::clone(&persona_store),
        Arc::clone(&synapse_store),
        Arc::clone(&energy_store),
    );

    // Recall without persona — all cues get energy 1.0, no bias
    let result = engine.recall(&[NodeId(1)], None);

    println!("  Raw recall from NodeId(1):");
    for (nid, energy) in &result.top_nodes {
        println!("    {} → {:.4}", nid.0, energy);
    }

    assert!(
        !result.activation.is_empty(),
        "Should activate at least 1 node"
    );
    assert!(result.persona_id.is_none(), "Should have no persona");

    // Feedback without persona — still reinforces shared synapses
    let fb = engine.feedback(result.trail_id, true).unwrap();
    assert!(fb.synapses_reinforced > 0);
    assert!(fb.knows_discovered == 0, "No persona = no KNOWS changes");

    println!("\n  ✅ Raw recall without persona: SUCCESS\n");
}

#[test]
fn enrichment_from_any_agent() {
    println!("\n{}", "═".repeat(60));
    println!("  TEST: Any agent can enrich the shared graph");
    println!("{}\n", "═".repeat(60));

    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));

    // Start with sparse graph
    synapse_store.reinforce(NodeId(1), NodeId(2), 0.3);
    energy_store.boost(NodeId(1), 1.0);
    energy_store.boost(NodeId(2), 1.0);
    energy_store.boost(NodeId(3), 1.0);

    let persona_store = Arc::new(PersonaStore::new());
    let p1 = persona_store.create("Agent1", "");
    let p2 = persona_store.create("Agent2", "");

    let engine = PersonaRecallEngine::new(
        Arc::clone(&persona_store),
        Arc::clone(&synapse_store),
        Arc::clone(&energy_store),
    );

    // Agent 1: queries and gets rewarded → synapse 1↔2 reinforced
    let r1 = engine.recall(&[NodeId(1), NodeId(2)], Some(p1));
    engine.feedback(r1.trail_id, true).unwrap();

    // Get the synapse weight after Agent 1's reward
    let weight_after_a1 = synapse_store
        .get_synapse(NodeId(1), NodeId(2))
        .map_or(0.0, |s| s.current_weight());
    println!("  Synapse 1↔2 after Agent 1 reward: {:.4}", weight_after_a1);

    // Agent 2: queries same nodes → gets rewarded too → synapse further reinforced
    let r2 = engine.recall(&[NodeId(1), NodeId(2)], Some(p2));
    engine.feedback(r2.trail_id, true).unwrap();

    let weight_after_a2 = synapse_store
        .get_synapse(NodeId(1), NodeId(2))
        .map_or(0.0, |s| s.current_weight());
    println!("  Synapse 1↔2 after Agent 2 reward: {:.4}", weight_after_a2);

    assert!(
        weight_after_a2 > weight_after_a1,
        "Second agent's reward should further strengthen the synapse"
    );

    // Both personas should have learned about these nodes
    let p1_knows = persona_store.get(p1).unwrap().knows_count();
    let p2_knows = persona_store.get(p2).unwrap().knows_count();
    println!("  Agent 1 KNOWS: {} nodes", p1_knows);
    println!("  Agent 2 KNOWS: {} nodes", p2_knows);

    assert!(p1_knows > 0, "Agent 1 should have discovered nodes");
    assert!(p2_knows > 0, "Agent 2 should have discovered nodes");

    println!("\n  ✅ Enrichment from any agent: SUCCESS\n");
}
