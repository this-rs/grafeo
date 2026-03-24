//! Integration tests for the provenance system.
//!
//! Tests verify:
//! 1. Boost energy 3x → 3 CognitiveEvent nodes with HAS_COGNITIVE_EVENT edges
//! 2. Decay sweep creates a batch event
//! 3. Query history returns events in chronological order
//! 4. DERIVED_FROM edges link consolidated nodes to sources

use grafeo_cognitive::provenance::{
    CognitiveEvent, CognitiveEventType, ProvenanceRecorder, EDGE_DERIVED_FROM,
    EDGE_HAS_COGNITIVE_EVENT,
};
use grafeo_common::types::NodeId;

fn node(id: u64) -> NodeId {
    NodeId(id)
}

// ---------------------------------------------------------------------------
// Test 1: boost energy 3x → verify 3 CognitiveEvent nodes with HAS_COGNITIVE_EVENT edges
// ---------------------------------------------------------------------------

#[test]
fn provenance_auto_boost_energy_3x_creates_3_events() {
    let recorder = ProvenanceRecorder::new();
    let n1 = node(42);

    // Simulate 3 energy boosts (as the engine would do automatically)
    recorder.record_energy_boost(n1, None, 1.0, "mutation_listener");
    recorder.record_energy_boost(n1, Some(1.0), 2.0, "mutation_listener");
    recorder.record_energy_boost(n1, Some(2.0), 3.0, "mutation_listener");

    // Verify 3 CognitiveEvent nodes exist
    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 3, "Expected 3 events for 3 boosts");

    // All should be EnergyBoost type
    for event in &history {
        assert_eq!(event.event_type, CognitiveEventType::EnergyBoost);
        assert_eq!(event.target_node, n1);
    }

    // Verify HAS_COGNITIVE_EVENT edges
    let edges = recorder.get_edges(n1);
    assert_eq!(
        edges.len(),
        3,
        "Expected 3 HAS_COGNITIVE_EVENT edges for 3 boosts"
    );
    for (src, _event_id) in &edges {
        assert_eq!(*src, n1, "Edge source should be the target node");
    }

    // Verify edge type constant
    assert_eq!(EDGE_HAS_COGNITIVE_EVENT, "HAS_COGNITIVE_EVENT");

    // Verify values track correctly
    assert_eq!(history[0].old_value, None); // First boost has no old value
    assert_eq!(history[0].new_value, Some(1.0));
    assert_eq!(history[1].old_value, Some(1.0));
    assert_eq!(history[1].new_value, Some(2.0));
    assert_eq!(history[2].old_value, Some(2.0));
    assert_eq!(history[2].new_value, Some(3.0));
}

// ---------------------------------------------------------------------------
// Test 2: decay sweep creates batch event
// ---------------------------------------------------------------------------

#[test]
fn provenance_auto_decay_sweep_creates_batch_event() {
    let recorder = ProvenanceRecorder::new();

    // Simulate a decay sweep affecting multiple nodes
    let nodes = vec![node(1), node(2), node(3), node(4)];
    for n in &nodes {
        recorder.record_decay_sweep(*n, 5.0, 2.5);
    }

    // Verify each node got a DecaySweep event
    for n in &nodes {
        let history = recorder.get_history(*n);
        assert_eq!(
            history.len(),
            1,
            "Each swept node should have 1 decay event"
        );
        assert_eq!(history[0].event_type, CognitiveEventType::DecaySweep);
        assert_eq!(history[0].old_value, Some(5.0));
        assert_eq!(history[0].new_value, Some(2.5));
        assert_eq!(history[0].trigger_source, "decay_sweep");
    }

    // Total events across all nodes
    assert_eq!(recorder.total_events(), 4);
    assert_eq!(recorder.nodes_with_events(), 4);
}

// ---------------------------------------------------------------------------
// Test 3: query history returns events in chronological order
// ---------------------------------------------------------------------------

#[test]
fn provenance_auto_history_chronological_order() {
    let recorder = ProvenanceRecorder::new();
    let n1 = node(100);

    // Create events with explicit timestamps to verify ordering
    // Insert out of order to test sorting
    let e_risk = CognitiveEvent::new_at(
        CognitiveEventType::RiskRecalc,
        n1,
        Some(0.1),
        Some(0.8),
        "risk_recalc",
        300, // t=300ms
    );
    let e_boost = CognitiveEvent::new_at(
        CognitiveEventType::EnergyBoost,
        n1,
        None,
        Some(1.0),
        "mutation_listener",
        100, // t=100ms (earliest)
    );
    let e_scar = CognitiveEvent::new_at(
        CognitiveEventType::ScarCreation,
        n1,
        None,
        Some(0.5),
        "scar_creation",
        200, // t=200ms
    );
    let e_prune = CognitiveEvent::new_at(
        CognitiveEventType::SynapsePrune,
        n1,
        Some(0.3),
        None,
        "synapse_prune",
        400, // t=400ms (latest)
    );

    // Record out of chronological order
    recorder.record(e_risk);
    recorder.record(e_boost);
    recorder.record(e_scar);
    recorder.record(e_prune);

    // get_history should return sorted by timestamp
    let history = recorder.get_history(n1);
    assert_eq!(history.len(), 4);

    assert_eq!(history[0].timestamp, 100);
    assert_eq!(history[0].event_type, CognitiveEventType::EnergyBoost);

    assert_eq!(history[1].timestamp, 200);
    assert_eq!(history[1].event_type, CognitiveEventType::ScarCreation);

    assert_eq!(history[2].timestamp, 300);
    assert_eq!(history[2].event_type, CognitiveEventType::RiskRecalc);

    assert_eq!(history[3].timestamp, 400);
    assert_eq!(history[3].event_type, CognitiveEventType::SynapsePrune);

    // Verify filtered queries also return chronological order
    let boosts = recorder.get_history_by_type(n1, &CognitiveEventType::EnergyBoost);
    assert_eq!(boosts.len(), 1);
    assert_eq!(boosts[0].timestamp, 100);
}

// ---------------------------------------------------------------------------
// Test 4: DERIVED_FROM edges link consolidated nodes to sources
// ---------------------------------------------------------------------------

#[test]
fn provenance_auto_derived_from_edges_consolidation() {
    let recorder = ProvenanceRecorder::new();

    // Simulate a consolidation: 3 source nodes are merged into 1 target
    let source_a = node(10);
    let source_b = node(20);
    let source_c = node(30);
    let consolidated = node(100);
    let sources = vec![source_a, source_b, source_c];

    recorder.record_derivation(consolidated, sources.clone(), "knowledge_consolidation");

    // Verify DERIVED_FROM edges
    let derived_edges = recorder.get_derived_from_edges(consolidated);
    assert_eq!(
        derived_edges.len(),
        3,
        "Should have 3 DERIVED_FROM edges (one per source)"
    );

    // Each edge should go from consolidated -> source
    let source_set: std::collections::HashSet<NodeId> =
        derived_edges.iter().map(|(_, src)| *src).collect();
    assert!(source_set.contains(&source_a));
    assert!(source_set.contains(&source_b));
    assert!(source_set.contains(&source_c));

    for (target, _) in &derived_edges {
        assert_eq!(*target, consolidated);
    }

    // Verify derivation records
    let derivations = recorder.get_derivations(consolidated);
    assert_eq!(derivations.len(), 1);
    assert_eq!(derivations[0].sources, sources);
    assert_eq!(derivations[0].operation, "knowledge_consolidation");

    // Verify edge type constant
    assert_eq!(EDGE_DERIVED_FROM, "DERIVED_FROM");

    // Multiple consolidations should accumulate
    let sources2 = vec![node(40), node(50)];
    recorder.record_derivation(consolidated, sources2.clone(), "second_consolidation");

    let derivations = recorder.get_derivations(consolidated);
    assert_eq!(derivations.len(), 2);

    let all_edges = recorder.get_derived_from_edges(consolidated);
    assert_eq!(all_edges.len(), 5, "3 + 2 = 5 DERIVED_FROM edges total");
}
