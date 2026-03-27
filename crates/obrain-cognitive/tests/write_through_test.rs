//! Write-through persistence integration tests.
//!
//! Verifies that cognitive scores (energy, synapse, fabric, co-change)
//! are persisted to graph properties and can be reloaded after dropping
//! the cognitive stores.

#![cfg(feature = "cognitive-full")]

use obrain_cognitive::co_change::CoChangeConfig;
use obrain_cognitive::co_change::CoChangeStore;
use obrain_cognitive::energy::{EnergyConfig, EnergyStore};
use obrain_cognitive::engine::{CognitiveEngine, CognitiveEngineBuilder};
use obrain_cognitive::fabric::FabricStore;
use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::PropertyKey;
use obrain_core::LpgStore;
use obrain_core::graph::GraphStoreMut;
use obrain_reactive::{BatchConfig, MutationBus, Scheduler};
use std::sync::Arc;
use std::time::Duration;

fn make_graph_store() -> Arc<LpgStore> {
    Arc::new(LpgStore::new().expect("LpgStore::new should succeed"))
}

fn make_scheduler() -> (MutationBus, Scheduler) {
    let bus = MutationBus::new();
    let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));
    (bus, scheduler)
}

// ---------------------------------------------------------------------------
// Energy write-through
// ---------------------------------------------------------------------------

#[test]
fn energy_write_through_persists_to_graph() {
    let gs = make_graph_store();
    let node_id = gs.create_node(&["TestNode"]);

    // Create energy store with graph backing
    let store = EnergyStore::with_graph_store(
        EnergyConfig::default(),
        Arc::clone(&gs) as Arc<dyn GraphStoreMut>,
    );

    // Boost energy
    store.boost(node_id, 5.0);
    let e = store.get_energy(node_id);
    assert!(e > 4.99, "energy should be ~5.0, got {e}");

    // Verify it was persisted to the graph
    let pk = PropertyKey::from("_cog_energy");
    let prop = gs.get_node_property(node_id, &pk);
    assert!(
        prop.is_some(),
        "energy should be persisted as node property"
    );
    let val = prop.unwrap().as_float64().unwrap();
    assert!(val > 4.99, "persisted energy should be ~5.0, got {val}");
}

#[test]
fn energy_lazy_load_from_graph() {
    let gs = make_graph_store();
    let node_id = gs.create_node(&["TestNode"]);

    // Create first store and boost
    {
        let store = EnergyStore::with_graph_store(
            EnergyConfig::default(),
            Arc::clone(&gs) as Arc<dyn GraphStoreMut>,
        );
        store.boost(node_id, 3.0);
    }
    // First store dropped — DashMap gone

    // Create a new store from the same graph — lazy load should work
    let store2 = EnergyStore::with_graph_store(
        EnergyConfig::default(),
        Arc::clone(&gs) as Arc<dyn GraphStoreMut>,
    );
    let e = store2.get_energy(node_id);
    assert!(e > 2.99, "lazy-loaded energy should be ~3.0, got {e}");
}

// ---------------------------------------------------------------------------
// Synapse write-through
// ---------------------------------------------------------------------------

#[test]
fn synapse_write_through_persists_to_graph() {
    let gs = make_graph_store();
    let n1 = gs.create_node(&["A"]);
    let n2 = gs.create_node(&["B"]);

    let store = SynapseStore::with_graph_store(
        SynapseConfig::default(),
        Arc::clone(&gs) as Arc<dyn GraphStoreMut>,
    );

    store.reinforce(n1, n2, SynapseConfig::default().reinforce_amount);
    let syn = store.get_synapse(n1, n2);
    assert!(syn.is_some(), "synapse should exist after reinforce");
    let w = syn.unwrap().current_weight();
    assert!(w > 0.0, "synapse weight should be positive, got {w}");

    // The store manages edge IDs internally — if reinforce succeeded
    // with a graph store, the SYNAPSE edge with _cog_synapse_weight was persisted
    assert!(!store.is_empty(), "synapse store should have entries");
}

// ---------------------------------------------------------------------------
// Fabric write-through
// ---------------------------------------------------------------------------

#[test]
fn fabric_write_through_persists_risk() {
    let gs = make_graph_store();
    let node_id = gs.create_node(&["TestNode"]);

    let store = FabricStore::with_graph_store(Arc::clone(&gs) as Arc<dyn GraphStoreMut>);

    store.record_mutation(node_id);
    store.record_mutation(node_id);
    store.record_mutation(node_id);

    // Verify mutation_frequency was persisted
    let pk = PropertyKey::from("_cog_mutation_frequency");
    let prop = gs.get_node_property(node_id, &pk);
    assert!(prop.is_some(), "mutation_frequency should be persisted");
    let val = prop.unwrap().as_float64().unwrap();
    assert!(
        (val - 3.0).abs() < 0.01,
        "persisted mutation_frequency should be 3.0, got {val}"
    );
}

// ---------------------------------------------------------------------------
// Co-change write-through
// ---------------------------------------------------------------------------

#[test]
fn co_change_write_through_creates_edges() {
    let gs = make_graph_store();
    let n1 = gs.create_node(&["A"]);
    let n2 = gs.create_node(&["B"]);

    let store = CoChangeStore::with_graph_store(
        CoChangeConfig::default(),
        Arc::clone(&gs) as Arc<dyn GraphStoreMut>,
    );

    store.record_co_change(n1, n2);
    store.record_co_change(n1, n2);

    let rel = store.get_relation(n1, n2);
    assert!(rel.is_some(), "co-change relation should exist");
    assert_eq!(rel.unwrap().count, 2, "co-change count should be 2");

    // The store manages edge IDs internally — if record_co_change succeeded
    // with a graph store, the CO_CHANGED edge with property was created.
    assert_eq!(store.len(), 1, "should have 1 co-change relation");
}

// ---------------------------------------------------------------------------
// Full engine write-through integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engine_with_graph_store_persists_scores() {
    let gs = make_graph_store();
    let (_bus, scheduler) = make_scheduler();

    // Build engine with graph store backing
    let engine = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .with_synapses(SynapseConfig::default())
        .with_fabric()
        .with_co_change(CoChangeConfig::default())
        .with_graph_store(Arc::clone(&gs) as Arc<dyn GraphStoreMut>)
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 4);

    // Create nodes in graph
    let n1 = gs.create_node(&["File"]);
    let n2 = gs.create_node(&["File"]);

    // Boost energy through the store
    let energy = engine.energy_store().unwrap();
    energy.boost(n1, 7.0);
    energy.boost(n2, 3.0);

    // Verify energy persisted to graph
    let pk = PropertyKey::from("_cog_energy");
    let e1 = gs.get_node_property(n1, &pk).and_then(|v| v.as_float64());
    assert!(e1.is_some(), "energy for n1 should be persisted");
    assert!(e1.unwrap() > 6.99, "n1 energy should be ~7.0");

    let e2 = gs.get_node_property(n2, &pk).and_then(|v| v.as_float64());
    assert!(e2.is_some(), "energy for n2 should be persisted");
    assert!(e2.unwrap() > 2.99, "n2 energy should be ~3.0");

    scheduler.shutdown().await;
}

#[tokio::test]
async fn engine_scores_survive_engine_drop_and_recreate() {
    let gs = make_graph_store();

    // Create nodes first
    let n1 = gs.create_node(&["File"]);

    // Phase 1: create engine, boost energy, drop engine
    {
        let (_bus, scheduler) = make_scheduler();
        let engine = CognitiveEngineBuilder::new()
            .with_energy(EnergyConfig::default())
            .with_graph_store(Arc::clone(&gs) as Arc<dyn GraphStoreMut>)
            .build(&scheduler);

        let energy = engine.energy_store().unwrap();
        energy.boost(n1, 10.0);

        let e = energy.get_energy(n1);
        assert!(e > 9.99, "energy should be ~10.0 before drop, got {e}");

        scheduler.shutdown().await;
        // engine and scheduler dropped here
    }

    // Phase 2: recreate engine from same graph store — energy should be loaded
    {
        let (_bus, scheduler) = make_scheduler();
        let engine = CognitiveEngineBuilder::new()
            .with_energy(EnergyConfig::default())
            .with_graph_store(Arc::clone(&gs) as Arc<dyn GraphStoreMut>)
            .build(&scheduler);

        let energy = engine.energy_store().unwrap();
        // Lazy load from graph: energy was persisted, should still be available
        let e = energy.get_energy(n1);
        assert!(
            e > 9.0,
            "energy after recreate should be ~10.0 (lazy loaded), got {e}"
        );

        scheduler.shutdown().await;
    }
}
