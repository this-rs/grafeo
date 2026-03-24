//! Integration tests for the CognitiveEngine builder and DefaultCognitiveEngine.

#![cfg(feature = "cognitive-full")]

use grafeo_cognitive::co_change::CoChangeConfig;
use grafeo_cognitive::config::CognitiveConfig;
use grafeo_cognitive::energy::EnergyConfig;
use grafeo_cognitive::engine::{CognitiveEngine, CognitiveEngineBuilder};
use grafeo_cognitive::synapse::SynapseConfig;
use grafeo_reactive::{BatchConfig, MutationBus, Scheduler};
use std::time::Duration;

fn make_scheduler() -> (MutationBus, Scheduler) {
    let bus = MutationBus::new();
    let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));
    (bus, scheduler)
}

// ---------------------------------------------------------------------------
// Builder::new() — empty
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_new_empty() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new().build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 0);
    assert!(engine.energy_store().is_none());
    assert!(engine.synapse_store().is_none());
    assert!(engine.fabric_store().is_none());
    assert!(engine.co_change_store().is_none());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder with all subsystems
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_with_all_subsystems() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .with_synapses(SynapseConfig::default())
        .with_fabric()
        .with_co_change(CoChangeConfig::default())
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 4);
    assert!(engine.energy_store().is_some());
    assert!(engine.synapse_store().is_some());
    assert!(engine.fabric_store().is_some());
    assert!(engine.co_change_store().is_some());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder::from_config()
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_from_config_full() {
    let (_bus, scheduler) = make_scheduler();
    let config = CognitiveConfig::full();
    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 4);
    assert!(engine.energy_store().is_some());
    assert!(engine.synapse_store().is_some());
    assert!(engine.fabric_store().is_some());
    assert!(engine.co_change_store().is_some());

    scheduler.shutdown().await;
}

#[tokio::test]
async fn builder_from_config_minimal() {
    let (_bus, scheduler) = make_scheduler();
    let config = CognitiveConfig::new(); // energy + synapse enabled by default
    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 2);
    assert!(engine.energy_store().is_some());
    assert!(engine.synapse_store().is_some());
    assert!(engine.fabric_store().is_none());
    assert!(engine.co_change_store().is_none());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Debug formatting
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_debug_formatting() {
    let builder = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .with_fabric();

    let dbg = format!("{:?}", builder);
    assert!(dbg.contains("CognitiveEngineBuilder"), "got: {dbg}");
    assert!(dbg.contains("energy"), "got: {dbg}");
    assert!(dbg.contains("fabric"), "got: {dbg}");
}

#[tokio::test]
async fn engine_debug_formatting() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .build(&scheduler);

    let dbg = format!("{:?}", engine);
    assert!(dbg.contains("DefaultCognitiveEngine"), "got: {dbg}");
    assert!(dbg.contains("active_subsystems"), "got: {dbg}");

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Default impl for CognitiveEngineBuilder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_default_is_same_as_new() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::default().build(&scheduler);
    assert_eq!(engine.active_subsystem_count(), 0);
    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Accessor methods on DefaultCognitiveEngine return correct stores
// ---------------------------------------------------------------------------

#[tokio::test]
async fn accessor_stores_are_functional() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .with_synapses(SynapseConfig::default())
        .with_fabric()
        .with_co_change(CoChangeConfig::default())
        .build(&scheduler);

    // Energy store is functional
    let energy = engine.energy_store().unwrap();
    assert!(energy.is_empty());
    energy.boost(grafeo_common::types::NodeId::new(1), 1.0);
    assert_eq!(energy.len(), 1);

    // Synapse store is functional
    let synapse = engine.synapse_store().unwrap();
    assert!(synapse.is_empty());

    // Fabric store is functional
    let fabric = engine.fabric_store().unwrap();
    assert_eq!(fabric.len(), 0);

    // Co-change store is functional
    let co_change = engine.co_change_store().unwrap();
    assert!(co_change.is_empty());

    scheduler.shutdown().await;
}
