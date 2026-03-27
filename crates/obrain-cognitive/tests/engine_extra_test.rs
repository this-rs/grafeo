//! Additional coverage tests for the CognitiveEngine builder and DefaultCognitiveEngine.

#![cfg(feature = "cognitive-full")]

use obrain_cognitive::co_change::CoChangeConfig;
use obrain_cognitive::config::CognitiveConfig;
use obrain_cognitive::energy::EnergyConfig;
use obrain_cognitive::engine::{CognitiveEngine, CognitiveEngineBuilder};
use obrain_cognitive::synapse::SynapseConfig;
use obrain_reactive::{BatchConfig, MutationBus, Scheduler};
use std::time::Duration;

fn make_scheduler() -> (MutationBus, Scheduler) {
    let bus = MutationBus::new();
    let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));
    (bus, scheduler)
}

// ---------------------------------------------------------------------------
// Builder with only energy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_energy_only() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 1);
    assert!(engine.energy_store().is_some());
    assert!(engine.synapse_store().is_none());
    assert!(engine.fabric_store().is_none());
    assert!(engine.co_change_store().is_none());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder with only synapse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_synapse_only() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_synapses(SynapseConfig::default())
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 1);
    assert!(engine.energy_store().is_none());
    assert!(engine.synapse_store().is_some());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder with only fabric
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_fabric_only() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_fabric()
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 1);
    assert!(engine.fabric_store().is_some());
    assert!(engine.energy_store().is_none());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder with only co-change
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_co_change_only() {
    let (_bus, scheduler) = make_scheduler();
    let engine = CognitiveEngineBuilder::new()
        .with_co_change(CoChangeConfig::default())
        .build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 1);
    assert!(engine.co_change_store().is_some());
    assert!(engine.energy_store().is_none());

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder::from_config with all disabled
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_from_config_all_disabled() {
    let (_bus, scheduler) = make_scheduler();
    let mut config = CognitiveConfig::default();
    config.energy.enabled = false;
    config.synapse.enabled = false;
    config.fabric.enabled = false;
    config.co_change.enabled = false;

    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);
    assert_eq!(engine.active_subsystem_count(), 0);

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Builder::from_config with custom params
// ---------------------------------------------------------------------------

#[tokio::test]
async fn builder_from_config_custom_energy() {
    let (_bus, scheduler) = make_scheduler();
    let toml_str = r#"
[energy]
enabled = true
boost_on_mutation = 5.0
half_life_secs = 7200

[synapse]
enabled = false

[fabric]
enabled = false

[co_change]
enabled = false
"#;
    let config = CognitiveConfig::from_toml(toml_str).unwrap();
    let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

    assert_eq!(engine.active_subsystem_count(), 1);
    let energy = engine.energy_store().unwrap();
    assert_eq!(energy.config().boost_on_mutation, 5.0);

    scheduler.shutdown().await;
}

// ---------------------------------------------------------------------------
// Engine is Send + Sync
// ---------------------------------------------------------------------------

#[test]
fn engine_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<obrain_cognitive::DefaultCognitiveEngine>();
}

// ---------------------------------------------------------------------------
// Builder Debug with mixed enabled/disabled
// ---------------------------------------------------------------------------

#[test]
fn builder_debug_mixed() {
    let builder = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .with_co_change(CoChangeConfig::default());

    let dbg = format!("{:?}", builder);
    assert!(dbg.contains("CognitiveEngineBuilder"), "got: {dbg}");
    assert!(dbg.contains("energy"), "got: {dbg}");
    assert!(dbg.contains("co_change"), "got: {dbg}");
}

// ---------------------------------------------------------------------------
// Multiple builds from same scheduler
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multiple_engines_from_same_scheduler() {
    let (_bus, scheduler) = make_scheduler();

    let engine1 = CognitiveEngineBuilder::new()
        .with_energy(EnergyConfig::default())
        .build(&scheduler);

    let engine2 = CognitiveEngineBuilder::new()
        .with_synapses(SynapseConfig::default())
        .build(&scheduler);

    assert_eq!(engine1.active_subsystem_count(), 1);
    assert_eq!(engine2.active_subsystem_count(), 1);

    scheduler.shutdown().await;
}
