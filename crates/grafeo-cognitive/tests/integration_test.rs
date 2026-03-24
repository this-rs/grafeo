//! End-to-end integration test for the CognitiveEngine.
//!
//! Opens a MutationBus + Scheduler, builds a CognitiveEngine with default config
//! (energy + synapses), publishes mutations, and verifies that energy boosts and
//! synapse reinforcement happen automatically.

#[cfg(feature = "cognitive")]
mod cognitive_integration {
    use grafeo_cognitive::{CognitiveConfig, CognitiveEngine, CognitiveEngineBuilder};
    use grafeo_common::types::NodeId;
    use grafeo_reactive::{
        BatchConfig, MutationBatch, MutationBus, MutationEvent, NodeSnapshot, Scheduler,
    };
    use smallvec::smallvec;
    use std::time::Duration;

    fn node_snapshot(id: u64) -> NodeSnapshot {
        NodeSnapshot {
            id: NodeId(id),
            labels: smallvec![],
            properties: vec![],
        }
    }

    #[tokio::test]
    async fn engine_from_default_config_activates_energy_and_synapse() {
        let bus = MutationBus::new();
        let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));

        let config = CognitiveConfig::default();
        let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

        // Default config: energy + synapse enabled
        assert_eq!(engine.active_subsystem_count(), 2);
        assert!(engine.energy_store().is_some());
        assert!(engine.synapse_store().is_some());

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn engine_builder_selective_activation() {
        let bus = MutationBus::new();
        let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));

        // Only energy, no synapses
        let engine = CognitiveEngineBuilder::new()
            .with_energy(grafeo_cognitive::EnergyConfig::default())
            .build(&scheduler);

        assert_eq!(engine.active_subsystem_count(), 1);
        assert!(engine.energy_store().is_some());
        assert!(engine.synapse_store().is_none());

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn mutation_triggers_energy_boost_and_synapse_reinforcement() {
        let bus = MutationBus::new();
        let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));

        let config = CognitiveConfig::default();
        let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

        let energy_store = engine.energy_store().unwrap().clone();
        let synapse_store = engine.synapse_store().unwrap().clone();

        // Before mutation: no energy tracked
        assert_eq!(energy_store.len(), 0);
        assert_eq!(synapse_store.len(), 0);

        // Publish a batch with two node creations (co-activation → synapse)
        let batch = MutationBatch::new(vec![
            MutationEvent::NodeCreated {
                node: node_snapshot(1),
            },
            MutationEvent::NodeCreated {
                node: node_snapshot(2),
            },
        ]);
        bus.publish_batch(batch);

        // Give the scheduler time to dispatch
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Energy should be boosted for both nodes
        assert!(
            energy_store.get_energy(NodeId(1)) > 0.0,
            "node 1 should have energy after mutation"
        );
        assert!(
            energy_store.get_energy(NodeId(2)) > 0.0,
            "node 2 should have energy after mutation"
        );

        // Synapse should exist between the two co-activated nodes
        let synapse = synapse_store.get_synapse(NodeId(1), NodeId(2));
        assert!(
            synapse.is_some(),
            "synapse should exist between co-activated nodes"
        );
        let synapse = synapse.unwrap();
        assert!(
            synapse.current_weight() > 0.0,
            "synapse weight should be positive"
        );

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn repeated_mutations_reinforce_synapse() {
        let bus = MutationBus::new();
        let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));

        let config = CognitiveConfig::default();
        let engine = CognitiveEngineBuilder::from_config(&config).build(&scheduler);

        let synapse_store = engine.synapse_store().unwrap().clone();

        // First batch
        bus.publish_batch(MutationBatch::new(vec![
            MutationEvent::NodeCreated {
                node: node_snapshot(10),
            },
            MutationEvent::NodeCreated {
                node: node_snapshot(20),
            },
        ]));
        tokio::time::sleep(Duration::from_millis(150)).await;

        let weight_after_first = synapse_store
            .get_synapse(NodeId(10), NodeId(20))
            .unwrap()
            .current_weight();

        // Second batch — same nodes → reinforces synapse
        bus.publish_batch(MutationBatch::new(vec![
            MutationEvent::NodeUpdated {
                before: node_snapshot(10),
                after: node_snapshot(10),
            },
            MutationEvent::NodeUpdated {
                before: node_snapshot(20),
                after: node_snapshot(20),
            },
        ]));
        tokio::time::sleep(Duration::from_millis(150)).await;

        let weight_after_second = synapse_store
            .get_synapse(NodeId(10), NodeId(20))
            .unwrap()
            .current_weight();

        assert!(
            weight_after_second > weight_after_first,
            "synapse weight should increase after repeated co-activation: {} > {}",
            weight_after_second,
            weight_after_first
        );

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn config_from_toml_roundtrip() {
        let toml_str = r#"
[energy]
enabled = true
boost_on_mutation = 2.0

[synapse]
enabled = true
reinforce_amount = 0.5

[fabric]
enabled = false
"#;
        let config = CognitiveConfig::from_toml(toml_str).unwrap();
        assert!(config.energy.enabled);
        assert_eq!(config.energy.boost_on_mutation, 2.0);
        assert!(config.synapse.enabled);
        assert_eq!(config.synapse.reinforce_amount, 0.5);
        assert!(!config.fabric.enabled);
    }
}
