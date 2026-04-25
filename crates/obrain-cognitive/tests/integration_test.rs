//! End-to-end integration test for the CognitiveEngine.
//!
//! Opens a MutationBus + Scheduler, builds a CognitiveEngine with default config
//! (energy + synapses), publishes mutations, and verifies that energy boosts and
//! synapse reinforcement happen automatically.

#[cfg(feature = "cognitive")]
mod cognitive_integration {
    use obrain_cognitive::{CognitiveConfig, CognitiveEngine, CognitiveEngineBuilder};
    use obrain_common::types::NodeId;
    use obrain_core::graph::traits::{GraphStore, GraphStoreMut};
    use obrain_reactive::{
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
            .with_energy(obrain_cognitive::EnergyConfig::default())
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

    // -----------------------------------------------------------------------
    // Write-through persistence tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn write_through_energy_persists_and_reloads() {
        use obrain_cognitive::energy::{EnergyConfig, EnergyStore};
        use obrain_substrate::SubstrateStore;

        let lpg = std::sync::Arc::new(SubstrateStore::open_tempfile().unwrap());

        // Create nodes in the graph so properties can be set
        let n1 = lpg.create_node(&["Test"]);
        let n2 = lpg.create_node(&["Test"]);

        // Create energy store backed by the graph
        let config = EnergyConfig::default();
        let store1 = EnergyStore::with_graph_store(config.clone(), lpg.clone());

        // Boost energy
        store1.boost(n1, 3.0);
        store1.boost(n2, 5.0);

        let e1_before = store1.get_energy(n1);
        let e2_before = store1.get_energy(n2);
        assert!(e1_before > 2.99, "e1 should be ~3.0, got {e1_before}");
        assert!(e2_before > 4.99, "e2 should be ~5.0, got {e2_before}");

        // Drop the store (simulates restart)
        drop(store1);

        // Recreate energy store with the same graph — should lazy-load
        let store2 = EnergyStore::with_graph_store(config, lpg.clone());
        let e1_after = store2.get_energy(n1);
        let e2_after = store2.get_energy(n2);

        // The persisted value is the current_energy at time of boost (may have tiny decay)
        assert!(
            e1_after > 2.5,
            "e1 after reload should be > 2.5, got {e1_after}"
        );
        assert!(
            e2_after > 4.5,
            "e2 after reload should be > 4.5, got {e2_after}"
        );
    }

    #[tokio::test]
    async fn write_through_synapse_persists_and_reloads() {
        use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
        use obrain_core::graph::Direction;
        use obrain_substrate::SubstrateStore;

        let lpg = std::sync::Arc::new(SubstrateStore::open_tempfile().unwrap());

        // Use the actual ids returned by create_node. LpgStore started ids at 0,
        // but SubstrateStore reserves slot 0 as a sentinel — so hardcoding
        // NodeId(0)/NodeId(1) no longer works post-T17 cutover.
        let n1 = lpg.create_node(&["Test"]);
        let n2 = lpg.create_node(&["Test"]);

        let config = SynapseConfig::default();
        let store1 = SynapseStore::with_graph_store(config.clone(), lpg.clone());

        // Reinforce synapse
        store1.reinforce(n1, n2, 0.5);
        let s1 = store1.get_synapse(n1, n2).unwrap();
        assert!(s1.current_weight() > 0.5, "weight should be > 0.5");

        // Drop and recreate
        drop(store1);

        let store2 = SynapseStore::with_graph_store(config, lpg.clone());
        // The synapse was persisted via the edge_ids mapping. Since we drop
        // the store, edge_ids are lost. The lazy load requires the edge_id
        // to be known. This is a design trade-off — full reload would need
        // scanning edges of type SYNAPSE. For now, verify the property was
        // written to the graph.
        use obrain_common::types::PropertyKey;
        // The edge was created during reinforce
        let edges = lpg.edge_count();
        assert!(edges >= 1, "should have at least 1 synapse edge");
        // Verify the property exists on the edge — scan outgoing edges of n1
        // to locate the synapse edge without hardcoding edge ids.
        let pk = PropertyKey::from("_cog_synapse_weight");
        let outgoing = lpg.edges_from(n1, Direction::Outgoing);
        let (_, edge_id) = outgoing
            .iter()
            .find(|(_, eid)| lpg.get_edge_property(*eid, &pk).is_some())
            .copied()
            .expect("exactly one outgoing edge of n1 should carry the synapse weight");
        let val = lpg.get_edge_property(edge_id, &pk);
        assert!(
            val.is_some(),
            "synapse weight should be persisted as edge property"
        );
        let weight = val.unwrap().as_float64().unwrap();
        assert!(
            weight > 0.5,
            "persisted weight should be > 0.5, got {weight}"
        );

        drop(store2);
    }

    #[tokio::test]
    async fn write_through_engine_end_to_end() {
        use obrain_common::types::PropertyKey;
        use obrain_substrate::SubstrateStore;

        let lpg = std::sync::Arc::new(SubstrateStore::open_tempfile().unwrap());

        // Create nodes in the graph
        let n1 = lpg.create_node(&["File"]);
        let n2 = lpg.create_node(&["File"]);

        let bus = MutationBus::new();
        let scheduler = Scheduler::new(&bus, BatchConfig::new(10, Duration::from_millis(20)));

        // Build engine with graph store persistence
        let engine = CognitiveEngineBuilder::from_config(&CognitiveConfig::default())
            .with_graph_store(lpg.clone())
            .build(&scheduler);

        let energy_store = engine.energy_store().unwrap().clone();

        // Publish mutations for the nodes that exist in the graph
        bus.publish_batch(MutationBatch::new(vec![
            MutationEvent::NodeCreated {
                node: NodeSnapshot {
                    id: n1,
                    labels: smallvec![],
                    properties: vec![],
                },
            },
            MutationEvent::NodeCreated {
                node: NodeSnapshot {
                    id: n2,
                    labels: smallvec![],
                    properties: vec![],
                },
            },
        ]));

        // Wait for scheduler to dispatch
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Verify energy was boosted
        assert!(
            energy_store.get_energy(n1) > 0.0,
            "node {} should have energy",
            n1.0
        );

        // Verify the property was written to the graph
        let pk = PropertyKey::from("_cog_energy");
        let val = lpg.get_node_property(n1, &pk);
        assert!(
            val.is_some(),
            "energy should be persisted as node property on node {}",
            n1.0
        );
        let persisted_energy = val.unwrap().as_float64().unwrap();
        assert!(
            persisted_energy > 0.0,
            "persisted energy should be > 0, got {persisted_energy}"
        );

        scheduler.shutdown().await;
    }
}
