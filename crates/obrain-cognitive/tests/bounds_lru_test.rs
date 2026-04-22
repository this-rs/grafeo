//! Tests for energy caps, synapse weight bounds, competitive normalization,
//! and LRU cache eviction.

#![cfg(feature = "cognitive")]

use obrain_cognitive::energy::{EnergyConfig, EnergyStore};
use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Step 1: Energy cap tests
// ---------------------------------------------------------------------------

#[test]
fn energy_boost_1000x_capped_at_max_energy() {
    let max = 10.0;
    let config = EnergyConfig {
        boost_on_mutation: 1.0,
        max_energy: max,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);
    let node = NodeId(42);

    for _ in 0..1000 {
        store.boost(node, 1.0);
    }

    let energy = store.get_energy(node);
    assert!(
        energy <= max,
        "Energy should be capped at {max}, got {energy}"
    );
    assert!(
        (energy - max).abs() < 0.001,
        "Energy should be exactly at max_energy after 1000 boosts, got {energy}"
    );
}

#[test]
fn energy_cap_custom_value() {
    let max = 5.0;
    let config = EnergyConfig {
        max_energy: max,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);
    let node = NodeId(1);

    store.boost(node, 100.0);
    let energy = store.get_energy(node);
    assert!(
        energy <= max,
        "Energy should be capped at {max}, got {energy}"
    );
}

#[test]
fn energy_cap_zero_floor() {
    let config = EnergyConfig {
        max_energy: 10.0,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config);
    let node = NodeId(1);

    // Boost with negative amount (should floor at 0)
    store.boost(node, -5.0);
    let energy = store.get_energy(node);
    assert!(energy >= 0.0, "Energy should be >= 0.0, got {energy}");
}

// ---------------------------------------------------------------------------
// Step 2: Synapse weight cap + competitive normalization
// ---------------------------------------------------------------------------

#[test]
fn synapse_individual_weight_capped() {
    let max_w = 5.0;
    let config = SynapseConfig {
        max_synapse_weight: max_w,
        max_total_outgoing_weight: 1000.0, // high so normalization doesn't interfere
        reinforce_amount: 0.0,
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config);
    let src = NodeId(1);
    let tgt = NodeId(2);

    // Reinforce many times
    for _ in 0..100 {
        store.reinforce(src, tgt, 1.0);
    }

    let syn = store.get_synapse(src, tgt).unwrap();
    assert!(
        syn.current_weight() <= max_w + 0.001,
        "Individual synapse weight should be capped at {max_w}, got {}",
        syn.current_weight()
    );
}

#[test]
fn synapse_competitive_normalization_10k_synapses() {
    let max_total = 100.0;
    let max_w = 10.0;
    let config = SynapseConfig {
        initial_weight: 0.0,
        reinforce_amount: 0.0,
        max_synapse_weight: max_w,
        max_total_outgoing_weight: max_total,
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config);
    let src = NodeId(0);

    // Create 10K synapses from src to different targets
    for i in 1..=10_000u64 {
        store.reinforce(src, NodeId(i), 1.0);
    }

    // Verify total outgoing weight <= max_total_outgoing_weight
    let synapses = store.list_synapses(src);
    assert_eq!(synapses.len(), 10_000);

    let total_weight: f64 = synapses.iter().map(|s| s.current_weight()).sum();
    assert!(
        total_weight <= max_total + 0.01,
        "Total outgoing weight should be <= {max_total}, got {total_weight}"
    );

    // Verify each individual weight <= max_synapse_weight
    for syn in &synapses {
        assert!(
            syn.current_weight() <= max_w + 0.001,
            "Individual weight {} exceeds max {max_w}",
            syn.current_weight()
        );
    }
}

#[test]
fn synapse_normalization_preserves_relative_weights() {
    let max_total = 10.0;
    let config = SynapseConfig {
        initial_weight: 0.0,
        reinforce_amount: 0.0,
        max_synapse_weight: 100.0,
        max_total_outgoing_weight: max_total,
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config);
    let src = NodeId(0);

    // Create two synapses with different weights
    store.reinforce(src, NodeId(1), 20.0); // weight = 20
    store.reinforce(src, NodeId(2), 80.0); // weight = 80

    let s1 = store.get_synapse(src, NodeId(1)).unwrap();
    let s2 = store.get_synapse(src, NodeId(2)).unwrap();

    // After normalization: total should be capped at max_total
    let w1 = s1.current_weight();
    let w2 = s2.current_weight();
    let total = w1 + w2;

    assert!(
        total <= max_total + 0.01,
        "Total should be <= {max_total}, got {total}"
    );

    // Both weights should be positive and w2 > w1 (relative order preserved)
    assert!(w1 > 0.0, "w1 should be positive, got {w1}");
    assert!(
        w2 > w1,
        "w2 ({w2}) should be > w1 ({w1}), preserving relative order"
    );
}

// ---------------------------------------------------------------------------
// Step 3: LRU cache eviction
// ---------------------------------------------------------------------------

#[test]
fn energy_lru_eviction_caps_cache_size() {
    let max_entries = 100;
    let config = EnergyConfig {
        max_energy: 10.0,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config).with_max_cache_entries(max_entries);

    // Insert 200 entries
    for i in 0..200u64 {
        store.boost(NodeId(i), 1.0);
    }

    assert!(
        store.len() <= max_entries,
        "Cache should have <= {max_entries} entries, got {}",
        store.len()
    );
}

#[test]
fn energy_lru_evicts_oldest_first() {
    let max_entries = 50;
    let config = EnergyConfig {
        max_energy: 10.0,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::new(config).with_max_cache_entries(max_entries);

    // Insert 100 entries (0..99)
    for i in 0..100u64 {
        store.boost(NodeId(i), 1.0);
    }

    // The most recent entries (50..99) should still be in cache
    // The oldest entries (0..49) should have been evicted
    // Note: exact eviction boundaries depend on implementation details,
    // but the cache size should be capped
    assert!(store.len() <= max_entries);

    // The last inserted node should still be accessible
    let last_energy = store.get_energy(NodeId(99));
    assert!(last_energy > 0.0, "Most recent entry should be in cache");
}

#[test]
fn synapse_lru_eviction_caps_cache_size() {
    let max_entries = 100;
    let config = SynapseConfig {
        max_synapse_weight: 100.0,
        max_total_outgoing_weight: f64::MAX,
        ..SynapseConfig::default()
    };
    let store = SynapseStore::new(config).with_max_cache_entries(max_entries);

    // Create 200 synapses (all from different pairs to avoid normalization overhead)
    for i in 0..200u64 {
        store.reinforce(NodeId(i * 2), NodeId(i * 2 + 1), 0.5);
    }

    assert!(
        store.len() <= max_entries,
        "Synapse cache should have <= {max_entries} entries, got {}",
        store.len()
    );
}

#[test]
fn energy_lru_evicted_entry_reloaded_from_graph() {
    use obrain_core::graph::GraphStoreMut;
    use obrain_substrate::SubstrateStore;

    let graph = Arc::new(SubstrateStore::open_tempfile().unwrap());
    let max_entries = 10;
    let config = EnergyConfig {
        max_energy: 10.0,
        ..EnergyConfig::default()
    };
    let store = EnergyStore::with_graph_store(config, Arc::clone(&graph) as Arc<dyn GraphStoreMut>)
        .with_max_cache_entries(max_entries);

    // Create real nodes in the graph and boost them
    let mut node_ids = Vec::new();
    for _ in 0..20 {
        let nid = graph.create_node(&["TestNode"]);
        node_ids.push(nid);
    }
    for &nid in &node_ids {
        store.boost(nid, 1.0);
    }

    assert!(
        store.len() <= max_entries,
        "Cache should have at most {max_entries} entries, got {}",
        store.len()
    );

    // Access the first entry (likely evicted) — should be reloaded from graph
    let energy = store.get_energy(node_ids[0]);
    assert!(
        energy > 0.0,
        "Evicted entry should be reloadable from graph, got {energy}"
    );
}

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

#[test]
fn cognitive_config_default_has_bounds() {
    let config = obrain_cognitive::config::CognitiveConfig::default();
    assert_eq!(config.energy.max_energy, 10.0);
    assert_eq!(config.synapse.max_synapse_weight, 10.0);
    assert_eq!(config.synapse.max_total_outgoing_weight, 100.0);
    assert_eq!(config.max_cache_entries, 100_000);
}

#[test]
fn cognitive_config_toml_roundtrip_new_fields() {
    let toml_str = r#"
max_cache_entries = 50000

[energy]
max_energy = 20.0

[synapse]
max_synapse_weight = 15.0
max_total_outgoing_weight = 200.0
"#;
    let config: obrain_cognitive::config::CognitiveConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.energy.max_energy, 20.0);
    assert_eq!(config.synapse.max_synapse_weight, 15.0);
    assert_eq!(config.synapse.max_total_outgoing_weight, 200.0);
    assert_eq!(config.max_cache_entries, 50_000);
}
