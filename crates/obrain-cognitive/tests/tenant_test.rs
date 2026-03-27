//! Integration tests for per-tenant isolation via named graphs.
//!
//! Verifies:
//! 1. Two tenants write simultaneously without interference
//! 2. Same node labels in different tenants have independent scores
//! 3. GDS/cognitive algorithms return only tenant-scoped results
//! 4. delete_tenant cleans up all cognitive state

#![cfg(feature = "cognitive")]

use grafeo_cognitive::TenantManager;
use grafeo_common::types::NodeId;
use std::sync::Arc;
use std::thread;

// ---------------------------------------------------------------------------
// Test 1: Two tenants write simultaneously without interference
// ---------------------------------------------------------------------------

#[test]
fn tenant_concurrent_writes_no_interference() {
    let tm = Arc::new(TenantManager::new());
    tm.create_tenant("alpha").unwrap();
    tm.create_tenant("beta").unwrap();

    let tm1 = Arc::clone(&tm);
    let tm2 = Arc::clone(&tm);

    // Spawn two threads writing to different tenants concurrently
    let h1 = thread::spawn(move || {
        let g = tm1.get_tenant("alpha").unwrap();
        for i in 0..100 {
            g.energy_store.boost(NodeId(i), 1.0);
        }
    });

    let h2 = thread::spawn(move || {
        let g = tm2.get_tenant("beta").unwrap();
        for i in 0..100 {
            g.energy_store.boost(NodeId(i), 2.0);
        }
    });

    h1.join().unwrap();
    h2.join().unwrap();

    // Verify each tenant has 100 nodes
    let alpha = tm.get_tenant("alpha").unwrap();
    let beta = tm.get_tenant("beta").unwrap();
    assert_eq!(alpha.energy_store.len(), 100);
    assert_eq!(beta.energy_store.len(), 100);

    // Verify energies are correct (no cross-contamination)
    let alpha_energy = alpha.energy_store.get_energy(NodeId(50));
    let beta_energy = beta.energy_store.get_energy(NodeId(50));
    assert!(
        (alpha_energy - 1.0).abs() < 0.1,
        "alpha energy for node 50 should be ~1.0, got {alpha_energy}"
    );
    assert!(
        (beta_energy - 2.0).abs() < 0.1,
        "beta energy for node 50 should be ~2.0, got {beta_energy}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Same node labels in different tenants have independent scores
// ---------------------------------------------------------------------------

#[test]
fn tenant_same_labels_independent_scores() {
    let tm = TenantManager::new();
    tm.create_tenant("prod").unwrap();
    tm.create_tenant("staging").unwrap();

    let node_a = NodeId(1);
    let node_b = NodeId(2);

    // In prod: high energy for node_a, synapse between a-b
    {
        let g = tm.get_tenant("prod").unwrap();
        g.energy_store.boost(node_a, 8.0);
        g.energy_store.boost(node_b, 1.0);
        g.synapse_store.reinforce(node_a, node_b, 5.0);
    }

    // In staging: low energy for node_a, no synapse
    {
        let g = tm.get_tenant("staging").unwrap();
        g.energy_store.boost(node_a, 0.5);
        g.energy_store.boost(node_b, 0.5);
        // No synapse reinforcement
    }

    // Verify energy independence
    let prod = tm.get_tenant("prod").unwrap();
    let staging = tm.get_tenant("staging").unwrap();

    let prod_energy_a = prod.energy_store.get_energy(node_a);
    let staging_energy_a = staging.energy_store.get_energy(node_a);
    assert!(
        prod_energy_a > 5.0,
        "prod node_a energy should be high, got {prod_energy_a}"
    );
    assert!(
        staging_energy_a < 2.0,
        "staging node_a energy should be low, got {staging_energy_a}"
    );

    // Verify synapse independence
    assert!(
        prod.synapse_store.get_synapse(node_a, node_b).is_some(),
        "prod should have synapse a-b"
    );
    assert!(
        staging.synapse_store.get_synapse(node_a, node_b).is_none(),
        "staging should NOT have synapse a-b"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Cognitive algorithms return only tenant-scoped results
// ---------------------------------------------------------------------------

#[test]
fn tenant_algorithms_scoped_results() {
    let tm = TenantManager::new();
    tm.create_tenant("tenant_x").unwrap();
    tm.create_tenant("tenant_y").unwrap();

    // tenant_x: 3 nodes, 2 synapses
    {
        let g = tm.get_tenant("tenant_x").unwrap();
        g.energy_store.boost(NodeId(1), 5.0);
        g.energy_store.boost(NodeId(2), 3.0);
        g.energy_store.boost(NodeId(3), 1.0);
        g.synapse_store.reinforce(NodeId(1), NodeId(2), 1.0);
        g.synapse_store.reinforce(NodeId(2), NodeId(3), 0.5);
    }

    // tenant_y: 1 node, 0 synapses
    {
        let g = tm.get_tenant("tenant_y").unwrap();
        g.energy_store.boost(NodeId(100), 10.0);
    }

    // EnergyStore.snapshot() returns only tenant-scoped nodes
    let x = tm.get_tenant("tenant_x").unwrap();
    let y = tm.get_tenant("tenant_y").unwrap();

    let x_snapshot = x.energy_store.snapshot();
    let y_snapshot = y.energy_store.snapshot();

    assert_eq!(x_snapshot.len(), 3, "tenant_x should have 3 energy nodes");
    assert_eq!(y_snapshot.len(), 1, "tenant_y should have 1 energy node");

    // Verify no node_ids leak between tenants
    let x_ids: Vec<NodeId> = x_snapshot.iter().map(|(id, _)| *id).collect();
    assert!(
        !x_ids.contains(&NodeId(100)),
        "tenant_x should not contain tenant_y's node"
    );

    let y_ids: Vec<NodeId> = y_snapshot.iter().map(|(id, _)| *id).collect();
    assert!(
        !y_ids.contains(&NodeId(1)),
        "tenant_y should not contain tenant_x's node"
    );

    // SynapseStore.list_synapses returns only tenant-scoped synapses
    let x_synapses = x.synapse_store.list_synapses(NodeId(2));
    assert_eq!(
        x_synapses.len(),
        2,
        "tenant_x node 2 should have 2 synapses"
    );

    let y_synapses = y.synapse_store.list_synapses(NodeId(2));
    assert_eq!(
        y_synapses.len(),
        0,
        "tenant_y should have 0 synapses for node 2"
    );

    // list_low_energy scoped to tenant
    let x_low = x.energy_store.list_low_energy(2.0);
    assert_eq!(x_low.len(), 1, "tenant_x should have 1 low-energy node");
    assert_eq!(x_low[0], NodeId(3));
}

// ---------------------------------------------------------------------------
// Test 4: delete_tenant cleans up all cognitive state
// ---------------------------------------------------------------------------

#[test]
fn tenant_delete_cleans_all_state() {
    let tm = TenantManager::new();
    tm.create_tenant("ephemeral").unwrap();
    tm.create_tenant("persistent").unwrap();

    // Populate both tenants
    {
        let g = tm.get_tenant("ephemeral").unwrap();
        g.energy_store.boost(NodeId(1), 5.0);
        g.energy_store.boost(NodeId(2), 3.0);
        g.synapse_store.reinforce(NodeId(1), NodeId(2), 2.0);
        assert_eq!(g.energy_store.len(), 2);
        assert_eq!(g.synapse_store.len(), 1);
    }
    {
        let g = tm.get_tenant("persistent").unwrap();
        g.energy_store.boost(NodeId(10), 7.0);
    }

    // Switch to ephemeral, then delete it
    tm.switch_tenant("ephemeral").unwrap();
    assert_eq!(tm.active_tenant_name().as_deref(), Some("ephemeral"));

    tm.delete_tenant("ephemeral").unwrap();

    // Verify cleanup
    assert!(tm.get_tenant("ephemeral").is_none());
    assert!(tm.active_tenant_name().is_none()); // active cleared
    assert_eq!(tm.tenant_count(), 1);

    // persistent tenant is unaffected
    let p = tm.get_tenant("persistent").unwrap();
    let energy = p.energy_store.get_energy(NodeId(10));
    assert!(
        (energy - 7.0).abs() < 0.5,
        "persistent tenant should still have node 10 energy ~7.0, got {energy}"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Switch tenant changes active scope
// ---------------------------------------------------------------------------

#[test]
fn tenant_switch_changes_active_scope() {
    let tm = TenantManager::new();
    tm.create_tenant("scope_a").unwrap();
    tm.create_tenant("scope_b").unwrap();

    // Populate different data
    {
        let g = tm.get_tenant("scope_a").unwrap();
        g.energy_store.boost(NodeId(1), 8.0);
    }
    {
        let g = tm.get_tenant("scope_b").unwrap();
        g.energy_store.boost(NodeId(1), 0.1);
    }

    // Switch to scope_a and verify active store
    tm.switch_tenant("scope_a").unwrap();
    let active = tm.active_graph().unwrap();
    let energy = active.energy_store.get_energy(NodeId(1));
    assert!(
        energy > 5.0,
        "active scope_a should have high energy for node 1"
    );

    // Switch to scope_b
    tm.switch_tenant("scope_b").unwrap();
    let active = tm.active_graph().unwrap();
    let energy = active.energy_store.get_energy(NodeId(1));
    assert!(
        energy < 1.0,
        "active scope_b should have low energy for node 1"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Concurrent tenant creation/deletion is safe
// ---------------------------------------------------------------------------

#[test]
fn tenant_concurrent_lifecycle() {
    let tm = Arc::new(TenantManager::new());
    let mut handles = vec![];

    // Spawn 10 threads each creating a tenant
    for i in 0..10 {
        let tm_clone = Arc::clone(&tm);
        handles.push(thread::spawn(move || {
            let name = format!("concurrent_{i}");
            tm_clone.create_tenant(&name).unwrap();
            let g = tm_clone.get_tenant(&name).unwrap();
            g.energy_store.boost(NodeId(i as u64), 1.0);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tm.tenant_count(), 10);

    // Delete half
    for i in 0..5 {
        tm.delete_tenant(&format!("concurrent_{i}")).unwrap();
    }
    assert_eq!(tm.tenant_count(), 5);

    // Remaining tenants still work
    for i in 5..10 {
        let g = tm.get_tenant(&format!("concurrent_{i}")).unwrap();
        let energy = g.energy_store.get_energy(NodeId(i as u64));
        assert!(energy > 0.5, "tenant concurrent_{i} should have energy");
    }
}
