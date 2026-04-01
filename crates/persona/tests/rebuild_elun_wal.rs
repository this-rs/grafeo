use persona::db::PersonaDB;
use obrain::ObrainDB;
use obrain_common::types::Value;

/// Rebuild Elun's WAL by creating a fresh DB with all current data.
/// Run with: cargo test -p persona --test rebuild_elun_wal -- --nocapture
#[test]
fn rebuild_elun_wal() {
    let src = "/tmp/elun";
    let dst = "/tmp/elun_rebuilt";
    let _ = std::fs::remove_dir_all(dst);

    // Open source
    let src_pdb = PersonaDB::open(src).expect("open src");
    let src_store = src_pdb.db.store();
    println!("Source: nodes={}, edges={}", src_store.node_count(), src_store.edge_count());

    // Create fresh destination
    let dst_db = ObrainDB::open(dst).expect("open dst");

    // Copy all nodes with properties
    let mut node_count = 0u32;
    for nid_raw in 0..2000u64 {
        let nid = obrain_common::types::NodeId(nid_raw);
        if let Some(node) = src_store.get_node(nid) {
            let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
            let props: Vec<(String, Value)> = node.properties.iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let new_id = dst_db.create_node_with_props(&labels, props);
            if new_id != nid {
                println!("  ⚠ ID mismatch: expected {}, got {}", nid_raw, new_id.0);
            }
            node_count += 1;
        }
    }
    println!("Copied {} nodes", node_count);

    // Copy all edges
    let mut edge_count = 0u32;
    for nid_raw in 0..2000u64 {
        let nid = obrain_common::types::NodeId(nid_raw);
        for (target_nid, eid) in src_store.edges_from(nid, obrain_core::graph::Direction::Outgoing) {
            if let Some(edge) = src_store.get_edge(eid) {
                dst_db.create_edge(nid, target_nid, edge.edge_type.as_str());
                edge_count += 1;
            }
        }
    }
    println!("Copied {} edges", edge_count);
    drop(src_store);
    drop(dst_db);

    // Seed formulas in rebuilt DB
    let dst_pdb = PersonaDB::open(dst).expect("reopen dst");
    let n = dst_pdb.seed_formulas_if_empty();
    println!("Seeded {} formulas", n);

    let store = dst_pdb.db.store();
    println!("\nRebuilt: nodes={}, edges={}, formulas={}",
        store.node_count(), store.edge_count(),
        store.nodes_by_label("AttnFormula").len());
    drop(store);
    drop(dst_pdb);

    // Verify roundtrip
    let verify = PersonaDB::open(dst).expect("verify");
    let formulas = verify.list_formulas();
    println!("After reopen: {} formulas", formulas.len());
    for f in &formulas {
        println!("  ✅ {} (energy={:.2})", f.name, f.energy);
    }

    let vstore = verify.db.store();
    println!("Verify: nodes={}, facts={}, memories={}, patterns={}",
        vstore.node_count(),
        vstore.nodes_by_label("Fact").len(),
        vstore.nodes_by_label("Memory").len(),
        vstore.nodes_by_label("Pattern").len());

    if formulas.len() == 6 {
        println!("\n🎉 Rebuild successful!");
        println!("To apply: mv /tmp/elun /tmp/elun_old && mv /tmp/elun_rebuilt /tmp/elun");
        println!("(then: cp /tmp/elun_old/projection_net.bin /tmp/elun/)");
    }
}
