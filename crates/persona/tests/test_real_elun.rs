use persona::db::PersonaDB;

#[test]
fn inspect_real_elun() {
    let pdb = PersonaDB::open("/tmp/elun").expect("Failed to open /tmp/elun");
    let store = pdb.db.store();
    
    // Check all labels
    let all_labels = ["AttnFormula", "Conversation", "Message", "Fact", "Memory", 
                      "Pattern", "ConvTurn", "GNNWeights", "PersistNetWeights", "RewardToken",
                      "Self"];
    for label in &all_labels {
        let count = store.nodes_by_label(label).len();
        if count > 0 {
            println!("  :{:<20} = {}", label, count);
        }
    }
    
    // Total node count
    println!("  Total nodes: {}", store.node_count());
    println!("  Total edges: {}", store.edge_count());
    
    // Check if seed would create formulas
    drop(store);
    let would_seed = pdb.list_formulas().len();
    println!("\n  Formulas in DB: {}", would_seed);
    
    // Try seeding now
    let n = pdb.seed_formulas_if_empty();
    println!("  Seeded: {} new formulas", n);
    
    let after = pdb.list_formulas().len();
    println!("  After seed: {} formulas", after);
}
