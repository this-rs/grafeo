use persona::db::PersonaDB;
use obrain_common::PropertyKey;
use obrain_common::Value;

#[test]
fn dump_formulas() {
    let pdb = PersonaDB::open("/tmp/elun").expect("Failed to open /tmp/elun");

    let formulas = pdb.list_formulas();
    println!("\n=== ATTENTION FORMULAS ({}) ===", formulas.len());
    for f in &formulas {
        let status = if f.active { "✓ ACTIVE" } else { "✗ dead" };
        let parent = f.parent_id.map(|p| format!(" parent={}", p.0)).unwrap_or_default();
        println!("  [{}] {} (id={}, gen={}, energy={:.3}, avg_r={:.4}, activations={}, affinity={:?}){}",
            status, f.name, f.id.0, f.generation, f.energy, f.avg_reward, f.activation_count, f.context_affinity, parent);
        println!("       DSL: {}", f.dsl_json);
    }

    // GNN weights
    let store = pdb.db.store();
    let gnn_nodes = store.nodes_by_label("GNNWeights");
    println!("\n=== GNN WEIGHTS ({}) ===", gnn_nodes.len());
    for &nid in &gnn_nodes {
        if let Some(node) = store.get_node(nid) {
            let layer = node.properties.get(&PropertyKey::from("layer"))
                .and_then(|v| if let Value::Int64(i) = v { Some(*i) } else { None })
                .unwrap_or(-1);
            let dim = node.properties.get(&PropertyKey::from("dim"))
                .and_then(|v| if let Value::Int64(i) = v { Some(*i) } else { None })
                .unwrap_or(-1);
            println!("  GNN layer={}, dim={}, id={}", layer, dim, nid.0);
        }
    }

    // Reward tokens
    let reward_nodes = store.nodes_by_label("RewardToken");
    println!("\n=== REWARD TOKENS ({}) ===", reward_nodes.len());
    for &nid in &reward_nodes {
        if let Some(node) = store.get_node(nid) {
            let word = node.properties.get(&PropertyKey::from("word"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let polarity = node.properties.get(&PropertyKey::from("polarity"))
                .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
                .unwrap_or(0.0);
            let lang = node.properties.get(&PropertyKey::from("lang"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            print!("  [{:+.1}] \"{}\" ({})  |  ", polarity, word, lang);
        }
    }
    println!();

    // PersistNet weights
    let persist_nodes = store.nodes_by_label("PersistNetWeights");
    println!("\n=== PERSIST NET WEIGHTS ({}) ===", persist_nodes.len());
}
