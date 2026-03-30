use persona::db::PersonaDB;
use obrain_common::PropertyKey;
use obrain_common::Value;

#[test]
fn check_patterns() {
    let pdb = PersonaDB::open("/tmp/elun").expect("open");
    let store = pdb.db.store();
    
    // Check Pattern nodes
    let patterns = store.nodes_by_label("Pattern");
    println!("\n=== PATTERNS ({}) ===", patterns.len());
    for &nid in &patterns {
        if let Some(node) = store.get_node(nid) {
            let trigger = node.properties.get(&PropertyKey::from("trigger"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let key = node.properties.get(&PropertyKey::from("key_template"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let hits = node.properties.get(&PropertyKey::from("hit_count"))
                .and_then(|v| if let Value::Int64(i) = v { Some(*i) } else { None })
                .unwrap_or(0);
            let active = node.properties.get(&PropertyKey::from("active"))
                .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                .unwrap_or(true);
            println!("  [{:>2} hits] key={:<20} active={} trigger=\"{}\"", hits, key, active, trigger);
        }
    }
    
    // Check Facts
    let facts = store.nodes_by_label("Fact");
    println!("\n=== FACTS ({}) ===", facts.len());
    for &nid in &facts {
        if let Some(node) = store.get_node(nid) {
            let key = node.properties.get(&PropertyKey::from("key"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let value = node.properties.get(&PropertyKey::from("value"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let active = node.properties.get(&PropertyKey::from("active"))
                .and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                .unwrap_or(true);
            println!("  [{}] {} = \"{}\"", if active { "✓" } else { "✗" }, key, value);
        }
    }

    // Check Memories
    let memories = store.nodes_by_label("Memory");
    println!("\n=== MEMORIES ({}) ===", memories.len());
    for &nid in &memories {
        if let Some(node) = store.get_node(nid) {
            let text = node.properties.get(&PropertyKey::from("text"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let energy = node.properties.get(&PropertyKey::from("energy"))
                .and_then(|v| if let Value::Float64(f) = v { Some(*f) } else { None })
                .unwrap_or(0.0);
            let show: String = text.chars().take(80).collect();
            println!("  [e={:.2}] \"{}\"", energy, show);
        }
    }
    
    // Check ConvTurn
    let turns = store.nodes_by_label("ConvTurn");
    println!("\n=== CONV TURNS ({}) ===", turns.len());
    
    // Reward tokens — check the WORD property
    let rtokens = store.nodes_by_label("RewardToken");
    println!("\n=== REWARD TOKENS ({}) — first 10 ===", rtokens.len());
    for &nid in rtokens.iter().take(10) {
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
}
