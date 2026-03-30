// Quick persona DB dump script — run with:
// cargo test -p persona --test dump_persona -- --nocapture
// (copy to crates/persona/tests/ first)

use persona::db::PersonaDB;

#[test]
fn dump_elun_db() {
    let pdb = PersonaDB::open("/tmp/elun").expect("Failed to open /tmp/elun");
    let store = pdb.db.store();

    // Count all labels
    let labels = [
        "Conversation", "Message", "Fact", "Memory",
        "ConvTurn", "GNNWeights", "PersistNetWeights",
        "Self", "SelfEmbedWeights", "SelfAlphaWeights",
    ];
    println!("\n=== NODE COUNTS ===");
    let mut total = 0;
    for label in &labels {
        let count = store.nodes_by_label(label).len();
        if count > 0 {
            println!("  :{:<20} = {}", label, count);
            total += count;
        }
    }
    println!("  TOTAL              = {}", total);

    // Xi stats
    let stats = pdb.xi_stats();
    println!("\n=== Ξ(t) STATS ===");
    println!("  facts: {}/{} active, avg_energy={:.3}, avg_confidence={:.3}",
        stats.facts_active, stats.facts_total, stats.avg_energy, stats.avg_confidence);
    println!("  memories: {} (PersistNet)", stats.memories);
    println!("  conv_turns: {}, avg_reward_recent={:.3}, avg_mask_reward={:.3}",
        stats.conv_turns, stats.avg_reward_recent, stats.avg_mask_reward);

    // Dump all facts
    let facts = pdb.list_facts();
    println!("\n=== FACTS ({}) ===", facts.len());
    for (nid, key, value, _energy, active, confidence, _cost_eff, fact_type) in &facts {
        let status = if *active { "✓" } else { "✗" };
        println!("  [{}] ({}) {}: {} [conf={:.2}, type={}]",
            status, nid.0, key, value, confidence, fact_type);
    }

    // Dump memories
    let memories = store.nodes_by_label("Memory");
    println!("\n=== MEMORIES ({}) ===", memories.len());
    for &nid in &memories {
        if let Some(node) = store.get_node(nid) {
            let text = node.properties.get(&obrain::PropertyKey::from("text"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let score = node.properties.get(&obrain::PropertyKey::from("persist_score"))
                .and_then(|v| if let obrain::Value::Float64(f) = v { Some(*f) } else { None })
                .unwrap_or(0.0);
            let active = node.properties.get(&obrain::PropertyKey::from("active"))
                .and_then(|v| if let obrain::Value::Bool(b) = v { Some(*b) } else { None })
                .unwrap_or(true);
            let status = if active { "✓" } else { "✗" };
            let preview: String = text.chars().take(120).collect();
            println!("  [{}] (score={:.3}) {}", status, score, preview);
        }
    }

    // Dump conversations summary
    let convs = store.nodes_by_label("Conversation");
    println!("\n=== CONVERSATIONS ({}) ===", convs.len());
    for &cid in &convs {
        if let Some(node) = store.get_node(cid) {
            let title = node.properties.get(&obrain::PropertyKey::from("title"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let created = node.properties.get(&obrain::PropertyKey::from("created_at"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            // Count messages in this conv
            let msg_count = store.edges_from(cid).iter()
                .filter(|e| e.label == obrain::EdgeLabel::from("HAS_MSG"))
                .count();
            println!("  {} — {} ({} messages)", created, title, msg_count);
        }
    }

    // Dump ConvTurn rewards
    let turns = store.nodes_by_label("ConvTurn");
    println!("\n=== CONV TURNS with reward ({}) ===", turns.len());
    let mut rewarded = 0;
    for &tid in &turns {
        if let Some(node) = store.get_node(tid) {
            let reward = node.properties.get(&obrain::PropertyKey::from("reward"))
                .and_then(|v| if let obrain::Value::Float64(f) = v { Some(*f) } else { None });
            let mask_reward = node.properties.get(&obrain::PropertyKey::from("mask_reward"))
                .and_then(|v| if let obrain::Value::Float64(f) = v { Some(*f) } else { None });
            let query = node.properties.get(&obrain::PropertyKey::from("query_text"))
                .and_then(|v| v.as_str()).unwrap_or("");
            let turn_num = node.properties.get(&obrain::PropertyKey::from("turn_number"))
                .and_then(|v| if let obrain::Value::Int64(i) = v { Some(*i) } else { None })
                .unwrap_or(-1);
            if reward.is_some() || mask_reward.is_some() {
                rewarded += 1;
                let q_preview: String = query.chars().take(80).collect();
                println!("  T{}: reward={:?} mask_reward={:?} — {}",
                    turn_num, reward, mask_reward, q_preview);
            }
        }
    }
    println!("  ({} turns with reward signal out of {})", rewarded, turns.len());
}
