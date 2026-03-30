use obrain_common::types::{PropertyKey, Value};
use persona::db::PersonaDB;

macro_rules! prop {
    ($store:expr, $nid:expr, $key:expr, str) => {
        $store.get_node($nid)
            .and_then(|n| n.properties.get(&PropertyKey::from($key))
                .and_then(|v| v.as_str()).map(|s| s.to_string()))
            .unwrap_or_default()
    };
    ($store:expr, $nid:expr, $key:expr, f64) => {
        $store.get_node($nid)
            .and_then(|n| n.properties.get(&PropertyKey::from($key))
                .and_then(|v| match v { Value::Float64(f) => Some(*f), _ => None }))
    };
    ($store:expr, $nid:expr, $key:expr, i64) => {
        $store.get_node($nid)
            .and_then(|n| n.properties.get(&PropertyKey::from($key))
                .and_then(|v| match v { Value::Int64(i) => Some(*i), _ => None }))
    };
    ($store:expr, $nid:expr, $key:expr, bool) => {
        $store.get_node($nid)
            .and_then(|n| n.properties.get(&PropertyKey::from($key))
                .and_then(|v| match v { Value::Bool(b) => Some(*b), _ => None }))
    };
}

#[test]
fn dump_elun_db() {
    let pdb = PersonaDB::open("/tmp/elun").expect("Failed to open /tmp/elun");
    let store = pdb.db.store();

    // Count all labels
    let labels = [
        "Conversation", "Message", "Fact", "Memory", "Pattern",
        "ConvTurn", "GNNWeights", "PersistNetWeights", "RewardToken",
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
    println!("  patterns: {}/{} active ({} auto)",
        stats.patterns_active, stats.patterns_total, stats.patterns_auto);
    println!("  conv_turns: {}, avg_reward_recent={:.3}, avg_mask_reward={:.3}",
        stats.conv_turns, stats.avg_reward_recent, stats.avg_mask_reward);
    println!("  reward_tokens: {}", stats.reward_tokens);

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
        let text = prop!(store, nid, "text", str);
        let score = prop!(store, nid, "persist_score", f64).unwrap_or(0.0);
        let active = prop!(store, nid, "active", bool).unwrap_or(true);
        let status = if active { "✓" } else { "✗" };
        let preview: String = text.chars().take(150).collect();
        println!("  [{}] (score={:.3}) {}", status, score, preview);
    }

    // Dump conversations summary
    let convs = store.nodes_by_label("Conversation");
    let total_msgs = store.nodes_by_label("Message").len();
    println!("\n=== CONVERSATIONS ({}) — {} total messages ===", convs.len(), total_msgs);
    for &cid in &convs {
        let title = prop!(store, cid, "title", str);
        let created = prop!(store, cid, "created_at", str);
        println!("  {} — {}", created, title);
    }

    // Dump ConvTurn rewards
    let turns = store.nodes_by_label("ConvTurn");
    println!("\n=== CONV TURNS ({}) ===", turns.len());
    let mut rewarded = 0;
    for &tid in &turns {
        let reward = prop!(store, tid, "reward", f64);
        let mask_reward = prop!(store, tid, "mask_reward", f64);
        let query = prop!(store, tid, "query_text", str);
        let turn_num = prop!(store, tid, "turn_number", i64).unwrap_or(-1);
        let q_preview: String = query.chars().take(80).collect();
        if reward.is_some() || mask_reward.is_some() {
            rewarded += 1;
        }
        println!("  T{}: reward={:?} mask_r={:?} — {}",
            turn_num, reward, mask_reward, q_preview);
    }
    println!("  ({} with reward signal)", rewarded);

    // Dump patterns
    let patterns = store.nodes_by_label("Pattern");
    println!("\n=== PATTERNS ({}) ===", patterns.len());
    for &pid in &patterns {
        let trigger = prop!(store, pid, "trigger", str);
        let key_tpl = prop!(store, pid, "key_template", str);
        let hits = prop!(store, pid, "hit_count", i64).unwrap_or(0);
        let auto = prop!(store, pid, "auto_generated", bool).unwrap_or(false);
        let avg_r = prop!(store, pid, "avg_reward", f64).unwrap_or(0.0);
        println!("  [{}] trigger=\"{}\" → key=\"{}\" hits={} avg_r={:.2}",
            if auto { "auto" } else { "seed" }, trigger, key_tpl, hits, avg_r);
    }

    // Dump last 40 messages
    let msgs = store.nodes_by_label("Message");
    println!("\n=== LAST 40 MESSAGES (out of {}) ===", msgs.len());
    let mut msg_data: Vec<_> = msgs.iter().map(|&mid| {
        let role = prop!(store, mid, "role", str);
        let content = prop!(store, mid, "content", str);
        let order = prop!(store, mid, "order", i64).unwrap_or(0);
        let reward = prop!(store, mid, "reward", f64);
        (order, role, content, reward)
    }).collect();
    msg_data.sort_by_key(|(o, _, _, _)| *o);
    for (order, role, content, reward) in msg_data.iter().rev().take(40).rev() {
        let preview: String = content.chars().take(120).collect();
        let r_str = reward.map(|r| format!(" [r={:.2}]", r)).unwrap_or_default();
        println!("  #{} [{}]{}: {}", order, role, r_str, preview);
    }
}
