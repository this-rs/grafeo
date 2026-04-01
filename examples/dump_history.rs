//! Dump user history entries from PersonaDB (for readline debug).
use obrain::ObrainDB;
use obrain_common::types::PropertyKey;
use obrain_core::graph::Direction;

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let db_path = format!("{home}/.obrain/personna/elun");
    eprintln!("Opening DB at {db_path}");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");

    let store = db.store();
    let convs = store.nodes_by_label("Conversation");
    eprintln!("Conversations: {}", convs.len());

    let mut prompts: Vec<(String, i64, String)> = Vec::new();
    for &conv_id in &convs {
        for (target, _eid) in store.edges_from(conv_id, Direction::Outgoing).collect::<Vec<_>>() {
            if let Some(node) = store.get_node(target) {
                let is_msg = node.labels.iter().any(|l| { let s: &str = l.as_ref(); s == "Message" });
                if !is_msg { continue; }
                let role = node.properties.get(&PropertyKey::from("role"))
                    .and_then(|v| v.as_str()).unwrap_or("");
                if role != "user" { continue; }
                let content = node.properties.get(&PropertyKey::from("content"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                if content.is_empty() { continue; }
                let ts = node.properties.get(&PropertyKey::from("timestamp"))
                    .and_then(|v| v.as_str()).unwrap_or("").to_string();
                let order = node.properties.get(&PropertyKey::from("order"))
                    .and_then(|v| { if let obrain_common::types::Value::Int64(n) = v { Some(*n) } else { None } })
                    .unwrap_or(0);
                prompts.push((ts, order, content));
            }
        }
    }
    prompts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    eprintln!("\nUser history entries: {}", prompts.len());
    eprintln!("\n=== FIRST 15 (oldest) ===");
    for (i, (ts, _ord, content)) in prompts.iter().take(15).enumerate() {
        let preview: String = content.chars().take(100).collect();
        let ts_short = if ts.len() >= 19 { &ts[..19] } else { ts.as_str() };
        eprintln!("  [{:3}] [{}] {}", i, ts_short, preview);
    }
    eprintln!("\n=== LAST 15 (newest — first on arrow-up) ===");
    let start = prompts.len().saturating_sub(15);
    for (i, (ts, _ord, content)) in prompts.iter().skip(start).enumerate() {
        let preview: String = content.chars().take(100).collect();
        let ts_short = if ts.len() >= 19 { &ts[..19] } else { ts.as_str() };
        eprintln!("  [{:3}] [{}] {}", start + i, ts_short, preview);
    }
}
