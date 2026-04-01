use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey, Value};
use obrain_core::graph::Direction;

fn main() {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/elun".to_string());
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");
    let store = db.store();

    let convs = store.nodes_by_label("Conversation");
    eprintln!("Conversations: {}", convs.len());

    let mut prompts: Vec<(String, i64, String, NodeId)> = Vec::new();
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
                    .and_then(|v| { if let Value::Int64(n) = v { Some(*n) } else { None } })
                    .unwrap_or(0);
                prompts.push((ts, order, content, target));
            }
        }
    }
    prompts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    eprintln!("User history entries: {}\n", prompts.len());
    // Print ALL — look for any benchmark remnants
    for (i, (ts, _ord, content, nid)) in prompts.iter().enumerate() {
        let preview: String = content.chars().take(100).collect();
        let ts_short = if ts.len() >= 19 { &ts[..19] } else { ts.as_str() };
        let flag = if content.contains("Thomas") || content.contains("Marc") || content.contains("Lyon")
            || content.contains("développeur") || content.contains("benchmark") || content.contains("villes")
            { " ⚠️ BENCH?" } else { "" };
        eprintln!("[{:3}] {:?} [{}] {}{}", i, nid, ts_short, preview, flag);
    }
}
