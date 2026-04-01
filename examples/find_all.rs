use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey};

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let db_path = format!("{home}/.obrain/personna/elun");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");
    let store = db.store();

    // Scan EVERY node for "Thomas" or "Lyon" or "développeur"
    eprintln!("=== Scanning ALL nodes (0..2000) ===");
    for i in 0..2000u64 {
        let nid = NodeId::new(i);
        if let Some(node) = store.get_node(nid) {
            for (key, val) in node.properties.iter() {
                if let Some(s) = val.as_str() {
                    let sl = s.to_lowercase();
                    if sl.contains("thomas") || sl.contains("lyon") || sl.contains("développeur") || sl.contains("developpeur") {
                        let labels: Vec<&str> = node.labels.iter().map(|l| { let r: &str = l.as_ref(); r }).collect();
                        let preview: String = s.chars().take(120).collect();
                        eprintln!("  {:?} labels={:?} .{} = {}", nid, labels, key, preview);
                    }
                }
            }
        }
    }

    // Also check: is there a Fact with "name" or "location" key?
    eprintln!("\n=== All Fact nodes ===");
    let facts = store.nodes_by_label("Fact");
    eprintln!("Total Facts: {}", facts.len());
    for &nid in &facts {
        if let Some(node) = store.get_node(nid) {
            let props: Vec<String> = node.properties.iter()
                .map(|(k, v)| {
                    let vs: String = v.as_str().map(|s| s.chars().take(60).collect()).unwrap_or_else(|| format!("{:?}", v));
                    format!("{}={}", k, vs)
                })
                .collect();
            eprintln!("  {:?} → {}", nid, props.join(" | "));
        }
    }

    // Check Pattern nodes too (they generate facts)
    eprintln!("\n=== All Pattern nodes ===");
    let patterns = store.nodes_by_label("Pattern");
    eprintln!("Total Patterns: {}", patterns.len());
    for &nid in &patterns {
        if let Some(node) = store.get_node(nid) {
            for (key, val) in node.properties.iter() {
                if let Some(s) = val.as_str() {
                    if s.to_lowercase().contains("thomas") || s.to_lowercase().contains("lyon") {
                        let preview: String = s.chars().take(100).collect();
                        eprintln!("  {:?} .{} = {}", nid, key, preview);
                    }
                }
            }
        }
    }
}
