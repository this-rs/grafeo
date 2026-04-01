use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey};

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let db_path = format!("{home}/.obrain/personna/elun");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");
    let store = db.store();

    let target = "Je m'appelle Thomas";
    eprintln!("Searching ALL nodes for '{target}'...\n");

    // Check ALL nodes, not just Messages
    let all_labels = ["Message", "Fact", "Conversation", "Pattern", "ConvTurn", "Formula"];
    for label in &all_labels {
        let nodes = store.nodes_by_label(label);
        let mut found = 0;
        for &nid in &nodes {
            if let Some(node) = store.get_node(nid) {
                for (key, val) in node.properties.iter() {
                    if let Some(s) = val.as_str() {
                        if s.contains("Thomas") || s.contains("Marc Dupont") || s.contains("Lyon") {
                            let preview: String = s.chars().take(120).collect();
                            eprintln!("  [{}] {:?} .{} = {}", label, nid, key, preview);
                            found += 1;
                        }
                    }
                }
            }
        }
        if found > 0 {
            eprintln!("  → {} matches in {} nodes\n", found, nodes.len());
        }
    }

    // Also scan every single node regardless of label
    eprintln!("=== Brute force scan: all nodes with 'Thomas' ===");
    let mut total_scanned = 0u64;
    let mut total_found = 0u64;
    // Scan node IDs 0..2000
    for i in 0..2000u64 {
        let nid = NodeId::new(i);
        if let Some(node) = store.get_node(nid) {
            total_scanned += 1;
            for (key, val) in node.properties.iter() {
                if let Some(s) = val.as_str() {
                    if s.contains("Thomas") {
                        let preview: String = s.chars().take(100).collect();
                        let labels: Vec<&str> = node.labels.iter().map(|l| { let s: &str = l.as_ref(); s }).collect();
                        eprintln!("  {:?} labels={:?} .{} = {}", nid, labels, key, preview);
                        total_found += 1;
                    }
                }
            }
        }
    }
    eprintln!("\nScanned {} nodes, found {} with 'Thomas'", total_scanned, total_found);
}
