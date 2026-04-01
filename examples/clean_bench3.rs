use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey};

fn main() {
    let db_path = std::env::args().nth(1).expect("Usage: clean_bench3 <db_path>");
    eprintln!("Opening DB at {db_path}");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");

    let bench_patterns: &[&str] = &[
        "Thomas Rivière", "Thomas Riviere", "Marc Dupont",
        "Marc et Thomas", "Thomas et Marc",
        "Je m'appelle Thomas", "Je m'appelle Marc",
        "Liste toutes les villes", "villes mentionnées", "villes mentionnees",
        "Bonjour ! Comment vas-tu", "Comment tu t'appelles", "Quel est mon prénom",
        "j'habite à Lyon", "développeur Rust", "j'adore le vélo",
        "mon chat s'appelle Pixel", "Je travaille chez OVH",
    ];
    let labels = ["Message", "ConvTurn", "Memory", "Fact"];
    let props = ["content", "text", "query_text", "value", "key"];

    let mut to_delete: Vec<(NodeId, String)> = Vec::new();
    let store = db.store();
    for label in &labels {
        for &nid in &store.nodes_by_label(label) {
            if let Some(node) = store.get_node(nid) {
                for prop in &props {
                    if let Some(val) = node.properties.get(&PropertyKey::from(*prop)) {
                        if let Some(s) = val.as_str() {
                            if bench_patterns.iter().any(|pat| s.contains(pat)) {
                                let preview: String = s.chars().take(80).collect();
                                to_delete.push((nid, format!("[{label}] {preview}")));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Also find Fact nodes with benchmark keys
    let fact_pats = ["Thomas", "Marc Dupont", "Lyon", "OVH", "Pixel", "vélo"];
    for &nid in &store.nodes_by_label("Fact") {
        if to_delete.iter().any(|(id, _)| *id == nid) { continue; }
        if let Some(node) = store.get_node(nid) {
            for (key, val) in node.properties.iter() {
                if let Some(s) = val.as_str() {
                    if fact_pats.iter().any(|p| s.contains(p)) {
                        to_delete.push((nid, format!("[Fact] {key}={}", s.chars().take(60).collect::<String>())));
                        break;
                    }
                }
            }
        }
    }

    eprintln!("To delete: {}", to_delete.len());
    for (nid, desc) in &to_delete {
        eprintln!("  {:?} {}", nid, desc);
    }

    let mut deleted = 0;
    for (nid, _) in &to_delete {
        if db.delete_node(*nid) { deleted += 1; }
    }
    eprintln!("Deleted {} nodes.", deleted);

    // Verify
    let store2 = db.store();
    eprintln!("Remaining Messages: {}", store2.nodes_by_label("Message").len());
    let mut still = 0;
    for label in &labels {
        for &nid in &store2.nodes_by_label(label) {
            if let Some(node) = store2.get_node(nid) {
                for prop in &props {
                    if let Some(val) = node.properties.get(&PropertyKey::from(*prop)) {
                        if let Some(s) = val.as_str() {
                            if s.contains("Thomas") || s.contains("Marc Dupont") {
                                still += 1; break;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("Nodes still with Thomas/Marc: {}", still);
}
