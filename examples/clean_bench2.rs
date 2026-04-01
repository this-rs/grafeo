use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey};

fn main() {
    let home = std::env::var("HOME").expect("HOME not set");
    let db_path = format!("{home}/.obrain/personna/elun");
    eprintln!("Opening DB at {db_path}");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");

    // Broader benchmark patterns — includes graph query benchmark messages
    let bench_content_patterns: &[&str] = &[
        "Thomas Rivière",
        "Thomas Riviere",
        "Marc Dupont",
        "Marc et Thomas",
        "Thomas et Marc",
        "Je m'appelle Thomas",
        "Liste toutes les villes",
        "villes mentionnées",
        "villes mentionnees",
        "Bonjour ! Comment vas-tu",
        "Comment tu t'appelles",
        "Quel est mon prénom",
        "j'habite à Lyon",
        "développeur Rust",
        "j'adore le vélo",
        "mon chat s'appelle Pixel",
        "Je travaille chez OVH",
    ];

    let labels_to_scan = ["Message", "ConvTurn", "Memory", "Fact"];
    let props_to_check = ["content", "text", "query_text", "value", "key"];

    let mut to_delete: Vec<(NodeId, String, String)> = Vec::new(); // (nid, label, preview)

    let store = db.store();
    for label in &labels_to_scan {
        let nodes = store.nodes_by_label(label);
        for &nid in &nodes {
            if let Some(node) = store.get_node(nid) {
                let mut matched = false;
                let mut preview = String::new();
                for prop in &props_to_check {
                    if let Some(val) = node.properties.get(&PropertyKey::from(*prop)) {
                        if let Some(s) = val.as_str() {
                            if bench_content_patterns.iter().any(|pat| s.contains(pat)) {
                                preview = s.chars().take(80).collect();
                                matched = true;
                                break;
                            }
                        }
                    }
                }
                if matched {
                    to_delete.push((nid, label.to_string(), preview));
                }
            }
        }
    }

    eprintln!("Benchmark nodes to delete: {}", to_delete.len());
    for (nid, label, preview) in &to_delete {
        eprintln!("  [{label}] {:?} → {}", nid, preview);
    }

    if to_delete.is_empty() {
        eprintln!("Nothing to delete.");
        return;
    }

    drop(store);

    let mut deleted = 0;
    for (nid, _, _) in &to_delete {
        if db.delete_node(*nid) {
            deleted += 1;
        } else {
            eprintln!("  WARN: failed to delete {:?}", nid);
        }
    }
    eprintln!("\nDeleted {} nodes.", deleted);

    // Verify
    let store2 = db.store();
    let remaining = store2.nodes_by_label("Message").len();
    eprintln!("Remaining Message nodes: {}", remaining);

    // Check no Thomas left
    let mut thomas_left = 0;
    for label in &labels_to_scan {
        for &nid in &store2.nodes_by_label(label) {
            if let Some(node) = store2.get_node(nid) {
                for prop in &props_to_check {
                    if let Some(val) = node.properties.get(&PropertyKey::from(*prop)) {
                        if let Some(s) = val.as_str() {
                            if s.contains("Thomas") || s.contains("Marc Dupont") {
                                thomas_left += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("Nodes still containing 'Thomas'/'Marc Dupont': {}", thomas_left);
}
