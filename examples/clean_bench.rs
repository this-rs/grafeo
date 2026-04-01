//! Remove benchmark/test messages from the PersonaDB.
//! Usage: cargo run --release --example clean_bench -- [db_path]

use obrain::ObrainDB;
use obrain_common::types::{NodeId, PropertyKey};

fn main() {
    let db_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").expect("HOME not set");
            format!("{home}/.obrain/personna/elun")
        });

    eprintln!("Opening DB at {db_path}");
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");

    let bench_patterns: &[&str] = &[
        "Je m'appelle Thomas",
        "Je m'appelle Marc Dupont",
        "Marc Dupont",
        "Liste toutes les villes",
        "Bonjour ! Comment vas-tu",
        "Comment tu t'appelles",
        "Quel est mon prénom",
        "j'habite à Lyon",
        "développeur Rust",
        "j'adore le vélo",
        "mon chat s'appelle Pixel",
        "Je travaille chez OVH",
    ];

    let store = db.store();
    let msg_nodes: Vec<NodeId> = store.nodes_by_label("Message");

    eprintln!("Total Message nodes: {}", msg_nodes.len());

    let mut to_delete: Vec<(NodeId, String)> = Vec::new();

    for &nid in &msg_nodes {
        if let Some(node) = store.get_node(nid) {
            let content = node
                .properties
                .get(&PropertyKey::from("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let is_bench = bench_patterns.iter().any(|pat| content.contains(pat));
            if is_bench {
                let preview: String = content.chars().take(80).collect();
                to_delete.push((nid, preview));
            }
        }
    }

    eprintln!("\nBenchmark messages found: {}", to_delete.len());
    for (nid, preview) in &to_delete {
        eprintln!("  [{:?}] {}", nid, preview);
    }

    if to_delete.is_empty() {
        eprintln!("Nothing to delete.");
        return;
    }

    // Also find benchmark assistant responses (paired with benchmark user messages)
    // They typically follow immediately after a benchmark user message
    let mut bench_assistant_msgs: Vec<(NodeId, String)> = Vec::new();
    let assistant_bench_patterns: &[&str] = &[
        "Bonjour Thomas",
        "Enchanté Thomas",
        "prénom est Thomas",
        "vous appelez Thomas",
        "t'appelles Thomas",
        "tu t'appelles Thomas",
        "Marc Dupont",
        "villes mentionnées",
        "villes mentionnees",
    ];
    for &nid in &msg_nodes {
        // Skip if already in to_delete
        if to_delete.iter().any(|(id, _)| *id == nid) {
            continue;
        }
        if let Some(node) = store.get_node(nid) {
            let role = node
                .properties
                .get(&PropertyKey::from("role"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if role != "assistant" {
                continue;
            }
            let content = node
                .properties
                .get(&PropertyKey::from("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if assistant_bench_patterns.iter().any(|pat| content.contains(pat)) {
                let preview: String = content.chars().take(80).collect();
                bench_assistant_msgs.push((nid, preview));
            }
        }
    }
    if !bench_assistant_msgs.is_empty() {
        eprintln!("\nBenchmark assistant responses found: {}", bench_assistant_msgs.len());
        for (nid, preview) in &bench_assistant_msgs {
            eprintln!("  [{:?}] {}", nid, preview);
        }
    }

    // Also delete any Fact nodes that reference benchmark data
    let fact_nodes: Vec<NodeId> = store.nodes_by_label("Fact");
    let mut bench_facts: Vec<(NodeId, String)> = Vec::new();
    let fact_patterns: &[&str] = &[
        "Thomas", "Marc Dupont", "Lyon", "OVH", "Pixel", "vélo",
    ];
    for &nid in &fact_nodes {
        if let Some(node) = store.get_node(nid) {
            let value = node
                .properties
                .get(&PropertyKey::from("value"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let key = node
                .properties
                .get(&PropertyKey::from("key"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let combined = format!("{key}={value}");
            if fact_patterns.iter().any(|p| combined.contains(p)) {
                bench_facts.push((nid, combined));
            }
        }
    }

    if !bench_facts.is_empty() {
        eprintln!("\nBenchmark Facts found: {}", bench_facts.len());
        for (nid, desc) in &bench_facts {
            eprintln!("  [{:?}] {}", nid, desc);
        }
    }

    drop(store); // release borrow before mutations

    // Delete all benchmark nodes
    let mut deleted = 0;
    for (nid, _) in to_delete.iter().chain(bench_assistant_msgs.iter()).chain(bench_facts.iter()) {
        if db.delete_node(*nid) {
            deleted += 1;
        } else {
            eprintln!("  WARN: failed to delete {:?}", nid);
        }
    }

    eprintln!("\nDeleted {} nodes total. DB will persist on drop.", deleted);

    // Verify: count remaining messages
    let store3 = db.store();
    let remaining_msgs = store3.nodes_by_label("Message");
    let remaining = remaining_msgs.len();
    let zeta_count = remaining_msgs
        .iter()
        .filter(|&&nid| {
            store3
                .get_node(nid)
                .and_then(|n| {
                    n.properties
                        .get(&PropertyKey::from("content"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_lowercase())
                })
                .map(|s| s.contains("zeta") || s.contains("riemann"))
                .unwrap_or(false)
        })
        .count();

    eprintln!("\nAfter cleanup:");
    eprintln!("  Remaining Message nodes: {}", remaining);
    eprintln!("  Zeta/Riemann messages: {}", zeta_count);
}
