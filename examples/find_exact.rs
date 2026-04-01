use obrain::ObrainDB;
use obrain_common::types::NodeId;

fn main() {
    let db_path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/elun".to_string());
    let db = ObrainDB::open(&db_path).expect("Failed to open DB");
    let store = db.store();

    let needle = "Je m'appelle";
    eprintln!("Searching ALL nodes for exact '{needle}' in /tmp/elun...\n");
    
    for i in 0..2000u64 {
        let nid = NodeId::new(i);
        if let Some(node) = store.get_node(nid) {
            for (key, val) in node.properties.iter() {
                if let Some(s) = val.as_str() {
                    if s.contains(needle) || s.contains("développeur Rust") || s.contains("developpeur Rust") {
                        let labels: Vec<&str> = node.labels.iter().map(|l| { let r: &str = l.as_ref(); r }).collect();
                        eprintln!("  {:?} labels={:?} .{} = {}", nid, labels, key, &s[..s.len().min(200)]);
                    }
                }
            }
        }
    }

    // Check all Facts
    eprintln!("\n=== ALL Facts ===");
    for &nid in &store.nodes_by_label("Fact") {
        if let Some(node) = store.get_node(nid) {
            let props: Vec<String> = node.properties.iter()
                .filter(|(k, _)| {
                    let ks = k.to_string();
                    ks == "key" || ks == "value" || ks == "fact_type"
                })
                .map(|(k, v)| {
                    let vs = v.as_str().map(|s| s.chars().take(80).collect::<String>())
                        .unwrap_or_else(|| format!("{:?}", v));
                    format!("{}={}", k, vs)
                })
                .collect();
            eprintln!("  {:?} {}", nid, props.join(" | "));
        }
    }
}
