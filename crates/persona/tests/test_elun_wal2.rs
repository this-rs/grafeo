use persona::db::PersonaDB;

#[test]
fn test_node_id_conflict() {
    let tmp = "/tmp/elun_copy2";
    let _ = std::fs::remove_dir_all(tmp);
    std::fs::create_dir_all(format!("{}/wal", tmp)).unwrap();
    for entry in std::fs::read_dir("/tmp/elun/wal").unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), format!("{}/wal/{}", tmp, entry.file_name().to_str().unwrap())).unwrap();
    }
    
    // Open and inspect
    let pdb = PersonaDB::open(tmp).expect("open");
    let store = pdb.db.store();
    
    // Find the max node ID
    let all_node_count = store.node_count();
    println!("  Total nodes after replay: {}", all_node_count);
    
    // Try to find AttnFormula by scanning ALL node IDs
    let mut attn_count = 0;
    let formulas_by_label = store.nodes_by_label("AttnFormula");
    println!("  nodes_by_label(AttnFormula): {}", formulas_by_label.len());
    
    // Check some high IDs that might have been the formula IDs
    for nid_raw in 670..700 {
        let nid = obrain_common::types::NodeId(nid_raw);
        if let Some(node) = store.get_node(nid) {
            let labels: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
            let name_prop = node.properties.get(&obrain_common::PropertyKey::from("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            println!("  Node {} labels={:?} name={}", nid_raw, labels, name_prop);
            if labels.contains(&"AttnFormula") {
                attn_count += 1;
            }
        }
    }
    println!("  AttnFormula in range 670-700: {}", attn_count);
    
    let _ = std::fs::remove_dir_all(tmp);
}
