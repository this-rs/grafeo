use persona::db::PersonaDB;
use obrain_common::PropertyKey;

#[test]
fn verify_pattern_migration() {
    // Copy the DB so we don't modify the real one
    let tmp = std::path::PathBuf::from("/tmp/elun_migration_test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::path::Path::new("/tmp/elun");
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), tmp.join(entry.file_name())).unwrap();
        }
    }

    let pdb = PersonaDB::open(tmp.to_str().unwrap()).expect("open");
    let store = pdb.db.store();
    
    let before = store.nodes_by_label("Pattern").len();
    println!("Before migration: {} patterns", before);
    
    // Run seed_default_patterns (now additive)
    pdb.seed_default_patterns();
    
    let after = store.nodes_by_label("Pattern").len();
    println!("After migration: {} patterns (+{})", after, after - before);
    
    // List all patterns
    for &nid in &store.nodes_by_label("Pattern") {
        if let Some(node) = store.get_node(nid) {
            let trigger = node.properties.get(&PropertyKey::from("trigger"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            let key = node.properties.get(&PropertyKey::from("key_template"))
                .and_then(|v| v.as_str()).unwrap_or("?");
            println!("  key={:<20} trigger=\"{}\"", key, trigger);
        }
    }
    
    assert!(after > before, "Should have added new patterns");
    assert!(after >= 40, "Should have at least 40 patterns total");
    
    // Verify key patterns are present
    let triggers: Vec<String> = store.nodes_by_label("Pattern").iter()
        .filter_map(|&nid| {
            store.get_node(nid).and_then(|n| {
                n.properties.get(&PropertyKey::from("trigger"))
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
            })
        })
        .collect();
    
    assert!(triggers.contains(&"je m'appelle ".to_string()), "Missing 'je m'appelle'");
    assert!(triggers.contains(&"je suis ".to_string()), "Missing 'je suis'");
    assert!(triggers.contains(&"j'habite à ".to_string()), "Missing 'j'habite à'");
    assert!(triggers.contains(&"j'aime ".to_string()), "Missing 'j'aime'");
    
    println!("\n✅ All critical patterns present!");
}
