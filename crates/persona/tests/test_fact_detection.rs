use persona::db::PersonaDB;
use persona::facts::detect_facts_from_graph;

#[test]
fn test_fact_detection_matching() {
    let tmp = std::path::PathBuf::from("/tmp/elun_fact_test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    
    let pdb = PersonaDB::open(tmp.to_str().unwrap()).expect("open");
    pdb.seed_default_patterns();
    
    let store = pdb.db.store();
    let n_patterns = store.nodes_by_label("Pattern").len();
    println!("Patterns seeded: {}", n_patterns);
    
    // Test various inputs
    let test_cases = vec![
        ("Je m'appelle Thomas", "name", "thomas"),
        ("je suis développeur", "identity", "développeur"),
        ("J'habite à Lyon", "city", "lyon"),
        ("j'aime le chocolat", "preference", "le chocolat"),
        ("j'adore la musique", "preference", "la musique"),
        ("je déteste les araignées", "dislike", "les araignées"),
        ("retiens que je travaille chez Google", "memory", "je travaille chez google"),
        ("My name is John", "name", "john"),
        ("I live in Paris", "city", "paris"),
        ("mon nom est Marie", "name", "marie"),
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for (input, expected_key, expected_value) in &test_cases {
        let matches = detect_facts_from_graph(&pdb, input);
        if matches.is_empty() {
            println!("  ✗ MISS: \"{}\" → no match (expected {}={})", input, expected_key, expected_value);
            failed += 1;
        } else {
            let m = &matches[0];
            let ok = m.key == *expected_key && m.value.to_lowercase().contains(expected_value);
            if ok {
                println!("  ✓ HIT:  \"{}\" → {}={}", input, m.key, m.value);
                passed += 1;
            } else {
                println!("  ✗ WRONG: \"{}\" → {}={} (expected {}={})", input, m.key, m.value, expected_key, expected_value);
                failed += 1;
            }
        }
    }
    
    // Test non-matching inputs
    let negatives = vec![
        "Bonjour !",
        "Quelle heure est-il ?",
        "Peux-tu m'expliquer les fractales ?",
        "Merci beaucoup !",
    ];
    for input in &negatives {
        let matches = detect_facts_from_graph(&pdb, input);
        if matches.is_empty() {
            println!("  ✓ SKIP: \"{}\" → correctly no match", input);
            passed += 1;
        } else {
            println!("  ✗ FALSE+: \"{}\" → {}={}", input, matches[0].key, matches[0].value);
            failed += 1;
        }
    }
    
    println!("\n=== Results: {}/{} passed, {} failed ===", passed, passed+failed, failed);
    assert!(failed <= 1, "Too many failures");
    
    let _ = std::fs::remove_dir_all(&tmp);
}
