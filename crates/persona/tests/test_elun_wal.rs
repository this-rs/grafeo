use persona::db::PersonaDB;

#[test]
fn test_elun_wal_roundtrip() {
    let tmp = "/tmp/elun_copy_test";
    let _ = std::fs::remove_dir_all(tmp);
    std::fs::create_dir_all(format!("{}/wal", tmp)).unwrap();
    
    // Copy Elun's WAL
    for entry in std::fs::read_dir("/tmp/elun/wal").unwrap() {
        let entry = entry.unwrap();
        let dest = format!("{}/wal/{}", tmp, entry.file_name().to_str().unwrap());
        std::fs::copy(entry.path(), &dest).unwrap();
        println!("  Copied {} ({} bytes)", entry.file_name().to_str().unwrap(), entry.metadata().unwrap().len());
    }
    
    // Open 1: seed formulas
    {
        let pdb = PersonaDB::open(tmp).expect("open1");
        let before = pdb.list_formulas().len();
        let n = pdb.seed_formulas_if_empty();
        let after = pdb.list_formulas().len();
        println!("  [open1] before={}, seeded={}, after={}", before, n, after);
        let store = pdb.db.store();
        println!("  [open1] nodes={}, edges={}", store.node_count(), store.edge_count());
    }
    
    // Open 2: check survival
    {
        let pdb = PersonaDB::open(tmp).expect("open2");
        let count = pdb.list_formulas().len();
        let store = pdb.db.store();
        println!("  [open2] formulas={}, nodes={}, edges={}", count, store.node_count(), store.edge_count());
        
        if count == 0 {
            println!("  ❌ BUG CONFIRMED: formulas lost on Elun WAL reopen");
        } else {
            println!("  ✅ Formulas survived");
        }
    }
    
    let _ = std::fs::remove_dir_all(tmp);
}
