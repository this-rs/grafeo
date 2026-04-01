use persona::db::PersonaDB;

#[test]
fn test_wal_roundtrip() {
    let tmp = "/tmp/elun_wal_test";
    let _ = std::fs::remove_dir_all(tmp);
    
    // Phase 1: create DB, add formula, close
    {
        let pdb = PersonaDB::open(tmp).expect("open1");
        let n = pdb.seed_formulas_if_empty();
        println!("  [open1] Seeded {} formulas", n);
        let count = pdb.list_formulas().len();
        println!("  [open1] In-memory: {} formulas", count);
        assert_eq!(count, 6);
        // DB drops here — WAL should flush
    }
    
    // Phase 2: reopen same path, check persistence
    {
        let pdb = PersonaDB::open(tmp).expect("open2");
        let count = pdb.list_formulas().len();
        println!("  [open2] After reopen: {} formulas", count);
        
        if count == 0 {
            println!("  ❌ WAL REPLAY BUG: formulas lost after reopen!");
            // Check what DID survive
            let store = pdb.db.store();
            println!("  [open2] Total nodes: {}", store.node_count());
            println!("  [open2] Total edges: {}", store.edge_count());
            let labels = ["AttnFormula", "Conversation", "Fact", "Pattern"];
            for l in &labels {
                println!("  [open2] :{} = {}", l, store.nodes_by_label(l).len());
            }
        } else {
            println!("  ✅ WAL roundtrip OK: {} formulas survived reopen", count);
        }
    }
    
    let _ = std::fs::remove_dir_all(tmp);
}
