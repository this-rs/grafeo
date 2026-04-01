use persona::db::PersonaDB;

/// One-shot fix: open Elun, seed formulas, write checkpoint.
/// Run with: cargo test -p persona --test fix_elun_db -- --nocapture
#[test]
fn fix_elun_db() {
    let path = "/tmp/elun";
    
    // Backup WAL first
    let wal_path = format!("{}/wal/wal_00000000.log", path);
    let backup = format!("{}/wal/wal_00000000.log.bak", path);
    if !std::path::Path::new(&backup).exists() {
        std::fs::copy(&wal_path, &backup).expect("backup failed");
        println!("  📋 WAL backed up to .bak");
    }
    
    let pdb = PersonaDB::open(path).expect("open");
    
    // Current state
    let store = pdb.db.store();
    println!("  Before: nodes={}, edges={}, formulas={}", 
        store.node_count(), store.edge_count(), 
        store.nodes_by_label("AttnFormula").len());
    drop(store);
    
    // Seed formulas
    let n = pdb.seed_formulas_if_empty();
    println!("  Seeded {} formulas", n);
    
    // Write checkpoint so next open doesn't replay
    let ckpt_path = format!("{}/wal/checkpoint.meta", path);
    // The checkpoint format for obrain WAL is just the byte offset
    let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    std::fs::write(&ckpt_path, wal_size.to_le_bytes()).expect("write checkpoint");
    println!("  ✅ Checkpoint written at offset {} bytes", wal_size);
    
    // Verify
    let store = pdb.db.store();
    println!("  After: nodes={}, edges={}, formulas={}",
        store.node_count(), store.edge_count(),
        store.nodes_by_label("AttnFormula").len());
    
    let formulas = pdb.list_formulas();
    for f in &formulas {
        println!("    {} (energy={:.2}, active={})", f.name, f.energy, f.active);
    }
}
