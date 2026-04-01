use persona::db::PersonaDB;

#[test]
fn test_seed_and_list() {
    // Open in a temp copy to not corrupt the real DB
    let tmp = "/tmp/elun_test_seed";
    let _ = std::fs::remove_dir_all(tmp);
    std::fs::create_dir_all(tmp).unwrap();
    // Copy WAL
    for entry in std::fs::read_dir("/tmp/elun/wal").unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), format!("{}/wal/{}", tmp, entry.file_name().to_str().unwrap()))
            .ok();
    }
    std::fs::create_dir_all(format!("{}/wal", tmp)).unwrap();
    for entry in std::fs::read_dir("/tmp/elun/wal").unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), format!("{}/wal/{}", tmp, entry.file_name().to_str().unwrap()))
            .unwrap();
    }
    if std::fs::metadata("/tmp/elun/projection_net.bin").is_ok() {
        std::fs::copy("/tmp/elun/projection_net.bin", format!("{}/projection_net.bin", tmp)).unwrap();
    }

    let pdb = PersonaDB::open(tmp).expect("Failed to open copy");
    
    let before = pdb.list_formulas().len();
    println!("  Before seed: {} formulas", before);
    
    let seeded = pdb.seed_formulas_if_empty();
    println!("  Seeded: {} formulas", seeded);
    
    let after = pdb.list_formulas();
    println!("  After seed: {} formulas", after.len());
    for f in &after {
        println!("    {} (energy={:.2}, gen={}, active={})", f.name, f.energy, f.generation, f.active);
    }
    
    // Cleanup
    let _ = std::fs::remove_dir_all(tmp);
}
