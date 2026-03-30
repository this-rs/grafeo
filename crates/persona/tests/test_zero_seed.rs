/// Phase 6 — Zero-seed integration test.
/// Verifies that a brand new PersonaDB works without any hardcoded seeds.
use persona::db::PersonaDB;
use persona::reward::RewardDetector;

#[test]
fn test_virgin_persona_zero_seed() {
    let tmp = std::path::PathBuf::from("/tmp/zero_seed_test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let pdb = PersonaDB::open(tmp.to_str().unwrap()).expect("open virgin DB");
    let store = pdb.db.store();

    // Seed formulas (called from main.rs at startup)
    pdb.seed_formulas_if_empty();

    // === Verify: exactly 1 formula (Identity) ===
    let formulas = pdb.list_formulas();
    assert_eq!(formulas.len(), 1, "Should have exactly 1 seed formula");
    assert_eq!(formulas[0].name, "F0-Identity");
    assert_eq!(formulas[0].dsl_json, "\"Identity\"");
    assert_eq!(formulas[0].activation_count, 0);
    assert!(formulas[0].active);
    println!("✓ 1 formula (Identity)");

    // === Verify: 0 Pattern nodes ===
    let patterns = store.nodes_by_label("Pattern");
    assert_eq!(patterns.len(), 0, "Should have 0 patterns (zero-seed)");
    println!("✓ 0 patterns");

    // === Verify: 0 RewardToken nodes ===
    let reward_tokens = store.nodes_by_label("RewardToken");
    assert_eq!(reward_tokens.len(), 0, "Should have 0 reward tokens (zero-seed)");
    println!("✓ 0 reward tokens");

    // === Verify: PersistNet can be initialized ===
    let pnet = persona::PersistNet::new(128);
    assert_eq!(pnet.n_updates, 0);
    println!("✓ PersistNet initialized (n_embd=128)");

    // === Verify: RewardDetector works without token polarity ===
    let mut rd = RewardDetector::new();

    // Turn 1: always returns 0.0
    let s1 = rd.compute_reward(&[1, 2, 3], 1, None, None, None);
    assert_eq!(s1.reward, 0.0, "Turn 1 should always be 0.0");

    // Turn 2: structural signals only
    let s2 = rd.compute_reward(&[4, 5, 6], 2, None, None, Some(1.0));
    assert!(s2.reward.is_finite(), "Reward should be finite");
    assert!(s2.reward >= -1.0 && s2.reward <= 1.0, "Reward should be in [-1, 1]");
    println!("✓ RewardDetector: turn 2 reward = {:.4} (structural only)", s2.reward);

    // Turn 5: engagement bonus kicks in
    let s5 = rd.compute_reward(&[7, 8, 9], 5, None, None, Some(1.0));
    assert!(s5.reward > s2.reward, "Turn 5 should have higher reward than turn 2 (engagement)");
    println!("✓ RewardDetector: turn 5 reward = {:.4} (engagement bonus)", s5.reward);

    // Reformulation detection (same tokens → penalty)
    let s_reform = rd.compute_reward(&[7, 8, 9], 6, None, None, None);
    assert!(s_reform.reward < 0.0, "Reformulation should produce negative reward");
    println!("✓ RewardDetector: reformulation penalty = {:.4}", s_reform.reward);

    // Factual success signal
    let facts = vec![("name".to_string(), "Thomas".to_string())];
    let s_fact = rd.compute_reward(&[10, 11], 7, Some(&facts), Some("Bonjour Thomas !"), None);
    assert!(s_fact.factual_signal > 0.0, "Factual signal should be positive when facts match");
    println!("✓ RewardDetector: factual signal = {:.4}", s_fact.factual_signal);

    println!("\n✅ Zero-seed persona is fully functional!");

    let _ = std::fs::remove_dir_all(&tmp);
}
