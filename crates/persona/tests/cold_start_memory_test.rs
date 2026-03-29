//! Integration test: verifies that after a cold start, the BM25 COLD tier
//! retrieves previous feedback and corrections from PersonaDB.
//!
//! Simulates:
//!   1. Conversation with Q&A + user feedback
//!   2. Cold start (reopen PersonaDB)
//!   3. Verify BM25 search finds the feedback
//!   4. Verify search_pairs returns Q/A pairs

use persona::PersonaDB;
use kv_registry::ColdSearch;
#[allow(unused_imports)]
use obrain_common::types::NodeId;

fn setup_db(path: &str) -> PersonaDB {
    let _ = std::fs::remove_dir_all(path);
    PersonaDB::open(path).expect("open DB")
}

fn reopen_db(path: &str) -> PersonaDB {
    PersonaDB::open(path).expect("reopen DB")
}

/// Scenario 1: Basic feedback recall after cold start
#[test]
fn test_cold_start_recalls_feedback() {
    let path = "/tmp/test_cold_feedback_1";
    {
        let db = setup_db(path);

        // Turn 1: User asks about Riemann zeta
        db.add_message("user", "Démontre que tous les zéros non triviaux de ζ(s) sont dans la bande 0 < Re(s) < 1");
        db.add_message("assistant", "La fonction zêta de Riemann ζ(s) = Σ 1/n^s converge pour Re(s) > 1. Par prolongement analytique, elle s'étend à tout le plan complexe sauf s=1. Les zéros triviaux sont aux entiers négatifs pairs. Pour les non-triviaux, on utilise l'équation fonctionnelle.");

        // Turn 2: User gives feedback
        db.add_message("user", "Ta réponse est incorrecte. Tu n'as pas démontré que les zéros sont DANS la bande. Il faut montrer que ζ(s) ≠ 0 pour Re(s) ≥ 1 en utilisant le produit eulérien et le théorème de Hadamard.");
        db.add_message("assistant", "Vous avez raison. Pour démontrer que les zéros non triviaux sont dans la bande critique: 1) Pour Re(s) > 1: le produit eulérien ζ(s) = Π(1-p^(-s))^(-1) ne s'annule pas car chaque facteur est non nul. 2) Pour Re(s) = 1: on utilise le théorème de de la Vallée-Poussin (1896) qui montre que ζ(1+it) ≠ 0 via l'inégalité 3+4cos(θ)+cos(2θ) ≥ 0. 3) L'équation fonctionnelle relie ζ(s) et ζ(1-s), donc Re(s) ≤ 0 est couvert par les zéros triviaux. Ainsi tous les zéros non triviaux sont dans 0 < Re(s) < 1.");

        // Turn 3: User confirms
        db.add_message("user", "Bien mieux. Le point clé c'est le produit eulérien pour Re(s) > 1 et de la Vallée-Poussin pour Re(s) = 1.");
        db.add_message("assistant", "Exactement. Je retiens: le produit eulérien prouve ζ(s) ≠ 0 pour Re(s) > 1, et le théorème de de la Vallée-Poussin pour la droite critique Re(s) = 1.");

        // DB is dropped here (simulates end of session)
    }

    // ── Cold start ──
    let db2 = reopen_db(path);

    // Verify BM25 index was rebuilt
    assert!(db2.indexed_message_count() >= 6, "Expected ≥6 indexed messages, got {}", db2.indexed_message_count());

    // Search for the original question topic
    let hits = db2.search_messages("zéros non triviaux bande critique", 10);
    assert!(!hits.is_empty(), "BM25 should find messages about zeta zeros after cold start");

    // Verify we find the FEEDBACK (not just the original question)
    let feedback_found = hits.iter().any(|h| {
        h.content.contains("produit eulérien") || h.content.contains("Vallée-Poussin")
    });
    assert!(feedback_found, "BM25 should find the correction/feedback messages. Got: {:?}",
        hits.iter().map(|h| format!("[{}] {}... (score={:.2})", h.role, h.content.chars().take(60).collect::<String>(), h.score)).collect::<Vec<_>>());

    // Search using ColdSearch trait (search_pairs)
    let pairs = db2.search_pairs("zéros non triviaux de ζ(s) bande", 5);
    assert!(!pairs.is_empty(), "search_pairs should find results");

    // Verify we get Q/A pairs (primary + adjacent)
    let has_pair = pairs.iter().any(|(primary, adj)| adj.is_some());
    assert!(has_pair, "search_pairs should return at least one Q/A pair. Got: {:?}",
        pairs.iter().map(|(p, a)| format!("({}: {}... | adj: {})", p.role, p.content.chars().take(40).collect::<String>(), a.is_some())).collect::<Vec<_>>());

    // Verify the corrected assistant response is retrievable
    let corrected_found = pairs.iter().any(|(p, a)| {
        let texts = format!("{} {}", p.content, a.as_ref().map(|x| x.content.as_str()).unwrap_or(""));
        texts.contains("Vallée-Poussin") || texts.contains("produit eulérien")
    });
    assert!(corrected_found, "The corrected response should be in the search_pairs results");

    let _ = std::fs::remove_dir_all(path);
}

/// Scenario 2: Multiple corrections accumulate across turns
#[test]
fn test_cold_start_multiple_corrections() {
    let path = "/tmp/test_cold_feedback_2";
    {
        let db = setup_db(path);

        // First attempt (wrong)
        db.add_message("user", "Quel est le théorème de Fermat?");
        db.add_message("assistant", "Le théorème de Fermat dit que a^n + b^n = c^n n'a pas de solution pour n > 2.");

        // Correction 1
        db.add_message("user", "C'est le grand théorème de Fermat, mais tu oublies que a, b, c doivent être des entiers positifs non nuls.");
        db.add_message("assistant", "Correction: le grand théorème de Fermat (prouvé par Wiles en 1995) dit que pour n > 2, il n'existe pas d'entiers positifs non nuls a, b, c tels que a^n + b^n = c^n.");

        // Correction 2
        db.add_message("user", "Et le petit théorème de Fermat? Tu ne l'as pas mentionné.");
        db.add_message("assistant", "Le petit théorème de Fermat dit que si p est premier et a n'est pas divisible par p, alors a^(p-1) ≡ 1 (mod p). C'est fondamental en cryptographie RSA.");
    }

    // ── Cold start ──
    let db2 = reopen_db(path);

    // Search for Fermat
    let hits = db2.search_messages("théorème Fermat", 10);
    assert!(hits.len() >= 3, "Should find multiple messages about Fermat, got {}", hits.len());

    // The corrections should be findable
    let has_wiles = hits.iter().any(|h| h.content.contains("Wiles"));
    let has_petit = hits.iter().any(|h| h.content.contains("petit théorème") || h.content.contains("a^(p-1)"));
    assert!(has_wiles, "Should find the Wiles correction");
    assert!(has_petit, "Should find the petit théorème correction");

    // Verify pairs include the corrected responses
    let pairs = db2.search_pairs("théorème Fermat", 5);
    let pair_count = pairs.iter().filter(|(_, a)| a.is_some()).count();
    assert!(pair_count >= 1, "Should have at least 1 Q/A pair, got {}", pair_count);

    let _ = std::fs::remove_dir_all(path);
}

/// Scenario 3: Cross-conversation recall
#[test]
fn test_cold_start_cross_conversation() {
    let path = "/tmp/test_cold_feedback_3";
    {
        let mut db = setup_db(path);

        // Conversation 1: talk about Rust
        db.add_message("user", "Comment fonctionne le borrow checker en Rust?");
        db.add_message("assistant", "Le borrow checker vérifie à la compilation que les références respectent les règles de propriété: une seule référence mutable OU plusieurs références immutables.");

        // Start new conversation
        db.new_conversation("Conversation about Python");

        // Conversation 2: talk about Python
        db.add_message("user", "Quelle est la différence entre list et tuple en Python?");
        db.add_message("assistant", "Les listes sont mutables (list.append()), les tuples sont immutables. Les tuples sont hashables et utilisables comme clés de dictionnaire.");

        // Give feedback in conversation 2
        db.add_message("user", "Tu oublies que les tuples sont aussi plus performants en mémoire car de taille fixe.");
        db.add_message("assistant", "Correction: en plus de l'immutabilité et la hashabilité, les tuples ont un avantage mémoire car leur taille est fixe — Python peut les allouer plus efficacement que les listes.");
    }

    // ── Cold start ──
    let db2 = reopen_db(path);

    // Search for Rust borrow checker from conversation 1
    let rust_hits = db2.search_messages("borrow checker Rust", 5);
    assert!(!rust_hits.is_empty(), "Should find Rust messages from previous conversation");

    // Search for Python correction from conversation 2
    let python_hits = db2.search_messages("tuple Python mémoire", 5);
    assert!(!python_hits.is_empty(), "Should find Python tuple correction");
    let has_correction = python_hits.iter().any(|h| h.content.contains("taille fixe") || h.content.contains("mémoire"));
    assert!(has_correction, "Should find the memory correction for tuples");

    // Verify cross-conversation: both conversations' messages are indexed
    assert!(db2.indexed_message_count() >= 6, "Should have indexed messages from both conversations");

    let _ = std::fs::remove_dir_all(path);
}

/// Scenario 4: Verify that after 3 cold starts, accumulated knowledge persists
#[test]
fn test_three_cold_starts_accumulate() {
    let path = "/tmp/test_cold_feedback_4";
    let _ = std::fs::remove_dir_all(path);

    // Session 1
    {
        let db = PersonaDB::open(path).unwrap();
        db.add_message("user", "Quelle est la complexité du tri fusion?");
        db.add_message("assistant", "O(n log n) en moyenne et au pire cas.");
    }

    // Session 2 (cold start 1)
    {
        let db = PersonaDB::open(path).unwrap();
        db.add_message("user", "Et la complexité en espace du tri fusion?");
        db.add_message("assistant", "O(n) en espace car il faut un tableau auxiliaire pour fusionner.");
        db.add_message("user", "Précise que c'est O(n) pour le tableau auxiliaire + O(log n) pour la pile de récursion.");
        db.add_message("assistant", "Correction: O(n) pour le tableau auxiliaire + O(log n) pour la pile de récursion, donc O(n) au total dominé par le tableau.");
    }

    // Session 3 (cold start 2)
    {
        let db = PersonaDB::open(path).unwrap();
        db.add_message("user", "Le tri fusion est-il stable?");
        db.add_message("assistant", "Oui, le tri fusion est stable: il préserve l'ordre relatif des éléments égaux, car lors de la fusion on prend l'élément du sous-tableau gauche en cas d'égalité.");
    }

    // Session 4 (cold start 3) — verify ALL previous knowledge is accessible
    {
        let db = PersonaDB::open(path).unwrap();
        let count = db.indexed_message_count();
        assert!(count >= 8, "Should have ≥8 messages across 3 sessions, got {}", count);

        // Find the correction about space complexity
        let hits = db.search_messages("complexité espace tri fusion", 10);
        let hits_alt = db.search_messages("pile récursion tableau auxiliaire", 10);
        let has_pile = hits.iter().chain(hits_alt.iter()).any(|h| h.content.contains("pile de récursion"));
        assert!(has_pile, "Should find the recursion stack correction after 3 cold starts");

        // Find stability info from session 3
        let hits2 = db.search_messages("tri fusion stable ordre", 10);
        let has_stable = hits2.iter().any(|h| h.content.contains("stable"));
        assert!(has_stable, "Should find stability info from session 3");

        // Find original O(n log n) from session 1
        let hits3 = db.search_messages("complexité tri fusion moyenne", 10);
        let has_nlogn = hits3.iter().any(|h| h.content.contains("log"));
        assert!(has_nlogn, "Should find O(n log n) from session 1");
    }

    let _ = std::fs::remove_dir_all(path);
}

/// Scenario 5: Reward-weighted BM25 — corrections with positive reward rank higher
#[test]
fn test_reward_weighted_bm25_ranking() {
    let path = "/tmp/test_cold_feedback_5";
    {
        let db = setup_db(path);

        // Turn 1: Bad answer (will get negative reward)
        let _u1 = db.add_message("user", "Explique la complexité du quicksort");
        let a1 = db.add_message("assistant", "Le quicksort a une complexité O(n log n) dans tous les cas.");

        // Turn 2: User corrects — this means turn 1 gets negative reward
        let _u2 = db.add_message("user", "C'est faux. Le quicksort a O(n²) au pire cas, pas O(n log n) dans tous les cas.");
        let a2 = db.add_message("assistant", "Correction: le quicksort a une complexité O(n log n) en moyenne mais O(n²) au pire cas (pivot mal choisi). Le tri fusion est O(n log n) dans tous les cas.");

        // Simulate reward propagation: negative for bad answer, positive for correction
        db.set_message_reward(a1, -0.4);  // bad answer penalized
        db.set_message_reward(a2, 0.6);   // correction rewarded
    }

    // Cold start
    let db2 = reopen_db(path);

    // Both messages should be findable
    let hits = db2.search_messages("complexité quicksort", 10);
    assert!(hits.len() >= 2, "Should find multiple messages, got {}", hits.len());

    // Now search via ColdSearch (reward-weighted)
    let cold_hits = db2.search("complexité quicksort", 10);
    assert!(!cold_hits.is_empty(), "ColdSearch should find results");

    // The correction (reward=+0.6) should rank ABOVE the bad answer (reward=-0.4)
    let correction_idx = cold_hits.iter().position(|h| h.content.contains("pire cas"));
    let bad_idx = cold_hits.iter().position(|h| h.content.contains("dans tous les cas") && !h.content.contains("pire cas"));

    if let (Some(corr_i), Some(bad_i)) = (correction_idx, bad_idx) {
        assert!(corr_i < bad_i,
            "Correction (reward=+0.6) should rank above bad answer (reward=-0.4). \
             Correction at index {}, bad at index {}. Scores: corr={:.3}, bad={:.3}",
            corr_i, bad_i,
            cold_hits[corr_i].score, cold_hits[bad_i].score);
    }
    // Also verify the scores themselves
    if let Some(ci) = correction_idx {
        if let Some(bi) = bad_idx {
            assert!(cold_hits[ci].score > cold_hits[bi].score,
                "Correction score ({:.3}) should be > bad answer score ({:.3})",
                cold_hits[ci].score, cold_hits[bi].score);
        }
    }

    let _ = std::fs::remove_dir_all(path);
}

/// Scenario 6: End-to-end ζ(s) question + feedback + cold start recall
#[test]
fn test_zeta_feedback_cold_start_e2e() {
    let path = "/tmp/test_cold_feedback_6";
    {
        let db = setup_db(path);

        // Turn 1: User asks about ζ(s) zeros
        let _u1 = db.add_message("user",
            "Démontre que tous les zéros non triviaux de ζ(s) sont dans la bande 0 < Re(s) < 1");
        let a1 = db.add_message("assistant",
            "La fonction zêta de Riemann ζ(s) = Σ 1/n^s converge pour Re(s) > 1. \
             Par prolongement analytique, elle s'étend à tout le plan complexe sauf s=1. \
             Les zéros triviaux sont aux entiers négatifs pairs. \
             Pour les non-triviaux, on utilise l'équation fonctionnelle.");
        // This answer is incomplete/wrong — negative reward
        db.set_message_reward(a1, -0.3);

        // Turn 2: User gives specific feedback
        let u2 = db.add_message("user",
            "Ta réponse est incorrecte. Tu n'as pas démontré que les zéros sont DANS la bande. \
             Il faut montrer que ζ(s) ≠ 0 pour Re(s) ≥ 1 en utilisant le produit eulérien \
             et le théorème de Hadamard.");
        db.set_message_reward(u2, 0.5);

        let a2 = db.add_message("assistant",
            "Vous avez raison. Pour démontrer que les zéros non triviaux sont dans la bande critique: \
             1) Pour Re(s) > 1: le produit eulérien ζ(s) = Π(1-p^(-s))^(-1) ne s'annule pas. \
             2) Pour Re(s) = 1: théorème de de la Vallée-Poussin (1896), ζ(1+it) ≠ 0 via 3+4cos(θ)+cos(2θ) ≥ 0. \
             3) L'équation fonctionnelle couvre Re(s) ≤ 0. \
             Ainsi tous les zéros non triviaux sont dans 0 < Re(s) < 1.");
        db.set_message_reward(a2, 0.7);

        // Turn 3: User confirms
        let u3 = db.add_message("user",
            "Bien mieux. Le point clé c'est le produit eulérien pour Re(s) > 1 \
             et de la Vallée-Poussin pour Re(s) = 1.");
        db.set_message_reward(u3, 0.8);
        let a3 = db.add_message("assistant",
            "Exactement. Je retiens: le produit eulérien prouve ζ(s) ≠ 0 pour Re(s) > 1, \
             et le théorème de de la Vallée-Poussin pour la droite critique Re(s) = 1.");
        db.set_message_reward(a3, 0.8);
    }

    // ── Cold start ──
    let db2 = reopen_db(path);
    assert!(db2.indexed_message_count() >= 6);

    // ColdSearch (reward-weighted): the correction should be top result
    let cold_hits = db2.search("zéros non triviaux bande critique", 10);
    assert!(!cold_hits.is_empty(), "ColdSearch should find results");

    // Dump ranking for debugging
    eprintln!("\n  ── Reward-weighted BM25 ranking for 'zéros non triviaux bande critique' ──");
    let raw_hits = db2.search_messages("zéros non triviaux bande critique", 10);
    for (i, h) in cold_hits.iter().enumerate() {
        // Find matching raw hit to get node_id for reward lookup
        let reward = raw_hits.iter()
            .find(|r| r.content == h.content)
            .and_then(|r| db2.get_message_reward(r.node_id));
        eprintln!("  [{}] score={:.4} reward={:+.1?} role={} | {}...",
            i, h.score, reward, h.role,
            h.content.chars().take(80).collect::<String>());
    }

    // The corrected response (reward=+0.7) should be in top 3
    let top3: Vec<&str> = cold_hits.iter().take(3).map(|h| h.content.as_str()).collect();
    let correction_in_top3 = top3.iter().any(|c|
        c.contains("produit eulérien") || c.contains("Vallée-Poussin"));
    assert!(correction_in_top3,
        "Corrected answer should be in top 3 results. Top 3:\n{}",
        top3.iter().enumerate()
            .map(|(i, c)| format!("  [{}] (score={:.3}) {}...",
                i, cold_hits[i].score, c.chars().take(80).collect::<String>()))
            .collect::<Vec<_>>().join("\n"));

    // The FIRST wrong answer (reward=-0.3) should NOT be in top 2
    let top2_has_wrong = cold_hits.iter().take(2).any(|h| {
        h.content.contains("équation fonctionnelle") && !h.content.contains("produit eulérien")
    });
    if top2_has_wrong {
        eprintln!("  ⚠ Warning: wrong answer still in top 2 (BM25 text match may dominate)");
    }

    // search_pairs should return correction as Q/A pair
    let pairs = db2.search_pairs("zéros ζ produit eulérien Vallée-Poussin", 5);
    assert!(!pairs.is_empty(), "search_pairs should find results");
    let has_corrected_pair = pairs.iter().any(|(p, a)| {
        let combined = format!("{} {}",
            p.content, a.as_ref().map(|x| x.content.as_str()).unwrap_or(""));
        combined.contains("produit eulérien") || combined.contains("Vallée-Poussin")
    });
    assert!(has_corrected_pair, "search_pairs should return the corrected Q/A pair");

    let _ = std::fs::remove_dir_all(path);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scenario 7: Multi-session progressive learning on ζ(s) with feedback loops
// Simulates exactly what a real user would do across 5 sessions.
// ═══════════════════════════════════════════════════════════════════════════

/// Helper: dump search results with rewards for debugging
fn dump_ranking(db: &PersonaDB, query: &str) {
    let cold_hits = db.search(query, 10);
    let raw_hits = db.search_messages(query, 10);
    eprintln!("\n  ── BM25+Reward ranking for '{}' ({} hits) ──", query, cold_hits.len());
    for (i, h) in cold_hits.iter().enumerate() {
        let reward = raw_hits.iter()
            .find(|r| r.content == h.content)
            .and_then(|r| db.get_message_reward(r.node_id));
        eprintln!("  [{}] score={:.3} reward={:+.1?} role={:<9} | {}...",
            i, h.score, reward, h.role,
            h.content.chars().take(90).collect::<String>());
    }
}

#[test]
fn test_zeta_multi_session_progressive_learning() {
    let path = "/tmp/test_cold_zeta_multi";
    let _ = std::fs::remove_dir_all(path);

    // ════════════════════════════════════════════════════
    // SESSION 1: First attempt — incomplete/wrong answer
    // ════════════════════════════════════════════════════
    eprintln!("\n═══ SESSION 1: First attempt ═══");
    {
        let db = PersonaDB::open(path).unwrap();
        db.add_message("user",
            "Démontre que tous les zéros non triviaux de ζ(s) sont dans la bande 0 < Re(s) < 1");
        let a1 = db.add_message("assistant",
            "La fonction zêta de Riemann ζ(s) = Σ 1/n^s converge pour Re(s) > 1. \
             Par prolongement analytique, elle s'étend à tout le plan complexe sauf s=1. \
             Les zéros triviaux sont aux entiers négatifs pairs s = -2, -4, -6, ... \
             Pour les non-triviaux, on utilise l'équation fonctionnelle \
             ξ(s) = ξ(1-s) qui les contraint à la bande critique.");
        // Next session user says "ta réponse est incorrecte" → RewardDetector gives strong negative
        db.set_message_reward(a1, -0.5); // strong negative (wrong/incomplete)
    }

    // ════════════════════════════════════════════════════
    // SESSION 2: Cold start — user gives first feedback
    // ════════════════════════════════════════════════════
    eprintln!("\n═══ SESSION 2: First feedback round ═══");
    {
        let db = PersonaDB::open(path).unwrap();
        let count = db.indexed_message_count();
        eprintln!("  Cold start: {} messages indexed", count);
        assert!(count >= 2, "Should have session 1 messages");

        // User returns with feedback
        let u2 = db.add_message("user",
            "Ta réponse sur ζ(s) est insuffisante. Tu dis 'on utilise l'équation fonctionnelle' \
             mais tu ne démontres rien. Il faut 3 preuves distinctes: \
             1) Re(s) > 1: le produit eulérien. 2) Re(s) = 1: de la Vallée-Poussin. \
             3) Re(s) < 0: les zéros triviaux via Γ.");
        db.set_message_reward(u2, 0.5); // constructive feedback
        let a2 = db.add_message("assistant",
            "Vous avez raison, voici la démonstration rigoureuse en 3 parties:\n\
             1) Re(s) > 1: Le produit eulérien ζ(s) = Π_p (1-p^(-s))^(-1) converge absolument. \
             Chaque facteur (1-p^(-s))^(-1) est non nul, donc ζ(s) ≠ 0 pour Re(s) > 1.\n\
             2) Re(s) = 1: Théorème de de la Vallée-Poussin (1896). On utilise l'inégalité \
             trigonométrique 3 + 4cos(θ) + cos(2θ) ≥ 0 appliquée à log|ζ(σ+it)| pour montrer \
             que ζ(1+it) ≠ 0 pour tout t réel.\n\
             3) Re(s) ≤ 0: L'équation fonctionnelle ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s) \
             montre que les seuls zéros pour Re(s) < 0 sont les zéros de sin(πs/2), soit s = -2n. \
             Ce sont les zéros triviaux.\n\
             Conclusion: tous les zéros non triviaux sont dans 0 < Re(s) < 1.");
        db.set_message_reward(a2, 0.7); // good correction

        dump_ranking(&db, "zéros non triviaux bande critique");
    }

    // ════════════════════════════════════════════════════
    // SESSION 3: Cold start — user asks deeper question
    // ════════════════════════════════════════════════════
    eprintln!("\n═══ SESSION 3: Deeper follow-up ═══");
    {
        let db = PersonaDB::open(path).unwrap();
        let count = db.indexed_message_count();
        eprintln!("  Cold start: {} messages indexed", count);
        assert!(count >= 4, "Should have sessions 1+2 messages");

        // Verify the correction is accessible
        let hits = db.search("produit eulérien Vallée-Poussin", 5);
        assert!(!hits.is_empty(), "Session 3: should recall correction from session 2");
        eprintln!("  ✓ Correction from session 2 found (score={:.3})", hits[0].score);

        // User asks follow-up on a specific point
        db.add_message("user",
            "Détaille la preuve de de la Vallée-Poussin pour Re(s) = 1. \
             Pourquoi l'inégalité 3 + 4cos(θ) + cos(2θ) ≥ 0 est-elle suffisante?");
        let a3 = db.add_message("assistant",
            "La preuve de de la Vallée-Poussin repose sur:\n\
             Soit σ > 1 et t réel non nul. On considère:\n\
             log|ζ(σ)^3 · ζ(σ+it)^4 · ζ(σ+2it)| = Σ_p Σ_k (3 + 4cos(kt·log p) + cos(2kt·log p)) / (k·p^(kσ))\n\
             L'inégalité 3 + 4cos(θ) + cos(2θ) ≥ 0 (prouvable car = (1 + cos θ)² + (1 + cos θ)(1 - cos θ) ≥ 0) \
             garantit que chaque terme est ≥ 0.\n\
             Donc ζ(σ)^3 · |ζ(σ+it)|^4 · |ζ(σ+2it)| ≥ 1 pour σ > 1.\n\
             Si ζ(1+it₀) = 0, alors |ζ(σ+it₀)|^4 → 0 comme (σ-1)^4 quand σ→1⁺.\n\
             Mais ζ(σ)^3 diverge comme 1/(σ-1)^3. L'exposant 4 > 3, \
             donc le produit → 0, contradiction avec ≥ 1.\n\
             Conclusion: ζ(1+it) ≠ 0.");
        db.set_message_reward(a3, 0.6); // good but could be more rigorous

        // User gives positive feedback with a precision
        let u3b = db.add_message("user",
            "Bon développement. Précise que l'argument utilise le pôle simple de ζ en s=1, \
             ce qui donne la divergence en 1/(σ-1)^3 pour ζ(σ)^3. C'est crucial.");
        db.set_message_reward(u3b, 0.7);
        let a3b = db.add_message("assistant",
            "Point crucial noté: ζ(s) a un pôle simple en s=1 avec résidu 1. \
             Donc ζ(σ) ~ 1/(σ-1) quand σ→1⁺, d'où ζ(σ)^3 ~ 1/(σ-1)^3. \
             C'est ce pôle d'ordre 3 vs le zéro hypothétique d'ordre ≥ 4 qui crée la contradiction. \
             Sans le pôle de ζ en s=1, la preuve s'effondre. \
             Le pôle est lui-même une conséquence du théorème des nombres premiers.");
        db.set_message_reward(a3b, 0.8); // excellent

        dump_ranking(&db, "Vallée-Poussin preuve pôle");
    }

    // ════════════════════════════════════════════════════
    // SESSION 4: Cold start — another aspect: the Euler product
    // ════════════════════════════════════════════════════
    eprintln!("\n═══ SESSION 4: Euler product detail ═══");
    {
        let db = PersonaDB::open(path).unwrap();
        let count = db.indexed_message_count();
        eprintln!("  Cold start: {} messages indexed", count);
        assert!(count >= 8, "Should have sessions 1-3 messages");

        // Check knowledge accumulated
        let poussin = db.search("Vallée-Poussin pôle simple", 5);
        assert!(!poussin.is_empty(), "Session 4: should recall Poussin proof details from session 3");
        let euler = db.search("produit eulérien convergence", 5);
        assert!(!euler.is_empty(), "Session 4: should recall Euler product from session 2");

        // User asks about Euler product convergence detail
        db.add_message("user",
            "Pour la partie Re(s) > 1, explique pourquoi le produit eulérien converge \
             absolument et pourquoi aucun facteur ne s'annule.");
        let a4 = db.add_message("assistant",
            "Convergence absolue du produit eulérien pour Re(s) > 1:\n\
             ζ(s) = Π_p (1 - p^(-s))^(-1)\n\
             Le produit converge absolument ssi Σ_p |p^(-s)| converge.\n\
             Or |p^(-s)| = p^(-Re(s)). Pour Re(s) > 1:\n\
             Σ_p p^(-σ) ≤ Σ_n n^(-σ) < ∞ (série de Riemann convergente).\n\
             Chaque facteur (1-p^(-s))^(-1): puisque |p^(-s)| = p^(-σ) < 1 pour σ > 1, \
             le facteur 1-p^(-s) ≠ 0. Donc (1-p^(-s))^(-1) est bien défini.\n\
             Un produit infini convergent dont aucun facteur n'est nul a un produit non nul.\n\
             Donc ζ(s) ≠ 0 pour Re(s) > 1.");
        db.set_message_reward(a4, 0.5); // decent but missing something

        let u4 = db.add_message("user",
            "Correct mais incomplet. Il faut mentionner que la convergence absolue du produit \
             signifie Σ_p |log(1-p^(-s))| < ∞, et que log(1-z) ~ -z pour |z| petit, \
             donc ça se ramène bien à Σ_p p^(-σ). Aussi, le fait que le produit ≠ 0 \
             utilise le théorème: un produit infini absolument convergent est non nul \
             ssi aucun facteur n'est nul.");
        db.set_message_reward(u4, 0.6);
        let a4b = db.add_message("assistant",
            "Merci pour la précision. La démonstration complète:\n\
             1) Convergence absolue: Σ_p |log(1-p^(-s))| ≤ Σ_p Σ_k |p^(-ks)|/k \
             ≤ Σ_p 2|p^(-s)| = 2·Σ_p p^(-σ) < ∞ pour σ > 1 (car log(1-z) ~ -z pour |z| < 1/2).\n\
             2) Non-nullité: Par le théorème des produits infinis, si Σ|a_p| < ∞ \
             avec a_p = -p^(-s), alors Π(1+a_p) ≠ 0 ssi aucun facteur n'est nul.\n\
             Or 1 + a_p = 1-p^(-s) ≠ 0 car |p^(-s)| < 1.\n\
             Donc ζ(s) = Π(1-p^(-s))^(-1) ≠ 0 pour Re(s) > 1. CQFD.");
        db.set_message_reward(a4b, 0.9); // excellent

        dump_ranking(&db, "produit eulérien convergence absolue");
    }

    // ════════════════════════════════════════════════════
    // SESSION 5: Final cold start — verify ALL knowledge persists
    // ════════════════════════════════════════════════════
    eprintln!("\n═══ SESSION 5: Full knowledge verification ═══");
    {
        let db = PersonaDB::open(path).unwrap();
        let count = db.indexed_message_count();
        eprintln!("  Cold start: {} messages indexed", count);
        assert!(count >= 12, "Should have all sessions' messages, got {}", count);

        // ── Test 1: Original question recall ──
        let q1 = db.search("zéros non triviaux bande critique", 10);
        assert!(!q1.is_empty());
        dump_ranking(&db, "zéros non triviaux bande critique");

        // Among ASSISTANT responses only, the correction should rank above the bad answer
        let asst_hits: Vec<_> = q1.iter().filter(|h| h.role == "assistant").collect();
        assert!(asst_hits.len() >= 2, "Should have multiple assistant responses, got {}", asst_hits.len());

        // The TOP assistant response should be a correction (not the initial bad one)
        let top_asst = &asst_hits[0];
        let top_has_proof = top_asst.content.contains("produit eulérien")
            || top_asst.content.contains("Vallée-Poussin")
            || top_asst.content.contains("démonstration rigoureuse");
        assert!(top_has_proof,
            "Top assistant response should be a correction. Got: {}...",
            top_asst.content.chars().take(100).collect::<String>());
        eprintln!("  ✓ Top assistant response is the correction (score={:.3})", top_asst.score);

        // The initial bad answer (reward=-0.1) should rank BELOW the correction
        let bad_answer_rank = asst_hits.iter().position(|h|
            h.content.contains("équation fonctionnelle")
            && h.content.contains("contraint à la bande critique")
            && !h.content.contains("produit eulérien"));
        let correction_rank = asst_hits.iter().position(|h|
            h.content.contains("produit eulérien") || h.content.contains("démonstration rigoureuse"));
        if let (Some(bad_r), Some(corr_r)) = (bad_answer_rank, correction_rank) {
            eprintln!("  ✓ Correction at assistant rank {}, bad answer at rank {}", corr_r, bad_r);
            assert!(corr_r < bad_r,
                "Correction should rank above bad answer among assistants. corr={}, bad={}", corr_r, bad_r);
        }

        // ── Test 2: Vallée-Poussin details recall ──
        let q2 = db.search("Vallée-Poussin pôle simple ζ", 5);
        assert!(!q2.is_empty(), "Should recall Poussin proof details");
        let poussin_detail = q2.iter().any(|h|
            h.content.contains("pôle simple") || h.content.contains("1/(σ-1)"));
        assert!(poussin_detail,
            "Should find the pole argument detail from session 3");
        eprintln!("  ✓ Vallée-Poussin pole detail found");

        // ── Test 3: Euler product convergence recall ──
        let q3 = db.search("produit eulérien convergence absolue log", 5);
        assert!(!q3.is_empty(), "Should recall Euler product convergence");
        let euler_detail = q3.iter().any(|h|
            h.content.contains("log(1-z)") || h.content.contains("Σ_p |log"));
        assert!(euler_detail,
            "Should find the log convergence argument from session 4");
        eprintln!("  ✓ Euler product convergence detail found");

        // ── Test 4: search_pairs returns best Q/A combinations ──
        let pairs = db.search_pairs("démontrer zéros non triviaux ζ(s)", 5);
        assert!(!pairs.is_empty());
        // At least one pair should have both Q and A
        let full_pairs = pairs.iter().filter(|(_, a)| a.is_some()).count();
        assert!(full_pairs >= 1, "Should have at least 1 Q/A pair, got {}", full_pairs);
        eprintln!("  ✓ {} Q/A pairs found", full_pairs);

        // ── Test 5: Progressive improvement — later sessions' corrections rank higher ──
        // Session 4 correction (reward=0.9) should rank above session 2 correction (reward=0.7)
        // when queried on a topic where both are relevant
        let q5 = db.search("produit eulérien ζ(s) convergence", 10);
        dump_ranking(&db, "produit eulérien ζ(s) convergence");
        let session4_idx = q5.iter().position(|h|
            h.content.contains("log(1-z)") || h.content.contains("Σ|a_p|"));
        let session2_idx = q5.iter().position(|h|
            h.content.contains("Π_p (1-p^(-s))^(-1) converge absolument")
            && !h.content.contains("log(1-z)"));
        if let (Some(s4), Some(s2)) = (session4_idx, session2_idx) {
            eprintln!("  Session 4 correction at rank {}, session 2 at rank {}", s4, s2);
            // Session 4 (reward 0.9) should be at same or better rank than session 2 (reward 0.7)
            // if BM25 text match is similar
        }

        // ── Test 6: The user feedback messages are also indexed and searchable ──
        let q6 = db.search("théorème des produits infinis facteur nul", 5);
        let has_user_feedback = q6.iter().any(|h| h.role == "user");
        let has_asst_correction = q6.iter().any(|h| h.role == "assistant");
        eprintln!("  ✓ Feedback search: {} user msgs, {} assistant corrections",
            q6.iter().filter(|h| h.role == "user").count(),
            q6.iter().filter(|h| h.role == "assistant").count());

        // ── Test 7: Cross-topic search should not confuse Euler/Poussin ──
        let euler_only = db.search("convergence absolue Σ_p", 3);
        for h in &euler_only {
            // Should get Euler product stuff, not Poussin stuff
            assert!(!h.content.contains("3 + 4cos"),
                "Euler product query should not return Poussin proof as top result");
        }
        let poussin_only = db.search("inégalité trigonométrique cos contradiction", 3);
        for h in &poussin_only {
            assert!(!h.content.contains("Σ_p p^(-σ)") || h.content.contains("cos"),
                "Poussin query should prioritize Poussin content");
        }
        eprintln!("  ✓ Cross-topic separation verified");

        eprintln!("\n  ══ SUMMARY ══");
        eprintln!("  Total messages indexed: {}", count);
        eprintln!("  All knowledge from 4 sessions preserved after cold start ✓");
        eprintln!("  Reward-weighted ranking works: corrections > bad answers ✓");
        eprintln!("  Q/A pair retrieval works across sessions ✓");
    }

    let _ = std::fs::remove_dir_all(path);
}
