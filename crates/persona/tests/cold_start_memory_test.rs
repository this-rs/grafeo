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
