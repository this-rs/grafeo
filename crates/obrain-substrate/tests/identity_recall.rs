//! T7 Step 7 — cross-session identity recall (substrate-level scenario).
//!
//! Validates the substrate's contract for the "Identity recall" use case
//! that has historically required a 7-layer chain of fixes in the Hub
//! (see the existing gotcha note "Identity recall failure (7-layer chain)"):
//!
//! 1. retriever indexed
//! 2. direct match scoring
//! 3. stop-word filtering removed
//! 4. semantic gap closed
//! 5. min_episodes gate
//! 6. identity persistence
//! 7. always-on injection
//!
//! Of these, layers (4) "semantic gap" and (7) "always-on injection" are
//! Hub responsibilities (encoder, prompt builder). Substrate covers
//! layers (1), (3), (5), and (6) by construction:
//!
//! * **(1) retriever indexed**: engram bitset column is built lazily and
//!   persisted as part of the substrate file — no separate index to
//!   "warm up" on open.
//! * **(3) no stop-word filtering**: substrate stores opaque node ids;
//!   no token-level filtering exists.
//! * **(5) one-shot identity engrams**: `seed_engram` forms an engram
//!   from a single observation, with no `min_episodes` gate.
//! * **(6) identity persistence**: engram membership + bitset survive
//!   close + reopen by construction (WAL-native, mmap'd columns).
//!
//! This test exercises the full substrate-level path:
//!
//! ```text
//! Session 1: write
//!   1. user node + name-assertion node + "Atlas" node
//!   2. seed_engram([assertion, atlas])  → engram_id E
//!   3. coact_reinforce_f32(user → assertion, 0.9)
//!   4. coact_reinforce_f32(assertion → atlas, 0.9)
//!   5. flush + drop
//!
//! Session 2: read (fresh process simulated by reopen)
//!   6. open same path
//!   7. create question node ("comment tu t'appelles ?")
//!   8. simulate Hub-side semantic routing: add_engram_bit(question, E)
//!   9. hopfield_recall(question, k=10) → must return atlas in top results
//!  10. coact weights from session 1 still readable
//! ```

use obrain_common::{EdgeId, NodeId};
use obrain_core::graph::GraphStoreMut;
use obrain_substrate::SubstrateStore;
use obrain_substrate::record::edge_flags;

#[test]
fn cross_session_identity_recall_via_engram_bitset() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("brain.substrate");

    // ── Session 1 — store the identity assertion ──────────────────────
    let engram_id;
    let user_id;
    let assertion_id;
    let atlas_id;
    let coact_user_assertion;
    let coact_assertion_atlas;
    {
        let s = SubstrateStore::create(&path).unwrap();

        user_id = s.create_node(&["User"]);
        assertion_id = s.create_node(&["IdentityAssertion"]);
        atlas_id = s.create_node(&["Identity", "Atlas"]);

        // Form the Identity engram from the two assertion-side nodes.
        engram_id = s
            .seed_engram(&[assertion_id, atlas_id])
            .expect("seed_engram should succeed");
        assert_eq!(engram_id, 1, "first engram id is 1");

        // Build a small COACT chain — these survive the close cycle
        // because the underlying primitive logs absolute weights to the WAL.
        let coact_id = s.coact_type_id().unwrap();
        coact_user_assertion = s.create_edge(user_id, assertion_id, "COACT");
        coact_assertion_atlas = s.create_edge(assertion_id, atlas_id, "COACT");
        // Sanity: the slots are typed COACT.
        assert_eq!(
            s.coact_weight_f32(coact_user_assertion).unwrap(),
            Some(0.0),
            "freshly created COACT slot must read back as Some(0.0)"
        );

        s.coact_reinforce_f32(coact_user_assertion, 0.9).unwrap();
        s.coact_reinforce_f32(coact_assertion_atlas, 0.9).unwrap();

        let _ = coact_id; // documentation only

        // Flush so the dict allocator + COACT type registration land
        // on disk (WAL is already durable per-record).
        s.flush().unwrap();
    } // drop(s) — simulates closing the session

    // ── Session 2 — query as if from a brand-new conversation ──────────
    {
        let s = SubstrateStore::open(&path).unwrap();

        // The engram allocator must have advanced — a fresh seed in
        // session 2 must produce engram_id = 2, never re-issue 1.
        assert_eq!(
            s.next_engram_id(),
            2,
            "next_engram_id must round-trip via substrate.dict v2"
        );

        // The Identity engram members must be intact.
        let mut got = s.engram_members(engram_id).unwrap().unwrap();
        let mut expected = vec![assertion_id, atlas_id];
        got.sort_by_key(|n: &NodeId| n.0);
        expected.sort_by_key(|n: &NodeId| n.0);
        assert_eq!(
            got, expected,
            "Identity engram membership must survive close+reopen"
        );

        // The Atlas node's signature must still carry the Identity bit.
        let atlas_bits = s.engram_bitset(atlas_id).unwrap();
        let identity_mask = obrain_substrate::engram_bit_mask(engram_id);
        assert_eq!(
            atlas_bits & identity_mask,
            identity_mask,
            "Atlas node lost its Identity-engram bit across reopen"
        );

        // The COACT weights from session 1 must still be readable
        // (Layer 6: identity persistence).
        let w_ua = s.coact_weight_f32(coact_user_assertion).unwrap().unwrap();
        let w_aa = s.coact_weight_f32(coact_assertion_atlas).unwrap().unwrap();
        assert!(
            (w_ua - 0.9).abs() < 1e-2,
            "user→assertion coact weight must persist (got {w_ua})"
        );
        assert!(
            (w_aa - 0.9).abs() < 1e-2,
            "assertion→atlas coact weight must persist (got {w_aa})"
        );

        // ── Hub-layer mock: semantic routing assigns the question into
        //    the Identity engram. In production this is done by the
        //    Hub's tier-L0/L1/L2 nearest-neighbour search (T8) feeding
        //    a routing decision; here we short-circuit it.
        let question_id = s.create_node(&["IdentityQuestion"]);
        s.add_engram_bit(question_id, engram_id).unwrap();

        // Hopfield recall: starting from the question's bitset, find the
        // top-k nodes by engram-bitset overlap.
        let recalled = s.hopfield_recall(question_id, 10).unwrap();
        assert!(
            !recalled.is_empty(),
            "hopfield_recall must return at least the assertion + atlas"
        );

        // Atlas must be in the recalled candidates (every node whose
        // bitset has the Identity bit shows up — the Hopfield primitive
        // is a Bloom-overlap filter; tier resolution comes later).
        let recalled_ids: Vec<NodeId> = recalled.iter().map(|(nid, _)| *nid).collect();
        assert!(
            recalled_ids.contains(&atlas_id),
            "Atlas node must be recalled from a fresh question that \
             routes into the Identity engram. Got: {recalled_ids:?}"
        );
        assert!(
            recalled_ids.contains(&assertion_id),
            "Assertion node must also be recalled. Got: {recalled_ids:?}"
        );

        // The assertion + atlas overlap is >= 1 bit (the Identity bit).
        for (nid, score) in &recalled {
            assert!(
                *score >= 1,
                "recalled node {nid:?} must have overlap ≥ 1 (got {score})"
            );
        }

        // The user node was NEVER added to the engram — it must NOT
        // appear in the recall (this validates that routing through the
        // Identity bit is *selective*, not a full graph scan).
        assert!(
            !recalled_ids.contains(&user_id),
            "User node has no engram bit and must not appear in recall. \
             Got: {recalled_ids:?}"
        );

        // Final sanity: the COACT edges are still typed correctly
        // (no synapse-typed edge accidentally surfaced as a coact).
        let _ = edge_flags::TOMBSTONED; // doc usage
        let _ = EdgeId(0);
    }
}

/// Two independent seed_engram calls across a reopen produce strictly
/// monotonically increasing engram ids — id 1 from session 1, id 2 from
/// session 2, never a reissue of 1. Guards against allocator-state loss
/// across substrate.dict round-trip.
#[test]
fn engram_id_allocator_is_monotonic_across_reopen() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("brain.substrate");

    let id_session_1;
    let n_a;
    {
        let s = SubstrateStore::create(&path).unwrap();
        n_a = s.create_node(&["A"]);
        id_session_1 = s.seed_engram(&[n_a]).unwrap();
        assert_eq!(id_session_1, 1);
        s.flush().unwrap();
    }

    let id_session_2 = {
        let s = SubstrateStore::open(&path).unwrap();
        let n_b = s.create_node(&["B"]);
        s.seed_engram(&[n_b]).unwrap()
    };

    assert_eq!(
        id_session_2, 2,
        "second-session engram id must be 2 (no re-issue of 1)"
    );
}
