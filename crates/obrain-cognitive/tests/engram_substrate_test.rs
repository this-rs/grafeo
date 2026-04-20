//! T7 Step 2 — substrate-native EngramStore path.
//!
//! These tests exercise the new `form` / `recall` / `members` primitives and
//! verify that `insert` / `remove` correctly mirror to the substrate
//! side-columns (`substrate.engram_members`, `substrate.engram_bitset`).
//!
//! The JSON-via-CognitiveStorage persistence path is tested elsewhere
//! (`engram::store::tests`). Here we focus on the compact binary mirror.

#![cfg(all(feature = "engram", feature = "substrate"))]

use std::sync::Arc;

use obrain_cognitive::engram::{Engram, EngramStore};
use obrain_common::types::NodeId;
use obrain_substrate::{SubstrateFile, SyncMode, Writer as SubstrateWriter};

fn make_writer() -> (Arc<SubstrateWriter>, tempfile::TempDir) {
    let td = tempfile::tempdir().unwrap();
    let sub = SubstrateFile::create(td.path().join("kb")).unwrap();
    let w = SubstrateWriter::new(sub, SyncMode::Never).unwrap();
    (Arc::new(w), td)
}

#[test]
fn form_allocates_and_persists_members() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer.clone());

    let nids = vec![NodeId(10), NodeId(20), NodeId(30)];
    let eid = store.form(&nids).expect("form should succeed");

    // members() round-trips through the substrate side-table.
    let got = store.members(eid).expect("members() should succeed");
    assert_eq!(got, nids, "members snapshot must equal original ensemble");

    // Each node's bitset has the engram's bit OR-ed in.
    for nid in &nids {
        let bits = writer.engram_bitset(nid.0 as u32).unwrap();
        let eid_u16 = eid.0 as u16;
        let mask = 1u64 << (eid_u16 & 0x3F);
        assert_ne!(
            bits & mask,
            0,
            "bitset for {nid:?} must have bit for engram {eid} set (bits={bits:#018b})"
        );
    }
}

#[test]
fn recall_returns_engrams_containing_node() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer);

    // engram A: {1, 2}
    let a = store.form(&[NodeId(1), NodeId(2)]).unwrap();
    // engram B: {2, 3}
    let b = store.form(&[NodeId(2), NodeId(3)]).unwrap();
    // engram C: {4}
    let c = store.form(&[NodeId(4)]).unwrap();

    // Node 2 is in A and B.
    let got = store.recall(NodeId(2)).unwrap();
    assert!(got.contains(&a));
    assert!(got.contains(&b));
    assert!(!got.contains(&c));
    assert_eq!(got.len(), 2);

    // Node 4 is only in C.
    let got = store.recall(NodeId(4)).unwrap();
    assert_eq!(got, vec![c]);

    // Node 999 is in nothing.
    let got = store.recall(NodeId(999)).unwrap();
    assert!(got.is_empty());
}

#[test]
fn insert_mirrors_to_substrate() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer);

    // `insert` uses a pre-built Engram, whereas `form` is the substrate-only
    // fast path. Both must leave the substrate side-columns consistent.
    let id = store.next_id();
    let engram = Engram::new(id, vec![(NodeId(100), 1.0), (NodeId(200), 0.5)]);
    store.insert(engram);

    let got = store.members(id).unwrap();
    assert_eq!(got, vec![NodeId(100), NodeId(200)]);

    let recalled = store.recall(NodeId(100)).unwrap();
    assert_eq!(recalled, vec![id]);
    let recalled = store.recall(NodeId(200)).unwrap();
    assert_eq!(recalled, vec![id]);
}

#[test]
fn remove_clears_members_but_leaves_bitset_monotone() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer.clone());

    let eid = store.form(&[NodeId(7), NodeId(8)]).unwrap();
    // To get the engram into the cache (so recall can find candidates),
    // we also need to insert a matching Engram — form() writes only to
    // substrate, not to the cache. That's intentional: `form` is the
    // substrate-native primitive; the cognitive layer decides when to
    // materialise a full Engram. For this test we simulate the normal
    // insert+form pairing used by higher-level code.
    store.insert(Engram::new(eid, vec![(NodeId(7), 1.0), (NodeId(8), 1.0)]));

    assert_eq!(store.members(eid).unwrap().len(), 2);
    assert!(store.recall(NodeId(7)).unwrap().contains(&eid));

    // Remove wipes the members directory entry.
    store.remove(eid);
    assert!(
        store.members(eid).unwrap().is_empty(),
        "members snapshot must be cleared after remove"
    );

    // Bitset bits are intentionally left set (monotone-union). But recall
    // now returns an empty vector because verification against the empty
    // members table filters out the stale candidate.
    let eid_u16 = eid.0 as u16;
    let mask = 1u64 << (eid_u16 & 0x3F);
    let bits_after = writer.engram_bitset(7).unwrap();
    assert_eq!(
        bits_after & mask,
        mask,
        "bitset must keep the removed engram's bit (monotone-union)"
    );
    assert!(
        store.recall(NodeId(7)).unwrap().is_empty(),
        "recall must filter out removed engrams via the members table"
    );
}

#[test]
fn form_refuses_when_no_substrate_attached() {
    let store = EngramStore::new(None);
    let err = store.form(&[NodeId(1)]).unwrap_err();
    assert!(
        format!("{err}").contains("no substrate writer attached"),
        "expected attachment error, got: {err}"
    );
}

#[test]
fn form_survives_zero_and_large_ensembles() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer);

    let eid = store.form(&[]).expect("empty ensemble should succeed");
    assert_eq!(store.members(eid).unwrap(), Vec::<NodeId>::new());

    let many: Vec<NodeId> = (1_000..1_100).map(NodeId).collect();
    let eid = store.form(&many).expect("100-node ensemble should succeed");
    assert_eq!(store.members(eid).unwrap(), many);
    // Spot-check a handful of the bitset columns.
    for i in [0usize, 50, 99] {
        let got = store.recall(many[i]).unwrap();
        assert!(got.contains(&eid));
    }
}

#[test]
fn recall_filters_bloom_collisions_via_members_table() {
    let store = EngramStore::new(None);
    let (writer, _td) = make_writer();
    store.attach_substrate(writer);

    // Two engrams whose ids fold to the same bitset slot (engram_id & 0x3F).
    // Force id 1 and id 1+64 = 65 to collide on bit 1.
    // We can't pick ids directly (next_id is monotone from 1), but we can
    // burn ids until we land on 65 for the second engram.
    let e1 = store.form(&[NodeId(42)]).unwrap();
    assert_eq!(e1.0, 1);
    // Burn ids up to 64 so the next form() gets id 65.
    for _ in 0..63 {
        let _ = store.next_id();
    }
    let e65 = store.form(&[NodeId(99)]).unwrap();
    assert_eq!(e65.0, 65);
    // Verify the bit collision.
    assert_eq!(e1.0 as u16 & 0x3F, e65.0 as u16 & 0x3F);

    // Populate the cache so recall has candidates to enumerate.
    // (form() doesn't touch the cache; higher-level code pairs form+insert.)
    store.insert(Engram::new(e1, vec![(NodeId(42), 1.0)]));
    store.insert(Engram::new(e65, vec![(NodeId(99), 1.0)]));

    // Query node 42 — both engrams' bits are candidates (bit 1 in the bitset
    // column for node 42 would match both ids). Only e1 actually contains
    // node 42 — the members table must filter e65 out.
    let got = store.recall(NodeId(42)).unwrap();
    assert_eq!(got, vec![e1], "collision must be resolved by members table");

    let got = store.recall(NodeId(99)).unwrap();
    assert_eq!(got, vec![e65], "collision must be resolved by members table");
}
