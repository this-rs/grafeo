//! End-to-end integration tests for the substrate crate — records, pages, heap, WAL.

use obrain_substrate::*;

#[test]
fn node_record_layout_roundtrip_via_bytemuck() {
    let nodes = vec![
        NodeRecord {
            label_bitset: 1,
            first_edge_off: U48::from_u64(100),
            first_prop_off: U48::from_u64(200),
            community_id: 5,
            energy: f32_to_q1_15(0.5),
            scar_util_affinity: PackedScarUtilAff::new(10, 20, 30, true).pack(),
            centrality_cached: 0xAAAA,
            flags: node_flags::ENGRAM_SEED | node_flags::IDENTITY,
        };
        1000
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&nodes);
    assert_eq!(bytes.len(), NodeRecord::SIZE * 1000);
    let back: &[NodeRecord] = bytemuck::cast_slice(bytes);
    assert_eq!(back.len(), 1000);
    assert_eq!(back[0], nodes[0]);
    assert_eq!(back[999], nodes[999]);
}

#[test]
fn edge_record_layout_roundtrip() {
    let edges = (0..500u32)
        .map(|i| EdgeRecord {
            src: i,
            dst: i + 1,
            edge_type: (i as u16) % 10,
            weight_u16: f32_to_q1_15(0.3),
            next_from: U48::from_u64((i as u64) * 32),
            next_to: U48::from_u64((i as u64) * 64),
            first_prop_off: U48::from_u64((i as u64) * 128),
            ricci_u8: (i % 256) as u8,
            flags: edge_flags::SYNAPSE_ACTIVE,
            engram_tag: i as u16,
            _pad: [0; 2],
        })
        .collect::<Vec<_>>();
    let bytes: &[u8] = bytemuck::cast_slice(&edges);
    let back: &[EdgeRecord] = bytemuck::cast_slice(bytes);
    assert_eq!(back, edges.as_slice());
}

#[test]
fn property_page_crc_and_cast() {
    let mut p = PropertyPage::new(42);
    p.header.entry_count = 1;
    p.payload[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    p.seal_crc32();
    let bytes: &[u8] = bytemuck::bytes_of(&p);
    assert_eq!(bytes.len(), PAGE_SIZE);
    let p2: &PropertyPage = bytemuck::from_bytes(bytes);
    assert!(p2.verify_crc32());
    assert_eq!(p2.header.node_id, 42);
}

#[test]
fn string_heap_large_workload() {
    let mut h = StringHeap::new();
    let mut refs = Vec::new();
    for i in 0..10_000 {
        let s = format!("entry-{i:06}-with-some-padding");
        let r = h.intern(s.as_bytes());
        refs.push((r, s));
    }
    for ((page_id, off), expected) in refs {
        let got = h.get(page_id, off).expect("heap entry should exist");
        assert_eq!(got, expected.as_bytes());
    }
    // Should spill across many pages.
    assert!(h.pages.len() > 10);
}

#[test]
fn wal_sequence_replay_simulation() {
    // Simulate the hot path: many small records encoded to a buffer, then decoded
    // back in order — this is exactly what replay does.
    let ops = [
        WalPayload::NodeInsert {
            node_id: 1,
            label_bitset: 1,
        },
        WalPayload::NodeInsert {
            node_id: 2,
            label_bitset: 3,
        },
        WalPayload::EdgeInsert {
            edge_id: 1,
            src: 1,
            dst: 2,
            edge_type: 0,
            weight_u16: f32_to_q1_15(0.5),
        },
        WalPayload::EnergyReinforce {
            node_id: 1,
            new_energy: f32_to_q1_15(0.8),
        },
        WalPayload::SynapseReinforce {
            edge_id: 1,
            new_weight: f32_to_q1_15(0.9),
        },
        WalPayload::Checkpoint { at_lsn: 5 },
    ];
    let mut buf = Vec::new();
    for (i, payload) in ops.iter().enumerate() {
        let rec = WalRecord {
            lsn: i as u64 + 1,
            timestamp: 0,
            flags: if matches!(payload, WalPayload::Checkpoint { .. }) {
                WalRecord::FLAG_CHECKPOINT
            } else {
                0
            },
            payload: payload.clone(),
        };
        buf.extend_from_slice(&rec.encode().unwrap());
    }
    let mut cursor = 0;
    let mut replayed = Vec::new();
    while cursor < buf.len() {
        let (rec, used) = WalRecord::decode(&buf[cursor..]).unwrap();
        replayed.push(rec);
        cursor += used;
    }
    assert_eq!(replayed.len(), 6);
    assert!(replayed.last().unwrap().is_checkpoint());
}

#[test]
fn tiers_size_check() {
    assert_eq!(core::mem::size_of::<Tier0>(), 16);
    assert_eq!(core::mem::size_of::<Tier1>(), 64);
    assert_eq!(core::mem::size_of::<Tier2>(), L2_DIM * 2);
}

#[test]
fn meta_header_size() {
    assert_eq!(core::mem::size_of::<MetaHeader>(), META_HEADER_SIZE);
}
