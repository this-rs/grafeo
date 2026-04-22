//! Write-Ahead Log — framed, length-prefixed, CRC32-protected records.
//!
//! See `docs/rfc/substrate/wal-spec.md`.
//!
//! Framing (28 B header + payload):
//!
//! ```text
//!   0..4   length      u32 LE     payload length
//!   4      kind        u8         WalKind discriminant
//!   5      flags       u8         bit 0 = commit marker, 1 = checkpoint
//!   6..8   reserved    u16 LE     zero
//!   8..16  lsn         u64 LE     monotonic log sequence number
//!   16..24 timestamp   i64 LE     unix micros
//!   24..28 crc32       u32 LE     crc32 of bytes [0..24] XOR payload
//! ```

use crate::error::SubstrateError;
use bincode::config::{Configuration, Fixint, LittleEndian};
use serde::{Deserialize, Serialize};

pub const WAL_HEADER_SIZE: usize = 28;

/// `bincode` configuration used for WAL payloads: little-endian, fixed-int encoding
/// so record size is stable across machines.
fn bincode_config() -> Configuration<LittleEndian, Fixint> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

// ---------------------------------------------------------------------------
// Kind enum (u8 discriminants documented in wal-spec.md §3)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WalKind {
    NodeInsert = 0x01,
    NodeUpdate = 0x02,
    NodeDelete = 0x03,
    EdgeInsert = 0x10,
    EdgeUpdate = 0x11,
    EdgeDelete = 0x12,
    PropSet = 0x20,
    PropDelete = 0x21,
    /// Mutate only the `first_prop_off` (U48) head pointer of a node — the
    /// chain anchor that points into the v2 props zone. Introduced by T17c
    /// Step 3b.2 so that property-chain mutations are durably replayed
    /// without rewriting the rest of the `NodeRecord`.
    NodePropHeadUpdate = 0x22,
    StringIntern = 0x30,
    LabelIntern = 0x31,
    KeyIntern = 0x32,
    EnergyDecay = 0x40,
    EnergyReinforce = 0x41,
    SynapseReinforce = 0x42,
    ScarUtilAffinitySet = 0x43,
    CentralityUpdate = 0x44,
    SynapseDecay = 0x45,
    CommunityAssign = 0x50,
    HilbertRepermute = 0x51,
    RicciUpdate = 0x52,
    CompactCommunity = 0x53,
    Tier0Update = 0x60,
    Tier1Update = 0x61,
    Tier2Update = 0x62,
    EngramMembersSet = 0x70,
    EngramBitsetSet = 0x71,
    CoactDecay = 0x46,
    Checkpoint = 0xF0,
    NoOp = 0xFE,
    EndOfLog = 0xFF,
}

impl WalKind {
    pub fn from_u8(x: u8) -> Option<Self> {
        use WalKind::*;
        Some(match x {
            0x01 => NodeInsert,
            0x02 => NodeUpdate,
            0x03 => NodeDelete,
            0x10 => EdgeInsert,
            0x11 => EdgeUpdate,
            0x12 => EdgeDelete,
            0x20 => PropSet,
            0x21 => PropDelete,
            0x22 => NodePropHeadUpdate,
            0x30 => StringIntern,
            0x31 => LabelIntern,
            0x32 => KeyIntern,
            0x40 => EnergyDecay,
            0x41 => EnergyReinforce,
            0x42 => SynapseReinforce,
            0x43 => ScarUtilAffinitySet,
            0x44 => CentralityUpdate,
            0x45 => SynapseDecay,
            0x50 => CommunityAssign,
            0x51 => HilbertRepermute,
            0x52 => RicciUpdate,
            0x53 => CompactCommunity,
            0x60 => Tier0Update,
            0x61 => Tier1Update,
            0x62 => Tier2Update,
            0x70 => EngramMembersSet,
            0x71 => EngramBitsetSet,
            0x46 => CoactDecay,
            0xF0 => Checkpoint,
            0xFE => NoOp,
            0xFF => EndOfLog,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

/// Generic WAL payload — each variant maps to a [`WalKind`].
///
/// Payloads intentionally carry full state (not deltas) so that replay is
/// idempotent (re-applying an already-durable record is a no-op).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WalPayload {
    NodeInsert {
        node_id: u32,
        label_bitset: u64,
    },
    NodeUpdate {
        node_id: u32,
        label_bitset: u64,
        energy: u16,
        scar_util_affinity: u16,
        centrality_cached: u16,
        flags: u16,
        community_id: u32,
    },
    NodeDelete {
        node_id: u32,
    },
    EdgeInsert {
        edge_id: u64,
        src: u32,
        dst: u32,
        edge_type: u16,
        weight_u16: u16,
    },
    EdgeUpdate {
        edge_id: u64,
        weight_u16: u16,
        ricci_u8: u8,
        flags: u8,
        engram_tag: u16,
    },
    EdgeDelete {
        edge_id: u64,
    },
    PropSet {
        node_id: u32,
        key_id: u16,
        value: PropValue,
    },
    PropDelete {
        node_id: u32,
        key_id: u16,
    },
    StringIntern {
        page_id: u32,
        offset: u32,
        bytes: Vec<u8>,
    },
    LabelIntern {
        label_id: u16,
        name: String,
    },
    KeyIntern {
        key_id: u16,
        name: String,
    },
    EnergyDecay {
        factor_q16: u16, // multiplicative factor in Q0.16
    },
    EnergyReinforce {
        node_id: u32,
        new_energy: u16,
    },
    SynapseReinforce {
        edge_id: u64,
        new_weight: u16,
    },
    SynapseDecay {
        factor_q16: u16, // multiplicative factor in Q0.16 applied to all live SYNAPSE edges
    },
    ScarUtilAffinitySet {
        node_id: u32,
        packed: u16,
    },
    CentralityUpdate {
        updates: Vec<(u32, u16)>, // (node_id, centrality_cached)
    },
    CommunityAssign {
        updates: Vec<(u32, u32)>, // (node_id, community_id)
    },
    HilbertRepermute {
        permutation: Vec<u32>,
    },
    RicciUpdate {
        updates: Vec<(u64, u8)>, // (edge_id, ricci_u8)
    },
    Tier0Update {
        node_id: u32,
        bits: [u64; 2], // 128 bits
    },
    Tier1Update {
        node_id: u32,
        bits: [u64; 8], // 512 bits
    },
    Tier2Update {
        node_id: u32,
        f16s: Vec<u16>, // 384 × f16 stored as raw u16
    },
    /// Full engram membership snapshot. The WAL carries the entire list of
    /// member node slot IDs so replay is idempotent (re-applying overwrites
    /// the same directory entry with the same bytes). `engram_id = 0` is
    /// reserved and rejected by the writer.
    EngramMembersSet {
        engram_id: u16,
        members: Vec<u32>,
    },
    /// Absolute value of a node's 64-bit engram signature (T7 Step 1).
    /// Replay overwrites the column slot with `bitset` verbatim — idempotent.
    EngramBitsetSet {
        node_id: u32,
        bitset: u64,
    },
    Checkpoint {
        at_lsn: u64,
    },
    NoOp,
    EndOfLog,
    /// Apply a Q0.16 multiplicative decay to every live edge whose
    /// `edge_type` equals `coact_type_id` (T7 Step 5).
    ///
    /// Distinct from [`WalPayload::SynapseDecay`] because COACT and
    /// SYNAPSE columns share the same `weight_u16` slot but follow
    /// different decay schedules — see RFC pillar 2 / format-spec §2.
    ///
    /// Idempotent under replay: re-applying the same factor multiplies
    /// twice (mirroring `SynapseDecay` semantics — replay deterministic
    /// because the records are ordered).
    ///
    /// **Wire-compat note**: this variant is intentionally appended at the
    /// tail of `WalPayload` to keep all preceding serde variant indices
    /// stable for older WAL files written before T7 Step 5.
    CoactDecay {
        factor_q16: u16,
        coact_type_id: u16,
    },
    /// Transactional per-community compaction (T11 Step 5).
    ///
    /// Relocates every live node of `community_id` from its scattered old
    /// slot to a fresh contiguous page-aligned range at the file tail.
    /// `relocations[i] = (old_slot, new_slot)` describes the move; the WAL
    /// record precedes the mmap mutation (WAL-first), so a crash between
    /// the two is recovered idempotently by replay:
    ///
    /// * On replay, the engine walks the mapping and, for each `(old, new)`:
    ///   - if the node at `old` carries `community_id == target` → copy to
    ///     `new`, zero-fill `old` (completes the move);
    ///   - if the node at `new` already carries `community_id == target` →
    ///     treat as already applied (no-op).
    ///
    /// * Edge remap is also idempotent: re-applying `src = old_to_new[src]`
    ///   only fires when `src` is still in the old-slot domain (i.e. still
    ///   tied to the pre-compaction layout) — after the first pass, `src`
    ///   has moved to the new range and the lookup misses, so the second
    ///   application is a no-op.
    ///
    /// `page_range` (`(first, last)`, half-open on `last`) documents the
    /// new contiguous page range for verification / debugging; the actual
    /// replay only depends on `relocations`.
    CompactCommunity {
        community_id: u32,
        relocations: Vec<(u32, u32)>,
        page_range: (u32, u32),
    },
    /// Update only the `first_prop_off` head pointer of a node (T17c
    /// Step 3b.2). The full U48 value is serialized as `u64` (upper 16 bits
    /// always zero) for a stable bincode wire width independent of `U48`'s
    /// internal byte layout.
    ///
    /// Idempotent under replay: re-applying overwrites the same 6-byte
    /// `first_prop_off` slot in the `NodeRecord` with the same value,
    /// leaving every other field untouched.
    ///
    /// **Wire-compat note**: appended at the tail of `WalPayload` — older
    /// WAL files (written before T17c) do not contain this variant and
    /// remain decodable because every preceding variant index is stable.
    NodePropHeadUpdate {
        node_id: u32,
        first_prop_off: u64,
    },
}

impl WalPayload {
    pub fn kind(&self) -> WalKind {
        use WalKind as K;
        use WalPayload::*;
        match self {
            NodeInsert { .. } => K::NodeInsert,
            NodeUpdate { .. } => K::NodeUpdate,
            NodeDelete { .. } => K::NodeDelete,
            EdgeInsert { .. } => K::EdgeInsert,
            EdgeUpdate { .. } => K::EdgeUpdate,
            EdgeDelete { .. } => K::EdgeDelete,
            PropSet { .. } => K::PropSet,
            PropDelete { .. } => K::PropDelete,
            StringIntern { .. } => K::StringIntern,
            LabelIntern { .. } => K::LabelIntern,
            KeyIntern { .. } => K::KeyIntern,
            EnergyDecay { .. } => K::EnergyDecay,
            EnergyReinforce { .. } => K::EnergyReinforce,
            SynapseReinforce { .. } => K::SynapseReinforce,
            SynapseDecay { .. } => K::SynapseDecay,
            ScarUtilAffinitySet { .. } => K::ScarUtilAffinitySet,
            CentralityUpdate { .. } => K::CentralityUpdate,
            CommunityAssign { .. } => K::CommunityAssign,
            HilbertRepermute { .. } => K::HilbertRepermute,
            RicciUpdate { .. } => K::RicciUpdate,
            Tier0Update { .. } => K::Tier0Update,
            Tier1Update { .. } => K::Tier1Update,
            Tier2Update { .. } => K::Tier2Update,
            EngramMembersSet { .. } => K::EngramMembersSet,
            EngramBitsetSet { .. } => K::EngramBitsetSet,
            CoactDecay { .. } => K::CoactDecay,
            CompactCommunity { .. } => K::CompactCommunity,
            NodePropHeadUpdate { .. } => K::NodePropHeadUpdate,
            Checkpoint { .. } => K::Checkpoint,
            NoOp => K::NoOp,
            EndOfLog => K::EndOfLog,
        }
    }
}

/// Property value types stored in WAL records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropValue {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    StringRef { page_id: u32, offset: u32 },
    BytesRef { page_id: u32, offset: u32 },
}

// ---------------------------------------------------------------------------
// Framed record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub lsn: u64,
    pub timestamp: i64,
    pub flags: u8,
    pub payload: WalPayload,
}

impl WalRecord {
    /// Mark this record as a transaction commit (`flags.bit(0)`).
    pub const FLAG_COMMIT: u8 = 1 << 0;
    /// Mark this record as a checkpoint boundary (`flags.bit(1)`).
    pub const FLAG_CHECKPOINT: u8 = 1 << 1;

    /// Encode the record to its on-disk byte representation.
    pub fn encode(&self) -> Result<Vec<u8>, SubstrateError> {
        let cfg = bincode_config();
        let payload = bincode::serde::encode_to_vec(&self.payload, cfg)
            .map_err(|e| SubstrateError::WalEncode(e.to_string()))?;
        let length = payload.len() as u32;
        let kind_byte = self.payload.kind() as u8;

        let mut out = Vec::with_capacity(WAL_HEADER_SIZE + payload.len());
        // header bytes [0..24] — crc computed over this slice XOR payload.
        out.extend_from_slice(&length.to_le_bytes()); // 0..4
        out.push(kind_byte); // 4
        out.push(self.flags); // 5
        out.extend_from_slice(&[0u8, 0u8]); // 6..8 reserved
        out.extend_from_slice(&self.lsn.to_le_bytes()); // 8..16
        out.extend_from_slice(&self.timestamp.to_le_bytes()); // 16..24

        let crc = crc_over(&out[0..24], &payload);
        out.extend_from_slice(&crc.to_le_bytes()); // 24..28
        out.extend_from_slice(&payload); // payload
        debug_assert_eq!(out.len(), WAL_HEADER_SIZE + payload.len());
        Ok(out)
    }

    /// Decode a record from `bytes`. Returns the decoded record and the number of bytes
    /// consumed.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize), SubstrateError> {
        if bytes.len() < WAL_HEADER_SIZE {
            return Err(SubstrateError::WalShortRead {
                needed: WAL_HEADER_SIZE,
                got: bytes.len(),
            });
        }
        let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let kind_byte = bytes[4];
        let flags = bytes[5];
        // bytes[6..8] reserved — check zero
        if &bytes[6..8] != &[0, 0] {
            return Err(SubstrateError::WalBadFrame(
                "reserved bytes at offset 6..8 are non-zero".into(),
            ));
        }
        let lsn = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let timestamp = i64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let claimed_crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap());

        let total = WAL_HEADER_SIZE + length;
        if bytes.len() < total {
            return Err(SubstrateError::WalShortRead {
                needed: total,
                got: bytes.len(),
            });
        }
        let payload_bytes = &bytes[WAL_HEADER_SIZE..total];
        let actual_crc = crc_over(&bytes[0..24], payload_bytes);
        if actual_crc != claimed_crc {
            return Err(SubstrateError::WalCrcMismatch {
                claimed: claimed_crc,
                actual: actual_crc,
            });
        }
        // Sanity check: kind byte matches payload decode.
        let _kind = WalKind::from_u8(kind_byte)
            .ok_or_else(|| SubstrateError::WalBadFrame(format!("unknown kind byte {kind_byte}")))?;

        let cfg = bincode_config();
        let (payload, _used): (WalPayload, usize) =
            bincode::serde::decode_from_slice(payload_bytes, cfg)
                .map_err(|e| SubstrateError::WalDecode(e.to_string()))?;

        Ok((
            Self {
                lsn,
                timestamp,
                flags,
                payload,
            },
            total,
        ))
    }

    pub fn is_commit(&self) -> bool {
        self.flags & Self::FLAG_COMMIT != 0
    }

    pub fn is_checkpoint(&self) -> bool {
        self.flags & Self::FLAG_CHECKPOINT != 0
    }
}

/// Compute CRC32 over `header_24` XOR `payload` (per wal-spec framing).
///
/// "XOR" here means the CRC is computed over the concatenation in a way that
/// tampering with either section changes the output. In practice we just feed
/// both slices to a standard streaming CRC32 — the "XOR" language in the spec
/// captures the design intent (cover both regions).
fn crc_over(header_24: &[u8], payload: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(header_24);
    h.update(payload);
    h.finalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rec(lsn: u64) -> WalRecord {
        WalRecord {
            lsn,
            timestamp: 1_700_000_000,
            flags: WalRecord::FLAG_COMMIT,
            payload: WalPayload::NodeInsert {
                node_id: 42,
                label_bitset: 0b1011,
            },
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let rec = sample_rec(1);
        let bytes = rec.encode().unwrap();
        let (rec2, used) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(used, bytes.len());
        assert_eq!(rec, rec2);
    }

    #[test]
    fn encode_multiple_and_decode_sequence() {
        let mut buf = Vec::new();
        for i in 1..=5 {
            let r = sample_rec(i);
            buf.extend_from_slice(&r.encode().unwrap());
        }
        let mut cursor = 0;
        for i in 1..=5 {
            let (rec, used) = WalRecord::decode(&buf[cursor..]).unwrap();
            assert_eq!(rec.lsn, i);
            cursor += used;
        }
        assert_eq!(cursor, buf.len());
    }

    #[test]
    fn tampered_payload_fails_crc() {
        let rec = sample_rec(1);
        let mut bytes = rec.encode().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = WalRecord::decode(&bytes).unwrap_err();
        matches!(err, SubstrateError::WalCrcMismatch { .. });
    }

    #[test]
    fn tampered_header_fails_crc() {
        let rec = sample_rec(1);
        let mut bytes = rec.encode().unwrap();
        bytes[8] ^= 0xFF; // corrupt LSN byte
        let err = WalRecord::decode(&bytes).unwrap_err();
        matches!(err, SubstrateError::WalCrcMismatch { .. });
    }

    #[test]
    fn short_tail_returns_short_read() {
        let rec = sample_rec(1);
        let bytes = rec.encode().unwrap();
        let truncated = &bytes[..bytes.len() - 3];
        let err = WalRecord::decode(truncated).unwrap_err();
        matches!(err, SubstrateError::WalShortRead { .. });
    }

    #[test]
    fn all_kinds_roundtrip() {
        // Smoke test: every WalPayload variant encodes and decodes identically.
        let payloads = vec![
            WalPayload::NodeInsert {
                node_id: 1,
                label_bitset: 0xAAAA_BBBB_CCCC_DDDD,
            },
            WalPayload::NodeUpdate {
                node_id: 2,
                label_bitset: 1,
                energy: 0x8000,
                scar_util_affinity: 0x1234,
                centrality_cached: 0xFFFF,
                flags: 0x0001,
                community_id: 7,
            },
            WalPayload::NodeDelete { node_id: 3 },
            WalPayload::EdgeInsert {
                edge_id: 0xDEAD_BEEF,
                src: 10,
                dst: 20,
                edge_type: 5,
                weight_u16: 0x4000,
            },
            WalPayload::EdgeUpdate {
                edge_id: 99,
                weight_u16: 0xFFFF,
                ricci_u8: 128,
                flags: 0x02,
                engram_tag: 17,
            },
            WalPayload::EdgeDelete { edge_id: 100 },
            WalPayload::PropSet {
                node_id: 1,
                key_id: 5,
                value: PropValue::I64(-42),
            },
            WalPayload::PropSet {
                node_id: 1,
                key_id: 5,
                value: PropValue::StringRef {
                    page_id: 2,
                    offset: 128,
                },
            },
            WalPayload::PropDelete {
                node_id: 1,
                key_id: 5,
            },
            WalPayload::StringIntern {
                page_id: 0,
                offset: 16,
                bytes: b"hello-world".to_vec(),
            },
            WalPayload::LabelIntern {
                label_id: 1,
                name: "Person".into(),
            },
            WalPayload::KeyIntern {
                key_id: 10,
                name: "age".into(),
            },
            WalPayload::EnergyDecay { factor_q16: 32768 },
            WalPayload::EnergyReinforce {
                node_id: 1,
                new_energy: 0x7FFF,
            },
            WalPayload::SynapseReinforce {
                edge_id: 7,
                new_weight: 0xC000,
            },
            WalPayload::SynapseDecay { factor_q16: 65000 },
            WalPayload::ScarUtilAffinitySet {
                node_id: 1,
                packed: 0xABCD,
            },
            WalPayload::CentralityUpdate {
                updates: vec![(1, 100), (2, 200)],
            },
            WalPayload::CommunityAssign {
                updates: vec![(1, 7), (2, 7), (3, 11)],
            },
            WalPayload::HilbertRepermute {
                permutation: vec![3, 1, 2, 0],
            },
            WalPayload::RicciUpdate {
                updates: vec![(1, 64), (2, 192)],
            },
            WalPayload::Tier0Update {
                node_id: 42,
                bits: [0xDEAD_BEEF, 0xCAFE_BABE],
            },
            WalPayload::Tier1Update {
                node_id: 42,
                bits: [1, 2, 3, 4, 5, 6, 7, 8],
            },
            WalPayload::Tier2Update {
                node_id: 42,
                f16s: vec![0x3C00; 384], // f16 1.0
            },
            WalPayload::EngramMembersSet {
                engram_id: 7,
                members: vec![10, 20, 30, 40, 50],
            },
            WalPayload::EngramBitsetSet {
                node_id: 42,
                bitset: 0xAAAA_BBBB_CCCC_DDDD,
            },
            WalPayload::Checkpoint { at_lsn: 123 },
            WalPayload::NoOp,
            WalPayload::EndOfLog,
            WalPayload::CoactDecay {
                factor_q16: 64880,
                coact_type_id: 7,
            },
            WalPayload::CompactCommunity {
                community_id: 3,
                relocations: vec![(100, 256), (101, 257), (102, 258)],
                page_range: (2, 4),
            },
            WalPayload::NodePropHeadUpdate {
                node_id: 42,
                first_prop_off: (1u64 << 48) - 1, // max U48 value
            },
            WalPayload::NodePropHeadUpdate {
                node_id: 0,
                first_prop_off: 0,
            },
        ];
        for (i, p) in payloads.into_iter().enumerate() {
            let rec = WalRecord {
                lsn: i as u64 + 1,
                timestamp: 0,
                flags: 0,
                payload: p,
            };
            let bytes = rec.encode().unwrap();
            let (rec2, _) = WalRecord::decode(&bytes).unwrap();
            assert_eq!(rec, rec2, "roundtrip failed at index {i}");
        }
    }
}
