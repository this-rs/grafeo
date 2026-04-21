//! Fixed-size `#[repr(C)]` records for the Substrate on-disk format.
//!
//! See `docs/rfc/substrate/format-spec.md` for the bit-level layout.
//!
//! Invariants enforced at compile time:
//!   - `size_of::<NodeRecord>() == 32`
//!   - `size_of::<EdgeRecord>() == 32`
//!   - `align_of::<NodeRecord>() == 8`
//!   - `align_of::<EdgeRecord>() == 4`
//!   - Both are `#[repr(C)]` and `bytemuck::Pod + Zeroable`, so `bytemuck::cast_slice`
//!     from a 4 KiB page-aligned mmap region is legal.

use bytemuck::{Pod, Zeroable};

/// Unsigned 48-bit little-endian integer, stored as a `[u8; 6]`.
///
/// Used for file-internal offsets where `u32` is too small (records arrays can
/// exceed 4 GiB on megalaw-scale graphs) but `u64` wastes 2 B × hot-path density.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct U48(pub [u8; 6]);

impl U48 {
    pub const ZERO: Self = Self([0; 6]);

    #[inline]
    pub fn from_u64(v: u64) -> Self {
        debug_assert!(v < (1u64 << 48), "U48::from_u64: value exceeds 2^48");
        let b = v.to_le_bytes();
        Self([b[0], b[1], b[2], b[3], b[4], b[5]])
    }

    #[inline]
    pub fn to_u64(self) -> u64 {
        let [a, b, c, d, e, f] = self.0;
        u64::from_le_bytes([a, b, c, d, e, f, 0, 0])
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == [0; 6]
    }
}

impl core::fmt::Debug for U48 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "U48({})", self.to_u64())
    }
}

impl Default for U48 {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// NodeRecord — 32 B, aligned to 8.
// ---------------------------------------------------------------------------

/// Flags bit positions for [`NodeRecord::flags`].
pub mod node_flags {
    pub const TOMBSTONED: u16 = 1 << 0;
    pub const CENTRALITY_STALE: u16 = 1 << 1;
    pub const EMBEDDING_STALE: u16 = 1 << 2;
    pub const ENGRAM_SEED: u16 = 1 << 3;
    pub const HILBERT_DIRTY: u16 = 1 << 4;
    pub const IDENTITY: u16 = 1 << 5;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct NodeRecord {
    /// bit i set ⇒ node carries label interned as id `i` (up to 64 labels).
    pub label_bitset: u64,
    /// Offset into the `EdgeRecord` array of the first outgoing edge (linked list).
    pub first_edge_off: U48,
    /// Offset into the `PropertyPage` array of the first property page (linked list).
    pub first_prop_off: U48,
    /// LDleiden community id. `u32::MAX` ⇒ unassigned.
    pub community_id: u32,
    /// Energy in Q1.15 fixed-point; `0x8000` ≈ 1.0.
    pub energy: u16,
    /// Packed: scar(5b) | utility(5b) | affinity(5b) | dirty_flag(1b).
    pub scar_util_affinity: u16,
    /// Cached PageRank × 65535 (Q0.16). Stale flagged in `flags`.
    pub centrality_cached: u16,
    /// See [`node_flags`].
    pub flags: u16,
}

const _: [(); 1] = [(); (core::mem::size_of::<NodeRecord>() == 32) as usize];
const _: [(); 1] = [(); (core::mem::align_of::<NodeRecord>() == 8) as usize];

/// Number of `NodeRecord` slots that fit in a 4 KiB page.
///
/// `PAGE_SIZE (4096) / NodeRecord::SIZE (32) = 128`. The online-insertion
/// allocator (T11 Step 3) uses this to decide when a community's current
/// page is full and a new one must be opened.
pub const NODES_PER_PAGE: u32 = 128;

// Compile-time sanity check: adjust if NodeRecord grows / PAGE_SIZE changes.
const _: [(); 1] =
    [(); ((4096usize / NodeRecord::SIZE) == NODES_PER_PAGE as usize) as usize];

impl NodeRecord {
    pub const SIZE: usize = 32;

    #[inline]
    pub fn is_tombstoned(&self) -> bool {
        self.flags & node_flags::TOMBSTONED != 0
    }

    #[inline]
    pub fn set_tombstoned(&mut self) {
        self.flags |= node_flags::TOMBSTONED;
    }

    #[inline]
    pub fn energy_q15(&self) -> f32 {
        q1_15_to_f32(self.energy)
    }

    #[inline]
    pub fn set_energy_f32(&mut self, v: f32) {
        self.energy = f32_to_q1_15(v);
    }

    /// Unpack `(scar, utility, affinity, dirty_flag)` each in 0..31 (or 0..1 for flag).
    #[inline]
    pub fn unpack_scar_util_aff(&self) -> PackedScarUtilAff {
        PackedScarUtilAff::unpack(self.scar_util_affinity)
    }

    #[inline]
    pub fn set_scar_util_aff(&mut self, p: PackedScarUtilAff) {
        self.scar_util_affinity = p.pack();
    }

    #[inline]
    pub fn centrality_f32(&self) -> f32 {
        self.centrality_cached as f32 / 65535.0
    }

    #[inline]
    pub fn set_centrality_f32(&mut self, v: f32) {
        self.centrality_cached = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }
}

/// Unpacked representation of `NodeRecord.scar_util_affinity`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedScarUtilAff {
    pub scar: u8,     // 5 bits → 0..=31
    pub utility: u8,  // 5 bits
    pub affinity: u8, // 5 bits
    pub dirty: bool,  // 1 bit
}

impl PackedScarUtilAff {
    pub const fn new(scar: u8, utility: u8, affinity: u8, dirty: bool) -> Self {
        Self {
            scar,
            utility,
            affinity,
            dirty,
        }
    }

    #[inline]
    pub fn pack(self) -> u16 {
        let s = (self.scar & 0x1F) as u16;
        let u = (self.utility & 0x1F) as u16;
        let a = (self.affinity & 0x1F) as u16;
        let d = u16::from(self.dirty);
        s | (u << 5) | (a << 10) | (d << 15)
    }

    #[inline]
    pub fn unpack(x: u16) -> Self {
        Self {
            scar: (x & 0x1F) as u8,
            utility: ((x >> 5) & 0x1F) as u8,
            affinity: ((x >> 10) & 0x1F) as u8,
            dirty: (x >> 15) & 1 == 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization helpers for the packed scar/utility/affinity sub-fields.
//
// Each sub-field is 5 bits (0..=31). The cognitive stores choose a fixed
// value range that maps into those 32 levels. Callers that want to persist
// a float into the column go through these helpers; callers that want to
// read the column back as a float use the inverse helpers.
//
// These scales are the canonical on-disk contract — changing them is a
// substrate format break. They intentionally mirror the default
// `ScarConfig::min_intensity` × `max_scars_per_node` bound, the default
// `UtilityConfig::max_utility`, and the [0, 1] cosine range for affinity.
// ---------------------------------------------------------------------------

/// Upper bound for the cumulative scar intensity represented in the 5-bit
/// `scar` sub-field. Values above are clamped.
pub const SCAR_MAX_INTENSITY_Q5: f32 = 4.0;

/// Upper bound for the utility score represented in the 5-bit `utility`
/// sub-field. Values above are clamped.
pub const UTILITY_MAX_SCORE_Q5: f32 = 5.0;

#[inline]
pub fn scar_to_q5(x: f32) -> u8 {
    let norm = (x.clamp(0.0, SCAR_MAX_INTENSITY_Q5) / SCAR_MAX_INTENSITY_Q5) * 31.0;
    norm.round() as u8
}

#[inline]
pub fn q5_to_scar(q: u8) -> f32 {
    (q.min(31) as f32 / 31.0) * SCAR_MAX_INTENSITY_Q5
}

#[inline]
pub fn utility_to_q5(x: f32) -> u8 {
    let norm = (x.clamp(0.0, UTILITY_MAX_SCORE_Q5) / UTILITY_MAX_SCORE_Q5) * 31.0;
    norm.round() as u8
}

#[inline]
pub fn q5_to_utility(q: u8) -> f32 {
    (q.min(31) as f32 / 31.0) * UTILITY_MAX_SCORE_Q5
}

#[inline]
pub fn affinity_to_q5(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 31.0).round() as u8
}

#[inline]
pub fn q5_to_affinity(q: u8) -> f32 {
    q.min(31) as f32 / 31.0
}

// ---------------------------------------------------------------------------
// EdgeRecord — 32 B stride (28 B logical + 4 B pad), aligned to 4.
// ---------------------------------------------------------------------------

pub mod edge_flags {
    pub const TOMBSTONED: u8 = 1 << 0;
    pub const RICCI_STALE: u8 = 1 << 1;
    pub const COACT: u8 = 1 << 2;
    pub const SYNAPSE_ACTIVE: u8 = 1 << 3;
    pub const BRIDGE: u8 = 1 << 4;
}

/// Canonical edge-type name for cumulative coactivation edges (T7 Step 5).
///
/// Per RFC pillar 2 (substrate format-spec §2 — "Coactivation : edge type
/// COACT"), coactivation between two nodes is recorded as an explicit
/// edge type stored in [`EdgeRecord::edge_type`] (interned via the
/// dictionary), distinct from synapse-typed edges so that decay schedules
/// can differ:
///
/// * SYNAPSE edges decay fast (Hebbian short-term reinforcement),
/// * COACT   edges decay slowly (long-term coactivation evidence).
///
/// The associated [`edge_flags::COACT`] flag bit is orthogonal — it can
/// be set on any edge type as a quick boolean tag, but the canonical
/// "this is a coactivation edge" signal is `edge_type == coact_type_id`.
pub const COACT_EDGE_TYPE_NAME: &str = "COACT";

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct EdgeRecord {
    pub src: u32,
    pub dst: u32,
    pub edge_type: u16,
    /// Synapse weight, Q0.16 (Hebbian reinforcement).
    pub weight_u16: u16,
    /// Next `EdgeRecord` offset in the linked list of edges sharing this `src`.
    pub next_from: U48,
    /// Next `EdgeRecord` offset in the linked list of edges sharing this `dst`.
    pub next_to: U48,
    /// Quantized Ricci-Ollivier curvature in [-1, 1] → (x + 1) × 127.5.
    pub ricci_u8: u8,
    /// See [`edge_flags`].
    pub flags: u8,
    /// Interned engram-cluster id (0 = none).
    pub engram_tag: u16,
    /// Padding to reach 32 B stride.
    pub _pad: [u8; 4],
}

const _: [(); 1] = [(); (core::mem::size_of::<EdgeRecord>() == 32) as usize];
const _: [(); 1] = [(); (core::mem::align_of::<EdgeRecord>() == 4) as usize];

impl EdgeRecord {
    pub const SIZE: usize = 32;

    #[inline]
    pub fn is_tombstoned(&self) -> bool {
        self.flags & edge_flags::TOMBSTONED != 0
    }

    #[inline]
    pub fn set_tombstoned(&mut self) {
        self.flags |= edge_flags::TOMBSTONED;
    }

    #[inline]
    pub fn weight_f32(&self) -> f32 {
        self.weight_u16 as f32 / 65535.0
    }

    #[inline]
    pub fn set_weight_f32(&mut self, v: f32) {
        self.weight_u16 = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
    }

    #[inline]
    pub fn ricci_f32(&self) -> f32 {
        (self.ricci_u8 as f32 / 127.5) - 1.0
    }

    #[inline]
    pub fn set_ricci_f32(&mut self, v: f32) {
        self.ricci_u8 = ((v.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
    }
}

// ---------------------------------------------------------------------------
// Q1.15 fixed-point helpers (for energy, synapse weights, affinity scores).
// ---------------------------------------------------------------------------

/// Convert a `f32` in `[0.0, 1.0]` to Q1.15 (`u16`). Values outside the range
/// are clamped.
#[inline]
pub fn f32_to_q1_15(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 32768.0).round().min(65535.0) as u16
}

/// Convert a Q1.15 `u16` back to `f32` in `[0.0, 2.0)`.
#[inline]
pub fn q1_15_to_f32(x: u16) -> f32 {
    x as f32 / 32768.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_asserts() {
        assert_eq!(core::mem::size_of::<NodeRecord>(), 32);
        assert_eq!(core::mem::size_of::<EdgeRecord>(), 32);
        assert_eq!(core::mem::align_of::<NodeRecord>(), 8);
        assert_eq!(core::mem::align_of::<EdgeRecord>(), 4);
        assert_eq!(core::mem::size_of::<U48>(), 6);
    }

    #[test]
    fn u48_roundtrip() {
        for v in [0u64, 1, 42, 0x00FFFF, 0x0001_0000, (1u64 << 48) - 1] {
            let x = U48::from_u64(v);
            assert_eq!(x.to_u64(), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn u48_is_zero() {
        assert!(U48::ZERO.is_zero());
        assert!(!U48::from_u64(1).is_zero());
    }

    #[test]
    fn q1_15_roundtrip() {
        // Exact representations
        assert_eq!(f32_to_q1_15(0.0), 0);
        assert_eq!(f32_to_q1_15(1.0), 32768);
        assert!((q1_15_to_f32(f32_to_q1_15(0.5)) - 0.5).abs() < 1e-4);
        assert!((q1_15_to_f32(f32_to_q1_15(0.25)) - 0.25).abs() < 1e-4);
        // Clamping
        assert_eq!(f32_to_q1_15(-1.0), 0);
        assert_eq!(f32_to_q1_15(2.0), 32768);
    }

    #[test]
    fn scar_util_aff_pack_roundtrip() {
        for s in 0u8..=31 {
            for u in [0u8, 15, 31] {
                for a in [0u8, 7, 31] {
                    for d in [false, true] {
                        let p = PackedScarUtilAff::new(s, u, a, d);
                        let packed = p.pack();
                        let unpacked = PackedScarUtilAff::unpack(packed);
                        assert_eq!(unpacked, p, "roundtrip failed for {:?}", p);
                    }
                }
            }
        }
    }

    #[test]
    fn scar_util_aff_clips_to_5bits() {
        // Input beyond 5 bits is silently masked — documented behavior.
        let p = PackedScarUtilAff::new(0xFF, 0xFF, 0xFF, true);
        let r = PackedScarUtilAff::unpack(p.pack());
        assert_eq!(r.scar, 31);
        assert_eq!(r.utility, 31);
        assert_eq!(r.affinity, 31);
        assert!(r.dirty);
    }

    #[test]
    fn node_record_default_is_zeroed() {
        let n = NodeRecord::default();
        assert_eq!(n.label_bitset, 0);
        assert_eq!(n.community_id, 0);
        assert_eq!(n.energy, 0);
        assert_eq!(n.flags, 0);
        assert!(n.first_edge_off.is_zero());
        assert!(!n.is_tombstoned());
    }

    #[test]
    fn node_record_energy_accessor() {
        let mut n = NodeRecord::default();
        n.set_energy_f32(0.5);
        assert!((n.energy_q15() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn node_record_centrality_accessor() {
        let mut n = NodeRecord::default();
        n.set_centrality_f32(0.75);
        assert!((n.centrality_f32() - 0.75).abs() < 1e-4);
    }

    #[test]
    fn edge_record_weight_accessor() {
        let mut e = EdgeRecord::default();
        e.set_weight_f32(0.333);
        assert!((e.weight_f32() - 0.333).abs() < 1e-4);
    }

    #[test]
    fn edge_record_ricci_accessor() {
        let mut e = EdgeRecord::default();
        e.set_ricci_f32(-0.5);
        assert!((e.ricci_f32() - (-0.5)).abs() < 1e-2);
        e.set_ricci_f32(0.5);
        assert!((e.ricci_f32() - 0.5).abs() < 1e-2);
    }

    #[test]
    fn node_record_pod_cast() {
        let n = NodeRecord {
            label_bitset: 0x1234_5678_9ABC_DEF0,
            first_edge_off: U48::from_u64(0x0000_FF_FF_FF),
            first_prop_off: U48::ZERO,
            community_id: 42,
            energy: 0x4000,
            scar_util_affinity: 0x1234,
            centrality_cached: 0xABCD,
            flags: node_flags::ENGRAM_SEED,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&n);
        assert_eq!(bytes.len(), 32);
        let n2: NodeRecord = *bytemuck::from_bytes::<NodeRecord>(bytes);
        assert_eq!(n, n2);
    }

    #[test]
    fn edge_record_pod_cast() {
        let e = EdgeRecord {
            src: 7,
            dst: 11,
            edge_type: 3,
            weight_u16: 0x8000,
            next_from: U48::from_u64(128),
            next_to: U48::from_u64(256),
            ricci_u8: 64,
            flags: edge_flags::SYNAPSE_ACTIVE | edge_flags::COACT,
            engram_tag: 17,
            _pad: [0; 4],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&e);
        assert_eq!(bytes.len(), 32);
        let e2: EdgeRecord = *bytemuck::from_bytes::<EdgeRecord>(bytes);
        assert_eq!(e, e2);
    }
}
