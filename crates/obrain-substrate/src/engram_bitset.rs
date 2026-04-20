//! Per-node 64-bit engram signature column (T7 Step 1).
//!
//! Each node slot has an associated `u64` in the `substrate.engram_bitset`
//! zone. The mapping from `engram_id` to bit position is a 6-bit hash:
//!
//! ```text
//!   bit(engram_id) = engram_id & 0x3F         // engram_id % 64
//! ```
//!
//! The column is therefore a **Bloom-filter-like signature** over the set of
//! engrams a node belongs to, NOT a direct 1:1 mapping. Two engrams with the
//! same low 6 bits collide on the same bit, so `bitset[node] & (1 << b)` must
//! be treated as "maybe a member of some engram mapping to bit `b`" rather
//! than a definitive answer. Exact membership is always resolved through the
//! side-table [`crate::engram::EngramZone`].
//!
//! ## Why this shape
//!
//! Hopfield recall (Step 3) does a two-tier filter:
//!
//! 1. **Candidate prune**: `popcount(query_bitset & node_bitset[i])` over
//!    every live node, 8 slots per SIMD instruction (u64×8 AVX2 /
//!    NEON-popcount). Discards every node with zero bit overlap.
//! 2. **Exact check**: for each surviving candidate, consult the side-table
//!    to compute the true intersection of engram membership sets.
//!
//! Tier 1 turns a `O(N × avg_engrams²)` exact search into a single streaming
//! scan — the bottleneck moves from membership-set iteration to pure u64
//! arithmetic, which sits at peak memory bandwidth on modern cores.
//!
//! ## Column layout
//!
//! ```text
//!   offset 0: u64  (node slot 0 — reserved, always zero)
//!   offset 8: u64  (node slot 1 — bitset)
//!   offset 16: u64 (node slot 2 — bitset)
//!   ...
//! ```
//!
//! Column size = `node_high_water × 8` bytes. Growth mirrors `Zone::Nodes`:
//! exponential pre-alloc via [`crate::writer::ensure_room`] to keep remaps
//! O(log N). Tombstoned nodes keep their bitset in place; callers skip them
//! via the `TOMBSTONED` flag in the node zone.
//!
//! ## WAL
//!
//! Every mutation goes through [`WalPayload::EngramBitsetSet`](crate::wal::WalPayload::EngramBitsetSet),
//! which carries the full post-mutation u64 so replay is idempotent
//! (re-applying an already-durable record writes the same bytes). The
//! convenience RMW path `Writer::add_engram_bit` reads the current bitset,
//! ORs in the new bit, and logs the absolute post-state — it's a helper, not
//! a distinct WAL kind.

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::ZoneFile;
use crate::writer::ensure_room;

/// Size in bytes of one bitset entry.
pub const BITSET_ENTRY_SIZE: usize = 8;

/// Return the bit mask corresponding to `engram_id` in the node bitset. Ignores
/// the high 10 bits of `engram_id`; two engrams with the same low 6 bits share
/// the same mask (hash collision — resolved via the side-table).
#[inline(always)]
pub fn engram_bit_mask(engram_id: u16) -> u64 {
    1u64 << (engram_id & 0x3F) as u64
}

/// Stateless wrapper around a [`ZoneFile`] that exposes bitset primitives.
pub struct EngramBitsetColumn;

impl EngramBitsetColumn {
    /// Read the bitset for `node_id`. Returns 0 when the slot is beyond the
    /// current zone length (never written → implicit empty set).
    pub fn get(zf: &ZoneFile, node_id: u32) -> u64 {
        let offset = (node_id as usize) * BITSET_ENTRY_SIZE;
        let slice = zf.as_slice();
        if offset + BITSET_ENTRY_SIZE > slice.len() {
            return 0;
        }
        u64::from_le_bytes(slice[offset..offset + BITSET_ENTRY_SIZE].try_into().unwrap())
    }

    /// Write the bitset for `node_id`. Grows the zone with exponential
    /// pre-alloc if the slot is past the current length. This is the "raw"
    /// path — it does NOT log to the WAL. Replay calls it directly.
    pub fn set_raw(zf: &mut ZoneFile, node_id: u32, bitset: u64) -> SubstrateResult<()> {
        let offset = (node_id as usize) * BITSET_ENTRY_SIZE;
        ensure_room(zf, offset + BITSET_ENTRY_SIZE, 1 << 16)?;
        let slice = zf.as_slice_mut();
        slice[offset..offset + BITSET_ENTRY_SIZE].copy_from_slice(&bitset.to_le_bytes());
        Ok(())
    }

    /// Scan the full column and return the number of nodes with a non-zero
    /// bitset (i.e. in at least one engram). O(zone_len / 8). Used for
    /// observability; the hot Hopfield recall path never calls this.
    pub fn count_nonzero(zf: &ZoneFile) -> u64 {
        let slice = zf.as_slice();
        let mut n = 0u64;
        for chunk in slice.chunks_exact(BITSET_ENTRY_SIZE) {
            let v = u64::from_le_bytes(chunk.try_into().unwrap());
            if v != 0 {
                n += 1;
            }
        }
        n
    }

    /// Borrow the column as an aligned `&[u64]` slice. Returns `None` when
    /// the zone is unmapped (empty) or the mmap length isn't a multiple of
    /// `BITSET_ENTRY_SIZE` (corruption signal).
    pub fn as_u64_slice(zf: &ZoneFile) -> Option<&[u64]> {
        let slice = zf.as_slice();
        if slice.is_empty() {
            return None;
        }
        if slice.len() % BITSET_ENTRY_SIZE != 0 {
            return None;
        }
        // SAFETY: `bytemuck::try_cast_slice` would allocate; we use
        // `try_cast_slice` but in the error crate there's nothing unsafe to
        // surface. Using bytemuck's safe API:
        match bytemuck::try_cast_slice::<u8, u64>(slice) {
            Ok(s) => Some(s),
            Err(_) => None,
        }
    }
}

/// Validate that a caller-supplied `engram_id` is non-zero (0 = reserved
/// null engram). Mirrors the side-table's precondition so the two layers
/// agree.
pub fn check_nonzero_engram_id(engram_id: u16) -> SubstrateResult<()> {
    if engram_id == 0 {
        return Err(SubstrateError::WalBadFrame(
            "engram_id 0 is reserved (null engram)".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{SubstrateFile, Zone};

    #[test]
    fn bit_mask_folds_engram_id_mod_64() {
        assert_eq!(engram_bit_mask(1), 1 << 1);
        assert_eq!(engram_bit_mask(63), 1 << 63);
        assert_eq!(engram_bit_mask(64), 1 << 0); // 64 % 64 = 0
        assert_eq!(engram_bit_mask(65), 1 << 1); // collides with 1
        assert_eq!(engram_bit_mask(65535), 1 << 63); // 65535 % 64 = 63
    }

    #[test]
    fn empty_slot_returns_zero_bitset() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let zf = sub.open_zone(Zone::EngramBitset).unwrap();
        assert_eq!(EngramBitsetColumn::get(&zf, 1), 0);
        assert_eq!(EngramBitsetColumn::get(&zf, 100_000), 0);
    }

    #[test]
    fn set_and_get_roundtrip() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::EngramBitset).unwrap();
        let bits = 0xAAAA_BBBB_CCCC_DDDDu64;
        EngramBitsetColumn::set_raw(&mut zf, 42, bits).unwrap();
        assert_eq!(EngramBitsetColumn::get(&zf, 42), bits);
        // Other slots stay zero.
        assert_eq!(EngramBitsetColumn::get(&zf, 41), 0);
        assert_eq!(EngramBitsetColumn::get(&zf, 43), 0);
    }

    #[test]
    fn multiple_slots_independent() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::EngramBitset).unwrap();
        for i in 1..=20u32 {
            EngramBitsetColumn::set_raw(&mut zf, i, (i as u64) * 0x0101_0101_0101_0101).unwrap();
        }
        for i in 1..=20u32 {
            assert_eq!(
                EngramBitsetColumn::get(&zf, i),
                (i as u64) * 0x0101_0101_0101_0101
            );
        }
        assert_eq!(EngramBitsetColumn::count_nonzero(&zf), 20);
    }

    #[test]
    fn join_leave_semantics_via_full_rewrite() {
        // Bitset has hash collisions, so "remove" goes through a full
        // recompute from the caller's residual engram list.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::EngramBitset).unwrap();

        // Node joins engrams 3, 7, 100 (bits 3, 7, 36).
        let mut bits = 0u64;
        for eid in [3u16, 7, 100] {
            bits |= engram_bit_mask(eid);
        }
        EngramBitsetColumn::set_raw(&mut zf, 1, bits).unwrap();
        let got = EngramBitsetColumn::get(&zf, 1);
        assert_eq!(got, (1 << 3) | (1 << 7) | (1 << 36));

        // Node leaves engram 7 — caller recomputes from remaining {3, 100}.
        let mut bits = 0u64;
        for eid in [3u16, 100] {
            bits |= engram_bit_mask(eid);
        }
        EngramBitsetColumn::set_raw(&mut zf, 1, bits).unwrap();
        let got = EngramBitsetColumn::get(&zf, 1);
        assert_eq!(got, (1 << 3) | (1 << 36));
    }

    #[test]
    fn hash_collisions_are_observable() {
        // Engrams 1 and 65 both map to bit 1 — demonstrating why the bitset
        // is a filter, not an exact oracle.
        assert_eq!(engram_bit_mask(1), engram_bit_mask(65));
    }

    #[test]
    fn as_u64_slice_exposes_column() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::EngramBitset).unwrap();
        for i in 1..=8u32 {
            EngramBitsetColumn::set_raw(&mut zf, i, i as u64).unwrap();
        }
        let slice = EngramBitsetColumn::as_u64_slice(&zf).unwrap();
        // Slot 0 is zero, then 1..=8 carry their values.
        assert_eq!(slice[0], 0);
        for i in 1..=8u32 {
            assert_eq!(slice[i as usize], i as u64);
        }
    }

    #[test]
    fn close_reopen_preserves_bitset() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let sub_path = sub.path().to_path_buf();
        {
            let mut zf = sub.open_zone(Zone::EngramBitset).unwrap();
            EngramBitsetColumn::set_raw(&mut zf, 999, 0xDEAD_BEEF_CAFE_BABE).unwrap();
            zf.msync().unwrap();
            zf.fsync().unwrap();
        }
        // Re-open through the same substrate (its tempdir guard keeps the directory alive).
        let zf2 = sub.open_zone(Zone::EngramBitset).unwrap();
        assert_eq!(
            EngramBitsetColumn::get(&zf2, 999),
            0xDEAD_BEEF_CAFE_BABE
        );
        assert!(sub_path.exists());
    }

    #[test]
    fn engram_id_zero_check() {
        assert!(check_nonzero_engram_id(0).is_err());
        assert!(check_nonzero_engram_id(1).is_ok());
        assert!(check_nonzero_engram_id(65535).is_ok());
    }
}
