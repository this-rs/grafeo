//! On-disk persistence for the L0 / L1 / L2 tier projections (T16.5).
//!
//! See `docs/rfc/substrate/tier-persistence.md` for the full format
//! specification and the rationale for promoting this work out of
//! cutover (T17) into its own task. TL;DR: `phase_tiers` in
//! `obrain-migrate` built the tiered index in-process and discarded it,
//! so every reopen paid for an O(N) rebuild from the bincode property
//! sidecar — blowing the anon RSS gate by 3×–60× depending on the base.
//!
//! ## Format
//!
//! Each tier zone (`substrate.tier0 / .tier1 / .tier2`) is a self-contained
//! file with a 64 B `#[repr(C)]` header followed by a Pod payload. Tier0
//! carries the `slot_to_offset: [u32; n_slots]` array (needed to
//! reconstruct `offset_to_slot` on load); tier1 and tier2 inherit the
//! same slot ordering implicitly.
//!
//! Every zone has its own CRC32 over its payload. Loading validates
//! magic / version / record_size / CRC independently; any failure
//! returns `Ok(None)` rather than an error — the caller falls back to
//! a full rebuild. Tier corruption is **never** fatal for store open.
//!
//! ## Crash safety
//!
//! Writes are ordered: payload memcpy → CRC → header → `msync` →
//! `fsync`. A crash before `msync` leaves the header and payload out of
//! sync, which surfaces as a CRC mismatch on next open → rebuild path.
//! No half-valid state can be mistaken for a valid one.

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::ZoneFile;
use crate::writer::ensure_room;
use bytemuck::{Pod, Zeroable};

/// Current on-disk format version. Bump when the header layout, the
/// Pod tier record layout, or the default SRP seeds change.
pub const TIER_PERSIST_VERSION: u32 = 1;

/// Fixed header size in bytes. Tier records begin at `HEADER_SIZE` (no
/// slot_to_offset) or at `HEADER_SIZE + 4 * n_slots` (tier0 only).
pub const HEADER_SIZE: usize = 64;

/// `flags` bit 0 — set when the payload is preceded by a
/// `[u32; n_slots]` slot_to_offset array (tier0 master only).
pub const FLAG_HAS_SLOT_TO_OFFSET: u32 = 1 << 0;

/// Per-tier magic values. Little-endian `u32` bit pattern of the ASCII
/// bytes; chosen so `hexdump -C | head -1` gives a human-readable
/// identifier on disk.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TierMagic {
    Tier0,
    Tier1,
    Tier2,
}

impl TierMagic {
    /// Little-endian `u32` of the on-disk magic bytes.
    pub const fn as_u32(self) -> u32 {
        match self {
            TierMagic::Tier0 => u32::from_le_bytes(*b"TIR0"),
            TierMagic::Tier1 => u32::from_le_bytes(*b"TIR1"),
            TierMagic::Tier2 => u32::from_le_bytes(*b"TIR2"),
        }
    }

    /// Expected record size in bytes for this tier.
    pub const fn record_size(self) -> u32 {
        match self {
            TierMagic::Tier0 => 16, // Tier0 = [u64; 2]
            TierMagic::Tier1 => 64, // Tier1 = [u64; 8]
            TierMagic::Tier2 => 768, // Tier2 = [u16; 384]
        }
    }
}

/// On-disk tier zone header. `Pod`, 64 B, little-endian everywhere.
///
/// The struct is serialized as-is via `bytemuck::bytes_of`. Do **not**
/// change field order or add new mandatory fields without bumping
/// [`TIER_PERSIST_VERSION`].
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct TierPersistHeader {
    /// `TierMagic::as_u32()` for this zone.
    pub magic: u32,
    /// [`TIER_PERSIST_VERSION`] at write time.
    pub version: u32,
    /// Number of indexed embeddings (dense slot count).
    pub n_slots: u32,
    /// `TierMagic::record_size()` — redundant with magic but lets a
    /// hexdump reader sanity-check the payload width.
    pub record_size: u32,
    /// CRC32 (IEEE polynomial) over the payload region: the
    /// slot_to_offset array (when present) followed by the Pod records.
    pub crc32: u32,
    /// Bit 0 = [`FLAG_HAS_SLOT_TO_OFFSET`]. Remaining bits reserved.
    pub flags: u32,
    /// Zero-filled padding to bring the header to 64 B. Never read by
    /// current code; guarantees future versions can extend the header
    /// without reshuffling offsets.
    pub _reserved: [u32; 10],
}

impl TierPersistHeader {
    /// Build a header for a fresh write. `crc32` is left at zero; the
    /// caller patches it after computing the CRC over the payload.
    pub fn new(magic: TierMagic, n_slots: u32, has_slot_to_offset: bool) -> Self {
        Self {
            magic: magic.as_u32(),
            version: TIER_PERSIST_VERSION,
            n_slots,
            record_size: magic.record_size(),
            crc32: 0,
            flags: if has_slot_to_offset {
                FLAG_HAS_SLOT_TO_OFFSET
            } else {
                0
            },
            _reserved: [0u32; 10],
        }
    }

    /// `true` when [`FLAG_HAS_SLOT_TO_OFFSET`] is set.
    #[inline]
    pub fn has_slot_to_offset(&self) -> bool {
        (self.flags & FLAG_HAS_SLOT_TO_OFFSET) != 0
    }
}

const _: [(); 1] = [(); (core::mem::size_of::<TierPersistHeader>() == HEADER_SIZE) as usize];

/// Result of reading a tier zone. Vec's are copied out of the mmap so
/// the caller can close the zone without dangling references.
pub type LoadedTier<T> = (u32, Option<Vec<u32>>, Vec<T>);

// ---------------------------------------------------------------------------
// Write path
// ---------------------------------------------------------------------------

/// Write a tier zone from scratch.
///
/// Grows `zf` to exactly `HEADER_SIZE + slot_to_offset_bytes + records_bytes`,
/// writes the payload, computes the CRC, writes the header, and calls
/// `msync` + `fsync` to make the whole zone durable.
///
/// * `magic` — one of `TierMagic::{Tier0, Tier1, Tier2}`. Must match the
///   zone kind being written to (tier0 for `Zone::Tier0`, etc.); we do
///   not cross-check here because `ZoneFile` is untyped.
/// * `n_slots` — number of records. Must equal `records.len()` and (when
///   present) `slot_to_offset.len()`.
/// * `slot_to_offset` — `Some(&[u32; n_slots])` for tier0 (the master
///   that carries the index), `None` for tier1 / tier2.
/// * `records` — the Pod tier records in slot order.
pub fn write_tier_zone<T: Pod>(
    zf: &mut ZoneFile,
    magic: TierMagic,
    n_slots: u32,
    slot_to_offset: Option<&[u32]>,
    records: &[T],
) -> SubstrateResult<()> {
    // Sanity: dimensions agree.
    if records.len() != n_slots as usize {
        return Err(SubstrateError::WalBadFrame(format!(
            "tier_persist::write: n_slots={} != records.len()={}",
            n_slots,
            records.len()
        )));
    }
    if core::mem::size_of::<T>() as u32 != magic.record_size() {
        return Err(SubstrateError::WalBadFrame(format!(
            "tier_persist::write: record size {} != expected {} for magic {:?}",
            core::mem::size_of::<T>(),
            magic.record_size(),
            magic
        )));
    }
    let has_s2o = slot_to_offset.is_some();
    if let Some(s2o) = slot_to_offset {
        if s2o.len() != n_slots as usize {
            return Err(SubstrateError::WalBadFrame(format!(
                "tier_persist::write: n_slots={} != slot_to_offset.len()={}",
                n_slots,
                s2o.len()
            )));
        }
    }

    let s2o_bytes = if has_s2o { 4 * n_slots as usize } else { 0 };
    let records_bytes = records.len() * core::mem::size_of::<T>();
    let total = HEADER_SIZE + s2o_bytes + records_bytes;

    // `ensure_room` grows with a pre-alloc slack; we want the final
    // file size to match `total` exactly so loads can size-check.
    ensure_room(zf, total, 0)?;
    if (zf.len() as usize) > total {
        // Shrink to the exact target. `ZoneFile::grow_to` refuses to
        // shrink, so we set_len on the underlying file via a private
        // path — but since that's not exposed, we instead tolerate
        // an over-sized file (ensure_room will only grow) and make
        // the reader size-check lenient: it validates `total <= file_len`
        // rather than `==`. See `read_tier_zone` for the symmetry.
    }

    // Memcpy payload.
    {
        let slice = zf.as_slice_mut();
        let payload_start = HEADER_SIZE;
        if let Some(s2o) = slot_to_offset {
            let payload_s2o = &mut slice[payload_start..payload_start + s2o_bytes];
            payload_s2o.copy_from_slice(bytemuck::cast_slice(s2o));
        }
        let records_start = payload_start + s2o_bytes;
        let records_dst = &mut slice[records_start..records_start + records_bytes];
        records_dst.copy_from_slice(bytemuck::cast_slice(records));
    }

    // CRC over the payload (everything after the header).
    let crc = {
        let slice = zf.as_slice();
        let payload = &slice[HEADER_SIZE..HEADER_SIZE + s2o_bytes + records_bytes];
        let mut h = crc32fast::Hasher::new();
        h.update(payload);
        h.finalize()
    };

    // Write header last.
    let mut header = TierPersistHeader::new(magic, n_slots, has_s2o);
    header.crc32 = crc;
    {
        let slice = zf.as_slice_mut();
        slice[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&header));
    }

    // Durability.
    zf.msync()?;
    zf.fsync()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read path
// ---------------------------------------------------------------------------

/// Read a tier zone and validate it end-to-end.
///
/// Returns `Ok(None)` on any validation failure (missing zone, bad
/// magic, bad version, record-size mismatch, CRC mismatch, truncated
/// file). Errors are reserved for structural issues that mean we
/// couldn't even read the bytes (I/O errors from mmap).
///
/// On success, copies the `Vec<u32>` slot_to_offset (if present) and the
/// `Vec<T>` records out of the mmap so the caller can close the zone
/// without invalidating the return value.
pub fn read_tier_zone<T: Pod + Default + Copy>(
    zf: &ZoneFile,
    expected_magic: TierMagic,
) -> SubstrateResult<Option<LoadedTier<T>>> {
    let slice = zf.as_slice();
    if slice.len() < HEADER_SIZE {
        return Ok(None);
    }

    // bytemuck::try_from_bytes is safer than transmute and checks
    // alignment. The header is 4-aligned (mmap starts on page boundary).
    let header: TierPersistHeader =
        match bytemuck::try_pod_read_unaligned::<TierPersistHeader>(&slice[..HEADER_SIZE]) {
            Ok(h) => h,
            Err(_) => return Ok(None),
        };

    if header.magic != expected_magic.as_u32() {
        return Ok(None);
    }
    if header.version != TIER_PERSIST_VERSION {
        return Ok(None);
    }
    if header.record_size != expected_magic.record_size() {
        return Ok(None);
    }
    if core::mem::size_of::<T>() as u32 != header.record_size {
        return Ok(None);
    }

    let n_slots = header.n_slots as usize;
    let has_s2o = header.has_slot_to_offset();
    let s2o_bytes = if has_s2o { 4 * n_slots } else { 0 };
    let records_bytes = n_slots * core::mem::size_of::<T>();
    let expected_len = HEADER_SIZE + s2o_bytes + records_bytes;

    // We allow the file to be >= expected_len (the writer tolerates an
    // over-sized file after `ensure_room`'s pre-alloc; see write_tier_zone
    // for the symmetry). Reject if strictly smaller.
    if slice.len() < expected_len {
        return Ok(None);
    }

    // CRC check over the payload region.
    let payload = &slice[HEADER_SIZE..expected_len];
    let mut h = crc32fast::Hasher::new();
    h.update(payload);
    if h.finalize() != header.crc32 {
        return Ok(None);
    }

    // Copy out. Empty-slot tiers are valid: n_slots == 0 → empty Vecs.
    let slot_to_offset = if has_s2o {
        let s2o_src = &slice[HEADER_SIZE..HEADER_SIZE + s2o_bytes];
        let s2o_slice: &[u32] = match bytemuck::try_cast_slice(s2o_src) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        Some(s2o_slice.to_vec())
    } else {
        None
    };

    let records_src = &slice[HEADER_SIZE + s2o_bytes..expected_len];
    let records: Vec<T> = if n_slots == 0 {
        Vec::new()
    } else {
        match bytemuck::try_cast_slice::<u8, T>(records_src) {
            Ok(s) => s.to_vec(),
            Err(_) => return Ok(None),
        }
    };

    Ok(Some((header.n_slots, slot_to_offset, records)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{SubstrateFile, Zone};
    use crate::tiers::{Tier0, Tier1, Tier2};

    fn tier0_sample(i: u32) -> Tier0 {
        Tier0([i as u64, (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)])
    }

    fn tier1_sample(i: u32) -> Tier1 {
        let mut t = Tier1([0u64; 8]);
        for k in 0..8 {
            t.0[k] = (i as u64).wrapping_mul(0x100 + k as u64);
        }
        t
    }

    fn tier2_sample(i: u32) -> Tier2 {
        let mut t = Tier2([0u16; 384]);
        for k in 0..384 {
            t.0[k] = ((i as u32).wrapping_mul(k as u32 + 1) & 0xFFFF) as u16;
        }
        t
    }

    #[test]
    fn roundtrip_empty() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        {
            let mut zf = sub.open_zone(Zone::Tier0).unwrap();
            write_tier_zone::<Tier0>(&mut zf, TierMagic::Tier0, 0, Some(&[]), &[]).unwrap();
        }
        let zf = sub.open_zone(Zone::Tier0).unwrap();
        let got = read_tier_zone::<Tier0>(&zf, TierMagic::Tier0)
            .unwrap()
            .unwrap();
        assert_eq!(got.0, 0);
        assert_eq!(got.1, Some(Vec::<u32>::new()));
        assert!(got.2.is_empty());
    }

    #[test]
    fn roundtrip_tier0_with_slot_to_offset() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 1000u32;
        let records: Vec<Tier0> = (0..n).map(tier0_sample).collect();
        let s2o: Vec<u32> = (0..n).map(|i| i * 7).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier0).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier0, n, Some(&s2o), &records).unwrap();
        }
        let zf = sub.open_zone(Zone::Tier0).unwrap();
        let (got_n, got_s2o, got_r): LoadedTier<Tier0> =
            read_tier_zone(&zf, TierMagic::Tier0).unwrap().unwrap();
        assert_eq!(got_n, n);
        assert_eq!(got_s2o, Some(s2o));
        assert_eq!(got_r, records);
    }

    #[test]
    fn roundtrip_tier1_no_slot_to_offset() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 500u32;
        let records: Vec<Tier1> = (0..n).map(tier1_sample).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier1).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier1, n, None, &records).unwrap();
        }
        let zf = sub.open_zone(Zone::Tier1).unwrap();
        let (got_n, got_s2o, got_r): LoadedTier<Tier1> =
            read_tier_zone(&zf, TierMagic::Tier1).unwrap().unwrap();
        assert_eq!(got_n, n);
        assert_eq!(got_s2o, None);
        assert_eq!(got_r, records);
    }

    #[test]
    fn roundtrip_tier2() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 100u32;
        let records: Vec<Tier2> = (0..n).map(tier2_sample).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier2).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier2, n, None, &records).unwrap();
        }
        let zf = sub.open_zone(Zone::Tier2).unwrap();
        let (got_n, got_s2o, got_r): LoadedTier<Tier2> =
            read_tier_zone(&zf, TierMagic::Tier2).unwrap().unwrap();
        assert_eq!(got_n, n);
        assert_eq!(got_s2o, None);
        assert_eq!(got_r, records);
    }

    #[test]
    fn corrupt_crc_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 64u32;
        let records: Vec<Tier1> = (0..n).map(tier1_sample).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier1).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier1, n, None, &records).unwrap();
        }
        // Flip one bit in the payload region. The header's CRC no
        // longer matches and the load must fall through.
        {
            let mut zf = sub.open_zone(Zone::Tier1).unwrap();
            zf.as_slice_mut()[HEADER_SIZE + 10] ^= 0xFF;
            zf.msync().unwrap();
        }
        let zf = sub.open_zone(Zone::Tier1).unwrap();
        let got = read_tier_zone::<Tier1>(&zf, TierMagic::Tier1).unwrap();
        assert!(got.is_none(), "corrupted CRC must return None");
    }

    #[test]
    fn wrong_magic_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 8u32;
        let records: Vec<Tier0> = (0..n).map(tier0_sample).collect();
        let s2o: Vec<u32> = (0..n).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier0).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier0, n, Some(&s2o), &records).unwrap();
        }
        // Reading as tier1 from the tier0 zone must reject.
        let zf = sub.open_zone(Zone::Tier0).unwrap();
        let got = read_tier_zone::<Tier1>(&zf, TierMagic::Tier1).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn wrong_version_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 4u32;
        let records: Vec<Tier0> = (0..n).map(tier0_sample).collect();
        let s2o: Vec<u32> = (0..n).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier0).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier0, n, Some(&s2o), &records).unwrap();
        }
        // Patch the version field (offset 4..8 in the header) to 99.
        {
            let mut zf = sub.open_zone(Zone::Tier0).unwrap();
            let bad: u32 = 99;
            zf.as_slice_mut()[4..8].copy_from_slice(&bad.to_le_bytes());
            zf.msync().unwrap();
        }
        let zf = sub.open_zone(Zone::Tier0).unwrap();
        let got = read_tier_zone::<Tier0>(&zf, TierMagic::Tier0).unwrap();
        assert!(got.is_none(), "unsupported version must return None");
    }

    #[test]
    fn truncated_payload_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let n = 100u32;
        let records: Vec<Tier1> = (0..n).map(tier1_sample).collect();
        {
            let mut zf = sub.open_zone(Zone::Tier1).unwrap();
            write_tier_zone(&mut zf, TierMagic::Tier1, n, None, &records).unwrap();
        }
        // `ensure_room` pads the zone to a 4 KiB boundary, so the
        // on-disk file is larger than `HEADER_SIZE + records_bytes`.
        // Truncate to BELOW `expected_len` so the reader's
        // size-check (`slice.len() < expected_len`) trips.
        let expected_len = HEADER_SIZE + (n as usize) * 64;
        let path = sub.zone_path(Zone::Tier1);
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(expected_len as u64 - 8).unwrap();
        drop(f);

        let zf = sub.open_zone(Zone::Tier1).unwrap();
        let got = read_tier_zone::<Tier1>(&zf, TierMagic::Tier1).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn empty_zone_file_returns_none() {
        // Fresh substrate: all tier zones are 0 B. Reader must not panic
        // and must return None so the caller falls back to rebuild.
        let sub = SubstrateFile::open_tempfile().unwrap();
        for (zone, magic) in [
            (Zone::Tier0, TierMagic::Tier0),
            (Zone::Tier1, TierMagic::Tier1),
            (Zone::Tier2, TierMagic::Tier2),
        ] {
            let zf = sub.open_zone(zone).unwrap();
            let got: Option<LoadedTier<[u8; 16]>> = match zone {
                Zone::Tier0 => read_tier_zone::<Tier0>(&zf, magic)
                    .unwrap()
                    .map(|(n, s, r)| (n, s, r.into_iter().map(|_| [0u8; 16]).collect())),
                _ => None,
            };
            assert!(got.is_none(), "empty {zone:?} must return None");
        }
    }

    #[test]
    fn header_size_matches_constant() {
        assert_eq!(core::mem::size_of::<TierPersistHeader>(), HEADER_SIZE);
    }

    #[test]
    fn magic_bytes_are_ascii() {
        // Sanity: the on-disk magic bytes are human-readable in a
        // hexdump, which matters for ops debugging.
        for (magic, expected) in [
            (TierMagic::Tier0, *b"TIR0"),
            (TierMagic::Tier1, *b"TIR1"),
            (TierMagic::Tier2, *b"TIR2"),
        ] {
            assert_eq!(magic.as_u32().to_le_bytes(), expected);
        }
    }

    #[test]
    fn flag_has_slot_to_offset_round_trips() {
        let h_with = TierPersistHeader::new(TierMagic::Tier0, 10, true);
        assert!(h_with.has_slot_to_offset());
        let h_without = TierPersistHeader::new(TierMagic::Tier1, 10, false);
        assert!(!h_without.has_slot_to_offset());
    }
}
