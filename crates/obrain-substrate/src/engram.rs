//! Engram-membership side-table (T7 Step 0).
//!
//! Engrams are cluster identifiers: each engram groups a set of node slots
//! that fire together (Hebbian-style). In the pre-substrate world, this
//! mapping was serialised as a JSON blob on a `:Engram` label node, which
//! required O(engrams × avg_size) JSON encode/decode on every session open.
//!
//! In the topology-as-storage model, the membership relation lives in its
//! own zone (`substrate.engram_members`) as a fixed-slot directory + an
//! append-only payload of u32 node slot IDs:
//!
//! ```text
//!   0..64            header (magic "ENGM", version, payload_used, ...)
//!   64..524_352      directory: 65_536 entries × 8 B
//!                      each entry = (offset: u32, len: u32)
//!                      offset = byte offset into the payload region
//!                      len    = number of u32 node IDs in this engram
//!   524_352..        payload region: contiguous u32 node IDs,
//!                    one stretch per (engram_id, version) snapshot.
//! ```
//!
//! `engram_id = 0` is reserved (null engram), so addressable IDs are
//! `1..=65_535`. An unset engram has `(offset=0, len=0)` in its directory
//! slot. Updates append a fresh snapshot to the payload tail and rewrite
//! only the directory slot — old snapshots become garbage and are reclaimed
//! by the next checkpoint compaction (T7 Step 4).
//!
//! ## Durability
//!
//! Every mutation goes through the WAL-first path in [`Writer`]:
//!
//! 1. `Writer::set_engram_members` appends
//!    [`WalPayload::EngramMembersSet`](crate::wal::WalPayload::EngramMembersSet)
//!    with the full member list (idempotent on replay).
//! 2. The mmap is then updated via [`EngramZone::set_members_raw`].
//!
//! Replay (`replay.rs`) rebuilds the side-table from the WAL by re-calling
//! `set_members_raw` in order, so a wiped zone file is reconstructible from
//! the WAL alone — same invariant as Node / Edge zones.
//!
//! ## Layout constants
//!
//! The whole header + directory fit in the first 128 KiB × 4 = 512 KiB + 4 KiB
//! bracket, well below the default 1 MiB pre-alloc headroom from
//! [`crate::writer::ensure_room`]. For most graphs, the zone file grows in
//! exactly one remap step on first use.
//!
//! [`Writer`]: crate::writer::Writer

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::ZoneFile;
use crate::writer::ensure_room;

/// Magic bytes at the start of the engram-members zone: `b"ENGM"` little-endian.
pub const ENGRAM_MAGIC: u32 = u32::from_le_bytes(*b"ENGM");

/// Current on-disk format version for the engram-members zone. Bump this
/// whenever the header or directory layout changes; a mismatch on open
/// surfaces as [`SubstrateError::UnsupportedVersion`].
pub const ENGRAM_ZONE_VERSION: u32 = 1;

/// Size of the fixed header at the start of the zone file (bytes).
pub const ENGRAM_HEADER_SIZE: usize = 64;

/// Maximum addressable `engram_id` (inclusive). ID 0 is reserved.
pub const MAX_ENGRAM_ID: u16 = u16::MAX;

/// Total number of directory slots (indexed by `engram_id`, slot 0 reserved).
pub const DIRECTORY_SLOTS: usize = (MAX_ENGRAM_ID as usize) + 1;

/// Bytes per directory entry: `(offset: u32, len: u32)` = 8 B.
pub const DIRECTORY_ENTRY_SIZE: usize = 8;

/// Size of the directory region in bytes: 65_536 × 8 B = 512 KiB.
pub const DIRECTORY_SIZE: usize = DIRECTORY_SLOTS * DIRECTORY_ENTRY_SIZE;

/// Byte offset at which the payload region starts.
pub const PAYLOAD_OFFSET: usize = ENGRAM_HEADER_SIZE + DIRECTORY_SIZE;

/// Number of bytes pre-allocated the first time the zone is materialised —
/// header + directory + 1 MiB of payload headroom. Keeps the initial `grow_to`
/// at a 4 KiB-aligned value with enough slack for thousands of small engrams
/// before the next remap.
pub const INITIAL_ZONE_SIZE: u64 = (PAYLOAD_OFFSET as u64 + (1 << 20)) // header + dir + 1 MiB
    .next_multiple_of(4096);

/// Padding pushed into every new payload snapshot so consecutive snapshots
/// stay word-aligned (simplifies future SIMD-accelerated intersection).
/// Currently 4 B (a single u32 padding lane); no-op for correctness but
/// documents the alignment invariant.
pub const PAYLOAD_ALIGN: usize = 4;

/// A wrapper around a [`ZoneFile`] that exposes the engram-membership side-table
/// operations. Stateless — the zone file itself owns all persistent state.
pub struct EngramZone;

impl EngramZone {
    /// Initialise the header of a freshly-materialised zone file. Idempotent:
    /// if the header is already present and valid, this is a no-op. If the
    /// zone is empty, grows it to [`INITIAL_ZONE_SIZE`] and writes the header.
    ///
    /// Call this lazily on first use from the [`Writer`](crate::writer::Writer)
    /// path, or eagerly at replay time before applying engram records.
    pub fn init(zf: &mut ZoneFile) -> SubstrateResult<()> {
        if (zf.len() as usize) < PAYLOAD_OFFSET {
            // Grow to the initial pre-allocated size.
            ensure_room(zf, INITIAL_ZONE_SIZE as usize, 0)?;
            Self::write_header(zf, 0)?;
            return Ok(());
        }
        // Validate an existing header.
        Self::validate_header(zf)?;
        Ok(())
    }

    /// Write the fixed header bytes starting at offset 0.
    fn write_header(zf: &mut ZoneFile, payload_used: u64) -> SubstrateResult<()> {
        let slice = zf.as_slice_mut();
        slice[..4].copy_from_slice(&ENGRAM_MAGIC.to_le_bytes());
        slice[4..8].copy_from_slice(&ENGRAM_ZONE_VERSION.to_le_bytes());
        slice[8..16].copy_from_slice(&payload_used.to_le_bytes());
        // 16..64 reserved — zero-fill (already zero on grow, but be explicit).
        for b in &mut slice[16..ENGRAM_HEADER_SIZE] {
            *b = 0;
        }
        Ok(())
    }

    /// Return `payload_used` — the number of bytes currently consumed in the
    /// payload region. The next snapshot will be written at this offset.
    pub fn payload_used(zf: &ZoneFile) -> SubstrateResult<u64> {
        Self::validate_header(zf)?;
        let slice = zf.as_slice();
        Ok(u64::from_le_bytes(slice[8..16].try_into().unwrap()))
    }

    /// Validate the zone's magic + version. Returns an error if the zone is
    /// uninitialised or carries an unsupported version.
    fn validate_header(zf: &ZoneFile) -> SubstrateResult<()> {
        if (zf.len() as usize) < ENGRAM_HEADER_SIZE {
            return Err(SubstrateError::WalBadFrame(format!(
                "engram zone too short: {} < {}",
                zf.len(),
                ENGRAM_HEADER_SIZE
            )));
        }
        let slice = zf.as_slice();
        let magic = u32::from_le_bytes(slice[0..4].try_into().unwrap());
        if magic != ENGRAM_MAGIC {
            return Err(SubstrateError::BadMagic);
        }
        let version = u32::from_le_bytes(slice[4..8].try_into().unwrap());
        if version != ENGRAM_ZONE_VERSION {
            return Err(SubstrateError::UnsupportedVersion(
                version,
                ENGRAM_ZONE_VERSION,
            ));
        }
        Ok(())
    }

    /// Read the directory entry for `engram_id`: `(payload_offset, len_u32s)`.
    /// Returns `(0, 0)` when the engram is unset.
    fn read_directory(zf: &ZoneFile, engram_id: u16) -> (u32, u32) {
        let slot = ENGRAM_HEADER_SIZE + (engram_id as usize) * DIRECTORY_ENTRY_SIZE;
        let slice = zf.as_slice();
        let offset = u32::from_le_bytes(slice[slot..slot + 4].try_into().unwrap());
        let len = u32::from_le_bytes(slice[slot + 4..slot + 8].try_into().unwrap());
        (offset, len)
    }

    /// Write a directory entry for `engram_id`.
    fn write_directory(zf: &mut ZoneFile, engram_id: u16, offset: u32, len: u32) {
        let slot = ENGRAM_HEADER_SIZE + (engram_id as usize) * DIRECTORY_ENTRY_SIZE;
        let slice = zf.as_slice_mut();
        slice[slot..slot + 4].copy_from_slice(&offset.to_le_bytes());
        slice[slot + 4..slot + 8].copy_from_slice(&len.to_le_bytes());
    }

    /// Return the membership list for `engram_id`, or `None` if the engram
    /// has no members (either never set or explicitly cleared).
    ///
    /// Does not mutate the zone. Copies the u32s out of the mmap so callers
    /// can freely use the result across zone growth events.
    pub fn members(zf: &ZoneFile, engram_id: u16) -> SubstrateResult<Option<Vec<u32>>> {
        if engram_id == 0 {
            return Ok(None);
        }
        Self::validate_header(zf)?;
        let (off, len) = Self::read_directory(zf, engram_id);
        if len == 0 {
            return Ok(None);
        }
        let start = PAYLOAD_OFFSET + off as usize;
        let byte_len = (len as usize) * 4;
        let slice = zf.as_slice();
        if start + byte_len > slice.len() {
            return Err(SubstrateError::WalBadFrame(format!(
                "engram {} dir points past zone end: start={} len={} zone_len={}",
                engram_id,
                start,
                byte_len,
                slice.len()
            )));
        }
        let raw = &slice[start..start + byte_len];
        let ids: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Some(ids))
    }

    /// Count the number of engrams currently with non-empty members. O(65_536),
    /// called only on rare observability paths.
    pub fn count(zf: &ZoneFile) -> SubstrateResult<u32> {
        Self::validate_header(zf)?;
        let mut n = 0u32;
        let slice = zf.as_slice();
        for id in 1..DIRECTORY_SLOTS {
            let slot = ENGRAM_HEADER_SIZE + id * DIRECTORY_ENTRY_SIZE;
            let len = u32::from_le_bytes(slice[slot + 4..slot + 8].try_into().unwrap());
            if len != 0 {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Write the full membership list for `engram_id`. Appends a new snapshot
    /// at the tail of the payload region, updates the directory entry, and
    /// advances `payload_used` in the header. Leaves the previous snapshot
    /// (if any) orphaned — compaction is the checkpoint's job.
    ///
    /// This is the "raw" path — it does NOT log to the WAL. The WAL-first
    /// path goes through
    /// [`Writer::set_engram_members`](crate::writer::Writer::set_engram_members),
    /// which appends a [`WalPayload::EngramMembersSet`](crate::wal::WalPayload::EngramMembersSet)
    /// record before calling this function. Replay also calls this function
    /// directly to reconstruct the zone from the WAL.
    pub fn set_members_raw(
        zf: &mut ZoneFile,
        engram_id: u16,
        members: &[u32],
    ) -> SubstrateResult<()> {
        if engram_id == 0 {
            return Err(SubstrateError::WalBadFrame(
                "engram_id 0 is reserved (null engram)".into(),
            ));
        }
        Self::init(zf)?;

        // Handle the empty case: clear the directory slot without touching payload.
        if members.is_empty() {
            Self::write_directory(zf, engram_id, 0, 0);
            return Ok(());
        }

        // Compute the new snapshot offset (aligned to 4 B) and required size.
        let mut payload_used = Self::payload_used(zf)? as usize;
        // Align up to PAYLOAD_ALIGN — each snapshot starts on a u32 boundary.
        let misalign = payload_used % PAYLOAD_ALIGN;
        if misalign != 0 {
            payload_used += PAYLOAD_ALIGN - misalign;
        }
        let snapshot_bytes = members.len() * 4;
        let snapshot_end = PAYLOAD_OFFSET + payload_used + snapshot_bytes;

        // Ensure the zone is big enough.
        ensure_room(zf, snapshot_end, 1 << 20)?;

        // Write payload bytes.
        let write_start = PAYLOAD_OFFSET + payload_used;
        {
            let slice = zf.as_slice_mut();
            for (i, &id) in members.iter().enumerate() {
                let off = write_start + i * 4;
                slice[off..off + 4].copy_from_slice(&id.to_le_bytes());
            }
        }

        // Bounds-check: engram_id × member count must fit in u32. A single
        // engram with > u32::MAX members is an obvious corruption signal.
        if members.len() > u32::MAX as usize {
            return Err(SubstrateError::WalBadFrame(format!(
                "engram {} member count {} exceeds u32",
                engram_id,
                members.len()
            )));
        }

        // Update the directory entry (offset = payload-relative).
        Self::write_directory(zf, engram_id, payload_used as u32, members.len() as u32);

        // Update payload_used in the header.
        let new_payload_used = (payload_used + snapshot_bytes) as u64;
        let slice = zf.as_slice_mut();
        slice[8..16].copy_from_slice(&new_payload_used.to_le_bytes());

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::{SubstrateFile, Zone};

    /// Open a fresh engram zone backed by a tempfile substrate.
    fn fresh_zone() -> (SubstrateFile, ZoneFile) {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let mut zf = sub.open_zone(Zone::EngramMembers).unwrap();
        EngramZone::init(&mut zf).unwrap();
        (sub, zf)
    }

    #[test]
    fn init_writes_valid_header() {
        let (_sub, zf) = fresh_zone();
        // Magic + version round-trip.
        let magic = u32::from_le_bytes(zf.as_slice()[0..4].try_into().unwrap());
        assert_eq!(magic, ENGRAM_MAGIC);
        let version = u32::from_le_bytes(zf.as_slice()[4..8].try_into().unwrap());
        assert_eq!(version, ENGRAM_ZONE_VERSION);
        assert_eq!(EngramZone::payload_used(&zf).unwrap(), 0);
    }

    #[test]
    fn init_is_idempotent() {
        let (_sub, mut zf) = fresh_zone();
        let len_after_first = zf.len();
        EngramZone::init(&mut zf).unwrap();
        EngramZone::init(&mut zf).unwrap();
        // Size does not grow — header is re-validated, not rewritten.
        assert_eq!(zf.len(), len_after_first);
    }

    #[test]
    fn empty_engram_returns_none() {
        let (_sub, zf) = fresh_zone();
        assert!(EngramZone::members(&zf, 42).unwrap().is_none());
        assert_eq!(EngramZone::count(&zf).unwrap(), 0);
    }

    #[test]
    fn set_and_get_single_engram() {
        let (_sub, mut zf) = fresh_zone();
        let members = vec![10u32, 20, 30, 40, 50];
        EngramZone::set_members_raw(&mut zf, 7, &members).unwrap();
        let got = EngramZone::members(&zf, 7).unwrap().unwrap();
        assert_eq!(got, members);
        assert_eq!(EngramZone::count(&zf).unwrap(), 1);
    }

    #[test]
    fn set_100_members_roundtrip() {
        let (_sub, mut zf) = fresh_zone();
        let members: Vec<u32> = (1..=100).collect();
        EngramZone::set_members_raw(&mut zf, 1234, &members).unwrap();
        let got = EngramZone::members(&zf, 1234).unwrap().unwrap();
        assert_eq!(got, members);
    }

    #[test]
    fn multiple_engrams_independent() {
        let (_sub, mut zf) = fresh_zone();
        EngramZone::set_members_raw(&mut zf, 1, &[100, 200, 300]).unwrap();
        EngramZone::set_members_raw(&mut zf, 2, &[400, 500]).unwrap();
        EngramZone::set_members_raw(&mut zf, 65535, &[999]).unwrap();
        assert_eq!(
            EngramZone::members(&zf, 1).unwrap().unwrap(),
            vec![100, 200, 300]
        );
        assert_eq!(
            EngramZone::members(&zf, 2).unwrap().unwrap(),
            vec![400, 500]
        );
        assert_eq!(EngramZone::members(&zf, 65535).unwrap().unwrap(), vec![999]);
        assert!(EngramZone::members(&zf, 3).unwrap().is_none());
        assert_eq!(EngramZone::count(&zf).unwrap(), 3);
    }

    #[test]
    fn update_orphans_old_snapshot() {
        let (_sub, mut zf) = fresh_zone();
        EngramZone::set_members_raw(&mut zf, 42, &[1, 2, 3]).unwrap();
        let first_used = EngramZone::payload_used(&zf).unwrap();
        assert_eq!(first_used, 12); // 3 × 4 bytes

        // Update with a different list.
        EngramZone::set_members_raw(&mut zf, 42, &[10, 20, 30, 40, 50]).unwrap();
        let second_used = EngramZone::payload_used(&zf).unwrap();
        // Payload advanced past old snapshot + new snapshot.
        assert!(second_used >= first_used + 20); // 5 × 4 = 20

        // Latest snapshot wins.
        let got = EngramZone::members(&zf, 42).unwrap().unwrap();
        assert_eq!(got, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn clear_engram_with_empty_list() {
        let (_sub, mut zf) = fresh_zone();
        EngramZone::set_members_raw(&mut zf, 5, &[1, 2, 3]).unwrap();
        assert!(EngramZone::members(&zf, 5).unwrap().is_some());

        EngramZone::set_members_raw(&mut zf, 5, &[]).unwrap();
        assert!(EngramZone::members(&zf, 5).unwrap().is_none());
        assert_eq!(EngramZone::count(&zf).unwrap(), 0);
    }

    #[test]
    fn engram_id_zero_rejected() {
        let (_sub, mut zf) = fresh_zone();
        let err = EngramZone::set_members_raw(&mut zf, 0, &[1, 2]).unwrap_err();
        assert!(matches!(err, SubstrateError::WalBadFrame(_)));
    }

    #[test]
    fn close_reopen_preserves_members() {
        // The acceptance criterion for T7 Step 0: create 100 members → close →
        // reopen → members() returns same list.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let sub_path = sub.path().to_path_buf();
        let members: Vec<u32> = (1..=100).map(|i| i * 7).collect();
        {
            let mut zf = sub.open_zone(Zone::EngramMembers).unwrap();
            EngramZone::init(&mut zf).unwrap();
            EngramZone::set_members_raw(&mut zf, 999, &members).unwrap();
            zf.msync().unwrap();
            zf.fsync().unwrap();
        }
        // Drop the zone handle (sub keeps the directory alive via its _temp guard).
        // Re-open the zone from the same substrate.
        let zf2 = sub.open_zone(Zone::EngramMembers).unwrap();
        let got = EngramZone::members(&zf2, 999).unwrap().unwrap();
        assert_eq!(got, members);
        // Directory slot 999 still claims the same (offset, len).
        let (off, len) = EngramZone::read_directory(&zf2, 999);
        assert_eq!(len, 100);
        assert!(off < 1 << 20); // first snapshot sits near the start of payload
        // Path still valid — substrate directory was not removed.
        assert!(sub_path.exists());
    }

    #[test]
    fn many_engrams_grow_zone_logarithmically() {
        // Sanity: inserting 1000 engrams of 10 members each should trigger few remaps.
        let (_sub, mut zf) = fresh_zone();
        for id in 1..=1000u16 {
            let members: Vec<u32> = (0..10).map(|j| id as u32 * 100 + j).collect();
            EngramZone::set_members_raw(&mut zf, id, &members).unwrap();
        }
        assert_eq!(EngramZone::count(&zf).unwrap(), 1000);
        assert!(
            zf.remap_count() < 10,
            "expected < 10 remaps, got {}",
            zf.remap_count()
        );
        // Spot-check a few.
        let got = EngramZone::members(&zf, 500).unwrap().unwrap();
        assert_eq!(
            got,
            vec![
                50000, 50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008, 50009
            ]
        );
    }
}
