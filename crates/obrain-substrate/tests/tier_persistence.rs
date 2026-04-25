//! Integration tests for T16.5 tier persistence.
//!
//! These tests validate the contract described in
//! `docs/rfc/substrate/tier-persistence.md`:
//!
//! 1. **Round-trip fidelity** — persist → reopen store → load reconstructs
//!    the same projections and returns identical `search_top_k` hits as
//!    the in-process source of truth.
//! 2. **Fallback-to-none on corruption** — a corrupted CRC, wrong magic,
//!    or n_slots disagreement surfaces as `Ok(None)` so callers fall back
//!    to `rebuild_from_props`. Corruption is never fatal.
//! 3. **Empty zones** — a fresh substrate with no prior persist should
//!    return `Ok(None)` from `load_tier_index`, again never `Err`.
//! 4. **Crash-before-msync** — dropping the index without calling
//!    `persist_to_zones` yields `Ok(None)` on reopen (nothing to load).
//!
//! The tests drive the full substrate stack (`SubstrateFile` → `Writer` →
//! `SubstrateStore`) rather than unit-testing `tier_persist.rs` directly,
//! because the integration-level invariants (n_slots agreement across
//! three zones, zone path resolution, ensure_room growth) are exactly
//! what T16 gates depend on.

use obrain_substrate::retrieval::{NodeOffset, SubstrateTieredIndex, VectorIndex};
use obrain_substrate::{L2_DIM, SubstrateFile, Zone};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// Build a deterministic f32 vector of dimension `L2_DIM`. Different
/// `seed` values produce linearly-independent vectors so the tier
/// projections do not collide — we want to verify ordering is preserved
/// after persist/load, not defeat the cascade with near-duplicate inputs.
fn synthetic_embedding(seed: u32) -> Vec<f32> {
    let s = seed as f32;
    (0..L2_DIM)
        .map(|i| {
            let i = i as f32;
            ((s * 0.013 + i * 0.0007).sin() * 0.5) + ((s * 0.031 + i * 0.0019).cos() * 0.5)
        })
        .collect()
}

/// Helper: populate an index with `count` deterministic vectors.
fn populated_index(count: u32) -> (SubstrateTieredIndex, Vec<(NodeOffset, Vec<f32>)>) {
    let idx = SubstrateTieredIndex::new(L2_DIM);
    let mut pairs = Vec::with_capacity(count as usize);
    for i in 0..count {
        let emb = synthetic_embedding(i);
        pairs.push((i as NodeOffset, emb));
    }
    idx.rebuild(&pairs);
    (idx, pairs)
}

// ---------------------------------------------------------------------------
// 1. Round-trip
// ---------------------------------------------------------------------------

/// Build an index in-memory, persist it to a fresh substrate, reopen the
/// substrate, and verify `load_from_zones` returns an index whose top-K
/// hits match the source for a known query.
#[test]
fn persist_and_reload_round_trip() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");
    std::fs::create_dir_all(&sub_path).unwrap();

    // --- Phase 1: build + persist -----------------------------------------
    let (src_index, pairs) = populated_index(128);
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        src_index.persist_to_zones(&sub).unwrap();
        drop(sub);
    }

    // Verify the three tier zone files exist and are non-empty.
    for name in ["substrate.tier0", "substrate.tier1", "substrate.tier2"] {
        let p = sub_path.join(name);
        let m = std::fs::metadata(&p)
            .unwrap_or_else(|e| panic!("{} should exist after persist: {e}", name));
        assert!(m.len() > 0, "{} should be non-empty after persist", name);
    }

    // --- Phase 2: reopen substrate and load -------------------------------
    let sub2 = SubstrateFile::open(&sub_path).unwrap();
    let loaded = SubstrateTieredIndex::load_from_zones(&sub2, L2_DIM)
        .expect("load_from_zones returns Ok")
        .expect("load_from_zones returns Some after persist");

    assert_eq!(loaded.len(), src_index.len(), "corpus size preserved");
    assert_eq!(loaded.dim(), src_index.dim(), "dim preserved");

    // --- Phase 3: top-K hits on the loaded index must track the source ---
    let query = synthetic_embedding(42);
    let src_hits = src_index.search_top_k(&query, 8);
    let loaded_hits = loaded.search_top_k(&query, 8);

    // The tier cascade is deterministic for identical (seeds, slot order,
    // projections). Since persist preserves slot order and we use the
    // default seed on both sides, hits must match exactly.
    assert_eq!(
        src_hits.len(),
        loaded_hits.len(),
        "hit count parity on query seed 42"
    );
    for (src, loaded) in src_hits.iter().zip(loaded_hits.iter()) {
        assert_eq!(
            src.0, loaded.0,
            "NodeOffset must match at rank (src={:?}, loaded={:?})",
            src, loaded
        );
        assert!(
            (src.1 - loaded.1).abs() < 1e-5,
            "score must match within f16 round-trip tolerance (src={}, loaded={})",
            src.1,
            loaded.1
        );
    }

    // And double-check that the pairs we fed in are recoverable as exact
    // self-matches (nearest hit for a query equal to one of the corpus
    // embeddings is itself, modulo ties).
    let self_query = &pairs[17].1;
    let hits = loaded.search_top_k(self_query, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 17, "self-match must recover the source offset");
}

// ---------------------------------------------------------------------------
// 2. Empty zones → Ok(None)
// ---------------------------------------------------------------------------

/// A freshly-created substrate that never had `persist_to_zones` called
/// must surface as `Ok(None)` — not an error, not a bogus empty index.
#[test]
fn load_from_empty_substrate_returns_none() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");
    std::fs::create_dir_all(&sub_path).unwrap();

    let sub = SubstrateFile::create(&sub_path).unwrap();
    let out = SubstrateTieredIndex::load_from_zones(&sub, L2_DIM).unwrap();
    assert!(out.is_none(), "empty substrate → Ok(None)");
}

// ---------------------------------------------------------------------------
// 3. CRC corruption → Ok(None)
// ---------------------------------------------------------------------------

/// Flip a byte in the tier0 payload (past the 64 B header) and verify the
/// CRC check rejects the zone, falling back to `Ok(None)` without
/// touching tier1 / tier2.
#[test]
fn corrupted_tier0_payload_returns_none() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");
    std::fs::create_dir_all(&sub_path).unwrap();

    let (src, _pairs) = populated_index(32);
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        src.persist_to_zones(&sub).unwrap();
    }

    // Corrupt a single byte at offset 128 (well past the 64 B header,
    // inside the slot_to_offset or l0 payload region).
    flip_byte(&sub_path.join("substrate.tier0"), 128);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let out = SubstrateTieredIndex::load_from_zones(&sub, L2_DIM).unwrap();
    assert!(
        out.is_none(),
        "corrupted tier0 CRC must surface as Ok(None), not Err or bogus Some"
    );
}

// ---------------------------------------------------------------------------
// 4. Truncated zone → Ok(None)
// ---------------------------------------------------------------------------

/// Truncate tier1 below its declared size so `expected_len` no longer
/// matches. The loader must return `Ok(None)`.
#[test]
fn truncated_tier1_returns_none() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");
    std::fs::create_dir_all(&sub_path).unwrap();

    let (src, _pairs) = populated_index(32);
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        src.persist_to_zones(&sub).unwrap();
    }

    // tier1 file = 64B header + 64B * 32 records = 2112 B minimum.
    // Truncate to 512 bytes — header intact, payload mostly gone.
    let t1 = sub_path.join("substrate.tier1");
    let f = OpenOptions::new().write(true).open(&t1).unwrap();
    f.set_len(512).unwrap();
    drop(f);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let out = SubstrateTieredIndex::load_from_zones(&sub, L2_DIM).unwrap();
    assert!(out.is_none(), "truncated tier1 must surface as Ok(None)");
}

// ---------------------------------------------------------------------------
// 5. Mismatched n_slots across zones → Ok(None)
// ---------------------------------------------------------------------------

/// Corrupt the n_slots field in tier1's header so it disagrees with
/// tier0. The loader detects the mismatch and returns `Ok(None)`.
/// This catches partial-persist scenarios where (hypothetically) one
/// zone was re-written and another was not — we default to rebuild
/// rather than constructing an index with a torn view.
#[test]
fn mismatched_n_slots_returns_none() {
    let td = tempfile::tempdir().unwrap();
    let sub_path = td.path().join("kb");
    std::fs::create_dir_all(&sub_path).unwrap();

    let (src, _pairs) = populated_index(32);
    {
        let sub = SubstrateFile::create(&sub_path).unwrap();
        src.persist_to_zones(&sub).unwrap();
    }

    // n_slots lives at header offset 8..12 (u32 LE). Overwrite with 999.
    // This also invalidates the CRC (different header flips payload
    // interpretation bits), but the n_slots mismatch check would have
    // caught it independently.
    let t1 = sub_path.join("substrate.tier1");
    let mut f = OpenOptions::new().write(true).open(&t1).unwrap();
    f.seek(SeekFrom::Start(8)).unwrap();
    f.write_all(&999u32.to_le_bytes()).unwrap();
    drop(f);

    let sub = SubstrateFile::open(&sub_path).unwrap();
    let out = SubstrateTieredIndex::load_from_zones(&sub, L2_DIM).unwrap();
    assert!(out.is_none(), "mismatched n_slots must surface as Ok(None)");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Flip one bit in the byte at `offset` of `path`. Used to simulate
/// storage-layer bitrot without pulling in a full fault-injection crate.
fn flip_byte(path: &Path, offset: u64) {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut buf = [0u8; 1];
    use std::io::Read;
    f.read_exact(&mut buf).unwrap();
    buf[0] ^= 0x5a;
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&buf).unwrap();
}

// ---------------------------------------------------------------------------
// Sanity: ensure Zone enum has the expected tier variants (compile-time
// guard — if someone renames Zone::Tier0 this test stops compiling).
// ---------------------------------------------------------------------------
#[test]
fn zone_variants_exist() {
    let _ = Zone::Tier0;
    let _ = Zone::Tier1;
    let _ = Zone::Tier2;
}
