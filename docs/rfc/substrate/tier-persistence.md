# Tier persistence — T16.5 on-disk format

*Status: in progress (T16.5). Supersedes the "T17's concern" comment at `converter.rs:phase_tiers` — tier persistence is promoted out of cutover because **it is the root cause of T16 gate failures** (embeddings fall into the `substrate.props` bincode sidecar when tier zones are empty → 3–60 GB anon RSS on open).*

## 1. Problem

`phase_tiers` in `obrain-migrate/src/converter.rs` builds a `SubstrateTieredIndex` from every `_st_embedding` property, then discards it (`let _ = substrate;`). The in-process L0 / L1 / L2 projections never reach disk. At reopen, the tier zones are empty; any caller that wants retrieval must either

1. re-scan every node's `_st_embedding` property from `substrate.props` (the bincode sidecar, loaded whole into anon heap), and
2. re-project through `Tier0Builder` / `Tier1Builder` / `Tier2Builder` from scratch.

T16 measured the fallout: PO at 3.86 s / 3.39 GB anon, Wikipedia at 56.66 s / 58.13 GB anon — far outside the ≤ 100 ms / ≤ 1 GB gates.

## 2. Goals

* Persist the three tier projections to the pre-existing `substrate.tier0 / .tier1 / .tier2` zones so reopens skip the rebuild step.
* Keep `SubstrateTieredIndex` as the single source of truth for the projection format — writers and readers go through the same struct.
* Fall back to full rebuild on any corruption signal (missing zone, bad magic, bad version, CRC mismatch). Never abort the store open.

Non-goals (deferred to later steps):

* Dropping `_st_embedding` from `substrate.props` entirely (that removes the 2.86 GB/7.72 GB sidecar on PO/Wikipedia but requires touching the prop-persistence pipeline — T17 work).
* Supporting tier2 widths other than `L2_DIM = 384`.

## 3. On-disk layout

Each tier zone is a **stand-alone file** with its own header + CRC. Zones carry enough metadata to be validated independently; cross-zone consistency (same `n_slots`) is checked at load time.

### 3.1 Header (64 B, `#[repr(C)]`, `Pod`)

```
offset  size  name              notes
─────────────────────────────────────
0       4     magic             b"TIR0" / b"TIR1" / b"TIR2" (little-endian u32)
4       4     version           1
8       4     n_slots           number of indexed embeddings (dense)
12      4     record_size       16 (tier0) / 64 (tier1) / 768 (tier2)
16      4     crc32             CRC32 (IEEE poly) of the payload region
20      4     flags             bit 0 = has_slot_to_offset (tier0 master)
24      40    _reserved         zero-filled; bumps header to 64 B
```

Every tier zone carries an **independent CRC** covering its payload. This lets the loader drop a single corrupted tier without invalidating the others (it still forces a rebuild — we don't do partial loads — but tests and diagnostics stay surgical).

### 3.2 Payload (per zone)

| zone  | layout                                               | size                  |
|-------|------------------------------------------------------|-----------------------|
| tier0 | `[u32; n_slots]` slot_to_offset, then `[Tier0; n]`   | `4·n + 16·n = 20·n`   |
| tier1 | `[Tier1; n_slots]`                                   | `64·n`                |
| tier2 | `[Tier2; n_slots]`                                   | `768·n`               |

**Tier0 is the master.** It carries the `slot_to_offset: Vec<u32>` array because it is the smallest zone and therefore cheapest to scan when the loader needs to rebuild the `offset_to_slot` HashMap. Tier1 and Tier2 inherit the same slot ordering implicitly; at load time we verify their headers' `n_slots` match tier0's before reconstructing.

All three tiers are **`bytemuck::Pod`** (`Tier0 = [u64; 2]`, `Tier1 = [u64; 8]`, `Tier2 = [u16; 384]`) so the payload is a single `cast_slice` — no per-record serialization.

### 3.3 Alignment

The `u32` slot_to_offset array in tier0 begins at file offset 64 (4-byte aligned). Records start at `64 + 4·n_slots`, which is 4-aligned for all n and 8-aligned when n is even. We **do not** pad to 16 B for tier0's `[u64; 2]` — on aarch64 and x86_64 the unaligned-but-naturally-read u64 load is fine; bytemuck's `cast_slice` does require 4-byte alignment of the source which we meet. If a future ISA requires 8-B alignment we revisit via a version bump.

## 4. Format invariants (checked at load)

| Check                                           | Failure → |
|-------------------------------------------------|-----------|
| File size ≥ 64                                  | `Ok(None)` |
| Magic matches expected tier                     | `Ok(None)` |
| Version == 1                                    | `Ok(None)` |
| `record_size` matches expected                  | `Ok(None)` |
| File size == `64 + slot_to_offset_bytes + record_size·n_slots` | `Ok(None)` |
| CRC32 of payload matches header                 | `Ok(None)` |
| `n_slots` equal across the three zones          | `Ok(None)` |

All failures return `Ok(None)` (not an `Err`) — tier corruption is **not fatal**. The caller falls back to `rebuild_from_props` or any other warm-up strategy.

## 5. Write path

```rust
impl SubstrateTieredIndex {
    pub fn persist_to_zones(&self, sub: &SubstrateFile) -> SubstrateResult<()> {
        let guard = self.inner.read();
        let n_slots = guard.slot_to_offset.len() as u32;

        tier_persist::write_tier_zone(
            &mut sub.open_zone(Zone::Tier0)?,
            TierMagic::Tier0, n_slots,
            Some(&guard.slot_to_offset), &guard.l0,
        )?;
        tier_persist::write_tier_zone(
            &mut sub.open_zone(Zone::Tier1)?,
            TierMagic::Tier1, n_slots,
            None, &guard.l1,
        )?;
        tier_persist::write_tier_zone(
            &mut sub.open_zone(Zone::Tier2)?,
            TierMagic::Tier2, n_slots,
            None, &guard.l2,
        )?;
        Ok(())
    }
}
```

Each `write_tier_zone` call: grow to `header_size + payload_size`, memcpy header + slot_to_offset (tier0 only) + records, compute CRC over the payload, patch the header CRC field, `msync()`, `fsync()`.

## 6. Read path

```rust
impl SubstrateTieredIndex {
    pub fn load_from_zones(
        sub: &SubstrateFile,
        dim: usize,
    ) -> SubstrateResult<Option<Self>> {
        let t0 = tier_persist::read_tier_zone::<Tier0>(
            &sub.open_zone(Zone::Tier0)?, TierMagic::Tier0,
        )?;
        let Some((n0, Some(slot_to_offset), l0)) = t0 else { return Ok(None); };

        let t1 = tier_persist::read_tier_zone::<Tier1>(
            &sub.open_zone(Zone::Tier1)?, TierMagic::Tier1,
        )?;
        let Some((n1, None, l1)) = t1 else { return Ok(None); };
        if n0 != n1 { return Ok(None); }

        let t2 = tier_persist::read_tier_zone::<Tier2>(
            &sub.open_zone(Zone::Tier2)?, TierMagic::Tier2,
        )?;
        let Some((n2, None, l2)) = t2 else { return Ok(None); };
        if n0 != n2 { return Ok(None); }

        // Reconstruct offset_to_slot from slot_to_offset.
        let offset_to_slot = slot_to_offset
            .iter().enumerate()
            .map(|(slot, off)| (*off, slot))
            .collect();

        Ok(Some(SubstrateTieredIndex::from_parts(
            dim, slot_to_offset, offset_to_slot, l0, l1, l2,
        )))
    }
}
```

## 7. Crash safety

Write order: payload first (bulk memcpy into mmap) → CRC computed in-memory → header written last → `msync` → `fsync`.

A crash **before `msync`** leaves the tier zones at whatever state the kernel's write-back decided to flush. On next open, the header CRC will not match the on-disk payload and the load returns `Ok(None)` — the caller falls back to rebuild. No half-valid states surface.

A crash **between `msync` and `fsync`** is equivalent because `msync` already issued the dirty-page writeback; `fsync` is belt-and-braces for metadata. Either the full zone is durable or neither is.

## 8. Migration integration

`obrain-migrate::converter::phase_tiers`:

```rust
// old:
let index = SubstrateTieredIndex::new(L2_DIM);
index.rebuild(&pairs);
let _ = substrate;

// new:
let index = SubstrateTieredIndex::new(L2_DIM);
index.rebuild(&pairs);
let sub_file = substrate.substrate_file().lock();
index.persist_to_zones(&sub_file)?;
drop(sub_file);
tracing::info!(
    "Phase::Tiers persisted {n} slots to tier0/1/2",
    n = pairs.len(),
);
```

## 9. Versioning

Any change to `Tier0 / Tier1 / Tier2` field layout, or to the default seeds of `Tier0Builder / Tier1Builder`, requires a **version bump** (1 → 2). Old tier zones with `version = 1` then fail validation at load, trigger a rebuild, and the new write emits `version = 2`. No in-place upgrade — the rebuild step is already idempotent.

## 10. Observability

`admin(action: "get_fabric_stats")` and the SubstrateStore open path should log:

* tier zone sizes on open (each of the three)
* whether `load_from_zones` returned `Some` or `None`
* rebuild latency when `None` forced a fallback

This lets us catch a silent "everyone falls back to rebuild" regression in the wild.

## 11. Open questions

* Should `SubstrateStore::from_substrate` attempt `load_from_zones` eagerly? We default to **no** — loading is driver-owned (obrain-hub decides when to warm the retrieval index). This keeps substrate open cheap for callers that don't need retrieval (T11 geometric activation, bench harnesses, tooling like `cognitive_quality`).

* Should we add a separate `substrate.slot_to_offset` zone so tier0 can be pure `[Tier0]`? Rejected: the slot_to_offset array is 4 B × n_slots = 20 MB at 5M embeddings, and keeping it inside tier0 means only one syscall to open on the hot path.

---

*See also*: `crates/obrain-substrate/src/tier_persist.rs` (implementation), `crates/obrain-substrate/src/retrieval.rs` (`SubstrateTieredIndex::persist_to_zones` / `load_from_zones`).
