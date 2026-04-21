# Substrate Storage — Binary Format Specification

**Status**: Draft
**RFC**: Substrate — Replacement of LpgStore with mmap-native, topology-as-storage substrate
**Target crate**: `obrain-substrate` (new, workspace member of `lab/grafeo`)
**Endianness**: little-endian (native for x86_64 and aarch64 — no byte-swapping on these platforms)
**Alignment**: all records are `#[repr(C)]` and `bytemuck::Pod`. Each section is 4 KiB page-aligned.
**Version**: `SUBSTRATE_FORMAT_VERSION = 1`

## 1. File layout

A Substrate database is a directory:

```
<db_path>/
  substrate.meta          # 4 KiB — magic, version, offsets, counters
  substrate.nodes         # array of NodeRecord (32 B each)
  substrate.edges         # array of EdgeRecord (30 B each, padded to 32 B on disk)
  substrate.props         # array of PropertyPage (4 KiB each)
  substrate.strings       # StringHeap — variable-length, page-indexed
  substrate.tier0         # L0 binary embeddings — 16 B per node (128 bits)
  substrate.tier1         # L1 binary embeddings — 64 B per node (512 bits)
  substrate.tier2         # L2 f16 embeddings — 768 B per node (384 × f16)
  substrate.hilbert       # Hilbert-ordered node_id permutation (u32 × node_count)
  substrate.community     # community_id per node (u32 × node_count) — may alias NodeRecord.community_id
  substrate.wal           # append-only write-ahead log (see wal-spec.md)
  substrate.checkpoint    # periodic checkpoint marker (timestamp + last fsync'd WAL offset)
```

All data files except `substrate.wal` and `substrate.checkpoint` are memory-mapped read-only via `memmap2::Mmap`. Writers mutate through a write-through mmap + WAL path (see wal-spec.md §3).

## 2. Meta file (`substrate.meta` — 4096 B)

```
offset  size  field
------  ----  -----
0       8     magic             = b"SUBSTRT\0"
8       4     format_version    = 1 (u32 LE)
12      4     flags             (u32 LE, bit 0 = "cognitive state active", bit 1 = "hilbert sorted")
16      8     node_count        (u64 LE)
24      8     edge_count        (u64 LE)
32      8     property_page_count (u64 LE)
40      8     string_heap_size  (u64 LE, bytes)
48      8     created_at        (i64 LE, Unix seconds)
56      8     last_checkpoint   (i64 LE, Unix seconds)
64      8     last_wal_offset   (u64 LE, bytes written to WAL at last checkpoint)
72      4     tier0_dim_bits    = 128
76      4     tier1_dim_bits    = 512
80      4     tier2_dim         = 384 (elements, f16 each)
84      4     hilbert_order     (u32 LE — 0 = node_id, 1 = hilbert curve rank)
88      4     schema_crc32      (u32 LE — crc32 of the label/property-key dictionary)
92      4     reserved
96     ...    — reserved (4000 B of zeros, for future extensions)
```

The meta file is rewritten atomically at every checkpoint (`tmpfile + rename + fsync`).

## 3. NodeRecord (32 B, `#[repr(C)]`, Pod)

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeRecord {
    pub label_bitset:         u64,  // bit i set ⇒ node has label i (up to 64 labels — see §9)
    pub first_edge_off:       [u8; 6],  // u48 LE — offset into EdgeRecord array of first outgoing edge
    pub first_prop_off:       [u8; 6],  // u48 LE — offset into PropertyPage array of first property page
    pub community_id:         u32,  // LDleiden community id (u32::MAX ⇒ unassigned)
    pub energy:               u16,  // Q1.15 fixed-point in [0,1) — 0x8000 = 1.0
    pub scar_util_affinity:   u16,  // packed: scar(5b) | utility(5b) | affinity(5b) | flags(1b)
    pub centrality_cached:    u16,  // Q0.16 — pagerank × 65535, stale_bit in flags
    pub flags:                u16,  // bit 0 = tombstoned, 1 = centrality_stale, 2 = embedding_stale,
                                    // 3 = engram_seed, 4 = hilbert_dirty, 5 = identity,
                                    // 6..15 reserved
}
```

**Invariants**:
- `size_of::<NodeRecord>() == 32`, `align_of == 8`
- `first_edge_off == 0` ⇒ no outgoing edges (offset 0 is reserved as sentinel — EdgeRecord at index 0 is the null edge)
- `flags.bit(0) == 1` ⇒ record is logically deleted; excluded from scans
- `community_id == u32::MAX` ⇒ node pending first LDleiden assignment
- `energy` is reinforced on activation, decayed on maintenance tick (see cognitive/energy.md)

**`scar_util_affinity` layout (16 bits, LSB first)**:
```
bits  0..5   scar       (u5, 0..31)  — scar tissue accumulator
bits  5..10  utility    (u5, 0..31)  — how often used in successful retrievals
bits 10..15  affinity   (u5, 0..31)  — user-facing relevance boost
bit  15      dirty_flag (u1)         — any of the three fields updated since last checkpoint
```

## 4. EdgeRecord (30 B logical, 32 B on-disk stride)

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeRecord {
    pub src:          u32,  // source node index
    pub dst:          u32,  // target node index
    pub edge_type:    u16,  // interned edge-type id (u16 ⇒ 65 535 types max)
    pub weight_u16:   u16,  // Q0.16 synapse weight (Hebbian reinforcement)
    pub next_from:    [u8; 6],  // u48 LE — next EdgeRecord with same src (linked list)
    pub next_to:      [u8; 6],  // u48 LE — next EdgeRecord with same dst (linked list)
    pub ricci_u8:     u8,   // quantized Ricci-Ollivier curvature in [-1, 1] ⇒ (x+1)*127.5
    pub flags:        u8,   // bit 0 = tombstoned, 1 = ricci_stale, 2 = coact, 3 = synapse_active,
                            // 4 = bridge, 5..7 reserved
    pub engram_tag:   u16,  // interned engram cluster id (0 = none)
    // Total: 4 + 4 + 2 + 2 + 6 + 6 + 1 + 1 + 2 = 28 B + 4 B padding ⇒ stride 32 B
    pub _pad:         [u8; 4],
}
```

**Invariants**:
- `size_of::<EdgeRecord>() == 32`, `align_of == 4`
- `next_from`, `next_to` form singly-linked intrusive lists — O(1) insertion at head
- `src == 0 && dst == 0` is the sentinel null edge at index 0 — never a real edge
- `weight_u16` is Q0.16; rescale to f32 by `w_u16 as f32 / 65535.0`
- `ricci_u8` — unsigned quantization: `curvature = (ricci_u8 as f32 / 127.5) - 1.0`

## 5. PropertyPage (4096 B, `#[repr(C)]`, Pod)

A property page holds a heterogeneous, variable-sized property list for a single node (or a chain of pages if overflow).

```
offset  size  field
------  ----  -----
0       4     magic            = 0xPROP_PAGE (0xF5075FA6 — FNV of "property")
4       4     node_id          (u32 LE — owning node)
8       6     next_page        (u48 LE — offset of next PropertyPage in chain, 0 = tail)
14      2     entry_count      (u16 LE)
16      2     free_offset      (u16 LE — first free byte within page, from page start)
18      2     tombstones       (u16 LE — count of tombstoned entries in page)
20      4     crc32            (u32 LE — crc32 of bytes [0..20] XOR [24..end])
24   ...4072  entries          (see PropertyEntry layout below)
```

**PropertyEntry** (variable):
```
offset  size  field
------  ----  -----
0       2     key_id         (u16 LE — interned property-key id)
2       1     value_tag      (u8  — 0=null, 1=bool, 2=i64, 3=f64, 4=string_ref,
                              5=bytes_ref, 6=arr_i64, 7=arr_f64, 8=arr_string_ref)
3       1     reserved
4     ...     value          (layout depends on value_tag)
```

- value_tag=1 (bool)        → 1 B `0x00` / `0x01` (total 6 B entry, padded to 8 B)
- value_tag=2 (i64)         → 8 B little-endian (total 12 B, padded to 16 B)
- value_tag=3 (f64)         → 8 B little-endian (total 12 B, padded to 16 B)
- value_tag=4 (string_ref)  → 8 B StringRef { heap_page_id: u32, heap_offset: u32 }
- value_tag=5 (bytes_ref)   → 8 B StringRef
- value_tag=6..8 (arrays)   → 8 B ArrayRef { heap_page_id: u32, heap_offset: u32 } → heap entry has { len: u32, then `len` elements }

**Invariants**:
- `size_of::<PropertyPage>() == 4096`, `align_of == 4096` (page-aligned)
- A single node's properties are chained across pages via `next_page` (at most 4 pages in practice; longer chains indicate schema abuse)
- Tombstoned entries are reclaimed at checkpoint-time compaction

## 6. StringHeap (variable size, page-indexed)

The StringHeap is a sequence of 4 KiB heap pages. String/bytes values referenced by `value_tag ∈ {4,5,6,7,8}` are stored here.

```
HeapPage (4096 B):
offset  size  field
------  ----  -----
0       4     magic            = 0xHEAP_PAGE (0xF1EA95FF)
4       4     page_id          (u32 LE)
8       2     entry_count      (u16 LE)
10      2     free_offset      (u16 LE)
12      4     crc32            (u32 LE)
16   ...4080  entries          (see HeapEntry layout)
```

**HeapEntry** (variable):
```
offset  size  field
------  ----  -----
0       4     len              (u32 LE — bytes for strings/raw, count for arrays)
4     ...     payload          (UTF-8 for strings, raw for bytes, array elements for arrays)
```

Entries are **never resized in place** — updates append a new entry and mark the old one tombstoned (reclaimed at checkpoint compaction).

## 7. Tier files (binary and f16 embeddings)

Each tier is a flat, 4 KiB-page-aligned array indexed by node index (same index space as `substrate.nodes`).

| Tier | Stride | Content | Purpose |
|------|--------|---------|---------|
| `substrate.tier0` | 16 B | 128-bit signed random projection (`u128`) | AVX-512 vpopcntq / NEON scalar — first filter |
| `substrate.tier1` | 64 B | 512-bit random projection (`[u64; 8]`) | refine candidates from tier0 |
| `substrate.tier2` | 768 B | 384 × f16 | final rerank (cosine similarity) |

**Invariants**:
- All tiers are 64-byte aligned for SIMD loads (AVX-512 lane width)
- Re-computing a node's embedding sets `NodeRecord.flags.bit(2) = 1` (embedding_stale). The retrieval path MUST either skip stale nodes or rerank them in tier2.
- The random projection matrix used to compute tier0/tier1 is deterministic (xxh3_64 seeded) — stored in `substrate.meta` as `embedding_matrix_seed: u64` (in flags word — TBD).

## 8. Hilbert and Community files

- `substrate.hilbert` — `u32 × node_count` — a permutation of node indices, ordered by Hilbert curve rank in (community_id, pagerank-rank) 2D space. Rewritten at checkpoint when `flags.bit(4)` (hilbert_dirty) is set on ≥ 0.5% of nodes.
- `substrate.community` — `u32 × node_count` — materialized community id (redundant with `NodeRecord.community_id` but cache-line contiguous for community-scan queries).

## 9. Label & property-key dictionaries

Labels and property keys are interned as `u16` ids. The dictionary is stored inline at the head of `substrate.meta` after offset 4096:

```
section "dict" (grows, ends at next 4 KiB boundary):
  label_count: u16
  for i in 0..label_count:
    len: u8
    name: [u8; len]  (UTF-8)
  — padded to 2 B alignment —
  prop_key_count: u16
  for i in 0..prop_key_count:
    len: u8
    name: [u8; len]
  — padded to 4 KiB —
  crc32: u32  (covers the dict section)
```

**Invariants**:
- Dictionary is append-only during a database's lifetime (labels/keys never removed, only deprecated)
- `schema_crc32` in meta header covers the dictionary section
- Loading verifies `schema_crc32`; mismatch ⇒ rebuild from WAL or abort with diagnostic

## 10. Size envelope (design targets)

| Artefact | megalaw (12 M nodes, 90 M edges) | PO (500 k nodes, 4 M edges) |
|----------|----------------------------------|-----------------------------|
| `substrate.nodes`    | 384 MiB (12 M × 32 B)     | 16 MiB   |
| `substrate.edges`    | 2.88 GiB (90 M × 32 B)    | 128 MiB  |
| `substrate.props`    | ~500 MiB (sparse)          | ~20 MiB  |
| `substrate.strings`  | ~800 MiB                   | ~30 MiB  |
| `substrate.tier0`    | 192 MiB                    | 8 MiB    |
| `substrate.tier1`    | 768 MiB                    | 32 MiB   |
| `substrate.tier2`    | 9.2 GiB                    | 384 MiB  |
| **Total on disk**    | **~15 GiB**                | **~620 MiB** |
| **Resident mmap (hot)** | **~1.2 GiB** (nodes + edges + tier0 + Hilbert window) | **~180 MiB** |

Compared to the current LpgStore snapshot approach (~110 GiB RSS for megalaw), this is a **≥60× reduction in resident memory** with a warm-cache retrieval path that hits tier0 directly from mmap.

## 11. Invariants summary (bytemuck / ABI)

1. All records are `#[repr(C)]`, contain only `Pod` fields, and have static asserts:
   ```rust
   const _: () = assert!(std::mem::size_of::<NodeRecord>() == 32);
   const _: () = assert!(std::mem::size_of::<EdgeRecord>() == 32);
   const _: () = assert!(std::mem::size_of::<PropertyPage>() == 4096);
   ```
2. All multi-byte integers are little-endian; u48 fields use a `[u8; 6]` wrapper with `from_le_bytes` / `to_le_bytes` accessors.
3. Reading via `bytemuck::cast_slice::<u8, NodeRecord>` is legal iff the mmap is aligned to `align_of::<NodeRecord>() == 8` — enforced by 4 KiB page alignment of file headers.
4. Writers MUST NOT tear-write cognitive-state fields (`energy`, `scar_util_affinity`, `centrality_cached`) in the mmap without going through the WAL path — atomicity is guaranteed by WAL replay on crash.

## 12. Versioning and forward compatibility

- Bumping `format_version` requires a migration step in `obrain-migrate`.
- Unknown `flags` bits MUST be preserved on read-modify-write (mask-preserving update).
- `reserved` bytes in `substrate.meta` MUST be zero; readers verify and log a warning if non-zero.

## 13. Validation Gates

These are the pass/fail criteria enforced by `make bench-compare` (T9) after both baseline and substrate JSON outputs are produced. Failing any gate → non-zero exit, blocks merge.

**Reference workloads**:
- **PO**: small fixture (~500 k nodes, ~4 M edges) — `~/.obrain/hub.db`
- **megalaw**: large fixture (~12 M nodes, ~90 M edges) — `~/.obrain/megalaw.obrain` (or user-provided path)

### Hard gates (MUST pass before T15 merge)

| Metric | Target (Substrate) | Baseline reference | Source |
|--------|-------------------|--------------------|--------|
| `rss_anon_peak` on megalaw | ≤ 1 GiB (≈1.07 × 10⁹ B) | 5–10 GiB | `RssSampler::peaks().0` |
| `rss_total_peak` on megalaw | ≤ 18 GiB | ~110 GiB | `RssSampler::peaks().1` |
| `open_duration` on megalaw | ≤ 100 ms | 5 000–15 000 ms | `bench-{baseline,substrate} --target megalaw` |
| `open_duration` on PO | ≤ 20 ms | 200–800 ms | same, `--target po` |
| `retrieval_latency` @ p95 on megalaw | ≤ 1 000 µs | 3 000–10 000 µs | criterion `retrieval` bench |
| `activation_latency` @ p95 on megalaw | ≤ 1 000 µs | 5 000–20 000 µs | criterion `activation` bench |
| `recall@10` vs baseline on megalaw | ≥ 99 % | — (Substrate must match LpgStore top-10 on a fixed query set of ≥ 1 000 queries) | T9 compare harness |
| `recall@100` vs baseline on megalaw | ≥ 99.5 % | — | T9 compare harness |

### Soft gates (warn, don't block)

| Metric | Target | Notes |
|--------|--------|-------|
| `crud_latency` @ p99 (single-node insert) on megalaw | ≤ 2 × baseline | Trade-off accepted: WAL roundtrip is heavier than in-memory LpgStore insert, but durability is the compensation. |
| `rss_anon_peak` steady-state growth over 1 h cognitive workload | ≤ 50 MiB | Catches thinker leaks. |
| `wal_size_bytes` growth rate under write workload | ≤ 2 × mutation throughput | Sanity check on WAL format overhead. |

### Correctness gates (green/red binary)

| Gate | Verification |
|------|--------------|
| Idempotent replay | SIGKILL test: 1 000 random mutations + SIGKILL at random offset → reopen → graph equal to state at last commit marker. Zero mismatches across 100 seeds. |
| Static ABI asserts | `cargo test -p obrain-substrate -- static_asserts` → all size_of / align_of match the format spec. |
| Schema CRC stability | Opening a database with mismatched `schema_crc32` → diagnostic error, no silent corruption. |
| LDleiden convergence | Modularity on megalaw ≥ 90 % of Louvain batch reference. |

### Running the gates

```bash
# Produce baseline JSONs
make bench-baseline PO_DB=/path/to/po.obrain MEGALAW_DB=/path/to/megalaw.obrain

# After T3 + T8 + T10 land — produce substrate JSONs
make bench-substrate

# Evaluate all gates (T9)
make bench-compare
```

`bench-compare` reads the two JSON pairs, applies the table above, emits a
markdown report to `target/bench/gates-report.md`, and exits non-zero on any
hard-gate failure.

## 14. Open questions (for T2 implementation review)

- Should `EdgeRecord.edge_type` promote to `u32` to accommodate richer typed-graph use cases? (Trade-off: +2 B per edge, 180 MiB on megalaw.)
- Should we add a `substrate.engram` sidecar for the engram cluster bitset, or keep `NodeRecord.flags.bit(3)` + `EdgeRecord.engram_tag`? (Trade-off: O(1) engram lookup vs. +16 B/node.)
- Hilbert rank: recompute incrementally on every 1% of dirty nodes, or defer to maintenance tick? (Trade-off: write amplification vs. stale locality.)
