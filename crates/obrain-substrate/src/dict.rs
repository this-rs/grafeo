//! `substrate.dict` — persistent name dictionaries + slot high-water marks.
//!
//! Three name-to-id registries live here:
//!
//! * **Labels** — `name → bit_index` (0..=63), packed into
//!   `NodeRecord.label_bitset`.
//! * **Edge types** — `name → u16`, stored in `EdgeRecord.edge_type`.
//! * **Property keys** — `name → u16`, used when properties land on
//!   `PropertyPage` zones (step 4+).
//!
//! Alongside the names, two slot high-water counters are persisted so
//! the store can distinguish "never-written slot" from "written but
//! tombstoned" at reopen time — mmap zones are grown with significant
//! zero-initialised headroom (1 MiB aligned to 4 KiB), so a raw scan
//! cannot tell them apart from a live zero-initialised record.
//!
//! ## On-disk layout
//!
//! ```text
//! [u32 magic              ] = 0xD1C7_0DB1 ("dict" little-ish)
//! [u32 format_version     ] = 1
//! [u16 label_count        ]
//!   per label:
//!     [u8 len] [u8; len name]  — UTF-8, strict ascii encouraged
//! [u16 edge_type_count    ]
//!   per edge type:
//!     [u8 len] [u8; len name]
//! [u16 prop_key_count     ]
//!   per property key:
//!     [u8 len] [u8; len name]
//! [u64 next_node_id       ]  — slot allocator state (≥ 1; 0 = null sentinel)
//! [u64 next_edge_id       ]  — slot allocator state (≥ 1; 0 = null sentinel)
//! [pad to 4 B boundary    ]
//! [u32 crc32c             ]  — crc32c of all preceding bytes
//! ```
//!
//! **Endian**: little-endian (native for x86_64/aarch64).
//!
//! This is NOT the layout described in `format-spec.md §9` (which places
//! the dict inside `substrate.meta`'s 4 KiB reserved tail). A
//! step-3-sized change would need a format bump; the separate file is
//! a pragmatic interim.

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use crate::blob_column::BlobColSpec;
use crate::error::{SubstrateError, SubstrateResult};
use crate::vec_column::{EntityKind, VecColSpec, VecDType};

const MAGIC: u32 = 0xD1C7_0DB1;
/// Dict on-disk format version.
///
/// * v1 (T6): labels, edge_types, prop_keys, next_node_id, next_edge_id.
/// * v2 (T7 Step 6): adds `next_engram_id: u16` immediately after
///   `next_edge_id`. v1 dicts load with `next_engram_id = 1` (default —
///   the slot allocator starts at 1, slot 0 is reserved).
/// * v3 (T16.7 Step 2): appends a list of `VecColSpec`s recording
///   which property keys have been promoted to dense mmap vector
///   columns. Each spec is 8 bytes on wire: `u16 prop_key_id`,
///   `u8 entity_kind`, `u8 dtype`, `u32 dim`. v1 / v2 dicts load with
///   an empty list (no vec columns seen yet).
/// * v4 (T16.7 Step 4): appends a list of `BlobColSpec`s recording
///   which property keys have been promoted to two-file mmap blob
///   columns (for oversized `Value::String` / `Value::Bytes` payloads).
///   Each spec is 3 bytes on wire: `u16 prop_key_id`, `u8 entity_kind`.
///   v1 / v2 / v3 dicts load with an empty list.
/// * v5 (T17h T1): appends a `PersistedCounters` block : total live
///   nodes (u64), total live edges (u64), fixed 64×u64 label counts,
///   then u16 entry-count + (u16 type_id, u64 count) entries for
///   edge-type counters. v1..=v4 load with `counters = None` — the
///   caller rebuilds from a one-shot zone scan and persists on next
///   flush (which writes v5).
/// * v6 (T17i T2): appends two per-edge-type label-bitset histograms
///   to the counters block — `edge_type_target_label_counts` and
///   `edge_type_source_label_counts`. Each is a `u16 n_entries`
///   prefix followed by `(u16 type_id, [u64; 64] counts)` entries,
///   encoding the distribution of peer labels observed across every
///   edge of that type. Enables O(1) introspection
///   `edge_target_labels(edge_type)` / `edge_source_labels`, which
///   the T17i T3 Cypher-planner rewrite uses to safely route
///   peer-label-constrained queries. v5 dicts load these fields as
///   empty `Vec` and the caller rebuilds them on the next full
///   counter scan.
const FORMAT_VERSION: u32 = 6;

/// Wire size of one `VecColSpec` in the v3+ tail of the dict.
const VEC_COL_SPEC_WIRE_SIZE: usize = 2 + 1 + 1 + 4;

/// Wire size of one `BlobColSpec` in the v4+ tail of the dict.
const BLOB_COL_SPEC_WIRE_SIZE: usize = 2 + 1;

/// T17h T1 — live counters persisted alongside the dict snapshot.
/// Stored as a single block at the end of the dict file (v5 tail),
/// absent in v1..=v4 dicts. A `None` load triggers a one-shot rebuild
/// from the on-disk zones; subsequent flushes write `Some` (v5 bump).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedCounters {
    /// Total number of non-tombstoned nodes (matches `node_count()`).
    pub total_live_nodes: u64,
    /// Total number of non-tombstoned edges (matches `edge_count()`).
    pub total_live_edges: u64,
    /// Per-label live count, indexed by label bit (0..=63). Fixed size
    /// because `NodeRecord.label_bitset` is a u64 bitset. Sparse
    /// labels have count 0 (cheap on wire — always 8 B × 64 = 512 B).
    pub label_counts: [u64; 64],
    /// Per-edge-type live count, indexed by `edge_type_id: u16`. Wire
    /// format uses a count-prefixed list to avoid writing zeros for
    /// unused type ids in the sparse u16 id space.
    pub edge_type_counts: Vec<(u16, u64)>,
    /// T17i T2 — per-edge-type histogram of **target** label bits.
    /// Each entry is `(type_id, [u64; 64])` where slot `i` is the
    /// count of live edges of `type_id` whose `dst.label_bitset` has
    /// bit `i` set. Used by `edge_target_labels(edge_type)` to decide
    /// whether all targets of an edge type share a single label
    /// (planner-safety gate for peer-label rewrites).
    pub edge_type_target_label_counts: Vec<(u16, [u64; 64])>,
    /// T17i T2 — symmetric histogram for **source** labels
    /// (`src.label_bitset` per edge). Fed by the same delete/create
    /// hooks as `target` counts, just reading `rec.src` instead of
    /// `rec.dst`.
    pub edge_type_source_label_counts: Vec<(u16, [u64; 64])>,
}

impl Default for PersistedCounters {
    fn default() -> Self {
        Self {
            total_live_nodes: 0,
            total_live_edges: 0,
            label_counts: [0u64; 64],
            edge_type_counts: Vec::new(),
            edge_type_target_label_counts: Vec::new(),
            edge_type_source_label_counts: Vec::new(),
        }
    }
}

/// Wire size of the fixed-prefix portion of a `PersistedCounters` block
/// (total_live_nodes + total_live_edges + 64× label counts + edge-type
/// entry count prefix). Variable tail = 10 B per edge_type entry.
const COUNTERS_FIXED_PREFIX_SIZE: usize = 8 + 8 + 64 * 8 + 2;

/// Wire size of one edge-type entry in the v5+ counters tail.
const EDGE_TYPE_COUNT_ENTRY_SIZE: usize = 2 + 8;

/// Wire size of one edge-type × label-histogram entry in the v6+
/// counters tail : u16 type_id + 64×u64 per-label counts.
const EDGE_TYPE_LABEL_HIST_ENTRY_SIZE: usize = 2 + 64 * 8;

/// All three registries + allocator state captured at one point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictSnapshot {
    /// Label names, indexed by bit position (0..=63).
    pub labels: Vec<String>,
    /// Edge-type names, indexed by u16 id.
    pub edge_types: Vec<String>,
    /// Property-key names, indexed by u16 id.
    pub prop_keys: Vec<String>,
    /// Next node slot to allocate. Starts at 1 (slot 0 = null sentinel).
    pub next_node_id: u64,
    /// Next edge slot to allocate. Starts at 1 (slot 0 = null sentinel).
    pub next_edge_id: u64,
    /// Next engram id to allocate. Starts at 1 (id 0 = null engram, see
    /// [`crate::engram::MAX_ENGRAM_ID`]). Persisted from format v2 (T7
    /// Step 6); v1 dicts load this as 1.
    pub next_engram_id: u16,
    /// Vector columns that have been materialised for this substrate.
    /// Each entry tells the store "property key N, with this dtype and
    /// dim, lives in a dense mmap zone on disk and not in the bincode
    /// sidecar". Persisted from format v3 (T16.7 Step 2); v1 / v2
    /// dicts load with an empty list.
    pub vec_columns: Vec<VecColSpec>,
    /// Blob columns that have been materialised for this substrate.
    /// Each entry tells the store "property key N (for this entity
    /// kind) lives in a two-file mmap blob-column zone pair on disk
    /// and not in the bincode sidecar". Persisted from format v4
    /// (T16.7 Step 4); v1 / v2 / v3 dicts load with an empty list.
    pub blob_columns: Vec<BlobColSpec>,
    /// Live counters snapshot (T17h T1) — `Some` when loaded from a v5+
    /// dict, `None` when loaded from v1..=v4. The caller treats `None`
    /// as "rebuild from zones at open time, persist on next flush".
    /// Writers always populate this from the in-memory atomics before
    /// calling [`Self::persist`].
    pub counters: Option<PersistedCounters>,
}

impl Default for DictSnapshot {
    fn default() -> Self {
        Self {
            labels: Vec::new(),
            edge_types: Vec::new(),
            prop_keys: Vec::new(),
            next_node_id: 1,
            next_edge_id: 1,
            next_engram_id: 1,
            vec_columns: Vec::new(),
            blob_columns: Vec::new(),
            // T17h T1: a default (fresh) snapshot is already v5-compatible.
            // Producing `Some(Default)` here ensures roundtrip invariance
            // (write → read → equal) and makes in-memory snapshots consistent
            // with freshly-written on-disk files. Load of a legacy v1..=v4
            // file produces `counters: None` (see `from_bytes`) so the store
            // knows to rebuild.
            counters: Some(PersistedCounters::default()),
        }
    }
}

impl DictSnapshot {
    /// Serialize the snapshot to bytes. Caller writes atomically.
    pub fn to_bytes(&self) -> SubstrateResult<Vec<u8>> {
        if self.labels.len() > 64 {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} labels (max 64 — label_bitset is u64)",
                self.labels.len()
            )));
        }
        if self.edge_types.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} edge types (max {} — EdgeRecord.edge_type is u16)",
                self.edge_types.len(),
                u16::MAX
            )));
        }
        if self.prop_keys.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} property keys (max {})",
                self.prop_keys.len(),
                u16::MAX
            )));
        }

        let mut buf = Vec::with_capacity(4096);
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

        write_name_list(&mut buf, &self.labels)?;
        write_name_list(&mut buf, &self.edge_types)?;
        write_name_list(&mut buf, &self.prop_keys)?;

        buf.extend_from_slice(&self.next_node_id.to_le_bytes());
        buf.extend_from_slice(&self.next_edge_id.to_le_bytes());
        // v2: next_engram_id (u16). Always written by current format.
        buf.extend_from_slice(&self.next_engram_id.to_le_bytes());

        // v3: vec_columns list. `u16 count` followed by 8 bytes per spec
        // (prop_key_id u16, entity_kind u8, dtype u8, dim u32).
        if self.vec_columns.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} vec_columns (max {})",
                self.vec_columns.len(),
                u16::MAX
            )));
        }
        buf.extend_from_slice(&(self.vec_columns.len() as u16).to_le_bytes());
        for spec in &self.vec_columns {
            buf.extend_from_slice(&spec.prop_key_id.to_le_bytes());
            buf.push(spec.entity_kind as u8);
            buf.push(spec.dtype as u8);
            buf.extend_from_slice(&spec.dim.to_le_bytes());
        }

        // v4: blob_columns list. `u16 count` followed by 3 bytes per spec
        // (prop_key_id u16, entity_kind u8). No dim/dtype — blobs are
        // variable-length opaque byte payloads identified solely by
        // (prop_key_id, entity_kind).
        if self.blob_columns.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} blob_columns (max {})",
                self.blob_columns.len(),
                u16::MAX
            )));
        }
        buf.extend_from_slice(&(self.blob_columns.len() as u16).to_le_bytes());
        for spec in &self.blob_columns {
            buf.extend_from_slice(&spec.prop_key_id.to_le_bytes());
            buf.push(spec.entity_kind as u8);
        }

        // v5: PersistedCounters block. Always write `Default` when
        // `self.counters` is None so the file format is uniform per
        // version — the caller can tell "counters present" from the
        // version field alone (v5 = always present).
        //
        // Wire format:
        //   [u64 total_live_nodes]
        //   [u64 total_live_edges]
        //   [64 × u64 label_counts]  (fixed, sparse labels are 0)
        //   [u16 edge_type_count]
        //   per entry: [u16 type_id] [u64 count]
        //
        // Overall fixed prefix = 530 B + 10 B per edge_type entry.
        let counters = self.counters.clone().unwrap_or_default();
        if counters.edge_type_counts.len() > u16::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: {} edge_type_counts (max {})",
                counters.edge_type_counts.len(),
                u16::MAX
            )));
        }
        buf.extend_from_slice(&counters.total_live_nodes.to_le_bytes());
        buf.extend_from_slice(&counters.total_live_edges.to_le_bytes());
        for c in &counters.label_counts {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf.extend_from_slice(&(counters.edge_type_counts.len() as u16).to_le_bytes());
        for (type_id, count) in &counters.edge_type_counts {
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
        }

        // v6 (T17i T2): two per-edge-type label histograms — target + source.
        // Each : [u16 n_entries] [(u16 type_id, [u64; 64] counts)] × n.
        if counters.edge_type_target_label_counts.len() > u16::MAX as usize
            || counters.edge_type_source_label_counts.len() > u16::MAX as usize
        {
            return Err(SubstrateError::Internal(format!(
                "dict overflow: label histogram size exceeds u16"
            )));
        }
        for hist in [
            &counters.edge_type_target_label_counts,
            &counters.edge_type_source_label_counts,
        ] {
            buf.extend_from_slice(&(hist.len() as u16).to_le_bytes());
            for (type_id, label_counts) in hist {
                buf.extend_from_slice(&type_id.to_le_bytes());
                for c in label_counts {
                    buf.extend_from_slice(&c.to_le_bytes());
                }
            }
        }

        // Pad to 4-byte boundary before crc to keep alignment clean.
        while buf.len() % 4 != 0 {
            buf.push(0);
        }

        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        Ok(buf)
    }

    /// Deserialize a snapshot from bytes. Validates magic, version, crc.
    pub fn from_bytes(bytes: &[u8]) -> SubstrateResult<Self> {
        if bytes.len() < 8 + 4 {
            return Err(SubstrateError::Internal(format!(
                "dict too short: {} bytes",
                bytes.len()
            )));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(SubstrateError::Internal(format!(
                "dict magic mismatch: got {magic:#x}, expected {MAGIC:#x}"
            )));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        // Accept all backward-compatible versions:
        // * v1 → no `next_engram_id`, no `vec_columns`, no `blob_columns`, no counters.
        // * v2 → `next_engram_id` present, no `vec_columns`, no `blob_columns`, no counters.
        // * v3 → `vec_columns` present, no `blob_columns`, no counters.
        // * v4 → all structural fields present, no counters.
        // * v5 → `PersistedCounters` block (T17h T1).
        // * v6 → current: adds per-edge-type label histograms (T17i T2).
        if !(version >= 1 && version <= FORMAT_VERSION) {
            return Err(SubstrateError::Internal(format!(
                "dict version {version} unsupported (expected 1..={FORMAT_VERSION})"
            )));
        }

        // Separate the CRC tail and validate.
        let body = &bytes[..bytes.len() - 4];
        let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        let computed_crc = crc32c(body);
        if stored_crc != computed_crc {
            return Err(SubstrateError::Internal(format!(
                "dict CRC mismatch: stored {stored_crc:#x}, computed {computed_crc:#x}"
            )));
        }

        let mut cur = &body[8..];
        let labels = read_name_list(&mut cur)?;
        let edge_types = read_name_list(&mut cur)?;
        let prop_keys = read_name_list(&mut cur)?;
        if labels.len() > 64 {
            return Err(SubstrateError::Internal(format!(
                "dict invariant violation: {} labels (>64)",
                labels.len()
            )));
        }
        if cur.len() < 16 {
            return Err(SubstrateError::Internal(format!(
                "dict truncated at allocator counters (need 16, have {})",
                cur.len()
            )));
        }
        let next_node_id = u64::from_le_bytes(cur[0..8].try_into().unwrap());
        let next_edge_id = u64::from_le_bytes(cur[8..16].try_into().unwrap());
        if next_node_id == 0 || next_edge_id == 0 {
            return Err(SubstrateError::Internal(format!(
                "dict invariant violation: next_node_id={next_node_id} \
                 next_edge_id={next_edge_id} (both must be ≥ 1; slot 0 = null sentinel)"
            )));
        }
        // v2+: read `next_engram_id` (u16) immediately after `next_edge_id`.
        // v1: default to 1 (slot 0 reserved for null engram).
        let next_engram_id: u16 = if version >= 2 {
            if cur.len() < 18 {
                return Err(SubstrateError::Internal(format!(
                    "dict v2 truncated at next_engram_id (need 18, have {})",
                    cur.len()
                )));
            }
            let v = u16::from_le_bytes(cur[16..18].try_into().unwrap());
            if v == 0 {
                return Err(SubstrateError::Internal(
                    "dict invariant violation: next_engram_id=0 (must be ≥ 1; \
                     id 0 = null engram sentinel)"
                        .into(),
                ));
            }
            v
        } else {
            1
        };

        // Advance cursor past the allocator counters.
        let after_counters = if version >= 2 { 18 } else { 16 };
        let mut cur = &cur[after_counters..];

        // v3+: vec_columns list. u16 count + 8 B per spec.
        // v1 / v2: empty list — nothing more to read.
        let vec_columns: Vec<VecColSpec> = if version >= 3 {
            if cur.len() < 2 {
                return Err(SubstrateError::Internal(
                    "dict v3 truncated at vec_columns count".into(),
                ));
            }
            let n = u16::from_le_bytes([cur[0], cur[1]]) as usize;
            cur = &cur[2..];
            let need = n * VEC_COL_SPEC_WIRE_SIZE;
            if cur.len() < need {
                return Err(SubstrateError::Internal(format!(
                    "dict v3 truncated at vec_columns body (need {need}, have {})",
                    cur.len()
                )));
            }
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let base = i * VEC_COL_SPEC_WIRE_SIZE;
                let prop_key_id = u16::from_le_bytes([cur[base], cur[base + 1]]);
                let entity_byte = cur[base + 2];
                let dtype_byte = cur[base + 3];
                let dim = u32::from_le_bytes([
                    cur[base + 4],
                    cur[base + 5],
                    cur[base + 6],
                    cur[base + 7],
                ]);
                let Some(entity_kind) = EntityKind::from_u8(entity_byte) else {
                    return Err(SubstrateError::Internal(format!(
                        "dict v3: invalid entity_kind byte {entity_byte} at vec_column[{i}]"
                    )));
                };
                let Some(dtype) = VecDType::from_u8(dtype_byte) else {
                    return Err(SubstrateError::Internal(format!(
                        "dict v3: invalid dtype byte {dtype_byte} at vec_column[{i}]"
                    )));
                };
                if dim == 0 {
                    return Err(SubstrateError::Internal(format!(
                        "dict v3: vec_column[{i}] has dim=0 (invalid)"
                    )));
                }
                out.push(VecColSpec {
                    prop_key_id,
                    entity_kind,
                    dim,
                    dtype,
                });
            }
            // Advance cursor past the vec_columns payload so the v4
            // blob_columns parse below starts at the right offset.
            cur = &cur[need..];
            out
        } else {
            Vec::new()
        };

        // v4+: blob_columns list. u16 count + 3 B per spec
        // (prop_key_id u16, entity_kind u8). v1 / v2 / v3: empty list.
        let blob_columns: Vec<BlobColSpec> = if version >= 4 {
            if cur.len() < 2 {
                return Err(SubstrateError::Internal(
                    "dict v4 truncated at blob_columns count".into(),
                ));
            }
            let n = u16::from_le_bytes([cur[0], cur[1]]) as usize;
            cur = &cur[2..];
            let need = n * BLOB_COL_SPEC_WIRE_SIZE;
            if cur.len() < need {
                return Err(SubstrateError::Internal(format!(
                    "dict v4 truncated at blob_columns body (need {need}, have {})",
                    cur.len()
                )));
            }
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let base = i * BLOB_COL_SPEC_WIRE_SIZE;
                let prop_key_id = u16::from_le_bytes([cur[base], cur[base + 1]]);
                let entity_byte = cur[base + 2];
                let Some(entity_kind) = EntityKind::from_u8(entity_byte) else {
                    return Err(SubstrateError::Internal(format!(
                        "dict v4: invalid entity_kind byte {entity_byte} at blob_column[{i}]"
                    )));
                };
                out.push(BlobColSpec {
                    prop_key_id,
                    entity_kind,
                });
            }
            // Advance cursor past the blob_columns payload so v5
            // counters parsing starts at the right offset.
            cur = &cur[need..];
            out
        } else {
            Vec::new()
        };

        // v5+: PersistedCounters block. u64 total_nodes + u64 total_edges
        // + 64×u64 label_counts + u16 count + (u16, u64) entries.
        // v1..=v4: counters = None → caller rebuilds.
        let counters: Option<PersistedCounters> = if version >= 5 {
            if cur.len() < COUNTERS_FIXED_PREFIX_SIZE {
                return Err(SubstrateError::Internal(format!(
                    "dict v5 truncated at counters block (need ≥ {}, have {})",
                    COUNTERS_FIXED_PREFIX_SIZE,
                    cur.len()
                )));
            }
            let total_live_nodes = u64::from_le_bytes(cur[0..8].try_into().unwrap());
            let total_live_edges = u64::from_le_bytes(cur[8..16].try_into().unwrap());
            let mut label_counts = [0u64; 64];
            for (i, count) in label_counts.iter_mut().enumerate() {
                let off = 16 + i * 8;
                *count = u64::from_le_bytes(cur[off..off + 8].try_into().unwrap());
            }
            let entries_off = 16 + 64 * 8;
            let n_entries =
                u16::from_le_bytes(cur[entries_off..entries_off + 2].try_into().unwrap()) as usize;
            let entries_start = entries_off + 2;
            let need = n_entries * EDGE_TYPE_COUNT_ENTRY_SIZE;
            if cur.len() < entries_start + need {
                return Err(SubstrateError::Internal(format!(
                    "dict v5 truncated at edge_type_counts body (need {need}, have {})",
                    cur.len().saturating_sub(entries_start)
                )));
            }
            let mut edge_type_counts = Vec::with_capacity(n_entries);
            for i in 0..n_entries {
                let base = entries_start + i * EDGE_TYPE_COUNT_ENTRY_SIZE;
                let type_id = u16::from_le_bytes(cur[base..base + 2].try_into().unwrap());
                let count = u64::from_le_bytes(cur[base + 2..base + 10].try_into().unwrap());
                edge_type_counts.push((type_id, count));
            }
            cur = &cur[entries_start + need..];

            // v6 : two per-edge-type × label histograms (target + source).
            // v5 files stop here ; we load empty Vecs and the caller
            // rebuilds on next scan (first flush bumps to v6).
            let (edge_type_target_label_counts, edge_type_source_label_counts) = if version >= 6 {
                let mut histograms: [Vec<(u16, [u64; 64])>; 2] = [Vec::new(), Vec::new()];
                for h in histograms.iter_mut() {
                    if cur.len() < 2 {
                        return Err(SubstrateError::Internal(
                            "dict v6 truncated at label-histogram n_entries".into(),
                        ));
                    }
                    let n = u16::from_le_bytes(cur[0..2].try_into().unwrap()) as usize;
                    cur = &cur[2..];
                    let need = n * EDGE_TYPE_LABEL_HIST_ENTRY_SIZE;
                    if cur.len() < need {
                        return Err(SubstrateError::Internal(format!(
                            "dict v6 truncated at label-histogram body (need {need}, have {})",
                            cur.len()
                        )));
                    }
                    h.reserve(n);
                    for i in 0..n {
                        let base = i * EDGE_TYPE_LABEL_HIST_ENTRY_SIZE;
                        let type_id = u16::from_le_bytes(cur[base..base + 2].try_into().unwrap());
                        let mut arr = [0u64; 64];
                        for (j, slot) in arr.iter_mut().enumerate() {
                            let off = base + 2 + j * 8;
                            *slot = u64::from_le_bytes(cur[off..off + 8].try_into().unwrap());
                        }
                        h.push((type_id, arr));
                    }
                    cur = &cur[need..];
                }
                let [target, source] = histograms;
                (target, source)
            } else {
                (Vec::new(), Vec::new())
            };
            // Touch `cur` so future versions can continue parsing from here.
            let _ = &cur;
            Some(PersistedCounters {
                total_live_nodes,
                total_live_edges,
                label_counts,
                edge_type_counts,
                edge_type_target_label_counts,
                edge_type_source_label_counts,
            })
        } else {
            None
        };

        Ok(Self {
            labels,
            edge_types,
            prop_keys,
            next_node_id,
            next_edge_id,
            next_engram_id,
            vec_columns,
            blob_columns,
            counters,
        })
    }

    /// Atomically write the snapshot to `path` (tmpfile + rename + fsync).
    pub fn persist(&self, path: &Path) -> SubstrateResult<()> {
        let bytes = self.to_bytes()?;
        let tmp = path.with_extension("dict.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, path)?;
        // fsync parent dir to make the rename durable.
        if let Some(parent) = path.parent() {
            if let Ok(d) = fs::File::open(parent) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    }

    /// Load a snapshot from `path`. Returns an empty snapshot if the file
    /// does not exist (fresh database).
    pub fn load(path: &Path) -> SubstrateResult<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let mut f = fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }
}

fn write_name_list(buf: &mut Vec<u8>, names: &[String]) -> SubstrateResult<()> {
    buf.extend_from_slice(&(names.len() as u16).to_le_bytes());
    for name in names {
        let bytes = name.as_bytes();
        if bytes.len() > u8::MAX as usize {
            return Err(SubstrateError::Internal(format!(
                "dict name too long ({} bytes > 255): {name:?}",
                bytes.len()
            )));
        }
        buf.push(bytes.len() as u8);
        buf.extend_from_slice(bytes);
    }
    Ok(())
}

fn read_name_list(cur: &mut &[u8]) -> SubstrateResult<Vec<String>> {
    if cur.len() < 2 {
        return Err(SubstrateError::Internal(
            "dict truncated (name list count)".into(),
        ));
    }
    let count = u16::from_le_bytes([cur[0], cur[1]]) as usize;
    *cur = &cur[2..];
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        if cur.is_empty() {
            return Err(SubstrateError::Internal(format!(
                "dict truncated at entry {i}/{count} (len byte)"
            )));
        }
        let len = cur[0] as usize;
        *cur = &cur[1..];
        if cur.len() < len {
            return Err(SubstrateError::Internal(format!(
                "dict truncated at entry {i}/{count} (payload needs {len}, have {})",
                cur.len()
            )));
        }
        let name = std::str::from_utf8(&cur[..len]).map_err(|e| {
            SubstrateError::Internal(format!("dict entry {i}/{count} is not UTF-8: {e}"))
        })?;
        out.push(name.to_string());
        *cur = &cur[len..];
    }
    Ok(out)
}

/// Bit-wise CRC-32C (Castagnoli) in software, no table — adequate for
/// small dicts (a few kilobytes). Replace with `crc32fast` if perf matters.
fn crc32c(bytes: &[u8]) -> u32 {
    const POLY: u32 = 0x82F6_3B78;
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_counters_are_one() {
        let s = DictSnapshot::default();
        assert_eq!(s.next_node_id, 1);
        assert_eq!(s.next_edge_id, 1);
    }

    #[test]
    fn roundtrip_empty() {
        let s = DictSnapshot::default();
        let bytes = s.to_bytes().unwrap();
        let back = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn roundtrip_populated() {
        let s = DictSnapshot {
            labels: vec!["Person".into(), "Company".into(), "Engineer".into()],
            edge_types: vec!["WORKS_AT".into(), "KNOWS".into()],
            prop_keys: vec!["name".into(), "age".into(), "role".into()],
            next_node_id: 12345,
            next_edge_id: 67_890,
            next_engram_id: 4321,
            vec_columns: Vec::new(),
            blob_columns: Vec::new(),
            counters: Some(PersistedCounters::default()),
        };
        let bytes = s.to_bytes().unwrap();
        let back = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn persist_and_load_via_disk() {
        let td = tempdir().unwrap();
        let path = td.path().join("substrate.dict");
        let s = DictSnapshot {
            labels: vec!["Node".into()],
            edge_types: vec!["REL".into(), "PARENT".into()],
            prop_keys: vec!["id".into()],
            next_node_id: 42,
            next_edge_id: 99,
            next_engram_id: 7,
            vec_columns: Vec::new(),
            blob_columns: Vec::new(),
            counters: Some(PersistedCounters::default()),
        };
        s.persist(&path).unwrap();
        let back = DictSnapshot::load(&path).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn roundtrip_v3_with_vec_columns() {
        let s = DictSnapshot {
            labels: vec!["Person".into()],
            edge_types: vec!["KNOWS".into()],
            prop_keys: vec!["embedding".into(), "data".into(), "score".into()],
            next_node_id: 1_000_000,
            next_edge_id: 2_000_000,
            next_engram_id: 42,
            vec_columns: vec![
                VecColSpec {
                    prop_key_id: 0,
                    entity_kind: EntityKind::Node,
                    dim: 80,
                    dtype: VecDType::F32,
                },
                VecColSpec {
                    prop_key_id: 1,
                    entity_kind: EntityKind::Edge,
                    dim: 384,
                    dtype: VecDType::F16,
                },
            ],
            blob_columns: Vec::new(),
            counters: Some(PersistedCounters::default()),
        };
        let bytes = s.to_bytes().unwrap();
        let back = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn v2_dict_loads_as_v3_with_empty_vec_columns() {
        // Simulate a pre-T16.7 dict on disk by writing a v2-format
        // byte stream by hand and verifying we load it as v3 with an
        // empty vec_columns list.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // v2

        // Three empty name lists.
        for _ in 0..3 {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_node_id
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_edge_id
        buf.extend_from_slice(&1u16.to_le_bytes()); // next_engram_id
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        let back = DictSnapshot::from_bytes(&buf).unwrap();
        assert_eq!(back.vec_columns, Vec::<VecColSpec>::new());
        assert_eq!(back.blob_columns, Vec::<BlobColSpec>::new());
        assert_eq!(back.next_engram_id, 1);
    }

    #[test]
    fn roundtrip_v4_with_blob_columns() {
        // Mixed vec + blob columns — the common shape post-Step 4d.
        let s = DictSnapshot {
            labels: vec!["Message".into()],
            edge_types: vec!["HAS_EVENT".into()],
            prop_keys: vec!["embedding".into(), "data".into(), "file_path".into()],
            next_node_id: 100,
            next_edge_id: 200,
            next_engram_id: 3,
            vec_columns: vec![VecColSpec {
                prop_key_id: 0,
                entity_kind: EntityKind::Node,
                dim: 80,
                dtype: VecDType::F32,
            }],
            blob_columns: vec![
                BlobColSpec {
                    prop_key_id: 1, // "data"
                    entity_kind: EntityKind::Node,
                },
                BlobColSpec {
                    prop_key_id: 1, // "data" on edges too
                    entity_kind: EntityKind::Edge,
                },
            ],
            counters: Some(PersistedCounters::default()),
        };
        let bytes = s.to_bytes().unwrap();
        let back = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn v3_dict_loads_as_v4_with_empty_blob_columns() {
        // Simulate a pre-Step-4 dict on disk by writing a v3-format
        // byte stream by hand (with a non-empty vec_columns list to
        // exercise the cursor advance) and verifying we load it as v4
        // with an empty blob_columns list.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // v3

        // Three empty name lists.
        for _ in 0..3 {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_node_id
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_edge_id
        buf.extend_from_slice(&1u16.to_le_bytes()); // next_engram_id

        // One vec_column spec to make sure the cursor reaches the
        // tail cleanly after the payload.
        buf.extend_from_slice(&1u16.to_le_bytes()); // vec_columns count
        buf.extend_from_slice(&0u16.to_le_bytes()); // prop_key_id
        buf.push(EntityKind::Node as u8);
        buf.push(VecDType::F32 as u8);
        buf.extend_from_slice(&16u32.to_le_bytes()); // dim

        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        let back = DictSnapshot::from_bytes(&buf).unwrap();
        assert_eq!(back.vec_columns.len(), 1);
        assert_eq!(back.blob_columns, Vec::<BlobColSpec>::new());
    }

    #[test]
    fn v4_blob_columns_truncated_body_is_rejected() {
        // Hand-craft a v4 header that claims 2 blob_columns but has
        // only 3 bytes in the payload region (enough for 1).
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        for _ in 0..3 {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // vec_columns count = 0
        buf.extend_from_slice(&2u16.to_le_bytes()); // blob_columns count = 2
        // Only 3 bytes of blob_columns payload (enough for 1 spec, not 2).
        buf.extend_from_slice(&0u16.to_le_bytes()); // prop_key_id
        buf.push(EntityKind::Node as u8);
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        let err = DictSnapshot::from_bytes(&buf).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("blob_columns") && msg.contains("truncated"),
            "got: {msg}"
        );
    }

    #[test]
    fn v4_blob_columns_bad_entity_kind_rejected() {
        // u8 entity_kind = 42 is not a valid EntityKind.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        for _ in 0..3 {
            buf.extend_from_slice(&0u16.to_le_bytes());
        }
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // vec_columns count = 0
        buf.extend_from_slice(&1u16.to_le_bytes()); // blob_columns count = 1
        buf.extend_from_slice(&0u16.to_le_bytes()); // prop_key_id
        buf.push(42); // bogus entity_kind
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        let err = DictSnapshot::from_bytes(&buf).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("entity_kind") && msg.contains("blob_column"),
            "got: {msg}"
        );
    }

    #[test]
    fn missing_file_loads_empty() {
        let td = tempdir().unwrap();
        let path = td.path().join("substrate.dict");
        let back = DictSnapshot::load(&path).unwrap();
        assert_eq!(back, DictSnapshot::default());
    }

    #[test]
    fn crc_failure_is_detected() {
        let s = DictSnapshot {
            labels: vec!["A".into()],
            ..Default::default()
        };
        let mut bytes = s.to_bytes().unwrap();
        // Flip a bit in the middle.
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        let err = DictSnapshot::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("CRC") || msg.contains("crc"), "got: {msg}");
    }

    #[test]
    fn magic_mismatch_detected() {
        let mut bytes = DictSnapshot::default().to_bytes().unwrap();
        bytes[0] = 0;
        let err = DictSnapshot::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("magic"), "got: {msg}");
    }

    #[test]
    fn too_many_labels_rejected() {
        let s = DictSnapshot {
            labels: (0..65).map(|i| format!("L{i}")).collect(),
            ..Default::default()
        };
        let err = s.to_bytes().unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("64"), "got: {msg}");
    }

    #[test]
    fn zero_counter_rejected() {
        // Manually craft bytes with a zero counter — must fail.
        let s = DictSnapshot::default();
        let mut bytes = s.to_bytes().unwrap();
        // Locate and corrupt: the two u64 counters are the 16 bytes before
        // padding + crc. Easiest: decode and check error path by patching
        // the bytes.
        let counter_start = bytes.len() - 4 - /*pad≤3*/0;
        // Simpler approach — search for the u64(1) little-endian pattern.
        let pattern = 1u64.to_le_bytes();
        let pos = bytes
            .windows(8)
            .rposition(|w| w == pattern)
            .expect("should find u64(1) for a counter");
        bytes[pos..pos + 8].copy_from_slice(&0u64.to_le_bytes());
        // Recompute CRC over the tampered body so we hit the zero-counter
        // invariant, not the CRC guard.
        let body_end = bytes.len() - 4;
        let new_crc = crc32c(&bytes[..body_end]);
        bytes[body_end..].copy_from_slice(&new_crc.to_le_bytes());
        let _ = counter_start; // silence unused if we ever skip the search
        let err = DictSnapshot::from_bytes(&bytes).unwrap_err();
        let msg = format!("{err:?}");
        // After v5 bump (T17h T1), the rposition search for u64(1) may
        // now overlap next_engram_id + vec_count + blob_count + start of
        // counters block (all zeros for a default dict), so the corrupted
        // 8-byte slice can land on any of the three counters depending on
        // the precise byte layout. All three are legitimate targets of
        // this invariant check.
        assert!(
            msg.contains("next_node_id")
                || msg.contains("next_edge_id")
                || msg.contains("next_engram_id"),
            "got: {msg}"
        );
    }

    /// Build a synthetic v1 dict on the wire — exactly the layout that
    /// shipped before T7 Step 6 (no `next_engram_id`). The new loader
    /// must accept it and default the engram allocator to 1.
    fn build_v1_bytes(
        labels: &[&str],
        edge_types: &[&str],
        prop_keys: &[&str],
        next_node_id: u64,
        next_edge_id: u64,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
        let write_names = |buf: &mut Vec<u8>, names: &[&str]| {
            buf.extend_from_slice(&(names.len() as u16).to_le_bytes());
            for n in names {
                let b = n.as_bytes();
                buf.push(b.len() as u8);
                buf.extend_from_slice(b);
            }
        };
        write_names(&mut buf, labels);
        write_names(&mut buf, edge_types);
        write_names(&mut buf, prop_keys);
        buf.extend_from_slice(&next_node_id.to_le_bytes());
        buf.extend_from_slice(&next_edge_id.to_le_bytes());
        // No next_engram_id.
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    #[test]
    fn v1_dict_loads_with_default_next_engram_id() {
        let bytes = build_v1_bytes(&["Person"], &["KNOWS", "COACT"], &["name"], 17, 42);
        let snap = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snap.labels, vec!["Person".to_string()]);
        assert_eq!(
            snap.edge_types,
            vec!["KNOWS".to_string(), "COACT".to_string()]
        );
        assert_eq!(snap.prop_keys, vec!["name".to_string()]);
        assert_eq!(snap.next_node_id, 17);
        assert_eq!(snap.next_edge_id, 42);
        // v1 has no engram allocator → defaults to 1.
        assert_eq!(snap.next_engram_id, 1);
    }

    #[test]
    fn v2_dict_roundtrips_engram_counter() {
        let s = DictSnapshot {
            labels: vec!["A".into()],
            edge_types: vec!["E".into()],
            prop_keys: vec!["k".into()],
            next_node_id: 5,
            next_edge_id: 6,
            next_engram_id: 9999,
            vec_columns: Vec::new(),
            blob_columns: Vec::new(),
            counters: Some(PersistedCounters::default()),
        };
        let bytes = s.to_bytes().unwrap();
        let back = DictSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(back.next_engram_id, 9999);
        assert_eq!(s, back);
    }

    #[test]
    fn v3_zero_engram_counter_rejected() {
        // Manually craft v-current bytes with next_engram_id = 0.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        // Empty name lists
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_node_id
        buf.extend_from_slice(&1u64.to_le_bytes()); // next_edge_id
        buf.extend_from_slice(&0u16.to_le_bytes()); // next_engram_id = 0 (illegal)
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        let err = DictSnapshot::from_bytes(&buf).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("next_engram_id"), "got: {msg}");
    }

    #[test]
    fn unknown_version_rejected() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes()); // future / unknown version
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
        let crc = crc32c(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        let err = DictSnapshot::from_bytes(&buf).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("999"), "got: {msg}");
    }
}
