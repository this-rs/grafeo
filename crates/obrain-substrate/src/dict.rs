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
const FORMAT_VERSION: u32 = 3;

/// Wire size of one `VecColSpec` in the v3 tail of the dict.
const VEC_COL_SPEC_WIRE_SIZE: usize = 2 + 1 + 1 + 4;

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
        // Accept v1, v2, and current version:
        // * v1 → no `next_engram_id`, no `vec_columns`.
        // * v2 → `next_engram_id` present, no `vec_columns`.
        // * v3 → all fields present.
        if !(version == 1 || version == 2 || version == FORMAT_VERSION) {
            return Err(SubstrateError::Internal(format!(
                "dict version {version} unsupported (expected 1, 2, or {FORMAT_VERSION})"
            )));
        }

        // Separate the CRC tail and validate.
        let body = &bytes[..bytes.len() - 4];
        let stored_crc =
            u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
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
            out
        } else {
            Vec::new()
        };

        Ok(Self {
            labels,
            edge_types,
            prop_keys,
            next_node_id,
            next_edge_id,
            next_engram_id,
            vec_columns,
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
            SubstrateError::Internal(format!(
                "dict entry {i}/{count} is not UTF-8: {e}"
            ))
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
        assert_eq!(back.next_engram_id, 1);
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
        assert!(
            msg.contains("next_node_id") || msg.contains("next_edge_id"),
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
        let bytes = build_v1_bytes(
            &["Person"],
            &["KNOWS", "COACT"],
            &["name"],
            17,
            42,
        );
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
