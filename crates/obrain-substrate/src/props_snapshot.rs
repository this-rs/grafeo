//! # Properties sidecar snapshot
//!
//! **Transitional solution** for property durability, shipped ahead of the
//! full property-pages subsystem (RFC-SUBSTRATE §Properties).
//!
//! ## Problem it solves
//!
//! `SubstrateStore::set_node_property` / `set_edge_property` originally
//! wrote only to the in-memory `DashMap<_, _InMem>`. The WAL variant
//! `PropSet` exists in [`crate::wal::WalPayload`] but is not emitted by
//! any code path, and `flush()` persists only the dict (registries +
//! slot counters) and the Nodes / Edges mmap zones. Properties were
//! silently discarded on close — a 30-minute `neo4j2obrain` Stage 2 run
//! (477 805 embeddings × 1.5 KB = 700 MB in RSS) left disk unchanged at
//! 385 MB.
//!
//! ## What this module adds
//!
//! A flat bincode snapshot written atomically to
//! `<substrate-dir>/substrate.props` at every [`SubstrateStore::flush`]
//! boundary, and loaded at open. Durability is **at-flush-granularity**:
//! a crash between two flushes loses the interim properties. Callers
//! that want tighter durability (e.g. the hub's per-message write path)
//! should call `flush()` after each batch.
//!
//! ## What this module does NOT solve
//!
//! * WAL-backed property durability — full `PropSet` replay is T17's
//!   scope (tracked in `RFC-SUBSTRATE §Properties`).
//! * Property pages in the mmap — also T17.
//! * Delta persistence — every flush rewrites the full snapshot. At
//!   4.5 M nodes × 1.5 KB embedding the snapshot is ~7 GB per flush.
//!   The migration and warden flush infrequently; runtime callers that
//!   churn properties will want the T17 implementation.
//!
//! ## On-disk format (v1)
//!
//! ```text
//! [u32 magic = 0x5052_4F50]   // "PROP" little-endian (0x50_52_4F_50 → 'P','R','O','P')
//! [u32 version = 1]
//! [bincode(PropertiesSnapshotV1)]
//! [u32 crc32c of the bincode payload]
//! ```
//!
//! The CRC is the trailing 4 bytes; everything before it is the magic +
//! version header plus the bincode-serialized body. A corrupted file
//! (magic mismatch, CRC mismatch, bincode decode error) is treated as
//! "no snapshot" — the loader returns `Ok(Default::default())` and
//! tracing a warning rather than refusing to open the store.

use crate::{SubstrateError, SubstrateResult};
use bincode::config::{Configuration, Fixint, LittleEndian};
use obrain_common::types::{PropertyKey, Value};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Fast table-driven CRC-32 (IEEE polynomial) over the whole buffer.
/// Uses the `crc32fast` workspace dep — measured at ~2 GB/s on the M2
/// which keeps the CRC step well below the bincode encode cost even at
/// 7 GB snapshots.
fn crc32_fast(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

/// Sidecar filename inside the substrate directory.
pub const PROPS_FILENAME: &str = "substrate.props";

const MAGIC: u32 = 0x5052_4F50; // b"PROP" little-endian
const FORMAT_VERSION: u32 = 1;

fn bincode_config() -> Configuration<LittleEndian, Fixint> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

/// One entity's property map, serialised as a flat Vec to sidestep
/// `PropertyMap`'s lack of serde derive.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PropEntry {
    /// Opaque id: for nodes this is `NodeId.as_u64()`, for edges `EdgeId.as_u64()`.
    pub id: u64,
    /// `(key, value)` pairs. Keys are stored as `String` (the interned
    /// name) rather than `PropertyKey` — the key registry is rebuilt at
    /// open time from the dict.
    pub props: Vec<(String, Value)>,
}

/// v1 snapshot of all node and edge properties at one point in time.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PropertiesSnapshotV1 {
    pub nodes: Vec<PropEntry>,
    pub edges: Vec<PropEntry>,
}

impl PropertiesSnapshotV1 {
    /// Encode to bytes with magic, version, and trailing CRC32C.
    fn to_bytes(&self) -> SubstrateResult<Vec<u8>> {
        let body = bincode::serde::encode_to_vec(self, bincode_config())
            .map_err(|e| SubstrateError::Internal(format!("props encode: {e}")))?;
        let mut out = Vec::with_capacity(4 + 4 + body.len() + 4);
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&body);
        let crc = crc32_fast(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        Ok(out)
    }

    /// Decode from bytes. Validates magic, version, CRC. Returns
    /// `Ok(Default)` with a tracing warning on any recoverable error.
    fn from_bytes(bytes: &[u8]) -> Self {
        // Minimum: 4 (magic) + 4 (version) + 0 (empty body possible) + 4 (crc) = 12.
        if bytes.len() < 12 {
            tracing::warn!(
                "props snapshot too short ({} bytes); treating as empty",
                bytes.len()
            );
            return Self::default();
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != MAGIC {
            tracing::warn!(
                "props snapshot magic mismatch: got {magic:#x}, expected {MAGIC:#x}; treating as empty"
            );
            return Self::default();
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != FORMAT_VERSION {
            tracing::warn!(
                "props snapshot version {version} unsupported (expected {FORMAT_VERSION}); treating as empty"
            );
            return Self::default();
        }
        let body = &bytes[..bytes.len() - 4];
        let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        let computed_crc = crc32_fast(body);
        if stored_crc != computed_crc {
            tracing::warn!(
                "props snapshot CRC mismatch: stored {stored_crc:#x}, computed {computed_crc:#x}; treating as empty"
            );
            return Self::default();
        }
        match bincode::serde::decode_from_slice::<Self, _>(&body[8..], bincode_config()) {
            Ok((snap, _)) => snap,
            Err(e) => {
                tracing::warn!("props snapshot decode error: {e}; treating as empty");
                Self::default()
            }
        }
    }

    /// Atomically write the snapshot to `<path>`. Uses a tmp file + rename
    /// so a crash during write never leaves a half-written `.props` file
    /// for the next open to trip over.
    pub fn persist(&self, path: &Path) -> SubstrateResult<()> {
        let bytes = self.to_bytes()?;
        let tmp_path = path.with_extension("props.tmp");
        std::fs::write(&tmp_path, &bytes).map_err(SubstrateError::Io)?;
        std::fs::rename(&tmp_path, path).map_err(SubstrateError::Io)?;
        Ok(())
    }

    /// Load the snapshot from `<path>`. Returns `Default` if the file
    /// does not exist (fresh store).
    pub fn load(path: &Path) -> SubstrateResult<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path).map_err(SubstrateError::Io)?;
        Ok(Self::from_bytes(&bytes))
    }
}

/// Helper: turn one entity's `PropertyMap` into a flat `Vec<(String, Value)>`.
pub fn map_to_entries(
    map: &obrain_common::types::PropertyMap,
) -> Vec<(String, Value)> {
    map.iter()
        .map(|(k, v)| (k.as_str().to_string(), v.clone()))
        .collect()
}

/// Helper: turn a flat `Vec<(String, Value)>` back into a `PropertyMap`.
pub fn entries_to_map(
    entries: Vec<(String, Value)>,
) -> obrain_common::types::PropertyMap {
    let mut m = obrain_common::types::PropertyMap::with_capacity(entries.len());
    for (k, v) in entries {
        m.insert(PropertyKey::new(k), v);
    }
    m
}

// ---------------------------------------------------------------------------
// Streaming writer — DashMap-bypass path for bulk migrations
// ---------------------------------------------------------------------------
//
// The batch writer above rewrites the whole `substrate.props` sidecar
// from an in-memory `DashMap<_, PropertyMap>` on every flush. That's
// fine for incremental hub traffic (few-MB deltas) but blows out when
// bulk-migrating a source whose properties are several GB — every
// `set_node_property` call pins the value in the DashMap forever, so
// RSS grows monotonically until the OOM killer fires.
//
// [`PropertiesStreamingWriter`] sidesteps this. It writes one
// `PropEntry` at a time, directly to `substrate.props.stream.tmp`, in
// the same on-disk format the batch writer produces. The node count,
// edge count, and trailing CRC32 are patched in at [`Self::finish`]
// time (seek-back-and-patch for the counts, single streaming pass
// through the finished file for the CRC).
//
// The output is byte-identical to what the batch writer would have
// produced for the same (nodes, edges) sequence — see
// `streaming_matches_batch_format` in the tests for the invariant.
//
// ## Scope
//
// This is a **transitional tool** for `obrain-migrate` until the T17
// property-pages subsystem lands. Runtime callers should keep using
// `SubstrateStore::set_node_property` / `set_edge_property` and rely
// on `flush()`'s batch write.
//
// ## Coexistence with `flush()`
//
// The migrator MUST call `finish()` **after** any `substrate.flush()`
// calls. `flush()` → `persist_properties()` writes `substrate.props`
// (via a `substrate.props.tmp` intermediary); if `finish()` ran first,
// the flush would clobber the streamed file with an empty snapshot
// (the migration path never populates the DashMap).
//
// Recommended order:
//   1. Open streaming writer.
//   2. Run all migration phases (writer receives appends).
//   3. Call `substrate.flush()` (persists dict + empty props snapshot).
//   4. Call `writer.finish()` — atomic rename wins the last write.

/// Writer phase: enforced by a small state machine so
/// `append_node` / `append_edge` land on the right side of the
/// nodes/edges delimiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterPhase {
    /// Accepting `append_node`.
    Nodes,
    /// Accepting `append_edge`.
    Edges,
    /// Terminal — all counts patched, CRC appended, tmp renamed into place.
    Finished,
}

/// Serialization-time shim: lets us encode a single `PropEntry` from
/// borrowed props without cloning the `Vec<(String, Value)>`. Bincode
/// serializes `&[T]` identically to `Vec<T>` (u64 LE length prefix +
/// elements), so the on-disk layout matches [`PropEntry`] exactly.
#[derive(Serialize)]
struct PropEntryRef<'a> {
    id: u64,
    props: &'a [(String, Value)],
}

/// Streaming writer for `substrate.props`. See module docs for the
/// rationale and the coexistence contract with `SubstrateStore::flush`.
pub struct PropertiesStreamingWriter {
    /// Destination path (what `load()` reads). Filled in at `finish()`.
    final_path: PathBuf,
    /// Temp file being streamed into.
    tmp_path: PathBuf,
    /// Buffered writer over the tmp file. Wrapped in `Option` so
    /// `finish()` can take ownership without conflicting with the
    /// `Drop` impl (which needs access to the same fields to clean up
    /// the tmp file on abnormal exit).
    writer: Option<BufWriter<File>>,
    /// Byte offset in the tmp file where the `nodes.len()` u64 LE
    /// placeholder lives. Filled at header time.
    nodes_len_offset: u64,
    /// Byte offset of the `edges.len()` u64 LE placeholder. Set when
    /// `begin_edges()` runs (implicitly on first `append_edge`, or
    /// explicitly on `finish()` if no edges were appended).
    edges_len_offset: Option<u64>,
    /// Monotonically incremented on every successful `append_*`. Patched
    /// into the placeholders at `finish()`.
    nodes_written: u64,
    edges_written: u64,
    /// State machine guard.
    phase: WriterPhase,
}

impl PropertiesStreamingWriter {
    /// Open a fresh streaming writer targeting `<final_path>`.
    ///
    /// * `final_path` — usually `<substrate-dir>/substrate.props`. The
    ///   writer creates `<final_path>.stream.tmp` and renames it on
    ///   `finish()`.
    ///
    /// Writes the `[MAGIC][VERSION]` header and a zero placeholder for
    /// the nodes vec length; subsequent `append_node` calls stream one
    /// `PropEntry` each.
    pub fn open(final_path: impl AsRef<Path>) -> SubstrateResult<Self> {
        let final_path = final_path.as_ref().to_path_buf();
        // Distinct suffix from `substrate.props.tmp` (used by
        // `PropertiesSnapshotV1::persist`) so a concurrent
        // `persist_properties()` can't collide on the tmp name.
        let tmp_path = final_path.with_extension("props.stream.tmp");

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .read(true)
            .open(&tmp_path)
            .map_err(SubstrateError::Io)?;
        // 1 MiB buffer — big enough that most PropEntry encodes land in
        // one syscall, small enough to not balloon RSS on a 4.5M-node
        // run (the write amplifier is 1× the buffer size, not N).
        let mut writer: BufWriter<File> = BufWriter::with_capacity(1 << 20, file);

        // Header.
        writer
            .write_all(&MAGIC.to_le_bytes())
            .map_err(SubstrateError::Io)?;
        writer
            .write_all(&FORMAT_VERSION.to_le_bytes())
            .map_err(SubstrateError::Io)?;

        // Placeholder for `nodes.len()` (u64 LE, fixint). Patched in
        // `finish()`.
        let nodes_len_offset: u64 = 8;
        writer
            .write_all(&0u64.to_le_bytes())
            .map_err(SubstrateError::Io)?;

        Ok(Self {
            final_path,
            tmp_path,
            writer: Some(writer),
            nodes_len_offset,
            edges_len_offset: None,
            nodes_written: 0,
            edges_written: 0,
            phase: WriterPhase::Nodes,
        })
    }

    /// Convenience accessor — panics if called after `finish()` or on a
    /// writer whose inner `BufWriter` has been stolen for any reason.
    /// All internal methods use this to get a `&mut BufWriter<File>`.
    fn writer_mut(&mut self) -> &mut BufWriter<File> {
        self.writer
            .as_mut()
            .expect("streaming writer already finalised or moved")
    }

    /// Append one node's property map. No-op if `props.is_empty()` —
    /// matches `persist_properties()`'s "skip entries with no properties"
    /// invariant so the streamed file is byte-compatible with the batch
    /// writer's output for the same input.
    pub fn append_node(
        &mut self,
        id: u64,
        props: &[(String, Value)],
    ) -> SubstrateResult<()> {
        if !matches!(self.phase, WriterPhase::Nodes) {
            return Err(SubstrateError::Internal(format!(
                "append_node after phase transition ({:?})",
                self.phase
            )));
        }
        if props.is_empty() {
            return Ok(());
        }
        let entry = PropEntryRef { id, props };
        let cfg = bincode_config();
        let w = self.writer_mut();
        bincode::serde::encode_into_std_write(&entry, w, cfg)
            .map_err(|e| SubstrateError::Internal(format!("stream encode: {e}")))?;
        self.nodes_written += 1;
        Ok(())
    }

    /// Explicitly close the nodes block and open the edges block.
    /// Called automatically on the first `append_edge`; exposed so
    /// callers can force the transition for nodes-only migrations whose
    /// final file should still declare `[u64 LE 0]` for the edge count.
    pub fn begin_edges(&mut self) -> SubstrateResult<()> {
        match self.phase {
            WriterPhase::Nodes => {}
            WriterPhase::Edges => return Ok(()), // idempotent
            WriterPhase::Finished => {
                return Err(SubstrateError::Internal(
                    "begin_edges called after finish".into(),
                ));
            }
        }
        let w = self.writer_mut();
        // Flush so the underlying File's cursor reflects everything we
        // have written so far; we record the absolute offset of the
        // edges.len() placeholder for the later patch.
        w.flush().map_err(SubstrateError::Io)?;
        let pos = w.get_mut().stream_position().map_err(SubstrateError::Io)?;
        self.edges_len_offset = Some(pos);
        let w = self.writer_mut();
        w.write_all(&0u64.to_le_bytes()).map_err(SubstrateError::Io)?;
        self.phase = WriterPhase::Edges;
        Ok(())
    }

    /// Append one edge's property map. Transitions into the edges
    /// phase automatically on first call.
    pub fn append_edge(
        &mut self,
        id: u64,
        props: &[(String, Value)],
    ) -> SubstrateResult<()> {
        if matches!(self.phase, WriterPhase::Nodes) {
            self.begin_edges()?;
        }
        if !matches!(self.phase, WriterPhase::Edges) {
            return Err(SubstrateError::Internal(format!(
                "append_edge in phase {:?}",
                self.phase
            )));
        }
        if props.is_empty() {
            return Ok(());
        }
        let entry = PropEntryRef { id, props };
        let cfg = bincode_config();
        let w = self.writer_mut();
        bincode::serde::encode_into_std_write(&entry, w, cfg)
            .map_err(|e| SubstrateError::Internal(format!("stream encode: {e}")))?;
        self.edges_written += 1;
        Ok(())
    }

    /// Running totals, mostly for progress logging.
    pub fn counts(&self) -> (u64, u64) {
        (self.nodes_written, self.edges_written)
    }

    /// Finalise the file:
    ///   1. Force the edges header to exist (for nodes-only runs).
    ///   2. Patch node / edge counts into their placeholders.
    ///   3. Stream-hash the full file to compute CRC32.
    ///   4. Append the CRC.
    ///   5. Fsync and atomically rename tmp → final.
    ///
    /// The loader accepts the file iff magic + version match and the
    /// CRC validates, so we must not reorder step 4 ahead of 3.
    pub fn finish(mut self) -> SubstrateResult<()> {
        // (1) Always emit an edges-length slot, even if empty.
        if self.edges_len_offset.is_none() {
            self.begin_edges()?;
        }
        let edges_len_offset = self
            .edges_len_offset
            .expect("begin_edges invariant: edges_len_offset set");

        // Steal the BufWriter out of the Option so the Drop impl
        // (which runs on early-return) sees `writer = None` and skips
        // its cleanup — we're taking responsibility for finalisation
        // here.
        let bw = self
            .writer
            .take()
            .expect("writer already taken (double finish?)");

        // Pre-patch: make sure every buffered byte hits the File before
        // we seek / read it back.
        let mut file = bw
            .into_inner()
            .map_err(|e| SubstrateError::Internal(format!("buf into_inner: {e}")))?;

        // (2) Patch counts.
        file.seek(SeekFrom::Start(self.nodes_len_offset))
            .map_err(SubstrateError::Io)?;
        file.write_all(&self.nodes_written.to_le_bytes())
            .map_err(SubstrateError::Io)?;
        file.seek(SeekFrom::Start(edges_len_offset))
            .map_err(SubstrateError::Io)?;
        file.write_all(&self.edges_written.to_le_bytes())
            .map_err(SubstrateError::Io)?;

        // Make counts durable before the CRC scan reads them back.
        file.sync_data().map_err(SubstrateError::Io)?;

        // (3) Streaming CRC over the entire pre-CRC body.
        file.seek(SeekFrom::Start(0)).map_err(SubstrateError::Io)?;
        let mut hasher = crc32fast::Hasher::new();
        // 64 KiB read buffer is a sweet spot on M2: comfortably above
        // the page size, well under L1d (192 KB).
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf).map_err(SubstrateError::Io)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let crc = hasher.finalize();

        // (4) Append CRC.
        file.seek(SeekFrom::End(0)).map_err(SubstrateError::Io)?;
        file.write_all(&crc.to_le_bytes())
            .map_err(SubstrateError::Io)?;
        file.sync_data().map_err(SubstrateError::Io)?;
        drop(file);

        // (5) Atomic rename.
        std::fs::rename(&self.tmp_path, &self.final_path)
            .map_err(SubstrateError::Io)?;

        self.phase = WriterPhase::Finished;
        Ok(())
    }
}

impl Drop for PropertiesStreamingWriter {
    fn drop(&mut self) {
        // If the writer was not explicitly finished — abnormal exit,
        // ?-propagated error, caller forgot to call finish — nuke the
        // tmp file so the next migration run doesn't mistake a partial
        // prefix for a valid snapshot.
        //
        // We key on `writer.is_some()` rather than `phase != Finished`:
        // `finish()` takes the BufWriter out of the Option, so any
        // Drop after a successful finish sees `None` and leaves the
        // now-renamed final file alone.
        if self.writer.is_some() {
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_empty() {
        let s = PropertiesSnapshotV1::default();
        let bytes = s.to_bytes().unwrap();
        let back = PropertiesSnapshotV1::from_bytes(&bytes);
        assert_eq!(back, s);
    }

    #[test]
    fn roundtrip_with_vector_embedding() {
        let mut s = PropertiesSnapshotV1::default();
        s.nodes.push(PropEntry {
            id: 42,
            props: vec![
                (
                    "_st_embedding".to_string(),
                    Value::Vector(Arc::from(vec![0.1_f32, 0.2, -0.3, 0.5].into_boxed_slice())),
                ),
                ("title".to_string(), Value::String("hello".into())),
            ],
        });
        s.edges.push(PropEntry {
            id: 7,
            props: vec![("created_at".to_string(), Value::Int64(1_700_000_000))],
        });
        let bytes = s.to_bytes().unwrap();
        let back = PropertiesSnapshotV1::from_bytes(&bytes);
        assert_eq!(back, s);
    }

    #[test]
    fn persist_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(PROPS_FILENAME);
        let mut s = PropertiesSnapshotV1::default();
        s.nodes.push(PropEntry {
            id: 1,
            props: vec![("k".to_string(), Value::Int64(99))],
        });
        s.persist(&path).unwrap();
        let back = PropertiesSnapshotV1::load(&path).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn missing_file_yields_default() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does-not-exist.props");
        let back = PropertiesSnapshotV1::load(&path).unwrap();
        assert_eq!(back, PropertiesSnapshotV1::default());
    }

    #[test]
    fn corrupt_magic_yields_default() {
        // Valid-length but wrong magic.
        let mut bytes = vec![0u8; 12];
        bytes[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        let back = PropertiesSnapshotV1::from_bytes(&bytes);
        assert_eq!(back, PropertiesSnapshotV1::default());
    }

    #[test]
    fn corrupt_crc_yields_default() {
        let s = PropertiesSnapshotV1::default();
        let mut bytes = s.to_bytes().unwrap();
        // Flip the last byte of the CRC.
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let back = PropertiesSnapshotV1::from_bytes(&bytes);
        assert_eq!(back, PropertiesSnapshotV1::default());
    }

    // --- PropertiesStreamingWriter -----------------------------------------

    /// Build a small, mixed-type input for the streaming / batch parity tests.
    fn sample_snapshot() -> PropertiesSnapshotV1 {
        let mut s = PropertiesSnapshotV1::default();
        s.nodes.push(PropEntry {
            id: 1,
            props: vec![
                ("title".to_string(), Value::String("alpha".into())),
                ("rank".to_string(), Value::Int64(42)),
            ],
        });
        s.nodes.push(PropEntry {
            id: 2,
            props: vec![(
                "vec".to_string(),
                Value::Vector(Arc::from(
                    vec![0.1_f32, -0.2, 0.3].into_boxed_slice(),
                )),
            )],
        });
        s.edges.push(PropEntry {
            id: 100,
            props: vec![("weight".to_string(), Value::Float64(0.75))],
        });
        s
    }

    #[test]
    fn streaming_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(PROPS_FILENAME);

        let mut w = PropertiesStreamingWriter::open(&path).unwrap();
        let src = sample_snapshot();
        for n in &src.nodes {
            w.append_node(n.id, &n.props).unwrap();
        }
        for e in &src.edges {
            w.append_edge(e.id, &e.props).unwrap();
        }
        assert_eq!(w.counts(), (src.nodes.len() as u64, src.edges.len() as u64));
        w.finish().unwrap();

        let back = PropertiesSnapshotV1::load(&path).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn streaming_nodes_only_then_finish() {
        // Nodes-only migration path: finish() must still emit the
        // edges-length slot (0) so the loader doesn't truncate.
        let dir = tempdir().unwrap();
        let path = dir.path().join(PROPS_FILENAME);

        let mut w = PropertiesStreamingWriter::open(&path).unwrap();
        w.append_node(
            7,
            &[("k".to_string(), Value::Int64(99))],
        )
        .unwrap();
        w.finish().unwrap();

        let back = PropertiesSnapshotV1::load(&path).unwrap();
        assert_eq!(back.nodes.len(), 1);
        assert_eq!(back.nodes[0].id, 7);
        assert!(back.edges.is_empty());
    }

    #[test]
    fn streaming_matches_batch_format() {
        // The byte-identical invariant: for the same (nodes, edges)
        // sequence, the streaming writer and the batch writer produce
        // identical files. This guarantees load() can consume either
        // blindly.
        let dir = tempdir().unwrap();
        let batch_path = dir.path().join("batch.props");
        let stream_path = dir.path().join("stream.props");

        let src = sample_snapshot();
        src.persist(&batch_path).unwrap();

        let mut w = PropertiesStreamingWriter::open(&stream_path).unwrap();
        for n in &src.nodes {
            w.append_node(n.id, &n.props).unwrap();
        }
        for e in &src.edges {
            w.append_edge(e.id, &e.props).unwrap();
        }
        w.finish().unwrap();

        let batch_bytes = std::fs::read(&batch_path).unwrap();
        let stream_bytes = std::fs::read(&stream_path).unwrap();
        assert_eq!(
            batch_bytes, stream_bytes,
            "streamed file must be byte-identical to batch output"
        );
    }

    #[test]
    fn streaming_skips_empty_props() {
        // append_node with empty props MUST be a no-op — persist_properties()
        // skips empty maps, and the streaming writer preserves byte-identity.
        let dir = tempdir().unwrap();
        let stream_path = dir.path().join("stream.props");
        let batch_path = dir.path().join("batch.props");

        let mut src = PropertiesSnapshotV1::default();
        src.nodes.push(PropEntry {
            id: 1,
            props: vec![("k".into(), Value::Int64(1))],
        });
        // Note: src has no "empty node" entry — persist_properties()
        // strips them, so the batch output is exactly this shape.
        src.persist(&batch_path).unwrap();

        let mut w = PropertiesStreamingWriter::open(&stream_path).unwrap();
        w.append_node(99, &[]).unwrap(); // must be dropped
        w.append_node(1, &[("k".into(), Value::Int64(1))]).unwrap();
        w.append_edge(5, &[]).unwrap(); // must be dropped
        w.finish().unwrap();

        assert_eq!(
            std::fs::read(&batch_path).unwrap(),
            std::fs::read(&stream_path).unwrap()
        );
    }

    #[test]
    fn streaming_drop_cleans_up_tmp() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(PROPS_FILENAME);
        let tmp = path.with_extension("props.stream.tmp");

        {
            let mut w = PropertiesStreamingWriter::open(&path).unwrap();
            w.append_node(1, &[("k".into(), Value::Int64(1))]).unwrap();
            // Drop without finish() — should clean tmp.
            drop(w);
        }

        assert!(!tmp.exists(), "tmp file must be removed on drop");
        assert!(!path.exists(), "final file must not exist");
    }
}
