//! `BlobColumnRegistry` — runtime routing of oversized `Value::String`
//! and `Value::Bytes` properties to the two-file mmap blob-column
//! zones defined in [`crate::blob_column`].
//!
//! # Role in the stack
//!
//! ```text
//! SubstrateStore::set_node_property(id, key, Value::String(s))
//!     └──► if s.len() > BLOB_COLUMN_THRESHOLD_BYTES
//!            └──► BlobColumnRegistry::write(key, Node, id, s.as_bytes())
//!                   └──► get-or-create BlobColumnWriter for (key, Node)
//!                          └──► write_slot(id as u32, bytes)
//! ```
//!
//! The registry is the sole gatekeeper of the `substrate.blobcol.*`
//! zone family, mirroring [`crate::vec_column_registry::VecColumnRegistry`]:
//!
//! 1. **One writer per (key, entity)** — kept open for the store's
//!    lifetime so a migration loop does not pay per-write `mmap` costs.
//! 2. **Read consistency** — reads issued after a write see the new
//!    payload via the writer's own mmap, even before the header CRCs
//!    have been `sync`'d.
//! 3. **Durability at flush cadence** — [`Self::sync_all`] walks every
//!    writer and `sync`s it, computing both CRCs and `msync`+`fsync`'ing
//!    the two files of each column.
//!
//! # Crash safety
//!
//! Identical model to the vec-column registry: writes are visible in
//! process but not durable until the next `flush`; a crash between
//! flushes loses writes back to the last sync-point, and on reopen the
//! column's reader rejects the stale header via the CRC check and the
//! zone degrades to "absent".

use std::sync::Arc;

use dashmap::DashMap;
use obrain_common::types::PropertyKey;
use parking_lot::Mutex;

use crate::blob_column::{BlobColSpec, BlobColumnWriter};
use crate::error::{SubstrateError, SubstrateResult};
use crate::file::SubstrateFile;
use crate::vec_column::EntityKind;

/// Default routing threshold for scalar string / bytes payloads:
/// values shorter than this land in the in-memory property map (and
/// thus in `substrate.props` as before), values longer are routed to a
/// blob column. 256 B was picked to cover every chat/event `data`
/// payload on PO (avg 1.4 KiB) while leaving small identifiers
/// (`file_path` avg 97 B, short `name` / `id` keys) in the scalar path.
pub const BLOB_COLUMN_THRESHOLD_BYTES: usize = 256;

/// Routing table + open-writer cache for blob-typed properties.
pub(crate) struct BlobColumnRegistry {
    /// `(PropertyKey, EntityKind) -> BlobColSpec`. Populated on first
    /// oversized write for a given key, and pre-populated on open via
    /// [`Self::hydrate_from_dict`] from the persisted
    /// `DictSnapshot.blob_columns` list plus the prop-key names.
    by_key: DashMap<(PropertyKey, EntityKind), BlobColSpec>,
    /// `BlobColSpec -> Arc<Mutex<BlobColumnWriter>>`. A Mutex (rather
    /// than an RwLock) because `write_slot` may grow either mmap via
    /// `ensure_room`, which invalidates outstanding borrows into the
    /// old mapping.
    writers: DashMap<BlobColSpec, Arc<Mutex<BlobColumnWriter>>>,
}

impl BlobColumnRegistry {
    pub(crate) fn new() -> Self {
        Self {
            by_key: DashMap::new(),
            writers: DashMap::new(),
        }
    }

    /// Rehydrate the registry from persisted specs. Takes the
    /// `blob_columns` list from [`crate::dict::DictSnapshot`] plus the
    /// property-key name table (indexed by `prop_key_id`). For each
    /// spec, opens a writer (the backing files are expected to already
    /// exist from a prior session; missing files are tolerated and
    /// start empty) and records the key → spec mapping.
    ///
    /// Mismatching on-disk headers are also tolerated — the writer
    /// inherits `n_slots=0` + `arena_bytes=0` and overwrites on next
    /// write, which is the same "degrade to absent then re-emit" policy
    /// as the vec-column registry.
    pub(crate) fn hydrate_from_dict(
        &self,
        sub: &SubstrateFile,
        prop_key_names: &[String],
        specs: &[BlobColSpec],
    ) -> SubstrateResult<()> {
        for spec in specs {
            let Some(name) = prop_key_names.get(spec.prop_key_id as usize) else {
                // Persisted spec references a prop_key_id past the
                // known table. Dict corruption or forward-compat drift
                // — skip with a log, same policy as vec_column_registry.
                tracing::warn!(
                    target: "substrate::blob_columns",
                    "hydrate: blob_column spec references unknown prop_key_id {} \
                     (only {} names known); skipping",
                    spec.prop_key_id,
                    prop_key_names.len()
                );
                continue;
            };
            let key = PropertyKey::new(name.as_str());
            let writer = BlobColumnWriter::create(sub, *spec)?;
            self.writers
                .insert(*spec, Arc::new(Mutex::new(writer)));
            self.by_key.insert((key, spec.entity_kind), *spec);
        }
        Ok(())
    }

    /// Look up the spec for a (key, entity_kind) pair, if any. This is
    /// the cheap fast-path used by the read-side router — on a miss
    /// the caller falls through to the in-memory property map.
    pub(crate) fn spec_for(&self, key: &PropertyKey, ek: EntityKind) -> Option<BlobColSpec> {
        self.by_key.get(&(key.clone(), ek)).map(|r| *r)
    }

    /// Route an oversized `Value::String` / `Value::Bytes` write for a
    /// node/edge slot. Creates the blob column on first use;
    /// `prop_key_id` must already be interned in the store's
    /// `PropKeyRegistry`. Empty payloads are rejected by the underlying
    /// writer (absence is encoded by `len == 0`, so a "present but
    /// empty" payload cannot be round-tripped).
    pub(crate) fn write(
        &self,
        sub: &SubstrateFile,
        key: &PropertyKey,
        ek: EntityKind,
        prop_key_id: u16,
        slot: u32,
        bytes: &[u8],
    ) -> SubstrateResult<()> {
        if bytes.is_empty() {
            return Err(SubstrateError::WalBadFrame(
                "blob_column: cannot store empty payload".into(),
            ));
        }
        let spec = BlobColSpec {
            prop_key_id,
            entity_kind: ek,
        };

        // Register the spec on first sighting. Unlike vec_columns
        // there is no dim/dtype to enforce — only the (key, ek) pair
        // identifies the column.
        self.by_key.entry((key.clone(), ek)).or_insert(spec);

        // Get-or-create the writer.
        let writer = if let Some(w) = self.writers.get(&spec) {
            w.clone()
        } else {
            let w = Arc::new(Mutex::new(BlobColumnWriter::create(sub, spec)?));
            self.writers
                .entry(spec)
                .or_insert_with(|| w.clone())
                .clone()
        };

        writer.lock().write_slot(slot, bytes)
    }

    /// Copy the payload for `slot` out of the column registered for
    /// `(key, ek)`. Returns `None` if the key is not a blob column, if
    /// the slot is past the writer's high-water mark, or if the slot
    /// is marked absent (`len == 0`).
    ///
    /// The returned `Arc<[u8]>` is an owned copy — the writer's mmap
    /// borrow cannot escape the Mutex guard.
    pub(crate) fn read(
        &self,
        key: &PropertyKey,
        ek: EntityKind,
        slot: u32,
    ) -> Option<Arc<[u8]>> {
        let spec = self.spec_for(key, ek)?;
        let writer = self.writers.get(&spec)?.clone();
        let guard = writer.lock();
        let data: &[u8] = guard.read_slot(slot)?;
        Some(Arc::from(data))
    }

    /// Iterate every currently-registered spec. Used by
    /// `SubstrateStore::build_dict_snapshot` to populate the v4
    /// `blob_columns` list.
    pub(crate) fn specs_snapshot(&self) -> Vec<BlobColSpec> {
        let mut out: Vec<BlobColSpec> =
            self.writers.iter().map(|e| *e.key()).collect();
        // Deterministic order: (entity_kind, prop_key_id) matches the
        // vec-column snapshot convention for easier diffing.
        out.sort_by_key(|s| (s.entity_kind as u8, s.prop_key_id));
        out
    }

    /// Durability barrier: walk every open writer and `sync` it
    /// (both CRCs + header + msync + fsync on idx **and** dat). Called
    /// from `SubstrateStore::flush` so blob columns become durable at
    /// the same cadence as the rest of the substrate.
    pub(crate) fn sync_all(&self) -> SubstrateResult<()> {
        for entry in self.writers.iter() {
            entry.value().lock().sync()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_column::BlobColumnReader;
    use crate::file::SubstrateFile;

    fn pk(s: &str) -> PropertyKey {
        PropertyKey::new(s)
    }

    #[test]
    fn write_then_read_node_roundtrip() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();

        let payload = b"hello world, this is a chat payload blob".to_vec();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 5, &payload)
            .unwrap();

        let got = reg.read(&pk("data"), EntityKind::Node, 5).unwrap();
        assert_eq!(got.as_ref(), &payload[..]);
    }

    #[test]
    fn read_missing_key_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let payload = b"x".repeat(512);
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 0, &payload)
            .unwrap();
        assert!(reg.read(&pk("unknown"), EntityKind::Node, 0).is_none());
    }

    #[test]
    fn read_missing_slot_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let payload = b"some bytes".to_vec();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 0, &payload)
            .unwrap();
        assert!(reg.read(&pk("data"), EntityKind::Node, 99).is_none());
    }

    #[test]
    fn node_and_edge_keys_are_independent() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();

        let node_payload = b"node-side payload".to_vec();
        let edge_payload = b"edge-side much longer payload with different length".to_vec();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 0, &node_payload)
            .unwrap();
        reg.write(&sub, &pk("data"), EntityKind::Edge, 0, 0, &edge_payload)
            .unwrap();

        let got_node = reg
            .read(&pk("data"), EntityKind::Node, 0)
            .unwrap();
        let got_edge = reg
            .read(&pk("data"), EntityKind::Edge, 0)
            .unwrap();
        assert_eq!(got_node.as_ref(), &node_payload[..]);
        assert_eq!(got_edge.as_ref(), &edge_payload[..]);
    }

    #[test]
    fn empty_payload_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let err = reg
            .write(&sub, &pk("k"), EntityKind::Node, 0, 0, &[])
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("empty"), "unexpected err: {msg}");
    }

    #[test]
    fn variable_length_writes_share_one_column() {
        // Two writes for the same (key, ek) with different payload
        // sizes land in the same column — unlike vec_columns, there is
        // no dim check to reject.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let short = b"tiny".to_vec();
        let long = b"x".repeat(4096);
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 0, &short)
            .unwrap();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 1, &long)
            .unwrap();

        assert_eq!(reg.read(&pk("data"), EntityKind::Node, 0).unwrap().as_ref(), &short[..]);
        assert_eq!(reg.read(&pk("data"), EntityKind::Node, 1).unwrap().as_ref(), &long[..]);
        // Only one spec registered for the key.
        assert_eq!(reg.specs_snapshot().len(), 1);
    }

    #[test]
    fn overwrite_returns_new_value() {
        // Blob columns are append-only: overwriting a slot rewrites
        // the slot entry and appends new bytes; the read must reflect
        // the latest write.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 4, b"first-value")
            .unwrap();
        reg.write(
            &sub,
            &pk("data"),
            EntityKind::Node,
            0,
            4,
            b"second-and-rather-longer-value",
        )
        .unwrap();
        let got = reg.read(&pk("data"), EntityKind::Node, 4).unwrap();
        assert_eq!(got.as_ref(), b"second-and-rather-longer-value");
    }

    #[test]
    fn specs_snapshot_is_deterministic() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let payload = b"some bytes to store".to_vec();
        // Insert in a scrambled order: edge key 2, node key 0, node key 1.
        reg.write(&sub, &pk("b"), EntityKind::Edge, 2, 0, &payload).unwrap();
        reg.write(&sub, &pk("a"), EntityKind::Node, 0, 0, &payload).unwrap();
        reg.write(&sub, &pk("c"), EntityKind::Node, 1, 0, &payload).unwrap();

        let specs = reg.specs_snapshot();
        assert_eq!(specs.len(), 3);
        // Sorted by (entity_kind, prop_key_id): node first.
        assert_eq!(specs[0].entity_kind, EntityKind::Node);
        assert_eq!(specs[0].prop_key_id, 0);
        assert_eq!(specs[1].entity_kind, EntityKind::Node);
        assert_eq!(specs[1].prop_key_id, 1);
        assert_eq!(specs[2].entity_kind, EntityKind::Edge);
        assert_eq!(specs[2].prop_key_id, 2);
    }

    #[test]
    fn sync_all_makes_data_durable_to_a_fresh_reader() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let payload = b"persist me across reopen".to_vec();
        reg.write(&sub, &pk("data"), EntityKind::Node, 0, 3, &payload)
            .unwrap();
        reg.sync_all().unwrap();

        let spec = reg
            .spec_for(&pk("data"), EntityKind::Node)
            .expect("spec should be registered");
        let r = BlobColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 4); // slots 0..=3 allocated
        assert_eq!(r.read_slot(3).unwrap(), &payload[..]);
    }

    #[test]
    fn hydrate_from_dict_recovers_registered_key() {
        // Write + sync a column, then simulate a reopen via a fresh
        // registry that re-hydrates from (spec, prop-key-name) as the
        // dict would carry.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg1 = BlobColumnRegistry::new();
        let payload = b"content that must survive a reopen cycle".to_vec();
        reg1.write(&sub, &pk("data"), EntityKind::Node, 0, 7, &payload)
            .unwrap();
        reg1.sync_all().unwrap();
        let specs = reg1.specs_snapshot();
        drop(reg1);

        // Fresh registry. Hydrate from what the dict would carry.
        let reg2 = BlobColumnRegistry::new();
        reg2.hydrate_from_dict(&sub, &["data".to_string()], &specs)
            .unwrap();

        let got = reg2.read(&pk("data"), EntityKind::Node, 7).unwrap();
        assert_eq!(got.as_ref(), &payload[..]);
    }

    #[test]
    fn hydrate_tolerates_unknown_prop_key_id() {
        // A persisted spec references a prop_key_id past the name
        // table — should log + skip, not error.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = BlobColumnRegistry::new();
        let bogus_spec = BlobColSpec {
            prop_key_id: 42, // past the names table below
            entity_kind: EntityKind::Node,
        };
        // Only one name registered, id 0. Spec's id 42 must be skipped.
        reg.hydrate_from_dict(&sub, &["only_one".to_string()], &[bogus_spec])
            .unwrap();
        assert!(reg.spec_for(&pk("only_one"), EntityKind::Node).is_none());
        assert_eq!(reg.specs_snapshot().len(), 0);
    }

    #[test]
    fn threshold_constant_matches_spec() {
        // Guard against accidentally bumping the threshold without an
        // RFC / plan-level discussion — the value is part of the
        // public routing contract.
        assert_eq!(BLOB_COLUMN_THRESHOLD_BYTES, 256);
    }
}
