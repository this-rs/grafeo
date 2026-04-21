//! `VecColumnRegistry` — runtime routing of `Value::Vector` properties
//! to the dense mmap vec-column zones defined in [`crate::vec_column`].
//!
//! # Role in the stack
//!
//! ```text
//! SubstrateStore::set_node_property(id, key, Value::Vector(v))
//!     └──► VecColumnRegistry::write_node(id, key, v)
//!            └──► get-or-create VecColumnWriter for (key, Node, dim)
//!                   └──► write_slot(id as u32, cast_slice(&v))
//! ```
//!
//! A `VecColumnRegistry` is the sole gatekeeper of the
//! `substrate.veccol.*` zone family. No other code in the crate
//! opens those files directly — going through the registry ensures:
//!
//! 1. **Spec uniqueness per (key, entity)** — the first `Value::Vector`
//!    write for a given property key establishes its dim + dtype;
//!    subsequent writes with a mismatching dim are rejected rather
//!    than silently corrupting the zone.
//! 2. **Writer reuse** — we keep one open writer per `VecColSpec`
//!    for the lifetime of the store, so a migration loop of
//!    "`set_node_property` per node" does not pay per-write
//!    `mmap` or `fsync` costs. Durability is batched to
//!    [`SubstrateStore::flush`].
//! 3. **Read consistency** — reads issued in the same session
//!    after a `set_node_property` see the most recent value, even
//!    though the on-disk header hasn't been `sync`'d yet. The
//!    writer serves reads from its own mmap (see
//!    [`VecColumnWriter::read_slot`]).
//!
//! # Crash safety
//!
//! Writes are **not** individually durable. The contract is:
//!
//! - Between two `SubstrateStore::flush` calls, all writes are
//!   visible in-process but the on-disk header still carries the
//!   stale CRC from the last `sync`.
//! - `flush` iterates every open writer and calls `sync` on it:
//!   CRC is recomputed, header is overwritten, the zone is
//!   `msync`+`fsync`'d. Post-`flush`, the zone is durable.
//! - A crash between two flushes loses the writes back to the last
//!   `sync`-point; the next open validates the CRC and, on
//!   mismatch, the reader returns `None`, which the store
//!   translates to "property absent".
//!
//! This is the same durability model as the bincode sidecar it
//! replaces. Tightening crash-safety (per-write WAL) is a T17
//! concern, not T16.7.

use std::sync::Arc;

use dashmap::DashMap;
use obrain_common::types::PropertyKey;
use parking_lot::Mutex;

use crate::error::{SubstrateError, SubstrateResult};
use crate::file::SubstrateFile;
use crate::vec_column::{EntityKind, VecColSpec, VecColumnWriter, VecDType};

/// Routing table + open-writer cache for vector-typed properties.
///
/// Cloneable via `Arc`-internal DashMaps, so the `SubstrateStore`
/// can share the registry across its various method paths without
/// refactoring the store's ownership model.
pub(crate) struct VecColumnRegistry {
    /// `(PropertyKey, EntityKind) -> VecColSpec`. Populated on every
    /// first write of a vector for a given key, and pre-populated on
    /// open from the persisted `DictSnapshot.vec_columns` list plus
    /// the prop-key names (so reads on a freshly-opened store can
    /// find the right spec from the property-key alone).
    by_key: DashMap<(PropertyKey, EntityKind), VecColSpec>,
    /// `VecColSpec -> Arc<Mutex<VecColumnWriter>>`. A Mutex (rather
    /// than an RwLock) because `write_slot` may grow the mmap via
    /// `ensure_room`, which invalidates outstanding read borrows
    /// into the old mapping — the lock-out is intentional.
    writers: DashMap<VecColSpec, Arc<Mutex<VecColumnWriter>>>,
}

impl VecColumnRegistry {
    pub(crate) fn new() -> Self {
        Self {
            by_key: DashMap::new(),
            writers: DashMap::new(),
        }
    }

    /// Rehydrate the registry from persisted specs. Takes the
    /// `vec_columns` list from [`crate::dict::DictSnapshot`] plus the
    /// property-key name table (indexed by `prop_key_id`). For each
    /// spec, opens a writer (creating the zone file if absent, but
    /// the file is expected to already exist from a prior session)
    /// and records the key→spec mapping.
    ///
    /// Missing files are tolerated: the writer will be created empty
    /// and the first read will return `None`. Mismatching on-disk
    /// headers are also tolerated — the writer inherits `n_slots=0`
    /// and overwrites on next write.
    pub(crate) fn hydrate_from_dict(
        &self,
        sub: &SubstrateFile,
        prop_key_names: &[String],
        specs: &[VecColSpec],
    ) -> SubstrateResult<()> {
        for spec in specs {
            let Some(name) = prop_key_names.get(spec.prop_key_id as usize) else {
                // Persisted spec references a prop_key_id past the
                // known table. Dict corruption or forward-compat
                // drift — skip with a log.
                tracing::warn!(
                    target: "substrate::vec_columns",
                    "hydrate: vec_column spec references unknown prop_key_id {} \
                     (only {} names known); skipping",
                    spec.prop_key_id,
                    prop_key_names.len()
                );
                continue;
            };
            let key = PropertyKey::new(name.as_str());
            let writer = VecColumnWriter::create(sub, *spec)?;
            self.writers
                .insert(*spec, Arc::new(Mutex::new(writer)));
            self.by_key.insert((key, spec.entity_kind), *spec);
        }
        Ok(())
    }

    /// Look up the spec for a (key, entity_kind) pair, if any. This
    /// is the cheap fast-path used by the read-side router — on a
    /// miss the caller falls through to the in-memory property map.
    pub(crate) fn spec_for(&self, key: &PropertyKey, ek: EntityKind) -> Option<VecColSpec> {
        self.by_key.get(&(key.clone(), ek)).map(|r| *r)
    }

    /// Route a `Value::Vector` write for a node/edge slot. Creates
    /// the vec column on first use; rejects dim/dtype mismatches for
    /// an already-registered key. `prop_key_id` must already be
    /// interned in the store's `PropertyKeyRegistry`.
    pub(crate) fn write(
        &self,
        sub: &SubstrateFile,
        key: &PropertyKey,
        ek: EntityKind,
        prop_key_id: u16,
        slot: u32,
        vector: &[f32],
    ) -> SubstrateResult<()> {
        if vector.is_empty() {
            return Err(SubstrateError::WalBadFrame(
                "vec_column: cannot store empty vector".into(),
            ));
        }
        if vector.len() > u32::MAX as usize {
            return Err(SubstrateError::WalBadFrame(format!(
                "vec_column: vector length {} exceeds u32::MAX",
                vector.len()
            )));
        }
        let dim = vector.len() as u32;
        let spec = VecColSpec {
            prop_key_id,
            entity_kind: ek,
            dim,
            dtype: VecDType::F32,
        };

        // First: enforce spec-consistency per (key, ek). The first
        // vector determines the shape; later writes with a different
        // dim must be rejected rather than silently written to a
        // different file (which would leave the old column stranded).
        if let Some(existing) = self.by_key.get(&(key.clone(), ek)) {
            let ex = *existing;
            drop(existing);
            if ex != spec {
                return Err(SubstrateError::WalBadFrame(format!(
                    "vec_column: cannot change spec for key {:?} ({:?}): \
                     already registered as {:?}, now got {:?}",
                    key, ek, ex, spec
                )));
            }
        } else {
            self.by_key.insert((key.clone(), ek), spec);
        }

        // Second: get-or-create the writer.
        let writer = if let Some(w) = self.writers.get(&spec) {
            w.clone()
        } else {
            let w = Arc::new(Mutex::new(VecColumnWriter::create(sub, spec)?));
            // `entry().or_insert_with` would race with another thread
            // creating the same spec. Use `entry` to be explicit.
            self.writers
                .entry(spec)
                .or_insert_with(|| w.clone())
                .clone()
        };

        // Third: write the payload. `write_slot` takes `&mut self`
        // so the Mutex lock brackets the mmap mutation.
        let bytes: &[u8] = bytemuck::cast_slice(vector);
        writer.lock().write_slot(slot, bytes)
    }

    /// Copy the vector for `slot` out of the column registered for
    /// `(key, ek)`. Returns `None` if the key is not a vec column,
    /// if the slot is past the writer's high-water mark, or if the
    /// dtype is not F32. Other dtypes are not surfaced to
    /// `Value::Vector` today (it is `Arc<[f32]>`).
    pub(crate) fn read(
        &self,
        key: &PropertyKey,
        ek: EntityKind,
        slot: u32,
    ) -> Option<Arc<[f32]>> {
        let spec = self.spec_for(key, ek)?;
        let writer = self.writers.get(&spec)?.clone();
        let guard = writer.lock();
        let data: &[f32] = guard.read_slot_f32(slot)?;
        // Copy out into an owned Arc<[f32]>. The borrow dies with
        // the Mutex guard, so we cannot hand out the mmap slice.
        Some(Arc::from(data))
    }

    /// Iterate every currently-registered spec. Used by
    /// `SubstrateStore::build_dict_snapshot` to populate the v3
    /// `vec_columns` list.
    pub(crate) fn specs_snapshot(&self) -> Vec<VecColSpec> {
        let mut out: Vec<VecColSpec> =
            self.writers.iter().map(|e| *e.key()).collect();
        // Deterministic order for snapshot roundtrips + easier diffing.
        out.sort_by_key(|s| {
            (
                s.entity_kind as u8,
                s.prop_key_id,
                s.dtype as u8,
                s.dim,
            )
        });
        out
    }

    /// Durability barrier: walk every open writer and `sync` it
    /// (header + msync + fsync). Called from `SubstrateStore::flush`
    /// so the vec columns become durable at the same cadence as the
    /// rest of the substrate.
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
    use crate::file::SubstrateFile;

    fn pk(s: &str) -> PropertyKey {
        PropertyKey::new(s)
    }

    #[test]
    fn write_then_read_node_roundtrip() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();

        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        reg.write(&sub, &pk("embedding"), EntityKind::Node, 0, 5, &v)
            .unwrap();

        let got = reg.read(&pk("embedding"), EntityKind::Node, 5).unwrap();
        assert_eq!(got.as_ref(), &v[..]);
    }

    #[test]
    fn read_missing_key_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let v = vec![0.5_f32; 3];
        reg.write(&sub, &pk("embedding"), EntityKind::Node, 0, 0, &v)
            .unwrap();
        assert!(reg.read(&pk("unknown"), EntityKind::Node, 0).is_none());
    }

    #[test]
    fn read_missing_slot_returns_none() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let v = vec![0.5_f32; 3];
        reg.write(&sub, &pk("embedding"), EntityKind::Node, 0, 0, &v)
            .unwrap();
        assert!(reg.read(&pk("embedding"), EntityKind::Node, 99).is_none());
    }

    #[test]
    fn node_and_edge_keys_are_independent() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();

        let v_node = vec![1.0_f32, 2.0, 3.0];
        let v_edge = vec![10.0_f32, 20.0, 30.0];
        reg.write(&sub, &pk("weight"), EntityKind::Node, 0, 0, &v_node)
            .unwrap();
        reg.write(&sub, &pk("weight"), EntityKind::Edge, 0, 0, &v_edge)
            .unwrap();

        let got_node = reg
            .read(&pk("weight"), EntityKind::Node, 0)
            .unwrap();
        let got_edge = reg
            .read(&pk("weight"), EntityKind::Edge, 0)
            .unwrap();
        assert_eq!(got_node.as_ref(), &v_node[..]);
        assert_eq!(got_edge.as_ref(), &v_edge[..]);
    }

    #[test]
    fn dim_mismatch_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let v4 = vec![0.5_f32; 4];
        let v3 = vec![0.1_f32; 3];
        reg.write(&sub, &pk("k"), EntityKind::Node, 0, 0, &v4)
            .unwrap();
        let err = reg
            .write(&sub, &pk("k"), EntityKind::Node, 0, 1, &v3)
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("cannot change spec"),
            "unexpected err: {msg}"
        );
    }

    #[test]
    fn empty_vector_is_rejected() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let err = reg
            .write(&sub, &pk("k"), EntityKind::Node, 0, 0, &[])
            .unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("empty"), "unexpected err: {msg}");
    }

    #[test]
    fn specs_snapshot_is_deterministic() {
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let v = vec![0.5_f32; 4];
        // Insert in a scrambled order.
        reg.write(&sub, &pk("b"), EntityKind::Edge, 2, 0, &v).unwrap();
        reg.write(&sub, &pk("a"), EntityKind::Node, 0, 0, &v).unwrap();
        reg.write(&sub, &pk("c"), EntityKind::Node, 1, 0, &v).unwrap();

        let specs = reg.specs_snapshot();
        assert_eq!(specs.len(), 3);
        // Sorted by (entity_kind, prop_key_id, dtype, dim):
        // entity_kind Node(0) first, then Edge(1).
        assert_eq!(specs[0].prop_key_id, 0);
        assert_eq!(specs[0].entity_kind, EntityKind::Node);
        assert_eq!(specs[1].prop_key_id, 1);
        assert_eq!(specs[1].entity_kind, EntityKind::Node);
        assert_eq!(specs[2].prop_key_id, 2);
        assert_eq!(specs[2].entity_kind, EntityKind::Edge);
    }

    #[test]
    fn sync_all_makes_data_durable_to_a_fresh_reader() {
        use crate::vec_column::VecColumnReader;
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg = VecColumnRegistry::new();
        let v = vec![7.0_f32; 5];
        reg.write(&sub, &pk("emb"), EntityKind::Node, 0, 3, &v)
            .unwrap();
        reg.sync_all().unwrap();

        let spec = reg
            .spec_for(&pk("emb"), EntityKind::Node)
            .expect("spec should be registered");
        let r = VecColumnReader::open(&sub, spec).unwrap().unwrap();
        assert_eq!(r.n_slots(), 4); // slots 0..=3 allocated, so n_slots = 4
        assert_eq!(r.read_slot_f32(3).unwrap(), &v[..]);
    }

    #[test]
    fn hydrate_from_dict_recovers_registered_key() {
        // Write + sync a column, then simulate a reopen via a fresh
        // registry that re-hydrates from the (spec, prop-key-name)
        // list that would have been persisted in the dict.
        let sub = SubstrateFile::open_tempfile().unwrap();
        let reg1 = VecColumnRegistry::new();
        let v = vec![1.0_f32, 2.0, 3.0];
        reg1.write(&sub, &pk("embedding"), EntityKind::Node, 0, 7, &v)
            .unwrap();
        reg1.sync_all().unwrap();
        let specs = reg1.specs_snapshot();
        drop(reg1);

        // Fresh registry. Hydrate from what the dict would carry.
        let reg2 = VecColumnRegistry::new();
        reg2.hydrate_from_dict(&sub, &["embedding".to_string()], &specs)
            .unwrap();

        let got = reg2.read(&pk("embedding"), EntityKind::Node, 7).unwrap();
        assert_eq!(got.as_ref(), &v[..]);
    }
}
