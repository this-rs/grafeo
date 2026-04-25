//! EngramStore — in-memory cache with write-through to CognitiveStorage.
//!
//! Provides concurrent access to engrams via `DashMap` and atomic ID generation.
//!
//! ## Substrate backend (feature = "substrate")
//!
//! When the `substrate` feature is enabled and a [`SubstrateWriter`] is
//! attached via [`EngramStore::attach_substrate`], mutations are mirrored
//! to the substrate side-column layout introduced in T7 Step 1:
//!
//! * `substrate.engram_members` — directory-indexed snapshot of each engram's
//!   node ensemble (engram_id u16 → Vec<node_id u32>).
//! * `substrate.engram_bitset` — per-node 64-bit folded Bloom signature
//!   (bit `i` set ⇔ some engram with `id & 0x3F == i` claims the node).
//!
//! The bitset is monotone-union: [`EngramStore::remove`] clears the members
//! directory entry but leaves the bitset bits alone — a stale bit is a cheap
//! false positive resolved by a members-table lookup (source of truth).
//!
//! Three new semantic primitives are exposed on top of the substrate path:
//!
//! * [`EngramStore::form`] — allocate a new id, persist members, set each
//!   node's bit.
//! * [`EngramStore::recall`] — two-tier candidate retrieval (bitset pre-filter
//!   over the in-memory engram set, then members-table verification).
//! * [`EngramStore::members`] — read the members snapshot for an engram id.
//!
//! The existing public API (`insert`, `update`, `remove`, `load_from_graph`,
//! `find_by_node`, `list`, `get`, `next_id`, `count`) is preserved byte-for-byte
//! and continues to work without substrate: the substrate writer is strictly
//! optional and additive.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use obrain_common::types::{NodeId, Value};

use super::traits::CognitiveStorage;
use super::types::{Engram, EngramId};

#[cfg(feature = "substrate")]
use obrain_substrate::{
    SubstrateError, SubstrateResult, Writer as SubstrateWriter, engram_bit_mask,
};
#[cfg(feature = "substrate")]
use parking_lot::RwLock as PlRwLock;

/// Label used for engram nodes in the backing graph store.
const ENGRAM_LABEL: &str = ":Engram";

/// Property key carrying the full serialized engram (JSON).
const PROP_ENGRAM_DATA: &str = "_engram_data";

/// Property key carrying the engram ID (convenience for filtering).
const PROP_ENGRAM_ID: &str = "_engram_id";

/// Substrate engram ids are `u16` — we refuse to mirror engrams with ids
/// beyond this range. In practice the cognitive layer allocates ids
/// monotonically from 1 via [`EngramStore::next_id`]; the 65 535-slot
/// directory is sized to exhaust the namespace well before the ABI limit.
#[cfg(feature = "substrate")]
const MAX_SUBSTRATE_ENGRAM_ID: u64 = u16::MAX as u64;

/// In-memory engram store with optional write-through to [`CognitiveStorage`]
/// and mirrored substrate side-columns.
pub struct EngramStore {
    /// Concurrent cache of all engrams keyed by their ID.
    cache: DashMap<EngramId, Engram>,
    /// Optional backing storage for persistence (write-through).
    storage: Option<Arc<dyn CognitiveStorage>>,
    /// Mapping from EngramId to the backing graph node, when persisted.
    /// Used for `update` / `remove` write-through.
    node_ids: DashMap<EngramId, NodeId>,
    /// Monotonically increasing counter for ID generation.
    next_id: AtomicU64,
    /// Optional substrate writer for side-table + bitset mirror (T7 Step 2).
    #[cfg(feature = "substrate")]
    substrate: PlRwLock<Option<Arc<SubstrateWriter>>>,
}

impl std::fmt::Debug for EngramStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("EngramStore");
        dbg.field("cache_len", &self.cache.len())
            .field("has_storage", &self.storage.is_some())
            .field("next_id", &self.next_id.load(Ordering::Relaxed));
        #[cfg(feature = "substrate")]
        dbg.field("has_substrate", &self.substrate.read().is_some());
        dbg.finish()
    }
}

impl EngramStore {
    /// Creates a new `EngramStore`.
    ///
    /// When `storage` is provided, every `insert` / `update` / `remove` is
    /// written through to the backing [`CognitiveStorage`] so engrams survive
    /// process restarts. Use [`EngramStore::load_from_graph`] on startup to
    /// rehydrate the cache from persisted `:Engram` nodes.
    pub fn new(storage: Option<Arc<dyn CognitiveStorage>>) -> Self {
        Self {
            cache: DashMap::new(),
            storage,
            node_ids: DashMap::new(),
            next_id: AtomicU64::new(1),
            #[cfg(feature = "substrate")]
            substrate: PlRwLock::new(None),
        }
    }

    /// Attach a substrate writer. After this call every subsequent
    /// `insert` / `remove` / `load_from_graph` mutation is mirrored to the
    /// substrate engram side-column zones (members + bitset), and the
    /// substrate-native primitives ([`Self::form`], [`Self::recall`],
    /// [`Self::members`]) become usable.
    ///
    /// Pass `None`-wrapped to detach; re-attaching overwrites the previous
    /// writer.
    #[cfg(feature = "substrate")]
    pub fn attach_substrate(&self, writer: Arc<SubstrateWriter>) {
        *self.substrate.write() = Some(writer);
    }

    /// Returns `true` when a substrate writer is attached.
    #[cfg(feature = "substrate")]
    pub fn has_substrate(&self) -> bool {
        self.substrate.read().is_some()
    }

    /// Returns the next unique [`EngramId`] and advances the counter.
    pub fn next_id(&self) -> EngramId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        EngramId(id)
    }

    /// Retrieves a clone of the engram with the given `id`, if present.
    pub fn get(&self, id: EngramId) -> Option<Engram> {
        self.cache.get(&id).map(|entry| entry.value().clone())
    }

    /// Serializes an engram into graph node properties.
    fn engram_properties(engram: &Engram) -> HashMap<String, Value> {
        let mut props = HashMap::new();
        props.insert(PROP_ENGRAM_ID.to_string(), Value::Int64(engram.id.0 as i64));
        // Full serialization as JSON — simplest path that preserves every
        // field (ensemble, spectral_signature, FSRS state, history, ...).
        match serde_json::to_string(engram) {
            Ok(json) => {
                props.insert(PROP_ENGRAM_DATA.to_string(), Value::String(json.into()));
            }
            Err(e) => {
                tracing::warn!(
                    engram_id = engram.id.0,
                    error = %e,
                    "Failed to serialize engram, skipping persistence"
                );
            }
        }
        props
    }

    /// Inserts an engram into the cache and writes through to storage.
    pub fn insert(&self, engram: Engram) {
        let id = engram.id;
        if let Some(storage) = &self.storage {
            // If we already have a node for this id, update it in place.
            let props = Self::engram_properties(&engram);
            if !props.contains_key(PROP_ENGRAM_DATA) {
                // Serialization failed, still cache but skip persistence.
                self.cache.insert(id, engram);
                return;
            }
            if let Some(nid) = self.node_ids.get(&id).map(|e| *e.value()) {
                storage.update_node(nid, &props);
            } else {
                let nid = storage.create_node(ENGRAM_LABEL, &props);
                self.node_ids.insert(id, nid);
            }
        }
        #[cfg(feature = "substrate")]
        self.mirror_substrate_insert(&engram);
        self.cache.insert(id, engram);
    }

    /// Updates an engram in-place by applying `f` to its mutable reference.
    ///
    /// This is a no-op if no engram with the given `id` exists.
    pub fn update(&self, id: EngramId, f: impl FnOnce(&mut Engram)) {
        if let Some(mut entry) = self.cache.get_mut(&id) {
            f(entry.value_mut());
            // Write-through: persist the updated state.
            if let Some(storage) = &self.storage {
                let props = Self::engram_properties(entry.value());
                if let Some(nid) = self.node_ids.get(&id).map(|e| *e.value()) {
                    storage.update_node(nid, &props);
                } else {
                    // No node yet — create one now.
                    let nid = storage.create_node(ENGRAM_LABEL, &props);
                    self.node_ids.insert(id, nid);
                }
            }
            // Substrate mirror: refresh members snapshot (ensemble may have
            // drifted). Bitset stays monotone — we only ever add bits here,
            // matching the update's non-destructive semantics.
            #[cfg(feature = "substrate")]
            {
                let engram_snapshot = entry.value().clone();
                drop(entry); // release the cache entry before hitting substrate
                self.mirror_substrate_insert(&engram_snapshot);
            }
        }
    }

    /// Removes and returns the engram with the given `id`, if present.
    pub fn remove(&self, id: EngramId) -> Option<Engram> {
        let removed = self.cache.remove(&id).map(|(_, engram)| engram);
        if removed.is_some()
            && let Some(storage) = &self.storage
            && let Some((_, nid)) = self.node_ids.remove(&id)
        {
            storage.delete_node(nid);
        }
        // Substrate mirror: clear the members directory entry. Bitset bits
        // are intentionally left in place (monotone-union semantics) —
        // stale bits are a cheap false positive resolved by the members
        // lookup in `recall`.
        #[cfg(feature = "substrate")]
        if removed.is_some() {
            self.mirror_substrate_remove(id);
        }
        removed
    }

    /// Rehydrates the cache from persisted `:Engram` nodes in the backing
    /// graph store. Returns the number of engrams loaded.
    ///
    /// Also advances the internal ID counter past the highest persisted ID
    /// so newly generated IDs do not collide with existing ones.
    pub fn load_from_graph(&self) -> usize {
        let Some(storage) = &self.storage else {
            return 0;
        };
        let nodes = storage.query_nodes(ENGRAM_LABEL, None);
        let mut loaded = 0usize;
        let mut max_id: u64 = 0;
        for node in nodes {
            let Some(Value::String(json)) = node.properties.get(PROP_ENGRAM_DATA) else {
                continue;
            };
            match serde_json::from_str::<Engram>(json.as_ref()) {
                Ok(engram) => {
                    let eid = engram.id;
                    max_id = max_id.max(eid.0);
                    self.node_ids.insert(eid, node.id);
                    #[cfg(feature = "substrate")]
                    self.mirror_substrate_insert(&engram);
                    self.cache.insert(eid, engram);
                    loaded += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        node_id = node.id.0,
                        error = %e,
                        "Failed to deserialize persisted engram"
                    );
                }
            }
        }
        // Advance the ID counter so next_id() starts above the max loaded.
        if max_id > 0 {
            // fetch_max is stable — bumps the counter if ours is smaller.
            let next = max_id.saturating_add(1);
            let current = self.next_id.load(Ordering::Relaxed);
            if next > current {
                self.next_id.store(next, Ordering::Relaxed);
            }
        }
        loaded
    }

    /// Returns a snapshot of all engrams currently in the cache.
    pub fn list(&self) -> Vec<Engram> {
        self.cache
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Finds all engram IDs whose ensemble contains the given `node_id`.
    pub fn find_by_node(&self, node_id: NodeId) -> Vec<EngramId> {
        self.cache
            .iter()
            .filter(|entry| {
                entry
                    .value()
                    .ensemble
                    .iter()
                    .any(|(nid, _)| *nid == node_id)
            })
            .map(|entry| *entry.key())
            .collect()
    }

    /// Returns the number of engrams in the cache.
    pub fn count(&self) -> usize {
        self.cache.len()
    }

    // =================================================================
    // Substrate-native semantic primitives (T7 Step 2)
    // =================================================================

    /// Allocate a fresh engram id, materialise a minimal [`Engram`] in the
    /// cache, and mirror the ensemble to the substrate side-columns.
    ///
    /// For each node in `nids` the folded-Bloom bit for the new id is OR-ed
    /// into the node's signature column, and the members snapshot is written
    /// once as a contiguous blob. The cache gets a default-strength engram
    /// so [`Self::recall`] / [`Self::find_by_node`] / [`Self::list`] all see
    /// the new engram without any external `insert` call.
    ///
    /// Returns an error if no substrate writer is attached, or if the
    /// allocated id would overflow the substrate `u16` engram namespace.
    /// Overflow is pre-checked before the id counter is bumped so a failed
    /// `form` does not waste a slot.
    #[cfg(feature = "substrate")]
    pub fn form(&self, nids: &[NodeId]) -> SubstrateResult<EngramId> {
        if self.substrate_writer().is_none() {
            return Err(SubstrateError::WalBadFrame(
                "no substrate writer attached".into(),
            ));
        }
        // Peek the next id without bumping the counter so we can fail before
        // mutating state.
        let peek = self.next_id.load(Ordering::Relaxed);
        if peek == 0 || peek > MAX_SUBSTRATE_ENGRAM_ID {
            return Err(SubstrateError::WalBadFrame(format!(
                "engram id {peek} overflows substrate u16 namespace (max {})",
                MAX_SUBSTRATE_ENGRAM_ID
            )));
        }
        let eid = self.next_id();
        let ensemble: Vec<(NodeId, f64)> = nids.iter().map(|&n| (n, 1.0)).collect();
        let engram = Engram::new(eid, ensemble);
        // `insert` handles: cache population, optional JSON write-through,
        // and substrate mirror (members + bitset) via mirror_substrate_insert.
        // Mirror failures are logged rather than propagated to preserve
        // insert's contract; we pre-validated the namespace above so the
        // happy path never hits a substrate error.
        self.insert(engram);
        Ok(eid)
    }

    /// Candidate-recall for `query_nid` via the two-tier substrate path.
    ///
    /// 1. Read the 64-bit bitset column for the node.
    /// 2. Enumerate cached engrams whose id mod 64 matches a set bit
    ///    (cheap in-memory filter).
    /// 3. Verify actual membership against the `substrate.engram_members`
    ///    side-table — stale bits from removed engrams or from Bloom
    ///    collisions are filtered out here.
    ///
    /// Returns the verified engram ids in ascending order. An empty
    /// bitset short-circuits to an empty result.
    #[cfg(feature = "substrate")]
    pub fn recall(&self, query_nid: NodeId) -> SubstrateResult<Vec<EngramId>> {
        let writer = self
            .substrate_writer()
            .ok_or_else(|| SubstrateError::WalBadFrame("no substrate writer attached".into()))?;
        let nid_u32 = Self::node_id_to_u32(query_nid).ok_or_else(|| {
            SubstrateError::WalBadFrame("node id overflows u32 substrate addressing".into())
        })?;
        let bitset = writer.engram_bitset(nid_u32)?;
        if bitset == 0 {
            return Ok(Vec::new());
        }

        // Two-tier filter: collect cached engrams whose bit is set, then
        // verify via the substrate members snapshot.
        let mut candidates: Vec<EngramId> = Vec::new();
        for entry in self.cache.iter() {
            let eid = *entry.key();
            let Some(eid_u16) = Self::engram_id_to_u16(eid) else {
                continue;
            };
            if bitset & engram_bit_mask(eid_u16) == 0 {
                continue;
            }
            // Bitset says "maybe" — verify against the members table.
            match writer.engram_members(eid_u16)? {
                Some(members) if members.contains(&nid_u32) => candidates.push(eid),
                _ => {}
            }
        }
        candidates.sort_by_key(|eid| eid.0);
        Ok(candidates)
    }

    /// Read the members snapshot for `eid` directly from the substrate
    /// side-table. Returns an empty vector if the engram is unknown to
    /// substrate (never formed, or was cleared via [`Self::remove`]).
    #[cfg(feature = "substrate")]
    pub fn members(&self, eid: EngramId) -> SubstrateResult<Vec<NodeId>> {
        let writer = self
            .substrate_writer()
            .ok_or_else(|| SubstrateError::WalBadFrame("no substrate writer attached".into()))?;
        let Some(eid_u16) = Self::engram_id_to_u16(eid) else {
            return Ok(Vec::new());
        };
        let raw = writer.engram_members(eid_u16)?.unwrap_or_default();
        Ok(raw.into_iter().map(|n| NodeId(n as u64)).collect())
    }

    // --- substrate internals -----------------------------------------

    #[cfg(feature = "substrate")]
    fn substrate_writer(&self) -> Option<Arc<SubstrateWriter>> {
        self.substrate.read().as_ref().map(Arc::clone)
    }

    /// Persist the (id, members) pair + per-node bit for an engram whose id
    /// was allocated externally (used by `form` and by the insert mirror).
    #[cfg(feature = "substrate")]
    fn form_with_id(&self, eid: EngramId, nids: &[NodeId]) -> SubstrateResult<()> {
        let writer = self
            .substrate_writer()
            .ok_or_else(|| SubstrateError::WalBadFrame("no substrate writer attached".into()))?;
        let eid_u16 = Self::engram_id_to_u16(eid).ok_or_else(|| {
            SubstrateError::WalBadFrame(format!(
                "engram id {} overflows substrate u16 namespace (max {})",
                eid.0, MAX_SUBSTRATE_ENGRAM_ID
            ))
        })?;
        let mut member_u32s: Vec<u32> = Vec::with_capacity(nids.len());
        for nid in nids {
            if let Some(n) = Self::node_id_to_u32(*nid) {
                member_u32s.push(n);
            } else {
                tracing::warn!(
                    node_id = nid.0,
                    "node id overflows substrate u32 namespace; skipping engram mirror"
                );
            }
        }
        writer.set_engram_members(eid_u16, member_u32s.clone())?;
        for nid in &member_u32s {
            writer.add_engram_bit(*nid, eid_u16)?;
        }
        Ok(())
    }

    /// Best-effort insert/update mirror: refreshes members snapshot and
    /// OR-s membership bits. Failures are logged but not propagated — the
    /// cognitive layer treats substrate as a derived view, not the source
    /// of truth.
    #[cfg(feature = "substrate")]
    fn mirror_substrate_insert(&self, engram: &Engram) {
        if self.substrate.read().is_none() {
            return;
        }
        let nids: Vec<NodeId> = engram.ensemble.iter().map(|(n, _)| *n).collect();
        if let Err(e) = self.form_with_id(engram.id, &nids) {
            tracing::warn!(
                engram_id = engram.id.0,
                error = %e,
                "substrate engram mirror failed (insert/update)"
            );
        }
    }

    /// Best-effort remove mirror: clears the members directory entry.
    /// Bitset bits are deliberately left in place; stale bits are cheap
    /// false positives resolved by `recall` via the members lookup.
    #[cfg(feature = "substrate")]
    fn mirror_substrate_remove(&self, id: EngramId) {
        let Some(writer) = self.substrate_writer() else {
            return;
        };
        let Some(eid_u16) = Self::engram_id_to_u16(id) else {
            return;
        };
        if let Err(e) = writer.set_engram_members(eid_u16, Vec::new()) {
            tracing::warn!(
                engram_id = id.0,
                error = %e,
                "substrate engram mirror failed (remove)"
            );
        }
    }

    #[cfg(feature = "substrate")]
    fn engram_id_to_u16(id: EngramId) -> Option<u16> {
        if id.0 == 0 || id.0 > MAX_SUBSTRATE_ENGRAM_ID {
            None
        } else {
            Some(id.0 as u16)
        }
    }

    #[cfg(feature = "substrate")]
    fn node_id_to_u32(nid: NodeId) -> Option<u32> {
        if nid.0 > u32::MAX as u64 {
            None
        } else {
            Some(nid.0 as u32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove_roundtrip() {
        let store = EngramStore::new(None);
        let id = store.next_id();
        let engram = Engram::new(id, vec![(NodeId(1), 1.0), (NodeId(2), 0.8)]);

        store.insert(engram.clone());
        assert_eq!(store.count(), 1);

        let fetched = store.get(id).expect("engram should exist");
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.ensemble.len(), 2);

        let removed = store.remove(id).expect("engram should be removable");
        assert_eq!(removed.id, id);
        assert_eq!(store.count(), 0);
        assert!(store.get(id).is_none());
    }

    #[test]
    fn update_modifies_in_place() {
        let store = EngramStore::new(None);
        let id = store.next_id();
        let engram = Engram::new(id, vec![(NodeId(1), 1.0)]);
        store.insert(engram);

        store.update(id, |e| {
            e.strength = 0.99;
        });

        let updated = store.get(id).unwrap();
        assert!((updated.strength - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn find_by_node_returns_matching_engrams() {
        let store = EngramStore::new(None);

        let id1 = store.next_id();
        store.insert(Engram::new(id1, vec![(NodeId(10), 1.0), (NodeId(20), 0.5)]));

        let id2 = store.next_id();
        store.insert(Engram::new(id2, vec![(NodeId(20), 1.0), (NodeId(30), 0.5)]));

        let id3 = store.next_id();
        store.insert(Engram::new(id3, vec![(NodeId(30), 1.0)]));

        let matches = store.find_by_node(NodeId(20));
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&id1));
        assert!(matches.contains(&id2));
    }

    #[test]
    fn list_returns_all_engrams() {
        let store = EngramStore::new(None);
        for _ in 0..5 {
            let id = store.next_id();
            store.insert(Engram::new(id, vec![(NodeId(1), 1.0)]));
        }
        assert_eq!(store.list().len(), 5);
    }

    #[test]
    fn next_id_is_monotonic() {
        let store = EngramStore::new(None);
        let a = store.next_id();
        let b = store.next_id();
        let c = store.next_id();
        assert!(a.0 < b.0);
        assert!(b.0 < c.0);
    }
}
