//! EngramStore — in-memory cache with write-through to CognitiveStorage.
//!
//! Provides concurrent access to engrams via `DashMap` and atomic ID generation.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use obrain_common::types::NodeId;

use super::traits::CognitiveStorage;
use super::types::{Engram, EngramId};

/// In-memory engram store with optional write-through to [`CognitiveStorage`].
pub struct EngramStore {
    /// Concurrent cache of all engrams keyed by their ID.
    cache: DashMap<EngramId, Engram>,
    /// Optional backing storage for persistence (unused for now — cache only).
    #[allow(dead_code)]
    storage: Option<Arc<dyn CognitiveStorage>>,
    /// Monotonically increasing counter for ID generation.
    next_id: AtomicU64,
}

impl std::fmt::Debug for EngramStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramStore")
            .field("cache_len", &self.cache.len())
            .field("has_storage", &self.storage.is_some())
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl EngramStore {
    /// Creates a new `EngramStore`.
    ///
    /// If `storage` is provided it will be kept for future write-through
    /// support, but all operations currently hit the in-memory cache only.
    pub fn new(storage: Option<Arc<dyn CognitiveStorage>>) -> Self {
        Self {
            cache: DashMap::new(),
            storage,
            next_id: AtomicU64::new(1),
        }
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

    /// Inserts an engram into the cache (and, in the future, into storage).
    pub fn insert(&self, engram: Engram) {
        self.cache.insert(engram.id, engram);
    }

    /// Updates an engram in-place by applying `f` to its mutable reference.
    ///
    /// This is a no-op if no engram with the given `id` exists.
    pub fn update(&self, id: EngramId, f: impl FnOnce(&mut Engram)) {
        if let Some(mut entry) = self.cache.get_mut(&id) {
            f(entry.value_mut());
        }
    }

    /// Removes and returns the engram with the given `id`, if present.
    pub fn remove(&self, id: EngramId) -> Option<Engram> {
        self.cache.remove(&id).map(|(_, engram)| engram)
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
