//! Standard collection type aliases for Obrain.
//!
//! Use these instead of direct HashMap/HashSet to allow future optimization
//! and ensure consistent hashing across the codebase.
//!
//! # Type Aliases
//!
//! | Type | Use Case |
//! |------|----------|
//! | [`ObrainMap`] | Single-threaded hash map |
//! | [`ObrainSet`] | Single-threaded hash set |
//! | [`ObrainConcurrentMap`] | Multi-threaded hash map |
//! | [`ObrainConcurrentSet`] | Multi-threaded hash set |
//! | [`ObrainIndexMap`] | Insertion-order preserving map |
//! | [`ObrainIndexSet`] | Insertion-order preserving set |
//!
//! # Example
//!
//! ```rust
//! use obrain_common::collections::{ObrainMap, ObrainSet};
//!
//! let mut map: ObrainMap<String, i32> = ObrainMap::default();
//! map.insert("key".to_string(), 42);
//!
//! let mut set: ObrainSet<i32> = ObrainSet::default();
//! set.insert(1);
//! ```

use crate::utils::hash::FxBuildHasher;

/// Standard HashMap with FxHash (fast, non-cryptographic).
///
/// FxHash is optimized for small keys and provides excellent performance
/// for integer and string keys common in graph databases.
pub type ObrainMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;

/// Standard HashSet with FxHash.
pub type ObrainSet<T> = hashbrown::HashSet<T, FxBuildHasher>;

/// Concurrent HashMap for multi-threaded access.
///
/// Uses fine-grained locking for high concurrent throughput.
/// Prefer this over `Arc<Mutex<HashMap>>` for shared mutable state.
pub type ObrainConcurrentMap<K, V> = dashmap::DashMap<K, V, FxBuildHasher>;

/// Concurrent HashSet for multi-threaded access.
pub type ObrainConcurrentSet<T> = dashmap::DashSet<T, FxBuildHasher>;

/// Ordered map preserving insertion order.
///
/// Useful when iteration order matters (e.g., property serialization).
pub type ObrainIndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;

/// Ordered set preserving insertion order.
pub type ObrainIndexSet<T> = indexmap::IndexSet<T, FxBuildHasher>;

/// Create a new empty [`ObrainMap`].
#[inline]
#[must_use]
pub fn obrain_map<K, V>() -> ObrainMap<K, V> {
    ObrainMap::with_hasher(FxBuildHasher::default())
}

/// Create a new [`ObrainMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn obrain_map_with_capacity<K, V>(capacity: usize) -> ObrainMap<K, V> {
    ObrainMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new empty [`ObrainSet`].
#[inline]
#[must_use]
pub fn obrain_set<T>() -> ObrainSet<T> {
    ObrainSet::with_hasher(FxBuildHasher::default())
}

/// Create a new [`ObrainSet`] with the specified capacity.
#[inline]
#[must_use]
pub fn obrain_set_with_capacity<T>(capacity: usize) -> ObrainSet<T> {
    ObrainSet::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new empty [`ObrainConcurrentMap`].
#[inline]
#[must_use]
pub fn obrain_concurrent_map<K, V>() -> ObrainConcurrentMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    ObrainConcurrentMap::with_hasher(FxBuildHasher::default())
}

/// Create a new [`ObrainConcurrentMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn obrain_concurrent_map_with_capacity<K, V>(capacity: usize) -> ObrainConcurrentMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    ObrainConcurrentMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

/// Create a new empty [`ObrainIndexMap`].
#[inline]
#[must_use]
pub fn obrain_index_map<K, V>() -> ObrainIndexMap<K, V> {
    ObrainIndexMap::with_hasher(FxBuildHasher::default())
}

/// Create a new [`ObrainIndexMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn obrain_index_map_with_capacity<K, V>(capacity: usize) -> ObrainIndexMap<K, V> {
    ObrainIndexMap::with_capacity_and_hasher(capacity, FxBuildHasher::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obrain_map() {
        let mut map = obrain_map::<String, i32>();
        map.insert("key".to_string(), 42);
        assert_eq!(map.get("key"), Some(&42));
    }

    #[test]
    fn test_obrain_set() {
        let mut set = obrain_set::<i32>();
        set.insert(1);
        set.insert(2);
        assert!(set.contains(&1));
        assert!(!set.contains(&3));
    }

    #[test]
    fn test_obrain_concurrent_map() {
        let map = obrain_concurrent_map::<String, i32>();
        map.insert("key".to_string(), 42);
        assert_eq!(*map.get("key").unwrap(), 42);
    }

    #[test]
    fn test_obrain_index_map_preserves_order() {
        let mut map = obrain_index_map::<&str, i32>();
        map.insert("c", 3);
        map.insert("a", 1);
        map.insert("b", 2);

        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec!["c", "a", "b"]);
    }
}
