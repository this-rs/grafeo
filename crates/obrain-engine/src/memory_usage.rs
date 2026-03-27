//! Hierarchical memory usage breakdown for the database.
//!
//! Store-level types (`StoreMemory`, `IndexMemory`, etc.) live in grafeo-common.
//! This module defines the top-level `MemoryUsage` aggregate and engine-specific
//! types (`CacheMemory`, `BufferManagerMemory`).

pub use grafeo_common::memory::usage::{
    IndexMemory, MvccMemory, NamedMemory, StoreMemory, StringPoolMemory,
};
use serde::{Deserialize, Serialize};

/// Hierarchical memory usage breakdown for the entire database.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total estimated memory usage in bytes.
    pub total_bytes: usize,
    /// Graph storage (nodes, edges, properties).
    pub store: StoreMemory,
    /// Index structures.
    pub indexes: IndexMemory,
    /// MVCC versioning overhead.
    pub mvcc: MvccMemory,
    /// Caches (query plans, etc.).
    pub caches: CacheMemory,
    /// String interning (ArcStr label/type registries).
    pub string_pool: StringPoolMemory,
    /// Buffer manager tracked allocations.
    pub buffer_manager: BufferManagerMemory,
}

impl MemoryUsage {
    /// Recomputes `total_bytes` from child totals.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.store.total_bytes
            + self.indexes.total_bytes
            + self.mvcc.total_bytes
            + self.caches.total_bytes
            + self.string_pool.total_bytes
            + self.buffer_manager.allocated_bytes;
    }
}

/// Cache memory usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMemory {
    /// Total cache memory.
    pub total_bytes: usize,
    /// Parsed plan cache.
    pub parsed_plan_cache_bytes: usize,
    /// Optimized plan cache.
    pub optimized_plan_cache_bytes: usize,
    /// Number of cached plans (parsed + optimized).
    pub cached_plan_count: usize,
}

impl CacheMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.parsed_plan_cache_bytes + self.optimized_plan_cache_bytes;
    }
}

/// Buffer manager tracked allocations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BufferManagerMemory {
    /// Budget configured for the buffer manager.
    pub budget_bytes: usize,
    /// Currently allocated via grants.
    pub allocated_bytes: usize,
    /// Graph storage region.
    pub graph_storage_bytes: usize,
    /// Index buffers region.
    pub index_buffers_bytes: usize,
    /// Execution buffers region.
    pub execution_buffers_bytes: usize,
    /// Spill staging region.
    pub spill_staging_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_memory_usage_is_zero() {
        let usage = MemoryUsage::default();
        assert_eq!(usage.total_bytes, 0);
        assert_eq!(usage.store.total_bytes, 0);
        assert_eq!(usage.indexes.total_bytes, 0);
        assert_eq!(usage.mvcc.total_bytes, 0);
        assert_eq!(usage.caches.total_bytes, 0);
        assert_eq!(usage.string_pool.total_bytes, 0);
        assert_eq!(usage.buffer_manager.allocated_bytes, 0);
    }

    #[test]
    fn compute_total_sums_children() {
        let mut usage = MemoryUsage {
            store: StoreMemory {
                total_bytes: 100,
                ..Default::default()
            },
            indexes: IndexMemory {
                total_bytes: 200,
                ..Default::default()
            },
            mvcc: MvccMemory {
                total_bytes: 50,
                ..Default::default()
            },
            caches: CacheMemory {
                total_bytes: 30,
                ..Default::default()
            },
            string_pool: StringPoolMemory {
                total_bytes: 10,
                ..Default::default()
            },
            buffer_manager: BufferManagerMemory {
                allocated_bytes: 20,
                ..Default::default()
            },
            ..Default::default()
        };
        usage.compute_total();
        assert_eq!(usage.total_bytes, 410);
    }

    #[test]
    fn serde_roundtrip() {
        let mut usage = MemoryUsage::default();
        usage.store.nodes_bytes = 1024;
        usage.indexes.vector_indexes.push(NamedMemory {
            name: "vec_idx".to_string(),
            bytes: 512,
            item_count: 100,
        });
        usage.mvcc.average_chain_depth = 1.5;

        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: MemoryUsage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.store.nodes_bytes, 1024);
        assert_eq!(deserialized.indexes.vector_indexes.len(), 1);
        assert_eq!(deserialized.indexes.vector_indexes[0].name, "vec_idx");
        assert!((deserialized.mvcc.average_chain_depth - 1.5).abs() < f64::EPSILON);
    }
}
