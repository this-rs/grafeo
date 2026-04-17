//! OverlayStore — mmap base layer + LpgStore delta for mutations.
//!
//! Reads check the delta first, then fallback to the mmap base.
//! Writes go only to the delta (LpgStore).
//!
//! This avoids materializing all properties from the mmap, which is
//! the bottleneck for large databases (e.g. 32GB megalaw with 8M nodes
//! and 6+ vector properties each → 20+ min materialization).
//!
//! The overlay materializes only the structure (nodes + labels + edges +
//! adjacency) into the LpgStore, not the property values. New properties
//! written via `set_node_property` go into the LpgStore delta and are
//! visible immediately. Existing properties are served from mmap.
//!
//! # Merge for save
//!
//! When saving, the caller must iterate ALL nodes/edges and merge
//! properties from both layers. The `iter_merged_node_properties()`
//! method provides this merged view.

use std::sync::Arc;

use obrain_common::types::{NodeId, EdgeId, PropertyKey, Value};
use obrain_core::graph::lpg::LpgStore;
use obrain_core::graph::GraphStore;

use super::mmap_store::MmapStore;

/// Tracks which properties were overwritten in the delta for a node/edge.
/// When a property exists in both mmap and delta, delta wins.
pub struct OverlayStore {
    /// The mmap base layer (read-only, zero-copy).
    mmap: Arc<MmapStore>,
    /// The LpgStore delta (only contains mutations: new/changed properties, new edges).
    delta: Arc<LpgStore>,
}

impl OverlayStore {
    /// Create a new overlay from mmap + delta.
    ///
    /// The delta should already have the structure (nodes + edges) materialized
    /// via `materialize_structure_only()`. Properties in the delta are initially
    /// empty — reads fallback to mmap.
    pub fn new(mmap: Arc<MmapStore>, delta: Arc<LpgStore>) -> Self {
        Self { mmap, delta }
    }

    /// Get the mmap base layer.
    pub fn mmap(&self) -> &MmapStore {
        &self.mmap
    }

    /// Get the delta LpgStore (for mutations).
    pub fn delta(&self) -> &Arc<LpgStore> {
        &self.delta
    }

    /// Get a node property: delta first, then mmap fallback.
    pub fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        // Check delta first (new/overwritten properties)
        if let Some(v) = self.delta.get_node_property(id, key) {
            return Some(v);
        }
        // Fallback to mmap (original properties)
        self.mmap.get_node_property(id, key)
    }

    /// Get an edge property: delta first, then mmap fallback.
    pub fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if let Some(v) = self.delta.get_edge_property(id, key) {
            return Some(v);
        }
        self.mmap.get_edge_property(id, key)
    }

    /// Set a node property (writes to delta only).
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        self.delta.set_node_property(id, key, value);
    }

    /// Set an edge property (writes to delta only).
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.delta.set_edge_property(id, key, value);
    }

    /// Create a new edge (writes to delta only).
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.delta.create_edge(src, dst, edge_type)
    }

    /// Iterate merged node properties for save/export.
    ///
    /// For each property key on a node, returns the delta value if it exists,
    /// otherwise the mmap value. This provides the complete merged view.
    pub fn merged_node_properties(&self, id: NodeId) -> Vec<(String, Value)> {
        let mut props = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();

        // Delta properties first (overrides)
        if let Some(node) = GraphStore::get_node(&*self.delta, id) {
            for (key, value) in node.properties.iter() {
                seen_keys.insert(key.as_str().to_string());
                props.push((key.as_str().to_string(), value.clone()));
            }
        }

        // Mmap properties (only those not overridden)
        if let Some(node) = GraphStore::get_node(&*self.mmap, id) {
            for (key, value) in node.properties.iter() {
                let k = key.as_str().to_string();
                if !seen_keys.contains(&k) {
                    props.push((k, value.clone()));
                }
            }
        }

        props
    }

    /// Merge ALL mmap properties into the delta LpgStore (deferred materialization).
    ///
    /// This must be called before `db.save()` when in overlay mode, otherwise
    /// the save only writes delta properties and mmap data is lost.
    ///
    /// For each node in the mmap, copies properties that are NOT already
    /// present in the delta. Delta properties (new/changed) take precedence.
    ///
    /// Returns `(nodes_merged, properties_copied)`.
    pub fn merge_all_for_save(&self) -> (usize, usize) {
        let node_ids = GraphStore::node_ids(&*self.mmap);
        let total = node_ids.len();
        let mut nodes_merged = 0usize;
        let mut props_copied = 0usize;

        for (i, &nid) in node_ids.iter().enumerate() {
            // Progress logging for large databases
            if total > 100_000 && i > 0 && i % 500_000 == 0 {
                eprintln!(
                    "    merge_all_for_save: {}/{} nodes ({:.1}%)",
                    i, total, i as f64 / total as f64 * 100.0
                );
            }

            let mmap_node = match GraphStore::get_node(&*self.mmap, nid) {
                Some(n) => n,
                None => continue,
            };

            if mmap_node.properties.is_empty() {
                continue;
            }

            // Get delta property keys to skip (delta wins)
            let delta_keys: std::collections::HashSet<String> =
                if let Some(dn) = GraphStore::get_node(&*self.delta, nid) {
                    dn.properties.iter().map(|(k, _)| k.as_str().to_string()).collect()
                } else {
                    std::collections::HashSet::new()
                };

            let mut node_copied = false;
            for (key, value) in mmap_node.properties.iter() {
                let k = key.as_str();
                if !delta_keys.contains(k) {
                    self.delta.set_node_property(nid, k, value.clone());
                    props_copied += 1;
                    node_copied = true;
                }
            }
            if node_copied {
                nodes_merged += 1;
            }
        }

        (nodes_merged, props_copied)
    }
}
