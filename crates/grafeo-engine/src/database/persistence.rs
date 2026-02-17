//! Persistence, snapshots, and data export for GrafeoDB.

use std::path::Path;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};

use crate::config::Config;

#[cfg(feature = "wal")]
use grafeo_adapters::storage::wal::WalRecord;

/// Binary snapshot format for database export/import.
#[derive(serde::Serialize, serde::Deserialize)]
struct Snapshot {
    version: u8,
    nodes: Vec<SnapshotNode>,
    edges: Vec<SnapshotEdge>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotNode {
    id: NodeId,
    labels: Vec<String>,
    properties: Vec<(String, Value)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotEdge {
    id: EdgeId,
    src: NodeId,
    dst: NodeId,
    edge_type: String,
    properties: Vec<(String, Value)>,
}

impl super::GrafeoDB {
    // =========================================================================
    // ADMIN API: Persistence Control
    // =========================================================================

    /// Saves the database to a file path.
    ///
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the save operation fails.
    ///
    /// Requires the `wal` feature for persistence support.
    #[cfg(feature = "wal")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Create target database with WAL enabled
        let target_config = Config::persistent(path);
        let target = Self::with_config(target_config)?;

        // Copy all nodes using WAL-enabled methods
        for node in self.store.all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target.store.create_node_with_id(node.id, &label_refs);

            // Log to WAL
            target.log_wal(&WalRecord::CreateNode {
                id: node.id,
                labels: node.labels.iter().map(|s| s.to_string()).collect(),
            })?;

            // Copy properties
            for (key, value) in node.properties {
                target
                    .store
                    .set_node_property(node.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetNodeProperty {
                    id: node.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy all edges using WAL-enabled methods
        for edge in self.store.all_edges() {
            target
                .store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type);

            // Log to WAL
            target.log_wal(&WalRecord::CreateEdge {
                id: edge.id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.to_string(),
            })?;

            // Copy properties
            for (key, value) in edge.properties {
                target
                    .store
                    .set_edge_property(edge.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetEdgeProperty {
                    id: edge.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Checkpoint and close the target database
        target.close()?;

        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent.
    /// Useful for:
    /// - Testing modifications without affecting the original
    /// - Faster operations when persistence isn't needed
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    pub fn to_memory(&self) -> Result<Self> {
        let config = Config::in_memory();
        let target = Self::with_config(config)?;

        // Copy all nodes
        for node in self.store.all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target.store.create_node_with_id(node.id, &label_refs);

            // Copy properties
            for (key, value) in node.properties {
                target.store.set_node_property(node.id, key.as_str(), value);
            }
        }

        // Copy all edges
        for edge in self.store.all_edges() {
            target
                .store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type);

            // Copy properties
            for (key, value) in edge.properties {
                target.store.set_edge_property(edge.id, key.as_str(), value);
            }
        }

        Ok(target)
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or loaded.
    #[cfg(feature = "wal")]
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        // Open the source database (triggers WAL recovery)
        let source = Self::open(path)?;

        // Create in-memory copy
        let target = source.to_memory()?;

        // Close the source (releases file handles)
        source.close()?;

        Ok(target)
    }

    // =========================================================================
    // ADMIN API: Snapshot Export/Import
    // =========================================================================

    /// Exports the entire database to a binary snapshot.
    ///
    /// The returned bytes can be stored (e.g. in IndexedDB) and later
    /// restored with [`import_snapshot()`](Self::import_snapshot).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_snapshot(&self) -> Result<Vec<u8>> {
        let nodes: Vec<SnapshotNode> = self
            .store
            .all_nodes()
            .map(|n| SnapshotNode {
                id: n.id,
                labels: n.labels.iter().map(|l| l.to_string()).collect(),
                properties: n
                    .properties
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            })
            .collect();

        let edges: Vec<SnapshotEdge> = self
            .store
            .all_edges()
            .map(|e| SnapshotEdge {
                id: e.id,
                src: e.src,
                dst: e.dst,
                edge_type: e.edge_type.to_string(),
                properties: e
                    .properties
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            })
            .collect();

        let snapshot = Snapshot {
            version: 1,
            nodes,
            edges,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("snapshot export failed: {e}")))
    }

    /// Creates a new in-memory database from a binary snapshot.
    ///
    /// The `data` must have been produced by [`export_snapshot()`](Self::export_snapshot).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid or deserialization fails.
    pub fn import_snapshot(data: &[u8]) -> Result<Self> {
        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Internal(format!("snapshot import failed: {e}")))?;

        if snapshot.version != 1 {
            return Err(Error::Internal(format!(
                "unsupported snapshot version: {}",
                snapshot.version
            )));
        }

        let db = Self::new_in_memory();

        for node in snapshot.nodes {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
            db.store.create_node_with_id(node.id, &label_refs);
            for (key, value) in node.properties {
                db.store.set_node_property(node.id, &key, value);
            }
        }

        for edge in snapshot.edges {
            db.store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type);
            for (key, value) in edge.properties {
                db.store.set_edge_property(edge.id, &key, value);
            }
        }

        Ok(db)
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        self.store.all_nodes()
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        self.store.all_edges()
    }
}
