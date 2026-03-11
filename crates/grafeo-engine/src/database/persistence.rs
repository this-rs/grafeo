//! Persistence, snapshots, and data export for GrafeoDB.

#[cfg(feature = "wal")]
use std::path::Path;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use hashbrown::HashSet;

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
            target.store.create_node_with_id(node.id, &label_refs)?;

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
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;

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
            target.store.create_node_with_id(node.id, &label_refs)?;

            // Copy properties
            for (key, value) in node.properties {
                target.store.set_node_property(node.id, key.as_str(), value);
            }
        }

        // Copy all edges
        for edge in self.store.all_edges() {
            target
                .store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;

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
    /// All edge references are validated before any data is inserted: every
    /// edge's source and destination must reference a node present in the
    /// snapshot, and duplicate node/edge IDs are rejected. If validation
    /// fails, no database is created.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
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

        // Pre-validate: collect all node IDs and check for duplicates
        let mut node_ids = HashSet::with_capacity(snapshot.nodes.len());
        for node in &snapshot.nodes {
            if !node_ids.insert(node.id) {
                return Err(Error::Internal(format!(
                    "snapshot contains duplicate node ID {}",
                    node.id
                )));
            }
        }

        // Validate edge references and check for duplicate edge IDs
        let mut edge_ids = HashSet::with_capacity(snapshot.edges.len());
        for edge in &snapshot.edges {
            if !edge_ids.insert(edge.id) {
                return Err(Error::Internal(format!(
                    "snapshot contains duplicate edge ID {}",
                    edge.id
                )));
            }
            if !node_ids.contains(&edge.src) {
                return Err(Error::Internal(format!(
                    "snapshot edge {} references non-existent source node {}",
                    edge.id, edge.src
                )));
            }
            if !node_ids.contains(&edge.dst) {
                return Err(Error::Internal(format!(
                    "snapshot edge {} references non-existent destination node {}",
                    edge.id, edge.dst
                )));
            }
        }

        // Validation passed: build the database
        let db = Self::new_in_memory();

        for node in snapshot.nodes {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
            db.store.create_node_with_id(node.id, &label_refs)?;
            for (key, value) in node.properties {
                db.store.set_node_property(node.id, &key, value);
            }
        }

        for edge in snapshot.edges {
            db.store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
            for (key, value) in edge.properties {
                db.store.set_edge_property(edge.id, &key, value);
            }
        }

        Ok(db)
    }

    /// Replaces the current database contents with data from a binary snapshot.
    ///
    /// The `data` must have been produced by [`export_snapshot()`](Self::export_snapshot).
    ///
    /// All validation (duplicate IDs, dangling edge references) is performed
    /// before any data is modified. If validation fails, the current database
    /// is left unchanged. If validation passes, the store is cleared and
    /// rebuilt from the snapshot atomically (from the perspective of
    /// subsequent queries).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid, contains dangling edge
    /// references, has duplicate IDs, or deserialization fails.
    pub fn restore_snapshot(&self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, _): (Snapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| Error::Internal(format!("snapshot restore failed: {e}")))?;

        if snapshot.version != 1 {
            return Err(Error::Internal(format!(
                "unsupported snapshot version: {}",
                snapshot.version
            )));
        }

        // Pre-validate: collect all node IDs and check for duplicates
        let mut node_ids = HashSet::with_capacity(snapshot.nodes.len());
        for node in &snapshot.nodes {
            if !node_ids.insert(node.id) {
                return Err(Error::Internal(format!(
                    "snapshot contains duplicate node ID {}",
                    node.id
                )));
            }
        }

        // Validate edge references and check for duplicate edge IDs
        let mut edge_ids = HashSet::with_capacity(snapshot.edges.len());
        for edge in &snapshot.edges {
            if !edge_ids.insert(edge.id) {
                return Err(Error::Internal(format!(
                    "snapshot contains duplicate edge ID {}",
                    edge.id
                )));
            }
            if !node_ids.contains(&edge.src) {
                return Err(Error::Internal(format!(
                    "snapshot edge {} references non-existent source node {}",
                    edge.id, edge.src
                )));
            }
            if !node_ids.contains(&edge.dst) {
                return Err(Error::Internal(format!(
                    "snapshot edge {} references non-existent destination node {}",
                    edge.id, edge.dst
                )));
            }
        }

        // Validation passed: clear and rebuild
        self.store.clear();

        for node in snapshot.nodes {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
            self.store.create_node_with_id(node.id, &label_refs)?;
            for (key, value) in node.properties {
                self.store.set_node_property(node.id, &key, value);
            }
        }

        for edge in snapshot.edges {
            self.store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
            for (key, value) in edge.properties {
                self.store.set_edge_property(edge.id, &key, value);
            }
        }

        Ok(())
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

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId, Value};

    use super::super::GrafeoDB;
    use super::{Snapshot, SnapshotEdge, SnapshotNode};

    #[test]
    fn test_restore_snapshot_basic() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Populate
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        assert_eq!(db.store.node_count(), 3);

        // Restore original
        db.restore_snapshot(&snapshot).unwrap();

        assert_eq!(db.store.node_count(), 2);
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_restore_snapshot_validation_failure() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        // Corrupt snapshot: just garbage bytes
        let result = db.restore_snapshot(b"garbage");
        assert!(result.is_err());

        // DB should be unchanged
        assert_eq!(db.store.node_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_empty_db() {
        let db = GrafeoDB::new_in_memory();

        // Export empty snapshot, then populate, then restore to empty
        let empty_snapshot = db.export_snapshot().unwrap();

        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        assert_eq!(db.store.node_count(), 1);

        db.restore_snapshot(&empty_snapshot).unwrap();
        assert_eq!(db.store.node_count(), 0);
    }

    #[test]
    fn test_restore_snapshot_with_edges() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        assert_eq!(db.store.edge_count(), 1);

        // Modify: add more data
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(db.store.node_count(), 2);
        assert_eq!(db.store.edge_count(), 1);
    }

    #[test]
    fn test_restore_snapshot_preserves_sessions() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let snapshot = db.export_snapshot().unwrap();

        // Modify
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Restore
        db.restore_snapshot(&snapshot).unwrap();

        // Session should still work and see restored data
        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_export_import_roundtrip() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();

        let snapshot = db.export_snapshot().unwrap();
        let db2 = GrafeoDB::import_snapshot(&snapshot).unwrap();
        let session2 = db2.session();

        let result = session2.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    // --- to_memory() ---

    #[test]
    fn test_to_memory_empty() {
        let db = GrafeoDB::new_in_memory();
        let copy = db.to_memory().unwrap();
        assert_eq!(copy.store.node_count(), 0);
        assert_eq!(copy.store.edge_count(), 0);
    }

    #[test]
    fn test_to_memory_copies_nodes_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 30})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', age: 25})")
            .unwrap();

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.store.node_count(), 2);

        let s2 = copy.session();
        let result = s2
            .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
            .unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][0], Value::String("Alix".into()));
        assert_eq!(result.rows[1][0], Value::String("Gus".into()));
    }

    #[test]
    fn test_to_memory_copies_edges_and_properties() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["Person"]);
        db.set_node_property(a, "name", "Alix".into());
        let b = db.create_node(&["Person"]);
        db.set_node_property(b, "name", "Gus".into());
        let edge = db.create_edge(a, b, "KNOWS");
        db.set_edge_property(edge, "since", Value::Int64(2020));

        let copy = db.to_memory().unwrap();
        assert_eq!(copy.store.node_count(), 2);
        assert_eq!(copy.store.edge_count(), 1);

        let s2 = copy.session();
        let result = s2.execute("MATCH ()-[e:KNOWS]->() RETURN e.since").unwrap();
        assert_eq!(result.rows[0][0], Value::Int64(2020));
    }

    #[test]
    fn test_to_memory_is_independent() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let copy = db.to_memory().unwrap();

        // Mutating original should not affect copy
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        assert_eq!(db.store.node_count(), 2);
        assert_eq!(copy.store.node_count(), 1);
    }

    // --- iter_nodes() / iter_edges() ---

    #[test]
    fn test_iter_nodes_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_nodes().count(), 0);
    }

    #[test]
    fn test_iter_nodes_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let id1 = db.create_node(&["Person"]);
        db.set_node_property(id1, "name", "Alix".into());
        let id2 = db.create_node(&["Animal"]);
        db.set_node_property(id2, "name", "Fido".into());

        let nodes: Vec<_> = db.iter_nodes().collect();
        assert_eq!(nodes.len(), 2);

        let names: Vec<_> = nodes
            .iter()
            .filter_map(|n| n.properties.iter().find(|(k, _)| k.as_str() == "name"))
            .map(|(_, v)| v.clone())
            .collect();
        assert!(names.contains(&Value::String("Alix".into())));
        assert!(names.contains(&Value::String("Fido".into())));
    }

    #[test]
    fn test_iter_edges_empty() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.iter_edges().count(), 0);
    }

    #[test]
    fn test_iter_edges_returns_all() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let c = db.create_node(&["C"]);
        db.create_edge(a, b, "R1");
        db.create_edge(b, c, "R2");

        let edges: Vec<_> = db.iter_edges().collect();
        assert_eq!(edges.len(), 2);

        let types: Vec<_> = edges.iter().map(|e| e.edge_type.as_ref()).collect();
        assert!(types.contains(&"R1"));
        assert!(types.contains(&"R2"));
    }

    // --- restore_snapshot() validation ---

    fn encode_snapshot(snap: &Snapshot) -> Vec<u8> {
        bincode::serde::encode_to_vec(snap, bincode::config::standard()).unwrap()
    }

    #[test]
    fn test_restore_rejects_unsupported_version() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let snap = Snapshot {
            version: 99,
            nodes: vec![],
            edges: vec![],
        };
        let bytes = encode_snapshot(&snap);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("unsupported snapshot version"), "got: {err}");

        // DB unchanged
        assert_eq!(db.store.node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_node_ids() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let snap = Snapshot {
            version: 1,
            nodes: vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["A".into()],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec!["B".into()],
                    properties: vec![],
                },
            ],
            edges: vec![],
        };
        let bytes = encode_snapshot(&snap);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate node ID"), "got: {err}");
        assert_eq!(db.store.node_count(), 1);
    }

    #[test]
    fn test_restore_rejects_duplicate_edge_ids() {
        let db = GrafeoDB::new_in_memory();

        let snap = Snapshot {
            version: 1,
            nodes: vec![
                SnapshotNode {
                    id: NodeId::new(0),
                    labels: vec![],
                    properties: vec![],
                },
                SnapshotNode {
                    id: NodeId::new(1),
                    labels: vec![],
                    properties: vec![],
                },
            ],
            edges: vec![
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
                SnapshotEdge {
                    id: EdgeId::new(0),
                    src: NodeId::new(0),
                    dst: NodeId::new(1),
                    edge_type: "REL".into(),
                    properties: vec![],
                },
            ],
        };
        let bytes = encode_snapshot(&snap);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate edge ID"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_source() {
        let db = GrafeoDB::new_in_memory();

        let snap = Snapshot {
            version: 1,
            nodes: vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            edges: vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(999),
                dst: NodeId::new(0),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        };
        let bytes = encode_snapshot(&snap);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent source node"), "got: {err}");
    }

    #[test]
    fn test_restore_rejects_dangling_destination() {
        let db = GrafeoDB::new_in_memory();

        let snap = Snapshot {
            version: 1,
            nodes: vec![SnapshotNode {
                id: NodeId::new(0),
                labels: vec![],
                properties: vec![],
            }],
            edges: vec![SnapshotEdge {
                id: EdgeId::new(0),
                src: NodeId::new(0),
                dst: NodeId::new(999),
                edge_type: "REL".into(),
                properties: vec![],
            }],
        };
        let bytes = encode_snapshot(&snap);

        let result = db.restore_snapshot(&bytes);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("non-existent destination node"), "got: {err}");
    }
}
