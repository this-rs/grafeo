//! WAL record types and the [`WalEntry`] trait.

use grafeo_common::types::{EdgeId, NodeId, TxId, Value};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Trait for WAL record types, enabling type-safe WAL instances.
///
/// Implement this for each storage model's record enum (e.g., [`WalRecord`]
/// for LPG, a future `RdfWalRecord` for RDF). The [`TypedWal`](super::TypedWal)
/// wrapper uses these methods to handle durability decisions and transaction
/// semantics without knowing the concrete record type.
pub trait WalEntry: Serialize + DeserializeOwned + Send + Sync + std::fmt::Debug + Clone {
    /// Whether this record should force an immediate fsync in Sync durability mode.
    ///
    /// Returns `true` for commit markers.
    fn requires_sync(&self) -> bool;

    /// Whether this is a transaction commit record.
    fn is_commit(&self) -> bool;

    /// Whether this is a transaction abort record.
    fn is_abort(&self) -> bool;

    /// Whether this is a checkpoint record.
    fn is_checkpoint(&self) -> bool;

    /// Creates a checkpoint record for this WAL type.
    fn make_checkpoint(tx_id: TxId) -> Self;
}

/// A record in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    /// Create a new node.
    CreateNode {
        /// Node ID.
        id: NodeId,
        /// Labels for the node.
        labels: Vec<String>,
    },

    /// Delete a node.
    DeleteNode {
        /// Node ID.
        id: NodeId,
    },

    /// Create a new edge.
    CreateEdge {
        /// Edge ID.
        id: EdgeId,
        /// Source node ID.
        src: NodeId,
        /// Destination node ID.
        dst: NodeId,
        /// Edge type.
        edge_type: String,
    },

    /// Delete an edge.
    DeleteEdge {
        /// Edge ID.
        id: EdgeId,
    },

    /// Set a property on a node.
    SetNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Property key.
        key: String,
        /// Property value.
        value: Value,
    },

    /// Set a property on an edge.
    SetEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Property key.
        key: String,
        /// Property value.
        value: Value,
    },

    /// Add a label to a node.
    AddNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label to add.
        label: String,
    },

    /// Remove a label from a node.
    RemoveNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label to remove.
        label: String,
    },

    // === Schema DDL Records ===
    /// Register a node type definition.
    CreateNodeType {
        /// Type name (corresponds to a label).
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names). kind = "unique", "primary_key", "not_null".
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop a node type definition.
    DropNodeType {
        /// Type name.
        name: String,
    },

    /// Register an edge type definition.
    CreateEdgeType {
        /// Type name.
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names).
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop an edge type definition.
    DropEdgeType {
        /// Type name.
        name: String,
    },

    /// Create an index.
    CreateIndex {
        /// Index name.
        name: String,
        /// Target label.
        label: String,
        /// Target property.
        property: String,
        /// Index kind: "property", "text", "vector", "btree".
        index_type: String,
    },

    /// Drop an index.
    DropIndex {
        /// Index name.
        name: String,
    },

    /// Create a constraint.
    CreateConstraint {
        /// Constraint name.
        name: String,
        /// Target label.
        label: String,
        /// Target properties.
        properties: Vec<String>,
        /// Constraint kind: "unique", "node_key", "not_null", "exists".
        kind: String,
    },

    /// Drop a constraint.
    DropConstraint {
        /// Constraint name.
        name: String,
    },

    /// Register a graph type definition.
    CreateGraphType {
        /// Type name.
        name: String,
        /// Allowed node types.
        node_types: Vec<String>,
        /// Allowed edge types.
        edge_types: Vec<String>,
        /// Whether unlisted types are allowed.
        open: bool,
    },

    /// Drop a graph type definition.
    DropGraphType {
        /// Type name.
        name: String,
    },

    /// Register a schema namespace.
    CreateSchema {
        /// Schema name.
        name: String,
    },

    /// Drop a schema namespace.
    DropSchema {
        /// Schema name.
        name: String,
    },

    // === Transaction Control ===
    /// Transaction commit.
    TxCommit {
        /// Transaction ID.
        tx_id: TxId,
    },

    /// Transaction abort.
    TxAbort {
        /// Transaction ID.
        tx_id: TxId,
    },

    /// Checkpoint marker.
    Checkpoint {
        /// Transaction ID at checkpoint.
        tx_id: TxId,
    },
}

impl WalEntry for WalRecord {
    fn requires_sync(&self) -> bool {
        matches!(self, WalRecord::TxCommit { .. })
    }

    fn is_commit(&self) -> bool {
        matches!(self, WalRecord::TxCommit { .. })
    }

    fn is_abort(&self) -> bool {
        matches!(self, WalRecord::TxAbort { .. })
    }

    fn is_checkpoint(&self) -> bool {
        matches!(self, WalRecord::Checkpoint { .. })
    }

    fn make_checkpoint(tx_id: TxId) -> Self {
        WalRecord::Checkpoint { tx_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(record: &WalRecord) -> WalRecord {
        let json = serde_json::to_string(record).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn test_create_node_roundtrip() {
        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string(), "Employee".to_string()],
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateNode { id, labels } => {
                assert_eq!(id, NodeId::new(1));
                assert_eq!(labels, vec!["Person", "Employee"]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_node_roundtrip() {
        let record = WalRecord::DeleteNode {
            id: NodeId::new(42),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::DeleteNode { id } => assert_eq!(id, NodeId::new(42)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_edge_roundtrip() {
        let record = WalRecord::CreateEdge {
            id: EdgeId::new(10),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "KNOWS".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateEdge {
                id,
                src,
                dst,
                edge_type,
            } => {
                assert_eq!(id, EdgeId::new(10));
                assert_eq!(src, NodeId::new(1));
                assert_eq!(dst, NodeId::new(2));
                assert_eq!(edge_type, "KNOWS");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_edge_roundtrip() {
        let record = WalRecord::DeleteEdge {
            id: EdgeId::new(99),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::DeleteEdge { id } => assert_eq!(id, EdgeId::new(99)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_node_property_roundtrip() {
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(5),
            key: "name".to_string(),
            value: Value::String("Alice".into()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { id, key, value } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "name");
                assert_eq!(value, Value::String("Alice".into()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_edge_property_roundtrip() {
        let record = WalRecord::SetEdgeProperty {
            id: EdgeId::new(7),
            key: "weight".to_string(),
            value: Value::Float64(std::f64::consts::PI),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetEdgeProperty { id, key, value } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
                assert_eq!(value, Value::Float64(std::f64::consts::PI));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_add_node_label_roundtrip() {
        let record = WalRecord::AddNodeLabel {
            id: NodeId::new(3),
            label: "Admin".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::AddNodeLabel { id, label } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Admin");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_node_label_roundtrip() {
        let record = WalRecord::RemoveNodeLabel {
            id: NodeId::new(3),
            label: "Temp".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveNodeLabel { id, label } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Temp");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_commit_roundtrip() {
        let record = WalRecord::TxCommit {
            tx_id: TxId::new(100),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TxCommit { tx_id } => assert_eq!(tx_id, TxId::new(100)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_abort_roundtrip() {
        let record = WalRecord::TxAbort {
            tx_id: TxId::new(200),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TxAbort { tx_id } => assert_eq!(tx_id, TxId::new(200)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let record = WalRecord::Checkpoint {
            tx_id: TxId::new(50),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::Checkpoint { tx_id } => assert_eq!(tx_id, TxId::new(50)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_node_empty_labels() {
        let record = WalRecord::CreateNode {
            id: NodeId::new(0),
            labels: Vec::new(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateNode { labels, .. } => assert!(labels.is_empty()),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_wal_entry_requires_sync() {
        use super::WalEntry;

        // Only TxCommit should require sync
        assert!(
            WalRecord::TxCommit {
                tx_id: TxId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec![]
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::TxAbort {
                tx_id: TxId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::Checkpoint {
                tx_id: TxId::new(1)
            }
            .requires_sync()
        );
    }

    #[test]
    fn test_wal_entry_transaction_markers() {
        use super::WalEntry;

        let commit = WalRecord::TxCommit {
            tx_id: TxId::new(1),
        };
        assert!(commit.is_commit());
        assert!(!commit.is_abort());
        assert!(!commit.is_checkpoint());

        let abort = WalRecord::TxAbort {
            tx_id: TxId::new(2),
        };
        assert!(!abort.is_commit());
        assert!(abort.is_abort());
        assert!(!abort.is_checkpoint());

        let checkpoint = WalRecord::Checkpoint {
            tx_id: TxId::new(3),
        };
        assert!(!checkpoint.is_commit());
        assert!(!checkpoint.is_abort());
        assert!(checkpoint.is_checkpoint());

        // Data records are none of the above
        let data = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        };
        assert!(!data.is_commit());
        assert!(!data.is_abort());
        assert!(!data.is_checkpoint());
    }

    #[test]
    fn test_wal_entry_make_checkpoint() {
        use super::WalEntry;

        let record = WalRecord::make_checkpoint(TxId::new(42));
        match record {
            WalRecord::Checkpoint { tx_id } => assert_eq!(tx_id, TxId::new(42)),
            _ => panic!("make_checkpoint should produce Checkpoint variant"),
        }
    }
}
