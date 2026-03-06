//! WAL record types and the [`WalEntry`] trait.

use grafeo_common::types::{EdgeId, NodeId, TransactionId, Value};
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
    fn make_checkpoint(transaction_id: TransactionId) -> Self;
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

    /// Remove a property from a node.
    RemoveNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Property key.
        key: String,
    },

    /// Remove a property from an edge.
    RemoveEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Property key.
        key: String,
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

    /// Alter a node type (add/drop properties).
    AlterNodeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter an edge type (add/drop properties).
    AlterEdgeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter a graph type (add/drop node/edge types).
    AlterGraphType {
        /// Graph type name.
        name: String,
        /// Alterations: ("add_node_type"|"drop_node_type"|"add_edge_type"|"drop_edge_type", type_name).
        alterations: Vec<(String, String)>,
    },

    /// Create a stored procedure.
    CreateProcedure {
        /// Procedure name.
        name: String,
        /// Parameters: (name, type).
        params: Vec<(String, String)>,
        /// Return columns: (name, type).
        returns: Vec<(String, String)>,
        /// Raw GQL body.
        body: String,
    },

    /// Drop a stored procedure.
    DropProcedure {
        /// Procedure name.
        name: String,
    },

    // === Transaction Control ===
    /// Transaction commit.
    TransactionCommit {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Transaction abort.
    TransactionAbort {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Checkpoint marker.
    Checkpoint {
        /// Transaction ID at checkpoint.
        transaction_id: TransactionId,
    },
}

impl WalEntry for WalRecord {
    fn requires_sync(&self) -> bool {
        matches!(self, WalRecord::TransactionCommit { .. })
    }

    fn is_commit(&self) -> bool {
        matches!(self, WalRecord::TransactionCommit { .. })
    }

    fn is_abort(&self) -> bool {
        matches!(self, WalRecord::TransactionAbort { .. })
    }

    fn is_checkpoint(&self) -> bool {
        matches!(self, WalRecord::Checkpoint { .. })
    }

    fn make_checkpoint(transaction_id: TransactionId) -> Self {
        WalRecord::Checkpoint { transaction_id }
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
            value: Value::String("Alix".into()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { id, key, value } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "name");
                assert_eq!(value, Value::String("Alix".into()));
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
    fn test_remove_node_property_roundtrip() {
        let record = WalRecord::RemoveNodeProperty {
            id: NodeId::new(5),
            key: "age".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveNodeProperty { id, key } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "age");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_edge_property_roundtrip() {
        let record = WalRecord::RemoveEdgeProperty {
            id: EdgeId::new(7),
            key: "weight".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveEdgeProperty { id, key } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
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
        let record = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(100),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionCommit { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(100));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_abort_roundtrip() {
        let record = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(200),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionAbort { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(200));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let record = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(50),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(50));
            }
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

        // Only TransactionCommit should require sync
        assert!(
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1)
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
            !WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::Checkpoint {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );
    }

    #[test]
    fn test_wal_entry_transaction_markers() {
        use super::WalEntry;

        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };
        assert!(commit.is_commit());
        assert!(!commit.is_abort());
        assert!(!commit.is_checkpoint());

        let abort = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(2),
        };
        assert!(!abort.is_commit());
        assert!(abort.is_abort());
        assert!(!abort.is_checkpoint());

        let checkpoint = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(3),
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

        let record = WalRecord::make_checkpoint(TransactionId::new(42));
        match record {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(42));
            }
            _ => panic!("make_checkpoint should produce Checkpoint variant"),
        }
    }
}
