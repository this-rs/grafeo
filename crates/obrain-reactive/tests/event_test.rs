//! Comprehensive tests for event types: MutationEvent, MutationBatch,
//! NodeSnapshot, EdgeSnapshot, EntityRef.

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_reactive::{EdgeSnapshot, MutationBatch, MutationEvent, NodeSnapshot};

// --- Helpers ---

fn make_node(id: u64, labels: &[&str]) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: labels.iter().map(|l| arcstr::ArcStr::from(*l)).collect(),
        properties: vec![],
    }
}

fn make_node_with_props(
    id: u64,
    labels: &[&str],
    props: Vec<(PropertyKey, Value)>,
) -> NodeSnapshot {
    NodeSnapshot {
        id: NodeId::new(id),
        labels: labels.iter().map(|l| arcstr::ArcStr::from(*l)).collect(),
        properties: props,
    }
}

fn make_edge(id: u64, src: u64, dst: u64, edge_type: &str) -> EdgeSnapshot {
    EdgeSnapshot {
        id: EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: arcstr::ArcStr::from(edge_type),
        properties: vec![],
    }
}

fn make_edge_with_props(
    id: u64,
    src: u64,
    dst: u64,
    edge_type: &str,
    props: Vec<(PropertyKey, Value)>,
) -> EdgeSnapshot {
    EdgeSnapshot {
        id: EdgeId::new(id),
        src: NodeId::new(src),
        dst: NodeId::new(dst),
        edge_type: arcstr::ArcStr::from(edge_type),
        properties: props,
    }
}

// ========================================================================
// NodeSnapshot tests
// ========================================================================

#[test]
fn node_snapshot_fields() {
    let snap = make_node_with_props(
        10,
        &["Person", "Employee"],
        vec![
            (
                PropertyKey::from("name"),
                Value::String(arcstr::literal!("Alice")),
            ),
            (PropertyKey::from("age"), Value::Int64(30)),
        ],
    );
    assert_eq!(snap.id, NodeId::new(10));
    assert_eq!(snap.labels.len(), 2);
    assert_eq!(snap.labels[0].as_str(), "Person");
    assert_eq!(snap.labels[1].as_str(), "Employee");
    assert_eq!(snap.properties.len(), 2);
}

#[test]
fn node_snapshot_no_labels_no_props() {
    let snap = make_node(1, &[]);
    assert_eq!(snap.id, NodeId::new(1));
    assert!(snap.labels.is_empty());
    assert!(snap.properties.is_empty());
}

#[test]
fn node_snapshot_clone() {
    let snap = make_node_with_props(
        5,
        &["A"],
        vec![(PropertyKey::from("key"), Value::Int64(42))],
    );
    let cloned = snap.clone();
    assert_eq!(cloned.id, snap.id);
    assert_eq!(cloned.labels.len(), snap.labels.len());
    assert_eq!(cloned.properties.len(), snap.properties.len());
}

#[test]
fn node_snapshot_debug() {
    let snap = make_node(1, &["Test"]);
    let debug = format!("{:?}", snap);
    assert!(debug.contains("NodeSnapshot"));
    assert!(debug.contains("Test"));
}

// ========================================================================
// EdgeSnapshot tests
// ========================================================================

#[test]
fn edge_snapshot_fields() {
    let snap = make_edge_with_props(
        20,
        1,
        2,
        "KNOWS",
        vec![(PropertyKey::from("weight"), Value::Float64(0.5))],
    );
    assert_eq!(snap.id, EdgeId::new(20));
    assert_eq!(snap.src, NodeId::new(1));
    assert_eq!(snap.dst, NodeId::new(2));
    assert_eq!(snap.edge_type.as_str(), "KNOWS");
    assert_eq!(snap.properties.len(), 1);
}

#[test]
fn edge_snapshot_no_props() {
    let snap = make_edge(1, 10, 20, "FOLLOWS");
    assert_eq!(snap.id, EdgeId::new(1));
    assert_eq!(snap.src, NodeId::new(10));
    assert_eq!(snap.dst, NodeId::new(20));
    assert_eq!(snap.edge_type.as_str(), "FOLLOWS");
    assert!(snap.properties.is_empty());
}

#[test]
fn edge_snapshot_clone() {
    let snap = make_edge_with_props(
        3,
        1,
        2,
        "LIKES",
        vec![(
            PropertyKey::from("since"),
            Value::String(arcstr::literal!("2024")),
        )],
    );
    let cloned = snap.clone();
    assert_eq!(cloned.id, snap.id);
    assert_eq!(cloned.src, snap.src);
    assert_eq!(cloned.dst, snap.dst);
    assert_eq!(cloned.edge_type, snap.edge_type);
    assert_eq!(cloned.properties.len(), snap.properties.len());
}

#[test]
fn edge_snapshot_debug() {
    let snap = make_edge(1, 1, 2, "KNOWS");
    let debug = format!("{:?}", snap);
    assert!(debug.contains("EdgeSnapshot"));
    assert!(debug.contains("KNOWS"));
}

// ========================================================================
// MutationEvent::kind() — all six variants
// ========================================================================

#[test]
fn event_kind_node_created() {
    let event = MutationEvent::NodeCreated {
        node: make_node(1, &["A"]),
    };
    assert_eq!(event.kind(), "node_created");
}

#[test]
fn event_kind_node_updated() {
    let event = MutationEvent::NodeUpdated {
        before: make_node(1, &["A"]),
        after: make_node(1, &["A", "B"]),
    };
    assert_eq!(event.kind(), "node_updated");
}

#[test]
fn event_kind_node_deleted() {
    let event = MutationEvent::NodeDeleted {
        node: make_node(1, &["A"]),
    };
    assert_eq!(event.kind(), "node_deleted");
}

#[test]
fn event_kind_edge_created() {
    let event = MutationEvent::EdgeCreated {
        edge: make_edge(1, 1, 2, "KNOWS"),
    };
    assert_eq!(event.kind(), "edge_created");
}

#[test]
fn event_kind_edge_updated() {
    let event = MutationEvent::EdgeUpdated {
        before: make_edge(1, 1, 2, "KNOWS"),
        after: make_edge(1, 1, 2, "KNOWS"),
    };
    assert_eq!(event.kind(), "edge_updated");
}

#[test]
fn event_kind_edge_deleted() {
    let event = MutationEvent::EdgeDeleted {
        edge: make_edge(1, 1, 2, "KNOWS"),
    };
    assert_eq!(event.kind(), "edge_deleted");
}

// ========================================================================
// MutationEvent::entity_id() — all six variants
// ========================================================================

#[test]
fn entity_id_node_created() {
    let event = MutationEvent::NodeCreated {
        node: make_node(42, &["Test"]),
    };
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Node"));
}

#[test]
fn entity_id_node_updated() {
    let event = MutationEvent::NodeUpdated {
        before: make_node(10, &["A"]),
        after: make_node(10, &["A", "B"]),
    };
    // entity_id uses `after.id`
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Node"));
}

#[test]
fn entity_id_node_deleted() {
    let event = MutationEvent::NodeDeleted {
        node: make_node(99, &["Temp"]),
    };
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Node"));
}

#[test]
fn entity_id_edge_created() {
    let event = MutationEvent::EdgeCreated {
        edge: make_edge(7, 1, 2, "KNOWS"),
    };
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Edge"));
}

#[test]
fn entity_id_edge_updated() {
    let event = MutationEvent::EdgeUpdated {
        before: make_edge(5, 1, 2, "LIKES"),
        after: make_edge(5, 1, 2, "LIKES"),
    };
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Edge"));
}

#[test]
fn entity_id_edge_deleted() {
    let event = MutationEvent::EdgeDeleted {
        edge: make_edge(3, 2, 3, "FOLLOWS"),
    };
    let id = event.entity_id();
    assert!(format!("{:?}", id).contains("Edge"));
}

// ========================================================================
// MutationEvent Clone + Debug
// ========================================================================

#[test]
fn event_clone_preserves_data() {
    let event = MutationEvent::NodeCreated {
        node: make_node_with_props(
            1,
            &["Person"],
            vec![(PropertyKey::from("name"), Value::String("Alice".into()))],
        ),
    };
    let cloned = event.clone();
    assert_eq!(cloned.kind(), "node_created");
    if let MutationEvent::NodeCreated { node } = &cloned {
        assert_eq!(node.id, NodeId::new(1));
        assert_eq!(node.properties.len(), 1);
    } else {
        panic!("expected NodeCreated");
    }
}

#[test]
fn event_debug_all_variants() {
    let events = vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["A"]),
        },
        MutationEvent::NodeUpdated {
            before: make_node(1, &["A"]),
            after: make_node(1, &["A", "B"]),
        },
        MutationEvent::NodeDeleted {
            node: make_node(1, &["A"]),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
        MutationEvent::EdgeUpdated {
            before: make_edge(1, 1, 2, "KNOWS"),
            after: make_edge(1, 1, 2, "KNOWS"),
        },
        MutationEvent::EdgeDeleted {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
    ];
    for event in &events {
        let debug = format!("{:?}", event);
        assert!(!debug.is_empty());
    }
}

// ========================================================================
// MutationBatch tests
// ========================================================================

#[test]
fn batch_new_empty() {
    let batch = MutationBatch::new(vec![]);
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
}

#[test]
fn batch_new_with_events() {
    let events = vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["A"]),
        },
        MutationEvent::NodeCreated {
            node: make_node(2, &["B"]),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
    ];
    let batch = MutationBatch::new(events);
    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 3);
}

#[test]
fn batch_timestamp_is_recent() {
    let before = std::time::Instant::now();
    let batch = MutationBatch::new(vec![MutationEvent::NodeCreated {
        node: make_node(1, &[]),
    }]);
    let after = std::time::Instant::now();
    assert!(batch.timestamp >= before);
    assert!(batch.timestamp <= after);
}

#[test]
fn batch_clone() {
    let batch = MutationBatch::new(vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["A"]),
        },
        MutationEvent::EdgeCreated {
            edge: make_edge(1, 1, 2, "KNOWS"),
        },
    ]);
    let cloned = batch.clone();
    assert_eq!(cloned.len(), batch.len());
    assert_eq!(cloned.events[0].kind(), batch.events[0].kind());
    assert_eq!(cloned.events[1].kind(), batch.events[1].kind());
}

#[test]
fn batch_debug() {
    let batch = MutationBatch::new(vec![MutationEvent::NodeCreated {
        node: make_node(1, &["Test"]),
    }]);
    let debug = format!("{:?}", batch);
    assert!(debug.contains("MutationBatch"));
}

#[test]
fn batch_events_accessible() {
    let batch = MutationBatch::new(vec![
        MutationEvent::NodeCreated {
            node: make_node(1, &["A"]),
        },
        MutationEvent::NodeDeleted {
            node: make_node(2, &["B"]),
        },
    ]);
    assert_eq!(batch.events[0].kind(), "node_created");
    assert_eq!(batch.events[1].kind(), "node_deleted");
}

#[test]
fn batch_single_event() {
    let batch = MutationBatch::new(vec![MutationEvent::EdgeDeleted {
        edge: make_edge(5, 1, 2, "LIKES"),
    }]);
    assert!(!batch.is_empty());
    assert_eq!(batch.len(), 1);
}

#[test]
fn batch_large() {
    let events: Vec<MutationEvent> = (0..1000)
        .map(|i| MutationEvent::NodeCreated {
            node: make_node(i, &["Bulk"]),
        })
        .collect();
    let batch = MutationBatch::new(events);
    assert_eq!(batch.len(), 1000);
    assert!(!batch.is_empty());
}
