//! Tests for obrain-reactive crate.

#[cfg(test)]
mod reactive_tests {
    use crate::bus::MutationBus;
    use crate::event::{EdgeSnapshot, MutationBatch, MutationEvent, NodeSnapshot};
    use obrain_common::types::{EdgeId, NodeId};

    fn make_node_snapshot(id: u64, labels: &[&str]) -> NodeSnapshot {
        NodeSnapshot {
            id: NodeId::new(id),
            labels: labels.iter().map(|l| arcstr::ArcStr::from(*l)).collect(),
            properties: vec![],
        }
    }

    fn make_edge_snapshot(id: u64, src: u64, dst: u64, edge_type: &str) -> EdgeSnapshot {
        EdgeSnapshot {
            id: EdgeId::new(id),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            edge_type: arcstr::ArcStr::from(edge_type),
            properties: vec![],
        }
    }

    #[test]
    fn bus_no_subscribers_returns_false() {
        let bus = MutationBus::new();
        let event = MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &["Person"]),
        };
        assert!(!bus.publish(event));
        assert_eq!(bus.total_events_published(), 0);
    }

    #[test]
    fn bus_empty_batch_returns_false() {
        let bus = MutationBus::new();
        let _rx = bus.subscribe();
        let batch = MutationBatch::new(vec![]);
        assert!(!bus.publish_batch(batch));
    }

    #[tokio::test]
    async fn bus_with_subscriber_receives_events() {
        let bus = MutationBus::new();
        let mut rx = bus.subscribe();

        let event = MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &["Person"]),
        };
        assert!(bus.publish(event));
        assert_eq!(bus.total_events_published(), 1);
        assert_eq!(bus.total_batches_published(), 1);

        let batch = rx.recv().await.unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.events[0].kind(), "node_created");
    }

    #[tokio::test]
    async fn bus_batch_multiple_events() {
        let bus = MutationBus::new();
        let mut rx = bus.subscribe();

        let events = vec![
            MutationEvent::NodeCreated {
                node: make_node_snapshot(1, &["Person"]),
            },
            MutationEvent::EdgeCreated {
                edge: make_edge_snapshot(1, 1, 2, "KNOWS"),
            },
            MutationEvent::NodeUpdated {
                before: make_node_snapshot(2, &["Person"]),
                after: make_node_snapshot(2, &["Person", "Employee"]),
            },
        ];

        let batch = MutationBatch::new(events);
        assert!(bus.publish_batch(batch));
        assert_eq!(bus.total_events_published(), 3);

        let received = rx.recv().await.unwrap();
        assert_eq!(received.len(), 3);
        assert_eq!(received.events[0].kind(), "node_created");
        assert_eq!(received.events[1].kind(), "edge_created");
        assert_eq!(received.events[2].kind(), "node_updated");
    }

    #[test]
    fn bus_subscriber_count() {
        let bus = MutationBus::new();
        assert_eq!(bus.subscriber_count(), 0);

        let _rx1 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        let _rx2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);

        drop(_rx1);
        assert_eq!(bus.subscriber_count(), 1);
    }

    #[tokio::test]
    async fn bus_multiple_subscribers() {
        let bus = MutationBus::new();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let event = MutationEvent::NodeDeleted {
            node: make_node_snapshot(1, &["Person"]),
        };
        bus.publish(event);

        let b1 = rx1.recv().await.unwrap();
        let b2 = rx2.recv().await.unwrap();
        assert_eq!(b1.len(), 1);
        assert_eq!(b2.len(), 1);
        assert_eq!(b1.events[0].kind(), "node_deleted");
        assert_eq!(b2.events[0].kind(), "node_deleted");
    }

    #[test]
    fn event_entity_ref() {
        use crate::event::EntityRef;

        let event = MutationEvent::NodeCreated {
            node: make_node_snapshot(42, &["Test"]),
        };
        assert_eq!(event.entity_id(), EntityRef::Node(NodeId::new(42)));

        let event = MutationEvent::EdgeDeleted {
            edge: make_edge_snapshot(7, 1, 2, "KNOWS"),
        };
        assert_eq!(event.entity_id(), EntityRef::Edge(EdgeId::new(7)));
    }

    #[test]
    fn mutation_batch_len_and_empty() {
        let empty = MutationBatch::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let batch = MutationBatch::new(vec![MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &[]),
        }]);
        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    // --- EntityRef for all 6 event variants ---

    #[test]
    fn entity_ref_all_variants() {
        use crate::event::EntityRef;

        // NodeCreated
        let e = MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &["A"]),
        };
        assert_eq!(e.entity_id(), EntityRef::Node(NodeId::new(1)));

        // NodeUpdated (uses after.id)
        let e = MutationEvent::NodeUpdated {
            before: make_node_snapshot(2, &["A"]),
            after: make_node_snapshot(2, &["A", "B"]),
        };
        assert_eq!(e.entity_id(), EntityRef::Node(NodeId::new(2)));

        // NodeDeleted
        let e = MutationEvent::NodeDeleted {
            node: make_node_snapshot(3, &["X"]),
        };
        assert_eq!(e.entity_id(), EntityRef::Node(NodeId::new(3)));

        // EdgeCreated
        let e = MutationEvent::EdgeCreated {
            edge: make_edge_snapshot(10, 1, 2, "KNOWS"),
        };
        assert_eq!(e.entity_id(), EntityRef::Edge(EdgeId::new(10)));

        // EdgeUpdated (uses after.id)
        let e = MutationEvent::EdgeUpdated {
            before: make_edge_snapshot(11, 1, 2, "LIKES"),
            after: make_edge_snapshot(11, 1, 2, "LIKES"),
        };
        assert_eq!(e.entity_id(), EntityRef::Edge(EdgeId::new(11)));

        // EdgeDeleted
        let e = MutationEvent::EdgeDeleted {
            edge: make_edge_snapshot(12, 3, 4, "FOLLOWS"),
        };
        assert_eq!(e.entity_id(), EntityRef::Edge(EdgeId::new(12)));
    }

    // --- kind() for all 6 event variants ---

    #[test]
    fn event_kind_all_variants() {
        assert_eq!(
            MutationEvent::NodeCreated {
                node: make_node_snapshot(1, &[])
            }
            .kind(),
            "node_created"
        );
        assert_eq!(
            MutationEvent::NodeUpdated {
                before: make_node_snapshot(1, &[]),
                after: make_node_snapshot(1, &[])
            }
            .kind(),
            "node_updated"
        );
        assert_eq!(
            MutationEvent::NodeDeleted {
                node: make_node_snapshot(1, &[])
            }
            .kind(),
            "node_deleted"
        );
        assert_eq!(
            MutationEvent::EdgeCreated {
                edge: make_edge_snapshot(1, 1, 2, "X")
            }
            .kind(),
            "edge_created"
        );
        assert_eq!(
            MutationEvent::EdgeUpdated {
                before: make_edge_snapshot(1, 1, 2, "X"),
                after: make_edge_snapshot(1, 1, 2, "X")
            }
            .kind(),
            "edge_updated"
        );
        assert_eq!(
            MutationEvent::EdgeDeleted {
                edge: make_edge_snapshot(1, 1, 2, "X")
            }
            .kind(),
            "edge_deleted"
        );
    }

    // --- EntityRef traits ---

    #[test]
    fn entity_ref_debug_clone_copy_eq_hash() {
        use crate::event::EntityRef;
        use std::collections::HashSet;

        let node_ref = EntityRef::Node(NodeId::new(1));
        let edge_ref = EntityRef::Edge(EdgeId::new(2));

        // Debug
        assert!(format!("{:?}", node_ref).contains("Node"));
        assert!(format!("{:?}", edge_ref).contains("Edge"));

        // Clone/Copy
        let copied = node_ref;
        assert_eq!(copied, node_ref);

        // Eq
        assert_eq!(node_ref, EntityRef::Node(NodeId::new(1)));
        assert_ne!(node_ref, EntityRef::Node(NodeId::new(2)));
        assert_ne!(node_ref, edge_ref);

        // Hash — can be used as a HashSet key
        let mut set = HashSet::new();
        set.insert(node_ref);
        set.insert(edge_ref);
        set.insert(EntityRef::Node(NodeId::new(1))); // duplicate
        assert_eq!(set.len(), 2);
    }

    // --- MutationBatch clone and timestamp ---

    #[test]
    fn mutation_batch_clone_preserves_events() {
        let batch = MutationBatch::new(vec![
            MutationEvent::NodeCreated {
                node: make_node_snapshot(1, &["A"]),
            },
            MutationEvent::EdgeCreated {
                edge: make_edge_snapshot(1, 1, 2, "KNOWS"),
            },
        ]);
        let cloned = batch.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.events[0].kind(), "node_created");
        assert_eq!(cloned.events[1].kind(), "edge_created");
    }

    #[test]
    fn mutation_batch_timestamp_is_set() {
        let before = std::time::Instant::now();
        let batch = MutationBatch::new(vec![]);
        let after = std::time::Instant::now();
        assert!(batch.timestamp >= before);
        assert!(batch.timestamp <= after);
    }

    // --- MutationEvent clone and debug ---

    #[test]
    fn mutation_event_clone_preserves_variant() {
        let event = MutationEvent::EdgeUpdated {
            before: make_edge_snapshot(1, 1, 2, "X"),
            after: make_edge_snapshot(1, 1, 2, "X"),
        };
        let cloned = event.clone();
        assert_eq!(cloned.kind(), "edge_updated");
    }

    #[test]
    fn mutation_event_debug_is_nonempty() {
        let event = MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &["A"]),
        };
        let debug = format!("{:?}", event);
        assert!(!debug.is_empty());
        assert!(debug.contains("NodeCreated"));
    }

    // --- NodeSnapshot / EdgeSnapshot with properties ---

    #[test]
    fn node_snapshot_with_properties() {
        use obrain_common::types::{PropertyKey, Value};

        let snap = NodeSnapshot {
            id: NodeId::new(5),
            labels: smallvec::smallvec![arcstr::literal!("Person")],
            properties: vec![
                (PropertyKey::from("name"), Value::String("Alice".into())),
                (PropertyKey::from("age"), Value::Int64(30)),
            ],
        };
        assert_eq!(snap.properties.len(), 2);
        let debug = format!("{:?}", snap);
        assert!(debug.contains("Person"));
    }

    #[test]
    fn edge_snapshot_with_properties() {
        use obrain_common::types::{PropertyKey, Value};

        let snap = EdgeSnapshot {
            id: EdgeId::new(10),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: arcstr::literal!("KNOWS"),
            properties: vec![(PropertyKey::from("weight"), Value::Float64(0.5))],
        };
        assert_eq!(snap.properties.len(), 1);
        assert_eq!(snap.edge_type.as_str(), "KNOWS");
    }

    // --- Bus default and with_capacity ---

    #[test]
    fn bus_default_impl() {
        let bus = MutationBus::default();
        assert_eq!(bus.subscriber_count(), 0);
        assert_eq!(bus.total_events_published(), 0);
    }

    #[test]
    fn bus_with_capacity() {
        let bus = MutationBus::with_capacity(8);
        let _rx = bus.subscribe();
        // Should work normally
        let event = MutationEvent::NodeCreated {
            node: make_node_snapshot(1, &["A"]),
        };
        assert!(bus.publish(event));
        assert_eq!(bus.total_events_published(), 1);
    }

    #[test]
    fn bus_debug() {
        let bus = MutationBus::new();
        let debug = format!("{:?}", bus);
        assert!(debug.contains("MutationBus"));
    }
}
