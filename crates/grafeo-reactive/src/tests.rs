//! Tests for grafeo-reactive crate.

#[cfg(test)]
mod tests {
    use crate::bus::MutationBus;
    use crate::event::{EdgeSnapshot, MutationBatch, MutationEvent, NodeSnapshot};
    use grafeo_common::types::{EdgeId, NodeId};

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
}
