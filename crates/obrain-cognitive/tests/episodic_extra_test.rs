//! Extra coverage tests for the episodic memory subsystem.

#![cfg(feature = "episodic")]

use arcstr::ArcStr;
use obrain_cognitive::episodic::{
    ActivationStep, EpisodeConfig, EpisodeHorizon, EpisodeMemoryManager, EpisodeRecorder,
    EpisodeStore, Outcome, Stimulus, ValidationResult,
};
use obrain_reactive::{EdgeSnapshot, MutationEvent, MutationListener, NodeSnapshot};
use smallvec::smallvec;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn node_snap(id: u64) -> NodeSnapshot {
    NodeSnapshot {
        id: obrain_common::types::NodeId(id),
        labels: smallvec![ArcStr::from("Test")],
        properties: vec![],
    }
}

fn edge_snap(id: u64, src: u64, dst: u64) -> EdgeSnapshot {
    EdgeSnapshot {
        id: obrain_common::types::EdgeId(id),
        src: obrain_common::types::NodeId(src),
        dst: obrain_common::types::NodeId(dst),
        edge_type: ArcStr::from("RELATES_TO"),
        properties: vec![],
    }
}

fn make_store(config: EpisodeConfig) -> Arc<EpisodeStore> {
    Arc::new(EpisodeStore::new(config))
}

fn default_store() -> Arc<EpisodeStore> {
    make_store(EpisodeConfig::default())
}

fn sample_trace(node_id: u64, count: usize) -> Vec<ActivationStep> {
    (0..count)
        .map(|i| ActivationStep {
            node_id: node_id + i as u64,
            activation_level: 1.0 - (i as f64 * 0.1),
            source_node: if i == 0 { None } else { Some(node_id) },
            description: format!("step{i}"),
        })
        .collect()
}

fn record_episode(store: &EpisodeStore, node_ids: Vec<u64>) -> u64 {
    let first = node_ids.first().copied().unwrap_or(0);
    store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: node_ids.clone(),
        },
        sample_trace(first, 2),
        Outcome::with_counts(node_ids.len(), 0, 0, "ok"),
        ValidationResult::ok(),
        node_ids,
        vec![],
    )
}

// ===========================================================================
// 1. EpisodeRecorder::on_event() — all 6 MutationEvent variants
// ===========================================================================

#[tokio::test]
async fn on_event_node_created() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::NodeCreated {
        node: node_snap(10),
    };
    recorder.on_event(&event).await;

    assert_eq!(store.len(), 1);
    let ep = store.list_recent(1).pop().unwrap();
    assert_eq!(ep.process_trace.len(), 1);
    assert!(ep.process_trace[0].description.contains("NodeCreated"));
    assert_eq!(ep.involved_nodes, vec![10]);
}

#[tokio::test]
async fn on_event_node_updated() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::NodeUpdated {
        before: node_snap(20),
        after: node_snap(20),
    };
    recorder.on_event(&event).await;

    let ep = store.list_recent(1).pop().unwrap();
    assert!(ep.process_trace[0].description.contains("NodeUpdated"));
    assert_eq!(ep.involved_nodes, vec![20]);
}

#[tokio::test]
async fn on_event_node_deleted() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::NodeDeleted {
        node: node_snap(30),
    };
    recorder.on_event(&event).await;

    let ep = store.list_recent(1).pop().unwrap();
    assert!(ep.process_trace[0].description.contains("NodeDeleted"));
    assert_eq!(ep.involved_nodes, vec![30]);
}

#[tokio::test]
async fn on_event_edge_created() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::EdgeCreated {
        edge: edge_snap(1, 40, 50),
    };
    recorder.on_event(&event).await;

    let ep = store.list_recent(1).pop().unwrap();
    assert!(ep.process_trace[0].description.contains("EdgeCreated"));
    assert!(ep.involved_nodes.contains(&40));
    assert!(ep.involved_nodes.contains(&50));
}

#[tokio::test]
async fn on_event_edge_updated() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::EdgeUpdated {
        before: edge_snap(2, 60, 70),
        after: edge_snap(2, 60, 70),
    };
    recorder.on_event(&event).await;

    let ep = store.list_recent(1).pop().unwrap();
    assert!(ep.process_trace[0].description.contains("EdgeUpdated"));
    assert!(ep.involved_nodes.contains(&60));
    assert!(ep.involved_nodes.contains(&70));
}

#[tokio::test]
async fn on_event_edge_deleted() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let event = MutationEvent::EdgeDeleted {
        edge: edge_snap(3, 80, 90),
    };
    recorder.on_event(&event).await;

    let ep = store.list_recent(1).pop().unwrap();
    assert!(ep.process_trace[0].description.contains("EdgeDeleted"));
    assert!(ep.involved_nodes.contains(&80));
    assert!(ep.involved_nodes.contains(&90));
}

// ===========================================================================
// 2. EpisodeRecorder::on_batch()
// ===========================================================================

#[tokio::test]
async fn on_batch_aggregates_multiple_events() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeCreated { node: node_snap(1) },
        MutationEvent::EdgeCreated {
            edge: edge_snap(1, 2, 3),
        },
    ];
    recorder.on_batch(&events).await;

    // Should produce exactly one episode (batch aggregation).
    assert_eq!(store.len(), 1);
    let ep = store.list_recent(1).pop().unwrap();

    // Process trace should have one step per event.
    assert_eq!(ep.process_trace.len(), 2);

    // Mutation type should be aggregated with "+".
    match &ep.stimulus {
        Stimulus::Mutation { mutation_type, .. } => {
            assert!(
                mutation_type.contains("NodeCreated"),
                "expected NodeCreated in '{mutation_type}'"
            );
            assert!(
                mutation_type.contains("EdgeCreated"),
                "expected EdgeCreated in '{mutation_type}'"
            );
            assert!(
                mutation_type.contains('+'),
                "expected '+' separator in '{mutation_type}'"
            );
        }
        _ => panic!("expected Mutation stimulus"),
    }
}

#[tokio::test]
async fn on_batch_deduplicates_node_ids() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    // Both events involve node 5.
    let events = vec![
        MutationEvent::NodeCreated { node: node_snap(5) },
        MutationEvent::NodeUpdated {
            before: node_snap(5),
            after: node_snap(5),
        },
    ];
    recorder.on_batch(&events).await;

    let ep = store.list_recent(1).pop().unwrap();
    // Node 5 should appear only once after deduplication.
    assert_eq!(
        ep.involved_nodes.iter().filter(|&&n| n == 5).count(),
        1,
        "node 5 should be deduplicated"
    );
}

#[tokio::test]
async fn on_batch_deduplicates_mutation_types() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let events = vec![
        MutationEvent::NodeCreated { node: node_snap(1) },
        MutationEvent::NodeCreated { node: node_snap(2) },
    ];
    recorder.on_batch(&events).await;

    let ep = store.list_recent(1).pop().unwrap();
    match &ep.stimulus {
        Stimulus::Mutation { mutation_type, .. } => {
            // Should NOT be "NodeCreated+NodeCreated".
            assert_eq!(mutation_type, "NodeCreated");
        }
        _ => panic!("expected Mutation stimulus"),
    }
}

#[tokio::test]
async fn on_batch_empty_is_noop() {
    let store = default_store();
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    recorder.on_batch(&[]).await;
    assert_eq!(store.len(), 0);
}

// ===========================================================================
// 3. EpisodeStore::peek() does NOT increment access_count
// ===========================================================================

#[test]
fn peek_does_not_increment_access_count() {
    let store = default_store();
    let id = record_episode(&store, vec![1]);

    // Peek several times.
    for _ in 0..5 {
        let ep = store.peek(id).unwrap();
        assert_eq!(ep.access_count, 0);
    }

    // Contrast with get.
    store.get(id);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.access_count, 1);
}

#[test]
fn peek_returns_none_for_missing_id() {
    let store = default_store();
    assert!(store.peek(9999).is_none());
}

// ===========================================================================
// 4. process_trace truncation
// ===========================================================================

#[test]
fn process_trace_truncated_when_exceeding_max() {
    let config = EpisodeConfig {
        max_process_trace: 5,
        ..Default::default()
    };
    let store = EpisodeStore::new(config);

    // Provide 20 steps, expect truncation to 5.
    let trace = sample_trace(1, 20);
    assert_eq!(trace.len(), 20);

    let id = store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: vec![1],
        },
        trace,
        Outcome::success("ok"),
        ValidationResult::ok(),
        vec![1],
        vec![],
    );

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.process_trace.len(), 5);
    // Verify it kept the first 5 steps.
    assert_eq!(ep.process_trace[0].description, "step0");
    assert_eq!(ep.process_trace[4].description, "step4");
}

#[test]
fn process_trace_not_truncated_when_within_limit() {
    let config = EpisodeConfig {
        max_process_trace: 10,
        ..Default::default()
    };
    let store = EpisodeStore::new(config);

    let trace = sample_trace(1, 7);
    let id = store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: vec![1],
        },
        trace,
        Outcome::success("ok"),
        ValidationResult::ok(),
        vec![1],
        vec![],
    );

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.process_trace.len(), 7);
}

// ===========================================================================
// 5. EpisodeMemoryManager::sweep() state transitions
// ===========================================================================

#[test]
fn sweep_operational_to_consolidated() {
    // consolidation_age_secs = 0 means immediately eligible.
    let config = EpisodeConfig {
        consolidation_age_secs: 0,
        archive_age_secs: 999_999,
        min_access_for_retain: 3,
        ..Default::default()
    };
    let store = make_store(config);
    let id = record_episode(&store, vec![1]);

    // Must access once so access_count > 0.
    store.get(id);

    let mgr = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = mgr.sweep();

    assert_eq!(result.consolidated, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Consolidated);
}

#[test]
fn sweep_consolidated_to_archived() {
    let config = EpisodeConfig {
        consolidation_age_secs: 0,
        archive_age_secs: 0,
        min_access_for_retain: 999, // access_count will be far below this
        ..Default::default()
    };
    let store = make_store(config);
    let id = record_episode(&store, vec![1]);

    // Access once so it can consolidate first.
    store.get(id);

    let mgr = EpisodeMemoryManager::new(Arc::clone(&store));

    // First sweep: Operational -> Consolidated.
    let r1 = mgr.sweep();
    assert_eq!(r1.consolidated, 1);

    // Second sweep: Consolidated -> Archived (idle >= 0, access_count < 999).
    let r2 = mgr.sweep();
    assert_eq!(r2.archived, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Archived);
}

#[test]
fn sweep_emergency_operational_stays_if_not_accessed() {
    // An Operational episode that has never been accessed (access_count == 0)
    // should NOT be consolidated, even if old enough.
    let config = EpisodeConfig {
        consolidation_age_secs: 0,
        archive_age_secs: 0,
        min_access_for_retain: 1,
        ..Default::default()
    };
    let store = make_store(config);
    let _id = record_episode(&store, vec![1]);

    let mgr = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = mgr.sweep();

    // access_count == 0, so it stays Operational.
    assert_eq!(result.consolidated, 0);
}

#[test]
fn sweep_archived_to_operational_reactivation() {
    let config = EpisodeConfig {
        consolidation_age_secs: 86400, // Large threshold.
        archive_age_secs: 0,
        min_access_for_retain: 100,
        ..Default::default()
    };
    let store = make_store(config);
    let id = record_episode(&store, vec![1]);

    // Manually set to Archived.
    store.set_horizon(id, EpisodeHorizon::Archived);
    assert_eq!(store.peek(id).unwrap().horizon, EpisodeHorizon::Archived);

    // Access the episode (updates last_accessed to now).
    store.get(id);

    let mgr = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = mgr.sweep();

    assert_eq!(result.reactivated, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Operational);
}

#[test]
fn sweep_evicts_excess_archived_episodes() {
    let config = EpisodeConfig {
        max_episodes: 2,
        consolidation_age_secs: 0,
        archive_age_secs: 0,
        min_access_for_retain: 999,
        ..Default::default()
    };
    let store = make_store(config);

    // Record 4 episodes (2 over capacity).
    let mut ids = Vec::new();
    for i in 0..4u64 {
        ids.push(record_episode(&store, vec![i + 100]));
    }

    // Mark all as Archived so they can be evicted.
    for &id in &ids {
        // Access once so consolidation can happen (though we set manually).
        store.set_horizon(id, EpisodeHorizon::Archived);
    }
    assert_eq!(store.len(), 4);

    let mgr = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = mgr.sweep();

    assert!(
        result.evicted >= 2,
        "expected >= 2 evictions, got {}",
        result.evicted
    );
    assert!(
        store.len() <= 2,
        "expected <= 2 episodes, got {}",
        store.len()
    );
}

// ===========================================================================
// 6. evict_oldest() — oldest archived removed at capacity
// ===========================================================================

#[test]
fn evict_oldest_removes_oldest_archived_on_record() {
    let config = EpisodeConfig {
        max_episodes: 3,
        ..Default::default()
    };
    let store = EpisodeStore::new(config);

    // Record 3 episodes to fill up.
    let id1 = record_episode(&store, vec![1]);
    let id2 = record_episode(&store, vec![2]);
    let id3 = record_episode(&store, vec![3]);
    assert_eq!(store.len(), 3);

    // Mark id1 as Archived (oldest).
    store.set_horizon(id1, EpisodeHorizon::Archived);

    // Recording a 4th episode should trigger evict_oldest and remove the archived id1.
    let id4 = record_episode(&store, vec![4]);
    assert_eq!(store.len(), 3);

    // id1 should be gone.
    assert!(store.peek(id1).is_none(), "id1 should have been evicted");
    // Others should remain.
    assert!(store.peek(id2).is_some());
    assert!(store.peek(id3).is_some());
    assert!(store.peek(id4).is_some());
}

#[test]
fn evict_oldest_does_not_remove_non_archived() {
    let config = EpisodeConfig {
        max_episodes: 2,
        ..Default::default()
    };
    let store = EpisodeStore::new(config);

    // Record 2 episodes — all Operational.
    let id1 = record_episode(&store, vec![1]);
    let id2 = record_episode(&store, vec![2]);
    assert_eq!(store.len(), 2);

    // Record a 3rd. Since no episodes are Archived, evict_oldest won't remove any.
    let id3 = record_episode(&store, vec![3]);

    // All 3 remain because evict_oldest only targets Archived episodes.
    assert_eq!(store.len(), 3);
    assert!(store.peek(id1).is_some());
    assert!(store.peek(id2).is_some());
    assert!(store.peek(id3).is_some());
}
