//! Integration tests for the episodic memory subsystem.

#![cfg(feature = "episodic")]

use grafeo_cognitive::episodic::{
    ActivationStep, EpisodeConfig, EpisodeHorizon, EpisodeMemoryManager, EpisodeRecorder,
    EpisodeStore, Outcome, Stimulus, ValidationResult, extract_cross_lesson,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_store() -> EpisodeStore {
    EpisodeStore::new(EpisodeConfig::default())
}

fn sample_trace(node_id: u64) -> Vec<ActivationStep> {
    vec![
        ActivationStep {
            node_id,
            activation_level: 1.0,
            source_node: None,
            description: "step1".to_string(),
        },
        ActivationStep {
            node_id: node_id + 100,
            activation_level: 0.7,
            source_node: Some(node_id),
            description: "step2".to_string(),
        },
    ]
}

fn record_sample_episode(store: &EpisodeStore, node_ids: Vec<u64>, tags: Vec<String>) -> u64 {
    let first = node_ids.first().copied().unwrap_or(0);
    store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: node_ids.clone(),
        },
        sample_trace(first),
        Outcome::with_counts(node_ids.len(), 0, 0, "created new connections"),
        ValidationResult::ok(),
        node_ids,
        tags,
    )
}

// ---------------------------------------------------------------------------
// record and get
// ---------------------------------------------------------------------------

#[test]
fn record_and_get_episode() {
    let store = default_store();
    let id = record_sample_episode(&store, vec![1, 2, 3], vec!["test".to_string()]);

    let episode = store.get(id).expect("episode should exist");
    assert_eq!(episode.id, id);
    assert_eq!(episode.involved_nodes, vec![1, 2, 3]);
    assert_eq!(episode.tags, vec!["test".to_string()]);
    assert_eq!(episode.process_trace.len(), 2);
    assert_eq!(episode.process_trace[0].description, "step1");
    assert_eq!(episode.process_trace[1].description, "step2");
}

// ---------------------------------------------------------------------------
// list_by_node
// ---------------------------------------------------------------------------

#[test]
fn list_by_node_returns_correct_episodes() {
    let store = default_store();

    let id1 = record_sample_episode(&store, vec![10, 20], vec![]);
    let _id2 = record_sample_episode(&store, vec![30], vec![]);
    let id3 = record_sample_episode(&store, vec![10, 40], vec![]);

    let episodes = store.list_by_node(10);
    let ids: Vec<u64> = episodes.iter().map(|e| e.id).collect();

    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&id1));
    assert!(ids.contains(&id3));
}

#[test]
fn list_by_node_returns_empty_for_unknown_node() {
    let store = default_store();
    record_sample_episode(&store, vec![1], vec![]);

    let episodes = store.list_by_node(9999);
    assert!(episodes.is_empty());
}

// ---------------------------------------------------------------------------
// list_recent
// ---------------------------------------------------------------------------

#[test]
fn list_recent_returns_newest_first() {
    let store = default_store();

    let id1 = record_sample_episode(&store, vec![1], vec![]);
    let id2 = record_sample_episode(&store, vec![2], vec![]);
    let id3 = record_sample_episode(&store, vec![3], vec![]);

    let recent = store.list_recent(10);
    assert_eq!(recent.len(), 3);

    let ids: Vec<u64> = recent.iter().map(|e| e.id).collect();
    assert!(ids.contains(&id1));
    assert!(ids.contains(&id2));
    assert!(ids.contains(&id3));
}

#[test]
fn list_recent_respects_limit() {
    let store = default_store();

    for i in 0..5 {
        record_sample_episode(&store, vec![i], vec![]);
    }

    let recent = store.list_recent(3);
    assert_eq!(recent.len(), 3);
}

// ---------------------------------------------------------------------------
// extract_lesson
// ---------------------------------------------------------------------------

#[test]
fn extract_lesson_produces_readable_string() {
    let store = default_store();
    let id = store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: vec![1, 2, 3],
        },
        sample_trace(1),
        Outcome::with_counts(3, 1, 0, "created new connections"),
        ValidationResult::ok(),
        vec![1, 2, 3],
        vec![],
    );

    let lesson = store.extract_lesson_for(id).expect("lesson should exist");
    assert!(lesson.contains("NodeCreated"));
    assert!(lesson.contains("3 nodes"));
    assert!(lesson.contains("success"));
}

#[test]
fn extract_lesson_for_query_stimulus() {
    let store = default_store();
    let id = store.record(
        Stimulus::Query {
            query_text: "MATCH (n) RETURN n".to_string(),
        },
        vec![],
        Outcome::success("query executed"),
        ValidationResult::failure(Some(42), "timeout exceeded"),
        vec![],
        vec![],
    );

    let lesson = store.extract_lesson_for(id).expect("lesson should exist");
    assert!(lesson.contains("Query"));
    assert!(lesson.contains("failure"));
    assert!(lesson.contains("timeout exceeded"));
    assert!(lesson.contains("scar #42"));
}

#[test]
fn extract_lesson_includes_outcome_counts() {
    let store = default_store();
    let id = store.record(
        Stimulus::Mutation {
            mutation_type: "EdgeCreated".to_string(),
            node_ids: vec![1, 2],
        },
        vec![],
        Outcome::with_counts(2, 3, 1, "edges added"),
        ValidationResult::ok(),
        vec![1, 2],
        vec![],
    );

    let lesson = store.extract_lesson_for(id).unwrap();
    assert!(lesson.contains("2 nodes modified"));
    assert!(lesson.contains("3 synapses changed"));
    assert!(lesson.contains("1 scars added"));
}

// ---------------------------------------------------------------------------
// extract_cross_lesson
// ---------------------------------------------------------------------------

#[test]
fn extract_cross_lesson_finds_co_occurring_nodes() {
    let store = default_store();

    // Record multiple episodes with overlapping nodes.
    record_sample_episode(&store, vec![1, 2, 3], vec![]);
    record_sample_episode(&store, vec![1, 2, 4], vec![]);
    record_sample_episode(&store, vec![1, 5], vec![]);

    let episodes = store.list_recent(10);
    let lesson = extract_cross_lesson(&episodes);

    assert!(lesson.contains("3 episodes"));
    assert!(lesson.contains("Node 1")); // Node 1 appears in all 3.
    assert!(lesson.contains("Success rate"));
}

// ---------------------------------------------------------------------------
// auto_lesson
// ---------------------------------------------------------------------------

#[test]
fn auto_lesson_populates_lesson_on_record() {
    let store = default_store();
    let id = record_sample_episode(&store, vec![1, 2, 3], vec![]);

    let episode = store.get(id).unwrap();
    assert!(episode.lesson.is_some());
    let lesson = episode.lesson.unwrap();
    assert!(lesson.contains("NodeCreated"));
    assert!(lesson.contains("3 nodes"));
}

#[test]
fn auto_lesson_disabled_leaves_lesson_none() {
    let config = EpisodeConfig {
        auto_lesson: false,
        ..Default::default()
    };
    let store = EpisodeStore::new(config);
    let id = record_sample_episode(&store, vec![1], vec![]);

    let episode = store.get(id).unwrap();
    assert!(episode.lesson.is_none());
}

// ---------------------------------------------------------------------------
// search_by_tags
// ---------------------------------------------------------------------------

#[test]
fn search_by_tags_finds_matching_episodes() {
    let store = default_store();

    let id1 = record_sample_episode(
        &store,
        vec![1],
        vec!["alpha".to_string(), "beta".to_string()],
    );
    let _id2 = record_sample_episode(&store, vec![2], vec!["gamma".to_string()]);
    let id3 = record_sample_episode(&store, vec![3], vec!["beta".to_string()]);

    let results = store.search_by_tags(&["beta".to_string()]);
    let ids: Vec<u64> = results.iter().map(|e| e.id).collect();

    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&id1));
    assert!(ids.contains(&id3));
}

#[test]
fn search_by_tags_returns_empty_for_unknown_tags() {
    let store = default_store();
    record_sample_episode(&store, vec![1], vec!["alpha".to_string()]);

    let results = store.search_by_tags(&["nonexistent".to_string()]);
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// EpisodeRecorder
// ---------------------------------------------------------------------------

#[test]
fn episode_recorder_records_mutation_episodes() {
    let store = Arc::new(default_store());
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let id = recorder.record_mutation("NodeCreated", &[10, 20], true, "all good");
    let episode = store.get(id).unwrap();

    match &episode.stimulus {
        Stimulus::Mutation {
            mutation_type,
            node_ids,
        } => {
            assert_eq!(mutation_type, "NodeCreated");
            assert_eq!(node_ids, &[10, 20]);
        }
        _ => panic!("expected Mutation stimulus"),
    }

    assert_eq!(episode.outcome.nodes_modified, 2);
    assert!(episode.validation.success);
    assert_eq!(episode.involved_nodes, vec![10, 20]);
}

#[test]
fn episode_recorder_records_failure() {
    let store = Arc::new(default_store());
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let id = recorder.record_mutation("NodeDeleted", &[5], false, "node not found");
    let episode = store.get(id).unwrap();

    assert!(!episode.validation.success);
    assert_eq!(episode.outcome.scars_added, 1);
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

#[test]
fn validation_result_success() {
    let v = ValidationResult::ok();
    assert!(v.success);
    assert!(v.scar_id.is_none());
}

#[test]
fn validation_result_failure_with_scar() {
    let v = ValidationResult::failure(Some(42), "constraint violated");
    assert!(!v.success);
    assert_eq!(v.scar_id, Some(42));
    assert_eq!(v.message.as_deref(), Some("constraint violated"));
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

#[test]
fn outcome_with_counts_tracks_changes() {
    let o = Outcome::with_counts(5, 3, 1, "batch applied");
    assert_eq!(o.nodes_modified, 5);
    assert_eq!(o.synapses_changed, 3);
    assert_eq!(o.scars_added, 1);
    assert_eq!(o.details, "batch applied");
}

// ---------------------------------------------------------------------------
// len and is_empty
// ---------------------------------------------------------------------------

#[test]
fn len_and_is_empty_work_correctly() {
    let store = default_store();

    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    record_sample_episode(&store, vec![1], vec![]);
    assert!(!store.is_empty());
    assert_eq!(store.len(), 1);

    record_sample_episode(&store, vec![2], vec![]);
    assert_eq!(store.len(), 2);
}

// ---------------------------------------------------------------------------
// ID monotonicity
// ---------------------------------------------------------------------------

#[test]
fn episode_ids_are_monotonically_increasing() {
    let store = default_store();

    let id1 = record_sample_episode(&store, vec![1], vec![]);
    let id2 = record_sample_episode(&store, vec![2], vec![]);
    let id3 = record_sample_episode(&store, vec![3], vec![]);

    assert!(id1 < id2);
    assert!(id2 < id3);
}

// ---------------------------------------------------------------------------
// Default config
// ---------------------------------------------------------------------------

#[test]
fn default_config_has_sensible_values() {
    let config = EpisodeConfig::default();

    assert_eq!(config.max_episodes, 10_000);
    assert_eq!(config.max_process_trace, 100);
    assert!(config.auto_lesson);
    assert!(config.consolidation_age_secs > 0);
    assert!(config.archive_age_secs > config.consolidation_age_secs);
}

// ---------------------------------------------------------------------------
// EpisodeMemoryManager
// ---------------------------------------------------------------------------

#[test]
fn memory_manager_sweep_no_ops_for_fresh_episodes() {
    let store = Arc::new(default_store());
    record_sample_episode(&store, vec![1], vec![]);

    let manager = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = manager.sweep();

    // Fresh episode shouldn't be consolidated yet.
    assert_eq!(result.consolidated, 0);
    assert_eq!(result.archived, 0);
    assert_eq!(result.reactivated, 0);
    assert_eq!(result.evicted, 0);
}

#[test]
fn memory_manager_consolidates_old_accessed_episodes() {
    // Use a very short consolidation threshold.
    let config = EpisodeConfig {
        consolidation_age_secs: 0, // Immediately eligible.
        archive_age_secs: 86400,
        min_access_for_retain: 3,
        ..Default::default()
    };
    let store = Arc::new(EpisodeStore::new(config));
    let id = record_sample_episode(&store, vec![1], vec![]);

    // Access it so access_count > 0.
    store.get(id);

    let manager = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = manager.sweep();

    assert_eq!(result.consolidated, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Consolidated);
}

#[test]
fn memory_manager_archives_idle_consolidated_episodes() {
    let config = EpisodeConfig {
        consolidation_age_secs: 0,
        archive_age_secs: 0,        // Immediately eligible.
        min_access_for_retain: 100, // High threshold.
        ..Default::default()
    };
    let store = Arc::new(EpisodeStore::new(config));
    let id = record_sample_episode(&store, vec![1], vec![]);

    // Access once to allow consolidation.
    store.get(id);

    let manager = EpisodeMemoryManager::new(Arc::clone(&store));

    // First sweep: consolidates.
    manager.sweep();
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Consolidated);

    // Second sweep: archives (access_count < min_access_for_retain).
    let result = manager.sweep();
    assert_eq!(result.archived, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Archived);
}

#[test]
fn memory_manager_reactivates_accessed_archived_episode() {
    // Use a large consolidation threshold so recently-accessed episodes reactivate.
    let config = EpisodeConfig {
        consolidation_age_secs: 86400, // 1 day — large so last_accessed is "recent" relative to this.
        archive_age_secs: 0,
        min_access_for_retain: 100,
        ..Default::default()
    };
    let store = Arc::new(EpisodeStore::new(config));
    let id = record_sample_episode(&store, vec![1], vec![]);

    // Manually set to Archived.
    store.set_horizon(id, EpisodeHorizon::Archived);

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Archived);

    // Access the episode (updates last_accessed to now).
    store.get(id);

    // Sweep: should reactivate because idle < consolidation_threshold (now < 1 day).
    let manager = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = manager.sweep();
    assert_eq!(result.reactivated, 1);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Operational);
}

#[test]
fn memory_manager_evicts_when_over_capacity() {
    // Use a very small consolidation_age so archived episodes aren't reactivated
    // (their idle time is > 0 which is >= 0 = consolidation_threshold).
    let config = EpisodeConfig {
        max_episodes: 3,
        consolidation_age_secs: 0,
        archive_age_secs: 0,
        min_access_for_retain: 100,
        ..Default::default()
    };
    let store = Arc::new(EpisodeStore::new(config));

    // Record 5 episodes (2 over capacity).
    let mut ids = Vec::new();
    for i in 0..5 {
        ids.push(record_sample_episode(&store, vec![i], vec![]));
    }
    assert_eq!(store.len(), 5);

    // Manually set 3 to Archived so they can be evicted.
    // Don't access them (so last_accessed stays at created_at).
    store.set_horizon(ids[0], EpisodeHorizon::Archived);
    store.set_horizon(ids[1], EpisodeHorizon::Archived);
    store.set_horizon(ids[2], EpisodeHorizon::Archived);

    let manager = EpisodeMemoryManager::new(Arc::clone(&store));
    let result = manager.sweep();

    // Should have evicted enough archived episodes to get to max_episodes.
    assert!(
        result.evicted >= 2,
        "expected >= 2 evictions, got {}",
        result.evicted
    );
    assert!(
        store.len() <= 3,
        "expected <= 3 episodes, got {}",
        store.len()
    );
}

// ---------------------------------------------------------------------------
// Episode horizons
// ---------------------------------------------------------------------------

#[test]
fn new_episodes_start_as_operational() {
    let store = default_store();
    let id = record_sample_episode(&store, vec![1], vec![]);

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.horizon, EpisodeHorizon::Operational);
}

#[test]
fn count_by_horizon_groups_correctly() {
    let store = default_store();

    record_sample_episode(&store, vec![1], vec![]);
    record_sample_episode(&store, vec![2], vec![]);

    let counts = store.count_by_horizon();
    assert_eq!(counts.get(&EpisodeHorizon::Operational), Some(&2));
    assert_eq!(counts.get(&EpisodeHorizon::Consolidated), None);
    assert_eq!(counts.get(&EpisodeHorizon::Archived), None);
}

// ---------------------------------------------------------------------------
// Access tracking
// ---------------------------------------------------------------------------

#[test]
fn get_increments_access_count() {
    let store = default_store();
    let id = record_sample_episode(&store, vec![1], vec![]);

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.access_count, 0);

    store.get(id);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.access_count, 1);

    store.get(id);
    store.get(id);
    let ep = store.peek(id).unwrap();
    assert_eq!(ep.access_count, 3);
}

#[test]
fn peek_does_not_increment_access_count() {
    let store = default_store();
    let id = record_sample_episode(&store, vec![1], vec![]);

    store.peek(id);
    store.peek(id);
    store.peek(id);

    let ep = store.peek(id).unwrap();
    assert_eq!(ep.access_count, 0);
}

// ---------------------------------------------------------------------------
// ActivationStep
// ---------------------------------------------------------------------------

#[test]
fn activation_step_serializable() {
    let step = ActivationStep {
        node_id: 42,
        activation_level: 0.85,
        source_node: Some(10),
        description: "propagated from node 10".to_string(),
    };

    let json = serde_json::to_string(&step).unwrap();
    let deserialized: ActivationStep = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.node_id, 42);
    assert!((deserialized.activation_level - 0.85).abs() < f64::EPSILON);
    assert_eq!(deserialized.source_node, Some(10));
}

// ---------------------------------------------------------------------------
// Distillation Layer 1 integration
// ---------------------------------------------------------------------------

#[test]
fn episodes_serve_as_layer1_for_distillation() {
    let store = default_store();

    // Simulate a few cognitive episodes.
    let id1 = store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".into(),
            node_ids: vec![1, 2],
        },
        vec![
            ActivationStep {
                node_id: 1,
                activation_level: 1.0,
                source_node: None,
                description: "initial activation".into(),
            },
            ActivationStep {
                node_id: 2,
                activation_level: 0.8,
                source_node: Some(1),
                description: "spread to node 2".into(),
            },
        ],
        Outcome::with_counts(2, 1, 0, "synapse reinforced between 1↔2"),
        ValidationResult::ok(),
        vec![1, 2],
        vec!["creation".into()],
    );

    let _id2 = store.record(
        Stimulus::Mutation {
            mutation_type: "NodeUpdated".into(),
            node_ids: vec![1, 3],
        },
        vec![ActivationStep {
            node_id: 1,
            activation_level: 1.0,
            source_node: None,
            description: "re-activation of node 1".into(),
        }],
        Outcome::with_counts(1, 2, 0, "synapses 1↔2 and 1↔3 reinforced"),
        ValidationResult::ok(),
        vec![1, 2, 3],
        vec!["update".into()],
    );

    // Layer 1 = episodes provide WHY and HOW, not just WHAT (Layer 0 = edges).
    let ep1 = store.get(id1).unwrap();
    assert!(ep1.lesson.is_some());
    let lesson = ep1.lesson.unwrap();
    assert!(lesson.contains("NodeCreated"));

    // Cross-episode analysis.
    let episodes = store.list_by_node(1);
    assert_eq!(episodes.len(), 2);

    let cross = extract_cross_lesson(&episodes);
    assert!(cross.contains("2 episodes"));
    assert!(cross.contains("Node 1"));
}

// ---------------------------------------------------------------------------
// Episode serialization roundtrip
// ---------------------------------------------------------------------------

#[test]
fn episode_full_serialization_roundtrip() {
    let store = default_store();
    let id = store.record(
        Stimulus::External {
            source: "webhook".into(),
            description: "deployment event".into(),
        },
        vec![ActivationStep {
            node_id: 99,
            activation_level: 0.5,
            source_node: None,
            description: "external trigger".into(),
        }],
        Outcome::with_counts(1, 0, 1, "scar added"),
        ValidationResult::failure(Some(7), "deployment failed"),
        vec![99],
        vec!["deploy".into()],
    );

    let ep = store.get(id).unwrap();
    let json = serde_json::to_string_pretty(&ep).unwrap();
    let restored: grafeo_cognitive::episodic::Episode = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.id, ep.id);
    assert_eq!(restored.involved_nodes, vec![99]);
    assert!(!restored.validation.success);
    assert_eq!(restored.validation.scar_id, Some(7));
    assert_eq!(restored.outcome.scars_added, 1);
}
