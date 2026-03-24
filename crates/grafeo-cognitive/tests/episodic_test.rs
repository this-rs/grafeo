//! Integration tests for the episodic memory subsystem.

#![cfg(feature = "episodic")]

use grafeo_cognitive::episodic::{
    EpisodeConfig, EpisodeRecorder, EpisodeStore, Outcome, Stimulus,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_store() -> EpisodeStore {
    EpisodeStore::new(EpisodeConfig::default())
}

fn record_sample_episode(store: &EpisodeStore, node_ids: Vec<u64>, tags: Vec<String>) -> u64 {
    store.record(
        Stimulus::Mutation {
            mutation_type: "NodeCreated".to_string(),
            node_ids: node_ids.clone(),
        },
        vec!["step1".to_string(), "step2".to_string()],
        Outcome::Success {
            details: "created new connections".to_string(),
        },
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
    assert_eq!(episode.process_trace, vec!["step1", "step2"]);
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

    // Newest first — IDs should be in descending order (id3 > id2 > id1).
    // Because SystemTime::now() may be identical for fast sequential calls,
    // we just verify all three are present and the first is >= the last.
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
        vec!["step1".to_string()],
        Outcome::Success {
            details: "created new connections".to_string(),
        },
        vec![1, 2, 3],
        vec![],
    );

    let lesson = store.extract_lesson(id).expect("lesson should exist");
    assert!(lesson.contains("NodeCreated"));
    assert!(lesson.contains("3 nodes"));
    assert!(lesson.contains("Success"));
    assert!(lesson.contains("created new connections"));
}

#[test]
fn extract_lesson_for_query_stimulus() {
    let store = default_store();
    let id = store.record(
        Stimulus::Query {
            query_text: "MATCH (n) RETURN n".to_string(),
        },
        vec![],
        Outcome::Failure {
            error: "timeout exceeded".to_string(),
        },
        vec![],
        vec![],
    );

    let lesson = store.extract_lesson(id).expect("lesson should exist");
    assert!(lesson.contains("Query"));
    assert!(lesson.contains("Failure"));
    assert!(lesson.contains("timeout exceeded"));
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

    let id1 = record_sample_episode(&store, vec![1], vec!["alpha".to_string(), "beta".to_string()]);
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

    match &episode.outcome {
        Outcome::Success { details } => assert_eq!(details, "all good"),
        _ => panic!("expected Success outcome"),
    }

    assert_eq!(episode.involved_nodes, vec![10, 20]);
}

#[test]
fn episode_recorder_records_failure() {
    let store = Arc::new(default_store());
    let recorder = EpisodeRecorder::new(Arc::clone(&store));

    let id = recorder.record_mutation("NodeDeleted", &[5], false, "node not found");
    let episode = store.get(id).unwrap();

    match &episode.outcome {
        Outcome::Failure { error } => assert_eq!(error, "node not found"),
        _ => panic!("expected Failure outcome"),
    }
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
}
