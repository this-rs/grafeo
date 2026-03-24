//! Episodic Memory — captures Stimulus -> Process -> Outcome -> Lesson episodes.
//!
//! The episodic memory module records operational episodes from graph mutations,
//! enabling the system to learn from its history. Each episode captures what
//! triggered it (stimulus), what steps were taken (process trace), what happened
//! (outcome), and what can be learned (lesson).

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------

/// What triggered the episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stimulus {
    /// A graph mutation occurred.
    Mutation {
        /// The type of mutation (e.g. "NodeCreated", "EdgeDeleted").
        mutation_type: String,
        /// The node IDs involved in the mutation.
        node_ids: Vec<u64>,
    },
    /// A query was executed.
    Query {
        /// The query text that was executed.
        query_text: String,
    },
    /// An external event.
    External {
        /// The source of the external event.
        source: String,
        /// A human-readable description of the event.
        description: String,
    },
}

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// The result of an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    /// The operation succeeded.
    Success {
        /// A description of the successful outcome.
        details: String,
    },
    /// The operation failed.
    Failure {
        /// The error message.
        error: String,
    },
    /// The operation partially succeeded.
    Partial {
        /// A description of what succeeded.
        details: String,
        /// A list of issues encountered.
        issues: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// Episode
// ---------------------------------------------------------------------------

/// A single episode in the episodic memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique episode identifier.
    pub id: u64,
    /// What triggered this episode.
    pub stimulus: Stimulus,
    /// Ordered steps taken during the episode.
    pub process_trace: Vec<String>,
    /// The outcome of the episode.
    pub outcome: Outcome,
    /// Extracted lesson (human/LLM-readable).
    pub lesson: Option<String>,
    /// When the episode was created.
    pub created_at: SystemTime,
    /// Node IDs involved in the episode.
    pub involved_nodes: Vec<u64>,
    /// Searchable tags.
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// EpisodeConfig
// ---------------------------------------------------------------------------

/// Configuration for the episodic memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeConfig {
    /// Maximum number of stored episodes.
    pub max_episodes: usize,
    /// Maximum steps in a process trace.
    pub max_process_trace: usize,
    /// Whether to automatically extract lessons on record.
    pub auto_lesson: bool,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            max_episodes: 10_000,
            max_process_trace: 100,
            auto_lesson: true,
        }
    }
}

// ---------------------------------------------------------------------------
// EpisodeStore
// ---------------------------------------------------------------------------

/// Thread-safe store for episodic memory.
pub struct EpisodeStore {
    episodes: DashMap<u64, Episode>,
    by_node: DashMap<u64, Vec<u64>>,
    next_id: AtomicU64,
    config: EpisodeConfig,
}

impl EpisodeStore {
    /// Creates a new episode store with the given configuration.
    pub fn new(config: EpisodeConfig) -> Self {
        Self {
            episodes: DashMap::new(),
            by_node: DashMap::new(),
            next_id: AtomicU64::new(1),
            config,
        }
    }

    /// Records a new episode and returns its ID.
    pub fn record(
        &self,
        stimulus: Stimulus,
        process_trace: Vec<String>,
        outcome: Outcome,
        involved_nodes: Vec<u64>,
        tags: Vec<String>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Truncate process trace if needed.
        let process_trace = if process_trace.len() > self.config.max_process_trace {
            process_trace[..self.config.max_process_trace].to_vec()
        } else {
            process_trace
        };

        let mut episode = Episode {
            id,
            stimulus,
            process_trace,
            outcome,
            lesson: None,
            created_at: SystemTime::now(),
            involved_nodes: involved_nodes.clone(),
            tags,
        };

        // Auto-extract lesson if configured.
        if self.config.auto_lesson {
            episode.lesson = Some(Self::build_lesson(&episode));
        }

        // Update by_node index.
        for &node_id in &involved_nodes {
            self.by_node.entry(node_id).or_default().push(id);
        }

        self.episodes.insert(id, episode);

        id
    }

    /// Gets an episode by ID.
    pub fn get(&self, id: u64) -> Option<Episode> {
        self.episodes.get(&id).map(|e| e.clone())
    }

    /// Lists episodes involving a specific node, sorted by creation time (newest first).
    pub fn list_by_node(&self, node_id: u64) -> Vec<Episode> {
        let episode_ids = match self.by_node.get(&node_id) {
            Some(ids) => ids.clone(),
            None => return Vec::new(),
        };

        let mut episodes: Vec<Episode> = episode_ids
            .iter()
            .filter_map(|id| self.episodes.get(id).map(|e| e.clone()))
            .collect();

        episodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        episodes
    }

    /// Lists the N most recent episodes.
    pub fn list_recent(&self, limit: usize) -> Vec<Episode> {
        let mut episodes: Vec<Episode> = self.episodes.iter().map(|e| e.value().clone()).collect();
        episodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        episodes.truncate(limit);
        episodes
    }

    /// Returns the total number of episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Returns true if there are no episodes.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Extracts a lesson from an episode based on patterns.
    pub fn extract_lesson(&self, episode_id: u64) -> Option<String> {
        self.episodes
            .get(&episode_id)
            .map(|e| Self::build_lesson(&e))
    }

    /// Builds a human-readable lesson string from an episode.
    fn build_lesson(episode: &Episode) -> String {
        let stimulus_desc = match &episode.stimulus {
            Stimulus::Mutation {
                mutation_type,
                node_ids,
            } => {
                format!("{mutation_type} on {} nodes", node_ids.len())
            }
            Stimulus::Query { query_text } => format!("Query \"{}\"", query_text),
            Stimulus::External {
                source,
                description: _,
            } => {
                format!("External event from {source}")
            }
        };

        let outcome_desc = match &episode.outcome {
            Outcome::Success { details } => format!("Success: {details}"),
            Outcome::Failure { error } => format!("Failure: {error}"),
            Outcome::Partial { details, issues } => {
                format!("Partial success: {details} ({} issues)", issues.len())
            }
        };

        format!("{stimulus_desc} resulted in {outcome_desc}")
    }

    /// Returns episodes matching any of the given tags.
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<Episode> {
        let mut episodes: Vec<Episode> = self
            .episodes
            .iter()
            .filter(|entry| {
                let ep = entry.value();
                ep.tags.iter().any(|t| tags.contains(t))
            })
            .map(|entry| entry.value().clone())
            .collect();

        episodes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        episodes
    }
}

// ---------------------------------------------------------------------------
// EpisodeRecorder
// ---------------------------------------------------------------------------

/// Simple mutation recorder that wraps an `EpisodeStore`.
pub struct EpisodeRecorder {
    store: Arc<EpisodeStore>,
}

impl EpisodeRecorder {
    /// Creates a new episode recorder.
    pub fn new(store: Arc<EpisodeStore>) -> Self {
        Self { store }
    }

    /// Records a mutation event as an episode.
    pub fn record_mutation(
        &self,
        mutation_type: &str,
        node_ids: &[u64],
        success: bool,
        details: &str,
    ) -> u64 {
        let stimulus = Stimulus::Mutation {
            mutation_type: mutation_type.to_string(),
            node_ids: node_ids.to_vec(),
        };

        let process_trace = vec![format!("Mutation: {mutation_type}")];

        let outcome = if success {
            Outcome::Success {
                details: details.to_string(),
            }
        } else {
            Outcome::Failure {
                error: details.to_string(),
            }
        };

        let involved_nodes = node_ids.to_vec();
        let tags = vec![mutation_type.to_string()];

        self.store
            .record(stimulus, process_trace, outcome, involved_nodes, tags)
    }
}
