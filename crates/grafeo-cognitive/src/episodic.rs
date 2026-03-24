//! Episodic Memory — captures Stimulus → Process → Outcome → Validation → Lesson episodes.
//!
//! The episodic memory module records operational episodes from graph mutations,
//! enabling the system to learn from its history. Each episode captures:
//!
//! - **Stimulus**: what triggered the episode (mutation, query, external event)
//! - **Process**: what activation steps were taken (spreading activation trace)
//! - **Outcome**: what changed (nodes modified, synapses changed, scars added)
//! - **Validation**: success/failure + optional scar reference
//! - **Lesson**: an extracted human/LLM-readable pattern
//!
//! Episodes are linked to the nodes they involve and serve as Layer 1 for
//! distillation (vs Layer 0 = raw edges/synapses).

use async_trait::async_trait;
use dashmap::DashMap;
use grafeo_reactive::{MutationEvent, MutationListener};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// ActivationStep
// ---------------------------------------------------------------------------

/// A single step in the spreading activation trace of an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStep {
    /// The node that was activated.
    pub node_id: u64,
    /// The activation level at this node.
    pub activation_level: f64,
    /// Where the activation came from (None for the source node).
    pub source_node: Option<u64>,
    /// A human-readable description of this step.
    pub description: String,
}

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

/// The structural outcome of an episode — what changed in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Number of nodes that were modified.
    pub nodes_modified: usize,
    /// Number of synapses that changed (created, reinforced, or decayed).
    pub synapses_changed: usize,
    /// Number of scars added by this episode.
    pub scars_added: usize,
    /// A human-readable summary of the outcome.
    pub details: String,
}

impl Outcome {
    /// Creates a simple success outcome.
    pub fn success(details: impl Into<String>) -> Self {
        Self {
            nodes_modified: 0,
            synapses_changed: 0,
            scars_added: 0,
            details: details.into(),
        }
    }

    /// Creates an outcome with specific counts.
    pub fn with_counts(
        nodes_modified: usize,
        synapses_changed: usize,
        scars_added: usize,
        details: impl Into<String>,
    ) -> Self {
        Self {
            nodes_modified,
            synapses_changed,
            scars_added,
            details: details.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationResult
// ---------------------------------------------------------------------------

/// Validation of the episode outcome — did it succeed or fail?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the episode completed successfully.
    pub success: bool,
    /// Optional scar ID if a scar was created due to failure.
    pub scar_id: Option<u64>,
    /// Optional message describing the validation.
    pub message: Option<String>,
}

impl ValidationResult {
    /// Creates a successful validation result.
    pub fn ok() -> Self {
        Self {
            success: true,
            scar_id: None,
            message: None,
        }
    }

    /// Creates a failure validation result with an optional scar reference.
    pub fn failure(scar_id: Option<u64>, message: impl Into<String>) -> Self {
        Self {
            success: false,
            scar_id,
            message: Some(message.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// EpisodeHorizon
// ---------------------------------------------------------------------------

/// The memory horizon for an episode (mirrors node MemoryHorizon).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpisodeHorizon {
    /// Hot: recently created, frequently accessed.
    Operational,
    /// Warm: confirmed patterns, consolidated knowledge.
    Consolidated,
    /// Cold: old, rarely accessed, candidate for eviction.
    Archived,
}

// ---------------------------------------------------------------------------
// Episode
// ---------------------------------------------------------------------------

/// A single episode in the episodic memory, capturing the full cognitive cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique episode identifier.
    pub id: u64,
    /// What triggered this episode.
    pub stimulus: Stimulus,
    /// Ordered activation steps taken during the episode.
    pub process_trace: Vec<ActivationStep>,
    /// What changed as a result.
    pub outcome: Outcome,
    /// Validation: success/failure + optional scar reference.
    pub validation: ValidationResult,
    /// Extracted lesson (human/LLM-readable).
    pub lesson: Option<String>,
    /// When the episode was created.
    pub created_at: SystemTime,
    /// Node IDs involved in the episode.
    pub involved_nodes: Vec<u64>,
    /// Searchable tags.
    pub tags: Vec<String>,
    /// Current memory horizon (for consolidation/archival).
    pub horizon: EpisodeHorizon,
    /// Number of times this episode has been accessed/consulted.
    pub access_count: u64,
    /// Last time this episode was accessed.
    pub last_accessed: SystemTime,
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
    /// Duration after which idle episodes are consolidated (seconds).
    pub consolidation_age_secs: u64,
    /// Duration after which consolidated episodes are archived (seconds).
    pub archive_age_secs: u64,
    /// Minimum access count to prevent archival.
    pub min_access_for_retain: u64,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            max_episodes: 10_000,
            max_process_trace: 100,
            auto_lesson: true,
            consolidation_age_secs: 3600,    // 1 hour
            archive_age_secs: 7 * 24 * 3600, // 7 days
            min_access_for_retain: 3,
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

    /// Returns a reference to the store's configuration.
    pub fn config(&self) -> &EpisodeConfig {
        &self.config
    }

    /// Records a new episode and returns its ID.
    pub fn record(
        &self,
        stimulus: Stimulus,
        process_trace: Vec<ActivationStep>,
        outcome: Outcome,
        validation: ValidationResult,
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

        let now = SystemTime::now();
        let mut episode = Episode {
            id,
            stimulus,
            process_trace,
            outcome,
            validation,
            lesson: None,
            created_at: now,
            involved_nodes: involved_nodes.clone(),
            tags,
            horizon: EpisodeHorizon::Operational,
            access_count: 0,
            last_accessed: now,
        };

        // Auto-extract lesson if configured.
        if self.config.auto_lesson {
            episode.lesson = Some(extract_lesson(&episode));
        }

        // Update by_node index.
        for &node_id in &involved_nodes {
            self.by_node.entry(node_id).or_default().push(id);
        }

        self.episodes.insert(id, episode);

        // Evict oldest if over capacity.
        if self.episodes.len() > self.config.max_episodes {
            self.evict_oldest();
        }

        id
    }

    /// Gets an episode by ID, incrementing its access count.
    pub fn get(&self, id: u64) -> Option<Episode> {
        if let Some(mut entry) = self.episodes.get_mut(&id) {
            entry.access_count += 1;
            entry.last_accessed = SystemTime::now();
            Some(entry.clone())
        } else {
            None
        }
    }

    /// Gets an episode by ID without incrementing access count.
    pub fn peek(&self, id: u64) -> Option<Episode> {
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
    pub fn extract_lesson_for(&self, episode_id: u64) -> Option<String> {
        self.episodes.get(&episode_id).map(|e| extract_lesson(&e))
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

    /// Returns episodes grouped by horizon.
    pub fn count_by_horizon(&self) -> HashMap<EpisodeHorizon, usize> {
        let mut counts = HashMap::new();
        for entry in &self.episodes {
            *counts.entry(entry.value().horizon).or_insert(0) += 1;
        }
        counts
    }

    /// Sets the horizon of a specific episode (for testing / manual override).
    pub fn set_horizon(&self, id: u64, horizon: EpisodeHorizon) {
        if let Some(mut entry) = self.episodes.get_mut(&id) {
            entry.horizon = horizon;
        }
    }

    /// Returns all episode IDs.
    fn all_ids(&self) -> Vec<u64> {
        self.episodes.iter().map(|e| *e.key()).collect()
    }

    /// Evicts the oldest archived episode when over capacity.
    fn evict_oldest(&self) {
        // Find the oldest archived episode.
        let oldest = self
            .episodes
            .iter()
            .filter(|e| e.value().horizon == EpisodeHorizon::Archived)
            .min_by_key(|e| e.value().created_at)
            .map(|e| *e.key());

        if let Some(id) = oldest {
            self.remove(id);
        }
    }

    /// Removes an episode by ID.
    fn remove(&self, id: u64) {
        if let Some((_, episode)) = self.episodes.remove(&id) {
            for node_id in &episode.involved_nodes {
                if let Some(mut ids) = self.by_node.get_mut(node_id) {
                    ids.retain(|&eid| eid != id);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// extract_lesson (free function)
// ---------------------------------------------------------------------------

/// Extracts a human/LLM-readable lesson from an episode.
///
/// The lesson summarizes the observed pattern: what was the stimulus,
/// which nodes were involved, what outcome occurred, and any validation info.
pub fn extract_lesson(episode: &Episode) -> String {
    let stimulus_desc = match &episode.stimulus {
        Stimulus::Mutation {
            mutation_type,
            node_ids,
        } => {
            format!("{mutation_type} on {} nodes", node_ids.len())
        }
        Stimulus::Query { query_text } => {
            let truncated = if query_text.len() > 50 {
                format!("{}...", &query_text[..50])
            } else {
                query_text.clone()
            };
            format!("Query \"{truncated}\"")
        }
        Stimulus::External {
            source,
            description: _,
        } => {
            format!("External event from {source}")
        }
    };

    let outcome_desc = format!(
        "{} nodes modified, {} synapses changed, {} scars added. {}",
        episode.outcome.nodes_modified,
        episode.outcome.synapses_changed,
        episode.outcome.scars_added,
        episode.outcome.details,
    );

    let validation_desc = if episode.validation.success {
        "Validated: success".to_string()
    } else {
        let scar_info = episode
            .validation
            .scar_id
            .map(|id| format!(" (scar #{id})"))
            .unwrap_or_default();
        let msg = episode
            .validation
            .message
            .as_deref()
            .unwrap_or("unknown error");
        format!("Validated: failure — {msg}{scar_info}")
    };

    let node_list = if episode.involved_nodes.len() <= 5 {
        format!("{:?}", episode.involved_nodes)
    } else {
        let first_five: Vec<u64> = episode.involved_nodes.iter().take(5).copied().collect();
        format!(
            "{:?} (+{} more)",
            first_five,
            episode.involved_nodes.len() - 5
        )
    };

    let process_len = episode.process_trace.len();

    format!(
        "Pattern: {stimulus_desc} → {process_len} activation steps → {outcome_desc}. \
         {validation_desc}. Nodes involved: {node_list}."
    )
}

/// Extracts a lesson by cross-referencing multiple episodes involving the same nodes.
///
/// This looks for recurring patterns: "When X and Y are modified together, Z is always impacted."
pub fn extract_cross_lesson(episodes: &[Episode]) -> String {
    if episodes.is_empty() {
        return "No episodes to analyze.".to_string();
    }

    // Count node co-occurrence across episodes.
    let mut node_frequency: HashMap<u64, usize> = HashMap::new();
    let mut pair_frequency: HashMap<(u64, u64), usize> = HashMap::new();

    for ep in episodes {
        for &nid in &ep.involved_nodes {
            *node_frequency.entry(nid).or_insert(0) += 1;
        }
        // Count pairs.
        let nodes = &ep.involved_nodes;
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let pair = if nodes[i] < nodes[j] {
                    (nodes[i], nodes[j])
                } else {
                    (nodes[j], nodes[i])
                };
                *pair_frequency.entry(pair).or_insert(0) += 1;
            }
        }
    }

    // Find the most frequently co-occurring pair.
    let mut lines = Vec::new();
    lines.push(format!(
        "Cross-episode analysis over {} episodes:",
        episodes.len()
    ));

    // Top co-occurring pairs.
    let mut pairs: Vec<_> = pair_frequency.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));

    for ((a, b), count) in pairs.iter().take(3) {
        if *count > 1 {
            lines.push(format!(
                "  - Nodes {a} and {b} are modified together in {count}/{} episodes",
                episodes.len()
            ));
        }
    }

    // Success/failure ratio.
    let success_count = episodes.iter().filter(|e| e.validation.success).count();
    let failure_count = episodes.len() - success_count;
    lines.push(format!(
        "  - Success rate: {success_count}/{} ({failure_count} failures)",
        episodes.len()
    ));

    // Top recurring nodes.
    let mut nodes: Vec<_> = node_frequency.into_iter().collect();
    nodes.sort_by(|a, b| b.1.cmp(&a.1));
    for (nid, count) in nodes.iter().take(5) {
        if *count > 1 {
            lines.push(format!(
                "  - Node {nid} appears in {count}/{} episodes",
                episodes.len()
            ));
        }
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// EpisodeRecorder (MutationListener)
// ---------------------------------------------------------------------------

/// Records episodes from mutation batches via the MutationListener interface.
///
/// Each batch of mutations creates a single episode with:
/// - Stimulus = Mutation (aggregated types + node IDs)
/// - Process trace = one step per mutation event
/// - Outcome = count of nodes modified, synapses changed
/// - Validation = always success (mutations already committed)
pub struct EpisodeRecorder {
    store: Arc<EpisodeStore>,
}

impl EpisodeRecorder {
    /// Creates a new episode recorder.
    pub fn new(store: Arc<EpisodeStore>) -> Self {
        Self { store }
    }

    /// Returns a reference to the underlying store.
    pub fn store(&self) -> &Arc<EpisodeStore> {
        &self.store
    }

    /// Records a mutation event as an episode (convenience method).
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

        let process_trace = vec![ActivationStep {
            node_id: node_ids.first().copied().unwrap_or(0),
            activation_level: 1.0,
            source_node: None,
            description: format!("Mutation: {mutation_type}"),
        }];

        let outcome = Outcome::with_counts(node_ids.len(), 0, usize::from(!success), details);

        let validation = if success {
            ValidationResult::ok()
        } else {
            ValidationResult::failure(None, details)
        };

        self.store.record(
            stimulus,
            process_trace,
            outcome,
            validation,
            node_ids.to_vec(),
            vec![mutation_type.to_string()],
        )
    }

    /// Extracts node IDs from a mutation event.
    fn extract_node_ids(event: &MutationEvent) -> Vec<u64> {
        match event {
            MutationEvent::NodeCreated { node } => vec![node.id.0],
            MutationEvent::NodeUpdated { before, .. } => vec![before.id.0],
            MutationEvent::NodeDeleted { node } => vec![node.id.0],
            MutationEvent::EdgeCreated { edge } => vec![edge.src.0, edge.dst.0],
            MutationEvent::EdgeUpdated { before, .. } => vec![before.src.0, before.dst.0],
            MutationEvent::EdgeDeleted { edge } => vec![edge.src.0, edge.dst.0],
        }
    }

    /// Returns the event type name for a mutation event.
    fn event_type_name(event: &MutationEvent) -> &'static str {
        match event {
            MutationEvent::NodeCreated { .. } => "NodeCreated",
            MutationEvent::NodeUpdated { .. } => "NodeUpdated",
            MutationEvent::NodeDeleted { .. } => "NodeDeleted",
            MutationEvent::EdgeCreated { .. } => "EdgeCreated",
            MutationEvent::EdgeUpdated { .. } => "EdgeUpdated",
            MutationEvent::EdgeDeleted { .. } => "EdgeDeleted",
        }
    }
}

#[async_trait]
impl MutationListener for EpisodeRecorder {
    fn name(&self) -> &str {
        "EpisodeRecorder"
    }

    async fn on_event(&self, event: &MutationEvent) {
        let node_ids = Self::extract_node_ids(event);
        let event_type = Self::event_type_name(event);

        let stimulus = Stimulus::Mutation {
            mutation_type: event_type.to_string(),
            node_ids: node_ids.clone(),
        };

        let process_trace = vec![ActivationStep {
            node_id: node_ids.first().copied().unwrap_or(0),
            activation_level: 1.0,
            source_node: None,
            description: format!("{event_type} event processed"),
        }];

        let outcome = Outcome::with_counts(node_ids.len(), 0, 0, format!("{event_type} applied"));

        self.store.record(
            stimulus,
            process_trace,
            outcome,
            ValidationResult::ok(),
            node_ids,
            vec![event_type.to_string()],
        );
    }

    async fn on_batch(&self, events: &[MutationEvent]) {
        if events.is_empty() {
            return;
        }

        // Aggregate all node IDs and mutation types from the batch.
        let mut all_node_ids: Vec<u64> = Vec::new();
        let mut mutation_types: Vec<String> = Vec::new();
        let mut process_trace: Vec<ActivationStep> = Vec::new();
        let mut nodes_modified: usize = 0;

        for event in events {
            let node_ids = Self::extract_node_ids(event);
            let event_type = Self::event_type_name(event);

            process_trace.push(ActivationStep {
                node_id: node_ids.first().copied().unwrap_or(0),
                activation_level: 1.0,
                source_node: None,
                description: format!("{event_type} event"),
            });

            nodes_modified += node_ids.len();
            all_node_ids.extend_from_slice(&node_ids);

            if !mutation_types.contains(&event_type.to_string()) {
                mutation_types.push(event_type.to_string());
            }
        }

        // Deduplicate node IDs.
        all_node_ids.sort_unstable();
        all_node_ids.dedup();

        let stimulus = Stimulus::Mutation {
            mutation_type: mutation_types.join("+"),
            node_ids: all_node_ids.clone(),
        };

        let outcome = Outcome::with_counts(
            nodes_modified,
            0,
            0,
            format!("Batch of {} mutations applied", events.len()),
        );

        self.store.record(
            stimulus,
            process_trace,
            outcome,
            ValidationResult::ok(),
            all_node_ids,
            mutation_types,
        );
    }
}

// ---------------------------------------------------------------------------
// EpisodeMemoryManager
// ---------------------------------------------------------------------------

/// Manages episode lifecycle through memory horizons.
///
/// Applies the same consolidation/archival rules as [`MemoryManager`](crate::memory::MemoryManager)
/// but for episodes:
///
/// - **Operational** → **Consolidated**: episode older than `consolidation_age_secs`
///   and has been accessed at least once.
/// - **Consolidated** → **Archived**: episode older than `archive_age_secs`
///   and not frequently accessed (access_count < `min_access_for_retain`).
/// - **Archived** → **Operational**: reactivation when episode is accessed.
///
/// Old archived episodes are evicted when the store exceeds `max_episodes`.
pub struct EpisodeMemoryManager {
    store: Arc<EpisodeStore>,
}

/// Result of a memory sweep.
#[derive(Debug, Clone, Default)]
pub struct EpisodeSweepResult {
    /// Number of episodes promoted to Consolidated.
    pub consolidated: usize,
    /// Number of episodes demoted to Archived.
    pub archived: usize,
    /// Number of episodes reactivated to Operational.
    pub reactivated: usize,
    /// Number of episodes evicted.
    pub evicted: usize,
}

impl EpisodeMemoryManager {
    /// Creates a new episode memory manager.
    pub fn new(store: Arc<EpisodeStore>) -> Self {
        Self { store }
    }

    /// Runs a single sweep, transitioning episodes between horizons.
    pub fn sweep(&self) -> EpisodeSweepResult {
        let mut result = EpisodeSweepResult::default();
        let now = SystemTime::now();

        let consolidation_threshold =
            Duration::from_secs(self.store.config().consolidation_age_secs);
        let archive_threshold = Duration::from_secs(self.store.config().archive_age_secs);
        let min_access = self.store.config().min_access_for_retain;

        let ids = self.store.all_ids();
        for id in ids {
            if let Some(mut entry) = self.store.episodes.get_mut(&id) {
                let age = now
                    .duration_since(entry.created_at)
                    .unwrap_or(Duration::ZERO);
                let idle = now
                    .duration_since(entry.last_accessed)
                    .unwrap_or(Duration::ZERO);

                match entry.horizon {
                    EpisodeHorizon::Operational => {
                        // Promote to consolidated if old enough and accessed.
                        if age >= consolidation_threshold && entry.access_count > 0 {
                            entry.horizon = EpisodeHorizon::Consolidated;
                            result.consolidated += 1;
                        }
                    }
                    EpisodeHorizon::Consolidated => {
                        // Archive if old and not frequently accessed.
                        if idle >= archive_threshold && entry.access_count < min_access {
                            entry.horizon = EpisodeHorizon::Archived;
                            result.archived += 1;
                        }
                    }
                    EpisodeHorizon::Archived => {
                        // Reactivate if recently accessed.
                        if idle < consolidation_threshold {
                            entry.horizon = EpisodeHorizon::Operational;
                            result.reactivated += 1;
                        }
                    }
                }
            }
        }

        // Evict excess archived episodes.
        let max = self.store.config().max_episodes;
        while self.store.len() > max {
            let oldest_archived = self
                .store
                .episodes
                .iter()
                .filter(|e| e.value().horizon == EpisodeHorizon::Archived)
                .min_by_key(|e| e.value().created_at)
                .map(|e| *e.key());

            if let Some(id) = oldest_archived {
                self.store.remove(id);
                result.evicted += 1;
            } else {
                break; // No more archived episodes to evict.
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stimulus_covers_three_types() {
        let m = Stimulus::Mutation {
            mutation_type: "NodeCreated".into(),
            node_ids: vec![1],
        };
        let q = Stimulus::Query {
            query_text: "MATCH (n) RETURN n".into(),
        };
        let e = Stimulus::External {
            source: "api".into(),
            description: "webhook".into(),
        };

        // Just check they serialize.
        let _ = serde_json::to_string(&m).unwrap();
        let _ = serde_json::to_string(&q).unwrap();
        let _ = serde_json::to_string(&e).unwrap();
    }

    #[test]
    fn episode_serializable() {
        let ep = Episode {
            id: 1,
            stimulus: Stimulus::Mutation {
                mutation_type: "NodeCreated".into(),
                node_ids: vec![1, 2],
            },
            process_trace: vec![ActivationStep {
                node_id: 1,
                activation_level: 0.8,
                source_node: None,
                description: "initial".into(),
            }],
            outcome: Outcome::success("ok"),
            validation: ValidationResult::ok(),
            lesson: Some("lesson".into()),
            created_at: SystemTime::now(),
            involved_nodes: vec![1, 2],
            tags: vec!["test".into()],
            horizon: EpisodeHorizon::Operational,
            access_count: 0,
            last_accessed: SystemTime::now(),
        };

        let json = serde_json::to_string(&ep).unwrap();
        let deserialized: Episode = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, 1);
        assert_eq!(deserialized.involved_nodes, vec![1, 2]);
    }

    #[test]
    fn validation_result_with_scar() {
        let v = ValidationResult::failure(Some(42), "constraint violated");
        assert!(!v.success);
        assert_eq!(v.scar_id, Some(42));
        assert_eq!(v.message.as_deref(), Some("constraint violated"));
    }
}
