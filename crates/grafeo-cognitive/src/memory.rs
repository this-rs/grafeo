//! Memory Horizons — node lifecycle management with tiered storage.
//!
//! Nodes move through three memory horizons based on energy, age, and activity:
//!
//! ```text
//! Operational ──(promotion)──> Consolidated ──(demotion)──> Archived
//!     ^                             |                          |
//!     └─────────(reactivation)──────┘──────────────────────────┘
//! ```
//!
//! - **Operational**: hot, frequently accessed, recent nodes
//! - **Consolidated**: warm, confirmed patterns, energy above threshold
//! - **Archived**: cold, rarely accessed, candidates for eviction
//!
//! The [`MemoryManager`] runs a periodic sweep to promote/demote nodes
//! based on rules tied to the [`EnergyStore`](crate::energy::EnergyStore).

use async_trait::async_trait;
use dashmap::DashMap;
use grafeo_common::types::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;

#[cfg(feature = "energy")]
use crate::energy::EnergyStore;

// ---------------------------------------------------------------------------
// MemoryHorizon
// ---------------------------------------------------------------------------

/// The three tiers of memory storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryHorizon {
    /// Hot: recently created or actively used nodes.
    Operational,
    /// Warm: confirmed patterns with sustained energy above threshold.
    Consolidated,
    /// Cold: low energy, inactive nodes — candidates for eviction.
    Archived,
}

impl fmt::Display for MemoryHorizon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operational => write!(f, "operational"),
            Self::Consolidated => write!(f, "consolidated"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryConfig
// ---------------------------------------------------------------------------

/// Configuration for the memory horizon system.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Energy threshold for promotion from Operational to Consolidated.
    /// Node must have energy >= this value AND be old enough.
    pub promotion_energy_threshold: f64,
    /// Minimum age before a node can be promoted to Consolidated.
    pub promotion_min_age: Duration,
    /// Energy threshold for demotion to Archived.
    /// Nodes with energy < this value AND idle for too long are demoted.
    pub demotion_energy_threshold: f64,
    /// Maximum idle duration before demotion to Archived.
    pub demotion_max_idle: Duration,
    /// Interval between periodic sweeps.
    pub sweep_interval: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            promotion_energy_threshold: 2.0,
            promotion_min_age: Duration::from_secs(3600), // 1 hour
            demotion_energy_threshold: 0.1,
            demotion_max_idle: Duration::from_secs(7 * 24 * 3600), // 7 days
            sweep_interval: Duration::from_secs(3600),             // 1 hour
        }
    }
}

// ---------------------------------------------------------------------------
// NodeMemoryState
// ---------------------------------------------------------------------------

/// Tracks a node's current memory horizon and transition history.
#[derive(Debug, Clone)]
pub struct NodeMemoryState {
    /// Current memory horizon.
    pub horizon: MemoryHorizon,
    /// When the node entered its current horizon.
    pub entered_current_horizon: Instant,
    /// When the node was first tracked.
    pub created_at: Instant,
    /// Total number of horizon transitions.
    pub transition_count: u32,
}

impl NodeMemoryState {
    /// Creates a new state at the Operational horizon.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            horizon: MemoryHorizon::Operational,
            entered_current_horizon: now,
            created_at: now,
            transition_count: 0,
        }
    }

    /// Creates a state with a specific creation time (for testing).
    pub fn new_at(created_at: Instant) -> Self {
        Self {
            horizon: MemoryHorizon::Operational,
            entered_current_horizon: created_at,
            created_at,
            transition_count: 0,
        }
    }

    /// Transitions to a new horizon.
    pub fn transition_to(&mut self, horizon: MemoryHorizon) {
        if self.horizon != horizon {
            self.horizon = horizon;
            self.entered_current_horizon = Instant::now();
            self.transition_count += 1;
        }
    }

    /// Transitions to a new horizon at a specific time (for testing).
    pub fn transition_to_at(&mut self, horizon: MemoryHorizon, now: Instant) {
        if self.horizon != horizon {
            self.horizon = horizon;
            self.entered_current_horizon = now;
            self.transition_count += 1;
        }
    }

    /// Duration since the node entered its current horizon.
    pub fn time_in_current_horizon(&self) -> Duration {
        Instant::now().duration_since(self.entered_current_horizon)
    }

    /// Total age of the node since first tracked.
    pub fn age(&self) -> Duration {
        Instant::now().duration_since(self.created_at)
    }
}

impl Default for NodeMemoryState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

/// Thread-safe store for node memory states.
pub struct MemoryStore {
    /// Per-node memory state.
    nodes: DashMap<NodeId, NodeMemoryState>,
}

impl MemoryStore {
    /// Creates a new, empty memory store.
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
        }
    }

    /// Returns the horizon for a node, defaulting to Operational for unknown nodes.
    pub fn get_horizon(&self, node_id: NodeId) -> MemoryHorizon {
        self.nodes
            .get(&node_id)
            .map(|s| s.horizon)
            .unwrap_or(MemoryHorizon::Operational)
    }

    /// Returns the full memory state for a node, if tracked.
    pub fn get_state(&self, node_id: NodeId) -> Option<NodeMemoryState> {
        self.nodes.get(&node_id).map(|s| s.clone())
    }

    /// Sets or creates a node's horizon.
    pub fn set_horizon(&self, node_id: NodeId, horizon: MemoryHorizon) {
        self.nodes
            .entry(node_id)
            .and_modify(|s| s.transition_to(horizon))
            .or_insert_with(|| {
                let mut s = NodeMemoryState::new();
                s.horizon = horizon;
                s
            });
    }

    /// Ensures a node is tracked (creates at Operational if absent).
    pub fn track(&self, node_id: NodeId) {
        self.nodes
            .entry(node_id)
            .or_insert_with(NodeMemoryState::new);
    }

    /// Returns all node IDs in a given horizon.
    pub fn list_by_horizon(&self, horizon: MemoryHorizon) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|entry| entry.value().horizon == horizon)
            .map(|entry| *entry.key())
            .collect()
    }

    /// Returns all tracked node states as a snapshot.
    pub fn snapshot(&self) -> Vec<(NodeId, NodeMemoryState)> {
        self.nodes
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    /// Returns the number of tracked nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes are tracked.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns counts per horizon.
    pub fn horizon_counts(&self) -> HashMap<MemoryHorizon, usize> {
        let mut counts = HashMap::new();
        for entry in self.nodes.iter() {
            *counts.entry(entry.value().horizon).or_insert(0) += 1;
        }
        counts
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryStore")
            .field("tracked_nodes", &self.nodes.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SweepResult
// ---------------------------------------------------------------------------

/// Result of a single sweep operation.
#[derive(Debug, Clone, Default)]
pub struct SweepResult {
    /// Nodes promoted from Operational → Consolidated.
    pub promoted: Vec<NodeId>,
    /// Nodes demoted from Consolidated → Archived (or Operational → Archived).
    pub demoted: Vec<NodeId>,
    /// Nodes reactivated from Archived → Operational.
    pub reactivated: Vec<NodeId>,
}

impl SweepResult {
    /// Total number of transitions in this sweep.
    pub fn total_transitions(&self) -> usize {
        self.promoted.len() + self.demoted.len() + self.reactivated.len()
    }
}

// ---------------------------------------------------------------------------
// ArchiveBackend trait
// ---------------------------------------------------------------------------

/// Pluggable backend for archiving node data.
///
/// Implementations handle serialization and storage of archived node data
/// (e.g., to files, S3, or a cold storage tier).
#[async_trait]
pub trait ArchiveBackend: Send + Sync + fmt::Debug {
    /// Archives data for a node.
    async fn archive(&self, node_id: NodeId, data: &[u8]) -> Result<(), std::io::Error>;

    /// Restores archived data for a node. Returns `None` if not archived.
    async fn restore(&self, node_id: NodeId) -> Result<Option<Vec<u8>>, std::io::Error>;

    /// Removes archived data for a node (e.g., on reactivation).
    async fn remove(&self, node_id: NodeId) -> Result<(), std::io::Error>;

    /// Returns `true` if data exists for this node.
    async fn exists(&self, node_id: NodeId) -> Result<bool, std::io::Error>;
}

// ---------------------------------------------------------------------------
// FileArchiveBackend
// ---------------------------------------------------------------------------

/// File-based archive backend. Stores node data as JSON files in a directory.
#[derive(Debug, Clone)]
pub struct FileArchiveBackend {
    /// Directory where archived node data is stored.
    dir: PathBuf,
}

impl FileArchiveBackend {
    /// Creates a new file archive backend in the given directory.
    ///
    /// Creates the directory if it doesn't exist.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref().to_path_buf();
        Self { dir }
    }

    fn node_path(&self, node_id: NodeId) -> PathBuf {
        self.dir.join(format!("node_{}.archive", node_id.0))
    }
}

#[async_trait]
impl ArchiveBackend for FileArchiveBackend {
    async fn archive(&self, node_id: NodeId, data: &[u8]) -> Result<(), std::io::Error> {
        tokio::fs::create_dir_all(&self.dir).await?;
        tokio::fs::write(self.node_path(node_id), data).await
    }

    async fn restore(&self, node_id: NodeId) -> Result<Option<Vec<u8>>, std::io::Error> {
        let path = self.node_path(node_id);
        match tokio::fs::read(&path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn remove(&self, node_id: NodeId) -> Result<(), std::io::Error> {
        let path = self.node_path(node_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn exists(&self, node_id: NodeId) -> Result<bool, std::io::Error> {
        Ok(self.node_path(node_id).exists())
    }
}

// ---------------------------------------------------------------------------
// InMemoryArchiveBackend (for testing)
// ---------------------------------------------------------------------------

/// In-memory archive backend for testing.
#[derive(Debug, Default)]
pub struct InMemoryArchiveBackend {
    data: DashMap<NodeId, Vec<u8>>,
}

impl InMemoryArchiveBackend {
    /// Creates a new in-memory archive backend.
    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
        }
    }

    /// Returns the number of archived entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if no entries are archived.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[async_trait]
impl ArchiveBackend for InMemoryArchiveBackend {
    async fn archive(&self, node_id: NodeId, data: &[u8]) -> Result<(), std::io::Error> {
        self.data.insert(node_id, data.to_vec());
        Ok(())
    }

    async fn restore(&self, node_id: NodeId) -> Result<Option<Vec<u8>>, std::io::Error> {
        Ok(self.data.get(&node_id).map(|v| v.clone()))
    }

    async fn remove(&self, node_id: NodeId) -> Result<(), std::io::Error> {
        self.data.remove(&node_id);
        Ok(())
    }

    async fn exists(&self, node_id: NodeId) -> Result<bool, std::io::Error> {
        Ok(self.data.contains_key(&node_id))
    }
}

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

/// Manages periodic sweep of node memory horizons.
///
/// The manager reads energy levels from the [`EnergyStore`] and applies
/// promotion/demotion rules to move nodes between horizons.
pub struct MemoryManager {
    /// Shared memory store.
    store: Arc<MemoryStore>,
    /// Configuration.
    config: MemoryConfig,
    /// Energy store for reading node energy levels.
    #[cfg(feature = "energy")]
    energy_store: Arc<EnergyStore>,
    /// Optional archive backend for persisting archived node data.
    archive: Option<Arc<dyn ArchiveBackend>>,
    /// Shutdown signal.
    shutdown: Arc<Notify>,
}

impl MemoryManager {
    /// Creates a new memory manager.
    #[cfg(feature = "energy")]
    pub fn new(
        store: Arc<MemoryStore>,
        config: MemoryConfig,
        energy_store: Arc<EnergyStore>,
    ) -> Self {
        Self {
            store,
            config,
            energy_store,
            archive: None,
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Sets the archive backend.
    pub fn with_archive(mut self, archive: Arc<dyn ArchiveBackend>) -> Self {
        self.archive = Some(archive);
        self
    }

    /// Returns a reference to the memory store.
    pub fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }

    /// Returns a reference to the config.
    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Performs a single sweep: evaluates all tracked nodes and applies
    /// promotion/demotion rules.
    ///
    /// # Rules
    ///
    /// - **Promotion** (Operational → Consolidated):
    ///   energy >= `promotion_energy_threshold` AND age >= `promotion_min_age`
    ///
    /// - **Demotion** (Consolidated → Archived):
    ///   energy < `demotion_energy_threshold` AND time in horizon > `demotion_max_idle`
    ///
    /// - **Emergency demotion** (Operational → Archived):
    ///   energy < `demotion_energy_threshold` AND age > `demotion_max_idle`
    ///
    /// - **Reactivation** (Archived → Operational):
    ///   energy >= `promotion_energy_threshold` (node was re-boosted)
    #[cfg(feature = "energy")]
    pub fn sweep(&self) -> SweepResult {
        self.sweep_at(Instant::now())
    }

    /// Sweep at a specific instant (for testing).
    #[cfg(feature = "energy")]
    pub fn sweep_at(&self, now: Instant) -> SweepResult {
        let mut result = SweepResult::default();

        for mut entry in self.store.nodes.iter_mut() {
            let node_id = *entry.key();
            let state = entry.value_mut();
            let energy = self.energy_store.get_energy(node_id);

            match state.horizon {
                MemoryHorizon::Operational => {
                    let age = now.duration_since(state.created_at);

                    if energy >= self.config.promotion_energy_threshold
                        && age >= self.config.promotion_min_age
                    {
                        // Promote to Consolidated
                        state.transition_to_at(MemoryHorizon::Consolidated, now);
                        result.promoted.push(node_id);
                    } else if energy < self.config.demotion_energy_threshold
                        && age > self.config.demotion_max_idle
                    {
                        // Emergency demotion — skip Consolidated
                        state.transition_to_at(MemoryHorizon::Archived, now);
                        result.demoted.push(node_id);
                    }
                }
                MemoryHorizon::Consolidated => {
                    let time_in_horizon = now.duration_since(state.entered_current_horizon);

                    if energy < self.config.demotion_energy_threshold
                        && time_in_horizon > self.config.demotion_max_idle
                    {
                        // Demote to Archived
                        state.transition_to_at(MemoryHorizon::Archived, now);
                        result.demoted.push(node_id);
                    }
                }
                MemoryHorizon::Archived => {
                    if energy >= self.config.promotion_energy_threshold {
                        // Reactivation — back to Operational
                        state.transition_to_at(MemoryHorizon::Operational, now);
                        result.reactivated.push(node_id);
                    }
                }
            }
        }

        if result.total_transitions() > 0 {
            tracing::info!(
                promoted = result.promoted.len(),
                demoted = result.demoted.len(),
                reactivated = result.reactivated.len(),
                "memory sweep completed with {} transition(s)",
                result.total_transitions()
            );
        }

        result
    }

    /// Starts the periodic sweep as a background tokio task.
    ///
    /// Returns a handle to the spawned task.
    #[cfg(feature = "energy")]
    pub fn start_periodic(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval = self.config.sweep_interval;
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.tick().await; // Skip first immediate tick

            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        self.sweep();
                    }
                    () = shutdown.notified() => {
                        tracing::info!("memory manager shutting down");
                        // Final sweep before shutdown
                        self.sweep();
                        break;
                    }
                }
            }
        })
    }

    /// Signals the periodic task to shut down.
    pub fn shutdown(&self) {
        self.shutdown.notify_one();
    }
}

impl fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryManager")
            .field("store", &self.store)
            .field("config", &self.config)
            .field("has_archive", &self.archive.is_some())
            .finish()
    }
}
