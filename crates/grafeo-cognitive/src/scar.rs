//! Scar System — persistent memory of errors and problems.
//!
//! A scar is placed on a node or edge when an operation has caused a problem
//! (rollback, error, invalidation). Scars have an intensity that decays
//! exponentially over time (natural healing). Active scars influence the
//! risk_score in the Knowledge Fabric.
//!
//! ```text
//! intensity(t) = I0 × 2^(-Δt / half_life)
//! ```

use dashmap::DashMap;
use grafeo_common::types::NodeId;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// ScarConfig
// ---------------------------------------------------------------------------

/// Configuration for the scar system.
#[derive(Debug, Clone)]
pub struct ScarConfig {
    /// Default half-life for scar intensity decay.
    pub default_half_life: Duration,
    /// Minimum intensity threshold — scars below this are inactive.
    pub min_intensity: f64,
    /// Maximum scars per node before oldest are pruned.
    pub max_scars_per_node: usize,
}

impl Default for ScarConfig {
    fn default() -> Self {
        Self {
            default_half_life: Duration::from_secs(30 * 24 * 3600), // 30 days
            min_intensity: 0.01,
            max_scars_per_node: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// ScarId
// ---------------------------------------------------------------------------

/// Unique identifier for a scar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScarId(pub u64);

static NEXT_SCAR_ID: AtomicU64 = AtomicU64::new(1);

impl ScarId {
    fn next() -> Self {
        Self(NEXT_SCAR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for ScarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scar:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ScarReason
// ---------------------------------------------------------------------------

/// Why a scar was placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScarReason {
    /// A transaction was rolled back.
    Rollback,
    /// An operation caused an error.
    Error(String),
    /// Data was invalidated.
    Invalidation,
    /// A constraint was violated.
    ConstraintViolation(String),
    /// Custom reason.
    Custom(String),
}

impl fmt::Display for ScarReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rollback => write!(f, "rollback"),
            Self::Error(msg) => write!(f, "error: {}", msg),
            Self::Invalidation => write!(f, "invalidation"),
            Self::ConstraintViolation(msg) => write!(f, "constraint violation: {}", msg),
            Self::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Scar
// ---------------------------------------------------------------------------

/// A scar on a node — records that a problem occurred.
#[derive(Debug, Clone)]
pub struct Scar {
    /// Unique scar identifier.
    pub id: ScarId,
    /// The node this scar is placed on.
    pub target: NodeId,
    /// Initial intensity (severity of the problem).
    intensity: f64,
    /// Why the scar was placed.
    pub reason: ScarReason,
    /// When the scar was placed.
    pub created_at: Instant,
    /// When the scar was explicitly healed (None if not yet healed).
    pub healed_at: Option<Instant>,
    /// Half-life for intensity decay.
    half_life: Duration,
}

impl Scar {
    /// Creates a new scar.
    fn new(target: NodeId, intensity: f64, reason: ScarReason, half_life: Duration) -> Self {
        Self {
            id: ScarId::next(),
            target,
            intensity: intensity.max(0.0),
            reason,
            created_at: Instant::now(),
            healed_at: None,
            half_life,
        }
    }

    /// Creates a scar at a specific time (for testing).
    pub fn new_at(
        target: NodeId,
        intensity: f64,
        reason: ScarReason,
        half_life: Duration,
        created_at: Instant,
    ) -> Self {
        Self {
            id: ScarId::next(),
            target,
            intensity: intensity.max(0.0),
            reason,
            created_at,
            healed_at: None,
            half_life,
        }
    }

    /// Returns the current intensity after decay.
    ///
    /// `I(t) = I0 × 2^(-Δt / half_life)`
    ///
    /// Returns 0.0 if healed.
    pub fn current_intensity(&self) -> f64 {
        self.intensity_at(Instant::now())
    }

    /// Returns the intensity at a specific instant.
    pub fn intensity_at(&self, now: Instant) -> f64 {
        if self.healed_at.is_some() {
            return 0.0;
        }
        let elapsed = now.duration_since(self.created_at);
        let half_lives = elapsed.as_secs_f64() / self.half_life.as_secs_f64();
        self.intensity * 2.0_f64.powf(-half_lives)
    }

    /// Returns the raw (non-decayed) intensity.
    pub fn raw_intensity(&self) -> f64 {
        self.intensity
    }

    /// Returns `true` if the scar has been explicitly healed.
    pub fn is_healed(&self) -> bool {
        self.healed_at.is_some()
    }

    /// Returns `true` if the scar is still active (intensity > threshold).
    pub fn is_active(&self, min_intensity: f64) -> bool {
        !self.is_healed() && self.current_intensity() >= min_intensity
    }

    /// Explicitly heals this scar.
    pub fn heal(&mut self) {
        self.healed_at = Some(Instant::now());
    }

    /// Heals at a specific time (for testing).
    pub fn heal_at(&mut self, now: Instant) {
        self.healed_at = Some(now);
    }
}

// ---------------------------------------------------------------------------
// ScarStore
// ---------------------------------------------------------------------------

/// Thread-safe store for scars.
///
/// Indexed by target `NodeId` for efficient lookup.
pub struct ScarStore {
    /// Scars indexed by node ID → vec of scars.
    scars: DashMap<NodeId, Vec<Scar>>,
    /// Global index by scar ID → (node_id, index).
    index: DashMap<ScarId, NodeId>,
    /// Configuration.
    config: ScarConfig,
}

impl ScarStore {
    /// Creates a new scar store.
    pub fn new(config: ScarConfig) -> Self {
        Self {
            scars: DashMap::new(),
            index: DashMap::new(),
            config,
        }
    }

    /// Places a scar on a node. Returns the scar ID.
    pub fn add_scar(&self, target: NodeId, intensity: f64, reason: ScarReason) -> ScarId {
        let scar = Scar::new(target, intensity, reason, self.config.default_half_life);
        let id = scar.id;

        self.scars
            .entry(target)
            .and_modify(|scars| {
                scars.push(scar.clone());
                // Prune if too many scars — remove oldest inactive ones
                if scars.len() > self.config.max_scars_per_node {
                    scars.sort_by(|a, b| {
                        b.current_intensity()
                            .partial_cmp(&a.current_intensity())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    scars.truncate(self.config.max_scars_per_node);
                }
            })
            .or_insert_with(|| vec![scar]);

        self.index.insert(id, target);
        id
    }

    /// Heals a scar by ID.
    pub fn heal(&self, scar_id: ScarId) -> bool {
        if let Some(node_id) = self.index.get(&scar_id) {
            if let Some(mut scars) = self.scars.get_mut(&*node_id) {
                if let Some(scar) = scars.iter_mut().find(|s| s.id == scar_id) {
                    scar.heal();
                    return true;
                }
            }
        }
        false
    }

    /// Returns all scars for a node (active and healed).
    pub fn get_scars(&self, node_id: NodeId) -> Vec<Scar> {
        self.scars
            .get(&node_id)
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Returns only active scars for a node (above min_intensity).
    pub fn get_active_scars(&self, node_id: NodeId) -> Vec<Scar> {
        let min = self.config.min_intensity;
        self.scars
            .get(&node_id)
            .map(|s| s.iter().filter(|s| s.is_active(min)).cloned().collect())
            .unwrap_or_default()
    }

    /// Returns the cumulative active scar intensity for a node.
    ///
    /// This can be used to influence the risk_score in the Knowledge Fabric.
    pub fn cumulative_intensity(&self, node_id: NodeId) -> f64 {
        let min = self.config.min_intensity;
        self.scars
            .get(&node_id)
            .map(|scars| {
                scars
                    .iter()
                    .filter(|s| s.is_active(min))
                    .map(|s| s.current_intensity())
                    .sum()
            })
            .unwrap_or(0.0)
    }

    /// Returns all nodes that have active scars above `min_intensity`.
    pub fn nodes_with_active_scars(&self, min_intensity: f64) -> Vec<(NodeId, f64)> {
        self.scars
            .iter()
            .filter_map(|entry| {
                let cumulative: f64 = entry
                    .value()
                    .iter()
                    .filter(|s| s.is_active(min_intensity))
                    .map(|s| s.current_intensity())
                    .sum();
                if cumulative > 0.0 {
                    Some((*entry.key(), cumulative))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Prunes all healed and expired scars (intensity < min_intensity).
    pub fn prune(&self) -> usize {
        let min = self.config.min_intensity;
        let mut pruned = 0;

        for mut entry in self.scars.iter_mut() {
            let before = entry.value().len();
            entry.value_mut().retain(|s| s.is_active(min));
            let after = entry.value().len();
            pruned += before - after;
        }

        // Clean up empty entries
        self.scars.retain(|_, v| !v.is_empty());

        if pruned > 0 {
            tracing::debug!(pruned, "pruned inactive scars");
        }
        pruned
    }

    /// Returns the total number of tracked scars.
    pub fn total_scars(&self) -> usize {
        self.scars.iter().map(|e| e.value().len()).sum()
    }

    /// Returns the number of active scars.
    pub fn active_scar_count(&self) -> usize {
        let min = self.config.min_intensity;
        self.scars
            .iter()
            .map(|e| e.value().iter().filter(|s| s.is_active(min)).count())
            .sum()
    }

    /// Returns a reference to the config.
    pub fn config(&self) -> &ScarConfig {
        &self.config
    }
}

impl Default for ScarStore {
    fn default() -> Self {
        Self::new(ScarConfig::default())
    }
}

impl fmt::Debug for ScarStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScarStore")
            .field("total_scars", &self.total_scars())
            .field("config", &self.config)
            .finish()
    }
}
