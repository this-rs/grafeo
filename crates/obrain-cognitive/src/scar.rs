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
use obrain_common::types::NodeId;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::store_trait::{
    OptionalGraphStore, PROP_SCAR_COUNT, PROP_SCAR_INTENSITY, load_node_f64, persist_node_f64,
};

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
    /// LLM output was truncated.
    Truncation,
    /// LLM output contained repetitive content.
    Repetition,
    /// The user explicitly corrected the output.
    UserCorrection,
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
            Self::Truncation => write!(f, "truncation"),
            Self::Repetition => write!(f, "repetition"),
            Self::UserCorrection => write!(f, "user correction"),
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
/// When a backing [`GraphStoreMut`](obrain_core::graph::GraphStoreMut) is
/// provided, scar count and cumulative intensity are persisted as node
/// properties (write-through). On read, if the node is not in the hot cache,
/// the store does NOT lazy-load individual scars (since scar details like
/// reason/created_at are not stored in properties), but cumulative metrics
/// are persisted for the fabric layer.
pub struct ScarStore {
    /// Scars indexed by node ID → vec of scars.
    scars: DashMap<NodeId, Vec<Scar>>,
    /// Global index by scar ID → (node_id, index).
    index: DashMap<ScarId, NodeId>,
    /// Configuration.
    config: ScarConfig,
    /// Optional backing graph store for write-through persistence.
    graph_store: OptionalGraphStore,
    /// Optional substrate backend — when present, the 5-bit scar sub-field
    /// of `NodeRecord.scar_util_affinity` is the durable source of truth for
    /// cumulative intensity (individual `Scar` records with `reason`/timestamps
    /// are in-memory only, same fidelity as the legacy `load_from_graph`).
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
}

impl ScarStore {
    /// Creates a new scar store (in-memory only).
    pub fn new(config: ScarConfig) -> Self {
        Self {
            scars: DashMap::new(),
            index: DashMap::new(),
            config,
            graph_store: None,
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a new scar store with write-through persistence.
    pub fn with_graph_store(
        config: ScarConfig,
        graph_store: Arc<dyn obrain_core::graph::GraphStoreMut>,
    ) -> Self {
        Self {
            scars: DashMap::new(),
            index: DashMap::new(),
            config,
            graph_store: Some(graph_store),
            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a scar store backed by a substrate column view (T6).
    ///
    /// Every intensity change writes through to the 5-bit `scar` sub-field of
    /// `NodeRecord.scar_util_affinity`; `utility` and `affinity` bits are
    /// preserved via read-modify-write under the zone lock (see
    /// [`obrain_substrate::writer::Writer::update_scar_field`]).
    ///
    /// **Fidelity**: individual `Scar` records (reason, timestamps, healed_at)
    /// are in-memory only — identical to the legacy `load_from_graph` which
    /// also only persists aggregate intensity/count. The 5-bit quantization
    /// clamps cumulative intensity to `[0, SCAR_MAX_INTENSITY_Q5]` (4.0 by
    /// default).
    #[cfg(feature = "substrate")]
    pub fn with_substrate(
        config: ScarConfig,
        substrate: Arc<obrain_substrate::SubstrateStore>,
    ) -> Self {
        Self {
            scars: DashMap::new(),
            index: DashMap::new(),
            config,
            graph_store: None,
            substrate: Some(substrate),
        }
    }

    /// Returns `true` if this store routes through a substrate column view.
    #[cfg(feature = "substrate")]
    pub fn is_substrate_backed(&self) -> bool {
        self.substrate.is_some()
    }

    /// Computes the cumulative active scar intensity for a target node.
    fn current_cumulative_intensity(&self, target: NodeId) -> f64 {
        let min = self.config.min_intensity;
        self.scars.get(&target).map_or(0.0, |scars| {
            scars
                .iter()
                .filter(|s| s.is_active(min))
                .map(|s| s.current_intensity())
                .sum()
        })
    }

    /// Persists the scar summary (count + cumulative intensity) for a node.
    fn persist_scar_summary(&self, target: NodeId) {
        // Substrate path: write-through quantized intensity to the 5-bit
        // scar sub-field. Utility / affinity bits are preserved by the
        // writer's read-modify-write primitive.
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            let intensity = self.current_cumulative_intensity(target) as f32;
            let _ = sub.set_node_scar_field_f32(target, intensity);
            return;
        }

        if let Some(gs) = &self.graph_store {
            let min = self.config.min_intensity;
            if let Some(scars) = self.scars.get(&target) {
                let count = scars.iter().filter(|s| s.is_active(min)).count() as f64;
                let intensity: f64 = scars
                    .iter()
                    .filter(|s| s.is_active(min))
                    .map(|s| s.current_intensity())
                    .sum();
                persist_node_f64(gs.as_ref(), target, PROP_SCAR_COUNT, count);
                persist_node_f64(gs.as_ref(), target, PROP_SCAR_INTENSITY, intensity);
            }
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
        // Write-through
        self.persist_scar_summary(target);
        id
    }

    /// Heals a scar by ID.
    pub fn heal(&self, scar_id: ScarId) -> bool {
        if let Some(node_id) = self.index.get(&scar_id)
            && let Some(mut scars) = self.scars.get_mut(&*node_id)
            && let Some(scar) = scars.iter_mut().find(|s| s.id == scar_id)
        {
            scar.heal();
            let target = *node_id;
            drop(scars);
            drop(node_id);
            self.persist_scar_summary(target);
            return true;
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
    ///
    /// **Substrate mode**: reads the 5-bit sub-field and dequantizes to the
    /// `[0, SCAR_MAX_INTENSITY_Q5]` range. The hot cache (if populated) is
    /// NOT consulted because the column is authoritative.
    pub fn cumulative_intensity(&self, node_id: NodeId) -> f64 {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            return sub
                .get_node_scar_field_f32(node_id)
                .ok()
                .flatten()
                .map(|v| v as f64)
                .unwrap_or(0.0);
        }
        self.current_cumulative_intensity(node_id)
    }

    /// Returns all nodes that have active scars above `min_intensity`.
    ///
    /// **Substrate mode**: iterates the authoritative column and returns
    /// every live node whose dequantized scar value is `> min_intensity`.
    pub fn nodes_with_active_scars(&self, min_intensity: f64) -> Vec<(NodeId, f64)> {
        #[cfg(feature = "substrate")]
        if let Some(sub) = &self.substrate {
            return match sub.iter_live_scar_util_affinity() {
                Ok(pairs) => pairs
                    .into_iter()
                    .filter_map(|(id, p)| {
                        let v = obrain_substrate::q5_to_scar(p.scar) as f64;
                        (v > min_intensity).then_some((id, v))
                    })
                    .collect(),
                Err(_) => Vec::new(),
            };
        }

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

    /// Rehydrates the scar cache from the backing graph store.
    ///
    /// **Substrate mode**: this is a no-op returning `0`. The 5-bit scar
    /// sub-field on `NodeRecord.scar_util_affinity` is the source of truth
    /// and is already crash-consistent on open. Callers that need the
    /// cumulative intensity go through [`Self::cumulative_intensity`] which
    /// reads the column directly.
    ///
    /// **Legacy mode**: only aggregate metrics (`PROP_SCAR_COUNT`,
    /// `PROP_SCAR_INTENSITY`) are persisted — individual scar details
    /// (reason, timestamps) are lost across restarts. To preserve
    /// `total_scars()`, `active_scar_count()`, and `cumulative_intensity()`
    /// after reload, this method reconstructs `count` synthetic scars per
    /// node, each carrying `intensity/count` and a
    /// `ScarReason::Custom("restored")` tag.
    ///
    /// Returns the total number of synthetic scars loaded.
    pub fn load_from_graph(&self) -> usize {
        #[cfg(feature = "substrate")]
        if self.substrate.is_some() {
            tracing::trace!("scar::load_from_graph: no-op (state is column-resident)");
            return 0;
        }

        let Some(gs) = self.graph_store.as_ref() else {
            return 0;
        };
        let mut loaded = 0usize;
        for nid in gs.node_ids() {
            if self.scars.contains_key(&nid) {
                continue;
            }
            let Some(count) = load_node_f64(gs.as_ref(), nid, PROP_SCAR_COUNT) else {
                continue;
            };
            let count = count as usize;
            if count == 0 {
                continue;
            }
            let total_intensity =
                load_node_f64(gs.as_ref(), nid, PROP_SCAR_INTENSITY).unwrap_or(0.0);
            if total_intensity <= 0.0 {
                continue;
            }
            let per_scar = total_intensity / count as f64;
            let mut synthetic = Vec::with_capacity(count);
            for _ in 0..count {
                let scar = Scar::new(
                    nid,
                    per_scar,
                    ScarReason::Custom("restored".to_string()),
                    self.config.default_half_life,
                );
                self.index.insert(scar.id, nid);
                synthetic.push(scar);
            }
            loaded += synthetic.len();
            self.scars.insert(nid, synthetic);
        }
        loaded
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

    /// Returns the cumulative active scar intensity for a node, or `0.0` if
    /// no scars exist.
    ///
    /// This is a convenience wrapper around [`cumulative_intensity`](Self::cumulative_intensity)
    /// that the hub layer uses for quick risk checks.
    pub fn get_scar_intensity(&self, node_id: NodeId) -> f64 {
        self.cumulative_intensity(node_id)
    }

    /// Reduces the intensity of all active scars on `node_id` by `amount`.
    ///
    /// Each scar's raw intensity is decreased (clamped to `0.0`). Scars whose
    /// intensity drops to zero are effectively healed. The scar summary is
    /// persisted after the operation.
    pub fn partial_heal(&self, node_id: NodeId, amount: f64) {
        if let Some(mut scars) = self.scars.get_mut(&node_id) {
            for scar in scars.iter_mut() {
                if !scar.is_healed() {
                    scar.intensity = (scar.intensity - amount).max(0.0);
                }
            }
        }
        self.persist_scar_summary(node_id);
    }

    /// Increases the intensity of the most recent active scar on `node_id`, or
    /// creates a new scar if none exist.
    ///
    /// The scar's raw intensity is increased by `intensity` (clamped to `1.0`).
    /// If `reason` differs from the existing scar's reason, a new scar is added
    /// instead. The scar summary is persisted after the operation.
    pub fn boost_scar(&self, node_id: NodeId, intensity: f64, reason: ScarReason) {
        let boosted = if let Some(mut scars) = self.scars.get_mut(&node_id) {
            if let Some(scar) = scars
                .iter_mut()
                .rev()
                .find(|s| !s.is_healed() && s.reason == reason)
            {
                scar.intensity = (scar.intensity + intensity).min(1.0);
                true
            } else {
                false
            }
        } else {
            false
        };

        if boosted {
            self.persist_scar_summary(node_id);
        } else {
            self.add_scar(node_id, intensity.min(1.0), reason);
        }
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

// ---------------------------------------------------------------------------
// Substrate-backed integration tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "substrate"))]
mod substrate_tests {
    use super::*;
    use obrain_core::graph::traits::GraphStoreMut;
    use obrain_substrate::SubstrateStore;

    fn make_substrate(n: usize) -> (Arc<SubstrateStore>, Vec<NodeId>, tempfile::TempDir) {
        let td = tempfile::tempdir().unwrap();
        let sub = SubstrateStore::create(td.path().join("kb")).unwrap();
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(sub.create_node(&["n"]));
        }
        sub.flush().unwrap();
        (Arc::new(sub), ids, td)
    }

    #[test]
    fn add_scar_writes_through_column() {
        let (sub, ids, _td) = make_substrate(3);
        let store = ScarStore::with_substrate(ScarConfig::default(), sub.clone());
        assert!(store.is_substrate_backed());
        let nid = ids[0];
        store.add_scar(nid, 1.5, ScarReason::Error("test".into()));
        // Column scar sub-field should quantize 1.5 / 4.0 * 31 ≈ 12.
        let packed = sub.get_node_scar_util_affinity(nid).unwrap().unwrap();
        assert!(
            packed.scar >= 10 && packed.scar <= 14,
            "expected ~12, got {}",
            packed.scar
        );
        assert!(packed.dirty, "dirty must be set by write-through");
    }

    #[test]
    fn cumulative_intensity_reads_from_column() {
        let (sub, ids, _td) = make_substrate(3);
        let store = ScarStore::with_substrate(ScarConfig::default(), sub.clone());
        let nid = ids[1];
        store.add_scar(nid, 2.0, ScarReason::Rollback);
        // Clear the hot cache to force a column read.
        store.scars.clear();
        let v = store.cumulative_intensity(nid);
        // Quantized to 5 bits → max error ≈ 4.0 / 31 ≈ 0.13.
        assert!((v - 2.0).abs() < 0.2, "expected ~2.0, got {v}");
    }

    #[test]
    fn update_preserves_utility_and_affinity_bits() {
        let (sub, ids, _td) = make_substrate(2);
        let store = ScarStore::with_substrate(ScarConfig::default(), sub.clone());
        let nid = ids[0];
        // Pre-seed utility / affinity via direct substrate write.
        sub.set_node_utility_field_f32(nid, 3.0).unwrap();
        sub.set_node_affinity_field_f32(nid, 0.7).unwrap();
        // Now add a scar — it must not clobber the other fields.
        store.add_scar(nid, 1.0, ScarReason::Invalidation);
        let packed = sub.get_node_scar_util_affinity(nid).unwrap().unwrap();
        // utility_to_q5(3.0) = (3/5 * 31).round() = 19
        assert_eq!(packed.utility, 19);
        // affinity_to_q5(0.7) = (0.7 * 31).round() = 22
        assert_eq!(packed.affinity, 22);
        assert!(packed.scar >= 7 && packed.scar <= 9); // 1.0 / 4.0 * 31 ≈ 8
    }

    #[test]
    fn load_from_graph_is_noop_in_substrate_mode() {
        let (sub, _ids, _td) = make_substrate(1);
        let store = ScarStore::with_substrate(ScarConfig::default(), sub.clone());
        let loaded = store.load_from_graph();
        assert_eq!(loaded, 0);
    }

    #[test]
    fn nodes_with_active_scars_iterates_column() {
        let (sub, ids, _td) = make_substrate(5);
        let store = ScarStore::with_substrate(ScarConfig::default(), sub.clone());
        store.add_scar(ids[0], 1.0, ScarReason::Rollback);
        store.add_scar(ids[2], 2.0, ScarReason::Error("e".into()));
        let mut hits = store.nodes_with_active_scars(0.1);
        hits.sort_by_key(|(id, _)| id.0);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, ids[0]);
        assert_eq!(hits[1].0, ids[2]);
    }
}
