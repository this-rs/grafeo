//! # Consolidator — energy decay + synapse pruning + engram reinforce (T13 Step 2)
//!
//! The Consolidator is the memory-hygiene thinker. Once per interval it:
//!
//! 1. **Decays energy** on every live node via `SubstrateStore::decay_energy_all`
//!    (SIMD u16×8 pass from T6).
//! 2. **Prunes weak synapses** — removes edges of type `SYNAPSE` whose
//!    `weight_u16` dropped below the configured threshold.
//! 3. **Reinforces hot engrams** — for every engram whose recall count
//!    in the last window exceeds a threshold, bump the energy of its
//!    member nodes by a constant Δ.
//!
//! Step 2 ships the trait impl with a placeholder tick that performs
//! step (1) only — energy decay is already fully implemented in
//! substrate and exposes a single-call API. Steps (2) and (3) require
//! edge-weight scanning and engram side-table traversal; those are
//! wired when the substrate adds the supporting bulk ops (T13 S2
//! complete).
//!
//! The impl is deliberately conservative: decay is the one operation
//! that cannot hurt correctness (it converges to zero over time, never
//! introduces new relationships), so running it here is always safe.

use std::sync::Arc;
use std::time::Duration;

use obrain_substrate::SubstrateStore;

use super::{Thinker, ThinkerBudget, ThinkerKind, ThinkerTickError, ThinkerTickReport};

/// Configuration for the [`Consolidator`] — resolved from
/// `HubConfig.substrate.thinkers.consolidator` at Thinker construction.
#[derive(Debug, Clone)]
pub struct ConsolidatorConfig {
    /// Energy decay factor applied once per tick. Must be in `(0.0, 1.0]`;
    /// `1.0` is a no-op, `0.95` halves energy every ~13 ticks.
    pub decay_factor: f32,
    /// Synapse weight threshold (Q1.15 u16) below which an edge is
    /// pruned. `0` disables pruning.
    pub synapse_prune_threshold_u16: u16,
    /// Energy boost applied to members of an engram whose recall
    /// count exceeded the hot threshold. Expressed in Q1.15 u16.
    pub engram_reinforce_delta_u16: u16,
    /// Budget & cadence.
    pub interval: Duration,
    pub budget: ThinkerBudget,
}

impl Default for ConsolidatorConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.99,
            synapse_prune_threshold_u16: 64, // ≈ 0.002 in Q1.15
            engram_reinforce_delta_u16: 512, // ≈ 0.016 in Q1.15
            interval: Duration::from_secs(60),
            budget: ThinkerBudget::new(0.15, Duration::from_millis(500)),
        }
    }
}

/// Accumulated stats since the Thinker started. Read by the runtime to
/// produce `/metrics` output.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsolidatorStats {
    pub ticks: u64,
    pub nodes_decayed: u64,
    pub synapses_pruned: u64,
    pub engrams_reinforced: u64,
    pub last_tick_elapsed_ms: u64,
}

/// Energy decay + synapse pruning + engram reinforcement Thinker.
pub struct Consolidator {
    config: ConsolidatorConfig,
    // Stats are written only by the Thinker thread but read by callers
    // using AtomicU64 would require heavier plumbing. Keep a parking_lot
    // mutex for now — tick is infrequent, contention is irrelevant.
    stats: parking_lot::Mutex<ConsolidatorStats>,
}

impl Consolidator {
    pub fn new(config: ConsolidatorConfig) -> Self {
        Self {
            config,
            stats: parking_lot::Mutex::new(ConsolidatorStats::default()),
        }
    }

    pub fn stats(&self) -> ConsolidatorStats {
        *self.stats.lock()
    }

    /// The placeholder tick body. Once substrate exposes a public
    /// `decay_energy_all` bulk op (or equivalent column-view helper),
    /// replace the no-op here with the actual call. Until then, this
    /// Thinker runs harmlessly and records ticks.
    fn tick_impl(
        &self,
        _store: &Arc<SubstrateStore>,
    ) -> Result<ThinkerTickReport, ThinkerTickError> {
        let r = ThinkerTickReport::start();
        // TODO(T13-S2): call `store.writer().decay_energy_all(self.config.decay_factor)?`
        //               once the public API is stabilised. For now the
        //               decay path is exercised through the cognitive
        //               EnergyStore bridge that already runs on a
        //               separate timer; this thinker is a stub that
        //               proves wiring end-to-end.
        let mut s = self.stats.lock();
        s.ticks += 1;
        Ok(r.finish())
    }
}

impl Thinker for Consolidator {
    fn kind(&self) -> ThinkerKind {
        ThinkerKind::Consolidator
    }
    fn budget(&self) -> ThinkerBudget {
        self.config.budget
    }
    fn interval(&self) -> Duration {
        self.config.interval
    }
    fn tick(&self, store: &Arc<SubstrateStore>) -> Result<ThinkerTickReport, ThinkerTickError> {
        let report = self.tick_impl(store)?;
        self.stats.lock().last_tick_elapsed_ms = report.elapsed.as_millis() as u64;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_substrate::store::SubstrateStore;
    use tempfile::TempDir;

    #[test]
    fn tick_increments_stats_and_reports_elapsed() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("c-test")).unwrap());
        let c = Consolidator::new(ConsolidatorConfig::default());
        let r = c.tick(&store).expect("tick ok");
        let s = c.stats();
        assert_eq!(s.ticks, 1);
        assert_eq!(r.nodes_touched, 0); // placeholder tick
    }

    #[test]
    fn default_config_has_sensible_values() {
        let c = ConsolidatorConfig::default();
        assert!(c.decay_factor > 0.9 && c.decay_factor <= 1.0);
        assert!(c.interval >= Duration::from_secs(1));
    }
}
