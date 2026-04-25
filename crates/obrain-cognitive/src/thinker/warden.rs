//! # CommunityWardenThinker — LDleiden + Hilbert compaction (T13 Step 3)
//!
//! Thin wrapper that drives the existing
//! [`obrain_substrate::CommunityWarden`] from T11 on a periodic tick.
//! The heavy lifting (fragmentation scan + transactional per-community
//! compaction) already lives in substrate; this Thinker only provides
//! the cadence and the tracing envelope.
//!
//! The LDleiden online update from T10 is driven by edge mutations
//! directly (MutationListener path). This Thinker's tick therefore
//! focuses exclusively on **fragmentation**: scan the Nodes zone, and
//! compact any community whose page ratio exceeds the trigger.

use std::sync::Arc;
use std::time::Duration;

use obrain_substrate::{CommunityWarden, DEFAULT_FRAGMENTATION_TRIGGER, SubstrateStore};

use super::{Thinker, ThinkerBudget, ThinkerKind, ThinkerTickError, ThinkerTickReport};

/// Configuration for the [`CommunityWardenThinker`].
#[derive(Debug, Clone)]
pub struct WardenConfig {
    /// Fragmentation trigger — see
    /// [`obrain_substrate::CommunityWarden::new_with_trigger`].
    /// Default: `1.30` (30% more pages than perfect packing).
    pub fragmentation_trigger: f32,
    pub interval: Duration,
    pub budget: ThinkerBudget,
}

impl Default for WardenConfig {
    fn default() -> Self {
        Self {
            fragmentation_trigger: DEFAULT_FRAGMENTATION_TRIGGER,
            interval: Duration::from_secs(300),
            budget: ThinkerBudget::new(0.25, Duration::from_secs(5)),
        }
    }
}

/// Stats accumulated by the Thinker. Reset-safe across ticks.
#[derive(Debug, Clone, Copy, Default)]
pub struct WardenStats {
    pub ticks: u64,
    pub communities_compacted: u64,
    pub last_tick_elapsed_ms: u64,
}

/// Fragmentation-scanner Thinker. Wraps
/// [`obrain_substrate::CommunityWarden::tick`].
pub struct CommunityWardenThinker {
    config: WardenConfig,
    stats: parking_lot::Mutex<WardenStats>,
}

impl CommunityWardenThinker {
    pub fn new(config: WardenConfig) -> Self {
        Self {
            config,
            stats: parking_lot::Mutex::new(WardenStats::default()),
        }
    }

    pub fn stats(&self) -> WardenStats {
        *self.stats.lock()
    }
}

impl Thinker for CommunityWardenThinker {
    fn kind(&self) -> ThinkerKind {
        ThinkerKind::CommunityWarden
    }
    fn budget(&self) -> ThinkerBudget {
        self.config.budget
    }
    fn interval(&self) -> Duration {
        self.config.interval
    }
    fn tick(&self, store: &Arc<SubstrateStore>) -> Result<ThinkerTickReport, ThinkerTickError> {
        let mut r = ThinkerTickReport::start();

        // Build a warden on each tick — it's cheap (a shared Arc clone)
        // and avoids stale state if the store is reconfigured. The
        // warden is intentionally stateless beyond its trigger.
        let warden =
            CommunityWarden::with_trigger(store.clone(), self.config.fragmentation_trigger);
        let compacted = warden.tick()?;
        r.side_counter = compacted.len() as u64;

        let mut s = self.stats.lock();
        s.ticks += 1;
        s.communities_compacted += compacted.len() as u64;
        let r = r.finish();
        s.last_tick_elapsed_ms = r.elapsed.as_millis() as u64;
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obrain_substrate::store::SubstrateStore;
    use tempfile::TempDir;

    #[test]
    fn tick_on_empty_store_is_noop() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("w-test")).unwrap());
        let w = CommunityWardenThinker::new(WardenConfig::default());
        let r = w.tick(&store).expect("tick ok");
        assert_eq!(r.side_counter, 0);
        assert_eq!(w.stats().communities_compacted, 0);
    }
}
