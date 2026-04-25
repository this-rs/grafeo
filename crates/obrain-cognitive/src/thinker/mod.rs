//! # Thinker — self-maintaining background agents (T13)
//!
//! The substrate is designed to maintain itself: rather than relying on
//! external cron-like loops or explicit maintenance calls, each
//! [`SubstrateStore`](obrain_substrate::SubstrateStore) spawns a set of
//! **Thinker** threads at open time. Each Thinker runs a small, bounded
//! amount of work at a fixed interval, and is responsible for one
//! cognitive invariant of the graph.
//!
//! This module defines the trait contract and the scheduling primitives;
//! individual thinkers live in sibling files:
//!
//! | Thinker              | Responsibility                                              | Spec step |
//! |----------------------|-------------------------------------------------------------|-----------|
//! | [`Consolidator`]     | energy decay + synapse pruning + engram reinforcement       | T13 S2    |
//! | [`CommunityWardenThinker`] | LDleiden online update + Hilbert page compaction      | T13 S3    |
//! | [`Predictor`]        | topic tracking + `madvise(WILLNEED)` on likely communities  | T13 S4    |
//! | [`Dreamer`]          | bottleneck-driven cross-community synapse proposals         | T13 S5    |
//!
//! ## Design
//!
//! - **Trait-level contract** ([`Thinker`]): every Thinker exposes a
//!   cheap `tick` that runs one bounded pass, a `budget` describing the
//!   CPU envelope it is willing to spend, an `interval` between ticks,
//!   and a `kind` for diagnostics.
//! - **Scheduler** ([`ThinkerRuntime`]): owns one OS thread per Thinker,
//!   sleeps for the declared interval, runs `tick`, re-sleeps. Pressure
//!   pause (T13 S8) is implemented by the runtime — thinkers stay
//!   unaware of global load.
//! - **Shutdown** ([`ThinkerRuntime::shutdown`]): sends a cooperative
//!   stop signal and joins every thread with a bounded timeout.
//!
//! ## Budget
//!
//! Budgets are expressed as a CPU percentage (0.0..=1.0) and a max
//! wall-clock duration per tick. The scheduler interprets them as an
//! **upper bound** — the thinker itself is responsible for staying
//! within the allotted time and returning early via its own
//! cooperative checkpointing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use obrain_substrate::SubstrateStore;

pub mod config;
pub mod consolidator;
pub mod dreamer;
#[cfg(feature = "enrichment")]
pub mod hilbert_enricher;
pub mod predictor;
pub mod runtime;
pub mod warden;

pub use config::{ThinkerFleetConfig, ThinkersConfig};
pub use consolidator::{Consolidator, ConsolidatorConfig, ConsolidatorStats};
pub use dreamer::{Dreamer, DreamerConfig, DreamerStats};
#[cfg(feature = "enrichment")]
pub use hilbert_enricher::{HilbertEnricher, HilbertEnricherConfig, HilbertEnricherStats};
pub use predictor::{Predictor, PredictorConfig, PredictorStats};
pub use runtime::{
    NeverOverloadedSensor, PressureSensor, ThinkerHandle, ThinkerRuntime, ThinkerRuntimeConfig,
    ThinkerRuntimeStats,
};
pub use warden::{CommunityWardenThinker, WardenConfig, WardenStats};

/// Build a [`ThinkerRuntime`] wired with the full standard fleet
/// (Consolidator + CommunityWarden + Predictor + Dreamer). Respects the
/// `enabled` flags in the provided [`ThinkersConfig`].
///
/// This is the single entry point used by `SubstrateStore::open()` (or
/// an equivalent higher-level facade) to auto-start thinkers at boot.
/// Callers that want a silent store (tests, migrations) simply skip
/// calling this function — there is no ambient auto-start behaviour.
pub fn spawn_standard_fleet(
    store: std::sync::Arc<obrain_substrate::SubstrateStore>,
    cfg: &ThinkersConfig,
    sensor: std::sync::Arc<dyn PressureSensor>,
) -> ThinkerRuntime {
    let mut rt = ThinkerRuntime::new(store, cfg.runtime.clone(), sensor);
    if cfg.consolidator.enabled {
        rt.spawn(Consolidator::new(cfg.consolidator.inner()));
    }
    if cfg.warden.enabled {
        rt.spawn(CommunityWardenThinker::new(cfg.warden.inner()));
    }
    if cfg.predictor.enabled {
        let ring = std::sync::Arc::new(predictor::TopicRing::with_capacity(
            cfg.predictor_topic_ring_capacity,
        ));
        rt.spawn(Predictor::new(cfg.predictor.inner(), ring));
    }
    if cfg.dreamer.enabled {
        rt.spawn(Dreamer::new(cfg.dreamer.inner()));
    }
    // T17l — HilbertEnricher (canonical `_hilbert_features` 64-72d)
    // only compiled when the `enrichment` feature is active. This
    // pulls obrain-adapters with the `algos` sub-feature. Binary
    // callers (hub, user-brain) opt in by adding `enrichment` to
    // their `obrain-cognitive` feature list.
    #[cfg(feature = "enrichment")]
    if cfg.hilbert_enricher.enabled {
        rt.spawn(hilbert_enricher::HilbertEnricher::new(
            cfg.hilbert_enricher.inner(),
        ));
    }
    rt
}

/// Diagnostic tag identifying a Thinker kind (used for thread names,
/// metrics counters, pressure pause keying, and config lookups).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThinkerKind {
    Consolidator,
    CommunityWarden,
    Predictor,
    Dreamer,
    /// T17l — Computes canonical `_hilbert_features` (64-72d) via
    /// `obrain-adapters::algorithms::hilbert_features`. Gated by the
    /// `enrichment` feature. First stage of the canonical retrieval
    /// composite (feeds KernelEnricher + StEnricher).
    HilbertEnricher,
}

impl ThinkerKind {
    /// Canonical lowercase name — used as the OS thread name prefix and
    /// the key under `[substrate.thinkers.<name>]` in `HubConfig`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkerKind::Consolidator => "consolidator",
            ThinkerKind::CommunityWarden => "warden",
            ThinkerKind::Predictor => "predictor",
            ThinkerKind::Dreamer => "dreamer",
            ThinkerKind::HilbertEnricher => "hilbert_enricher",
        }
    }
}

/// CPU / latency envelope a Thinker is willing to honour.
///
/// `cpu_fraction` is the fraction of one core the thinker is allowed to
/// spend on average (e.g. `0.25` = 25% of one core). It is enforced
/// cooperatively by the Thinker itself — the runtime does not pin, nor
/// throttle. In practice most thinkers stay well under their budget; the
/// field is used to advertise intent and drive the aggregate CPU cap
/// (T13 S7).
///
/// `max_tick_duration` is the hard wall-clock cap per tick. A thinker
/// that does not return before `max_tick_duration` is flagged in
/// [`ThinkerRuntimeStats::over_budget_ticks`]; no automatic cancellation
/// is attempted — the workload is expected to reach a natural yield point.
#[derive(Debug, Clone, Copy)]
pub struct ThinkerBudget {
    pub cpu_fraction: f32,
    pub max_tick_duration: Duration,
}

impl ThinkerBudget {
    pub const fn new(cpu_fraction: f32, max_tick_duration: Duration) -> Self {
        Self {
            cpu_fraction,
            max_tick_duration,
        }
    }

    /// Minimal budget — used by stubs and tests. 1% of one core,
    /// 10 ms per tick.
    pub const fn minimal() -> Self {
        Self {
            cpu_fraction: 0.01,
            max_tick_duration: Duration::from_millis(10),
        }
    }
}

/// Contract every Thinker implements. A Thinker is a long-lived
/// maintenance agent that runs [`tick`](Thinker::tick) at a fixed
/// interval against a [`SubstrateStore`].
///
/// Ticks must be:
/// - **Idempotent under repetition** — re-running a tick without
///   intervening graph mutations must be a no-op (or a cheap no-op-like
///   scan).
/// - **Crash-safe** — any mutation must go through the substrate's
///   normal write path (WAL-logged).
/// - **Bounded** — respect [`budget`](Thinker::budget); in particular,
///   never hold a read/write lock across an I/O wait.
/// - **Observable** — emit `tracing` events at `debug` level on entry
///   and exit with the elapsed duration and the number of affected
///   nodes/edges.
///
/// The trait is synchronous on purpose: each Thinker owns one OS thread
/// (`std::thread::spawn`) with a simple sleep/tick loop. Async is not
/// used here because (1) the work is CPU-bound, (2) we want the OS
/// scheduler (not Tokio) to preempt under system pressure.
pub trait Thinker: Send + Sync + 'static {
    /// Identifying tag for logs and metrics.
    fn kind(&self) -> ThinkerKind;

    /// CPU / latency envelope. Constant for the lifetime of the
    /// Thinker; changes require a restart.
    fn budget(&self) -> ThinkerBudget;

    /// Sleep duration between consecutive ticks. Must be ≥ 1 s to avoid
    /// pegging a core — the runtime rejects shorter intervals at
    /// registration time.
    fn interval(&self) -> Duration;

    /// Run one bounded maintenance pass. Returning an error is logged
    /// but does not stop the Thinker — the runtime retries at the next
    /// tick. Panics are caught by the runtime and convert into
    /// [`ThinkerTickError::Panicked`].
    fn tick(&self, store: &Arc<SubstrateStore>) -> Result<ThinkerTickReport, ThinkerTickError>;
}

/// Diagnostic summary returned by a single tick. Every Thinker extends
/// this with richer stats via its own side-channel; this struct stays
/// minimal so the runtime can aggregate across heterogeneous thinkers
/// without boxing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkerTickReport {
    pub started_at: Option<Instant>,
    pub elapsed: Duration,
    pub nodes_touched: u64,
    pub edges_touched: u64,
    /// Arbitrary secondary counter — semantics is thinker-specific
    /// (e.g. `Predictor` reports LRU hits, `Dreamer` reports proposed
    /// synapses).
    pub side_counter: u64,
}

impl ThinkerTickReport {
    pub fn start() -> Self {
        Self {
            started_at: Some(Instant::now()),
            elapsed: Duration::ZERO,
            nodes_touched: 0,
            edges_touched: 0,
            side_counter: 0,
        }
    }

    pub fn finish(mut self) -> Self {
        if let Some(s) = self.started_at {
            self.elapsed = s.elapsed();
        }
        self
    }
}

/// Error kinds a tick can return. Every variant is logged at `warn`
/// level by the runtime; only `Panicked` triggers an extra counter so
/// flakiness is easy to spot.
#[derive(Debug, thiserror::Error)]
pub enum ThinkerTickError {
    #[error("substrate error: {0}")]
    Substrate(#[from] obrain_substrate::SubstrateError),

    #[error("store busy — tick skipped")]
    Busy,

    #[error("panic in tick: {0}")]
    Panicked(String),

    #[error("other: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial stub used to verify the trait surface compiles and
    /// the default reporter behaves correctly.
    struct NoopThinker {
        kind: ThinkerKind,
    }

    impl Thinker for NoopThinker {
        fn kind(&self) -> ThinkerKind {
            self.kind
        }
        fn budget(&self) -> ThinkerBudget {
            ThinkerBudget::minimal()
        }
        fn interval(&self) -> Duration {
            Duration::from_secs(1)
        }
        fn tick(
            &self,
            _store: &Arc<SubstrateStore>,
        ) -> Result<ThinkerTickReport, ThinkerTickError> {
            let r = ThinkerTickReport::start();
            Ok(r.finish())
        }
    }

    #[test]
    fn kind_as_str_is_stable() {
        assert_eq!(ThinkerKind::Consolidator.as_str(), "consolidator");
        assert_eq!(ThinkerKind::CommunityWarden.as_str(), "warden");
        assert_eq!(ThinkerKind::Predictor.as_str(), "predictor");
        assert_eq!(ThinkerKind::Dreamer.as_str(), "dreamer");
    }

    #[test]
    fn budget_minimal_constructs() {
        let b = ThinkerBudget::minimal();
        assert!(b.cpu_fraction > 0.0);
        assert!(b.max_tick_duration > Duration::ZERO);
    }

    #[test]
    fn tick_report_records_elapsed() {
        let r = ThinkerTickReport::start();
        std::thread::sleep(Duration::from_millis(1));
        let r = r.finish();
        assert!(r.elapsed >= Duration::from_millis(1));
    }

    #[test]
    fn noop_thinker_trait_object_is_ok() {
        // Compile-check: trait object construction, object safety.
        let n: Box<dyn Thinker> = Box::new(NoopThinker {
            kind: ThinkerKind::Consolidator,
        });
        assert_eq!(n.kind(), ThinkerKind::Consolidator);
        assert_eq!(n.interval(), Duration::from_secs(1));
    }
}
