//! # HilbertEnricher — canonical `_hilbert_features` production (T17l)
//!
//! Background thinker that computes the 64-72d Hilbert topology-derived
//! features ([`hilbert_features`][1]) for nodes lacking them, and
//! persists the result as the canonical `_hilbert_features` property
//! on the substrate. This is the **first phase** of the canonical
//! cognitive feature chain :
//!
//! ```text
//!   HilbertEnricher → _hilbert_features (64-72d)
//!         │
//!         ▼
//!   KernelEnricher  → _kernel_embedding (80d, depends on Hilbert)
//!         │
//!         ▼
//!   StEnricher      → _st_embedding (384d, ONNX MiniLM, text-based)
//!
//!   CompositeEmbedding pipeline (obrain-chat::retrieval) fuses the
//!   three into a ~532d composite vector for cosine retrieval.
//! ```
//!
//! ## Semantics
//!
//! - **Idempotent** : tick first probes coverage ; if ≥ threshold it
//!   returns a no-op report. Re-running on an already-enriched store
//!   is cheap and safe.
//! - **Crash-safe** : every `set_node_property` flows through the
//!   substrate WAL path — partial progress is persisted on crash.
//! - **Bounded** : single tick runs the full graph Hilbert computation
//!   then writes ; large bases take minutes per tick. Runtime respects
//!   `budget.max_tick_duration` via the over-budget counter (no hard
//!   kill — the algorithm yields naturally after `hilbert_features`
//!   returns).
//!
//! ## Dependency chain
//!
//! This module lives under `thinker/` but is gated by the `enrichment`
//! feature (see `Cargo.toml`). `enrichment` pulls in `obrain-adapters`
//! with the `algos` sub-feature to access [`hilbert_features`][1].
//!
//! [1]: obrain_adapters::plugins::algorithms::hilbert_features

use std::sync::Arc;
use std::time::Duration;

use obrain_adapters::plugins::algorithms::hilbert_features::{
    HilbertFeaturesConfig, hilbert_features,
};
use obrain_common::types::{PropertyKey, Value};
use obrain_core::graph::{GraphStore, GraphStoreMut};
use obrain_substrate::SubstrateStore;

use super::{
    Thinker, ThinkerBudget, ThinkerKind, ThinkerTickError, ThinkerTickReport,
};

const HILBERT_KEY: &str = "_hilbert_features";

/// Coverage threshold below which the enrichment tick actually runs.
/// `0.10` means : if less than 10% of nodes already have
/// `_hilbert_features`, recompute. Above that, treat as done.
pub const DEFAULT_COVERAGE_THRESHOLD: f64 = 0.10;

/// Sample size for the coverage probe. Scanning the whole graph just to
/// check coverage would itself take seconds on large bases ; a 1000-node
/// sample is statistically sufficient to distinguish "not populated"
/// from "populated".
const COVERAGE_SAMPLE_SIZE: usize = 1000;

/// Runtime configuration for [`HilbertEnricher`]. Exposed directly —
/// the `ThinkerFleetConfig<HilbertEnricherTomlCfg>::inner()` impl in
/// `config.rs` materialises this from the TOML overrides.
#[derive(Debug, Clone)]
pub struct HilbertEnricherConfig {
    /// Per-tick wall-clock + CPU budget. Full-graph Hilbert can take
    /// seconds to minutes ; the tick is NOT canceled mid-computation
    /// (that would leave state inconsistent), but over-budget ticks are
    /// counted in `ThinkerRuntimeStats.over_budget_ticks`.
    pub budget: ThinkerBudget,
    /// Sleep between ticks. Default 600s so the enricher is low-impact
    /// on a warm base ; the coverage probe fast-exits when the store is
    /// already enriched.
    pub interval: Duration,
    /// Coverage threshold (fraction in `[0.0, 1.0]`). See
    /// [`DEFAULT_COVERAGE_THRESHOLD`].
    pub coverage_threshold: f64,
    /// Tuning for the underlying Hilbert algorithm. `Default` keeps
    /// 8 levels × 8 base facettes = 64 dims.
    pub hilbert: HilbertFeaturesConfig,
}

impl Default for HilbertEnricherConfig {
    fn default() -> Self {
        Self {
            budget: ThinkerBudget::new(0.10, Duration::from_secs(300)),
            interval: Duration::from_secs(600),
            coverage_threshold: DEFAULT_COVERAGE_THRESHOLD,
            hilbert: HilbertFeaturesConfig::default(),
        }
    }
}

/// Accumulated stats since the thinker started — exposed through the
/// runtime's `/metrics` introspection.
#[derive(Debug, Clone, Copy, Default)]
pub struct HilbertEnricherStats {
    pub ticks: u64,
    pub ticks_skipped_coverage: u64,
    pub nodes_written: u64,
    pub last_coverage: f64,
    pub last_tick_elapsed_ms: u64,
}

/// Canonical `_hilbert_features` producer thinker.
pub struct HilbertEnricher {
    config: HilbertEnricherConfig,
    stats: parking_lot::Mutex<HilbertEnricherStats>,
}

impl HilbertEnricher {
    pub fn new(config: HilbertEnricherConfig) -> Self {
        Self {
            config,
            stats: parking_lot::Mutex::new(HilbertEnricherStats::default()),
        }
    }

    pub fn stats(&self) -> HilbertEnricherStats {
        *self.stats.lock()
    }

    /// Probe the sample coverage on `_hilbert_features`. Returns the
    /// fraction of sampled nodes that carry a non-empty vector property.
    fn probe_coverage(store: &Arc<SubstrateStore>) -> (f64, usize) {
        let node_ids = store.node_ids();
        if node_ids.is_empty() {
            return (1.0, 0);
        }
        let sample_n = node_ids.len().min(COVERAGE_SAMPLE_SIZE);
        let key = PropertyKey::from(HILBERT_KEY);
        let mut hits = 0usize;
        for nid in node_ids.iter().take(sample_n) {
            if let Some(Value::Vector(_)) = store.get_node_property(*nid, &key) {
                hits += 1;
            }
        }
        ((hits as f64) / (sample_n as f64), sample_n)
    }

    fn tick_impl(
        &self,
        store: &Arc<SubstrateStore>,
    ) -> Result<ThinkerTickReport, ThinkerTickError> {
        let mut report = ThinkerTickReport::start();

        let (coverage, sample_n) = Self::probe_coverage(store);
        {
            let mut s = self.stats.lock();
            s.ticks += 1;
            s.last_coverage = coverage;
        }

        if sample_n == 0 {
            tracing::debug!(
                kind = "hilbert_enricher",
                "tick skipped — empty store"
            );
            let mut s = self.stats.lock();
            s.ticks_skipped_coverage += 1;
            return Ok(report.finish());
        }

        if coverage >= self.config.coverage_threshold {
            tracing::debug!(
                kind = "hilbert_enricher",
                coverage = coverage,
                threshold = self.config.coverage_threshold,
                sample = sample_n,
                "tick skipped — coverage already above threshold"
            );
            let mut s = self.stats.lock();
            s.ticks_skipped_coverage += 1;
            return Ok(report.finish());
        }

        tracing::info!(
            kind = "hilbert_enricher",
            coverage = coverage,
            threshold = self.config.coverage_threshold,
            "tick starting — computing canonical _hilbert_features"
        );

        // Full-graph compute + persist. GraphStoreMut is a supertrait of
        // GraphStore (edition-2024 trait upcasting).
        let store_ref: &dyn GraphStore = store.as_ref();
        let result = hilbert_features(store_ref, &self.config.hilbert);

        let dims = result.dimensions;
        let mut nodes_written = 0u64;
        for (nid, features) in result.features.iter() {
            let arc: Arc<[f32]> = features.as_slice().into();
            store.set_node_property(*nid, HILBERT_KEY, Value::Vector(arc));
            nodes_written += 1;
        }

        report.nodes_touched = nodes_written;
        report.side_counter = dims as u64;
        let r = report.finish();
        {
            let mut s = self.stats.lock();
            s.nodes_written += nodes_written;
            s.last_tick_elapsed_ms = r.elapsed.as_millis() as u64;
        }

        tracing::info!(
            kind = "hilbert_enricher",
            nodes_written = nodes_written,
            dims = dims,
            elapsed_ms = r.elapsed.as_millis() as u64,
            "tick complete — _hilbert_features populated"
        );

        Ok(r)
    }
}

impl Thinker for HilbertEnricher {
    fn kind(&self) -> ThinkerKind {
        ThinkerKind::HilbertEnricher
    }
    fn budget(&self) -> ThinkerBudget {
        self.config.budget
    }
    fn interval(&self) -> Duration {
        self.config.interval
    }
    fn tick(
        &self,
        store: &Arc<SubstrateStore>,
    ) -> Result<ThinkerTickReport, ThinkerTickError> {
        self.tick_impl(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_store() -> Arc<SubstrateStore> {
        Arc::new(SubstrateStore::open_tempfile().expect("open tempfile"))
    }

    #[test]
    fn tick_on_empty_store_is_noop() {
        let store = fresh_store();
        let e = HilbertEnricher::new(HilbertEnricherConfig::default());
        let r = e.tick(&store).expect("tick");
        assert_eq!(r.nodes_touched, 0);
        let s = e.stats();
        assert_eq!(s.ticks, 1);
        assert_eq!(s.ticks_skipped_coverage, 1);
        assert_eq!(s.nodes_written, 0);
    }

    #[test]
    fn tick_populates_hilbert_features_on_nonempty_store() {
        let store = fresh_store();
        // Build a small 5-node + 6-edge graph so Hilbert has something
        // meaningful to compute. Any labels + edges are fine.
        let n: Vec<_> = (0..5).map(|_| store.create_node(&["Node"])).collect();
        store.create_edge(n[0], n[1], "E");
        store.create_edge(n[1], n[2], "E");
        store.create_edge(n[2], n[3], "E");
        store.create_edge(n[3], n[4], "E");
        store.create_edge(n[0], n[2], "E");
        store.create_edge(n[1], n[4], "E");

        let e = HilbertEnricher::new(HilbertEnricherConfig::default());
        let r = e.tick(&store).expect("tick");
        assert!(r.nodes_touched > 0, "at least one node must be written");

        // Probe each node : every node should now have a non-empty
        // `_hilbert_features` Vector property.
        let key = PropertyKey::from(HILBERT_KEY);
        for nid in &n {
            match store.get_node_property(*nid, &key) {
                Some(Value::Vector(v)) => assert!(!v.is_empty()),
                other => panic!("node {nid:?} missing _hilbert_features, got {other:?}"),
            }
        }

        let s = e.stats();
        assert_eq!(s.ticks, 1);
        assert_eq!(s.ticks_skipped_coverage, 0);
        assert!(s.nodes_written > 0);
        // `last_tick_elapsed_ms` can round to 0 on a tiny 5-node graph
        // where the tick finishes in a few microseconds — don't assert
        // a strict > 0 here, just that the field was populated (exists).
        let _ = s.last_tick_elapsed_ms;
    }

    #[test]
    fn second_tick_is_skipped_when_coverage_already_high() {
        let store = fresh_store();
        let n: Vec<_> = (0..3).map(|_| store.create_node(&["Node"])).collect();
        store.create_edge(n[0], n[1], "E");
        store.create_edge(n[1], n[2], "E");

        let e = HilbertEnricher::new(HilbertEnricherConfig::default());
        let _r1 = e.tick(&store).expect("tick 1");
        let r2 = e.tick(&store).expect("tick 2");
        // Second tick should skip because coverage ≥ threshold now.
        assert_eq!(
            r2.nodes_touched, 0,
            "second tick should be a skip (coverage high)"
        );
        let s = e.stats();
        assert_eq!(s.ticks, 2);
        assert_eq!(s.ticks_skipped_coverage, 1, "only the 2nd tick skipped");
    }
}
