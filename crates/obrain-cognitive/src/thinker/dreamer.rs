//! # Dreamer — bottleneck-driven synapse proposals (T13 Step 5)
//!
//! The Dreamer is the creative thinker. Once per interval it:
//!
//! 1. Scans the graph for **bottleneck nodes** — nodes with low Ricci
//!    curvature on their incident edges, i.e. frontier nodes between
//!    communities (T12 primitives).
//! 2. For each bottleneck node, selects two adjacent communities and
//!    proposes a cross-community `SYNAPSE` edge between their top-K
//!    high-centrality representatives, ranked by embedding geodesic
//!    distance (close in semantic space → good candidates).
//! 3. Registers the proposal in a bounded queue; a human-reviewable
//!    audit trail is kept through `tracing::info!` events. No edges
//!    are created without confirmation in the default conservative
//!    mode.
//!
//! Step 5 ships the Thinker trait surface and the proposal queue; the
//! Ricci / geodesic calls are threaded through the `CurvatureProvider`
//! interface so the Thinker stays testable in isolation (stub provider
//! returns empty bottleneck list → tick is a cheap no-op).

use std::sync::Arc;
use std::time::Duration;

use obrain_substrate::SubstrateStore;
use parking_lot::Mutex;

use super::{Thinker, ThinkerBudget, ThinkerKind, ThinkerTickError, ThinkerTickReport};

/// A single cross-community synapse proposal. Conservative default is
/// to register-only; a higher-privilege operator can materialise the
/// proposal via an explicit CLI or admin API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynapseProposal {
    pub src_node_id: u64,
    pub dst_node_id: u64,
    pub src_community_id: u32,
    pub dst_community_id: u32,
    /// Suggested initial weight in Q1.15 u16 (e.g. `0.1` → 3276).
    pub suggested_weight_u16: u16,
}

/// Configuration for the [`Dreamer`].
#[derive(Debug, Clone)]
pub struct DreamerConfig {
    /// Maximum proposals per tick — hard cap to bound the queue.
    pub max_proposals_per_tick: usize,
    /// Soft cap on retained proposals. Older ones are dropped when the
    /// queue overflows.
    pub proposal_queue_size: usize,
    /// Upper bound on Ricci curvature below which a node is considered
    /// a bottleneck candidate. Stored as Q-signed i16 (stride 1/127.5).
    pub bottleneck_ricci_max_q: i16,
    pub interval: Duration,
    pub budget: ThinkerBudget,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            max_proposals_per_tick: 4,
            proposal_queue_size: 256,
            bottleneck_ricci_max_q: -32, // ≈ -0.25 after stride 1/127.5
            interval: Duration::from_secs(600),
            budget: ThinkerBudget::new(0.10, Duration::from_secs(1)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DreamerStats {
    pub ticks: u64,
    pub bottlenecks_scanned: u64,
    pub proposals_emitted: u64,
    pub queue_depth: u64,
    pub last_tick_elapsed_ms: u64,
}

/// Cross-community synapse proposer.
pub struct Dreamer {
    config: DreamerConfig,
    queue: Mutex<std::collections::VecDeque<SynapseProposal>>,
    stats: Mutex<DreamerStats>,
}

impl Dreamer {
    pub fn new(config: DreamerConfig) -> Self {
        let qsz = config.proposal_queue_size;
        Self {
            config,
            queue: Mutex::new(std::collections::VecDeque::with_capacity(qsz)),
            stats: Mutex::new(DreamerStats::default()),
        }
    }

    pub fn stats(&self) -> DreamerStats {
        let s = *self.stats.lock();
        DreamerStats {
            queue_depth: self.queue.lock().len() as u64,
            ..s
        }
    }

    /// Drain the queue of pending proposals (consumes them). Used by
    /// admin surfaces to review/materialise dreams.
    pub fn drain_proposals(&self) -> Vec<SynapseProposal> {
        let mut q = self.queue.lock();
        q.drain(..).collect()
    }

    /// Push a proposal (internal; exposed for tests).
    pub(crate) fn push_proposal(&self, p: SynapseProposal) {
        let mut q = self.queue.lock();
        if q.len() >= self.config.proposal_queue_size {
            q.pop_front();
        }
        q.push_back(p);
    }
}

impl Thinker for Dreamer {
    fn kind(&self) -> ThinkerKind {
        ThinkerKind::Dreamer
    }
    fn budget(&self) -> ThinkerBudget {
        self.config.budget
    }
    fn interval(&self) -> Duration {
        self.config.interval
    }
    fn tick(&self, _store: &Arc<SubstrateStore>) -> Result<ThinkerTickReport, ThinkerTickError> {
        let mut r = ThinkerTickReport::start();

        // Stub tick: the real bottleneck detection + proposal logic
        // lives behind the T12 primitives and requires a curvature
        // provider; until the runtime plumbing is wired (T13 S5
        // follow-up), the tick simply increments the counter.
        //
        // TODO(T13-S5): run `bottleneck_nodes_from_store(store)` + for
        //               each bottleneck, pick 2 adjacent communities and
        //               propose one synapse between their centroids.
        let mut s = self.stats.lock();
        s.ticks += 1;

        r.side_counter = 0;
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
    fn tick_increments_and_returns_elapsed() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("d-test")).unwrap());
        let d = Dreamer::new(DreamerConfig::default());
        let _ = d.tick(&store).expect("tick ok");
        let _ = d.tick(&store).expect("tick ok");
        let s = d.stats();
        assert_eq!(s.ticks, 2);
    }

    #[test]
    fn queue_wraps_on_overflow() {
        let mut cfg = DreamerConfig::default();
        cfg.proposal_queue_size = 4;
        let d = Dreamer::new(cfg);
        for i in 0..10u64 {
            d.push_proposal(SynapseProposal {
                src_node_id: i,
                dst_node_id: i + 100,
                src_community_id: 1,
                dst_community_id: 2,
                suggested_weight_u16: 3276,
            });
        }
        assert_eq!(d.queue.lock().len(), 4);
        let drained = d.drain_proposals();
        // Oldest evicted, last 4 kept.
        assert_eq!(drained.len(), 4);
        assert_eq!(drained[0].src_node_id, 6);
        assert_eq!(drained[3].src_node_id, 9);
    }
}
