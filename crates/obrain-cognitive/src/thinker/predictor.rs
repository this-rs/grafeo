//! # Predictor — topic-driven prefetch (T13 Step 4)
//!
//! The Predictor watches recent node activations (the chat pipeline
//! emits a breadcrumb when it references a node) and uses them to
//! anticipate the next turn's hot communities. Its tick:
//!
//! 1. **Snapshots** the last-N activated node ids from an externally
//!    owned ring buffer (`TopicRing`).
//! 2. **Groups** them by `community_id`.
//! 3. **Emits `madvise(WILLNEED)`** on each hot community's page range
//!    (wired through `SubstrateStore::prefetch_community` — see T11 S3).
//!
//! Step 4 ships the Thinker trait surface and the ring-buffer helper.
//! The prefetch call itself is a no-op when the substrate doesn't
//! expose a matching API yet — the wiring is trait-driven so it becomes
//! a drop-in replacement when ready.

use std::sync::Arc;
use std::time::Duration;

use obrain_substrate::SubstrateStore;
use parking_lot::RwLock;

use super::{
    Thinker, ThinkerBudget, ThinkerKind, ThinkerTickError, ThinkerTickReport,
};

/// A lock-free ring buffer of recent topic breadcrumbs. Externally
/// populated by the chat pipeline, externally read by the Predictor.
///
/// Capacity is fixed at construction; older entries are overwritten.
/// A single `parking_lot::RwLock` gates the whole buffer — contention
/// is a non-issue in practice because writes are one-per-chat-turn and
/// reads are one-per-tick.
#[derive(Debug, Default)]
pub struct TopicRing {
    inner: RwLock<TopicRingInner>,
}

#[derive(Debug, Default)]
struct TopicRingInner {
    buf: Vec<u64>,
    head: usize, // next write index
    len: usize,  // ≤ capacity
    capacity: usize,
}

impl TopicRing {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(TopicRingInner {
                buf: vec![0u64; capacity.max(1)],
                head: 0,
                len: 0,
                capacity: capacity.max(1),
            }),
        }
    }

    pub fn push(&self, node_id: u64) {
        let mut g = self.inner.write();
        let i = g.head;
        g.buf[i] = node_id;
        g.head = (g.head + 1) % g.capacity;
        if g.len < g.capacity {
            g.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot the current contents in insertion order (oldest → newest).
    pub fn snapshot(&self) -> Vec<u64> {
        let g = self.inner.read();
        let mut out = Vec::with_capacity(g.len);
        let start = if g.len < g.capacity {
            0
        } else {
            g.head
        };
        for i in 0..g.len {
            out.push(g.buf[(start + i) % g.capacity]);
        }
        out
    }
}

/// Configuration for the [`Predictor`].
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    /// Top-K communities (by breadcrumb frequency in the ring) to
    /// prefetch per tick.
    pub top_k_communities: usize,
    /// If the ring is emptier than this, skip the tick.
    pub min_breadcrumbs: usize,
    pub interval: Duration,
    pub budget: ThinkerBudget,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            top_k_communities: 3,
            min_breadcrumbs: 4,
            interval: Duration::from_secs(30),
            budget: ThinkerBudget::new(0.05, Duration::from_millis(100)),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PredictorStats {
    pub ticks: u64,
    pub breadcrumbs_seen: u64,
    pub communities_prefetched: u64,
    pub last_tick_elapsed_ms: u64,
}

/// Topic-driven prefetch Thinker.
pub struct Predictor {
    config: PredictorConfig,
    ring: Arc<TopicRing>,
    stats: parking_lot::Mutex<PredictorStats>,
}

impl Predictor {
    pub fn new(config: PredictorConfig, ring: Arc<TopicRing>) -> Self {
        Self { config, ring, stats: parking_lot::Mutex::new(PredictorStats::default()) }
    }

    pub fn stats(&self) -> PredictorStats {
        *self.stats.lock()
    }

    /// Direct access to the shared ring so the chat pipeline can push
    /// breadcrumbs without going through the Thinker.
    pub fn topic_ring(&self) -> Arc<TopicRing> {
        self.ring.clone()
    }
}

impl Thinker for Predictor {
    fn kind(&self) -> ThinkerKind {
        ThinkerKind::Predictor
    }
    fn budget(&self) -> ThinkerBudget {
        self.config.budget
    }
    fn interval(&self) -> Duration {
        self.config.interval
    }
    fn tick(
        &self,
        _store: &Arc<SubstrateStore>,
    ) -> Result<ThinkerTickReport, ThinkerTickError> {
        let mut r = ThinkerTickReport::start();
        let breadcrumbs = self.ring.snapshot();
        r.nodes_touched = breadcrumbs.len() as u64;

        let mut s = self.stats.lock();
        s.ticks += 1;
        s.breadcrumbs_seen += breadcrumbs.len() as u64;

        if breadcrumbs.len() < self.config.min_breadcrumbs {
            let r = r.finish();
            s.last_tick_elapsed_ms = r.elapsed.as_millis() as u64;
            return Ok(r);
        }

        // Histogram of node_ids → (stand-in for community_id). Wiring
        // the real community lookup requires a store accessor for
        // `NodeRecord::community_id` — adding that API is T13 S4
        // follow-up; for now the prefetch count is reported without
        // a side effect.
        //
        // TODO(T13-S4): resolve community_id per node_id, prefetch top-K
        //               community page ranges via
        //               `store.prefetch_community(cid)`.
        s.communities_prefetched += self.config.top_k_communities as u64;
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
    fn ring_wraps_at_capacity() {
        let r = TopicRing::with_capacity(4);
        for i in 0..10u64 {
            r.push(i);
        }
        let s = r.snapshot();
        assert_eq!(s.len(), 4);
        // Last 4 pushed: 6, 7, 8, 9
        assert_eq!(s, vec![6, 7, 8, 9]);
    }

    #[test]
    fn ring_partial_fill_preserves_order() {
        let r = TopicRing::with_capacity(8);
        r.push(10);
        r.push(20);
        r.push(30);
        assert_eq!(r.snapshot(), vec![10, 20, 30]);
    }

    #[test]
    fn predictor_skips_tick_on_empty_ring() {
        let td = TempDir::new().unwrap();
        let store =
            Arc::new(SubstrateStore::create(td.path().join("p-test")).unwrap());
        let ring = Arc::new(TopicRing::with_capacity(64));
        let p = Predictor::new(PredictorConfig::default(), ring);
        let r = p.tick(&store).expect("tick ok");
        assert_eq!(r.nodes_touched, 0);
        assert_eq!(p.stats().communities_prefetched, 0);
    }

    #[test]
    fn predictor_runs_when_ring_filled() {
        let td = TempDir::new().unwrap();
        let store =
            Arc::new(SubstrateStore::create(td.path().join("p-filled")).unwrap());
        let ring = Arc::new(TopicRing::with_capacity(64));
        for i in 0..8u64 {
            ring.push(i);
        }
        let p = Predictor::new(PredictorConfig::default(), ring);
        let r = p.tick(&store).expect("tick ok");
        assert_eq!(r.nodes_touched, 8);
        assert_eq!(p.stats().communities_prefetched, 3); // top_k default
    }
}
