//! GDS Refresh Scheduler — periodic batch refresh of global graph metrics.
//!
//! Reuses the existing algorithms from `grafeo-adapters`:
//! - [`pagerank`](grafeo_adapters::plugins::algorithms::pagerank)
//! - [`betweenness_centrality`](grafeo_adapters::plugins::algorithms::betweenness_centrality)
//! - [`louvain`](grafeo_adapters::plugins::algorithms::louvain)
//!
//! The scheduler runs on a configurable interval or mutation-count threshold,
//! computing these metrics and injecting results into the [`FabricStore`].

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[cfg(feature = "gds-refresh")]
use grafeo_common::types::NodeId;
#[cfg(feature = "gds-refresh")]
use grafeo_core::graph::GraphStore;

use crate::fabric::FabricStore;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GDS refresh scheduler.
#[derive(Debug, Clone)]
pub struct GdsRefreshConfig {
    /// How often to run a full GDS refresh (default: 5 minutes).
    pub refresh_interval: Duration,
    /// Number of mutations that triggers an immediate refresh (default: 1000).
    pub mutation_threshold: u64,
    /// PageRank damping factor (default: 0.85).
    pub pagerank_damping: f64,
    /// PageRank max iterations (default: 100).
    pub pagerank_max_iterations: usize,
    /// PageRank convergence tolerance (default: 1e-6).
    pub pagerank_tolerance: f64,
    /// Louvain resolution parameter (default: 1.0).
    pub louvain_resolution: f64,
    /// Whether to normalize betweenness centrality (default: true).
    pub betweenness_normalized: bool,
}

impl Default for GdsRefreshConfig {
    fn default() -> Self {
        Self {
            refresh_interval: Duration::from_secs(5 * 60), // 5 minutes
            mutation_threshold: 1000,
            pagerank_damping: 0.85,
            pagerank_max_iterations: 100,
            pagerank_tolerance: 1e-6,
            louvain_resolution: 1.0,
            betweenness_normalized: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GdsRefreshScheduler
// ---------------------------------------------------------------------------

/// Scheduler that periodically refreshes global graph metrics (PageRank,
/// Betweenness, Louvain) and injects results into the [`FabricStore`].
///
/// The scheduler tracks mutation count and triggers a refresh either:
/// - When `mutation_threshold` mutations have accumulated, or
/// - When `refresh_interval` has elapsed (when used with a background timer).
pub struct GdsRefreshScheduler {
    /// Fabric store to update with computed metrics.
    fabric_store: Arc<FabricStore>,
    /// Configuration.
    config: GdsRefreshConfig,
    /// Number of mutations since last refresh.
    mutation_count: AtomicU64,
}

impl GdsRefreshScheduler {
    /// Creates a new GDS refresh scheduler.
    pub fn new(fabric_store: Arc<FabricStore>, config: GdsRefreshConfig) -> Self {
        Self {
            fabric_store,
            config,
            mutation_count: AtomicU64::new(0),
        }
    }

    /// Records N mutations. Returns `true` if the mutation threshold was reached.
    pub fn record_mutations(&self, count: u64) -> bool {
        let prev = self.mutation_count.fetch_add(count, Ordering::Relaxed);
        prev + count >= self.config.mutation_threshold
    }

    /// Returns the current mutation count since last refresh.
    pub fn mutations_since_refresh(&self) -> u64 {
        self.mutation_count.load(Ordering::Relaxed)
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &GdsRefreshConfig {
        &self.config
    }

    /// Returns a reference to the fabric store.
    pub fn fabric_store(&self) -> &Arc<FabricStore> {
        &self.fabric_store
    }

    /// Executes a full GDS refresh using the provided graph store.
    ///
    /// This reuses the algorithms from `grafeo-adapters`:
    /// - PageRank for link-structure importance
    /// - Betweenness centrality for path involvement
    /// - Louvain for community detection
    ///
    /// Results are injected into the [`FabricStore`] and risk scores are recalculated.
    #[cfg(feature = "gds-refresh")]
    pub fn refresh(&self, store: &dyn GraphStore) {
        use grafeo_adapters::plugins::algorithms::{betweenness_centrality, louvain, pagerank};

        tracing::info!("GDS refresh starting");

        // 1. PageRank
        let pr_scores = pagerank(
            store,
            self.config.pagerank_damping,
            self.config.pagerank_max_iterations,
            self.config.pagerank_tolerance,
        );

        // 2. Betweenness centrality
        let bc_scores = betweenness_centrality(store, self.config.betweenness_normalized);

        // 3. Louvain community detection
        let louvain_result = louvain(store, self.config.louvain_resolution);

        // 4. Inject results into FabricStore
        let all_nodes: std::collections::HashSet<NodeId> = pr_scores
            .keys()
            .chain(bc_scores.keys())
            .chain(louvain_result.communities.keys())
            .copied()
            .collect();

        for node_id in all_nodes {
            let pr = pr_scores.get(&node_id).copied().unwrap_or(0.0);
            let bc = bc_scores.get(&node_id).copied().unwrap_or(0.0);
            let community = louvain_result.communities.get(&node_id).copied();
            self.fabric_store
                .set_gds_metrics(node_id, pr, bc, community);
        }

        // 5. Recalculate all risk scores with updated global metrics
        self.fabric_store.recalculate_all_risks();

        // 6. Reset mutation counter
        self.mutation_count.store(0, Ordering::Relaxed);

        tracing::info!(
            nodes = self.fabric_store.len(),
            communities = louvain_result.num_communities,
            "GDS refresh completed"
        );
    }

    /// Executes a GDS refresh without grafeo-adapters (no-op stub).
    ///
    /// When the `gds-refresh` feature is not enabled, this method does nothing.
    /// Use the `gds-refresh` feature to enable full GDS computation.
    #[cfg(not(feature = "gds-refresh"))]
    pub fn refresh_stub(&self) {
        tracing::warn!("GDS refresh called but 'gds-refresh' feature is not enabled");
        self.mutation_count.store(0, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for GdsRefreshScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdsRefreshScheduler")
            .field("config", &self.config)
            .field(
                "mutations_since_refresh",
                &self.mutation_count.load(Ordering::Relaxed),
            )
            .finish()
    }
}
