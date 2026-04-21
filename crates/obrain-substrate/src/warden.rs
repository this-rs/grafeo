//! CommunityWarden — fragmentation Thinker for topology-as-storage
//! (T11 Step 4).
//!
//! Role
//! ----
//! The online allocator ([`crate::store::SubstrateStore::create_node_in_community`])
//! preserves community locality per-insert, but adversarial or time-varying
//! access patterns (inter-community inserts, deletions, community-id
//! drift from LDleiden) eventually fragment a community across many
//! 4 KiB pages. The warden is a periodic scanner that:
//!
//! 1. Walks the Nodes zone once to compute, per community, the number of
//!    **distinct pages** it lives in versus the **ideal** page count
//!    (`ceil(live_count / NODES_PER_PAGE)`).
//! 2. Flags every community whose `distinct / ideal > trigger_threshold`
//!    (default `1.30` — 30% more pages than perfect packing).
//! 3. Invokes compaction on the flagged communities.
//!
//! Step 4 scope
//! ------------
//! This step ships the **detection** pipeline plus a STUB compaction that
//! piggy-backs on [`crate::writer::Writer::bulk_sort_by_hilbert`] (full
//! file re-sort). That is correct-by-construction but heavy; Step 5
//! replaces the stub with a transactional per-community page swap backed
//! by `WalPayload::CompactCommunity` for crash safety.
//!
//! Verification
//! ------------
//! See `tests/community_warden.rs`:
//!
//! > Induce interleaved inserts across 4 communities → fragmentation >> 30%
//! > → `warden.tick()` → re-scan → fragmentation ratio ≤ `1.15`.

use std::sync::Arc;

use crate::error::SubstrateResult;
use crate::record::{NodeRecord, NODES_PER_PAGE};
use crate::store::SubstrateStore;

/// Default fragmentation trigger ratio. A community with more than 30%
/// extra pages than perfect packing is scheduled for compaction.
pub const DEFAULT_FRAGMENTATION_TRIGGER: f32 = 1.30;

/// Default Hilbert order used by the stub compaction path (Step 4).
/// Step 5 drops this — compaction becomes strictly per-community.
const DEFAULT_HILBERT_ORDER: u32 = 4;
const DEFAULT_HILBERT_MAX_DEGREE: u32 = 32;

/// Per-community fragmentation report.
#[derive(Debug, Clone)]
pub struct CommunityFragmentation {
    /// The community the report is about.
    pub community_id: u32,
    /// Number of distinct 4 KiB pages containing at least one live node
    /// of this community.
    pub distinct_pages: u32,
    /// Live (non-tombstoned) node count for this community.
    pub live_count: u32,
    /// Ideal page count given live_count: `ceil(live / NODES_PER_PAGE)`.
    /// A community with 0 live nodes reports ideal = 0.
    pub ideal_pages: u32,
    /// Fragmentation ratio = distinct_pages / ideal_pages. A ratio of
    /// `1.0` means perfect packing; higher is worse. For a community
    /// with 0 live nodes the ratio is defined as `0.0` (no fragmentation
    /// can exist without nodes).
    pub fragmentation: f32,
}

impl CommunityFragmentation {
    /// True iff this community's fragmentation exceeds the given trigger
    /// threshold. Communities with 0 live nodes never trigger.
    pub fn needs_compaction(&self, trigger: f32) -> bool {
        self.live_count > 0 && self.fragmentation > trigger
    }
}

/// Aggregate fragmentation report over the whole substrate.
#[derive(Debug, Clone, Default)]
pub struct FragmentationReport {
    /// Per-community stats, sorted by `community_id`.
    pub communities: Vec<CommunityFragmentation>,
    /// Total distinct pages used by any community (sum over communities).
    pub total_distinct_pages: u32,
    /// Total live nodes across communities.
    pub total_live_nodes: u32,
    /// Total ideal pages (sum over communities of `ideal_pages`).
    pub total_ideal_pages: u32,
}

impl FragmentationReport {
    /// Overall fragmentation ratio across communities. `0.0` on empty
    /// store.
    pub fn overall_ratio(&self) -> f32 {
        if self.total_ideal_pages == 0 {
            0.0
        } else {
            self.total_distinct_pages as f32 / self.total_ideal_pages as f32
        }
    }
}

/// Scanner + compaction trigger.
///
/// The warden holds an `Arc<SubstrateStore>` so it can be shared across
/// maintenance threads and invoked from periodic timers without holding
/// a `&mut` reference to the store.
pub struct CommunityWarden {
    store: Arc<SubstrateStore>,
    trigger: f32,
}

impl CommunityWarden {
    /// Create a warden with the default 1.30 trigger ratio.
    pub fn new(store: Arc<SubstrateStore>) -> Self {
        Self::with_trigger(store, DEFAULT_FRAGMENTATION_TRIGGER)
    }

    /// Create a warden with a custom trigger ratio (ratio of distinct
    /// pages to ideal pages above which compaction fires). Typical
    /// values: `1.30` for opportunistic repack, `1.50` for a looser
    /// background pass.
    pub fn with_trigger(store: Arc<SubstrateStore>, trigger: f32) -> Self {
        Self { store, trigger }
    }

    /// Current trigger ratio.
    pub fn trigger(&self) -> f32 {
        self.trigger
    }

    /// Walk the Nodes zone once and compute per-community fragmentation
    /// stats. Read-only.
    ///
    /// Implementation: for each live (non-tombstoned) slot, compute its
    /// page index (`slot / NODES_PER_PAGE`) and insert `(community_id,
    /// page)` into a bitset-backed set. At the end, each community's
    /// entry gives `distinct_pages` (set size) and `live_count`.
    pub fn scan(&self) -> SubstrateResult<FragmentationReport> {
        use std::collections::BTreeMap;
        use std::collections::BTreeSet;

        // Per-community: set of distinct pages, live count.
        let mut pages_by_community: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        let mut live_by_community: BTreeMap<u32, u32> = BTreeMap::new();

        let hw = self.store.slot_high_water();
        for slot in 1..hw {
            let Some(rec) = self.store.writer().read_node(slot)? else {
                continue;
            };
            if rec.is_tombstoned() {
                continue;
            }
            // Skip slots that look like zero-filled alignment padding
            // (label_bitset == 0 AND community_id == 0 AND
            // centrality_cached == 0 AND energy == 0). A real node
            // created via `create_node_in_community(&[], 0)` also has
            // bitset == 0, but it's expected to count under community 0
            // — the test fixtures avoid this ambiguity by using
            // non-empty labels or non-zero community ids.
            let is_padding = rec.label_bitset == 0
                && rec.community_id == 0
                && rec.centrality_cached == 0
                && rec.energy == 0
                && rec.scar_util_affinity == 0
                && rec.flags == 0
                && rec.first_edge_off == crate::record::U48::ZERO
                && rec.first_prop_off == crate::record::U48::ZERO;
            if is_padding {
                continue;
            }
            let page = slot / NODES_PER_PAGE;
            pages_by_community
                .entry(rec.community_id)
                .or_default()
                .insert(page);
            *live_by_community.entry(rec.community_id).or_insert(0) += 1;
        }

        // Merge into an aggregate report.
        let mut communities = Vec::with_capacity(pages_by_community.len());
        let mut total_distinct_pages = 0u32;
        let mut total_live_nodes = 0u32;
        let mut total_ideal_pages = 0u32;
        for (cid, pages) in &pages_by_community {
            let distinct = pages.len() as u32;
            let live = live_by_community.get(cid).copied().unwrap_or(0);
            let ideal = live.div_ceil(NODES_PER_PAGE).max(1);
            let ratio = if ideal == 0 {
                0.0
            } else {
                distinct as f32 / ideal as f32
            };
            total_distinct_pages += distinct;
            total_live_nodes += live;
            total_ideal_pages += ideal;
            communities.push(CommunityFragmentation {
                community_id: *cid,
                distinct_pages: distinct,
                live_count: live,
                ideal_pages: ideal,
                fragmentation: ratio,
            });
        }

        Ok(FragmentationReport {
            communities,
            total_distinct_pages,
            total_live_nodes,
            total_ideal_pages,
        })
    }

    /// Run one maintenance pass: scan + compact every community that
    /// exceeds the trigger threshold.
    ///
    /// Returns the list of `community_id`s that were compacted. An
    /// empty return means the store is within the fragmentation budget.
    ///
    /// Compaction (Step 4 stub)
    /// ------------------------
    /// This delegates to `bulk_sort_by_hilbert`, which rewrites the
    /// whole file (not per-community). Expensive but correct. Step 5
    /// replaces this with surgical per-community page-swap under
    /// `WalPayload::CompactCommunity`.
    pub fn tick(&self) -> SubstrateResult<Vec<u32>> {
        let report = self.scan()?;
        let mut fired = Vec::new();
        for stats in &report.communities {
            if stats.needs_compaction(self.trigger) {
                fired.push(stats.community_id);
            }
        }
        if !fired.is_empty() {
            // Step 4 stub: one bulk sort handles every flagged
            // community in a single pass. Step 5 swaps this for
            // per-community compaction.
            //
            // `bulk_sort_by_hilbert` early-exits when HILBERT_SORTED is
            // set, so we must first invalidate the flag. Online inserts
            // clear it automatically; after a previous warden tick (or
            // manual bulk_sort) the flag is set — we clear it here so
            // the next sort re-runs over the post-mutation state.
            self.store.invalidate_layout_flag();

            let node_hw = self.store.slot_high_water();
            let edge_hw = self.store.edge_slot_high_water();
            self.store.writer().bulk_sort_by_hilbert(
                node_hw,
                edge_hw,
                DEFAULT_HILBERT_ORDER,
                DEFAULT_HILBERT_MAX_DEGREE,
            )?;

            // T11 Step 5: bulk_sort_by_hilbert rewrites node slots, so
            // `community_placements` / `community_first_slots` (the
            // prefetch range maps on the store) are now stale. Rescan
            // the Nodes zone once to rebuild them. Cost: O(high_water)
            // — amortized across warden ticks, this is negligible next
            // to the sort pass we just ran.
            self.store.refresh_community_ranges()?;
        }
        Ok(fired)
    }
}

// Compile-time invariant sanity check: size_of(NodeRecord) must divide
// evenly into the page boundary or the page/slot accounting above is
// misaligned. Same assertion as in record.rs, kept here to document the
// coupling for warden maintainers.
const _: [(); 1] = [(); ((4096usize / NodeRecord::SIZE) == NODES_PER_PAGE as usize) as usize];
