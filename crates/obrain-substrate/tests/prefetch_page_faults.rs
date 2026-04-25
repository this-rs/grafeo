//! T11 Step 5 verification — prefetch hook reduces page faults.
//!
//! # What this bench proves
//!
//! `SubstrateStore::prefetch_community()` issues `madvise(WILLNEED)` on
//! the bounding page range of a community across the Nodes, Community
//! and Hilbert zones. The kernel schedules readahead for those pages
//! before the app touches them. On a cold page cache, the traversal
//! that follows must **see ≥ 50% fewer minor page faults** than the
//! unhinted path (the plan's Step-5 acceptance criterion).
//!
//! # Methodology
//!
//! We measure `ru_minflt` directly via `getrusage(RUSAGE_SELF)`, so the
//! result is a concrete count — not a latency proxy subject to
//! scheduler jitter or storage-device noise.
//!
//! Setup (per run):
//! 1. Build a substrate with `K_COMMUNITIES * NODES_PER_COMMUNITY` live
//!    nodes, interleaving community ids per round so each community
//!    lands on its own page group (the allocator's slow path).
//! 2. Close and re-open the store (drops in-process cache state).
//! 3. Call `ZoneFile::advise_dontneed()` on Nodes + Community +
//!    Hilbert to evict the kernel page cache for those zones.
//! 4. Snapshot `ru_minflt` via `getrusage`.
//! 5. Traverse every live node in `PROBED_COMMUNITIES` random
//!    communities, touching three `NodeRecord` fields per node so the
//!    loop can't be elided.
//! 6. Snapshot `ru_minflt` again; delta is the run's fault count.
//!
//! Run twice: once with `prefetch_community()` called before each
//! community's loop, once without. Assert
//! `prefetch_faults ≤ 0.5 * no_prefetch_faults`.
//!
//! # Why the test is `#[ignore]` and Linux-only
//!
//! * The cold-cache comparison requires actual `madvise(MADV_DONTNEED)`
//!   eviction. Darwin (macOS) silently ignores `DONTNEED` on file-
//!   backed mappings — pages stay resident — so the "no_prefetch"
//!   variant sees 0 faults and the comparison becomes non-diagnostic.
//!   Linux's file-backed `DONTNEED` is authoritative, so we gate the
//!   test on `target_os = "linux"`.
//! * The cold-cache effect needs a dataset large enough to saturate
//!   the kernel page cache; at small N both runs fault the same
//!   handful of pages.
//! * Meant as a one-shot developer-machine verification after
//!   touching the prefetch plumbing, not a CI signal.
//!
//! Run locally with:
//! ```text
//! cargo test -p obrain-substrate --release --test prefetch_page_faults -- \
//!   --ignored --nocapture
//! ```
//!
//! The `--release` flag matters — the traversal inner loop is hot and
//! unoptimised code causes spurious faults in debug builds.
//!
//! # macOS alternative verification
//!
//! On Darwin the bench still exercises the wiring (the fault-count
//! assertion never trips because `noprefetch_faults == 0` guards it),
//! but for a meaningful page-fault signal use Instruments.app's
//! VM Tracker or a Linux VM. The `prefetch_on_warm_cache_is_cheap`
//! test exercises the hot-path overhead on every OS.

use std::time::Instant;

#[cfg(target_os = "linux")]
use obrain_substrate::file::Zone;
use obrain_substrate::store::SubstrateStore;
use tempfile::tempdir;

/// Number of distinct communities seeded. Chosen large enough that
/// `PROBED_COMMUNITIES * NODES_PER_COMMUNITY * 32 B` exceeds the L2
/// cache on a typical developer machine (16 MB on Apple M2), forcing
/// each probed community to fault most of its pages in.
const K_COMMUNITIES: u32 = 64;

/// Nodes per community. 2048 * 32 B = 64 KiB per community = 16 full
/// 4 KiB pages. The prefetch hook advises all 16 at once — the
/// unhinted path faults them one by one.
const NODES_PER_COMMUNITY: u32 = 2048;

/// Communities probed per run. Random-access so readahead beyond the
/// current community is wasted.
#[cfg(target_os = "linux")]
const PROBED_COMMUNITIES: usize = 16;

/// Maximum acceptable fault ratio (`prefetch_faults / no_prefetch_faults`).
/// 0.5 encodes the plan's ≥ 50% reduction criterion.
#[cfg(target_os = "linux")]
const MAX_FAULT_RATIO: f64 = 0.5;

/// Seed a substrate with interleaved community inserts. Returns the
/// full list of community ids for probing.
fn seed(path: &std::path::Path) -> Vec<u32> {
    let s = SubstrateStore::create(path).unwrap();
    let mut cids = Vec::with_capacity(K_COMMUNITIES as usize);
    for round in 0..NODES_PER_COMMUNITY {
        for cid in 1..=K_COMMUNITIES {
            s.create_node_in_community(&["L"], cid);
            if round == 0 {
                cids.push(cid);
            }
        }
    }
    s.flush().unwrap();
    cids
}

/// Evict kernel page cache for the three prefetch-targeted zones.
///
/// # Safety
/// No outstanding `&[u8]` borrows into these zones at the call site
/// (the substrate was just opened; no traversal has begun).
#[cfg(target_os = "linux")]
unsafe fn evict_zones(s: &SubstrateStore) {
    let sub = s.writer().substrate();
    let sub = sub.lock();
    for zone in [Zone::Nodes, Zone::Community, Zone::Hilbert] {
        if let Ok(zf) = sub.open_zone(zone) {
            unsafe {
                let _ = zf.advise_dontneed();
            }
        }
    }
}

/// Current process minor page-fault count (`ru_minflt` from
/// `getrusage(RUSAGE_SELF)`). Returns `None` on non-Unix platforms.
#[cfg(target_os = "linux")]
fn minor_faults() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: `getrusage` is a read-only syscall that never
        // invalidates Rust's memory model.
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
        if rc != 0 {
            return None;
        }
        Some(usage.ru_minflt as u64)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Traverse every live node in each probed community. If `prefetch` is
/// true, call `prefetch_community` before each community's loop.
fn traverse(s: &SubstrateStore, probes: &[u32], prefetch: bool) -> u64 {
    let mut energy_sum: u64 = 0;
    for &cid in probes {
        if prefetch {
            s.prefetch_community(cid).unwrap();
        }
        let Some((lo, hi)) = s.community_slot_range(cid) else {
            continue;
        };
        for slot in lo..=hi {
            if let Some(rec) = s.writer().read_node(slot).unwrap() {
                // Touch fields at distinct offsets so the inner loop
                // demands the whole record, not just the first byte.
                energy_sum = energy_sum.wrapping_add(rec.energy as u64);
                energy_sum = energy_sum.wrapping_add(rec.community_id as u64);
                energy_sum = energy_sum.wrapping_add(rec.label_bitset);
            }
        }
    }
    energy_sum
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "cold-cache bench — run manually with --ignored"]
fn prefetch_reduces_page_faults() {
    let td = tempdir().unwrap();
    let path = td.path().join("kb-prefetch");
    let cids = seed(&path);
    assert_eq!(cids.len(), K_COMMUNITIES as usize);

    // Deterministic pseudo-random probe selection.
    let probes: Vec<u32> = (0..PROBED_COMMUNITIES)
        .map(|i| cids[(i * 7 + 3) % cids.len()])
        .collect();

    // ---- Run 1: no prefetch -----------------------------------------
    let s = SubstrateStore::open(&path).unwrap();
    unsafe { evict_zones(&s) };
    // Warm the in-process DashMaps so they don't contribute faults
    // during the measurement window.
    for &cid in &probes {
        let _ = s.community_slot_range(cid);
    }
    let faults_before = minor_faults().expect("getrusage failed");
    let t0 = Instant::now();
    let sum_noprefetch = traverse(&s, &probes, false);
    let elapsed_noprefetch = t0.elapsed();
    let faults_after = minor_faults().expect("getrusage failed");
    let noprefetch_faults = faults_after.saturating_sub(faults_before);
    drop(s);

    // ---- Run 2: with prefetch ---------------------------------------
    let s = SubstrateStore::open(&path).unwrap();
    unsafe { evict_zones(&s) };
    for &cid in &probes {
        let _ = s.community_slot_range(cid);
    }
    let faults_before = minor_faults().expect("getrusage failed");
    let t0 = Instant::now();
    let sum_prefetch = traverse(&s, &probes, true);
    let elapsed_prefetch = t0.elapsed();
    let faults_after = minor_faults().expect("getrusage failed");
    let prefetch_faults = faults_after.saturating_sub(faults_before);
    drop(s);

    assert_eq!(
        sum_noprefetch, sum_prefetch,
        "prefetch must not change observed traversal result"
    );

    let ratio = prefetch_faults as f64 / (noprefetch_faults.max(1) as f64);
    eprintln!(
        "prefetch bench: no_prefetch = {noprefetch_faults} faults / {elapsed_noprefetch:?}, \
         prefetch = {prefetch_faults} faults / {elapsed_prefetch:?}, \
         ratio = {ratio:.3} (target ≤ {MAX_FAULT_RATIO})"
    );

    // Guard against a degenerate baseline (if `evict_zones` was a
    // no-op on the current platform, both runs see ~0 faults and the
    // test becomes meaningless).
    assert!(
        noprefetch_faults >= 64,
        "baseline fault count too low ({noprefetch_faults}) — eviction may be \
         silently ignored on this platform; test is non-diagnostic"
    );

    assert!(
        ratio <= MAX_FAULT_RATIO,
        "prefetch hook must cut page faults by ≥ 50% \
         (got {ratio:.3}, no_prefetch = {noprefetch_faults}, prefetch = {prefetch_faults})"
    );
}

/// Sanity companion — on a *warm* cache the prefetch hook must be
/// essentially free, proving the hint is safe to call on every
/// activation event without tanking hot paths.
#[test]
fn prefetch_on_warm_cache_is_cheap() {
    let td = tempdir().unwrap();
    let path = td.path().join("kb-warm");
    let cids = seed(&path);
    let s = SubstrateStore::open(&path).unwrap();

    // Warm up by traversing everything once.
    let _ = traverse(&s, &cids, false);

    let t0 = Instant::now();
    for &cid in cids.iter().take(100).cycle().take(100) {
        s.prefetch_community(cid).unwrap();
    }
    let elapsed = t0.elapsed();
    eprintln!("warm-cache 100×prefetch: {elapsed:?}");
    assert!(
        elapsed.as_millis() < 500,
        "prefetch on warm cache should be ≤ 500 ms for 100 calls, got {elapsed:?}"
    );
}
