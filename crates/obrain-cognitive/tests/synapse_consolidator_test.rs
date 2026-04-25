//! Plan 69e59065 T3 step 2+3 — SynapseConsolidator background task.
//!
//! Validates the off-WAL metadata write path end-to-end:
//!   1. `reinforce()` queues `MetadataDelta` instead of synchronously
//!      persisting LPG properties (T3 step 1).
//!   2. The background `SynapseConsolidator` ticks every
//!      `config.consolidator_interval` and drains the queue.
//!   3. `ConsolidatorHandle::shutdown().await` triggers a final drain
//!      and only returns once the queue is empty (T3 step 3 — the
//!      SIGTERM-safe contract).
//!
//! The plan's verification text was "1000 reinforces puis sleep(35s)
//! → pending = 0". 35s is poison for CI, so we use a 50ms tick and
//! assert with bounded waits. The semantics are identical: the
//! consolidator exists to drain the queue periodically, the cadence
//! is just compressed for the test.
//!
//! Run: `cargo test -p obrain-cognitive --test synapse_consolidator_test \
//!        --features synapse,substrate -- --nocapture`

#![cfg(all(feature = "synapse", feature = "substrate"))]

use std::sync::Arc;
use std::time::Duration;

use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::NodeId;

/// Helper: open a fresh substrate-backed SynapseStore in a tempdir
/// with a tiny consolidator interval suitable for tests.
///
/// FD-leak hardening — see gotcha 8484a4b7: explicit drops + the
/// returned TempDir lifetimes ensure each test cleans up its substrate
/// fd handles before the next test runs.
fn fresh_store(interval: Duration) -> (Arc<SynapseStore>, tempfile::TempDir) {
    use obrain_substrate::SubstrateStore;

    let td = tempfile::tempdir().expect("tempdir");
    let substrate = Arc::new(
        SubstrateStore::create(td.path().join("kb")).expect("create substrate"),
    );

    // SubstrateStore implements GraphStoreMut directly — same pattern
    // as the other substrate_tests in synapse.rs.
    let graph_store: Arc<dyn obrain_core::graph::GraphStoreMut> = substrate.clone();

    // Pre-create node ids so reinforce has valid endpoints.
    for _ in 0..256 {
        graph_store.create_node(&["n"]);
    }

    let config = SynapseConfig {
        consolidator_interval: interval,
        ..SynapseConfig::default()
    };
    let store = Arc::new(SynapseStore::with_substrate(
        config,
        graph_store,
        substrate,
    ));
    (store, td)
}

/// T3 step 2 — the consolidator must drain `pending_metadata_writes`
/// on its tick. We push 100 reinforces, wait 3 ticks, assert empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consolidator_ticks_drain_queue() {
    let (store, _td) = fresh_store(Duration::from_millis(50));

    // 100 reinforces → 100 entries queued (each reinforce queues
    // exactly one MetadataDelta in substrate mode).
    for i in 1..=100u64 {
        store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
    }
    assert_eq!(
        store.pending_metadata_writes_count(),
        100,
        "reinforce must queue MetadataDelta instead of persisting synchronously"
    );

    // Spawn the consolidator; let it run a few ticks.
    let consolidator = Arc::clone(&store).spawn_consolidator();
    // First tick is skipped (it would fire immediately), then 50ms
    // intervals. Wait 200ms = enough for ~3 real ticks.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Queue must be drained now.
    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "consolidator must drain the queue within ~3 ticks"
    );

    consolidator.shutdown().await;
}

/// T3 step 2 + step 3 — push reinforces, then immediately shutdown
/// the consolidator. The final drain must catch every queued entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consolidator_shutdown_drains_queue_on_signal() {
    // Long interval (1s) so the periodic tick will NOT fire during
    // the test — we want to prove the shutdown signal is what
    // drains the queue, not the timer.
    let (store, _td) = fresh_store(Duration::from_secs(1));

    let consolidator = Arc::clone(&store).spawn_consolidator();

    // 200 reinforces, immediately followed by shutdown.
    for i in 1..=200u64 {
        store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
    }
    assert_eq!(store.pending_metadata_writes_count(), 200);

    // shutdown().await — guarantees the final drain has executed
    // BEFORE we return.
    consolidator.shutdown().await;

    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "shutdown drain must persist every queued MetadataDelta"
    );
}

/// T3 step 3 — `ConsolidatorRuntime::shutdown_blocking()` is the
/// SIGTERM-safe entry point usable from synchronous code (the hub's
/// `CognitiveBrain::open` is sync, so it cannot await). The runtime
/// owns its own tokio scheduler on a dedicated worker thread,
/// independent of the caller's runtime.
#[test]
fn consolidator_runtime_owned_drains_on_shutdown() {
    let (store, _td) = fresh_store(Duration::from_secs(60));

    // Spawn the consolidator on its own owned runtime — works from
    // sync context (no tokio runtime around this test).
    let consolidator =
        obrain_cognitive::synapse::ConsolidatorRuntime::spawn(Arc::clone(&store));

    // Push 150 reinforces.
    for i in 1..=150u64 {
        store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
    }
    assert_eq!(store.pending_metadata_writes_count(), 150);

    // Synchronous shutdown — drains the queue, joins the runtime.
    consolidator.shutdown_blocking();

    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "ConsolidatorRuntime::shutdown_blocking must persist every queued delta"
    );
}

/// T3 step 3 — REGRESSION GUARD for the "Cannot start a runtime from
/// within a runtime" panic observed in production 2026-04-25 (PID
/// 33362, see hub commit e41c21c9). Reproducer: `ConsolidatorRuntime`
/// is constructed from `CognitiveBrain::open` which is called from
/// the chat handler — itself running on a tokio worker thread. The
/// previous implementation called `runtime.block_on(...)` to bootstrap
/// the consolidator task; tokio refuses to nest `block_on` and aborts
/// the worker thread.
///
/// The fix uses `Handle::enter()` (non-blocking) instead of
/// `block_on`. This test exercises that exact pattern: spawn a
/// `ConsolidatorRuntime` from inside a `#[tokio::test(flavor = "multi_thread")]`
/// — equivalent to a production async hub handler — and verifies it
/// doesn't panic, drains 100 reinforces, and shuts down cleanly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consolidator_runtime_spawn_from_tokio_context_does_not_panic() {
    // We are inside a tokio runtime here.
    let (store, _td) = fresh_store(Duration::from_secs(60));

    // This call previously panicked with "Cannot start a runtime from
    // within a runtime". With the Handle::enter() fix it works.
    let consolidator =
        obrain_cognitive::synapse::ConsolidatorRuntime::spawn(Arc::clone(&store));

    // Push 100 reinforces.
    for i in 1..=100u64 {
        store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
    }
    assert_eq!(store.pending_metadata_writes_count(), 100);

    // Sync shutdown from inside the tokio context — must NOT panic
    // (the previous shutdown also used block_on, same nested-runtime
    // issue). The fix flushes synchronously via the SynapseStore.
    consolidator.shutdown_blocking();

    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "shutdown_blocking from tokio context must drain the queue"
    );
}

/// T3 step 3 — `Drop` on `ConsolidatorRuntime` is best-effort: it
/// triggers a synchronous shutdown with a 1s timeout. Verifies no
/// panic, no hang, and the final drain runs.
#[test]
fn consolidator_runtime_drop_is_safe() {
    let (store, _td) = fresh_store(Duration::from_secs(60));

    {
        let _consolidator =
            obrain_cognitive::synapse::ConsolidatorRuntime::spawn(Arc::clone(&store));
        for i in 1..=75u64 {
            store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
        }
        // Drop the runtime — should drain on the way out.
    }

    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "ConsolidatorRuntime::drop must drain remaining writes"
    );
}

/// T3 step 3 — the Drop path on the inner ConsolidatorHandle is
/// best-effort: signals the task but doesn't await. A subsequent
/// `flush_pending_metadata()` call must still see whatever was queued
/// (possibly empty if the task ticked in between, possibly populated
/// if the task hasn't run yet). Either way, no panics, no hangs, no
/// lost writes after a final manual flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consolidator_drop_path_is_safe() {
    let (store, _td) = fresh_store(Duration::from_millis(50));

    {
        let consolidator = Arc::clone(&store).spawn_consolidator();
        for i in 1..=50u64 {
            store.reinforce(NodeId(i), NodeId(i + 1), 0.05);
        }
        // Drop the handle without awaiting. The signal is sent, but
        // we don't wait for the task.
        drop(consolidator);
    }

    // Manual flush as the safety net — production hub does this
    // explicitly during shutdown if it can't await the consolidator.
    let _ = store.flush_pending_metadata();
    assert_eq!(
        store.pending_metadata_writes_count(),
        0,
        "after explicit flush, queue must be empty regardless of drop path"
    );
}
