//! Integration tests for the GDS refresh scheduler.

#![cfg(feature = "gds-refresh")]

use grafeo_cognitive::gds_refresh::{GdsRefreshConfig, GdsRefreshScheduler};
use grafeo_cognitive::fabric::FabricStore;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// GdsRefreshConfig tests
// ---------------------------------------------------------------------------

#[test]
fn config_default_values() {
    let config = GdsRefreshConfig::default();
    assert_eq!(config.refresh_interval, Duration::from_secs(5 * 60));
    assert_eq!(config.mutation_threshold, 1000);
    assert!((config.pagerank_damping - 0.85).abs() < 1e-10);
    assert_eq!(config.pagerank_max_iterations, 100);
    assert!((config.pagerank_tolerance - 1e-6).abs() < 1e-12);
    assert!((config.louvain_resolution - 1.0).abs() < 1e-10);
    assert!(config.betweenness_normalized);
}

#[test]
fn config_debug_formatting() {
    let config = GdsRefreshConfig::default();
    let dbg = format!("{:?}", config);
    assert!(dbg.contains("GdsRefreshConfig"), "got: {dbg}");
    assert!(dbg.contains("refresh_interval"), "got: {dbg}");
    assert!(dbg.contains("mutation_threshold"), "got: {dbg}");
}

// ---------------------------------------------------------------------------
// GdsRefreshScheduler tests
// ---------------------------------------------------------------------------

fn make_scheduler(threshold: u64) -> GdsRefreshScheduler {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig {
        mutation_threshold: threshold,
        ..GdsRefreshConfig::default()
    };
    GdsRefreshScheduler::new(fabric, config)
}

#[test]
fn scheduler_initial_mutations_zero() {
    let sched = make_scheduler(100);
    assert_eq!(sched.mutations_since_refresh(), 0);
}

#[test]
fn record_mutations_below_threshold_returns_false() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(10);
    assert!(!reached, "10 mutations < threshold of 100");
    assert_eq!(sched.mutations_since_refresh(), 10);
}

#[test]
fn record_mutations_at_threshold_returns_true() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(100);
    assert!(reached, "100 mutations >= threshold of 100");
    assert_eq!(sched.mutations_since_refresh(), 100);
}

#[test]
fn record_mutations_above_threshold_returns_true() {
    let sched = make_scheduler(100);
    let reached = sched.record_mutations(200);
    assert!(reached, "200 mutations >= threshold of 100");
}

#[test]
fn record_mutations_accumulates_across_calls() {
    let sched = make_scheduler(100);
    assert!(!sched.record_mutations(30));
    assert_eq!(sched.mutations_since_refresh(), 30);
    assert!(!sched.record_mutations(30));
    assert_eq!(sched.mutations_since_refresh(), 60);
    // 60 + 50 = 110 >= 100
    assert!(sched.record_mutations(50));
    assert_eq!(sched.mutations_since_refresh(), 110);
}

#[test]
fn config_accessor() {
    let sched = make_scheduler(42);
    assert_eq!(sched.config().mutation_threshold, 42);
    assert!((sched.config().pagerank_damping - 0.85).abs() < 1e-10);
}

#[test]
fn fabric_store_accessor() {
    let fabric = Arc::new(FabricStore::new());
    let config = GdsRefreshConfig::default();
    let sched = GdsRefreshScheduler::new(Arc::clone(&fabric), config);
    // Should return a reference to the same fabric store
    assert_eq!(sched.fabric_store().len(), 0);
}

#[test]
fn debug_formatting() {
    let sched = make_scheduler(50);
    sched.record_mutations(7);
    let dbg = format!("{:?}", sched);
    assert!(dbg.contains("GdsRefreshScheduler"), "got: {dbg}");
    assert!(dbg.contains("mutations_since_refresh"), "got: {dbg}");
    assert!(dbg.contains("config"), "got: {dbg}");
}
