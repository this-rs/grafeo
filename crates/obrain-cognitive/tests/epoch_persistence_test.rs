//! Tests for cross-session persistence via epoch timestamps.
//!
//! Validates that `epoch_to_instant` correctly reconstructs Instant values
//! and that synapse/energy write-through persists epoch data for proper
//! cross-session decay calculation.

use obrain_cognitive::store_trait::{epoch_to_instant, now_epoch_secs};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// epoch_to_instant — basic round-trip
// ---------------------------------------------------------------------------

#[test]
fn epoch_to_instant_none_returns_now() {
    let before = Instant::now();
    let result = epoch_to_instant(None);
    let after = Instant::now();
    // Should be essentially "now" — within a few ms
    assert!(result >= before.checked_sub(Duration::from_millis(10)).unwrap());
    assert!(result <= after + Duration::from_millis(10));
}

#[test]
fn epoch_to_instant_recent_past() {
    // Simulate an epoch 60 seconds ago
    let sixty_secs_ago_epoch = now_epoch_secs() - 60.0;
    let result = epoch_to_instant(Some(sixty_secs_ago_epoch));
    let now = Instant::now();

    // The reconstructed instant should be ~60s before now
    let delta = now.duration_since(result);
    assert!(
        (delta.as_secs_f64() - 60.0).abs() < 1.0,
        "expected ~60s delta, got {:.2}s",
        delta.as_secs_f64()
    );
}

#[test]
fn epoch_to_instant_clock_skew_future_clamped() {
    // Epoch in the future (clock skew) — should clamp to Instant::now()
    let future_epoch = now_epoch_secs() + 3600.0;
    let before = Instant::now();
    let result = epoch_to_instant(Some(future_epoch));
    let after = Instant::now();

    // Should be clamped to ~now (delta = 0), not in the future
    assert!(result >= before.checked_sub(Duration::from_millis(10)).unwrap());
    assert!(result <= after + Duration::from_millis(10));
}

#[test]
fn epoch_to_instant_very_old_capped_at_365_days() {
    // Epoch from 2 years ago — should be capped at 365 days
    let two_years_ago = now_epoch_secs() - (730.0 * 86400.0);
    let result = epoch_to_instant(Some(two_years_ago));
    let now = Instant::now();

    let delta = now.duration_since(result);
    // Should be capped at 365 days, not 730
    assert!(
        (delta.as_secs_f64() - 365.0 * 86400.0).abs() < 2.0,
        "expected ~365d delta, got {:.2}d",
        delta.as_secs_f64() / 86400.0
    );
}

// ---------------------------------------------------------------------------
// Synapse decay after simulated reload
// ---------------------------------------------------------------------------

#[test]
fn synapse_cross_session_decay_simulation() {
    use obrain_cognitive::{Synapse, SynapseConfig};
    use obrain_common::types::NodeId;

    let half_life = Duration::from_secs(7 * 86400); // 7 days
    let config = SynapseConfig {
        default_half_life: half_life,
        ..SynapseConfig::default()
    };

    // Simulate: synapse reinforced 3 days ago
    let three_days_ago_epoch = now_epoch_secs() - (3.0 * 86400.0);
    let reconstructed_instant = epoch_to_instant(Some(three_days_ago_epoch));

    let syn = Synapse::new_at(
        NodeId::new(1),
        NodeId::new(2),
        1.0,
        config.default_half_life,
        reconstructed_instant,
    );

    // After 3 days with 7-day half-life: W = 1.0 * 2^(-3/7) ≈ 0.7411
    let expected = 2.0_f64.powf(-3.0 / 7.0);
    let actual = syn.current_weight();
    assert!(
        (actual - expected).abs() < 0.01,
        "expected ~{expected:.4}, got {actual:.4}"
    );
}

// ---------------------------------------------------------------------------
// Energy decay after simulated reload
// ---------------------------------------------------------------------------

#[test]
fn energy_cross_session_decay_simulation() {
    use obrain_cognitive::NodeEnergy;

    let half_life = Duration::from_secs(86400); // 24h

    // Simulate: energy boosted 12 hours ago
    let twelve_hours_ago_epoch = now_epoch_secs() - (12.0 * 3600.0);
    let reconstructed_instant = epoch_to_instant(Some(twelve_hours_ago_epoch));

    let ne = NodeEnergy::new_at(1.0, half_life, reconstructed_instant);

    // After 12h with 24h half-life: E = 1.0 * 2^(-0.5) ≈ 0.7071
    let expected = 2.0_f64.powf(-0.5);
    let actual = ne.current_energy();
    assert!(
        (actual - expected).abs() < 0.01,
        "expected ~{expected:.4}, got {actual:.4}"
    );
}
