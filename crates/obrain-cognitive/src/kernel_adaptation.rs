//! # Kernel Adaptation
//!
//! Feedback loop that adjusts [`KernelParam`]s based on [`SessionMetric`]s.
//!
//! After each session, the adaptation engine:
//! 1. Reads the latest session metrics
//! 2. Computes a gradient of correction for each kernel parameter
//! 3. Applies EMA (Exponential Moving Average) update: `param += lr × gradient`
//! 4. Clamps values to `[min, max]`
//! 5. Detects oscillation and convergence for meta-adaptation
//!
//! ## Gradient Rules
//!
//! | Metric | High → | Low → |
//! |--------|--------|-------|
//! | context_noise_ratio > 0.5 | ↑ propagation_decay | |
//! | context_miss_ratio > 0.3 | | ↓ propagation_decay, ↑ max_hops |
//! | community_boundary_crossed > 3 | ↑ community_cohesion_threshold | |
//! | token_budget_usage > 0.9 | ↓ context_budget_tokens | |
//! | token_budget_usage < 0.3 | ↑ context_budget_tokens | |
//! | resolution_time ↑ over 3 sessions | review budgets | |
//!
//! ## Meta-Adaptation (Layer 3)
//!
//! - **Oscillation**: If a parameter changes sign 3 times consecutively → halve its learning_rate
//! - **Convergence**: If variance < ε over 5 sessions → halve learning_rate
//! - **Degradation**: If all metrics worsen globally → increase all learning_rates (exploration)

use crate::error::{CognitiveError, CognitiveResult};
use crate::kernel_params::KernelParamStore;

use crate::engram::traits::CognitiveStorage;

// ---------------------------------------------------------------------------
// Adaptation constants
// ---------------------------------------------------------------------------

/// Threshold above which context_noise_ratio triggers decay increase.
const NOISE_HIGH_THRESHOLD: f64 = 0.5;

/// Threshold above which context_miss_ratio triggers decay decrease.
const MISS_HIGH_THRESHOLD: f64 = 0.3;

/// Threshold for community boundary crossings.
const BOUNDARY_HIGH_THRESHOLD: f64 = 3.0;

/// Token budget usage upper bound before adjustment.
const TOKEN_USAGE_HIGH: f64 = 0.9;

/// Token budget usage lower bound before adjustment.
const TOKEN_USAGE_LOW: f64 = 0.3;

/// Number of consecutive sign changes to trigger oscillation detection.
const OSCILLATION_SIGN_CHANGES: usize = 3;

/// Variance threshold below which convergence is detected.
const CONVERGENCE_VARIANCE_EPSILON: f64 = 0.001;

/// Window size for convergence detection.
const CONVERGENCE_WINDOW: usize = 5;

/// Factor by which learning rate is reduced on oscillation/convergence.
const LR_REDUCTION_FACTOR: f64 = 0.5;

/// Factor by which learning rate is increased on global degradation.
#[allow(dead_code)]
const LR_INCREASE_FACTOR: f64 = 1.5;

/// Maximum learning rate cap.
const LR_MAX: f64 = 0.5;

/// Minimum learning rate floor.
const LR_MIN: f64 = 0.001;

// ---------------------------------------------------------------------------
// SessionMetric (inline definition for independence)
// ---------------------------------------------------------------------------

/// Session performance metrics used by the adaptation engine.
///
/// This struct mirrors `session_metrics::SessionMetric` but is defined here
/// to keep the adaptation module compilable independently.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionFeedback {
    /// Proportion of injected context that was unused.
    pub context_noise_ratio: f64,
    /// Proportion of times the agent re-requested info already in the graph.
    pub context_miss_ratio: f64,
    /// Average task resolution time in milliseconds.
    pub resolution_time_ms: f64,
    /// Maximum effective propagation depth used.
    pub propagation_depth_used: f64,
    /// Number of cross-community propagation events.
    pub community_boundary_crossed: f64,
    /// Ratio of tokens used vs budget max.
    pub token_budget_usage: f64,
}

impl Default for SessionFeedback {
    fn default() -> Self {
        Self {
            context_noise_ratio: 0.0,
            context_miss_ratio: 0.0,
            resolution_time_ms: 0.0,
            propagation_depth_used: 0.0,
            community_boundary_crossed: 0.0,
            token_budget_usage: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Computed gradient for a single kernel parameter.
#[derive(Debug, Clone)]
pub struct ParamGradient {
    /// Parameter name.
    pub param_name: String,
    /// Gradient value (positive = increase, negative = decrease).
    pub gradient: f64,
    /// Reason for the gradient.
    pub reason: String,
}

/// Computes gradients for all kernel parameters based on session feedback.
pub fn compute_gradients(feedback: &SessionFeedback) -> Vec<ParamGradient> {
    let mut gradients = Vec::new();

    // --- propagation_decay ---
    // High noise → propagate less (increase decay)
    // High miss → propagate more (decrease decay)
    let decay_gradient = if feedback.context_noise_ratio > NOISE_HIGH_THRESHOLD {
        let excess = feedback.context_noise_ratio - NOISE_HIGH_THRESHOLD;
        ParamGradient {
            param_name: "propagation_decay".into(),
            gradient: excess,
            reason: format!(
                "context_noise_ratio={:.2} > {NOISE_HIGH_THRESHOLD}: increase decay",
                feedback.context_noise_ratio
            ),
        }
    } else if feedback.context_miss_ratio > MISS_HIGH_THRESHOLD {
        let excess = feedback.context_miss_ratio - MISS_HIGH_THRESHOLD;
        ParamGradient {
            param_name: "propagation_decay".into(),
            gradient: -excess,
            reason: format!(
                "context_miss_ratio={:.2} > {MISS_HIGH_THRESHOLD}: decrease decay",
                feedback.context_miss_ratio
            ),
        }
    } else {
        ParamGradient {
            param_name: "propagation_decay".into(),
            gradient: 0.0,
            reason: "metrics within bounds".into(),
        }
    };
    gradients.push(decay_gradient);

    // --- max_hops ---
    // High miss → increase hops
    if feedback.context_miss_ratio > MISS_HIGH_THRESHOLD {
        let excess = feedback.context_miss_ratio - MISS_HIGH_THRESHOLD;
        gradients.push(ParamGradient {
            param_name: "max_hops".into(),
            gradient: excess * 2.0, // scale to hop range
            reason: format!(
                "context_miss_ratio={:.2} > {MISS_HIGH_THRESHOLD}: increase hops",
                feedback.context_miss_ratio
            ),
        });
    }

    // --- community_cohesion_threshold ---
    // Many boundary crossings → increase threshold (stop crossing)
    if feedback.community_boundary_crossed > BOUNDARY_HIGH_THRESHOLD {
        let excess = (feedback.community_boundary_crossed - BOUNDARY_HIGH_THRESHOLD) / 10.0;
        gradients.push(ParamGradient {
            param_name: "community_cohesion_threshold".into(),
            gradient: excess.min(0.2),
            reason: format!(
                "boundary_crossed={:.0} > {BOUNDARY_HIGH_THRESHOLD}: tighten boundary",
                feedback.community_boundary_crossed
            ),
        });
    }

    // --- context_budget_tokens ---
    // High token usage → increase budget (needs more room)
    // Low token usage → decrease budget (wasting space)
    if feedback.token_budget_usage > TOKEN_USAGE_HIGH {
        let excess = feedback.token_budget_usage - TOKEN_USAGE_HIGH;
        gradients.push(ParamGradient {
            param_name: "context_budget_tokens".into(),
            gradient: excess * 500.0, // scale to token range
            reason: format!(
                "token_usage={:.2} > {TOKEN_USAGE_HIGH}: increase budget",
                feedback.token_budget_usage
            ),
        });
    } else if feedback.token_budget_usage < TOKEN_USAGE_LOW {
        let deficit = TOKEN_USAGE_LOW - feedback.token_budget_usage;
        gradients.push(ParamGradient {
            param_name: "context_budget_tokens".into(),
            gradient: -deficit * 200.0, // reduce gradually
            reason: format!(
                "token_usage={:.2} < {TOKEN_USAGE_LOW}: decrease budget",
                feedback.token_budget_usage
            ),
        });
    }

    gradients
}

// ---------------------------------------------------------------------------
// AdaptationHistory — tracks gradient history for meta-adaptation
// ---------------------------------------------------------------------------

/// History of gradient values for a single parameter.
#[derive(Debug, Clone, Default)]
pub struct ParamHistory {
    /// Recent gradient values (most recent last).
    pub recent_gradients: Vec<f64>,
    /// Count of consecutive sign changes.
    pub sign_change_count: usize,
    /// Last gradient sign (true = positive, false = negative, None = zero).
    pub last_sign: Option<bool>,
}

impl ParamHistory {
    /// Records a new gradient value and updates oscillation detection.
    pub fn record(&mut self, gradient: f64) {
        self.recent_gradients.push(gradient);

        // Keep only the last CONVERGENCE_WINDOW entries
        if self.recent_gradients.len() > CONVERGENCE_WINDOW {
            self.recent_gradients
                .drain(..self.recent_gradients.len() - CONVERGENCE_WINDOW);
        }

        // Oscillation detection: count sign changes
        if gradient.abs() > 1e-9 {
            let current_sign = gradient > 0.0;
            if let Some(last) = self.last_sign {
                if current_sign != last {
                    self.sign_change_count += 1;
                }
            }
            self.last_sign = Some(current_sign);
        }
    }

    /// Returns true if the parameter is oscillating.
    pub fn is_oscillating(&self) -> bool {
        self.sign_change_count >= OSCILLATION_SIGN_CHANGES
    }

    /// Returns true if the parameter has converged (low variance).
    pub fn is_converged(&self) -> bool {
        if self.recent_gradients.len() < CONVERGENCE_WINDOW {
            return false;
        }
        let mean = self.recent_gradients.iter().sum::<f64>() / self.recent_gradients.len() as f64;
        let variance = self
            .recent_gradients
            .iter()
            .map(|g| (g - mean).powi(2))
            .sum::<f64>()
            / self.recent_gradients.len() as f64;
        variance < CONVERGENCE_VARIANCE_EPSILON
    }

    /// Resets oscillation tracking (after learning rate reduction).
    pub fn reset_oscillation(&mut self) {
        self.sign_change_count = 0;
        self.last_sign = None;
    }
}

/// Tracks gradient history for all parameters.
#[derive(Debug, Clone, Default)]
pub struct AdaptationHistory {
    /// Per-parameter gradient history.
    pub params: std::collections::HashMap<String, ParamHistory>,
}

impl AdaptationHistory {
    /// Creates a new empty history.
    pub fn new() -> Self {
        Self {
            params: std::collections::HashMap::new(),
        }
    }

    /// Records a gradient for a parameter.
    pub fn record(&mut self, param_name: &str, gradient: f64) {
        self.params
            .entry(param_name.to_string())
            .or_default()
            .record(gradient);
    }

    /// Returns the history for a specific parameter.
    pub fn get(&self, param_name: &str) -> Option<&ParamHistory> {
        self.params.get(param_name)
    }
}

// ---------------------------------------------------------------------------
// Adaptation Result
// ---------------------------------------------------------------------------

/// Result of a single adaptation step.
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    /// Adjustments applied.
    pub adjustments: Vec<ParamAdjustment>,
    /// Meta-adaptation events (oscillation, convergence).
    pub meta_events: Vec<MetaEvent>,
}

/// A single parameter adjustment.
#[derive(Debug, Clone)]
pub struct ParamAdjustment {
    /// Parameter name.
    pub param_name: String,
    /// Old value.
    pub old_value: f64,
    /// New value (after clamping).
    pub new_value: f64,
    /// Gradient applied.
    pub gradient: f64,
    /// Learning rate used.
    pub learning_rate: f64,
    /// Reason for the adjustment.
    pub reason: String,
}

/// Meta-adaptation event.
#[derive(Debug, Clone)]
pub enum MetaEvent {
    /// Parameter is oscillating — learning rate reduced.
    Oscillation {
        param_name: String,
        old_lr: f64,
        new_lr: f64,
    },
    /// Parameter has converged — learning rate reduced.
    Convergence {
        param_name: String,
        old_lr: f64,
        new_lr: f64,
    },
    /// Global degradation — all learning rates increased.
    GlobalDegradation {
        lr_factor: f64,
    },
}

// ---------------------------------------------------------------------------
// Main adaptation function
// ---------------------------------------------------------------------------

/// Runs one adaptation step: computes gradients from session feedback,
/// applies EMA updates to kernel parameters, and performs meta-adaptation.
///
/// # Arguments
///
/// * `storage` — the cognitive graph store
/// * `feedback` — session metrics from the completed session
/// * `history` — mutable reference to the adaptation history (persisted across sessions)
/// * `now_millis` — current timestamp in milliseconds
///
/// # Returns
///
/// An [`AdaptationResult`] describing all adjustments and meta-events.
pub fn adapt(
    storage: &dyn CognitiveStorage,
    feedback: &SessionFeedback,
    history: &mut AdaptationHistory,
    now_millis: u64,
) -> CognitiveResult<AdaptationResult> {
    let gradients = compute_gradients(feedback);
    let params = KernelParamStore::list_params(storage)?;

    let mut adjustments = Vec::new();
    let mut meta_events = Vec::new();

    for grad in &gradients {
        if grad.gradient.abs() < 1e-9 {
            continue; // skip zero gradients
        }

        // Record in history for meta-adaptation
        history.record(&grad.param_name, grad.gradient);

        // Find the corresponding parameter
        let param = match params.iter().find(|p| p.name == grad.param_name) {
            Some(p) => p,
            None => continue,
        };

        // Meta-adaptation: check for oscillation and convergence
        if let Some(ph) = history.get(&grad.param_name) {
            if ph.is_oscillating() {
                let old_lr = param.learning_rate;
                let new_lr = (old_lr * LR_REDUCTION_FACTOR).max(LR_MIN);
                if (new_lr - old_lr).abs() > 1e-9 {
                    // Update the learning rate in the store
                    update_learning_rate(storage, &grad.param_name, new_lr)?;
                    meta_events.push(MetaEvent::Oscillation {
                        param_name: grad.param_name.clone(),
                        old_lr,
                        new_lr,
                    });
                    // Reset oscillation counter
                    if let Some(ph) = history.params.get_mut(&grad.param_name) {
                        ph.reset_oscillation();
                    }
                }
            } else if ph.is_converged() {
                let old_lr = param.learning_rate;
                let new_lr = (old_lr * LR_REDUCTION_FACTOR).max(LR_MIN);
                if (new_lr - old_lr).abs() > 1e-9 {
                    update_learning_rate(storage, &grad.param_name, new_lr)?;
                    meta_events.push(MetaEvent::Convergence {
                        param_name: grad.param_name.clone(),
                        old_lr,
                        new_lr,
                    });
                }
            }
        }

        // Apply EMA update
        let old_value = param.value;
        let new_value = old_value + param.learning_rate * grad.gradient;

        KernelParamStore::set_param(storage, &grad.param_name, new_value, Some(now_millis))?;

        adjustments.push(ParamAdjustment {
            param_name: grad.param_name.clone(),
            old_value,
            new_value: new_value.clamp(param.min_value, param.max_value),
            gradient: grad.gradient,
            learning_rate: param.learning_rate,
            reason: grad.reason.clone(),
        });
    }

    Ok(AdaptationResult {
        adjustments,
        meta_events,
    })
}

/// Updates only the learning rate of a kernel parameter.
fn update_learning_rate(
    storage: &dyn CognitiveStorage,
    name: &str,
    new_lr: f64,
) -> CognitiveResult<()> {
    let filter = crate::engram::traits::CognitiveFilter::PropertyEquals(
        "kernel_name".to_string(),
        obrain_common::types::Value::from(name),
    );
    let nodes = storage.query_nodes(crate::kernel_params::LABEL_KERNEL_PARAM, Some(&filter));
    let node = nodes.first().ok_or_else(|| {
        CognitiveError::Store(format!("kernel param not found: {name}"))
    })?;

    let mut props = std::collections::HashMap::new();
    props.insert(
        "kernel_lr".to_string(),
        obrain_common::types::Value::Float64(new_lr.clamp(LR_MIN, LR_MAX)),
    );
    storage.update_node(node.id, &props);

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_high_noise_increases_decay() {
        let feedback = SessionFeedback {
            context_noise_ratio: 0.8,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let decay_grad = grads.iter().find(|g| g.param_name == "propagation_decay");
        assert!(decay_grad.is_some());
        assert!(decay_grad.unwrap().gradient > 0.0, "should increase decay");
    }

    #[test]
    fn gradient_high_miss_decreases_decay() {
        let feedback = SessionFeedback {
            context_miss_ratio: 0.6,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let decay_grad = grads.iter().find(|g| g.param_name == "propagation_decay");
        assert!(decay_grad.is_some());
        assert!(
            decay_grad.unwrap().gradient < 0.0,
            "should decrease decay"
        );
    }

    #[test]
    fn gradient_high_miss_increases_hops() {
        let feedback = SessionFeedback {
            context_miss_ratio: 0.6,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let hops_grad = grads.iter().find(|g| g.param_name == "max_hops");
        assert!(hops_grad.is_some());
        assert!(hops_grad.unwrap().gradient > 0.0, "should increase hops");
    }

    #[test]
    fn gradient_boundary_crossing_tightens_threshold() {
        let feedback = SessionFeedback {
            community_boundary_crossed: 7.0,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let cohesion_grad = grads
            .iter()
            .find(|g| g.param_name == "community_cohesion_threshold");
        assert!(cohesion_grad.is_some());
        assert!(
            cohesion_grad.unwrap().gradient > 0.0,
            "should tighten boundary"
        );
    }

    #[test]
    fn gradient_high_token_usage_increases_budget() {
        let feedback = SessionFeedback {
            token_budget_usage: 0.95,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let budget_grad = grads
            .iter()
            .find(|g| g.param_name == "context_budget_tokens");
        assert!(budget_grad.is_some());
        assert!(
            budget_grad.unwrap().gradient > 0.0,
            "should increase budget"
        );
    }

    #[test]
    fn gradient_low_token_usage_decreases_budget() {
        let feedback = SessionFeedback {
            token_budget_usage: 0.1,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let budget_grad = grads
            .iter()
            .find(|g| g.param_name == "context_budget_tokens");
        assert!(budget_grad.is_some());
        assert!(
            budget_grad.unwrap().gradient < 0.0,
            "should decrease budget"
        );
    }

    #[test]
    fn gradient_within_bounds_zero() {
        let feedback = SessionFeedback {
            context_noise_ratio: 0.3,
            context_miss_ratio: 0.1,
            token_budget_usage: 0.5,
            community_boundary_crossed: 1.0,
            ..Default::default()
        };
        let grads = compute_gradients(&feedback);
        let nonzero = grads.iter().filter(|g| g.gradient.abs() > 1e-9).count();
        assert_eq!(nonzero, 0, "all metrics within bounds → zero gradients");
    }

    #[test]
    fn oscillation_detection() {
        let mut ph = ParamHistory::default();
        ph.record(0.1);
        ph.record(-0.1);
        ph.record(0.1);
        ph.record(-0.1);
        assert!(ph.is_oscillating());
    }

    #[test]
    fn convergence_detection() {
        let mut ph = ParamHistory::default();
        for _ in 0..5 {
            ph.record(0.001);
        }
        assert!(ph.is_converged());
    }

    #[test]
    fn no_convergence_with_variance() {
        let mut ph = ParamHistory::default();
        ph.record(0.1);
        ph.record(-0.2);
        ph.record(0.3);
        ph.record(-0.1);
        ph.record(0.2);
        assert!(!ph.is_converged());
    }

    #[test]
    fn history_window_trim() {
        let mut ph = ParamHistory::default();
        for i in 0..20 {
            ph.record(i as f64 * 0.01);
        }
        assert_eq!(ph.recent_gradients.len(), CONVERGENCE_WINDOW);
    }
}
