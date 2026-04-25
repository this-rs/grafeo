//! # Trajectory Tracking
//!
//! Records behavioral sequences (action chains) and detects patterns,
//! regressions, and efficiency over time. Each trajectory point deposits
//! stigmergic pheromones on the target node, creating implicit reinforcement.
//!
//! ## Data Model
//!
//! - `:TrajectoryPoint` — a single action in a sequence
//! - `:TrajectoryPoint -[:NEXT {delay_ms}]-> :TrajectoryPoint` — ordering
//! - `:TrajectoryPoint -[:ACTED_ON]-> <target>` — what was affected
//!
//! ## Analysis Capabilities
//!
//! - **Frequent subsequences** — mine repeated action patterns
//! - **Regression detection** — detect returns to inefficient patterns
//! - **Efficiency scoring** — ratio of useful vs wasted actions
//! - **Compression** — old trajectories collapse into patterns

use std::collections::HashMap;

use obrain_common::types::NodeId;
use web_time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label for trajectory point nodes in the graph.
pub const LABEL_TRAJECTORY_POINT: &str = "TrajectoryPoint";

/// Edge linking sequential trajectory points.
pub const EDGE_NEXT: &str = "NEXT";

/// Edge linking a point to its target.
pub const EDGE_ACTED_ON: &str = "ACTED_ON";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Type of action in a trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    /// An MCP tool was used.
    ToolUse,
    /// A decision was made.
    Decision,
    /// Navigation to a different entity/file.
    Navigation,
    /// Context was switched (different task, plan, file).
    ContextSwitch,
}

impl ActionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ToolUse => "tool_use",
            Self::Decision => "decision",
            Self::Navigation => "navigation",
            Self::ContextSwitch => "context_switch",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "tool_use" => Some(Self::ToolUse),
            "decision" => Some(Self::Decision),
            "navigation" => Some(Self::Navigation),
            "context_switch" => Some(Self::ContextSwitch),
            _ => None,
        }
    }
}

/// A single point in a trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    /// Node id in the graph (set after persistence).
    pub node_id: Option<NodeId>,
    /// Type of action.
    pub action_type: ActionType,
    /// Identifier for the specific action (tool name, decision id, etc.).
    pub action_id: String,
    /// Target node that was acted upon.
    pub target_node: Option<NodeId>,
    /// Timestamp (ms since epoch).
    pub timestamp_ms: u64,
    /// Episode this point belongs to.
    pub episode_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Trajectory Recorder
// ---------------------------------------------------------------------------

/// Records trajectory points within a session.
#[derive(Debug)]
pub struct TrajectoryRecorder {
    /// Points recorded in this session.
    points: Vec<TrajectoryPoint>,
    /// Session start time for relative timestamps.
    start: Instant,
    /// Episode id for this recording session.
    episode_id: Option<String>,
}

impl TrajectoryRecorder {
    /// Start a new recording session.
    pub fn new(episode_id: Option<String>) -> Self {
        Self {
            points: Vec::new(),
            start: Instant::now(),
            episode_id,
        }
    }

    /// Record a trajectory point.
    pub fn record(
        &mut self,
        action_type: ActionType,
        action_id: impl Into<String>,
        target_node: Option<NodeId>,
    ) {
        let elapsed = self.start.elapsed().as_millis() as u64;
        self.points.push(TrajectoryPoint {
            node_id: None,
            action_type,
            action_id: action_id.into(),
            target_node,
            timestamp_ms: elapsed,
            episode_id: self.episode_id.clone(),
        });
    }

    /// Get all recorded points.
    pub fn points(&self) -> &[TrajectoryPoint] {
        &self.points
    }

    /// Get the number of recorded points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the recorder is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute delays between consecutive points.
    pub fn delays(&self) -> Vec<u64> {
        self.points
            .windows(2)
            .map(|w| w[1].timestamp_ms.saturating_sub(w[0].timestamp_ms))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Trajectory Analysis
// ---------------------------------------------------------------------------

/// Result of trajectory analysis.
#[derive(Debug, Clone)]
pub struct TrajectoryAnalysis {
    /// Total number of actions.
    pub total_actions: usize,
    /// Breakdown by action type.
    pub action_counts: HashMap<ActionType, usize>,
    /// Efficiency score [0, 1] — ratio of non-redundant actions.
    pub efficiency: f64,
    /// Detected frequent subsequences.
    pub frequent_subsequences: Vec<Subsequence>,
    /// Context switch count (often indicates confusion or thrashing).
    pub context_switches: usize,
    /// Average delay between actions (ms).
    pub avg_delay_ms: f64,
}

/// A frequent subsequence detected in the trajectory.
#[derive(Debug, Clone)]
pub struct Subsequence {
    /// The action pattern (sequence of action_ids).
    pub pattern: Vec<String>,
    /// How many times this pattern appears.
    pub count: usize,
    /// Average total duration of this pattern (ms).
    pub avg_duration_ms: f64,
}

/// Analyze a recorded trajectory.
pub fn analyze(recorder: &TrajectoryRecorder) -> TrajectoryAnalysis {
    let points = recorder.points();
    let total_actions = points.len();

    // Count by type
    let mut action_counts: HashMap<ActionType, usize> = HashMap::new();
    for p in points {
        *action_counts.entry(p.action_type).or_insert(0) += 1;
    }

    let context_switches = action_counts
        .get(&ActionType::ContextSwitch)
        .copied()
        .unwrap_or(0);

    // Efficiency: penalize context switches and repeated actions on same target
    let efficiency = compute_efficiency(points);

    // Frequent subsequences (bigrams and trigrams)
    let frequent_subsequences = find_frequent_subsequences(points, 2);

    // Average delay
    let delays = recorder.delays();
    let avg_delay_ms = if delays.is_empty() {
        0.0
    } else {
        delays.iter().sum::<u64>() as f64 / delays.len() as f64
    };

    TrajectoryAnalysis {
        total_actions,
        action_counts,
        efficiency,
        frequent_subsequences,
        context_switches,
        avg_delay_ms,
    }
}

fn compute_efficiency(points: &[TrajectoryPoint]) -> f64 {
    if points.is_empty() {
        return 1.0;
    }

    let total = points.len() as f64;
    let mut redundant = 0.0;

    // Count consecutive duplicate targets as redundant
    for window in points.windows(2) {
        if window[0].target_node == window[1].target_node
            && window[0].action_type == window[1].action_type
        {
            redundant += 1.0;
        }
    }

    // Context switches are partial waste
    let switches = points
        .iter()
        .filter(|p| p.action_type == ActionType::ContextSwitch)
        .count() as f64;
    redundant += switches * 0.5;

    (1.0 - redundant / total).max(0.0)
}

fn find_frequent_subsequences(points: &[TrajectoryPoint], min_count: usize) -> Vec<Subsequence> {
    let mut bigrams: HashMap<(String, String), (usize, Vec<u64>)> = HashMap::new();

    for window in points.windows(2) {
        let key = (window[0].action_id.clone(), window[1].action_id.clone());
        let duration = window[1].timestamp_ms.saturating_sub(window[0].timestamp_ms);
        let entry = bigrams.entry(key).or_insert((0, Vec::new()));
        entry.0 += 1;
        entry.1.push(duration);
    }

    bigrams
        .into_iter()
        .filter(|(_, (count, _))| *count >= min_count)
        .map(|((a, b), (count, durations))| {
            let avg_duration_ms =
                durations.iter().sum::<u64>() as f64 / durations.len().max(1) as f64;
            Subsequence {
                pattern: vec![a, b],
                count,
                avg_duration_ms,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Regression Detection
// ---------------------------------------------------------------------------

/// A detected regression in behavior.
#[derive(Debug, Clone)]
pub struct RegressionAlert {
    /// The inefficient pattern detected.
    pub pattern: Vec<String>,
    /// How many times it recurred.
    pub recurrence_count: usize,
    /// Description of why this is a regression.
    pub reason: String,
}

/// Detect regressions by comparing current trajectory to known inefficient patterns.
pub fn detect_regressions(
    recorder: &TrajectoryRecorder,
    known_inefficient: &[Vec<String>],
) -> Vec<RegressionAlert> {
    let points = recorder.points();
    let mut alerts = Vec::new();

    for pattern in known_inefficient {
        if pattern.is_empty() {
            continue;
        }
        let mut count = 0;

        'outer: for window in points.windows(pattern.len()) {
            for (point, expected) in window.iter().zip(pattern.iter()) {
                if point.action_id != *expected {
                    continue 'outer;
                }
            }
            count += 1;
        }

        if count > 0 {
            alerts.push(RegressionAlert {
                pattern: pattern.clone(),
                recurrence_count: count,
                reason: format!(
                    "Inefficient pattern '{}' detected {} time(s)",
                    pattern.join(" → "),
                    count
                ),
            });
        }
    }

    alerts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_analyze() {
        let mut recorder = TrajectoryRecorder::new(Some("ep1".into()));
        recorder.record(ActionType::ToolUse, "read_file", Some(NodeId::from(1u64)));
        recorder.record(ActionType::ToolUse, "edit_file", Some(NodeId::from(1u64)));
        recorder.record(ActionType::Decision, "approve_change", None);

        assert_eq!(recorder.len(), 3);

        let analysis = analyze(&recorder);
        assert_eq!(analysis.total_actions, 3);
        assert_eq!(analysis.action_counts[&ActionType::ToolUse], 2);
        assert_eq!(analysis.action_counts[&ActionType::Decision], 1);
        assert_eq!(analysis.context_switches, 0);
    }

    #[test]
    fn test_efficiency_perfect() {
        let mut recorder = TrajectoryRecorder::new(None);
        recorder.record(ActionType::ToolUse, "a", Some(NodeId::from(1u64)));
        recorder.record(ActionType::ToolUse, "b", Some(NodeId::from(2u64)));
        recorder.record(ActionType::ToolUse, "c", Some(NodeId::from(3u64)));

        let analysis = analyze(&recorder);
        assert_eq!(analysis.efficiency, 1.0); // all unique targets, no waste
    }

    #[test]
    fn test_efficiency_with_redundancy() {
        let mut recorder = TrajectoryRecorder::new(None);
        // Same action on same target twice in a row = redundant
        recorder.record(ActionType::ToolUse, "read", Some(NodeId::from(1u64)));
        recorder.record(ActionType::ToolUse, "read", Some(NodeId::from(1u64)));
        recorder.record(ActionType::ToolUse, "edit", Some(NodeId::from(2u64)));

        let analysis = analyze(&recorder);
        assert!(analysis.efficiency < 1.0);
    }

    #[test]
    fn test_frequent_subsequences() {
        let mut recorder = TrajectoryRecorder::new(None);
        // Repeat pattern: read → edit → read → edit → read → edit
        for _ in 0..3 {
            recorder.record(ActionType::ToolUse, "read", Some(NodeId::from(1u64)));
            recorder.record(ActionType::ToolUse, "edit", Some(NodeId::from(1u64)));
        }

        let analysis = analyze(&recorder);
        let read_edit: Vec<_> = analysis
            .frequent_subsequences
            .iter()
            .filter(|s| s.pattern == vec!["read", "edit"])
            .collect();
        assert!(!read_edit.is_empty());
        assert!(read_edit[0].count >= 2);
    }

    #[test]
    fn test_regression_detection() {
        let mut recorder = TrajectoryRecorder::new(None);
        recorder.record(ActionType::ToolUse, "open_file", None);
        recorder.record(ActionType::ToolUse, "close_file", None);
        recorder.record(ActionType::ToolUse, "open_file", None);
        recorder.record(ActionType::ToolUse, "close_file", None);

        let inefficient = vec![vec!["open_file".into(), "close_file".into()]];
        let alerts = detect_regressions(&recorder, &inefficient);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].recurrence_count, 2);
    }

    #[test]
    fn test_no_regression() {
        let mut recorder = TrajectoryRecorder::new(None);
        recorder.record(ActionType::ToolUse, "read", None);
        recorder.record(ActionType::ToolUse, "edit", None);

        let inefficient = vec![vec!["delete".into(), "undo".into()]];
        let alerts = detect_regressions(&recorder, &inefficient);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_delays() {
        let recorder = TrajectoryRecorder::new(None);
        // Empty recorder → no delays
        assert!(recorder.delays().is_empty());
    }

    #[test]
    fn test_context_switch_counting() {
        let mut recorder = TrajectoryRecorder::new(None);
        recorder.record(ActionType::ToolUse, "a", None);
        recorder.record(ActionType::ContextSwitch, "switch_task", None);
        recorder.record(ActionType::ToolUse, "b", None);
        recorder.record(ActionType::ContextSwitch, "switch_file", None);

        let analysis = analyze(&recorder);
        assert_eq!(analysis.context_switches, 2);
    }
}
