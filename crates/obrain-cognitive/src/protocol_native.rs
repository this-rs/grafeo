//! # Native Protocol FSM
//!
//! Protocol states as graph nodes with energy and synapses.
//! Transitions reinforced by Hebbian learning.

use obrain_common::types::NodeId;

// ---------------------------------------------------------------------------
// State and transition types
// ---------------------------------------------------------------------------

/// A protocol state (becomes a graph node).
#[derive(Debug, Clone)]
pub struct ProtocolState {
    /// Optional node ID if persisted to graph.
    pub node_id: Option<NodeId>,
    /// Name of the state (unique within a protocol).
    pub name: String,
    /// Type of the state.
    pub state_type: StateType,
    /// Energy associated with this state.
    pub energy: f64,
    /// Number of times this state has been visited.
    pub visit_count: u64,
}

/// Type of a protocol state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateType {
    /// Initial state of the protocol.
    Start,
    /// Intermediate processing state.
    Intermediate,
    /// Terminal (final) state.
    Terminal,
}

/// A transition between states (becomes a graph edge).
#[derive(Debug, Clone)]
pub struct ProtocolTransition {
    /// Source state name.
    pub from_state: String,
    /// Target state name.
    pub to_state: String,
    /// Trigger that fires this transition.
    pub trigger: String,
    /// Optional guard condition (evaluated externally).
    pub guard: Option<String>,
    /// Synapse weight, reinforced by Hebbian learning.
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// NativeProtocol — definition
// ---------------------------------------------------------------------------

/// A protocol definition containing states and transitions.
#[derive(Debug, Clone)]
pub struct NativeProtocol {
    /// Protocol name.
    pub name: String,
    /// States in this protocol.
    pub states: Vec<ProtocolState>,
    /// Transitions between states.
    pub transitions: Vec<ProtocolTransition>,
}

impl NativeProtocol {
    /// Get available transitions from the given state matching the given trigger.
    pub fn available_transitions(
        &self,
        state: &str,
        trigger: &str,
    ) -> Vec<&ProtocolTransition> {
        self.transitions
            .iter()
            .filter(|t| t.from_state == state && t.trigger == trigger)
            .collect()
    }

    /// Find the start state of the protocol, if any.
    pub fn start_state(&self) -> Option<&ProtocolState> {
        self.states.iter().find(|s| s.state_type == StateType::Start)
    }

    /// Check whether a state name is terminal.
    pub fn is_terminal(&self, state_name: &str) -> bool {
        self.states
            .iter()
            .any(|s| s.name == state_name && s.state_type == StateType::Terminal)
    }
}

// ---------------------------------------------------------------------------
// ProtocolRun — running instance
// ---------------------------------------------------------------------------

/// A running protocol instance.
#[derive(Debug, Clone)]
pub struct ProtocolRun {
    /// Name of the protocol being executed.
    pub protocol_name: String,
    /// Name of the current state.
    pub current_state: String,
    /// History of visited states as `(state_name, timestamp)`.
    pub states_visited: Vec<(String, u64)>,
    /// Current run status.
    pub status: RunStatus,
}

/// Status of a protocol run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Protocol is currently executing.
    Running,
    /// Protocol reached a terminal state successfully.
    Completed,
    /// Protocol encountered a failure.
    Failed,
    /// Protocol was externally cancelled.
    Cancelled,
}

impl ProtocolRun {
    /// Start a new run of the given protocol.
    ///
    /// Returns `None` if the protocol has no start state.
    pub fn start(protocol: &NativeProtocol) -> Option<Self> {
        let start = protocol.start_state()?;
        Some(Self {
            protocol_name: protocol.name.clone(),
            current_state: start.name.clone(),
            states_visited: vec![(start.name.clone(), 0)],
            status: RunStatus::Running,
        })
    }

    /// Fire a transition with the given trigger.
    ///
    /// Returns the new state name on success, or an error message on failure.
    pub fn fire(
        &mut self,
        trigger: &str,
        protocol: &NativeProtocol,
        now: u64,
    ) -> Result<&str, String> {
        if self.status != RunStatus::Running {
            return Err(format!(
                "cannot fire transition: run is {:?}",
                self.status
            ));
        }

        let transitions =
            protocol.available_transitions(&self.current_state, trigger);

        let transition = transitions.first().ok_or_else(|| {
            format!(
                "no transition from '{}' with trigger '{}'",
                self.current_state, trigger
            )
        })?;

        let target = transition.to_state.clone();
        self.current_state = target.clone();
        self.states_visited.push((target, now));

        // Check if we reached a terminal state
        if protocol.is_terminal(&self.current_state) {
            self.status = RunStatus::Completed;
        }

        Ok(&self.current_state)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_protocol() -> NativeProtocol {
        NativeProtocol {
            name: "review".to_string(),
            states: vec![
                ProtocolState {
                    node_id: None,
                    name: "draft".to_string(),
                    state_type: StateType::Start,
                    energy: 1.0,
                    visit_count: 0,
                },
                ProtocolState {
                    node_id: None,
                    name: "review".to_string(),
                    state_type: StateType::Intermediate,
                    energy: 0.8,
                    visit_count: 0,
                },
                ProtocolState {
                    node_id: None,
                    name: "approved".to_string(),
                    state_type: StateType::Terminal,
                    energy: 0.6,
                    visit_count: 0,
                },
                ProtocolState {
                    node_id: None,
                    name: "rejected".to_string(),
                    state_type: StateType::Terminal,
                    energy: 0.3,
                    visit_count: 0,
                },
            ],
            transitions: vec![
                ProtocolTransition {
                    from_state: "draft".to_string(),
                    to_state: "review".to_string(),
                    trigger: "submit".to_string(),
                    guard: None,
                    weight: 1.0,
                },
                ProtocolTransition {
                    from_state: "review".to_string(),
                    to_state: "approved".to_string(),
                    trigger: "approve".to_string(),
                    guard: None,
                    weight: 1.0,
                },
                ProtocolTransition {
                    from_state: "review".to_string(),
                    to_state: "rejected".to_string(),
                    trigger: "reject".to_string(),
                    guard: None,
                    weight: 0.5,
                },
            ],
        }
    }

    #[test]
    fn start_run_from_start_state() {
        let protocol = sample_protocol();
        let run = ProtocolRun::start(&protocol).unwrap();
        assert_eq!(run.current_state, "draft");
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.states_visited.len(), 1);
    }

    #[test]
    fn fire_transition_moves_state() {
        let protocol = sample_protocol();
        let mut run = ProtocolRun::start(&protocol).unwrap();

        let result = run.fire("submit", &protocol, 1000);
        assert!(result.is_ok());
        assert_eq!(run.current_state, "review");
        assert_eq!(run.status, RunStatus::Running);
        assert_eq!(run.states_visited.len(), 2);
    }

    #[test]
    fn reach_terminal_completes_run() {
        let protocol = sample_protocol();
        let mut run = ProtocolRun::start(&protocol).unwrap();

        run.fire("submit", &protocol, 1000).unwrap();
        run.fire("approve", &protocol, 2000).unwrap();

        assert_eq!(run.current_state, "approved");
        assert_eq!(run.status, RunStatus::Completed);
        assert_eq!(run.states_visited.len(), 3);
    }

    #[test]
    fn reach_terminal_via_reject() {
        let protocol = sample_protocol();
        let mut run = ProtocolRun::start(&protocol).unwrap();

        run.fire("submit", &protocol, 1000).unwrap();
        run.fire("reject", &protocol, 2000).unwrap();

        assert_eq!(run.current_state, "rejected");
        assert_eq!(run.status, RunStatus::Completed);
    }

    #[test]
    fn invalid_trigger_returns_error() {
        let protocol = sample_protocol();
        let mut run = ProtocolRun::start(&protocol).unwrap();

        let result = run.fire("approve", &protocol, 1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("no transition from 'draft'"));
    }

    #[test]
    fn cannot_fire_after_completion() {
        let protocol = sample_protocol();
        let mut run = ProtocolRun::start(&protocol).unwrap();

        run.fire("submit", &protocol, 1000).unwrap();
        run.fire("approve", &protocol, 2000).unwrap();

        let result = run.fire("submit", &protocol, 3000);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Completed"));
    }

    #[test]
    fn no_start_state_returns_none() {
        let protocol = NativeProtocol {
            name: "empty".to_string(),
            states: vec![ProtocolState {
                node_id: None,
                name: "mid".to_string(),
                state_type: StateType::Intermediate,
                energy: 0.5,
                visit_count: 0,
            }],
            transitions: vec![],
        };
        assert!(ProtocolRun::start(&protocol).is_none());
    }

    #[test]
    fn available_transitions_filters_correctly() {
        let protocol = sample_protocol();

        let from_draft = protocol.available_transitions("draft", "submit");
        assert_eq!(from_draft.len(), 1);
        assert_eq!(from_draft[0].to_state, "review");

        let from_review_approve =
            protocol.available_transitions("review", "approve");
        assert_eq!(from_review_approve.len(), 1);

        let from_review_reject =
            protocol.available_transitions("review", "reject");
        assert_eq!(from_review_reject.len(), 1);

        // No transition for this trigger from this state
        let none = protocol.available_transitions("draft", "approve");
        assert!(none.is_empty());
    }
}
