//! Provenance System — automatic tracking of cognitive events.
//!
//! Every significant cognitive mutation (energy boost, decay sweep, synapse prune,
//! scar creation, risk recalculation) automatically produces a [`CognitiveEvent`]
//! node connected via `HAS_COGNITIVE_EVENT` edges. Consolidation operations
//! additionally create `DERIVED_FROM` edges linking output nodes to their sources.
//!
//! This is a reactive provenance system: callers do not need to explicitly
//! record events — the [`ProvenanceRecorder`] is called internally by the
//! cognitive engine subsystems.

use grafeo_common::types::NodeId;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// CognitiveEventType
// ---------------------------------------------------------------------------

/// Types of cognitive events that are automatically tracked.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CognitiveEventType {
    /// Node energy was boosted.
    EnergyBoost,
    /// Decay sweep removed or reduced energy on low-energy nodes.
    DecaySweep,
    /// Synapses below threshold were pruned.
    SynapsePrune,
    /// A scar was placed on a node.
    ScarCreation,
    /// Risk score was recalculated for a node.
    RiskRecalc,
    /// Consolidation merged or derived new nodes from sources.
    Consolidation,
    /// Custom event type for extensibility.
    Custom(String),
}

impl fmt::Display for CognitiveEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EnergyBoost => write!(f, "energy_boost"),
            Self::DecaySweep => write!(f, "decay_sweep"),
            Self::SynapsePrune => write!(f, "synapse_prune"),
            Self::ScarCreation => write!(f, "scar_creation"),
            Self::RiskRecalc => write!(f, "risk_recalc"),
            Self::Consolidation => write!(f, "consolidation"),
            Self::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// CognitiveEventId
// ---------------------------------------------------------------------------

/// Unique identifier for a cognitive event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CognitiveEventId(pub u64);

static NEXT_EVENT_ID: AtomicU64 = AtomicU64::new(1);

impl CognitiveEventId {
    fn next() -> Self {
        Self(NEXT_EVENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for CognitiveEventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cog_event:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CognitiveEvent
// ---------------------------------------------------------------------------

/// A recorded cognitive event — captures what changed, when, and why.
#[derive(Debug, Clone)]
pub struct CognitiveEvent {
    /// Unique event identifier.
    pub id: CognitiveEventId,
    /// The type of cognitive operation that occurred.
    pub event_type: CognitiveEventType,
    /// When the event occurred (millis since UNIX epoch).
    pub timestamp: u64,
    /// The previous value (if applicable). `None` for creation events.
    pub old_value: Option<f64>,
    /// The new value after the operation.
    pub new_value: Option<f64>,
    /// What triggered this event (e.g., "mutation_listener", "manual_boost", "prune_sweep").
    pub trigger_source: String,
    /// The node this event is associated with.
    pub target_node: NodeId,
}

impl CognitiveEvent {
    /// Creates a new cognitive event with the current timestamp.
    pub fn new(
        event_type: CognitiveEventType,
        target_node: NodeId,
        old_value: Option<f64>,
        new_value: Option<f64>,
        trigger_source: impl Into<String>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        Self {
            id: CognitiveEventId::next(),
            event_type,
            timestamp,
            old_value,
            new_value,
            trigger_source: trigger_source.into(),
            target_node,
        }
    }

    /// Creates a cognitive event with a specific timestamp (for testing).
    pub fn new_at(
        event_type: CognitiveEventType,
        target_node: NodeId,
        old_value: Option<f64>,
        new_value: Option<f64>,
        trigger_source: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            id: CognitiveEventId::next(),
            event_type,
            timestamp,
            old_value,
            new_value,
            trigger_source: trigger_source.into(),
            target_node,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge types
// ---------------------------------------------------------------------------

/// Edge type for connecting a node to its cognitive events.
pub const EDGE_HAS_COGNITIVE_EVENT: &str = "HAS_COGNITIVE_EVENT";

/// Edge type for provenance: links a derived/consolidated node to its sources.
pub const EDGE_DERIVED_FROM: &str = "DERIVED_FROM";

// ---------------------------------------------------------------------------
// DerivedFrom record
// ---------------------------------------------------------------------------

/// Records a DERIVED_FROM relationship between a target node and its sources.
#[derive(Debug, Clone)]
pub struct DerivedFromRecord {
    /// The derived (output) node.
    pub target: NodeId,
    /// The source nodes from which the target was derived.
    pub sources: Vec<NodeId>,
    /// When the derivation occurred (millis since UNIX epoch).
    pub timestamp: u64,
    /// Description of the derivation operation.
    pub operation: String,
}

// ---------------------------------------------------------------------------
// ProvenanceRecorder
// ---------------------------------------------------------------------------

/// Records cognitive events and provenance edges.
///
/// This is a reactive recorder: cognitive subsystems call its methods
/// internally whenever a significant mutation occurs. Consumers query
/// the event history via [`get_history`](Self::get_history).
///
/// Events are stored in-memory, indexed by target node for efficient
/// history queries. The recorder also tracks `DERIVED_FROM` relationships
/// for consolidation operations.
pub struct ProvenanceRecorder {
    /// Events indexed by target node ID.
    events: dashmap::DashMap<NodeId, Vec<CognitiveEvent>>,
    /// DERIVED_FROM records.
    derivations: dashmap::DashMap<NodeId, Vec<DerivedFromRecord>>,
}

impl ProvenanceRecorder {
    /// Creates a new, empty provenance recorder.
    pub fn new() -> Self {
        Self {
            events: dashmap::DashMap::new(),
            derivations: dashmap::DashMap::new(),
        }
    }

    /// Records a cognitive event for a node.
    ///
    /// This creates a `CognitiveEvent` node and an implicit
    /// `HAS_COGNITIVE_EVENT` edge from the target node to the event.
    pub fn record(&self, event: CognitiveEvent) {
        let node_id = event.target_node;
        self.events
            .entry(node_id)
            .and_modify(|events| events.push(event.clone()))
            .or_insert_with(|| vec![event]);
    }

    /// Convenience: records an energy boost event.
    pub fn record_energy_boost(
        &self,
        node_id: NodeId,
        old_energy: Option<f64>,
        new_energy: f64,
        trigger: &str,
    ) {
        self.record(CognitiveEvent::new(
            CognitiveEventType::EnergyBoost,
            node_id,
            old_energy,
            Some(new_energy),
            trigger,
        ));
    }

    /// Convenience: records a decay sweep event.
    pub fn record_decay_sweep(&self, node_id: NodeId, old_energy: f64, new_energy: f64) {
        self.record(CognitiveEvent::new(
            CognitiveEventType::DecaySweep,
            node_id,
            Some(old_energy),
            Some(new_energy),
            "decay_sweep",
        ));
    }

    /// Convenience: records a synapse prune event.
    pub fn record_synapse_prune(&self, node_id: NodeId, pruned_weight: f64) {
        self.record(CognitiveEvent::new(
            CognitiveEventType::SynapsePrune,
            node_id,
            Some(pruned_weight),
            None,
            "synapse_prune",
        ));
    }

    /// Convenience: records a scar creation event.
    pub fn record_scar_creation(&self, node_id: NodeId, intensity: f64) {
        self.record(CognitiveEvent::new(
            CognitiveEventType::ScarCreation,
            node_id,
            None,
            Some(intensity),
            "scar_creation",
        ));
    }

    /// Convenience: records a risk recalculation event.
    pub fn record_risk_recalc(&self, node_id: NodeId, old_risk: f64, new_risk: f64) {
        self.record(CognitiveEvent::new(
            CognitiveEventType::RiskRecalc,
            node_id,
            Some(old_risk),
            Some(new_risk),
            "risk_recalc",
        ));
    }

    /// Records a DERIVED_FROM relationship for consolidation.
    ///
    /// The `target` is the consolidated/derived node, and `sources` are
    /// the original nodes that contributed to it.
    pub fn record_derivation(
        &self,
        target: NodeId,
        sources: Vec<NodeId>,
        operation: impl Into<String>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        let record = DerivedFromRecord {
            target,
            sources,
            timestamp,
            operation: operation.into(),
        };
        self.derivations
            .entry(target)
            .and_modify(|records| records.push(record.clone()))
            .or_insert_with(|| vec![record]);
    }

    /// Returns the cognitive event history for a node, sorted chronologically.
    pub fn get_history(&self, node_id: NodeId) -> Vec<CognitiveEvent> {
        self.events.get(&node_id).map_or_else(Vec::new, |events| {
            let mut sorted = events.clone();
            sorted.sort_by_key(|e| e.timestamp);
            sorted
        })
    }

    /// Returns events of a specific type for a node.
    pub fn get_history_by_type(
        &self,
        node_id: NodeId,
        event_type: &CognitiveEventType,
    ) -> Vec<CognitiveEvent> {
        self.events.get(&node_id).map_or_else(Vec::new, |events| {
            let mut filtered: Vec<CognitiveEvent> = events
                .iter()
                .filter(|e| &e.event_type == event_type)
                .cloned()
                .collect();
            filtered.sort_by_key(|e| e.timestamp);
            filtered
        })
    }

    /// Returns all DERIVED_FROM records for a node.
    pub fn get_derivations(&self, node_id: NodeId) -> Vec<DerivedFromRecord> {
        self.derivations
            .get(&node_id)
            .map_or_else(Vec::new, |records| records.clone())
    }

    /// Returns the total number of recorded events across all nodes.
    pub fn total_events(&self) -> usize {
        self.events.iter().map(|e| e.value().len()).sum()
    }

    /// Returns the number of nodes with recorded events.
    pub fn nodes_with_events(&self) -> usize {
        self.events.len()
    }

    /// Returns all HAS_COGNITIVE_EVENT edges as (source_node, event_id) pairs for a node.
    pub fn get_edges(&self, node_id: NodeId) -> Vec<(NodeId, CognitiveEventId)> {
        self.events.get(&node_id).map_or_else(Vec::new, |events| {
            events.iter().map(|e| (node_id, e.id)).collect()
        })
    }

    /// Returns all DERIVED_FROM edges as (target, source) pairs for a node.
    pub fn get_derived_from_edges(&self, node_id: NodeId) -> Vec<(NodeId, NodeId)> {
        self.derivations
            .get(&node_id)
            .map_or_else(Vec::new, |records| {
                records
                    .iter()
                    .flat_map(|r| r.sources.iter().map(move |src| (r.target, *src)))
                    .collect()
            })
    }
}

impl Default for ProvenanceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ProvenanceRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProvenanceRecorder")
            .field("total_events", &self.total_events())
            .field("nodes_tracked", &self.nodes_with_events())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId(id)
    }

    #[test]
    fn test_record_and_get_history() {
        let recorder = ProvenanceRecorder::new();
        let n1 = node(1);

        recorder.record_energy_boost(n1, None, 1.0, "test");
        recorder.record_energy_boost(n1, Some(1.0), 2.0, "test");
        recorder.record_energy_boost(n1, Some(2.0), 3.0, "test");

        let history = recorder.get_history(n1);
        assert_eq!(history.len(), 3);
        // All should be EnergyBoost
        for event in &history {
            assert_eq!(event.event_type, CognitiveEventType::EnergyBoost);
            assert_eq!(event.target_node, n1);
        }
    }

    #[test]
    fn test_has_cognitive_event_edges() {
        let recorder = ProvenanceRecorder::new();
        let n1 = node(1);

        recorder.record_energy_boost(n1, None, 1.0, "test");
        recorder.record_scar_creation(n1, 0.5);
        recorder.record_risk_recalc(n1, 0.1, 0.8);

        let edges = recorder.get_edges(n1);
        assert_eq!(edges.len(), 3);
        for (src, _event_id) in &edges {
            assert_eq!(*src, n1);
        }
    }

    #[test]
    fn test_derived_from_edges() {
        let recorder = ProvenanceRecorder::new();
        let target = node(10);
        let sources = vec![node(1), node(2), node(3)];

        recorder.record_derivation(target, sources.clone(), "consolidation");

        let derivations = recorder.get_derivations(target);
        assert_eq!(derivations.len(), 1);
        assert_eq!(derivations[0].sources, sources);

        let edges = recorder.get_derived_from_edges(target);
        assert_eq!(edges.len(), 3);
        for (t, _) in &edges {
            assert_eq!(*t, target);
        }
    }

    #[test]
    fn test_history_by_type() {
        let recorder = ProvenanceRecorder::new();
        let n1 = node(1);

        recorder.record_energy_boost(n1, None, 1.0, "test");
        recorder.record_scar_creation(n1, 0.5);
        recorder.record_energy_boost(n1, Some(1.0), 2.0, "test");

        let boosts = recorder.get_history_by_type(n1, &CognitiveEventType::EnergyBoost);
        assert_eq!(boosts.len(), 2);

        let scars = recorder.get_history_by_type(n1, &CognitiveEventType::ScarCreation);
        assert_eq!(scars.len(), 1);
    }

    #[test]
    fn test_chronological_order() {
        let recorder = ProvenanceRecorder::new();
        let n1 = node(1);

        // Create events with explicit timestamps to verify ordering
        let e1 = CognitiveEvent::new_at(
            CognitiveEventType::EnergyBoost,
            n1,
            None,
            Some(1.0),
            "test",
            100,
        );
        let e2 = CognitiveEvent::new_at(
            CognitiveEventType::ScarCreation,
            n1,
            None,
            Some(0.5),
            "test",
            50, // Earlier timestamp
        );
        let e3 = CognitiveEvent::new_at(
            CognitiveEventType::RiskRecalc,
            n1,
            Some(0.1),
            Some(0.8),
            "test",
            200,
        );

        recorder.record(e1);
        recorder.record(e2);
        recorder.record(e3);

        let history = recorder.get_history(n1);
        assert_eq!(history.len(), 3);
        // Should be sorted by timestamp: 50, 100, 200
        assert_eq!(history[0].timestamp, 50);
        assert_eq!(history[1].timestamp, 100);
        assert_eq!(history[2].timestamp, 200);
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(CognitiveEventType::EnergyBoost.to_string(), "energy_boost");
        assert_eq!(CognitiveEventType::DecaySweep.to_string(), "decay_sweep");
        assert_eq!(
            CognitiveEventType::SynapsePrune.to_string(),
            "synapse_prune"
        );
        assert_eq!(
            CognitiveEventType::ScarCreation.to_string(),
            "scar_creation"
        );
        assert_eq!(CognitiveEventType::RiskRecalc.to_string(), "risk_recalc");
        assert_eq!(
            CognitiveEventType::Consolidation.to_string(),
            "consolidation"
        );
        assert_eq!(
            CognitiveEventType::Custom("foo".into()).to_string(),
            "custom:foo"
        );
    }

    #[test]
    fn test_total_events_and_nodes() {
        let recorder = ProvenanceRecorder::new();

        recorder.record_energy_boost(node(1), None, 1.0, "test");
        recorder.record_energy_boost(node(1), Some(1.0), 2.0, "test");
        recorder.record_scar_creation(node(2), 0.5);

        assert_eq!(recorder.total_events(), 3);
        assert_eq!(recorder.nodes_with_events(), 2);
    }

    #[test]
    fn test_empty_history() {
        let recorder = ProvenanceRecorder::new();
        let history = recorder.get_history(node(999));
        assert!(history.is_empty());
    }

    #[test]
    fn test_decay_sweep_batch_event() {
        let recorder = ProvenanceRecorder::new();
        let n1 = node(1);

        recorder.record_decay_sweep(n1, 5.0, 2.5);

        let history = recorder.get_history(n1);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].event_type, CognitiveEventType::DecaySweep);
        assert_eq!(history[0].old_value, Some(5.0));
        assert_eq!(history[0].new_value, Some(2.5));
        assert_eq!(history[0].trigger_source, "decay_sweep");
    }
}
