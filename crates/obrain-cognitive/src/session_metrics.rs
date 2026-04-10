//! # Session Metrics
//!
//! Captures performance signals from work sessions for kernel adaptation.
//!
//! Each session produces a [`SessionMetric`] node in the graph, linked
//! to the [`KernelParam`]s it measures via `MEASURES` edges. The
//! [`KernelAdaptation`] loop (T3) uses these metrics to adjust parameters.
//!
//! ## Metrics Captured
//!
//! | Metric | Description | Influences |
//! |--------|-------------|------------|
//! | context_noise_ratio | Proportion of injected context unused | propagation_decay |
//! | context_miss_ratio | Proportion of re-requested info | propagation_decay |
//! | resolution_time_ms | Average task resolution time | context_budget_tokens |
//! | propagation_depth_used | Max effective propagation depth | max_hops |
//! | community_boundary_crossed | Cross-community propagation count | community_cohesion_threshold |
//! | token_budget_usage | Ratio tokens used / budget max | context_budget_tokens |

use std::collections::HashMap;

use obrain_common::types::{NodeId, Value};

use crate::engram::traits::{CognitiveFilter, CognitiveNode, CognitiveStorage};
use crate::error::{CognitiveError, CognitiveResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Label used for SessionMetric nodes in the graph.
pub const LABEL_SESSION_METRIC: &str = "SessionMetric";

/// Edge type linking a SessionMetric node to the KernelParam it measures.
pub const EDGE_MEASURES: &str = "MEASURES";

// Property keys for SessionMetric nodes.
const PROP_SESSION_ID: &str = "session_id";
const PROP_TIMESTAMP: &str = "timestamp";
const PROP_CONTEXT_NOISE_RATIO: &str = "context_noise_ratio";
const PROP_CONTEXT_MISS_RATIO: &str = "context_miss_ratio";
const PROP_RESOLUTION_TIME_MS: &str = "resolution_time_ms";
const PROP_PROPAGATION_DEPTH_USED: &str = "propagation_depth_used";
const PROP_COMMUNITY_BOUNDARY_CROSSED: &str = "community_boundary_crossed";
const PROP_TOKEN_BUDGET_USAGE: &str = "token_budget_usage";

/// Mapping from metric field names to the KernelParam names they influence.
pub const METRIC_PARAM_MAP: &[(&str, &[&str])] = &[
    ("context_noise_ratio", &["propagation_decay"]),
    ("context_miss_ratio", &["propagation_decay", "max_hops"]),
    ("resolution_time_ms", &["context_budget_tokens"]),
    ("propagation_depth_used", &["max_hops"]),
    (
        "community_boundary_crossed",
        &["community_cohesion_threshold"],
    ),
    ("token_budget_usage", &["context_budget_tokens"]),
];

// ---------------------------------------------------------------------------
// SessionMetric
// ---------------------------------------------------------------------------

/// A single session metric snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionMetric {
    /// The graph node id (set after persistence).
    pub node_id: Option<NodeId>,
    /// Unique session identifier.
    pub session_id: String,
    /// Timestamp in epoch milliseconds.
    pub timestamp: u64,
    /// Proportion of injected context unused (0.0-1.0).
    pub context_noise_ratio: f64,
    /// Proportion of re-requested info (0.0-1.0).
    pub context_miss_ratio: f64,
    /// Average task resolution time in milliseconds.
    pub resolution_time_ms: f64,
    /// Max effective propagation depth used.
    pub propagation_depth_used: f64,
    /// Number of cross-community boundary crossings.
    pub community_boundary_crossed: f64,
    /// Ratio of tokens used to budget max (0.0-1.0).
    pub token_budget_usage: f64,
}

impl SessionMetric {
    /// Converts this metric into a property map for graph storage.
    fn to_properties(&self) -> HashMap<String, Value> {
        let mut props = HashMap::new();
        props.insert(
            PROP_SESSION_ID.to_string(),
            Value::from(self.session_id.as_str()),
        );
        props.insert(
            PROP_TIMESTAMP.to_string(),
            Value::Int64(self.timestamp as i64),
        );
        props.insert(
            PROP_CONTEXT_NOISE_RATIO.to_string(),
            Value::Float64(self.context_noise_ratio),
        );
        props.insert(
            PROP_CONTEXT_MISS_RATIO.to_string(),
            Value::Float64(self.context_miss_ratio),
        );
        props.insert(
            PROP_RESOLUTION_TIME_MS.to_string(),
            Value::Float64(self.resolution_time_ms),
        );
        props.insert(
            PROP_PROPAGATION_DEPTH_USED.to_string(),
            Value::Float64(self.propagation_depth_used),
        );
        props.insert(
            PROP_COMMUNITY_BOUNDARY_CROSSED.to_string(),
            Value::Float64(self.community_boundary_crossed),
        );
        props.insert(
            PROP_TOKEN_BUDGET_USAGE.to_string(),
            Value::Float64(self.token_budget_usage),
        );
        props
    }

    /// Reconstructs a `SessionMetric` from a `CognitiveNode`.
    fn from_cognitive_node(node: &CognitiveNode) -> Option<Self> {
        let session_id = node
            .properties
            .get(PROP_SESSION_ID)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())?;
        let timestamp = node
            .properties
            .get(PROP_TIMESTAMP)
            .and_then(|v| v.as_int64())? as u64;
        let context_noise_ratio = node
            .properties
            .get(PROP_CONTEXT_NOISE_RATIO)
            .and_then(|v| v.as_float64())?;
        let context_miss_ratio = node
            .properties
            .get(PROP_CONTEXT_MISS_RATIO)
            .and_then(|v| v.as_float64())?;
        let resolution_time_ms = node
            .properties
            .get(PROP_RESOLUTION_TIME_MS)
            .and_then(|v| v.as_float64())?;
        let propagation_depth_used = node
            .properties
            .get(PROP_PROPAGATION_DEPTH_USED)
            .and_then(|v| v.as_float64())?;
        let community_boundary_crossed = node
            .properties
            .get(PROP_COMMUNITY_BOUNDARY_CROSSED)
            .and_then(|v| v.as_float64())?;
        let token_budget_usage = node
            .properties
            .get(PROP_TOKEN_BUDGET_USAGE)
            .and_then(|v| v.as_float64())?;

        Some(Self {
            node_id: Some(node.id),
            session_id,
            timestamp,
            context_noise_ratio,
            context_miss_ratio,
            resolution_time_ms,
            propagation_depth_used,
            community_boundary_crossed,
            token_budget_usage,
        })
    }
}

// ---------------------------------------------------------------------------
// SessionMetricCollector
// ---------------------------------------------------------------------------

/// Accumulates raw observations during a session and finalizes them into a
/// [`SessionMetric`].
#[derive(Debug)]
pub struct SessionMetricCollector {
    // Context noise tracking
    total_injected: usize,
    total_used: usize,

    // Context miss tracking
    miss_count: u32,
    context_request_count: u32,

    // Resolution time tracking
    resolution_times: Vec<u64>,

    // Propagation depth tracking
    max_depth: u32,

    // Boundary crossing tracking
    boundary_crossings: u32,

    // Token budget tracking
    token_samples: Vec<(u32, u32)>, // (used, budget)
}

impl SessionMetricCollector {
    /// Creates a new collector with empty accumulators.
    pub fn new() -> Self {
        Self {
            total_injected: 0,
            total_used: 0,
            miss_count: 0,
            context_request_count: 0,
            resolution_times: Vec::new(),
            max_depth: 0,
            boundary_crossings: 0,
            token_samples: Vec::new(),
        }
    }

    /// Records a context injection, tracking how much was actually used.
    pub fn record_context_use(&mut self, total_injected: usize, used: usize) {
        self.total_injected += total_injected;
        self.total_used += used;
        self.context_request_count += 1;
    }

    /// Records a context miss (re-requested information).
    pub fn record_context_miss(&mut self) {
        self.miss_count += 1;
        self.context_request_count += 1;
    }

    /// Records a task resolution duration.
    pub fn record_resolution(&mut self, duration_ms: u64) {
        self.resolution_times.push(duration_ms);
    }

    /// Records the propagation depth used (keeps the maximum).
    pub fn record_propagation_depth(&mut self, depth: u32) {
        if depth > self.max_depth {
            self.max_depth = depth;
        }
    }

    /// Records a community boundary crossing event.
    pub fn record_boundary_crossing(&mut self) {
        self.boundary_crossings += 1;
    }

    /// Records a token usage sample.
    pub fn record_token_usage(&mut self, used: u32, budget: u32) {
        self.token_samples.push((used, budget));
    }

    /// Finalizes the collector into a [`SessionMetric`].
    ///
    /// Computes ratios and averages from accumulated data.
    pub fn finalize(&self, session_id: &str) -> SessionMetric {
        let context_noise_ratio = if self.total_injected > 0 {
            let unused = self.total_injected.saturating_sub(self.total_used);
            unused as f64 / self.total_injected as f64
        } else {
            0.0
        };

        let context_miss_ratio = if self.context_request_count > 0 {
            self.miss_count as f64 / self.context_request_count as f64
        } else {
            0.0
        };

        let resolution_time_ms = if self.resolution_times.is_empty() {
            0.0
        } else {
            let sum: u64 = self.resolution_times.iter().sum();
            sum as f64 / self.resolution_times.len() as f64
        };

        let propagation_depth_used = f64::from(self.max_depth);

        let community_boundary_crossed = f64::from(self.boundary_crossings);

        let token_budget_usage = if self.token_samples.is_empty() {
            0.0
        } else {
            let sum: f64 = self
                .token_samples
                .iter()
                .map(|&(used, budget)| {
                    if budget > 0 {
                        f64::from(used) / f64::from(budget)
                    } else {
                        0.0
                    }
                })
                .sum();
            (sum / self.token_samples.len() as f64).clamp(0.0, 1.0)
        };

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        SessionMetric {
            node_id: None,
            session_id: session_id.to_string(),
            timestamp,
            context_noise_ratio,
            context_miss_ratio,
            resolution_time_ms,
            propagation_depth_used,
            community_boundary_crossed,
            token_budget_usage,
        }
    }
}

impl Default for SessionMetricCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SessionMetricStore
// ---------------------------------------------------------------------------

/// CRUD operations for `:SessionMetric` nodes via `CognitiveStorage`.
pub struct SessionMetricStore;

impl SessionMetricStore {
    /// Saves a metric to the graph store and returns the created node ID.
    pub fn save_metric(
        storage: &dyn CognitiveStorage,
        metric: &SessionMetric,
    ) -> CognitiveResult<NodeId> {
        let node_id = storage.create_node(LABEL_SESSION_METRIC, &metric.to_properties());
        Ok(node_id)
    }

    /// Retrieves the most recent metrics, up to `limit`.
    ///
    /// Results are sorted by timestamp descending (newest first).
    pub fn get_metrics(
        storage: &dyn CognitiveStorage,
        limit: usize,
    ) -> CognitiveResult<Vec<SessionMetric>> {
        let nodes = storage.query_nodes(LABEL_SESSION_METRIC, None);
        let mut metrics: Vec<SessionMetric> = nodes
            .iter()
            .filter_map(SessionMetric::from_cognitive_node)
            .collect();

        // Sort by timestamp descending
        metrics.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        metrics.truncate(limit);
        Ok(metrics)
    }

    /// Retrieves all metrics with a timestamp >= `since_timestamp`.
    ///
    /// Results are sorted by timestamp descending (newest first).
    pub fn get_metrics_since(
        storage: &dyn CognitiveStorage,
        since_timestamp: u64,
    ) -> CognitiveResult<Vec<SessionMetric>> {
        // Use PropertyGt to filter by timestamp (note: PropertyGt is exclusive >)
        // We want >= so we subtract 1 from the threshold.
        let threshold = if since_timestamp > 0 {
            since_timestamp - 1
        } else {
            0
        };
        let filter = CognitiveFilter::PropertyGt(PROP_TIMESTAMP.to_string(), threshold as f64);
        let nodes = storage.query_nodes(LABEL_SESSION_METRIC, Some(&filter));

        let mut metrics: Vec<SessionMetric> = nodes
            .iter()
            .filter_map(SessionMetric::from_cognitive_node)
            .collect();

        metrics.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(metrics)
    }

    /// Creates a `MEASURES` edge from a session metric node to a KernelParam node.
    ///
    /// The `param_name` is used to find the KernelParam node by its `kernel_name`
    /// property. If no matching param is found, an error is returned.
    pub fn link_to_param(
        storage: &dyn CognitiveStorage,
        metric_node_id: NodeId,
        param_name: &str,
    ) -> CognitiveResult<()> {
        use crate::kernel_params::LABEL_KERNEL_PARAM;

        let filter = CognitiveFilter::PropertyEquals(
            "kernel_name".to_string(),
            Value::from(param_name),
        );
        let param_nodes = storage.query_nodes(LABEL_KERNEL_PARAM, Some(&filter));

        let param_node = param_nodes.first().ok_or_else(|| {
            CognitiveError::Store(format!("kernel param not found: {param_name}"))
        })?;

        storage.create_edge(
            metric_node_id,
            param_node.id,
            EDGE_MEASURES,
            &HashMap::new(),
        );

        Ok(())
    }

    /// Creates `MEASURES` edges for all metric fields based on [`METRIC_PARAM_MAP`].
    ///
    /// This is a convenience method that links a metric node to all the
    /// KernelParam nodes it influences. Errors for missing params are
    /// silently ignored (the param may not have been seeded yet).
    pub fn link_all_params(
        storage: &dyn CognitiveStorage,
        metric_node_id: NodeId,
    ) -> CognitiveResult<()> {
        for (_metric_field, param_names) in METRIC_PARAM_MAP {
            for param_name in *param_names {
                // Best-effort: skip missing params
                let _ = Self::link_to_param(storage, metric_node_id, param_name);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::traits::{CognitiveEdge, CognitiveNode, CognitiveStorage};
    use obrain_common::types::{EdgeId, NodeId};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    // -----------------------------------------------------------------------
    // In-memory mock CognitiveStorage
    // -----------------------------------------------------------------------

    struct MockStorage {
        nodes: Mutex<Vec<CognitiveNode>>,
        edges: Mutex<Vec<(NodeId, NodeId, String)>>,
        next_id: AtomicU64,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                nodes: Mutex::new(Vec::new()),
                edges: Mutex::new(Vec::new()),
                next_id: AtomicU64::new(1),
            }
        }
    }

    impl CognitiveStorage for MockStorage {
        fn create_node(&self, label: &str, properties: &HashMap<String, Value>) -> NodeId {
            let id = NodeId::from(self.next_id.fetch_add(1, Ordering::Relaxed));
            let node = CognitiveNode {
                id,
                label: label.to_string(),
                properties: properties.clone(),
            };
            self.nodes.lock().unwrap().push(node);
            id
        }

        fn create_edge(
            &self,
            from: NodeId,
            to: NodeId,
            rel_type: &str,
            _properties: &HashMap<String, Value>,
        ) -> EdgeId {
            self.edges
                .lock()
                .unwrap()
                .push((from, to, rel_type.to_string()));
            EdgeId::from(self.next_id.fetch_add(1, Ordering::Relaxed))
        }

        fn query_nodes(&self, label: &str, filter: Option<&CognitiveFilter>) -> Vec<CognitiveNode> {
            let nodes = self.nodes.lock().unwrap();
            nodes
                .iter()
                .filter(|n| n.label == label)
                .filter(|n| match filter {
                    None => true,
                    Some(CognitiveFilter::PropertyEquals(key, val)) => {
                        n.properties.get(key).map_or(false, |v| match (v, val) {
                            (Value::String(a), Value::String(b)) => a == b,
                            (Value::Float64(a), Value::Float64(b)) => a == b,
                            (Value::Int64(a), Value::Int64(b)) => a == b,
                            _ => false,
                        })
                    }
                    Some(CognitiveFilter::PropertyGt(key, threshold)) => n
                        .properties
                        .get(key)
                        .and_then(|v| match v {
                            Value::Float64(f) => Some(*f > *threshold),
                            Value::Int64(i) => Some((*i as f64) > *threshold),
                            _ => None,
                        })
                        .unwrap_or(false),
                    _ => true,
                })
                .cloned()
                .collect()
        }

        fn update_node(&self, id: NodeId, properties: &HashMap<String, Value>) {
            let mut nodes = self.nodes.lock().unwrap();
            if let Some(node) = nodes.iter_mut().find(|n| n.id == id) {
                for (k, v) in properties {
                    node.properties.insert(k.clone(), v.clone());
                }
            }
        }

        fn delete_node(&self, id: NodeId) {
            let mut nodes = self.nodes.lock().unwrap();
            nodes.retain(|n| n.id != id);
        }

        fn delete_edge(&self, _id: EdgeId) {}

        fn query_edges(&self, _from: NodeId, _rel_type: &str) -> Vec<CognitiveEdge> {
            Vec::new()
        }
    }

    // -----------------------------------------------------------------------
    // Collector tests
    // -----------------------------------------------------------------------

    #[test]
    fn collector_empty_produces_zeroes() {
        let collector = SessionMetricCollector::new();
        let metric = collector.finalize("test-session");

        assert_eq!(metric.session_id, "test-session");
        assert!((metric.context_noise_ratio - 0.0).abs() < f64::EPSILON);
        assert!((metric.context_miss_ratio - 0.0).abs() < f64::EPSILON);
        assert!((metric.resolution_time_ms - 0.0).abs() < f64::EPSILON);
        assert!((metric.propagation_depth_used - 0.0).abs() < f64::EPSILON);
        assert!((metric.community_boundary_crossed - 0.0).abs() < f64::EPSILON);
        assert!((metric.token_budget_usage - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn collector_context_noise_ratio() {
        let mut collector = SessionMetricCollector::new();
        // Injected 100 tokens, used 30 => noise = 70/100 = 0.7
        collector.record_context_use(100, 30);
        let metric = collector.finalize("s1");
        assert!((metric.context_noise_ratio - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn collector_context_noise_accumulates() {
        let mut collector = SessionMetricCollector::new();
        collector.record_context_use(100, 50); // 50 unused
        collector.record_context_use(200, 150); // 50 unused
        // Total: 300 injected, 200 used => noise = 100/300 = 0.333...
        let metric = collector.finalize("s2");
        assert!((metric.context_noise_ratio - 100.0 / 300.0).abs() < 1e-10);
    }

    #[test]
    fn collector_context_miss_ratio() {
        let mut collector = SessionMetricCollector::new();
        collector.record_context_use(10, 5); // request_count = 1
        collector.record_context_miss(); // request_count = 2, miss = 1
        collector.record_context_miss(); // request_count = 3, miss = 2
        // miss_ratio = 2/3
        let metric = collector.finalize("s3");
        assert!((metric.context_miss_ratio - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn collector_resolution_time_average() {
        let mut collector = SessionMetricCollector::new();
        collector.record_resolution(100);
        collector.record_resolution(200);
        collector.record_resolution(300);
        let metric = collector.finalize("s4");
        assert!((metric.resolution_time_ms - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn collector_propagation_depth_takes_max() {
        let mut collector = SessionMetricCollector::new();
        collector.record_propagation_depth(2);
        collector.record_propagation_depth(5);
        collector.record_propagation_depth(3);
        let metric = collector.finalize("s5");
        assert!((metric.propagation_depth_used - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn collector_boundary_crossings_accumulate() {
        let mut collector = SessionMetricCollector::new();
        collector.record_boundary_crossing();
        collector.record_boundary_crossing();
        collector.record_boundary_crossing();
        let metric = collector.finalize("s6");
        assert!((metric.community_boundary_crossed - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn collector_token_budget_usage_average() {
        let mut collector = SessionMetricCollector::new();
        collector.record_token_usage(500, 1000); // 0.5
        collector.record_token_usage(800, 1000); // 0.8
        // average = (0.5 + 0.8) / 2 = 0.65
        let metric = collector.finalize("s7");
        assert!((metric.token_budget_usage - 0.65).abs() < 1e-10);
    }

    #[test]
    fn collector_token_budget_zero_budget_safe() {
        let mut collector = SessionMetricCollector::new();
        collector.record_token_usage(100, 0); // budget=0 => treated as 0.0
        let metric = collector.finalize("s8");
        assert!((metric.token_budget_usage - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Store roundtrip tests
    // -----------------------------------------------------------------------

    #[test]
    fn save_and_get_metric_roundtrip() {
        let storage = MockStorage::new();
        let metric = SessionMetric {
            node_id: None,
            session_id: "session-42".to_string(),
            timestamp: 1700000000000,
            context_noise_ratio: 0.3,
            context_miss_ratio: 0.1,
            resolution_time_ms: 150.0,
            propagation_depth_used: 3.0,
            community_boundary_crossed: 2.0,
            token_budget_usage: 0.75,
        };

        let node_id = SessionMetricStore::save_metric(&storage, &metric).unwrap();
        assert_eq!(node_id, NodeId::from(1_u64));

        let retrieved = SessionMetricStore::get_metrics(&storage, 10).unwrap();
        assert_eq!(retrieved.len(), 1);

        let r = &retrieved[0];
        assert_eq!(r.session_id, "session-42");
        assert_eq!(r.timestamp, 1700000000000);
        assert!((r.context_noise_ratio - 0.3).abs() < f64::EPSILON);
        assert!((r.context_miss_ratio - 0.1).abs() < f64::EPSILON);
        assert!((r.resolution_time_ms - 150.0).abs() < f64::EPSILON);
        assert!((r.propagation_depth_used - 3.0).abs() < f64::EPSILON);
        assert!((r.community_boundary_crossed - 2.0).abs() < f64::EPSILON);
        assert!((r.token_budget_usage - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn get_metrics_respects_limit() {
        let storage = MockStorage::new();

        for i in 0..5 {
            let metric = SessionMetric {
                node_id: None,
                session_id: format!("s-{i}"),
                timestamp: 1000 + i as u64,
                context_noise_ratio: 0.0,
                context_miss_ratio: 0.0,
                resolution_time_ms: 0.0,
                propagation_depth_used: 0.0,
                community_boundary_crossed: 0.0,
                token_budget_usage: 0.0,
            };
            SessionMetricStore::save_metric(&storage, &metric).unwrap();
        }

        let retrieved = SessionMetricStore::get_metrics(&storage, 3).unwrap();
        assert_eq!(retrieved.len(), 3);
        // Should be sorted newest first
        assert_eq!(retrieved[0].timestamp, 1004);
        assert_eq!(retrieved[1].timestamp, 1003);
        assert_eq!(retrieved[2].timestamp, 1002);
    }

    #[test]
    fn get_metrics_since_filters_by_timestamp() {
        let storage = MockStorage::new();

        for i in 0..5 {
            let metric = SessionMetric {
                node_id: None,
                session_id: format!("s-{i}"),
                timestamp: 1000 + i as u64 * 100,
                context_noise_ratio: 0.0,
                context_miss_ratio: 0.0,
                resolution_time_ms: 0.0,
                propagation_depth_used: 0.0,
                community_boundary_crossed: 0.0,
                token_budget_usage: 0.0,
            };
            SessionMetricStore::save_metric(&storage, &metric).unwrap();
        }

        // Timestamps: 1000, 1100, 1200, 1300, 1400
        // since 1200 => should get 1200, 1300, 1400
        let retrieved = SessionMetricStore::get_metrics_since(&storage, 1200).unwrap();
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0].timestamp, 1400);
        assert_eq!(retrieved[1].timestamp, 1300);
        assert_eq!(retrieved[2].timestamp, 1200);
    }

    #[test]
    fn link_to_param_creates_edge() {
        let storage = MockStorage::new();

        // Create a KernelParam node manually
        let mut props = HashMap::new();
        props.insert(
            "kernel_name".to_string(),
            Value::from("propagation_decay"),
        );
        props.insert("kernel_value".to_string(), Value::Float64(0.3));
        props.insert("kernel_min".to_string(), Value::Float64(0.05));
        props.insert("kernel_max".to_string(), Value::Float64(0.9));
        props.insert("kernel_lr".to_string(), Value::Float64(0.1));
        props.insert("kernel_adjustment_count".to_string(), Value::Int64(0));
        let _param_id = storage.create_node("KernelParam", &props);

        // Create a session metric node
        let metric = SessionMetric {
            node_id: None,
            session_id: "link-test".to_string(),
            timestamp: 5000,
            context_noise_ratio: 0.5,
            context_miss_ratio: 0.0,
            resolution_time_ms: 0.0,
            propagation_depth_used: 0.0,
            community_boundary_crossed: 0.0,
            token_budget_usage: 0.0,
        };
        let metric_id = SessionMetricStore::save_metric(&storage, &metric).unwrap();

        // Link it
        SessionMetricStore::link_to_param(&storage, metric_id, "propagation_decay").unwrap();

        // Verify edge was created
        let edges = storage.edges.lock().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, "MEASURES");
    }

    #[test]
    fn link_to_param_errors_on_missing() {
        let storage = MockStorage::new();

        let metric = SessionMetric {
            node_id: None,
            session_id: "err-test".to_string(),
            timestamp: 6000,
            context_noise_ratio: 0.0,
            context_miss_ratio: 0.0,
            resolution_time_ms: 0.0,
            propagation_depth_used: 0.0,
            community_boundary_crossed: 0.0,
            token_budget_usage: 0.0,
        };
        let metric_id = SessionMetricStore::save_metric(&storage, &metric).unwrap();

        let result =
            SessionMetricStore::link_to_param(&storage, metric_id, "nonexistent_param");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Metric-param mapping test
    // -----------------------------------------------------------------------

    #[test]
    fn metric_param_map_covers_all_metrics() {
        let expected_metrics = [
            "context_noise_ratio",
            "context_miss_ratio",
            "resolution_time_ms",
            "propagation_depth_used",
            "community_boundary_crossed",
            "token_budget_usage",
        ];

        for expected in &expected_metrics {
            let found = METRIC_PARAM_MAP.iter().any(|(m, _)| m == expected);
            assert!(found, "METRIC_PARAM_MAP missing entry for: {expected}");
        }
    }

    #[test]
    fn metric_param_map_references_valid_params() {
        // All param names referenced in the map should be known kernel param names.
        let known_params = [
            "propagation_decay",
            "community_cohesion_threshold",
            "max_hops",
            "min_propagated_energy",
            "cristallization_sessions",
            "cristallization_energy",
            "dissolution_hit_rate",
            "context_budget_tokens",
            "kernel_learning_rate",
        ];

        for (_metric, param_names) in METRIC_PARAM_MAP {
            for param_name in *param_names {
                assert!(
                    known_params.contains(param_name),
                    "METRIC_PARAM_MAP references unknown param: {param_name}"
                );
            }
        }
    }

    #[test]
    fn default_collector_is_same_as_new() {
        let c1 = SessionMetricCollector::new();
        let c2 = SessionMetricCollector::default();
        let m1 = c1.finalize("a");
        let m2 = c2.finalize("b");
        // Same ratios (both zero)
        assert!((m1.context_noise_ratio - m2.context_noise_ratio).abs() < f64::EPSILON);
    }
}
