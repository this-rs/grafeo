//! Built-in procedure registry for CALL statement execution.
//!
//! Maps procedure names to [`GraphAlgorithm`] implementations, enabling
//! `CALL grafeo.pagerank({damping: 0.85}) YIELD nodeId, score` from any
//! supported query language (GQL, Cypher, SQL/PGQ).

use std::sync::Arc;

use grafeo_adapters::plugins::algorithms::{
    ArticulationPointsAlgorithm, BellmanFordAlgorithm, BetweennessCentralityAlgorithm,
    BfsAlgorithm, BridgesAlgorithm, ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm,
    ConnectedComponentsAlgorithm, DegreeCentralityAlgorithm, DfsAlgorithm, DijkstraAlgorithm,
    FloydWarshallAlgorithm, GraphAlgorithm, HitsAlgorithm, KCoreAlgorithm, KHopAlgorithm,
    KruskalAlgorithm, LabelPropagationAlgorithm, LeidenAlgorithm, LouvainAlgorithm,
    MaxFlowAlgorithm, MinCostFlowAlgorithm, NodeSimilarityAlgorithm, PageRankAlgorithm,
    PrimAlgorithm, ProjectionConfig, ProjectionRegistry, SsspAlgorithm,
    StronglyConnectedComponentsAlgorithm, TopKSimilarAlgorithm, TopologicalSortAlgorithm,
};
use grafeo_adapters::plugins::{AlgorithmResult, ParameterDef, Parameters};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use hashbrown::HashMap;

use crate::query::plan::LogicalExpression;

/// Registry of built-in procedures backed by graph algorithms.
pub struct BuiltinProcedures {
    algorithms: HashMap<String, Arc<dyn GraphAlgorithm>>,
}

impl BuiltinProcedures {
    /// Creates a new registry with all built-in algorithms registered.
    pub fn new() -> Self {
        let mut algorithms: HashMap<String, Arc<dyn GraphAlgorithm>> = HashMap::new();
        let register = |map: &mut HashMap<String, Arc<dyn GraphAlgorithm>>,
                        algo: Arc<dyn GraphAlgorithm>| {
            map.insert(algo.name().to_string(), algo);
        };

        // Centrality
        register(&mut algorithms, Arc::new(PageRankAlgorithm));
        register(&mut algorithms, Arc::new(BetweennessCentralityAlgorithm));
        register(&mut algorithms, Arc::new(ClosenessCentralityAlgorithm));
        register(&mut algorithms, Arc::new(DegreeCentralityAlgorithm));
        register(&mut algorithms, Arc::new(HitsAlgorithm));

        // Traversal
        register(&mut algorithms, Arc::new(BfsAlgorithm));
        register(&mut algorithms, Arc::new(DfsAlgorithm));

        // Components
        register(&mut algorithms, Arc::new(ConnectedComponentsAlgorithm));
        register(
            &mut algorithms,
            Arc::new(StronglyConnectedComponentsAlgorithm),
        );
        register(&mut algorithms, Arc::new(TopologicalSortAlgorithm));

        // Shortest Path
        register(&mut algorithms, Arc::new(DijkstraAlgorithm));
        register(&mut algorithms, Arc::new(SsspAlgorithm));
        register(&mut algorithms, Arc::new(BellmanFordAlgorithm));
        register(&mut algorithms, Arc::new(FloydWarshallAlgorithm));

        // Clustering
        register(&mut algorithms, Arc::new(ClusteringCoefficientAlgorithm));

        // Community
        register(&mut algorithms, Arc::new(LabelPropagationAlgorithm));
        register(&mut algorithms, Arc::new(LouvainAlgorithm));
        register(&mut algorithms, Arc::new(LeidenAlgorithm));

        // MST
        register(&mut algorithms, Arc::new(KruskalAlgorithm));
        register(&mut algorithms, Arc::new(PrimAlgorithm));

        // Flow
        register(&mut algorithms, Arc::new(MaxFlowAlgorithm));
        register(&mut algorithms, Arc::new(MinCostFlowAlgorithm));

        // Structure
        register(&mut algorithms, Arc::new(ArticulationPointsAlgorithm));
        register(&mut algorithms, Arc::new(BridgesAlgorithm));
        register(&mut algorithms, Arc::new(KCoreAlgorithm));

        // Subgraph extraction
        register(&mut algorithms, Arc::new(KHopAlgorithm));

        // Node similarity
        register(&mut algorithms, Arc::new(NodeSimilarityAlgorithm));
        register(&mut algorithms, Arc::new(TopKSimilarAlgorithm));

        Self { algorithms }
    }

    /// Resolves a procedure name to its algorithm.
    ///
    /// Strips `"grafeo."` prefix if present:
    /// - `["grafeo", "pagerank"]` → looks up `"pagerank"`
    /// - `["pagerank"]` → looks up `"pagerank"`
    pub fn get(&self, name: &[String]) -> Option<Arc<dyn GraphAlgorithm>> {
        let key = resolve_name(name);
        self.algorithms.get(&key).cloned()
    }

    /// Returns info for all registered procedures.
    pub fn list(&self) -> Vec<ProcedureInfo> {
        let mut result: Vec<ProcedureInfo> = self
            .algorithms
            .values()
            .map(|algo| ProcedureInfo {
                name: format!("grafeo.{}", algo.name()),
                description: algo.description().to_string(),
                parameters: algo.parameters().to_vec(),
                output_columns: output_columns_for(algo.as_ref()),
            })
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }
}

impl Default for BuiltinProcedures {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a registered procedure.
pub struct ProcedureInfo {
    /// Qualified name (e.g., `"grafeo.pagerank"`).
    pub name: String,
    /// Description of what the procedure does.
    pub description: String,
    /// Parameter definitions.
    pub parameters: Vec<ParameterDef>,
    /// Output column names.
    pub output_columns: Vec<String>,
}

/// Resolves a dotted procedure name to its lookup key.
///
/// Strips the `"grafeo"` namespace prefix if present.
/// Supports multi-level names like `["grafeo", "subgraph", "khop"]` → `"subgraph.khop"`.
fn resolve_name(parts: &[String]) -> String {
    if parts.is_empty() {
        return String::new();
    }
    if parts[0].eq_ignore_ascii_case("grafeo") && parts.len() > 1 {
        parts[1..].join(".")
    } else {
        parts.join(".")
    }
}

/// Infers output column names from an algorithm.
///
/// Returns the standard column names for known algorithm categories.
pub fn output_columns_for_name(algo: &dyn GraphAlgorithm) -> Vec<String> {
    output_columns_for(algo)
}

/// Canonical output column names for each algorithm.
///
/// These must match the actual column count from each algorithm's `execute()`,
/// providing user-friendly names (e.g., `"score"` instead of `"pagerank"`).
fn output_columns_for(algo: &dyn GraphAlgorithm) -> Vec<String> {
    match algo.name() {
        "pagerank" => vec!["node_id".into(), "score".into()],
        "hits" => vec![
            "node_id".into(),
            "hub_score".into(),
            "authority_score".into(),
        ],
        "betweenness_centrality" => vec!["node_id".into(), "centrality".into()],
        "closeness_centrality" => vec!["node_id".into(), "centrality".into()],
        "degree_centrality" => {
            vec![
                "node_id".into(),
                "in_degree".into(),
                "out_degree".into(),
                "total_degree".into(),
            ]
        }
        "bfs" => vec!["node_id".into(), "depth".into()],
        "dfs" => vec!["node_id".into(), "depth".into()],
        "connected_components" | "strongly_connected_components" => {
            vec!["node_id".into(), "component_id".into()]
        }
        "topological_sort" => vec!["node_id".into(), "order".into()],
        "dijkstra" | "sssp" => vec!["node_id".into(), "distance".into()],
        "bellman_ford" => vec![
            "node_id".into(),
            "distance".into(),
            "has_negative_cycle".into(),
        ],
        "floyd_warshall" => vec!["source".into(), "target".into(), "distance".into()],
        "clustering_coefficient" => {
            vec![
                "node_id".into(),
                "coefficient".into(),
                "triangle_count".into(),
            ]
        }
        "label_propagation" => vec!["node_id".into(), "community_id".into()],
        "louvain" | "leiden" => vec!["node_id".into(), "community_id".into(), "modularity".into()],
        "kruskal" | "prim" => vec!["source".into(), "target".into(), "weight".into()],
        "max_flow" => {
            vec![
                "source".into(),
                "target".into(),
                "flow".into(),
                "max_flow".into(),
            ]
        }
        "min_cost_max_flow" => {
            vec![
                "source".into(),
                "target".into(),
                "flow".into(),
                "cost".into(),
                "max_flow".into(),
            ]
        }
        "articulation_points" => vec!["node_id".into()],
        "bridges" => vec!["source".into(), "target".into()],
        "k_core" => vec!["node_id".into(), "core_number".into(), "max_core".into()],
        "similarity" => vec![
            "node1".into(),
            "node2".into(),
            "metric".into(),
            "score".into(),
        ],
        "similarity.topk" => vec!["neighbor".into(), "score".into()],
        "subgraph.khop" => vec![
            "node_id".into(),
            "hop".into(),
            "source".into(),
            "target".into(),
            "edge_type".into(),
            "weight".into(),
        ],
        _ => vec!["node_id".into(), "value".into()],
    }
}

/// Converts logical expression arguments into algorithm [`Parameters`].
///
/// Supports two patterns:
/// 1. **Map literal**: `{damping: 0.85, iterations: 20}` → named parameters
/// 2. **Positional args**: `(42, 'weight')` → mapped by index to `ParameterDef` names
pub fn evaluate_arguments(args: &[LogicalExpression], param_defs: &[ParameterDef]) -> Parameters {
    let mut params = Parameters::new();

    if args.len() == 1
        && let LogicalExpression::Map(entries) = &args[0]
    {
        // Map literal: {damping: 0.85, iterations: 20}
        for (key, value_expr) in entries {
            set_param_from_expression(&mut params, key, value_expr);
        }
        return params;
    }

    // Positional arguments: map by index to parameter definitions
    for (i, arg) in args.iter().enumerate() {
        if let Some(def) = param_defs.get(i) {
            set_param_from_expression(&mut params, &def.name, arg);
        }
    }

    params
}

/// Sets a parameter from a `LogicalExpression` constant value.
fn set_param_from_expression(params: &mut Parameters, name: &str, expr: &LogicalExpression) {
    match expr {
        LogicalExpression::Literal(Value::Int64(v)) => params.set_int(name, *v),
        LogicalExpression::Literal(Value::Float64(v)) => params.set_float(name, *v),
        LogicalExpression::Literal(Value::String(v)) => {
            params.set_string(name, AsRef::<str>::as_ref(v));
        }
        LogicalExpression::Literal(Value::Bool(v)) => params.set_bool(name, *v),
        _ => {} // Non-constant expressions are ignored in Phase 1
    }
}

/// Builds a `grafeo.procedures()` result listing all registered procedures.
pub fn procedures_result(registry: &BuiltinProcedures) -> AlgorithmResult {
    let procedures = registry.list();
    let mut result = AlgorithmResult::new(vec![
        "name".into(),
        "description".into(),
        "parameters".into(),
        "output_columns".into(),
    ]);
    for proc in procedures {
        let param_desc: String = proc
            .parameters
            .iter()
            .map(|p| {
                if p.required {
                    format!("{} ({:?})", p.name, p.param_type)
                } else if let Some(ref default) = p.default {
                    format!("{} ({:?}, default={})", p.name, p.param_type, default)
                } else {
                    format!("{} ({:?}, optional)", p.name, p.param_type)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        let columns_desc = proc.output_columns.join(", ");

        result.add_row(vec![
            Value::from(proc.name.as_str()),
            Value::from(proc.description.as_str()),
            Value::from(param_desc.as_str()),
            Value::from(columns_desc.as_str()),
        ]);
    }
    result
}

// ============================================================================
// Projection Procedures
// ============================================================================

/// Global projection registry, shared across all sessions.
static PROJECTION_REGISTRY: std::sync::OnceLock<ProjectionRegistry> = std::sync::OnceLock::new();

/// Returns the global projection registry.
pub fn projection_registry() -> &'static ProjectionRegistry {
    PROJECTION_REGISTRY.get_or_init(ProjectionRegistry::new)
}

/// Checks if a procedure name matches a projection procedure and executes it.
///
/// Handles:
/// - `grafeo.projection.create(name, {node_labels: [...], edge_types: [...]})`
/// - `grafeo.projection.drop(name)`
/// - `grafeo.projection.list()`
///
/// Returns `Some(AlgorithmResult)` if the name matched, `None` otherwise.
pub fn try_execute_projection_procedure(
    resolved_name: &str,
    arguments: &[LogicalExpression],
) -> Result<Option<AlgorithmResult>> {
    match resolved_name {
        "grafeo.projection.create" | "projection.create" => {
            let result = execute_projection_create(arguments)?;
            Ok(Some(result))
        }
        "grafeo.projection.drop" | "projection.drop" => {
            let result = execute_projection_drop(arguments)?;
            Ok(Some(result))
        }
        "grafeo.projection.list" | "projection.list" => {
            let result = execute_projection_list();
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// `CALL grafeo.projection.create(name, {node_labels: [...], edge_types: [...]})`.
///
/// Creates a named projection in the global registry.
fn execute_projection_create(arguments: &[LogicalExpression]) -> Result<AlgorithmResult> {
    // First argument: projection name (string)
    let name = match arguments.first() {
        Some(LogicalExpression::Literal(Value::String(s))) => s.to_string(),
        Some(_) => {
            return Err(Error::Internal(
                "grafeo.projection.create(): first argument must be a projection name (string)"
                    .to_string(),
            ));
        }
        None => {
            return Err(Error::Internal(
                "grafeo.projection.create(): requires at least 1 argument (name)".to_string(),
            ));
        }
    };

    // Second argument: optional config map
    let mut config = ProjectionConfig::new(&name);

    if let Some(LogicalExpression::Map(entries)) = arguments.get(1) {
        for (key, value_expr) in entries {
            match key.as_str() {
                "node_labels" => {
                    if let LogicalExpression::List(items) = value_expr {
                        let labels: Vec<String> = items
                            .iter()
                            .filter_map(|item| {
                                if let LogicalExpression::Literal(Value::String(s)) = item {
                                    Some(s.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        config = config.node_labels(labels);
                    }
                }
                "edge_types" => {
                    if let LogicalExpression::List(items) = value_expr {
                        let types: Vec<String> = items
                            .iter()
                            .filter_map(|item| {
                                if let LogicalExpression::Literal(Value::String(s)) = item {
                                    Some(s.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        config = config.edge_types(types);
                    }
                }
                _ => {} // Ignore unknown keys
            }
        }
    }

    let registry = projection_registry();
    registry.create(config).map_err(Error::Internal)?;

    let mut result = AlgorithmResult::new(vec!["name".to_string(), "status".to_string()]);
    result.add_row(vec![Value::from(name.as_str()), Value::from("created")]);
    Ok(result)
}

/// `CALL grafeo.projection.drop(name)`.
///
/// Removes a named projection from the global registry.
fn execute_projection_drop(arguments: &[LogicalExpression]) -> Result<AlgorithmResult> {
    let name = match arguments.first() {
        Some(LogicalExpression::Literal(Value::String(s))) => s.to_string(),
        Some(_) => {
            return Err(Error::Internal(
                "grafeo.projection.drop(): first argument must be a projection name (string)"
                    .to_string(),
            ));
        }
        None => {
            return Err(Error::Internal(
                "grafeo.projection.drop(): requires 1 argument (name)".to_string(),
            ));
        }
    };

    let registry = projection_registry();
    let dropped = registry.drop_projection(&name);

    let mut result = AlgorithmResult::new(vec!["name".to_string(), "status".to_string()]);
    result.add_row(vec![
        Value::from(name.as_str()),
        Value::from(if dropped { "dropped" } else { "not_found" }),
    ]);
    Ok(result)
}

/// `CALL grafeo.projection.list()`.
///
/// Lists all named projections in the global registry.
fn execute_projection_list() -> AlgorithmResult {
    let registry = projection_registry();
    let projections = registry.list();

    let mut result = AlgorithmResult::new(vec![
        "name".to_string(),
        "node_labels".to_string(),
        "edge_types".to_string(),
    ]);

    for config in projections {
        result.add_row(vec![
            Value::from(config.name()),
            Value::from(config.node_labels_ref().join(", ").as_str()),
            Value::from(config.edge_types_ref().join(", ").as_str()),
        ]);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_algorithms() {
        let registry = BuiltinProcedures::new();
        let list = registry.list();
        assert!(
            list.len() >= 25,
            "Expected at least 25 algorithms, got {}",
            list.len()
        );
    }

    #[test]
    fn test_resolve_with_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[test]
    fn test_resolve_without_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["pagerank".to_string()];
        assert!(registry.get(&name).is_some());
    }

    #[test]
    fn test_resolve_unknown() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "nonexistent".to_string()];
        assert!(registry.get(&name).is_none());
    }

    #[test]
    fn test_evaluate_map_arguments() {
        let args = vec![LogicalExpression::Map(vec![
            (
                "damping".to_string(),
                LogicalExpression::Literal(Value::Float64(0.85)),
            ),
            (
                "max_iterations".to_string(),
                LogicalExpression::Literal(Value::Int64(20)),
            ),
        ])];
        let params = evaluate_arguments(&args, &[]);
        assert_eq!(params.get_float("damping"), Some(0.85));
        assert_eq!(params.get_int("max_iterations"), Some(20));
    }

    #[test]
    fn test_evaluate_empty_arguments() {
        let params = evaluate_arguments(&[], &[]);
        assert_eq!(params.get_float("damping"), None);
    }

    #[test]
    fn test_procedures_result() {
        let registry = BuiltinProcedures::new();
        let result = procedures_result(&registry);
        assert_eq!(
            result.columns,
            vec!["name", "description", "parameters", "output_columns"]
        );
        assert!(result.rows.len() >= 25);
    }

    #[test]
    fn test_khop_registered() {
        let registry = BuiltinProcedures::new();
        // Resolve via dotted namespace: grafeo.subgraph.khop
        let name = vec![
            "grafeo".to_string(),
            "subgraph".to_string(),
            "khop".to_string(),
        ];
        let algo = registry.get(&name);
        assert!(algo.is_some(), "subgraph.khop should be registered");
        assert_eq!(algo.unwrap().name(), "subgraph.khop");
    }

    #[test]
    fn test_khop_resolve_without_namespace() {
        let registry = BuiltinProcedures::new();
        let name = vec!["subgraph".to_string(), "khop".to_string()];
        let algo = registry.get(&name);
        assert!(
            algo.is_some(),
            "subgraph.khop should resolve without grafeo prefix"
        );
    }

    #[test]
    fn test_khop_execute_via_registry() {
        use grafeo_core::graph::lpg::LpgStore;

        let registry = BuiltinProcedures::new();
        let name = vec![
            "grafeo".to_string(),
            "subgraph".to_string(),
            "khop".to_string(),
        ];
        let algo = registry.get(&name).unwrap();

        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Person"]);
        let n1 = store.create_node(&["Person"]);
        store.create_edge(n0, n1, "KNOWS");

        let mut params = Parameters::new();
        params.set_int("center", n0.0 as i64);
        params.set_int("k", 1);

        let result = algo.execute(&store, &params).unwrap();
        // Should have node rows (2 nodes) + edge rows (1 edge)
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_registry_count_includes_khop() {
        let registry = BuiltinProcedures::new();
        let list = registry.list();
        assert!(
            list.len() >= 26,
            "Expected at least 26 algorithms (22 + khop + leiden + similarity + similarity.topk), got {}",
            list.len()
        );
        assert!(
            list.iter().any(|p| p.name == "grafeo.subgraph.khop"),
            "grafeo.subgraph.khop should be in the procedure list"
        );
    }

    #[test]
    fn test_leiden_registered() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "leiden".to_string()];
        let algo = registry.get(&name);
        assert!(algo.is_some(), "leiden should be registered");
        assert_eq!(algo.unwrap().name(), "leiden");
    }

    #[test]
    fn test_leiden_execute_via_registry() {
        use grafeo_core::graph::lpg::LpgStore;

        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "leiden".to_string()];
        let algo = registry.get(&name).unwrap();

        let store = LpgStore::new().unwrap();
        let n0 = store.create_node(&["Person"]);
        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        store.create_edge(n0, n1, "KNOWS");
        store.create_edge(n1, n0, "KNOWS");
        store.create_edge(n1, n2, "KNOWS");
        store.create_edge(n2, n1, "KNOWS");

        let mut params = Parameters::new();
        params.set_float("resolution", 1.0);
        params.set_float("gamma", 0.01);

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.rows.len(), 3);
        assert_eq!(
            result.columns,
            vec!["node_id", "community_id", "modularity"]
        );
    }

    #[test]
    fn test_leiden_output_columns() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "leiden".to_string()];
        let algo = registry.get(&name).unwrap();
        let cols = output_columns_for(algo.as_ref());
        assert_eq!(cols, vec!["node_id", "community_id", "modularity"]);
    }

    // ========================================================================
    // Projection procedure tests
    // ========================================================================

    // ========================================================================
    // Similarity procedure tests
    // ========================================================================

    #[test]
    fn test_similarity_registered() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "similarity".to_string()];
        let algo = registry.get(&name);
        assert!(algo.is_some(), "similarity should be registered");
        assert_eq!(algo.unwrap().name(), "similarity");
    }

    #[test]
    fn test_similarity_topk_registered() {
        let registry = BuiltinProcedures::new();
        let name = vec![
            "grafeo".to_string(),
            "similarity".to_string(),
            "topk".to_string(),
        ];
        let algo = registry.get(&name);
        assert!(algo.is_some(), "similarity.topk should be registered");
        assert_eq!(algo.unwrap().name(), "similarity.topk");
    }

    #[test]
    fn test_similarity_execute_via_registry() {
        use grafeo_core::graph::lpg::LpgStore;

        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "similarity".to_string()];
        let algo = registry.get(&name).unwrap();

        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        // A-C and B-C (common neighbor C)
        store.create_edge(a, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, b, "KNOWS");

        let mut params = Parameters::new();
        params.set_int("node1", a.0 as i64);
        params.set_int("node2", b.0 as i64);
        params.set_string("metric", "jaccard");

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["node1", "node2", "metric", "score"]);
        assert_eq!(result.rows.len(), 1);
        // N(A) = {C}, N(B) = {C}, intersection={C}, union={C}, jaccard=1.0
        if let Value::Float64(score) = &result.rows[0][3] {
            assert!((*score - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_similarity_topk_execute_via_registry() {
        use grafeo_core::graph::lpg::LpgStore;

        let registry = BuiltinProcedures::new();
        let name = vec![
            "grafeo".to_string(),
            "similarity".to_string(),
            "topk".to_string(),
        ];
        let algo = registry.get(&name).unwrap();

        let store = LpgStore::new().unwrap();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        store.create_edge(a, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, b, "KNOWS");

        let mut params = Parameters::new();
        params.set_int("node", a.0 as i64);
        params.set_int("k", 5);
        params.set_string("metric", "jaccard");

        let result = algo.execute(&store, &params).unwrap();
        assert_eq!(result.columns, vec!["neighbor", "score"]);
        assert!(!result.rows.is_empty());
    }

    #[test]
    fn test_similarity_output_columns() {
        let registry = BuiltinProcedures::new();
        let name = vec!["grafeo".to_string(), "similarity".to_string()];
        let algo = registry.get(&name).unwrap();
        let cols = output_columns_for(algo.as_ref());
        assert_eq!(cols, vec!["node1", "node2", "metric", "score"]);
    }

    #[test]
    fn test_similarity_topk_output_columns() {
        let registry = BuiltinProcedures::new();
        let name = vec![
            "grafeo".to_string(),
            "similarity".to_string(),
            "topk".to_string(),
        ];
        let algo = registry.get(&name).unwrap();
        let cols = output_columns_for(algo.as_ref());
        assert_eq!(cols, vec!["neighbor", "score"]);
    }

    // ========================================================================
    // Projection procedure tests
    // ========================================================================

    #[test]
    fn test_projection_create_procedure() {
        let args = vec![
            LogicalExpression::Literal(Value::from("test_proj")),
            LogicalExpression::Map(vec![
                (
                    "node_labels".to_string(),
                    LogicalExpression::List(vec![
                        LogicalExpression::Literal(Value::from("File")),
                        LogicalExpression::Literal(Value::from("Function")),
                    ]),
                ),
                (
                    "edge_types".to_string(),
                    LogicalExpression::List(vec![LogicalExpression::Literal(Value::from(
                        "IMPORTS",
                    ))]),
                ),
            ]),
        ];

        let result = try_execute_projection_procedure("grafeo.projection.create", &args).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.columns, vec!["name", "status"]);
        assert_eq!(result.rows.len(), 1);

        // Verify it's in the registry
        let config = projection_registry().get("test_proj").unwrap();
        assert_eq!(config.node_labels_ref(), &["File", "Function"]);
        assert_eq!(config.edge_types_ref(), &["IMPORTS"]);

        // Cleanup
        projection_registry().drop_projection("test_proj");
    }

    #[test]
    fn test_projection_drop_procedure() {
        // Create first
        let registry = projection_registry();
        let _ = registry.create(ProjectionConfig::new("drop_test"));

        let args = vec![LogicalExpression::Literal(Value::from("drop_test"))];
        let result = try_execute_projection_procedure("grafeo.projection.drop", &args).unwrap();
        assert!(result.is_some());

        // Should be gone
        assert!(registry.get("drop_test").is_none());
    }

    #[test]
    fn test_projection_list_procedure() {
        let registry = projection_registry();
        let _ = registry.create(ProjectionConfig::new("list_test_a"));
        let _ = registry.create(ProjectionConfig::new("list_test_b"));

        let result = try_execute_projection_procedure("grafeo.projection.list", &[]).unwrap();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.columns, vec!["name", "node_labels", "edge_types"]);
        assert!(result.rows.len() >= 2);

        // Cleanup
        registry.drop_projection("list_test_a");
        registry.drop_projection("list_test_b");
    }

    #[test]
    fn test_projection_procedure_no_match() {
        let result = try_execute_projection_procedure("grafeo.something_else", &[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_projection_create_missing_name() {
        let result = try_execute_projection_procedure("grafeo.projection.create", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_projection_drop_missing_name() {
        let result = try_execute_projection_procedure("grafeo.projection.drop", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_projection_create_short_name() {
        let args = vec![LogicalExpression::Literal(Value::from("short_proj"))];
        let result = try_execute_projection_procedure("projection.create", &args).unwrap();
        assert!(result.is_some());
        projection_registry().drop_projection("short_proj");
    }
}
