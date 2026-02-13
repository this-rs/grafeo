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
    FloydWarshallAlgorithm, GraphAlgorithm, KCoreAlgorithm, KruskalAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, MaxFlowAlgorithm, MinCostFlowAlgorithm,
    PageRankAlgorithm, PrimAlgorithm, StronglyConnectedComponentsAlgorithm,
    TopologicalSortAlgorithm,
};
use grafeo_adapters::plugins::{AlgorithmResult, ParameterDef, Parameters};
use grafeo_common::types::Value;
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
        register(&mut algorithms, Arc::new(BellmanFordAlgorithm));
        register(&mut algorithms, Arc::new(FloydWarshallAlgorithm));

        // Clustering
        register(&mut algorithms, Arc::new(ClusteringCoefficientAlgorithm));

        // Community
        register(&mut algorithms, Arc::new(LabelPropagationAlgorithm));
        register(&mut algorithms, Arc::new(LouvainAlgorithm));

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

        Self { algorithms }
    }

    /// Resolves a procedure name to its algorithm.
    ///
    /// Strips `"grafeo."` prefix if present:
    /// - `["grafeo", "pagerank"]` → looks up `"pagerank"`
    /// - `["pagerank"]` → looks up `"pagerank"`
    pub fn get(&self, name: &[String]) -> Option<Arc<dyn GraphAlgorithm>> {
        let key = resolve_name(name);
        self.algorithms.get(key).cloned()
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
fn resolve_name(parts: &[String]) -> &str {
    match parts {
        [_, name] if parts[0].eq_ignore_ascii_case("grafeo") => name.as_str(),
        [name] => name.as_str(),
        _ => parts.last().map_or("", String::as_str),
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
        "dijkstra" => vec!["node_id".into(), "distance".into()],
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
        "louvain" => vec!["node_id".into(), "community_id".into(), "modularity".into()],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_all_algorithms() {
        let registry = BuiltinProcedures::new();
        let list = registry.list();
        assert!(
            list.len() >= 22,
            "Expected at least 22 algorithms, got {}",
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
        assert!(result.rows.len() >= 22);
    }
}
