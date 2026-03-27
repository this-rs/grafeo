//! RDF Query Planner.
//!
//! Converts logical plans with RDF operators (TripleScan, etc.) to physical
//! operators that execute against an RDF store.
//!
//! This planner follows the same push-based, vectorized execution model as
//! the LPG planner for consistent performance characteristics.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{
    BinaryFilterOp, FilterExpression, FilterOperator, HashAggregateOperator, Operator,
    OperatorError, Predicate, ProjectExpr, ProjectOperator, SimpleAggregateOperator,
    SingleRowOperator, SortOperator, UnaryFilterOp,
};
use grafeo_core::graph::rdf::{Literal, RdfStore, Term, Triple, TriplePattern};

use crate::query::plan::{
    AddGraphOp, AggregateFunction as LogicalAggregateFunction, AggregateOp, AntiJoinOp, BindOp,
    ClearGraphOp, CopyGraphOp, CreateGraphOp, DeleteTripleOp, DistinctOp, DropGraphOp, FilterOp,
    InsertTripleOp, LeftJoinOp, LimitOp, LogicalExpression, LogicalOperator, LogicalPlan, ModifyOp,
    MoveGraphOp, SkipOp, SortOp, TripleComponent, TripleScanOp, TripleTemplate,
};
use crate::query::planner::{PhysicalPlan, convert_aggregate_function, convert_filter_expression};

#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;

/// Default chunk size for morsel-driven execution.
const DEFAULT_CHUNK_SIZE: usize = 1024;

/// Logs an RDF WAL record if a WAL reference is present.
#[cfg(feature = "wal")]
fn log_rdf_wal(wal: &Option<Arc<RdfWal>>, record: &grafeo_adapters::storage::wal::WalRecord) {
    if let Some(wal) = wal
        && let Err(err) = wal.log(record)
    {
        tracing::warn!("RDF WAL log failed: {err}");
    }
}

/// Converts a Term to its N-Triples string for WAL serialization.
#[cfg(feature = "wal")]
fn term_to_wal(term: &Term) -> String {
    term.to_string()
}

/// Converts logical plans with RDF operators to physical operators.
///
/// This planner produces push-based operators that process data in chunks
/// (morsels) for cache efficiency and parallelism compatibility.
/// Type alias for the WAL used by the RDF planner.
#[cfg(feature = "wal")]
type RdfWal = grafeo_adapters::storage::wal::LpgWal;

/// Converts logical plans with RDF operators to physical operators.
///
/// This planner produces push-based operators that process data in chunks
/// (morsels) for cache efficiency and parallelism compatibility.
pub struct RdfPlanner {
    /// The RDF store to query.
    store: Arc<RdfStore>,
    /// Chunk size for vectorized execution.
    chunk_size: usize,
    /// Optional transaction ID for transactional operations.
    transaction_id: Option<TransactionId>,
    /// When true, each physical operator is wrapped in `ProfiledOperator`.
    profiling: std::cell::Cell<bool>,
    /// Profile entries collected during planning (post-order).
    profile_entries: std::cell::RefCell<Vec<crate::query::profile::ProfileEntry>>,
    /// Optional WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfPlanner {
    /// Creates a new RDF planner with the given store.
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            chunk_size: DEFAULT_CHUNK_SIZE,
            transaction_id: None,
            profiling: std::cell::Cell::new(false),
            profile_entries: std::cell::RefCell::new(Vec::new()),
            #[cfg(feature = "wal")]
            wal: None,
        }
    }

    /// Sets the chunk size for vectorized execution.
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Sets the transaction ID for transactional operations.
    #[must_use]
    pub fn with_transaction_id(mut self, transaction_id: Option<TransactionId>) -> Self {
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the WAL for logging RDF mutations.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn with_wal(mut self, wal: Option<Arc<RdfWal>>) -> Self {
        self.wal = wal;
        self
    }

    /// Plans a logical plan into a physical operator tree.
    ///
    /// # Errors
    ///
    /// Returns an error if planning fails.
    pub fn plan(&self, logical_plan: &LogicalPlan) -> Result<PhysicalPlan> {
        let (operator, columns) = self.plan_operator(&logical_plan.root)?;
        Ok(PhysicalPlan {
            operator,
            columns,
            adaptive_context: None,
        })
    }

    /// Plans a logical plan with profiling: each physical operator is wrapped
    /// in [`ProfiledOperator`](grafeo_core::execution::ProfiledOperator) to
    /// collect row counts and timing.
    pub fn plan_profiled(
        &self,
        logical_plan: &LogicalPlan,
    ) -> Result<(PhysicalPlan, Vec<crate::query::profile::ProfileEntry>)> {
        self.profiling.set(true);
        self.profile_entries.borrow_mut().clear();

        let result = self.plan_operator(&logical_plan.root);

        self.profiling.set(false);
        let (operator, columns) = result?;
        let entries = self.profile_entries.borrow_mut().drain(..).collect();

        Ok((
            PhysicalPlan {
                operator,
                columns,
                adaptive_context: None,
            },
            entries,
        ))
    }

    /// If profiling is enabled, wraps a planned result in `ProfiledOperator`
    /// and records a [`ProfileEntry`](crate::query::profile::ProfileEntry).
    fn maybe_profile(
        &self,
        result: Result<(Box<dyn Operator>, Vec<String>)>,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if self.profiling.get() {
            let (physical, columns) = result?;
            let (entry, stats) =
                crate::query::profile::ProfileEntry::new(physical.name(), op.display_label());
            let profiled = grafeo_core::execution::ProfiledOperator::new(physical, stats);
            self.profile_entries.borrow_mut().push(entry);
            Ok((Box::new(profiled), columns))
        } else {
            result
        }
    }

    /// Plans a single logical operator.
    fn plan_operator(&self, op: &LogicalOperator) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let result = match op {
            LogicalOperator::TripleScan(scan) => self.plan_triple_scan(scan),
            LogicalOperator::Filter(filter) => self.plan_filter(filter),
            LogicalOperator::Project(project) => self.plan_project(project),
            LogicalOperator::Limit(limit) => self.plan_limit(limit),
            LogicalOperator::Skip(skip) => self.plan_skip(skip),
            LogicalOperator::Sort(sort) => self.plan_sort(sort),
            LogicalOperator::Aggregate(agg) => self.plan_aggregate(agg),
            LogicalOperator::Return(ret) => self.plan_return(ret),
            LogicalOperator::Join(join) => self.plan_join(join),
            LogicalOperator::LeftJoin(join) => self.plan_left_join(join),
            LogicalOperator::AntiJoin(join) => self.plan_anti_join(join),
            LogicalOperator::Union(union) => self.plan_union(union),
            LogicalOperator::Distinct(distinct) => self.plan_distinct(distinct),
            LogicalOperator::InsertTriple(insert) => self.plan_insert_triple(insert),
            LogicalOperator::DeleteTriple(delete) => self.plan_delete_triple(delete),
            LogicalOperator::Modify(modify) => self.plan_modify(modify),
            LogicalOperator::ClearGraph(clear) => self.plan_clear_graph(clear),
            LogicalOperator::CreateGraph(create) => self.plan_create_graph(create),
            LogicalOperator::DropGraph(drop_op) => self.plan_drop_graph(drop_op),
            LogicalOperator::CopyGraph(copy) => self.plan_copy_graph(copy),
            LogicalOperator::MoveGraph(move_op) => self.plan_move_graph(move_op),
            LogicalOperator::AddGraph(add) => self.plan_add_graph(add),
            LogicalOperator::Bind(bind) => self.plan_bind(bind),
            LogicalOperator::MultiWayJoin(mwj) => self.plan_multi_way_join(mwj),
            LogicalOperator::Empty => {
                let op: Box<dyn Operator> = Box::new(SingleRowOperator::new());
                Ok((op, vec![]))
            }
            _ => Err(Error::Internal(format!(
                "Unsupported RDF operator: {:?}",
                std::mem::discriminant(op)
            ))),
        };
        self.maybe_profile(result, op)
    }

    /// Plans a triple scan operator.
    ///
    /// Creates a lazy scanning operator that reads triples in chunks
    /// for cache-efficient, vectorized processing.
    fn plan_triple_scan(&self, scan: &TripleScanOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Build the triple pattern for querying the store
        let pattern = self.build_triple_pattern(scan);

        // Determine which columns are variables (and thus in output)
        let mut columns = Vec::new();
        let mut output_mask = [false, false, false, false]; // s, p, o, g

        if let TripleComponent::Variable(name) = &scan.subject {
            columns.push(name.clone());
            output_mask[0] = true;
        }
        if let TripleComponent::Variable(name) = &scan.predicate {
            columns.push(name.clone());
            output_mask[1] = true;
        }
        if let TripleComponent::Variable(name) = &scan.object {
            columns.push(name.clone());
            output_mask[2] = true;
        }
        if let Some(TripleComponent::Variable(name)) = &scan.graph {
            columns.push(name.clone());
            output_mask[3] = true;
        }

        // Resolve graph context
        let (graph_iri, scan_all_graphs) = match &scan.graph {
            Some(TripleComponent::Iri(iri)) => (Some(iri.clone()), false),
            Some(TripleComponent::Literal(Value::String(iri))) => (Some(iri.to_string()), false),
            Some(TripleComponent::Variable(_)) => (None, true),
            _ => (None, false),
        };

        // Create the lazy scanning operator
        let operator = Box::new(RdfTripleScanOperator::new(
            Arc::clone(&self.store),
            pattern,
            output_mask,
            self.chunk_size,
            graph_iri,
            scan_all_graphs,
        ));

        Ok((operator, columns))
    }

    /// Builds a TriplePattern from a TripleScanOp.
    fn build_triple_pattern(&self, scan: &TripleScanOp) -> TriplePattern {
        TriplePattern {
            subject: component_to_term(&scan.subject),
            predicate: component_to_term(&scan.predicate),
            object: component_to_term(&scan.object),
        }
    }

    /// Plans a RETURN clause.
    fn plan_return(
        &self,
        ret: &crate::query::plan::ReturnOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, _input_columns) = self.plan_operator(&ret.input)?;

        // Extract output column names
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| {
                item.alias
                    .clone()
                    .unwrap_or_else(|| expression_to_string(&item.expression))
            })
            .collect();

        Ok((input_op, columns))
    }

    /// Plans a filter operator.
    ///
    /// Handles EXISTS/NOT EXISTS patterns by transforming them into semi-joins/anti-joins.
    fn plan_filter(&self, filter: &FilterOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check for EXISTS/NOT EXISTS patterns and transform to semi/anti joins
        if let Some((subquery, is_negated)) = self.extract_exists_pattern(&filter.predicate) {
            return self.plan_exists_as_join(&filter.input, subquery, is_negated);
        }

        let (input_op, columns) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = convert_filter_expression(&filter.predicate)?;

        // Create RDF-specific predicate (doesn't need LpgStore)
        let predicate = RdfExpressionPredicate::new(filter_expr, variable_columns);

        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));
        Ok((operator, columns))
    }

    /// Extracts an EXISTS or NOT EXISTS pattern from a filter predicate.
    /// Returns the subquery operator and whether it's negated (NOT EXISTS).
    fn extract_exists_pattern<'a>(
        &self,
        predicate: &'a LogicalExpression,
    ) -> Option<(&'a LogicalOperator, bool)> {
        use crate::query::plan::UnaryOp;

        match predicate {
            // EXISTS { pattern }
            LogicalExpression::ExistsSubquery(subquery) => Some((subquery.as_ref(), false)),
            // NOT EXISTS { pattern }
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subquery) = operand.as_ref() {
                    Some((subquery.as_ref(), true))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Plans an EXISTS/NOT EXISTS pattern as a semi-join or anti-join.
    fn plan_exists_as_join(
        &self,
        input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;

        let (left_op, left_columns) = self.plan_operator(input)?;
        let (right_op, right_columns) = self.plan_operator(subquery)?;

        let schema = derive_rdf_schema(&left_columns);

        // Use Anti for NOT EXISTS, Semi for EXISTS
        let result = if is_negated {
            common::build_anti_join(left_op, right_op, left_columns, &right_columns, schema)
        } else {
            common::build_semi_join(left_op, right_op, left_columns, &right_columns, schema)
        };

        Ok(result)
    }

    /// Plans a DISTINCT operator.
    fn plan_distinct(&self, distinct: &DistinctOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (input_op, columns) = self.plan_operator(&distinct.input)?;
        let schema = derive_rdf_schema(&columns);
        Ok(common::build_distinct(
            input_op,
            columns,
            distinct.columns.as_deref(),
            schema,
        ))
    }

    /// Plans a LIMIT operator.
    fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (input_op, columns) = self.plan_operator(&limit.input)?;
        let schema = derive_rdf_schema(&columns);
        Ok(common::build_limit(
            input_op,
            columns,
            limit.count.value(),
            schema,
        ))
    }

    /// Plans a SKIP operator.
    fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let schema = derive_rdf_schema(&columns);
        Ok(common::build_skip(
            input_op,
            columns,
            skip.count.value(),
            schema,
        ))
    }

    /// Plans a SORT operator.
    fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::plan::SortOrder;
        use grafeo_core::execution::operators::{NullOrder, SortDirection, SortKey};

        let (input_op, columns) = self.plan_operator(&sort.input)?;

        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let physical_keys: Vec<SortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = resolve_expression(&key.expression, &variable_columns)?;
                Ok(SortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    null_order: NullOrder::NullsLast,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = derive_rdf_schema(&columns);
        let operator = Box::new(SortOperator::new(input_op, physical_keys, output_schema));
        Ok((operator, columns))
    }

    /// Plans a PROJECT operator.
    ///
    /// Projects only the requested columns from the input.
    fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::{ProjectExpr, ProjectOperator};

        let (input_op, input_columns) = self.plan_operator(&project.input)?;

        // Build mapping from variable name to column index
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let mut projections = Vec::new();
        let mut output_columns = Vec::new();
        let mut output_types = Vec::new();

        for proj in &project.projections {
            match &proj.expression {
                LogicalExpression::Variable(name) => {
                    if let Some(&col_idx) = variable_columns.get(name) {
                        projections.push(ProjectExpr::Column(col_idx));
                        output_columns.push(proj.alias.clone().unwrap_or_else(|| name.clone()));
                        output_types.push(LogicalType::Any); // preserve actual value types
                    } else {
                        return Err(Error::Internal(format!(
                            "Variable '{}' not found in input columns",
                            name
                        )));
                    }
                }
                LogicalExpression::Literal(value) => {
                    projections.push(ProjectExpr::Constant(value.clone()));
                    output_columns.push(proj.alias.clone().unwrap_or_else(|| format!("{value}")));
                    output_types.push(LogicalType::Any);
                }
                _ => {
                    // For non-variable expressions, we need to evaluate them
                    // For now, skip complex expressions in projection
                    continue;
                }
            }
        }

        // If no projections were extracted, just return the input as-is
        if projections.is_empty() {
            return Ok((input_op, input_columns));
        }

        let operator = Box::new(ProjectOperator::new(input_op, projections, output_types));
        Ok((operator, output_columns))
    }

    /// Plans a BIND operator.
    ///
    /// BIND adds a computed column to each row by evaluating an expression.
    /// For example: `BIND (CONCAT(?name, " (age ", STR(?age), ")") AS ?label)`
    fn plan_bind(&self, bind: &BindOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, input_columns) = self.plan_operator(&bind.input)?;

        // Build variable-to-column mapping for expression evaluation
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert the BIND expression to a FilterExpression
        let filter_expr = convert_filter_expression(&bind.expression)?;

        // Build output columns: all input columns + the new BIND variable
        let mut output_columns = input_columns;
        output_columns.push(bind.variable.clone());

        let operator = Box::new(RdfBindOperator::new(
            input_op,
            filter_expr,
            variable_columns,
        ));
        Ok((operator, output_columns))
    }

    /// Plans an AGGREGATE operator.
    fn plan_aggregate(&self, agg: &AggregateOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use grafeo_core::execution::operators::AggregateExpr as PhysicalAggregateExpr;

        let (mut input_op, input_columns) = self.plan_operator(&agg.input)?;

        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Pre-project complex expressions (CASE, Binary, etc.) inside aggregate arguments
        let mut expression_projections: Vec<(FilterExpression, String)> = Vec::new();
        let mut next_col_idx = input_columns.len();
        for agg_expr in &agg.aggregates {
            for expr_opt in [&agg_expr.expression, &agg_expr.expression2] {
                let Some(expr) = expr_opt else { continue };
                match expr {
                    LogicalExpression::Variable(_) => {}
                    _ => {
                        let col_name = format!("__expr_{:?}", expr);
                        if !variable_columns.contains_key(&col_name) {
                            let filter_expr = convert_filter_expression(expr)?;
                            expression_projections.push((filter_expr, col_name.clone()));
                            variable_columns.insert(col_name, next_col_idx);
                            next_col_idx += 1;
                        }
                    }
                }
            }
        }

        if !expression_projections.is_empty() {
            let mut projections: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            let mut output_types: Vec<LogicalType> =
                input_columns.iter().map(|_| LogicalType::String).collect();

            for (filter_expr, _col_name) in &expression_projections {
                projections.push(ProjectExpr::Expression {
                    expr: filter_expr.clone(),
                    variable_columns: variable_columns.clone(),
                });
                output_types.push(LogicalType::Any);
            }

            input_op = Box::new(ProjectOperator::new(input_op, projections, output_types));
        }

        let group_columns: Vec<usize> = agg
            .group_by
            .iter()
            .map(|expr| resolve_expression(expr, &variable_columns))
            .collect::<Result<Vec<_>>>()?;

        let physical_aggregates: Vec<PhysicalAggregateExpr> = agg
            .aggregates
            .iter()
            .map(|agg_expr| {
                let column = agg_expr
                    .expression
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                let column2 = agg_expr
                    .expression2
                    .as_ref()
                    .map(|e| resolve_expression(e, &variable_columns))
                    .transpose()?;

                Ok(PhysicalAggregateExpr {
                    function: convert_aggregate_function(agg_expr.function),
                    column,
                    column2,
                    distinct: agg_expr.distinct,
                    alias: agg_expr.alias.clone(),
                    percentile: agg_expr.percentile,
                    separator: agg_expr.separator.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut output_schema = Vec::new();
        let mut output_columns = Vec::new();

        for expr in &agg.group_by {
            output_schema.push(LogicalType::String);
            output_columns.push(expression_to_string(expr));
        }

        for agg_expr in &agg.aggregates {
            // For RDF, numeric values are strings that get converted to floats
            // So SUM should also output Float64 (since SumFloat returns Float64)
            let result_type = match agg_expr.function {
                LogicalAggregateFunction::Count | LogicalAggregateFunction::CountNonNull => {
                    LogicalType::Int64
                }
                LogicalAggregateFunction::Sum => LogicalType::Float64,
                LogicalAggregateFunction::Avg => LogicalType::Float64,
                _ => LogicalType::String,
            };
            output_schema.push(result_type);
            output_columns.push(
                agg_expr
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}(...)", agg_expr.function).to_lowercase()),
            );
        }

        let mut operator: Box<dyn Operator> = if group_columns.is_empty() {
            Box::new(SimpleAggregateOperator::new(
                input_op,
                physical_aggregates,
                output_schema,
            ))
        } else {
            Box::new(HashAggregateOperator::new(
                input_op,
                group_columns,
                physical_aggregates,
                output_schema,
            ))
        };

        // Apply HAVING clause filter if present
        if let Some(having_expr) = &agg.having {
            let having_var_columns: HashMap<String, usize> = output_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();

            let filter_expr = convert_filter_expression(having_expr)?;
            let predicate = RdfExpressionPredicate::new(filter_expr, having_var_columns);
            operator = Box::new(FilterOperator::new(operator, Box::new(predicate)));
        }

        Ok((operator, output_columns))
    }

    /// Plans a JOIN operator using HashJoin.
    ///
    /// For SPARQL, we join on shared variables (equi-join). When no shared
    /// variables exist, falls back to cross join.
    fn plan_join(
        &self,
        join: &crate::query::plan::JoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        // Estimate cardinalities for build-side selection
        let cardinalities = estimate_operator_cardinality(&join.left, &self.store)
            .zip(estimate_operator_cardinality(&join.right, &self.store));

        Ok(common::build_inner_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            derive_rdf_schema,
            cardinalities,
        ))
    }

    /// Plans a LEFT JOIN operator (for SPARQL OPTIONAL) using HashJoin.
    fn plan_left_join(&self, join: &LeftJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;
        Ok(common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            derive_rdf_schema,
        ))
    }

    /// Plans an ANTI JOIN operator (for SPARQL MINUS).
    fn plan_anti_join(&self, join: &AntiJoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;
        let schema = derive_rdf_schema(&left_columns);
        Ok(common::build_anti_join(
            left_op,
            right_op,
            left_columns,
            &right_columns,
            schema,
        ))
    }

    /// Plans a multi-way join as cascading pairwise hash joins.
    ///
    /// Sorts inputs by estimated cardinality (smallest first) to minimize
    /// intermediate result sizes, then folds them left-to-right.
    fn plan_multi_way_join(
        &self,
        mwj: &crate::query::plan::MultiWayJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::planner::common;

        if mwj.inputs.is_empty() {
            return Err(Error::Internal(
                "MultiWayJoin requires at least one input".to_string(),
            ));
        }

        // Plan all inputs and estimate cardinalities
        let mut planned: Vec<(Box<dyn Operator>, Vec<String>, f64)> = Vec::new();
        for input in &mwj.inputs {
            let (op, cols) = self.plan_operator(input)?;
            let card = estimate_operator_cardinality(input, &self.store).unwrap_or(1000.0);
            planned.push((op, cols, card));
        }

        // Sort by cardinality (smallest first) so we build on smaller inputs
        planned.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Fold left-to-right with pairwise hash joins
        let (mut current_op, mut current_cols, mut current_card) = planned.remove(0);
        for (right_op, right_cols, right_card) in planned {
            let cardinalities = Some((current_card, right_card));
            let (joined_op, joined_cols) = common::build_inner_join(
                current_op,
                right_op,
                &current_cols,
                &right_cols,
                derive_rdf_schema,
                cardinalities,
            );
            // Rough estimate for cascaded join output
            current_card = (current_card * right_card * 0.1).max(1.0);
            current_op = joined_op;
            current_cols = joined_cols;
        }

        Ok((current_op, current_cols))
    }

    /// Plans a UNION operator.
    fn plan_union(
        &self,
        union: &crate::query::plan::UnionOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal("Empty UNION".to_string()));
        }

        // For INSERT operations, we execute all operators in sequence
        let mut operators: Vec<Box<dyn Operator>> = Vec::new();
        let mut columns = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols) = self.plan_operator(input)?;
            operators.push(op);
            if i == 0 {
                columns = cols;
            }
        }

        if operators.len() == 1 {
            return Ok((
                operators
                    .into_iter()
                    .next()
                    .expect("single-element iterator"),
                columns,
            ));
        }

        // Create a chain operator that executes all operators in sequence
        let operator = Box::new(RdfUnionOperator::new(operators));
        Ok((operator, columns))
    }

    /// Plans an INSERT TRIPLE operator.
    fn plan_insert_triple(
        &self,
        insert: &InsertTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if this is a pattern-based insert (has variables in the template)
        let has_variables = matches!(&insert.subject, TripleComponent::Variable(_))
            || matches!(&insert.predicate, TripleComponent::Variable(_))
            || matches!(&insert.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based insertion: need to query first, then insert each match
            if let Some(ref input) = insert.input {
                let (input_op, input_columns) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfInsertPatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    insert.subject.clone(),
                    insert.predicate.clone(),
                    insert.object.clone(),
                    column_map,
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                ));

                return Ok((operator, Vec::new()));
            }
        }

        // Direct insertion with concrete terms
        let subject = self.component_to_term(&insert.subject)?;
        let predicate = self.component_to_term(&insert.predicate)?;
        let object = self.component_to_term(&insert.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfInsertTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            insert.graph.clone(),
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));

        // Insert operations don't output columns
        Ok((operator, Vec::new()))
    }

    /// Converts a TripleComponent to an RDF Term.
    fn component_to_term(&self, component: &TripleComponent) -> Result<Term> {
        match component {
            TripleComponent::Iri(iri) => Ok(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Ok(Term::Literal(lit))
            }
            TripleComponent::Variable(name) => {
                // Variables in INSERT DATA should have been bound
                Err(Error::Internal(format!(
                    "Unbound variable '{}' in INSERT DATA",
                    name
                )))
            }
        }
    }

    /// Plans a DELETE TRIPLE operator.
    fn plan_delete_triple(
        &self,
        delete: &DeleteTripleOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if this is a pattern-based delete (has variables in the template)
        let has_variables = matches!(&delete.subject, TripleComponent::Variable(_))
            || matches!(&delete.predicate, TripleComponent::Variable(_))
            || matches!(&delete.object, TripleComponent::Variable(_));

        if has_variables {
            // Pattern-based deletion: need to query first, then delete each match
            if let Some(ref input) = delete.input {
                let (input_op, input_columns) = self.plan_operator(input)?;

                // Build column index map for variable substitution
                let column_map: HashMap<String, usize> = input_columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();

                let operator = Box::new(RdfDeletePatternOperator::new(
                    Arc::clone(&self.store),
                    input_op,
                    delete.subject.clone(),
                    delete.predicate.clone(),
                    delete.object.clone(),
                    column_map,
                    #[cfg(feature = "wal")]
                    self.wal.clone(),
                ));

                return Ok((operator, Vec::new()));
            }
        }

        // Direct deletion with concrete terms
        let subject = self.component_to_term(&delete.subject)?;
        let predicate = self.component_to_term(&delete.predicate)?;
        let object = self.component_to_term(&delete.object)?;

        let triple = Triple::new(subject, predicate, object);
        let operator = Box::new(RdfDeleteTripleOperator::new(
            Arc::clone(&self.store),
            triple,
            delete.graph.clone(),
            self.transaction_id,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));

        Ok((operator, Vec::new()))
    }

    /// Plans a CLEAR GRAPH operator.
    fn plan_clear_graph(&self, clear: &ClearGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfClearGraphOperator::new(
            Arc::clone(&self.store),
            clear.graph.clone(),
            clear.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a CREATE GRAPH operator.
    fn plan_create_graph(
        &self,
        create: &CreateGraphOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfCreateGraphOperator::new(
            Arc::clone(&self.store),
            create.graph.clone(),
            create.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a DROP GRAPH operator.
    fn plan_drop_graph(&self, drop_op: &DropGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfDropGraphOperator::new(
            Arc::clone(&self.store),
            drop_op.graph.clone(),
            drop_op.silent,
            #[cfg(feature = "wal")]
            self.wal.clone(),
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a COPY graph operator.
    fn plan_copy_graph(&self, copy: &CopyGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfCopyGraphOperator::new(
            Arc::clone(&self.store),
            copy.source.clone(),
            copy.destination.clone(),
            copy.silent,
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a MOVE graph operator.
    fn plan_move_graph(&self, move_op: &MoveGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfMoveGraphOperator::new(
            Arc::clone(&self.store),
            move_op.source.clone(),
            move_op.destination.clone(),
            move_op.silent,
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans an ADD graph operator.
    fn plan_add_graph(&self, add: &AddGraphOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let operator = Box::new(RdfAddGraphOperator::new(
            Arc::clone(&self.store),
            add.source.clone(),
            add.destination.clone(),
            add.silent,
        ));
        Ok((operator, Vec::new()))
    }

    /// Plans a SPARQL MODIFY operator (DELETE/INSERT WHERE).
    ///
    /// Per SPARQL 1.1 spec:
    /// 1. Evaluate WHERE clause once to get bindings
    /// 2. Apply DELETE templates using those bindings
    /// 3. Apply INSERT templates using the SAME bindings
    fn plan_modify(&self, modify: &ModifyOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the WHERE clause
        let (where_op, where_columns) = self.plan_operator(&modify.where_clause)?;

        // Build column index map for variable substitution
        let column_map: HashMap<String, usize> = where_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let operator = Box::new(RdfModifyOperator::new(
            Arc::clone(&self.store),
            where_op,
            modify.delete_templates.clone(),
            modify.insert_templates.clone(),
            column_map,
        ));

        Ok((operator, Vec::new()))
    }
}

// ============================================================================
// RDF Insert Triple Operator
// ============================================================================

/// Operator that inserts a triple into the RDF store.
struct RdfInsertTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    inserted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfInsertTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            inserted: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfInsertTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.inserted {
            return Ok(None);
        }

        // Resolve target store: named graph or default
        let target = match &self.graph_name {
            Some(name) => self.store.graph_or_create(name),
            None => Arc::clone(&self.store),
        };

        // Insert the triple (buffered if in a transaction)
        if let Some(transaction_id) = self.transaction_id {
            target.insert_in_transaction(transaction_id, self.triple.clone());
        } else {
            target.insert(self.triple.clone());
        }

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_adapters::storage::wal::WalRecord::InsertRdfTriple {
                subject: term_to_wal(self.triple.subject()),
                predicate: term_to_wal(self.triple.predicate()),
                object: term_to_wal(self.triple.object()),
                graph: self.graph_name.clone(),
            },
        );

        self.inserted = true;

        // Return an empty result (INSERT doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.inserted = false;
    }

    fn name(&self) -> &'static str {
        "RdfInsertTriple"
    }
}

// ============================================================================
// RDF Insert Pattern Operator
// ============================================================================

/// Operator that inserts triples based on a pattern from the RDF store.
/// Used for INSERT { } WHERE { } operations where the triple template contains variables.
struct RdfInsertPatternOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfInsertPatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        column_map: HashMap<String, usize>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            input,
            subject,
            predicate,
            object,
            column_map,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                // Check if it looks like an IRI
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else if let Ok(n) = s.parse::<i64>() {
                    // Try to parse as integer
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    // Try to parse as float
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfInsertPatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to insert
        let mut triples_to_insert = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject = self.resolve_component(&self.subject, &chunk, row);
                let predicate = self.resolve_component(&self.predicate, &chunk, row);
                let object = self.resolve_component(&self.object, &chunk, row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    triples_to_insert.push(Triple::new(s, p, o));
                }
            }
        }

        // Insert all collected triples
        for triple in &triples_to_insert {
            self.store.insert(triple.clone());
        }

        #[cfg(feature = "wal")]
        for triple in &triples_to_insert {
            log_rdf_wal(
                &self.wal,
                &grafeo_adapters::storage::wal::WalRecord::InsertRdfTriple {
                    subject: term_to_wal(triple.subject()),
                    predicate: term_to_wal(triple.predicate()),
                    object: term_to_wal(triple.object()),
                    graph: None,
                },
            );
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfInsertPattern"
    }
}

// ============================================================================
// RDF Delete Triple Operator
// ============================================================================

/// Operator that deletes a triple from the RDF store.
struct RdfDeleteTripleOperator {
    store: Arc<RdfStore>,
    triple: Triple,
    graph_name: Option<String>,
    transaction_id: Option<TransactionId>,
    deleted: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfDeleteTripleOperator {
    fn new(
        store: Arc<RdfStore>,
        triple: Triple,
        graph_name: Option<String>,
        transaction_id: Option<TransactionId>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            triple,
            graph_name,
            transaction_id,
            deleted: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfDeleteTripleOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.deleted {
            return Ok(None);
        }

        // Resolve target store: named graph or default
        let target = match &self.graph_name {
            Some(name) => self.store.graph_or_create(name),
            None => Arc::clone(&self.store),
        };

        // Delete the triple (buffered if in a transaction)
        if let Some(transaction_id) = self.transaction_id {
            target.remove_in_transaction(transaction_id, self.triple.clone());
        } else {
            target.remove(&self.triple);
        }

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_adapters::storage::wal::WalRecord::DeleteRdfTriple {
                subject: term_to_wal(self.triple.subject()),
                predicate: term_to_wal(self.triple.predicate()),
                object: term_to_wal(self.triple.object()),
                graph: self.graph_name.clone(),
            },
        );

        self.deleted = true;

        // Return an empty result (DELETE doesn't produce rows)
        Ok(None)
    }

    fn reset(&mut self) {
        self.deleted = false;
    }

    fn name(&self) -> &'static str {
        "RdfDeleteTriple"
    }
}

// ============================================================================
// RDF Delete Pattern Operator
// ============================================================================

/// Operator that deletes triples matching a pattern from the RDF store.
/// Used for DELETE WHERE operations where the triple template contains variables.
struct RdfDeletePatternOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    subject: TripleComponent,
    predicate: TripleComponent,
    object: TripleComponent,
    column_map: HashMap<String, usize>,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfDeletePatternOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        column_map: HashMap<String, usize>,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            input,
            subject,
            predicate,
            object,
            column_map,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::Variable(name) => {
                // Remove the leading '?' if present
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                // Check if it looks like an IRI
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else if let Ok(n) = s.parse::<i64>() {
                    // Try to parse as integer
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    // Try to parse as float
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfDeletePatternOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Collect all triples to delete
        let mut triples_to_delete = Vec::new();

        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                let subject = self.resolve_component(&self.subject, &chunk, row);
                let predicate = self.resolve_component(&self.predicate, &chunk, row);
                let object = self.resolve_component(&self.object, &chunk, row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    triples_to_delete.push(Triple::new(s, p, o));
                }
            }
        }

        // Delete all collected triples
        for triple in &triples_to_delete {
            self.store.remove(triple);
        }

        #[cfg(feature = "wal")]
        for triple in &triples_to_delete {
            log_rdf_wal(
                &self.wal,
                &grafeo_adapters::storage::wal::WalRecord::DeleteRdfTriple {
                    subject: term_to_wal(triple.subject()),
                    predicate: term_to_wal(triple.predicate()),
                    object: term_to_wal(triple.object()),
                    graph: None,
                },
            );
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfDeletePattern"
    }
}

// ============================================================================
// RDF Clear Graph Operator
// ============================================================================

/// Operator that clears triples from a graph in the RDF store.
struct RdfClearGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    cleared: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfClearGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        _silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            cleared: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfClearGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.cleared {
            return Ok(None);
        }

        self.store.clear_graph(self.graph.as_deref());

        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_adapters::storage::wal::WalRecord::ClearRdfGraph {
                graph: self.graph.clone(),
            },
        );

        self.cleared = true;

        Ok(None)
    }

    fn reset(&mut self) {
        self.cleared = false;
    }

    fn name(&self) -> &'static str {
        "RdfClearGraph"
    }
}

// ============================================================================
// RDF CREATE/DROP Graph Operators
// ============================================================================

/// Operator that creates a named graph.
struct RdfCreateGraphOperator {
    store: Arc<RdfStore>,
    graph: String,
    silent: bool,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfCreateGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: String,
        silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfCreateGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        let created = self.store.create_graph(&self.graph);
        if !created && !self.silent {
            return Err(OperatorError::Execution(format!(
                "Graph <{}> already exists",
                self.graph
            )));
        }
        #[cfg(feature = "wal")]
        if created {
            log_rdf_wal(
                &self.wal,
                &grafeo_adapters::storage::wal::WalRecord::CreateRdfGraph {
                    name: self.graph.clone(),
                },
            );
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCreateGraph"
    }
}

/// Operator that drops a named graph.
struct RdfDropGraphOperator {
    store: Arc<RdfStore>,
    graph: Option<String>,
    silent: bool,
    done: bool,
    #[cfg(feature = "wal")]
    wal: Option<Arc<RdfWal>>,
}

impl RdfDropGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        graph: Option<String>,
        silent: bool,
        #[cfg(feature = "wal")] wal: Option<Arc<RdfWal>>,
    ) -> Self {
        Self {
            store,
            graph,
            silent,
            done: false,
            #[cfg(feature = "wal")]
            wal,
        }
    }
}

impl Operator for RdfDropGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;
        match &self.graph {
            None => {
                // DROP DEFAULT: clear the default graph
                self.store.clear();
            }
            Some(name) => {
                let dropped = self.store.drop_graph(name);
                if !dropped && !self.silent {
                    return Err(OperatorError::Execution(format!(
                        "Graph <{name}> does not exist"
                    )));
                }
            }
        }
        #[cfg(feature = "wal")]
        log_rdf_wal(
            &self.wal,
            &grafeo_adapters::storage::wal::WalRecord::DropRdfGraph {
                name: self.graph.clone(),
            },
        );
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfDropGraph"
    }
}

// ============================================================================
// RDF COPY/MOVE/ADD Graph Operators
// ============================================================================

/// Operator that copies all triples from one graph to another.
struct RdfCopyGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfCopyGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfCopyGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .copy_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfCopyGraph"
    }
}

/// Operator that moves all triples from one graph to another.
struct RdfMoveGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfMoveGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfMoveGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .move_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfMoveGraph"
    }
}

/// Operator that adds (merges) all triples from one graph into another.
struct RdfAddGraphOperator {
    store: Arc<RdfStore>,
    source: Option<String>,
    destination: Option<String>,
    silent: bool,
    done: bool,
}

impl RdfAddGraphOperator {
    fn new(
        store: Arc<RdfStore>,
        source: Option<String>,
        destination: Option<String>,
        silent: bool,
    ) -> Self {
        Self {
            store,
            source,
            destination,
            silent,
            done: false,
        }
    }
}

impl Operator for RdfAddGraphOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }
        self.done = true;

        // Check source exists (unless silent)
        if !self.silent
            && let Some(ref name) = self.source
            && self.store.graph(name).is_none()
        {
            return Err(OperatorError::Execution(format!(
                "Source graph <{name}> does not exist"
            )));
        }

        self.store
            .add_graph(self.source.as_deref(), self.destination.as_deref());
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "RdfAddGraph"
    }
}

// ============================================================================
// RDF Modify Operator (SPARQL DELETE/INSERT WHERE)
// ============================================================================

/// Operator that handles SPARQL MODIFY operations (DELETE/INSERT WHERE).
///
/// Per SPARQL 1.1 Update spec:
/// 1. Evaluate WHERE clause once to get all bindings
/// 2. Apply DELETE templates to each binding
/// 3. Apply INSERT templates to each binding (using SAME bindings)
struct RdfModifyOperator {
    store: Arc<RdfStore>,
    input: Box<dyn Operator>,
    delete_templates: Vec<TripleTemplate>,
    insert_templates: Vec<TripleTemplate>,
    column_map: HashMap<String, usize>,
    done: bool,
}

impl RdfModifyOperator {
    fn new(
        store: Arc<RdfStore>,
        input: Box<dyn Operator>,
        delete_templates: Vec<TripleTemplate>,
        insert_templates: Vec<TripleTemplate>,
        column_map: HashMap<String, usize>,
    ) -> Self {
        Self {
            store,
            input,
            delete_templates,
            insert_templates,
            column_map,
            done: false,
        }
    }

    fn resolve_component(
        &self,
        component: &TripleComponent,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Term> {
        match component {
            TripleComponent::Iri(iri) => Some(Term::Iri(iri.clone().into())),
            TripleComponent::Literal(value) => {
                let lit = match value {
                    Value::String(s) => Literal::simple(s.to_string()),
                    Value::Int64(n) => Literal::integer(*n),
                    Value::Float64(f) => {
                        Literal::typed(f.to_string(), "http://www.w3.org/2001/XMLSchema#double")
                    }
                    Value::Bool(b) => {
                        Literal::typed(b.to_string(), "http://www.w3.org/2001/XMLSchema#boolean")
                    }
                    _ => Literal::simple(format!("{:?}", value)),
                };
                Some(Term::Literal(lit))
            }
            TripleComponent::Variable(name) => {
                let var_name = name.strip_prefix('?').unwrap_or(name);
                if let Some(&col_idx) = self.column_map.get(var_name)
                    && let Some(col) = chunk.column(col_idx)
                    && let Some(value) = col.get_value(row)
                {
                    return Self::value_to_term(&value);
                }
                None
            }
        }
    }

    fn value_to_term(value: &Value) -> Option<Term> {
        match value {
            Value::String(s) => {
                if s.starts_with("http://") || s.starts_with("https://") || s.starts_with("urn:") {
                    Some(Term::Iri(s.to_string().into()))
                } else if let Ok(n) = s.parse::<i64>() {
                    Some(Term::Literal(Literal::integer(n)))
                } else if let Ok(f) = s.parse::<f64>() {
                    Some(Term::Literal(Literal::typed(
                        f.to_string(),
                        "http://www.w3.org/2001/XMLSchema#double",
                    )))
                } else {
                    Some(Term::Literal(Literal::simple(s.to_string())))
                }
            }
            Value::Int64(n) => Some(Term::Literal(Literal::integer(*n))),
            Value::Float64(f) => Some(Term::Literal(Literal::typed(
                f.to_string(),
                "http://www.w3.org/2001/XMLSchema#double",
            ))),
            Value::Bool(b) => Some(Term::Literal(Literal::typed(
                b.to_string(),
                "http://www.w3.org/2001/XMLSchema#boolean",
            ))),
            _ => None,
        }
    }
}

impl Operator for RdfModifyOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        if self.done {
            return Ok(None);
        }

        // Step 1: Collect all bindings from WHERE clause (before any modifications)
        let mut bindings: Vec<(DataChunk, usize)> = Vec::new();
        while let Some(chunk) = self.input.next()? {
            for row in 0..chunk.row_count() {
                bindings.push((chunk.clone(), row));
            }
        }

        // Step 2: Apply DELETE templates using collected bindings
        for template in &self.delete_templates {
            for (chunk, row) in &bindings {
                let subject = self.resolve_component(&template.subject, chunk, *row);
                let predicate = self.resolve_component(&template.predicate, chunk, *row);
                let object = self.resolve_component(&template.object, chunk, *row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let triple = Triple::new(s, p, o);
                    self.store.remove(&triple);
                }
            }
        }

        // Step 3: Apply INSERT templates using the SAME bindings
        for template in &self.insert_templates {
            for (chunk, row) in &bindings {
                let subject = self.resolve_component(&template.subject, chunk, *row);
                let predicate = self.resolve_component(&template.predicate, chunk, *row);
                let object = self.resolve_component(&template.object, chunk, *row);

                if let (Some(s), Some(p), Some(o)) = (subject, predicate, object) {
                    let triple = Triple::new(s, p, o);
                    self.store.insert(triple);
                }
            }
        }

        self.done = true;
        Ok(None)
    }

    fn reset(&mut self) {
        self.done = false;
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RdfModify"
    }
}

// ============================================================================
// RDF Union Operator
// ============================================================================

/// Operator that executes multiple operators in sequence.
/// Used for UNION of INSERT operations.
struct RdfUnionOperator {
    operators: Vec<Box<dyn Operator>>,
    current_idx: usize,
}

impl RdfUnionOperator {
    fn new(operators: Vec<Box<dyn Operator>>) -> Self {
        Self {
            operators,
            current_idx: 0,
        }
    }
}

impl Operator for RdfUnionOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        // Execute all operators
        while self.current_idx < self.operators.len() {
            let op = &mut self.operators[self.current_idx];
            match op.next()? {
                Some(chunk) => return Ok(Some(chunk)),
                None => self.current_idx += 1,
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.current_idx = 0;
        for op in &mut self.operators {
            op.reset();
        }
    }

    fn name(&self) -> &'static str {
        "RdfUnion"
    }
}

// ============================================================================
// RDF Bind Operator
// ============================================================================

/// Operator that appends a computed column to each row.
///
/// Used for SPARQL BIND expressions (e.g., `BIND (CONCAT(?x, ?y) AS ?z)`).
/// Evaluates an expression using `RdfExpressionPredicate` and appends the
/// result as a new column in the output chunk.
struct RdfBindOperator {
    /// Child operator providing input rows.
    child: Box<dyn Operator>,
    /// Expression to evaluate per row.
    expression: FilterExpression,
    /// Variable name to column index mapping for expression evaluation.
    variable_columns: HashMap<String, usize>,
}

impl RdfBindOperator {
    fn new(
        child: Box<dyn Operator>,
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
    ) -> Self {
        Self {
            child,
            expression,
            variable_columns,
        }
    }
}

impl Operator for RdfBindOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        let input_col_count = input.column_count();
        let row_count = input.row_count();

        // Build output schema: preserve input column types + one extra column for BIND result
        let mut output_types: Vec<LogicalType> = Vec::with_capacity(input_col_count + 1);
        for col_idx in 0..input_col_count {
            if let Some(col) = input.column(col_idx) {
                output_types.push(col.logical_type());
            } else {
                output_types.push(LogicalType::String);
            }
        }
        output_types.push(LogicalType::Any);

        let mut output = DataChunk::with_capacity(&output_types, row_count);

        // Copy existing columns
        for col_idx in 0..input_col_count {
            let output_col = output
                .column_mut(col_idx)
                .expect("column exists: index within schema bounds");
            if let Some(input_col) = input.column(col_idx) {
                for row in input.selected_indices() {
                    if let Some(value) = input_col.get_value(row) {
                        output_col.push_value(value);
                    } else {
                        output_col.push_value(Value::Null);
                    }
                }
            }
        }

        // Evaluate expression for each row and append as new column
        let evaluator =
            RdfExpressionPredicate::new(self.expression.clone(), self.variable_columns.clone());
        let bind_col = output
            .column_mut(input_col_count)
            .expect("column exists: bind column is last in schema");
        for row in input.selected_indices() {
            let value = evaluator.eval(&input, row).unwrap_or(Value::Null);
            bind_col.push_value(value);
        }

        output.set_count(row_count);
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "RdfBind"
    }
}

// ============================================================================
// RDF Triple Scan Operator
// ============================================================================

/// Lazy triple scan operator that processes triples in chunks.
///
/// This operator queries the RDF store and emits results in DataChunks
/// for efficient vectorized processing.
struct RdfTripleScanOperator {
    /// The RDF store to scan.
    store: Arc<RdfStore>,
    /// The pattern to match.
    pattern: TriplePattern,
    /// Which components to include in output [s, p, o, g].
    output_mask: [bool; 4],
    /// Named graph to query. `None` = default graph, `Some(iri)` = named graph.
    /// When the graph component is a variable, this is `None` and we scan all graphs.
    graph: Option<String>,
    /// Whether to scan ALL graphs (when GRAPH ?var is used).
    scan_all_graphs: bool,
    /// Chunk size for batching.
    chunk_size: usize,
    /// Cached matching triples with graph names (lazily populated).
    triples: Option<Vec<(Option<String>, Arc<Triple>)>>,
    /// Current position in the triples.
    position: usize,
}

impl RdfTripleScanOperator {
    fn new(
        store: Arc<RdfStore>,
        pattern: TriplePattern,
        output_mask: [bool; 4],
        chunk_size: usize,
        graph: Option<String>,
        scan_all_graphs: bool,
    ) -> Self {
        Self {
            store,
            pattern,
            output_mask,
            graph,
            scan_all_graphs,
            chunk_size,
            triples: None,
            position: 0,
        }
    }

    /// Lazily load matching triples on first access.
    fn ensure_triples(&mut self) {
        if self.triples.is_none() {
            self.triples = Some(if self.scan_all_graphs {
                // GRAPH ?var - scan all graphs (default + named)
                self.store.find_in_graphs(&self.pattern, Some(&[]))
            } else if let Some(ref graph_iri) = self.graph {
                // GRAPH <iri> - scan specific named graph
                self.store
                    .graph(graph_iri)
                    .map(|g| {
                        g.find(&self.pattern)
                            .into_iter()
                            .map(|t| (Some(graph_iri.clone()), t))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                // Default graph
                self.store
                    .find(&self.pattern)
                    .into_iter()
                    .map(|t| (None, t))
                    .collect()
            });
        }
    }

    /// Count how many output columns we have.
    fn output_column_count(&self) -> usize {
        self.output_mask.iter().filter(|&&b| b).count()
    }
}

impl Operator for RdfTripleScanOperator {
    fn next(&mut self) -> std::result::Result<Option<DataChunk>, OperatorError> {
        self.ensure_triples();

        let triples = self
            .triples
            .as_ref()
            .expect("triples populated by ensure_triples");

        if self.position >= triples.len() {
            return Ok(None);
        }

        let end = (self.position + self.chunk_size).min(triples.len());
        let batch_size = end - self.position;
        let col_count = self.output_column_count();

        // Create output schema (all String for RDF)
        let schema: Vec<LogicalType> = (0..col_count).map(|_| LogicalType::String).collect();
        let mut chunk = DataChunk::with_capacity(&schema, batch_size);

        // Fill the chunk
        for i in self.position..end {
            let (ref graph_name, ref triple) = triples[i];
            let mut col_idx = 0;

            if self.output_mask[0] {
                // Subject
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(triple.subject()));
                }
                col_idx += 1;
            }
            if self.output_mask[1] {
                // Predicate
                if let Some(col) = chunk.column_mut(col_idx) {
                    col.push_string(term_to_string(triple.predicate()));
                }
                col_idx += 1;
            }
            if self.output_mask[2] {
                // Object
                if let Some(col) = chunk.column_mut(col_idx) {
                    push_term_value(col, triple.object());
                }
                col_idx += 1;
            }
            if self.output_mask[3] {
                // Graph
                if let Some(col) = chunk.column_mut(col_idx) {
                    match graph_name {
                        Some(name) => col.push_string(name.clone()),
                        None => col.push_value(Value::Null),
                    }
                }
            }
        }

        chunk.set_count(batch_size);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        // Keep triples cached for re-execution
    }

    fn name(&self) -> &'static str {
        "RdfTripleScan"
    }
}

// ============================================================================
// RDF Expression Predicate
// ============================================================================

/// Expression predicate for RDF queries.
///
/// Unlike the LPG predicate, this doesn't need a store reference because
/// RDF values are already materialized in the DataChunk columns.
struct RdfExpressionPredicate {
    expression: FilterExpression,
    variable_columns: HashMap<String, usize>,
}

impl RdfExpressionPredicate {
    fn new(expression: FilterExpression, variable_columns: HashMap<String, usize>) -> Self {
        Self {
            expression,
            variable_columns,
        }
    }

    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Property { variable, .. } => {
                // For RDF, treat property access as variable access
                let col_idx = *self.variable_columns.get(variable)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Binary { left, op, right } => {
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Id(var)
            | FilterExpression::Labels(var)
            | FilterExpression::Type(var) => {
                // Treat Id/Labels/Type access as variable lookup for RDF
                let col_idx = *self.variable_columns.get(var)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function_call(name, args, chunk, row)
            }
            // These expression types are not commonly used in RDF FILTER clauses
            FilterExpression::List(_)
            | FilterExpression::Case { .. }
            | FilterExpression::Map(_)
            | FilterExpression::IndexAccess { .. }
            | FilterExpression::SliceAccess { .. }
            | FilterExpression::ListComprehension { .. }
            | FilterExpression::ListPredicate { .. }
            | FilterExpression::ExistsSubquery { .. }
            | FilterExpression::CountSubquery { .. }
            | FilterExpression::Reduce { .. } => None,
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => Some(Value::Bool(left.as_bool()? && right.as_bool()?)),
            BinaryFilterOp::Or => Some(Value::Bool(left.as_bool()? || right.as_bool()?)),
            BinaryFilterOp::Xor => Some(Value::Bool(left.as_bool()? != right.as_bool()?)),
            BinaryFilterOp::Eq => compare_values(left, right, |o| o.is_eq()),
            BinaryFilterOp::Ne => compare_values(left, right, |o| o.is_ne()),
            BinaryFilterOp::Lt => compare_values(left, right, |o| o.is_lt()),
            BinaryFilterOp::Le => compare_values(left, right, |o| o.is_le()),
            BinaryFilterOp::Gt => compare_values(left, right, |o| o.is_gt()),
            BinaryFilterOp::Ge => compare_values(left, right, |o| o.is_ge()),
            BinaryFilterOp::Add => {
                // Helper to convert to f64 for arithmetic
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l + r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l + r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 + r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l + *r as f64)),
                    // Handle string-to-numeric conversion for RDF
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l + r))
                    }
                }
            }
            BinaryFilterOp::Sub => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l - r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l - r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 - r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l - *r as f64)),
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l - r))
                    }
                }
            }
            BinaryFilterOp::Mul => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) => Some(Value::Int64(l * r)),
                    (Value::Float64(l), Value::Float64(r)) => Some(Value::Float64(l * r)),
                    (Value::Int64(l), Value::Float64(r)) => Some(Value::Float64(*l as f64 * r)),
                    (Value::Float64(l), Value::Int64(r)) => Some(Value::Float64(l * *r as f64)),
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        Some(Value::Float64(l * r))
                    }
                }
            }
            BinaryFilterOp::Div => {
                fn to_f64(v: &Value) -> Option<f64> {
                    match v {
                        Value::Int64(i) => Some(*i as f64),
                        Value::Float64(f) => Some(*f),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    }
                }
                match (left, right) {
                    (Value::Int64(l), Value::Int64(r)) if *r != 0 => Some(Value::Int64(l / r)),
                    (Value::Float64(l), Value::Float64(r)) if *r != 0.0 => {
                        Some(Value::Float64(l / r))
                    }
                    (Value::Int64(l), Value::Float64(r)) if *r != 0.0 => {
                        Some(Value::Float64(*l as f64 / r))
                    }
                    (Value::Float64(l), Value::Int64(r)) if *r != 0 => {
                        Some(Value::Float64(l / *r as f64))
                    }
                    _ => {
                        let l = to_f64(left)?;
                        let r = to_f64(right)?;
                        if r != 0.0 {
                            Some(Value::Float64(l / r))
                        } else {
                            None
                        }
                    }
                }
            }
            BinaryFilterOp::Mod => match (left, right) {
                (Value::Int64(l), Value::Int64(r)) if *r != 0 => Some(Value::Int64(l % r)),
                _ => None,
            },
            BinaryFilterOp::Contains => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.contains(&**r))),
                _ => None,
            },
            BinaryFilterOp::StartsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.starts_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::EndsWith => match (left, right) {
                (Value::String(l), Value::String(r)) => Some(Value::Bool(l.ends_with(&**r))),
                _ => None,
            },
            BinaryFilterOp::In => {
                // Not implemented for RDF filter evaluation
                None
            }
            BinaryFilterOp::Regex => {
                // SPARQL REGEX(string, pattern) - returns true if string matches pattern
                match (left, right) {
                    (Value::String(text), Value::String(pattern)) => {
                        // Compile the regex pattern
                        match Regex::new(pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(text))),
                            Err(_) => None, // Invalid regex pattern
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Pow => {
                // Power operation
                match (left, right) {
                    (Value::Int64(base), Value::Int64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp as f64)))
                    }
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Value::Float64(base.powf(*exp)))
                    }
                    (Value::Int64(base), Value::Float64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp)))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Value::Float64(base.powf(*exp as f64)))
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Like => {
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        // Convert SQL LIKE pattern to regex
                        let mut re_pat = String::with_capacity(pattern.len() + 4);
                        re_pat.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => re_pat.push_str(".*"),
                                '_' => re_pat.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        if ".+*?^${}()|[]\\".contains(next) {
                                            re_pat.push('\\');
                                        }
                                        re_pat.push(next);
                                    }
                                }
                                _ => {
                                    if ".+*?^${}()|[]\\".contains(ch) {
                                        re_pat.push('\\');
                                    }
                                    re_pat.push(ch);
                                }
                            }
                        }
                        re_pat.push('$');
                        match Regex::new(&re_pat) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None,
                        }
                    }
                    _ => None,
                }
            }
            BinaryFilterOp::Concat => match (left, right) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::String(a), other) => {
                    let b = match other {
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::Bool(b) => b.to_string(),
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = match other {
                        Value::Int64(i) => i.to_string(),
                        Value::Float64(f) => f.to_string(),
                        Value::Bool(bo) => bo.to_string(),
                        _ => return None,
                    };
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                _ => None,
            },
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => Some(Value::Bool(!val?.as_bool()?)),
            UnaryFilterOp::IsNull => Some(Value::Bool(val.is_none())),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(val.is_some())),
            UnaryFilterOp::Neg => match val? {
                Value::Int64(v) => Some(Value::Int64(-v)),
                Value::Float64(v) => Some(Value::Float64(-v)),
                _ => None,
            },
        }
    }

    /// Evaluates SPARQL function calls.
    fn eval_function_call(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        // Normalize function name to uppercase for case-insensitive matching
        let func_name = name.to_uppercase();

        match func_name.as_str() {
            // CONCAT - concatenate multiple strings
            "CONCAT" => {
                let mut result = String::new();
                for arg in args {
                    if let Some(Value::String(s)) = self.eval_expr(arg, chunk, row) {
                        result.push_str(&s);
                    } else if let Some(val) = self.eval_expr(arg, chunk, row) {
                        // Convert non-string values to string
                        result.push_str(&value_to_string(&val));
                    }
                }
                Some(Value::String(result.into()))
            }

            // REPLACE - replace occurrences of pattern with replacement
            "REPLACE" => {
                if args.len() < 3 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let replacement = match self.eval_expr(&args[2], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };

                // Check if the pattern should be treated as regex (4th argument with 'r' flag)
                if args.len() >= 4
                    && let Some(Value::String(flags)) = self.eval_expr(&args[3], chunk, row)
                    && (flags.contains('r') || flags.contains('i'))
                {
                    // Regex-based replace
                    let regex_pattern = if flags.contains('i') {
                        format!("(?i){}", &pattern)
                    } else {
                        pattern.clone()
                    };
                    if let Ok(re) = Regex::new(&regex_pattern) {
                        return Some(Value::String(re.replace_all(&text, &replacement).into()));
                    }
                }

                // Simple string replace
                Some(Value::String(text.replace(&pattern, &replacement).into()))
            }

            // STRLEN - string length
            "STRLEN" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Int64(text.chars().count() as i64))
            }

            // UCASE - uppercase
            "UCASE" | "UPPER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_uppercase().into()))
            }

            // LCASE - lowercase
            "LCASE" | "LOWER" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::String(text.to_lowercase().into()))
            }

            // SUBSTR - substring extraction
            "SUBSTR" | "SUBSTRING" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let start = match self.eval_expr(&args[1], chunk, row)? {
                    Value::Int64(i) => (i.max(1) - 1) as usize, // SPARQL uses 1-based indexing
                    _ => return None,
                };
                let len = if args.len() >= 3 {
                    match self.eval_expr(&args[2], chunk, row)? {
                        Value::Int64(i) => Some(i.max(0) as usize),
                        _ => return None,
                    }
                } else {
                    None
                };

                let chars: Vec<char> = text.chars().collect();
                let substr: String = if let Some(len) = len {
                    chars.iter().skip(start).take(len).collect()
                } else {
                    chars.iter().skip(start).collect()
                };
                Some(Value::String(substr.into()))
            }

            // STRSTARTS - check if string starts with prefix
            "STRSTARTS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let prefix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.starts_with(&prefix)))
            }

            // STRENDS - check if string ends with suffix
            "STRENDS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let suffix = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.ends_with(&suffix)))
            }

            // CONTAINS - check if string contains substring
            "CONTAINS" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                Some(Value::Bool(text.contains(&pattern)))
            }

            // STRBEFORE - substring before pattern
            "STRBEFORE" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(text[..pos].to_string().into()))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // STRAFTER - substring after pattern
            "STRAFTER" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                if let Some(pos) = text.find(&pattern) {
                    Some(Value::String(
                        text[pos + pattern.len()..].to_string().into(),
                    ))
                } else {
                    Some(Value::String("".into()))
                }
            }

            // ENCODE_FOR_URI - URL encode
            "ENCODE_FOR_URI" => {
                if args.is_empty() {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                // Simple URL encoding for common characters
                let encoded: String = text
                    .chars()
                    .map(|c| match c {
                        'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                        _ => format!("%{:02X}", c as u32),
                    })
                    .collect();
                Some(Value::String(encoded.into()))
            }

            // COALESCE - return first non-null value
            "COALESCE" => {
                for arg in args {
                    if let Some(val) = self.eval_expr(arg, chunk, row)
                        && !matches!(val, Value::Null)
                    {
                        return Some(val);
                    }
                }
                None
            }

            // IF - conditional expression
            "IF" => {
                if args.len() < 3 {
                    return None;
                }
                let condition = self.eval_expr(&args[0], chunk, row)?;
                if condition.as_bool()? {
                    self.eval_expr(&args[1], chunk, row)
                } else {
                    self.eval_expr(&args[2], chunk, row)
                }
            }

            // BOUND - check if variable is bound
            "BOUND" => {
                if args.is_empty() {
                    return None;
                }
                let is_bound = self.eval_expr(&args[0], chunk, row).is_some();
                Some(Value::Bool(is_bound))
            }

            // STR - convert to string
            "STR" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                Some(Value::String(value_to_string(&val).into()))
            }

            // ISIRI / ISURI - check if value is an IRI
            "ISIRI" | "ISURI" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                if let Value::String(s) = val {
                    // Check if it looks like an IRI (starts with a scheme)
                    let is_iri = s.contains("://") || s.starts_with("urn:");
                    Some(Value::Bool(is_iri))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISBLANK - check if value is a blank node
            "ISBLANK" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                if let Value::String(s) = val {
                    Some(Value::Bool(s.starts_with("_:")))
                } else {
                    Some(Value::Bool(false))
                }
            }

            // ISLITERAL - check if value is a literal
            "ISLITERAL" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                // In our model, non-IRI strings and other values are literals
                match &val {
                    Value::String(s) => {
                        Some(Value::Bool(!s.contains("://") && !s.starts_with("_:")))
                    }
                    _ => Some(Value::Bool(true)),
                }
            }

            // ISNUMERIC - check if value is numeric
            "ISNUMERIC" => {
                if args.is_empty() {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let is_numeric = matches!(val, Value::Int64(_) | Value::Float64(_))
                    || matches!(&val, Value::String(s) if s.parse::<f64>().is_ok());
                Some(Value::Bool(is_numeric))
            }

            // ABS - absolute value
            "ABS" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v.abs())),
                    Value::Float64(v) => Some(Value::Float64(v.abs())),
                    _ => None,
                }
            }

            // CEIL - ceiling
            "CEIL" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.ceil())),
                    _ => None,
                }
            }

            // FLOOR - floor
            "FLOOR" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.floor())),
                    _ => None,
                }
            }

            // ROUND - round to nearest integer
            "ROUND" => {
                if args.is_empty() {
                    return None;
                }
                match self.eval_expr(&args[0], chunk, row)? {
                    Value::Int64(v) => Some(Value::Int64(v)),
                    Value::Float64(v) => Some(Value::Float64(v.round())),
                    _ => None,
                }
            }

            // REGEX - regular expression matching
            "REGEX" => {
                if args.len() < 2 {
                    return None;
                }
                let text = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    v => value_to_string(&v),
                };
                let pattern = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s.to_string(),
                    _ => return None,
                };
                // Optional flags argument (3rd arg): "i" for case-insensitive
                let regex_pattern = if args.len() >= 3
                    && let Some(Value::String(flags)) = self.eval_expr(&args[2], chunk, row)
                    && flags.contains('i')
                {
                    format!("(?i){pattern}")
                } else {
                    pattern
                };
                match Regex::new(&regex_pattern) {
                    Ok(re) => Some(Value::Bool(re.is_match(&text))),
                    Err(_) => None,
                }
            }

            // ================================================================
            // Date/Time Functions (SPARQL 1.1 Section 17.4.5)
            // ================================================================

            // NOW - current datetime
            "NOW" => {
                let ts = grafeo_common::types::Timestamp::now();
                Some(Value::Timestamp(ts))
            }

            // YEAR - extract year from date/datetime
            "YEAR" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.year()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().year()))),
                    Value::String(s) => grafeo_common::types::Date::parse(&s)
                        .map(|d| Value::Int64(i64::from(d.year()))),
                    _ => None,
                }
            }

            // MONTH - extract month from date/datetime
            "MONTH" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.month()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().month()))),
                    Value::String(s) => grafeo_common::types::Date::parse(&s)
                        .map(|d| Value::Int64(i64::from(d.month()))),
                    _ => None,
                }
            }

            // DAY - extract day from date/datetime
            "DAY" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Int64(i64::from(d.day()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_date().day()))),
                    Value::String(s) => grafeo_common::types::Date::parse(&s)
                        .map(|d| Value::Int64(i64::from(d.day()))),
                    _ => None,
                }
            }

            // HOURS - extract hours from time/datetime
            "HOURS" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.hour()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().hour()))),
                    Value::String(s) => grafeo_common::types::Time::parse(&s)
                        .map(|t| Value::Int64(i64::from(t.hour()))),
                    _ => None,
                }
            }

            // MINUTES - extract minutes from time/datetime
            "MINUTES" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => Some(Value::Int64(i64::from(t.minute()))),
                    Value::Timestamp(ts) => Some(Value::Int64(i64::from(ts.to_time().minute()))),
                    Value::String(s) => grafeo_common::types::Time::parse(&s)
                        .map(|t| Value::Int64(i64::from(t.minute()))),
                    _ => None,
                }
            }

            // SECONDS - extract seconds (with fractional) from time/datetime
            "SECONDS" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let to_secs = |t: &grafeo_common::types::Time| {
                    f64::from(t.second()) + f64::from(t.nanosecond()) / 1_000_000_000.0
                };
                match val {
                    Value::Time(t) => Some(Value::Float64(to_secs(&t))),
                    Value::Timestamp(ts) => Some(Value::Float64(to_secs(&ts.to_time()))),
                    Value::String(s) => {
                        grafeo_common::types::Time::parse(&s).map(|t| Value::Float64(to_secs(&t)))
                    }
                    _ => None,
                }
            }

            // TIMEZONE - extract timezone as xsd:dayTimeDuration
            "TIMEZONE" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => t.offset_seconds().map(|offset| {
                        Value::Duration(grafeo_common::types::Duration::from_seconds(i64::from(
                            offset,
                        )))
                    }),
                    Value::ZonedDatetime(zdt) => Some(Value::Duration(
                        grafeo_common::types::Duration::from_seconds(i64::from(
                            zdt.offset_seconds(),
                        )),
                    )),
                    _ => None,
                }
            }

            // TZ - extract timezone as string ("+05:00", "Z", "")
            "TZ" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::Time(t) => {
                        if let Some(offset) = t.offset_seconds() {
                            Some(Value::String(format_tz_offset(offset).into()))
                        } else {
                            Some(Value::String(String::new().into()))
                        }
                    }
                    Value::ZonedDatetime(zdt) => {
                        Some(Value::String(format_tz_offset(zdt.offset_seconds()).into()))
                    }
                    _ => Some(Value::String(String::new().into())),
                }
            }

            // ================================================================
            // Hash Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================
            "MD5" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                let digest = md5::compute(s.as_bytes());
                Some(Value::String(format!("{digest:x}").into()))
            }

            "SHA1" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha1::Digest as _;
                let hash = sha1::Sha1::digest(s.as_bytes());
                Some(Value::String(format!("{hash:x}").into()))
            }

            "SHA256" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha256::digest(s.as_bytes());
                Some(Value::String(format!("{hash:x}").into()))
            }

            "SHA384" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha384::digest(s.as_bytes());
                Some(Value::String(format!("{hash:x}").into()))
            }

            "SHA512" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let s = value_to_string(&val);
                use sha2::Digest as _;
                let hash = sha2::Sha512::digest(s.as_bytes());
                Some(Value::String(format!("{hash:x}").into()))
            }

            // ================================================================
            // RDF Term Functions (SPARQL 1.1 Section 17.4.2)
            // ================================================================

            // LANG - language tag of a literal
            "LANG" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                // In RDF execution, language-tagged literals are stored as strings
                // with the language tag in metadata. For now, return empty string.
                match val {
                    Value::String(_) => Some(Value::String(String::new().into())),
                    _ => Some(Value::String(String::new().into())),
                }
            }

            // DATATYPE - datatype IRI of a literal
            "DATATYPE" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                let dt = match &val {
                    Value::String(_) => "http://www.w3.org/2001/XMLSchema#string",
                    Value::Int64(_) => "http://www.w3.org/2001/XMLSchema#integer",
                    Value::Float64(_) => "http://www.w3.org/2001/XMLSchema#double",
                    Value::Bool(_) => "http://www.w3.org/2001/XMLSchema#boolean",
                    Value::Date(_) => "http://www.w3.org/2001/XMLSchema#date",
                    Value::Time(_) => "http://www.w3.org/2001/XMLSchema#time",
                    Value::Timestamp(_) => "http://www.w3.org/2001/XMLSchema#dateTime",
                    Value::Duration(_) => "http://www.w3.org/2001/XMLSchema#duration",
                    _ => return None,
                };
                Some(Value::String(dt.to_string().into()))
            }

            // IRI / URI - construct an IRI
            "IRI" | "URI" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    v => Some(Value::String(value_to_string(&v).into())),
                }
            }

            // BNODE - construct or retrieve a blank node
            "BNODE" => {
                if args.is_empty() {
                    // Generate a unique blank node ID
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static BNODE_COUNTER: AtomicU64 = AtomicU64::new(0);
                    let id = BNODE_COUNTER.fetch_add(1, Ordering::Relaxed);
                    Some(Value::String(format!("_:b{id}").into()))
                } else {
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let label = value_to_string(&val);
                    Some(Value::String(format!("_:b{label}").into()))
                }
            }

            // STRDT - construct a typed literal
            "STRDT" => {
                if args.len() < 2 {
                    return None;
                }
                // Return the string value (type information is metadata)
                self.eval_expr(&args[0], chunk, row)
            }

            // STRLANG - construct a language-tagged literal
            "STRLANG" => {
                if args.len() < 2 {
                    return None;
                }
                // Return the string value (language tag is metadata)
                self.eval_expr(&args[0], chunk, row)
            }

            // UUID - generate a UUID IRI
            "UUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static UUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = UUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("urn:uuid:{ts:032x}-{id:04x}").into()))
            }

            // STRUUID - generate a UUID string (no urn: prefix)
            "STRUUID" => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static STRUUID_COUNTER: AtomicU64 = AtomicU64::new(0);
                let id = STRUUID_COUNTER.fetch_add(1, Ordering::Relaxed);
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                Some(Value::String(format!("{ts:032x}-{id:04x}").into()))
            }

            // sameTerm - strict RDF term equality
            "SAMETERM" => {
                if args.len() < 2 {
                    return None;
                }
                let a = self.eval_expr(&args[0], chunk, row)?;
                let b = self.eval_expr(&args[1], chunk, row)?;
                Some(Value::Bool(a == b))
            }

            // ================================================================
            // Numeric Functions (SPARQL 1.1 Section 17.4.4)
            // ================================================================

            // RAND - random double in [0, 1)
            "RAND" => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                use std::sync::atomic::{AtomicU64, Ordering};
                static RAND_STATE: AtomicU64 = AtomicU64::new(0);
                let state = RAND_STATE.fetch_add(1, Ordering::Relaxed);
                let mut hasher = DefaultHasher::new();
                state.hash(&mut hasher);
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0u64, |d| d.as_nanos() as u64)
                    .hash(&mut hasher);
                let bits = hasher.finish();
                let value = (bits >> 11) as f64 / (1u64 << 53) as f64;
                Some(Value::Float64(value))
            }

            // Unknown function
            _ => None,
        }
    }
}

/// Formats a timezone offset in seconds as "+HH:MM" or "Z".
fn format_tz_offset(offset_secs: i32) -> String {
    if offset_secs == 0 {
        return "Z".to_string();
    }
    let sign = if offset_secs >= 0 { '+' } else { '-' };
    let abs = offset_secs.unsigned_abs();
    let hours = abs / 3600;
    let minutes = (abs % 3600) / 60;
    format!("{sign}{hours:02}:{minutes:02}")
}

impl Predicate for RdfExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        matches!(self.eval(chunk, row), Some(Value::Bool(true)))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Converts an RDF Term to a string for IRI/blank node representation.
fn term_to_string(term: &Term) -> String {
    match term {
        Term::Iri(iri) => iri.as_str().to_string(),
        Term::BlankNode(bnode) => format!("_:{}", bnode.id()),
        Term::Literal(lit) => lit.value().to_string(),
    }
}

/// Pushes an RDF term value to a column, preserving type where possible.
/// Pushes an RDF term value to a column.
///
/// For RDF columns (which use String type), we always push as string to avoid
/// type mismatches. The typed literal's value is preserved as a string, and
/// numeric comparisons are handled at the filter level.
fn push_term_value(col: &mut grafeo_core::execution::ValueVector, term: &Term) {
    match term {
        Term::Iri(iri) => col.push_string(iri.as_str().to_string()),
        Term::BlankNode(bnode) => col.push_string(format!("_:{}", bnode.id())),
        Term::Literal(lit) => {
            // Always push as string since RDF columns are String type
            // Numeric operations are handled by the filter evaluation
            col.push_string(lit.value().to_string());
        }
    }
}

/// Converts a TripleComponent to an Option<Term> for pattern matching.
fn component_to_term(component: &TripleComponent) -> Option<Term> {
    match component {
        TripleComponent::Variable(_) => None,
        TripleComponent::Iri(iri) => Some(Term::iri(iri.clone())),
        TripleComponent::Literal(value) => match value {
            Value::String(s) => Some(Term::literal(s.clone())),
            Value::Int64(i) => Some(Term::typed_literal(i.to_string(), Literal::XSD_INTEGER)),
            Value::Float64(f) => Some(Term::typed_literal(f.to_string(), Literal::XSD_DOUBLE)),
            Value::Bool(b) => Some(Term::typed_literal(b.to_string(), Literal::XSD_BOOLEAN)),
            _ => Some(Term::literal(value.to_string())),
        },
    }
}

/// Derives RDF schema (all String type for simplicity).
fn derive_rdf_schema(columns: &[String]) -> Vec<LogicalType> {
    columns.iter().map(|_| LogicalType::String).collect()
}

/// Quick cardinality estimate for a logical operator subtree.
///
/// Uses store index sizes to estimate how many rows an operator produces,
/// without collecting full statistics. Returns `None` if estimation is
/// not possible for this operator type.
fn estimate_operator_cardinality(
    op: &crate::query::plan::LogicalOperator,
    store: &RdfStore,
) -> Option<f64> {
    use crate::query::plan::LogicalOperator;
    match op {
        LogicalOperator::TripleScan(scan) => {
            let stats = store.stats();
            let total = stats.triple_count as f64;
            if total == 0.0 {
                return Some(0.0);
            }

            // Estimate based on which components are bound
            let s_bound = matches!(
                scan.subject,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
            );
            let p_bound = matches!(
                scan.predicate,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
            );
            let o_bound = matches!(
                scan.object,
                crate::query::plan::TripleComponent::Iri(_)
                    | crate::query::plan::TripleComponent::Literal(_)
            );

            let estimate = match (s_bound, p_bound, o_bound) {
                (true, true, true) => 1.0,
                (true, true, false) => total / stats.subject_count.max(1) as f64,
                (true, false, true) => total / stats.subject_count.max(1) as f64,
                (false, true, true) => total / stats.predicate_count.max(1) as f64,
                (true, false, false) => total / stats.subject_count.max(1) as f64,
                (false, true, false) => total / stats.predicate_count.max(1) as f64,
                (false, false, true) => total / stats.object_count.max(1) as f64,
                (false, false, false) => total,
            };
            Some(estimate.max(1.0))
        }
        LogicalOperator::Filter(f) => {
            estimate_operator_cardinality(&f.input, store).map(|c| (c * 0.33).max(1.0))
        }
        LogicalOperator::Join(j) => {
            let left = estimate_operator_cardinality(&j.left, store)?;
            let right = estimate_operator_cardinality(&j.right, store)?;
            Some((left * right * 0.1).max(1.0))
        }
        LogicalOperator::Limit(l) => {
            if let crate::query::plan::CountExpr::Literal(n) = l.count {
                Some(n as f64)
            } else {
                estimate_operator_cardinality(&l.input, store)
            }
        }
        _ => None,
    }
}

/// Resolves an expression to a column index.
fn resolve_expression(
    expr: &LogicalExpression,
    variable_columns: &HashMap<String, usize>,
) -> Result<usize> {
    match expr {
        LogicalExpression::Variable(name) => variable_columns
            .get(name)
            .copied()
            .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
        _ => {
            // Complex expression (CASE, Binary, etc.): look up synthetic column
            let col_name = format!("__expr_{:?}", expr);
            variable_columns.get(&col_name).copied().ok_or_else(|| {
                Error::Internal(format!("Cannot resolve expression to column: {:?}", expr))
            })
        }
    }
}

// expression_to_string is now in planner/common.rs
use crate::query::planner::common::expression_to_string;

/// Converts a value to its string representation.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::String(s) => s.to_string(),
        Value::Bytes(b) => String::from_utf8_lossy(b).to_string(),
        Value::Timestamp(t) => t.to_string(),
        Value::Date(d) => d.to_string(),
        Value::Time(t) => t.to_string(),
        Value::Duration(d) => d.to_string(),
        Value::List(items) => {
            let parts: Vec<String> = items.iter().map(value_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Map(entries) => {
            let parts: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_string(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Vector(v) => {
            let parts: Vec<String> = v.iter().map(|f| f.to_string()).collect();
            format!("vector([{}])", parts.join(", "))
        }
        Value::ZonedDatetime(zdt) => zdt.to_string(),
        Value::Path { nodes, edges } => {
            format!("<path: {} nodes, {} edges>", nodes.len(), edges.len())
        }
    }
}

/// Compares two values and returns a boolean result.
fn compare_values<F>(left: &Value, right: &Value, cmp: F) -> Option<Value>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    let ordering = match (left, right) {
        (Value::Int64(l), Value::Int64(r)) => l.cmp(r),
        (Value::Float64(l), Value::Float64(r)) => l.partial_cmp(r)?,
        (Value::String(l), Value::String(r)) => {
            // Try numeric comparison first if both look like numbers
            if let (Ok(l_num), Ok(r_num)) = (l.parse::<f64>(), r.parse::<f64>()) {
                l_num.partial_cmp(&r_num)?
            } else {
                l.cmp(r)
            }
        }
        (Value::Int64(l), Value::Float64(r)) => (*l as f64).partial_cmp(r)?,
        (Value::Float64(l), Value::Int64(r)) => l.partial_cmp(&(*r as f64))?,
        // RDF values are often stored as strings - try numeric conversion
        (Value::String(s), Value::Int64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(&(*r as f64))?
        }
        (Value::String(s), Value::Float64(r)) => {
            let l_num = s.parse::<f64>().ok()?;
            l_num.partial_cmp(r)?
        }
        (Value::Int64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            (*l as f64).partial_cmp(&r_num)?
        }
        (Value::Float64(l), Value::String(s)) => {
            let r_num = s.parse::<f64>().ok()?;
            l.partial_cmp(&r_num)?
        }
        _ => return None,
    };
    Some(Value::Bool(cmp(ordering)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::LogicalPlan;

    #[test]
    fn test_rdf_planner_simple_scan() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Variable("p".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        assert_eq!(physical.columns, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_rdf_planner_with_pattern() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/age"),
            Term::typed_literal("30", "http://www.w3.org/2001/XMLSchema#integer"),
        ));

        let planner = RdfPlanner::new(store);

        let scan = TripleScanOp {
            subject: TripleComponent::Variable("s".to_string()),
            predicate: TripleComponent::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
            object: TripleComponent::Variable("o".to_string()),
            graph: None,
            input: None,
        };

        let plan = LogicalPlan::new(LogicalOperator::TripleScan(scan));
        let physical = planner.plan(&plan).unwrap();

        // Only s and o are variables (predicate is fixed)
        assert_eq!(physical.columns, vec!["s", "o"]);
    }

    #[test]
    fn test_rdf_scan_operator_chunking() {
        let store = Arc::new(RdfStore::new());

        // Insert 100 triples
        for i in 0..100 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/item{}", i)),
                Term::iri("http://example.org/value"),
                Term::literal(i.to_string()),
            ));
        }

        let pattern = TriplePattern {
            subject: None,
            predicate: None,
            object: None,
        };

        let mut operator = RdfTripleScanOperator::new(
            Arc::clone(&store),
            pattern,
            [true, true, true, false],
            30,
            None,
            false,
        );

        let mut total_rows = 0;
        while let Ok(Some(chunk)) = operator.next() {
            total_rows += chunk.row_count();
            assert!(chunk.row_count() <= 30); // Respects chunk size
        }

        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_copy_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Insert triples into a named graph
        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));
        assert_eq!(src.len(), 1);

        // Copy src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source still has its data
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has a copy
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_move_graph_operator() {
        let store = Arc::new(RdfStore::new());

        store.create_graph("http://example.org/src");
        let src = store.graph("http://example.org/src").unwrap();
        src.insert(Triple::new(
            Term::iri("http://example.org/a"),
            Term::iri("http://example.org/p"),
            Term::literal("val"),
        ));

        // Move src -> dst via operator
        let plan = LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source is gone (move drops it)
        assert!(store.graph("http://example.org/src").is_none());
        // Destination has the data
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 1);
    }

    #[test]
    fn test_add_graph_operator() {
        let store = Arc::new(RdfStore::new());

        // Create src with 1 triple
        store.create_graph("http://example.org/src");
        store
            .graph("http://example.org/src")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/a"),
                Term::iri("http://example.org/p"),
                Term::literal("from-src"),
            ));

        // Create dst with 1 different triple
        store.create_graph("http://example.org/dst");
        store
            .graph("http://example.org/dst")
            .unwrap()
            .insert(Triple::new(
                Term::iri("http://example.org/b"),
                Term::iri("http://example.org/q"),
                Term::literal("from-dst"),
            ));

        // Add src -> dst via operator (merges)
        let plan = LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: Some("http://example.org/src".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;
        while op.next().unwrap().is_some() {}

        // Source unchanged
        assert_eq!(store.graph("http://example.org/src").unwrap().len(), 1);
        // Destination has both triples (union)
        assert_eq!(store.graph("http://example.org/dst").unwrap().len(), 2);
    }

    #[test]
    fn test_copy_nonexistent_source_errors_without_silent() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: false,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should error because source doesn't exist
        let result = op.next();
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_nonexistent_source_silent_ok() {
        let store = Arc::new(RdfStore::new());

        let plan = LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: Some("http://example.org/nope".to_string()),
            destination: Some("http://example.org/dst".to_string()),
            silent: true,
        }));
        let planner = RdfPlanner::new(Arc::clone(&store));
        let physical = planner.plan(&plan).unwrap();
        let mut op = physical.operator;

        // Should succeed silently
        assert!(op.next().is_ok());
    }
}
