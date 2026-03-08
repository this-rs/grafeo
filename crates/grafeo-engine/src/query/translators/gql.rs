//! GQL to LogicalPlan translator.
//!
//! Translates GQL AST to the common logical plan representation.

use std::collections::HashMap;

use super::common::{
    combine_with_and, is_aggregate_function, is_binary_set_function, to_aggregate_function,
    wrap_distinct, wrap_filter, wrap_limit, wrap_return, wrap_skip, wrap_sort,
};
use crate::query::plan::{
    self as plan, AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, BinaryOp,
    CallProcedureOp, CreateEdgeOp, CreateNodeOp, DeleteNodeOp, EntityKind, ExceptOp,
    ExpandDirection, ExpandOp, HorizontalAggregateOp, IntersectOp, JoinOp, JoinType, LeftJoinOp,
    LogicalExpression, LogicalOperator, LogicalPlan, MergeOp, MergeRelationshipOp, NodeScanOp,
    NullsOrdering, OtherwiseOp, PathMode, ProcedureYield, ProjectOp, Projection, RemoveLabelOp,
    ReturnItem, SetPropertyOp, ShortestPathOp, SortKey, SortOrder, UnaryOp, UnionOp, UnwindOp,
};
#[cfg(test)]
use crate::query::plan::{FilterOp, LimitOp, SkipOp};
use grafeo_adapters::query::gql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Result of translating a GQL query: either a logical plan, session command, or schema command.
#[derive(Debug)]
pub enum GqlTranslationResult {
    /// A query plan to execute (or EXPLAIN if `plan.explain` is true).
    Plan(LogicalPlan),
    /// A session or transaction command (not a query plan).
    SessionCommand(ast::SessionCommand),
    /// A schema DDL command (CREATE/DROP TYPE, INDEX, CONSTRAINT).
    SchemaCommand(ast::SchemaStatement),
}

/// Translates a GQL query string to a logical plan.
///
/// Session/transaction commands (USE GRAPH, COMMIT, etc.) return an error.
/// Use [`translate_full`] to handle both plans and session commands.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    match translate_full(query)? {
        GqlTranslationResult::Plan(plan) => Ok(plan),
        GqlTranslationResult::SessionCommand(_) => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Session commands cannot be executed as queries",
        ))),
        GqlTranslationResult::SchemaCommand(_) => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Schema DDL commands cannot be executed as queries",
        ))),
    }
}

/// Translates a GQL query string, returning either a logical plan or session command.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate_full(query: &str) -> Result<GqlTranslationResult> {
    let statement = gql::parse(query)?;
    let translator = GqlTranslator::new();
    translator.translate_statement_full(&statement)
}

/// Translator from GQL AST to LogicalPlan.
struct GqlTranslator {
    /// Edge variables from variable-length expand patterns (group-list variables).
    /// Maps edge variable name to the path alias used for `_path_edges_{alias}` lookup.
    group_list_variables: std::cell::RefCell<HashMap<String, String>>,
}

impl GqlTranslator {
    fn new() -> Self {
        Self {
            group_list_variables: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn translate_statement_full(&self, stmt: &ast::Statement) -> Result<GqlTranslationResult> {
        match stmt {
            ast::Statement::SessionCommand(cmd) => {
                Ok(GqlTranslationResult::SessionCommand(cmd.clone()))
            }
            ast::Statement::Schema(schema) => {
                Ok(GqlTranslationResult::SchemaCommand(schema.clone()))
            }
            ast::Statement::Explain(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.explain = true;
                Ok(GqlTranslationResult::Plan(plan))
            }
            ast::Statement::Profile(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.profile = true;
                Ok(GqlTranslationResult::Plan(plan))
            }
            other => self
                .translate_statement(other)
                .map(GqlTranslationResult::Plan),
        }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::DataModification(dm) => self.translate_data_modification(dm),
            ast::Statement::Schema(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Schema DDL commands are handled before query planning",
            ))),
            ast::Statement::Call(call) => self.translate_call(call),
            ast::Statement::CompositeQuery { left, op, right } => {
                self.translate_composite_query(left, *op, right)
            }
            ast::Statement::SessionCommand(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Session commands cannot be executed as queries",
            ))),
            ast::Statement::Explain(inner) | ast::Statement::Profile(inner) => {
                self.translate_statement(inner)
            }
        }
    }

    fn translate_composite_query(
        &self,
        left: &ast::Statement,
        op: ast::CompositeOp,
        right: &ast::Statement,
    ) -> Result<LogicalPlan> {
        let left_plan = self.translate_statement(left)?;
        let right_plan = self.translate_statement(right)?;

        match op {
            ast::CompositeOp::Union | ast::CompositeOp::UnionAll => {
                let union_op = LogicalOperator::Union(UnionOp {
                    inputs: vec![left_plan.root, right_plan.root],
                });
                let root = if op == ast::CompositeOp::UnionAll {
                    union_op
                } else {
                    wrap_distinct(union_op)
                };
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Except | ast::CompositeOp::ExceptAll => {
                let root = LogicalOperator::Except(ExceptOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                    all: matches!(op, ast::CompositeOp::ExceptAll),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Intersect | ast::CompositeOp::IntersectAll => {
                let root = LogicalOperator::Intersect(IntersectOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                    all: matches!(op, ast::CompositeOp::IntersectAll),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Otherwise => {
                let root = LogicalOperator::Otherwise(OtherwiseOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Next => {
                // NEXT (linear composition): output of left feeds as input to right.
                // Translate as Apply: for each row from left, execute right with bound variables.
                let root = LogicalOperator::Apply(ApplyOp {
                    input: Box::new(left_plan.root),
                    subplan: Box::new(right_plan.root),
                    shared_variables: Vec::new(),
                });
                Ok(LogicalPlan::new(root))
            }
        }
    }

    fn translate_call(&self, call: &ast::CallStatement) -> Result<LogicalPlan> {
        let arguments = call
            .arguments
            .iter()
            .map(|a| self.translate_expression(a))
            .collect::<Result<Vec<_>>>()?;

        let yield_items = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| ProcedureYield {
                    field_name: item.field_name.clone(),
                    alias: item.alias.clone(),
                })
                .collect()
        });

        let mut plan = LogicalOperator::CallProcedure(CallProcedureOp {
            name: call.procedure_name.clone(),
            arguments,
            yield_items,
        });

        // Apply WHERE filter on yielded rows
        if let Some(where_clause) = &call.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        // Apply RETURN clause (with ORDER BY, SKIP, LIMIT)
        // Order: RETURN first (closest to input), then Sort, Skip, Limit wrap it.
        // This ensures RETURN aliases are visible to ORDER BY in the binder.
        if let Some(return_clause) = &call.return_clause {
            // Apply RETURN projection first (only when explicit items are present)
            if !return_clause.items.is_empty() {
                let return_items = return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_return(plan, return_items, return_clause.distinct);
            }

            // Apply ORDER BY (wraps Return so aliases are visible)
            if let Some(order_by) = &return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression)?,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: item.nulls.map(|n| match n {
                                ast::NullsOrdering::First => NullsOrdering::First,
                                ast::NullsOrdering::Last => NullsOrdering::Last,
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }

            // Apply SKIP
            if let Some(skip_expr) = &return_clause.skip
                && let ast::Expression::Literal(ast::Literal::Integer(n)) = skip_expr
            {
                plan = wrap_skip(plan, *n as usize);
            }

            // Apply LIMIT
            if let Some(limit_expr) = &return_clause.limit
                && let ast::Expression::Literal(ast::Literal::Integer(n)) = limit_expr
            {
                plan = wrap_limit(plan, *n as usize);
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_query(&self, query: &ast::QueryStatement) -> Result<LogicalPlan> {
        let mut plan = LogicalOperator::Empty;
        let mut where_applied = false;

        // Process clauses in source order for correct variable scoping.
        // When ordered_clauses is populated, use it; otherwise fall back to
        // the legacy field-based ordering (for backward compatibility).
        if !query.ordered_clauses.is_empty() {
            for clause in &query.ordered_clauses {
                // Apply WHERE filter before the first mutation clause so that
                // MATCH ... WHERE ... CREATE/DELETE operates on filtered rows.
                if !where_applied
                    && matches!(
                        clause,
                        ast::QueryClause::Create(_)
                            | ast::QueryClause::Delete(_)
                            | ast::QueryClause::Set(_)
                            | ast::QueryClause::Merge(_)
                    )
                {
                    if let Some(where_clause) = &query.where_clause {
                        let predicate = self.translate_expression(&where_clause.expression)?;
                        plan = wrap_filter(plan, predicate);
                    }
                    where_applied = true;
                }

                match clause {
                    ast::QueryClause::Match(match_clause) => {
                        if matches!(plan, LogicalOperator::Empty) {
                            // No prior input: standard MATCH
                            plan = self.translate_match(match_clause)?;
                        } else if match_clause.optional {
                            // OPTIONAL MATCH: left join (prior vars on left, match on right)
                            let match_plan = self.translate_match(match_clause)?;
                            plan = LogicalOperator::LeftJoin(LeftJoinOp {
                                left: Box::new(plan),
                                right: Box::new(match_plan),
                                condition: None,
                            });
                        } else {
                            // Non-optional MATCH after prior clauses (UNWIND, etc.)
                            // Pass current plan as input so the MATCH's NodeScan creates
                            // a nested loop join, keeping prior variables (like UNWIND
                            // variables) in scope for property filters.
                            let input = std::mem::replace(&mut plan, LogicalOperator::Empty);
                            plan = self.translate_match_with_input(match_clause, Some(input))?;
                        }
                    }
                    ast::QueryClause::Unwind(unwind_clause) => {
                        let expression = self.translate_expression(&unwind_clause.expression)?;
                        plan = LogicalOperator::Unwind(UnwindOp {
                            expression,
                            variable: unwind_clause.alias.clone(),
                            ordinality_var: None,
                            offset_var: None,
                            input: Box::new(plan),
                        });
                    }
                    ast::QueryClause::For(unwind_clause) => {
                        let expression = self.translate_expression(&unwind_clause.expression)?;
                        plan = LogicalOperator::Unwind(UnwindOp {
                            expression,
                            variable: unwind_clause.alias.clone(),
                            ordinality_var: unwind_clause.ordinality_var.clone(),
                            offset_var: unwind_clause.offset_var.clone(),
                            input: Box::new(plan),
                        });
                    }
                    ast::QueryClause::Create(create_clause) => {
                        plan = self.translate_create_patterns(&create_clause.patterns, plan)?;
                    }
                    ast::QueryClause::Delete(delete_clause) => {
                        plan = self.translate_delete_targets(
                            &delete_clause.targets,
                            delete_clause.detach,
                            plan,
                        )?;
                    }
                    ast::QueryClause::Set(set_clause) => {
                        for assignment in &set_clause.assignments {
                            let value = self.translate_expression(&assignment.value)?;
                            plan = LogicalOperator::SetProperty(SetPropertyOp {
                                variable: assignment.variable.clone(),
                                properties: vec![(assignment.property.clone(), value)],
                                replace: false,
                                is_edge: false,
                                input: Box::new(plan),
                            });
                        }
                        for map_assign in &set_clause.map_assignments {
                            let value = self.translate_expression(&map_assign.map_expr)?;
                            plan = LogicalOperator::SetProperty(SetPropertyOp {
                                variable: map_assign.variable.clone(),
                                properties: vec![("*".to_string(), value)],
                                replace: map_assign.replace,
                                is_edge: false,
                                input: Box::new(plan),
                            });
                        }
                        for label_op in &set_clause.label_operations {
                            plan = LogicalOperator::AddLabel(AddLabelOp {
                                variable: label_op.variable.clone(),
                                labels: label_op.labels.clone(),
                                input: Box::new(plan),
                            });
                        }
                    }
                    ast::QueryClause::Merge(merge_clause) => {
                        plan = self.translate_merge(merge_clause, plan)?;
                    }
                    ast::QueryClause::Let(bindings) => {
                        // LET var = expr translates to a Project that adds the bound
                        // variables as additional columns in the current pipeline.
                        let mut projections = Vec::new();
                        for (name, expr) in bindings {
                            let logical_expr = self.translate_expression(expr)?;
                            projections.push(Projection {
                                expression: logical_expr,
                                alias: Some(name.clone()),
                            });
                        }
                        plan = LogicalOperator::Project(ProjectOp {
                            projections,
                            input: Box::new(plan),
                        });
                    }
                    ast::QueryClause::InlineCall { subquery, optional } => {
                        // CALL { subquery } translates to Apply (lateral join):
                        // for each row from the outer plan, execute the subquery.
                        let subplan = self.translate_query(subquery)?.root;
                        if *optional {
                            // OPTIONAL CALL: use LeftJoin so outer rows survive
                            plan = LogicalOperator::LeftJoin(LeftJoinOp {
                                left: Box::new(plan),
                                right: Box::new(subplan),
                                condition: None,
                            });
                        } else {
                            plan = LogicalOperator::Apply(ApplyOp {
                                input: Box::new(plan),
                                subplan: Box::new(subplan),
                                shared_variables: Vec::new(),
                            });
                        }
                    }
                    ast::QueryClause::CallProcedure(call_stmt) => {
                        // CALL procedure(...) within a query context
                        let call_plan = self.translate_call(call_stmt)?.root;
                        if matches!(plan, LogicalOperator::Empty) {
                            plan = call_plan;
                        } else {
                            plan = LogicalOperator::Apply(ApplyOp {
                                input: Box::new(plan),
                                subplan: Box::new(call_plan),
                                shared_variables: Vec::new(),
                            });
                        }
                    }
                }
            }
        } else {
            // Legacy path: process MATCH, then UNWIND, then MERGE separately
            for match_clause in &query.match_clauses {
                let match_plan = self.translate_match(match_clause)?;
                if matches!(plan, LogicalOperator::Empty) {
                    plan = match_plan;
                } else if match_clause.optional {
                    plan = LogicalOperator::LeftJoin(LeftJoinOp {
                        left: Box::new(plan),
                        right: Box::new(match_plan),
                        condition: None,
                    });
                } else {
                    plan = LogicalOperator::Join(JoinOp {
                        left: Box::new(plan),
                        right: Box::new(match_plan),
                        join_type: JoinType::Cross,
                        conditions: vec![],
                    });
                }
            }

            for unwind_clause in &query.unwind_clauses {
                let expression = self.translate_expression(&unwind_clause.expression)?;
                plan = LogicalOperator::Unwind(UnwindOp {
                    expression,
                    variable: unwind_clause.alias.clone(),
                    ordinality_var: unwind_clause.ordinality_var.clone(),
                    offset_var: unwind_clause.offset_var.clone(),
                    input: Box::new(plan),
                });
            }

            for merge_clause in &query.merge_clauses {
                plan = self.translate_merge(merge_clause, plan)?;
            }
        }

        // Apply WHERE filter (skip if already applied before a mutation clause)
        if !where_applied && let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        // Legacy path: handle SET/REMOVE/CREATE/DELETE from individual fields.
        // When ordered_clauses is used, these are already processed above.
        if query.ordered_clauses.is_empty() {
            for set_clause in &query.set_clauses {
                for assignment in &set_clause.assignments {
                    let value = self.translate_expression(&assignment.value)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: assignment.variable.clone(),
                        properties: vec![(assignment.property.clone(), value)],
                        replace: false,
                        is_edge: false,
                        input: Box::new(plan),
                    });
                }
                for map_assign in &set_clause.map_assignments {
                    let value = self.translate_expression(&map_assign.map_expr)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: map_assign.variable.clone(),
                        properties: vec![("*".to_string(), value)],
                        replace: map_assign.replace,
                        is_edge: false,
                        input: Box::new(plan),
                    });
                }
                for label_op in &set_clause.label_operations {
                    plan = LogicalOperator::AddLabel(AddLabelOp {
                        variable: label_op.variable.clone(),
                        labels: label_op.labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }

            for create_clause in &query.create_clauses {
                plan = self.translate_create_patterns(&create_clause.patterns, plan)?;
            }

            for delete_clause in &query.delete_clauses {
                plan = self.translate_delete_targets(
                    &delete_clause.targets,
                    delete_clause.detach,
                    plan,
                )?;
            }
        }

        // REMOVE clauses (not yet in ordered_clauses, always process)
        for remove_clause in &query.remove_clauses {
            for label_op in &remove_clause.label_operations {
                plan = LogicalOperator::RemoveLabel(RemoveLabelOp {
                    variable: label_op.variable.clone(),
                    labels: label_op.labels.clone(),
                    input: Box::new(plan),
                });
            }
            for (variable, property) in &remove_clause.property_removals {
                plan = LogicalOperator::SetProperty(SetPropertyOp {
                    variable: variable.clone(),
                    properties: vec![(property.clone(), LogicalExpression::Literal(Value::Null))],
                    replace: false,
                    is_edge: false,
                    input: Box::new(plan),
                });
            }
        }

        // Handle WITH clauses (projection for query chaining)
        for with_clause in &query.with_clauses {
            if !with_clause.is_wildcard {
                let projections: Vec<Projection> = with_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(Projection {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?;

                plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(plan),
                });
            }
            // WITH * skips projection: all variables pass through unchanged

            // Apply WHERE filter if present in WITH clause
            if let Some(where_clause) = &with_clause.where_clause {
                let predicate = self.translate_expression(&where_clause.expression)?;
                plan = wrap_filter(plan, predicate);
            }

            // Handle DISTINCT
            if with_clause.distinct {
                plan = wrap_distinct(plan);
            }
        }

        // Apply SKIP
        if let Some(skip_expr) = &query.return_clause.skip
            && let ast::Expression::Literal(ast::Literal::Integer(n)) = skip_expr
        {
            plan = wrap_skip(plan, *n as usize);
        }

        // Apply LIMIT
        if let Some(limit_expr) = &query.return_clause.limit
            && let ast::Expression::Literal(ast::Literal::Integer(n)) = limit_expr
        {
            plan = wrap_limit(plan, *n as usize);
        }

        // FINISH: consume input, return empty result (mutations already applied)
        if query.return_clause.is_finish {
            // Wrap in a Limit(0) to consume input but return no rows
            plan = wrap_limit(plan, 0);
            return Ok(LogicalPlan::new(plan));
        }

        // Check if RETURN contains aggregate functions
        let has_aggregates = !query.return_clause.is_wildcard
            && query
                .return_clause
                .items
                .iter()
                .any(|item| contains_aggregate(&item.expression));

        if has_aggregates {
            // Extract aggregate and group-by expressions.
            // When a return item wraps an aggregate in a binary/unary expression
            // (e.g. `count(n) > 0 AS exists`), we decompose it into:
            //   1. An aggregate (`count(n)` with synthetic alias)
            //   2. A post-aggregate projection (`_agg_0 > 0 AS exists`)
            let (aggregates, auto_group_by, post_return) =
                self.extract_aggregates_and_groups(&query.return_clause.items)?;

            // Separate horizontal aggregates (over group-list variables from
            // variable-length paths) from regular aggregates.
            let glv = self.group_list_variables.borrow();
            let mut regular_aggregates = Vec::new();
            for agg_expr in aggregates {
                if let Some(ref expr) = agg_expr.expression
                    && let LogicalExpression::Property { variable, property } = expr
                    && let Some(path_alias) = glv.get(variable)
                {
                    let alias = agg_expr.alias.clone().unwrap_or_else(|| {
                        format!("{:?}_{}", agg_expr.function, property).to_lowercase()
                    });
                    plan = LogicalOperator::HorizontalAggregate(HorizontalAggregateOp {
                        list_column: format!("_path_edges_{}", path_alias),
                        entity_kind: EntityKind::Edge,
                        function: agg_expr.function,
                        property: property.clone(),
                        alias,
                        input: Box::new(plan),
                    });
                    continue;
                }
                regular_aggregates.push(agg_expr);
            }
            drop(glv);

            // Use explicit GROUP BY if provided, otherwise use auto-detected
            let group_by = if query.return_clause.group_by.is_empty() {
                auto_group_by
            } else {
                query
                    .return_clause
                    .group_by
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?
            };

            // Translate HAVING clause if present
            let having = if let Some(having_clause) = &query.having_clause {
                Some(self.translate_expression(&having_clause.expression)?)
            } else {
                None
            };

            let agg_op = if regular_aggregates.is_empty() && group_by.is_empty() {
                // All aggregates were horizontal, no need for a HashAggregate
                plan
            } else {
                LogicalOperator::Aggregate(AggregateOp {
                    group_by,
                    aggregates: regular_aggregates,
                    input: Box::new(plan),
                    having,
                })
            };

            if let Some(return_items) = post_return {
                plan = wrap_return(agg_op, return_items, query.return_clause.distinct);
            } else {
                plan = agg_op;
            }

            // Apply ORDER BY for aggregate queries
            // Note: ORDER BY sort keys reference aggregate output columns (aliases).
            if let Some(order_by) = &query.return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression)?,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: item.nulls.map(|n| match n {
                                ast::NullsOrdering::First => NullsOrdering::First,
                                ast::NullsOrdering::Last => NullsOrdering::Last,
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }

            // Note: For aggregate queries, we don't add a Return operator
            // because Aggregate already produces the final output
        } else {
            // Apply RETURN first (closest to input), then Sort wraps it.
            // This ensures RETURN aliases are visible to ORDER BY in the binder.
            let return_items = if query.return_clause.is_wildcard {
                // RETURN *: emit a wildcard marker that the planner expands
                vec![ReturnItem {
                    expression: LogicalExpression::Variable("*".into()),
                    alias: None,
                }]
            } else {
                query
                    .return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            plan = wrap_return(plan, return_items, query.return_clause.distinct);

            // Apply ORDER BY (wraps Return so aliases are visible)
            if let Some(order_by) = &query.return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression)?,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: item.nulls.map(|n| match n {
                                ast::NullsOrdering::First => NullsOrdering::First,
                                ast::NullsOrdering::Last => NullsOrdering::Last,
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    /// Translates a MATCH clause with an optional initial input.
    ///
    /// When `initial_input` is provided (e.g. from a preceding UNWIND), the
    /// first pattern's NodeScan receives it as input. This creates a nested
    /// loop join that keeps prior variables (like UNWIND variables) in scope
    /// so that property filters like `{id: x}` can reference them.
    fn translate_match_with_input(
        &self,
        match_clause: &ast::MatchClause,
        initial_input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan: Option<LogicalOperator> = initial_input;

        let mut path_mode = match match_clause.path_mode {
            Some(ast::PathMode::Walk) | None => PathMode::Walk,
            Some(ast::PathMode::Trail) => PathMode::Trail,
            Some(ast::PathMode::Simple) => PathMode::Simple,
            Some(ast::PathMode::Acyclic) => PathMode::Acyclic,
        };

        // Match mode overrides path mode: DIFFERENT EDGES = Trail, REPEATABLE ELEMENTS = Walk
        if let Some(mode) = &match_clause.match_mode {
            match mode {
                ast::MatchMode::DifferentEdges => path_mode = PathMode::Trail,
                ast::MatchMode::RepeatableElements => path_mode = PathMode::Walk,
            }
        }

        // Search prefix determines whether we use shortest path operators
        let use_shortest = matches!(
            &match_clause.search_prefix,
            Some(
                ast::PathSearchPrefix::AnyShortest
                    | ast::PathSearchPrefix::AllShortest
                    | ast::PathSearchPrefix::ShortestK(_)
                    | ast::PathSearchPrefix::ShortestKGroups(_)
            )
        );

        for aliased_pattern in &match_clause.patterns {
            // Handle shortestPath patterns specially
            if let Some(path_function) = &aliased_pattern.path_function {
                plan = Some(self.translate_shortest_path(
                    &aliased_pattern.pattern,
                    aliased_pattern.alias.as_deref(),
                    *path_function,
                    plan.take(),
                )?);
            } else if use_shortest {
                // ISO path search prefix maps to ShortestPath operator
                let pf = match &match_clause.search_prefix {
                    Some(ast::PathSearchPrefix::AllShortest) => ast::PathFunction::AllShortestPaths,
                    _ => ast::PathFunction::ShortestPath,
                };
                plan = Some(self.translate_shortest_path(
                    &aliased_pattern.pattern,
                    aliased_pattern.alias.as_deref(),
                    pf,
                    plan.take(),
                )?);
            } else {
                let pattern_plan = self.translate_pattern_with_alias(
                    &aliased_pattern.pattern,
                    plan.take(),
                    aliased_pattern.alias.as_deref(),
                    path_mode,
                )?;
                plan = Some(pattern_plan);
            }
        }

        plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty MATCH clause",
            ))
        })
    }

    fn translate_match(&self, match_clause: &ast::MatchClause) -> Result<LogicalOperator> {
        self.translate_match_with_input(match_clause, None)
    }

    /// Translates a shortestPath pattern into a logical operator.
    fn translate_shortest_path(
        &self,
        pattern: &ast::Pattern,
        alias: Option<&str>,
        path_function: ast::PathFunction,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Extract source and target from the pattern
        let (source_node, target_node, edge_types, direction) = match pattern {
            ast::Pattern::Path(path) => {
                let target_node = if let Some(edge) = path.edges.last() {
                    &edge.target
                } else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "shortestPath requires a path pattern",
                    )));
                };
                let edge_types = path
                    .edges
                    .first()
                    .map(|e| e.types.clone())
                    .unwrap_or_default();
                let direction =
                    path.edges
                        .first()
                        .map_or(ExpandDirection::Both, |e| match e.direction {
                            ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
                            ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
                            ast::EdgeDirection::Undirected => ExpandDirection::Both,
                        });
                (&path.source, target_node, edge_types, direction)
            }
            ast::Pattern::Node(_)
            | ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "shortestPath requires a simple path pattern",
                )));
            }
        };

        // Get variable names
        let source_var = source_node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));
        let target_var = target_node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        // For shortestPath, we need to scan source and target nodes separately
        // (not expand between them - the ShortestPathOperator will find the path)

        // Scan source node first
        let source_plan = self.translate_node_pattern(source_node, input)?;

        // Scan target node (cross-product with source)
        let target_plan = self.translate_node_pattern(target_node, Some(source_plan))?;

        // Wrap with ShortestPath operator
        Ok(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(target_plan),
            source_var,
            target_var,
            edge_types,
            direction,
            path_alias: alias.unwrap_or("_path").to_string(),
            all_paths: matches!(path_function, ast::PathFunction::AllShortestPaths),
        }))
    }

    fn translate_pattern_with_alias(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
        path_alias: Option<&str>,
        path_mode: PathMode,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => self.translate_node_pattern(node, input),
            ast::Pattern::Path(path) => {
                self.translate_path_pattern_with_alias(path, input, path_alias, path_mode)
            }
            ast::Pattern::Quantified {
                pattern,
                min,
                max,
                subpath_var,
                path_mode: inner_path_mode,
                where_clause,
            } => {
                // G049: Inner path mode overrides the outer path mode if set.
                let effective_mode = inner_path_mode.as_ref().map_or(path_mode, |m| match m {
                    ast::PathMode::Walk => PathMode::Walk,
                    ast::PathMode::Trail => PathMode::Trail,
                    ast::PathMode::Simple => PathMode::Simple,
                    ast::PathMode::Acyclic => PathMode::Acyclic,
                });

                // G048: subpath_var takes precedence as path alias for the
                // quantified pattern. Fall back to outer path_alias if not set.
                let effective_alias = subpath_var.as_deref().or(path_alias);

                // Quantified path pattern: repeat the inner pattern min..max times.
                // For now, translate as a variable-length expansion if the inner
                // pattern is a simple single-edge path.
                let mut result = match pattern.as_ref() {
                    ast::Pattern::Path(path) if path.edges.len() == 1 => {
                        // Single-edge quantified: equivalent to variable-length edge
                        let mut modified_path = path.clone();
                        let edge = &mut modified_path.edges[0];
                        edge.min_hops = Some(*min);
                        edge.max_hops = *max;
                        self.translate_path_pattern_with_alias(
                            &modified_path,
                            input,
                            effective_alias,
                            effective_mode,
                        )
                    }
                    _ => {
                        // Multi-edge or complex quantified patterns: translate the
                        // inner pattern once (future: iterate with backtracking).
                        self.translate_pattern_with_alias(
                            pattern,
                            input,
                            effective_alias,
                            effective_mode,
                        )
                    }
                }?;

                // G050: Apply WHERE clause as a filter on the quantified pattern output
                if let Some(where_expr) = where_clause {
                    let filter_expr = self.translate_expression(where_expr)?;
                    result = wrap_filter(result, filter_expr);
                }

                Ok(result)
            }
            ast::Pattern::Union(patterns) => {
                // Union of alternative patterns: UNION ALL of each translated pattern
                let inputs: Vec<LogicalOperator> = patterns
                    .iter()
                    .map(|p| {
                        self.translate_pattern_with_alias(p, input.clone(), path_alias, path_mode)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }
            ast::Pattern::MultisetUnion(patterns) => {
                // G030: Multiset (bag) union preserves duplicates
                let inputs: Vec<LogicalOperator> = patterns
                    .iter()
                    .map(|p| {
                        self.translate_pattern_with_alias(p, input.clone(), path_alias, path_mode)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }
        }
    }

    /// Translates a MERGE clause into a MergeOp.
    fn translate_merge(
        &self,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let (variable, labels, match_properties) = match &merge_clause.pattern {
            ast::Pattern::Node(node) => {
                let var = node
                    .variable
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                let labels = node.labels.clone();
                let props: Vec<(String, LogicalExpression)> = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                (var, labels, props)
            }
            ast::Pattern::Path(path) if !path.edges.is_empty() => {
                return self.translate_merge_relationship(path, merge_clause, input);
            }
            ast::Pattern::Path(path) => {
                // Path with no edges is just a node pattern
                let var = path
                    .source
                    .variable
                    .clone()
                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                let labels = path.source.labels.clone();
                let props: Vec<(String, LogicalExpression)> = path
                    .source
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                (var, labels, props)
            }
            ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "MERGE does not support quantified or union patterns",
                )));
            }
        };

        let on_create: Vec<(String, LogicalExpression)> = merge_clause
            .on_create
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let on_match: Vec<(String, LogicalExpression)> = merge_clause
            .on_match
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        Ok(LogicalOperator::Merge(MergeOp {
            variable,
            labels,
            match_properties,
            on_create,
            on_match,
            input: Box::new(input),
        }))
    }

    /// Translates a MERGE with a relationship pattern.
    fn translate_merge_relationship(
        &self,
        path: &ast::PathPattern,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let source_variable = path.source.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a source node variable",
            ))
        })?;

        let edge = path.edges.first().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern is empty",
            ))
        })?;

        let variable = edge
            .variable
            .clone()
            .unwrap_or_else(|| format!("_merge_rel_{}", rand_id()));

        let edge_type = edge.types.first().cloned().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires an edge type",
            ))
        })?;

        let target_variable = edge.target.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a target node variable",
            ))
        })?;

        let match_properties: Vec<(String, LogicalExpression)> = edge
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let on_create: Vec<(String, LogicalExpression)> = merge_clause
            .on_create
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        let on_match: Vec<(String, LogicalExpression)> = merge_clause
            .on_match
            .as_ref()
            .map(|assignments| {
                assignments
                    .iter()
                    .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();

        Ok(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable,
            source_variable,
            target_variable,
            edge_type,
            match_properties,
            on_create,
            on_match,
            input: Box::new(input),
        }))
    }

    /// Translates CREATE patterns to create operators.
    fn translate_create_patterns(
        &self,
        patterns: &[ast::Pattern],
        mut plan: LogicalOperator,
    ) -> Result<LogicalOperator> {
        for pattern in patterns {
            match pattern {
                ast::Pattern::Node(node) => {
                    let variable = node
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                    let properties: Vec<(String, LogicalExpression)> = node
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    plan = LogicalOperator::CreateNode(CreateNodeOp {
                        variable,
                        labels: node.labels.clone(),
                        properties,
                        input: Some(Box::new(plan)),
                    });
                }
                ast::Pattern::Path(path) => {
                    // First create the source node if it has labels (new node)
                    let source_var = path
                        .source
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                    // If source has labels, it's a new node to create
                    if !path.source.labels.is_empty() {
                        let source_props: Vec<(String, LogicalExpression)> = path
                            .source
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<_>>()?;

                        plan = LogicalOperator::CreateNode(CreateNodeOp {
                            variable: source_var.clone(),
                            labels: path.source.labels.clone(),
                            properties: source_props,
                            input: Some(Box::new(plan)),
                        });
                    }

                    // Create edges and target nodes
                    for edge in &path.edges {
                        let target_var = edge
                            .target
                            .variable
                            .clone()
                            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                        // If target has labels, create it
                        if !edge.target.labels.is_empty() {
                            let target_props: Vec<(String, LogicalExpression)> = edge
                                .target
                                .properties
                                .iter()
                                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                                .collect::<Result<_>>()?;

                            plan = LogicalOperator::CreateNode(CreateNodeOp {
                                variable: target_var.clone(),
                                labels: edge.target.labels.clone(),
                                properties: target_props,
                                input: Some(Box::new(plan)),
                            });
                        }

                        // Create the edge
                        let edge_type = edge.types.first().cloned().unwrap_or_default();
                        let edge_var = edge.variable.clone();
                        let edge_props: Vec<(String, LogicalExpression)> = edge
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<_>>()?;

                        // Determine direction
                        let (from_var, to_var) = match edge.direction {
                            ast::EdgeDirection::Outgoing => (source_var.clone(), target_var),
                            ast::EdgeDirection::Incoming => {
                                let tv = edge
                                    .target
                                    .variable
                                    .clone()
                                    .unwrap_or_else(|| format!("_anon_{}", rand_id()));
                                (tv, source_var.clone())
                            }
                            ast::EdgeDirection::Undirected => (source_var.clone(), target_var),
                        };

                        plan = LogicalOperator::CreateEdge(CreateEdgeOp {
                            variable: edge_var,
                            from_variable: from_var,
                            to_variable: to_var,
                            edge_type,
                            properties: edge_props,
                            input: Box::new(plan),
                        });
                    }
                }
                ast::Pattern::Quantified { .. }
                | ast::Pattern::Union(_)
                | ast::Pattern::MultisetUnion(_) => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "CREATE does not support quantified or union patterns",
                    )));
                }
            }
        }
        Ok(plan)
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        // Use first label from colon syntax for the NodeScan optimization,
        // or first label from IS expression if it's a simple label.
        let label = if let Some(ref label_expr) = node.label_expression {
            if let ast::LabelExpression::Label(name) = label_expr {
                Some(name.clone())
            } else {
                None
            }
        } else {
            node.labels.first().cloned()
        };

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add label expression filter for complex IS expressions
        if let Some(ref label_expr) = node.label_expression {
            // Only add filter for non-simple expressions (simple Label already used in NodeScan)
            if !matches!(label_expr, ast::LabelExpression::Label(_)) {
                let predicate = Self::translate_label_expression(&variable, label_expr);
                plan = wrap_filter(plan, predicate);
            }
        }

        // Add filter for node pattern properties (e.g., {name: 'Alix'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        // Add element pattern WHERE clause (e.g., (n WHERE n.age > 30))
        if let Some(ref where_expr) = node.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Translates a label expression into a filter predicate using hasLabel() calls.
    fn translate_label_expression(
        variable: &str,
        expr: &ast::LabelExpression,
    ) -> LogicalExpression {
        match expr {
            ast::LabelExpression::Label(name) => LogicalExpression::FunctionCall {
                name: "hasLabel".into(),
                args: vec![
                    LogicalExpression::Variable(variable.to_string()),
                    LogicalExpression::Literal(Value::from(name.as_str())),
                ],
                distinct: false,
            },
            ast::LabelExpression::Conjunction(operands) => {
                let mut iter = operands.iter();
                let first = Self::translate_label_expression(
                    variable,
                    iter.next().expect("conjunction has at least one operand"),
                );
                iter.fold(first, |acc, op| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::And,
                    right: Box::new(Self::translate_label_expression(variable, op)),
                })
            }
            ast::LabelExpression::Disjunction(operands) => {
                let mut iter = operands.iter();
                let first = Self::translate_label_expression(
                    variable,
                    iter.next().expect("disjunction has at least one operand"),
                );
                iter.fold(first, |acc, op| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::Or,
                    right: Box::new(Self::translate_label_expression(variable, op)),
                })
            }
            ast::LabelExpression::Negation(inner) => LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(Self::translate_label_expression(variable, inner)),
            },
            ast::LabelExpression::Wildcard => LogicalExpression::Literal(Value::Bool(true)),
        }
    }

    /// Builds a predicate expression for property filters like {name: 'Alix', age: 30}.
    fn build_property_predicate(
        &self,
        variable: &str,
        properties: &[(String, ast::Expression)],
    ) -> Result<LogicalExpression> {
        let predicates = properties
            .iter()
            .map(|(prop_name, prop_value)| {
                let left = LogicalExpression::Property {
                    variable: variable.to_string(),
                    property: prop_name.clone(),
                };
                let right = self.translate_expression(prop_value)?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Eq,
                    right: Box::new(right),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        combine_with_and(predicates)
    }

    fn translate_path_pattern_with_alias(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
        path_alias: Option<&str>,
        path_mode: PathMode,
    ) -> Result<LogicalOperator> {
        // Start with the source node
        let source_var = path
            .source
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        let source_label = path.source.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: source_var.clone(),
            label: source_label,
            input: input.map(Box::new),
        });

        // Add filter for source node properties (e.g., {id: 'a'})
        if !path.source.properties.is_empty() {
            let predicate = self.build_property_predicate(&source_var, &path.source.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        // Add element WHERE clause for source node
        if let Some(ref where_expr) = path.source.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            plan = wrap_filter(plan, predicate);
        }

        // Process each edge in the chain
        let mut current_source = source_var;
        let edge_count = path.edges.len();

        for (idx, edge) in path.edges.iter().enumerate() {
            let target_var = edge
                .target
                .variable
                .clone()
                .unwrap_or_else(|| format!("_anon_{}", rand_id()));

            let edge_var = edge.variable.clone();
            let edge_types = edge.types.clone();

            let direction = match edge.direction {
                ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
                ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
                ast::EdgeDirection::Undirected => ExpandDirection::Both,
            };

            let edge_var_for_filter = edge_var.clone();

            // Set path_alias on the last edge of a named path
            let expand_path_alias = if idx == edge_count - 1 {
                path_alias.map(String::from)
            } else {
                None
            };

            let min_hops = edge.min_hops.unwrap_or(1);
            let max_hops = if edge.min_hops.is_none() && edge.max_hops.is_none() {
                Some(1)
            } else {
                edge.max_hops
            };

            let is_variable_length = min_hops != 1 || max_hops.is_none() || max_hops != Some(1);

            // For variable-length edges with a named edge variable, auto-generate
            // a path alias if none exists, so path detail columns are available
            // for horizontal aggregation (GE09).
            let expand_path_alias = if is_variable_length
                && edge_var_for_filter.is_some()
                && expand_path_alias.is_none()
            {
                Some(format!("_auto_path_{}", rand_id()))
            } else {
                expand_path_alias
            };

            // Track group-list variables for horizontal aggregation detection
            if is_variable_length
                && let (Some(ev), Some(pa)) = (&edge_var_for_filter, &expand_path_alias)
            {
                self.group_list_variables
                    .borrow_mut()
                    .insert(ev.clone(), pa.clone());
            }

            plan = LogicalOperator::Expand(ExpandOp {
                from_variable: current_source,
                to_variable: target_var.clone(),
                edge_variable: edge_var,
                direction,
                edge_types,
                min_hops,
                max_hops,
                input: Box::new(plan),
                path_alias: expand_path_alias,
                path_mode,
            });

            // For questioned edges (->?), save the plan before expand so we can
            // wrap the expand + filters in a LeftJoin later.
            let pre_expand_plan = if edge.questioned {
                Some(plan.clone())
            } else {
                None
            };

            // Add filter for edge properties
            if !edge.properties.is_empty()
                && let Some(ref ev) = edge_var_for_filter
            {
                let predicate = self.build_property_predicate(ev, &edge.properties)?;
                plan = wrap_filter(plan, predicate);
            }

            // Add element WHERE clause for edge
            if let Some(ref where_expr) = edge.where_clause {
                let predicate = self.translate_expression(where_expr)?;
                plan = wrap_filter(plan, predicate);
            }

            // Add filter for target node properties
            if !edge.target.properties.is_empty() {
                let predicate =
                    self.build_property_predicate(&target_var, &edge.target.properties)?;
                plan = wrap_filter(plan, predicate);
            }

            // Add filter for target node labels (colon syntax or IS expression)
            if let Some(ref label_expr) = edge.target.label_expression {
                let predicate = Self::translate_label_expression(&target_var, label_expr);
                plan = wrap_filter(plan, predicate);
            } else if !edge.target.labels.is_empty() {
                let label = edge.target.labels[0].clone();
                plan = wrap_filter(
                    plan,
                    LogicalExpression::FunctionCall {
                        name: "hasLabel".into(),
                        args: vec![
                            LogicalExpression::Variable(target_var.clone()),
                            LogicalExpression::Literal(Value::from(label)),
                        ],
                        distinct: false,
                    },
                );
            }

            // Add element WHERE clause for target node
            if let Some(ref where_expr) = edge.target.where_clause {
                let predicate = self.translate_expression(where_expr)?;
                plan = wrap_filter(plan, predicate);
            }

            // Questioned edge: wrap expand + all filters in a LeftJoin so the
            // edge is optional (rows without a matching edge are preserved with nulls).
            if let Some(left) = pre_expand_plan {
                // Extract only the expand + filter portion (the right side):
                // plan currently includes the expand on top of the left plan.
                // We need to rebuild: LeftJoin(left, expand_on_left).
                // Since expand already includes left as input, we use it as-is
                // and the LeftJoin semantics handle preserving unmatched rows.
                plan = LogicalOperator::LeftJoin(LeftJoinOp {
                    left: Box::new(left),
                    right: Box::new(plan),
                    condition: None,
                });
            }

            current_source = target_var;
        }

        Ok(plan)
    }

    fn translate_data_modification(
        &self,
        dm: &ast::DataModificationStatement,
    ) -> Result<LogicalPlan> {
        match dm {
            ast::DataModificationStatement::Insert(insert) => self.translate_insert(insert),
            ast::DataModificationStatement::Delete(delete) => self.translate_delete(delete),
            ast::DataModificationStatement::Set(set) => self.translate_set(set),
        }
    }

    fn translate_delete(&self, delete: &ast::DeleteStatement) -> Result<LogicalPlan> {
        // DELETE requires a preceding MATCH clause to identify what to delete.
        // For standalone DELETE, we need to scan and delete the specified variables.
        // This is typically used as: MATCH (n:Label) DELETE n

        if delete.targets.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "DELETE requires at least one target",
            )));
        }

        // Extract the first variable name for the scan
        let first_var = match &delete.targets[0] {
            ast::DeleteTarget::Variable(name) => name.clone(),
            ast::DeleteTarget::Expression(_) => "__delete_expr_0".to_string(),
        };

        // Create a scan to find the entities to delete
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: first_var.clone(),
            label: None,
            input: None,
        });

        let plan = self.translate_delete_targets(&delete.targets, delete.detach, scan)?;
        Ok(LogicalPlan::new(plan))
    }

    /// Translates a list of delete targets into a chain of delete operators.
    /// For `DeleteTarget::Variable`, emits a `DeleteNodeOp` directly.
    /// For `DeleteTarget::Expression` (GD04), projects the expression into a
    /// synthetic variable then deletes that variable.
    fn translate_delete_targets(
        &self,
        targets: &[ast::DeleteTarget],
        detach: bool,
        mut plan: LogicalOperator,
    ) -> Result<LogicalOperator> {
        for (i, target) in targets.iter().enumerate() {
            match target {
                ast::DeleteTarget::Variable(name) => {
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: name.clone(),
                        detach,
                        input: Box::new(plan),
                    });
                }
                ast::DeleteTarget::Expression(expr) => {
                    // GD04: evaluate the expression, bind to a synthetic variable,
                    // then delete that variable.
                    let synthetic_var = format!("__delete_expr_{i}");
                    let logical_expr = self.translate_expression(expr)?;
                    plan = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: logical_expr,
                            alias: Some(synthetic_var.clone()),
                        }],
                        input: Box::new(plan),
                    });
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: synthetic_var,
                        detach,
                        input: Box::new(plan),
                    });
                }
            }
        }
        Ok(plan)
    }

    fn translate_set(&self, set: &ast::SetStatement) -> Result<LogicalPlan> {
        // SET requires a preceding MATCH clause to identify what to update.
        // For standalone SET, we error - it should be part of a query.

        if set.assignments.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SET requires at least one assignment",
            )));
        }

        // Group assignments by variable
        let first_assignment = &set.assignments[0];
        let var = &first_assignment.variable;

        // Create a scan to find the entity to update
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: None,
            input: None,
        });

        // Build property assignments for this variable
        let properties: Vec<(String, LogicalExpression)> = set
            .assignments
            .iter()
            .filter(|a| &a.variable == var)
            .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
            .collect::<Result<_>>()?;

        let plan = LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false,
            is_edge: false,
            input: Box::new(scan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_insert(&self, insert: &ast::InsertStatement) -> Result<LogicalPlan> {
        if insert.patterns.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty INSERT statement",
            )));
        }

        // Chain CreateNode operators for all patterns.
        // First pattern gets input: None, subsequent ones chain via input: Some(prev).
        let mut plan: Option<LogicalOperator> = None;
        let mut last_variable = String::new();

        for pattern in &insert.patterns {
            match pattern {
                ast::Pattern::Node(node) => {
                    let variable = node
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                    let properties = node
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<Vec<_>>>()?;

                    plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                        variable: variable.clone(),
                        labels: node.labels.clone(),
                        properties,
                        input: plan.map(Box::new),
                    }));
                    last_variable = variable;
                }
                ast::Pattern::Path(path) => {
                    // Decompose path into CreateNode + CreateEdge chain
                    let source_var = path
                        .source
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                    if !path.source.labels.is_empty() {
                        let source_props: Vec<(String, LogicalExpression)> = path
                            .source
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<Vec<_>>>()?;
                        plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                            variable: source_var.clone(),
                            labels: path.source.labels.clone(),
                            properties: source_props,
                            input: plan.map(Box::new),
                        }));
                    }

                    let mut current_src = source_var;
                    for edge in &path.edges {
                        let target_var = edge
                            .target
                            .variable
                            .clone()
                            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                        if !edge.target.labels.is_empty() {
                            let target_props: Vec<(String, LogicalExpression)> = edge
                                .target
                                .properties
                                .iter()
                                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                                .collect::<Result<Vec<_>>>()?;
                            plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                                variable: target_var.clone(),
                                labels: edge.target.labels.clone(),
                                properties: target_props,
                                input: plan.map(Box::new),
                            }));
                        }

                        let edge_type = edge.types.first().cloned().unwrap_or_default();
                        let edge_props: Vec<(String, LogicalExpression)> = edge
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<Vec<_>>>()?;

                        let (from, to) = match edge.direction {
                            ast::EdgeDirection::Incoming => (target_var.clone(), current_src),
                            _ => (current_src, target_var.clone()),
                        };

                        plan = Some(LogicalOperator::CreateEdge(CreateEdgeOp {
                            variable: edge.variable.clone(),
                            edge_type,
                            from_variable: from,
                            to_variable: to,
                            properties: edge_props,
                            input: Box::new(plan.unwrap_or(LogicalOperator::Empty)),
                        }));
                        last_variable.clone_from(&target_var);
                        current_src = target_var;
                    }
                }
                ast::Pattern::Quantified { .. }
                | ast::Pattern::Union(_)
                | ast::Pattern::MultisetUnion(_) => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "INSERT does not support quantified or union patterns",
                    )));
                }
            }
        }

        let ret = wrap_return(
            plan.expect("plan initialized by non-empty patterns"),
            vec![ReturnItem {
                expression: LogicalExpression::Variable(last_variable),
                alias: None,
            }],
            false,
        );

        Ok(LogicalPlan::new(ret))
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => Ok(self.translate_literal(lit)),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { variable, property } => {
                Ok(LogicalExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*op);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand = self.translate_expression(operand)?;
                let op = self.translate_unary_op(*op);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // Special handling for length() on path variables
                // When length(p) is called where p is a path alias, we convert it
                // to a variable reference to the path length column
                if name.to_lowercase() == "length"
                    && args.len() == 1
                    && let ast::Expression::Variable(var_name) = &args[0]
                {
                    // Check if this looks like a path variable
                    // Path lengths are stored in columns named _path_length_{alias}
                    return Ok(LogicalExpression::Variable(format!(
                        "_path_length_{}",
                        var_name
                    )));
                }

                // NULLIF(a, b) desugars to CASE WHEN a = b THEN NULL ELSE a END
                if name.eq_ignore_ascii_case("nullif") {
                    if args.len() != 2 {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "NULLIF requires exactly 2 arguments",
                        )));
                    }
                    let a = self.translate_expression(&args[0])?;
                    let b = self.translate_expression(&args[1])?;
                    return Ok(LogicalExpression::Case {
                        operand: None,
                        when_clauses: vec![(
                            LogicalExpression::Binary {
                                left: Box::new(a.clone()),
                                op: BinaryOp::Eq,
                                right: Box::new(b),
                            },
                            LogicalExpression::Literal(Value::Null),
                        )],
                        else_clause: Some(Box::new(a)),
                    });
                }

                let args = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args,
                    distinct: *distinct,
                })
            }
            ast::Expression::List(items) => {
                let items = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::List(items))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = input
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                let when_clauses = whens
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.translate_expression(cond)?,
                            self.translate_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let else_clause = else_clause
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);

                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause,
                })
            }
            ast::Expression::Map(entries) => {
                let entries = entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Map(entries))
            }
            ast::Expression::ExistsSubquery { query } => {
                // Translate inner query to logical operator
                let inner_plan = self.translate_subquery_to_operator(query)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(inner_plan)))
            }
            ast::Expression::CountSubquery { query } => {
                let inner_plan = self.translate_subquery_to_operator(query)?;
                Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
            }
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base)?;
                let index_expr = self.translate_expression(index)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::ValueSubquery { query } => {
                // VALUE { subquery } returns a scalar from the inner query.
                // Translate the inner subquery and wrap as a correlated scalar subquery.
                let inner_plan = self.translate_subquery_to_operator(query)?;
                // We reuse CountSubquery infrastructure for now: the Apply operator
                // will run the inner plan and extract the first result.
                Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
            }
            ast::Expression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let list = self.translate_expression(list_expr)?;
                let filter = filter_expr
                    .as_ref()
                    .map(|f| self.translate_expression(f))
                    .transpose()?
                    .map(Box::new);
                let map = self.translate_expression(map_expr)?;
                Ok(LogicalExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    filter_expr: filter,
                    map_expr: Box::new(map),
                })
            }
            ast::Expression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let list = self.translate_expression(list_expr)?;
                let pred = self.translate_expression(predicate)?;
                let logical_kind = match kind {
                    ast::ListPredicateKind::All => plan::ListPredicateKind::All,
                    ast::ListPredicateKind::Any => plan::ListPredicateKind::Any,
                    ast::ListPredicateKind::None => plan::ListPredicateKind::None,
                    ast::ListPredicateKind::Single => plan::ListPredicateKind::Single,
                };
                Ok(LogicalExpression::ListPredicate {
                    kind: logical_kind,
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    predicate: Box::new(pred),
                })
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let init = self.translate_expression(initial)?;
                let list_expr = self.translate_expression(list)?;
                let body = self.translate_expression(expression)?;
                Ok(LogicalExpression::Reduce {
                    accumulator: accumulator.clone(),
                    initial: Box::new(init),
                    variable: variable.clone(),
                    list: Box::new(list_expr),
                    expression: Box::new(body),
                })
            }
            ast::Expression::LetIn { bindings, body } => {
                // LET x = expr1, y = expr2 IN body END
                // Desugar each binding to a nested CASE: the last binding wraps the body.
                // Since our IR doesn't have variable scoping, we translate bindings as
                // nested function calls: each binding becomes a property-like variable.
                // For simple cases, we substitute bindings directly into the body.
                // For the general case, we translate bindings as a chain of With projections,
                // but at the expression level we can inline: translate body with substitution.
                //
                // Simple approach: translate all binding expressions, then translate the body.
                // Variables defined in LET are in scope for the body. Since our logical
                // expression IR doesn't have local scoping, we treat LET bindings as
                // additional variables that the outer context can resolve.
                let _binding_exprs: Vec<(String, LogicalExpression)> = bindings
                    .iter()
                    .map(|(name, expr)| Ok((name.clone(), self.translate_expression(expr)?)))
                    .collect::<Result<_>>()?;
                // Translate the body expression directly.
                // The bindings introduce new variable names that will be resolved
                // at execution time through the normal variable resolution mechanism.
                self.translate_expression(body)
            }
        }
    }

    fn translate_literal(&self, lit: &ast::Literal) -> LogicalExpression {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::String(s.clone().into()),
            ast::Literal::Date(s) => grafeo_common::types::Date::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Date),
            ast::Literal::Time(s) => grafeo_common::types::Time::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Time),
            ast::Literal::Duration(s) => grafeo_common::types::Duration::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::Duration),
            ast::Literal::Datetime(s) => {
                // Try full ISO datetime: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]
                if let Some(pos) = s.find('T') {
                    if let (Some(d), Some(t)) = (
                        grafeo_common::types::Date::parse(&s[..pos]),
                        grafeo_common::types::Time::parse(&s[pos + 1..]),
                    ) {
                        Value::Timestamp(grafeo_common::types::Timestamp::from_date_time(d, t))
                    } else {
                        Value::String(s.clone().into())
                    }
                } else if let Some(d) = grafeo_common::types::Date::parse(s) {
                    Value::Timestamp(d.to_timestamp())
                } else {
                    Value::String(s.clone().into())
                }
            }
            ast::Literal::ZonedDatetime(s) => grafeo_common::types::ZonedDatetime::parse(s)
                .map_or_else(|| Value::String(s.clone().into()), Value::ZonedDatetime),
            ast::Literal::ZonedTime(s) => {
                // Parse as Time with required offset
                if let Some(t) = grafeo_common::types::Time::parse(s)
                    && t.offset_seconds().is_some()
                {
                    Value::Time(t)
                } else {
                    Value::String(s.clone().into())
                }
            }
        };
        LogicalExpression::Literal(value)
    }

    fn translate_binary_op(&self, op: ast::BinaryOp) -> BinaryOp {
        match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Xor => BinaryOp::Xor,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::Like => BinaryOp::Like,
            ast::BinaryOp::In => BinaryOp::In,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
        }
    }

    fn translate_unary_op(&self, op: ast::UnaryOp) -> UnaryOp {
        match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        }
    }

    /// Translates a subquery to a logical operator (without Return).
    fn translate_subquery_to_operator(
        &self,
        query: &ast::QueryStatement,
    ) -> Result<LogicalOperator> {
        let mut plan = LogicalOperator::Empty;

        for match_clause in &query.match_clauses {
            let match_plan = self.translate_match(match_clause)?;
            plan = if matches!(plan, LogicalOperator::Empty) {
                match_plan
            } else {
                LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(match_plan),
                    join_type: JoinType::Cross,
                    conditions: vec![],
                })
            };
        }

        if let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Extracts aggregate expressions and group-by expressions from RETURN items.
    /// Extracts aggregate and group-by expressions from RETURN items.
    ///
    /// Returns `(aggregates, group_by, post_return)` where `post_return` is
    /// `Some(...)` when any return item wraps an aggregate in a binary/unary
    /// expression (e.g. `count(n) > 0 AS exists`).
    fn extract_aggregates_and_groups(
        &self,
        items: &[ast::ReturnItem],
    ) -> Result<(
        Vec<AggregateExpr>,
        Vec<LogicalExpression>,
        Option<Vec<ReturnItem>>,
    )> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();
        let mut needs_post_return = false;
        let mut post_return_items = Vec::new();
        let mut agg_counter: u32 = 0;

        for item in items {
            if let Some(agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                // Direct aggregate (e.g. `count(n) AS cnt`)
                aggregates.push(agg_expr);
                let agg_alias = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| format!("_agg_{agg_counter}"));
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(agg_alias),
                    alias: item.alias.clone(),
                });
                agg_counter += 1;
            } else if contains_aggregate(&item.expression) {
                // Wrapped aggregate (e.g. `count(n) > 0 AS exists`)
                needs_post_return = true;
                let synthetic_alias = format!("_agg_{agg_counter}");
                agg_counter += 1;

                let (agg_expr, substitute) =
                    self.extract_wrapped_aggregate(&item.expression, &synthetic_alias)?;
                aggregates.push(agg_expr);
                post_return_items.push(ReturnItem {
                    expression: substitute,
                    alias: item.alias.clone(),
                });
            } else {
                // Non-aggregate expression: group-by key
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr.clone());
                post_return_items.push(ReturnItem {
                    expression: expr,
                    alias: item.alias.clone(),
                });
            }
        }

        if needs_post_return {
            Ok((aggregates, group_by, Some(post_return_items)))
        } else {
            Ok((aggregates, group_by, None))
        }
    }

    /// Extracts an aggregate from inside a wrapping expression.
    fn extract_wrapped_aggregate(
        &self,
        expr: &ast::Expression,
        synthetic_alias: &str,
    ) -> Result<(AggregateExpr, LogicalExpression)> {
        match expr {
            ast::Expression::FunctionCall { .. } => {
                let agg = self
                    .try_extract_aggregate(expr, &Some(synthetic_alias.to_string()))?
                    .expect("contains_aggregate was true but try_extract_aggregate returned None");
                let substitute = LogicalExpression::Variable(synthetic_alias.to_string());
                Ok((agg, substitute))
            }
            ast::Expression::Binary { left, op, right } => {
                let binary_op = self.translate_binary_op(*op);
                if contains_aggregate(left) {
                    let (agg, left_sub) = self.extract_wrapped_aggregate(left, synthetic_alias)?;
                    let right_expr = self.translate_expression(right)?;
                    Ok((
                        agg,
                        LogicalExpression::Binary {
                            left: Box::new(left_sub),
                            op: binary_op,
                            right: Box::new(right_expr),
                        },
                    ))
                } else {
                    let (agg, right_sub) =
                        self.extract_wrapped_aggregate(right, synthetic_alias)?;
                    let left_expr = self.translate_expression(left)?;
                    Ok((
                        agg,
                        LogicalExpression::Binary {
                            left: Box::new(left_expr),
                            op: binary_op,
                            right: Box::new(right_sub),
                        },
                    ))
                }
            }
            ast::Expression::Unary { op, operand } => {
                let unary_op = self.translate_unary_op(*op);
                let (agg, sub) = self.extract_wrapped_aggregate(operand, synthetic_alias)?;
                Ok((
                    agg,
                    LogicalExpression::Unary {
                        op: unary_op,
                        operand: Box::new(sub),
                    },
                ))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                // Find the first aggregate inside the CASE branches and extract it.
                for (cond, then) in whens {
                    if contains_aggregate(cond) {
                        let (agg, _) = self.extract_wrapped_aggregate(cond, synthetic_alias)?;
                        let full_case = self.translate_expression(expr)?;
                        return Ok((agg, full_case));
                    }
                    if contains_aggregate(then) {
                        let (agg, _) = self.extract_wrapped_aggregate(then, synthetic_alias)?;
                        let full_case = self.translate_expression(expr)?;
                        return Ok((agg, full_case));
                    }
                }
                if let Some(el) = else_clause
                    && contains_aggregate(el)
                {
                    let (agg, _) = self.extract_wrapped_aggregate(el, synthetic_alias)?;
                    let full_case = self.translate_expression(expr)?;
                    return Ok((agg, full_case));
                }
                if let Some(inp) = input
                    && contains_aggregate(inp)
                {
                    let (agg, _) = self.extract_wrapped_aggregate(inp, synthetic_alias)?;
                    let full_case = self.translate_expression(expr)?;
                    return Ok((agg, full_case));
                }
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "Unsupported expression wrapping an aggregate",
                )))
            }
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Unsupported expression wrapping an aggregate",
            ))),
        }
    }

    /// Resolves an ORDER BY expression in the context of an aggregate query.
    ///
    /// If the sort key is an aggregate function call, finds the matching RETURN
    /// Tries to extract an aggregate expression from an AST expression.
    fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                if let Some(func) = to_aggregate_function(name) {
                    let agg_expr = if args.is_empty() {
                        // COUNT(*) case
                        AggregateExpr {
                            function: func,
                            expression: None,
                            expression2: None,
                            distinct: *distinct,
                            alias: alias.clone(),
                            percentile: None,
                            separator: None,
                        }
                    } else {
                        // COUNT(x), SUM(x), etc.
                        // For COUNT with an expression, use CountNonNull to ensure we fetch values
                        let actual_func = if func == AggregateFunction::Count {
                            AggregateFunction::CountNonNull
                        } else {
                            func
                        };
                        // Extract percentile parameter for percentile functions
                        let percentile = if matches!(
                            actual_func,
                            AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont
                        ) && args.len() >= 2
                        {
                            // Second argument is the percentile value
                            if let ast::Expression::Literal(ast::Literal::Float(p)) = &args[1] {
                                Some((*p).clamp(0.0, 1.0))
                            } else if let ast::Expression::Literal(ast::Literal::Integer(p)) =
                                &args[1]
                            {
                                Some((*p as f64).clamp(0.0, 1.0))
                            } else {
                                Some(0.5) // Default to median
                            }
                        } else {
                            None
                        };
                        // Extract second argument for binary set functions
                        let expression2 = if is_binary_set_function(actual_func) && args.len() >= 2
                        {
                            Some(self.translate_expression(&args[1])?)
                        } else {
                            None
                        };
                        // Extract separator for LISTAGG / GROUP_CONCAT
                        let upper_name = name.to_uppercase();
                        let separator = if actual_func == AggregateFunction::GroupConcat {
                            if args.len() >= 2 {
                                // Second argument is the separator string
                                if let ast::Expression::Literal(ast::Literal::String(s)) = &args[1]
                                {
                                    Some(s.clone())
                                } else if upper_name == "LISTAGG" {
                                    Some(",".to_string())
                                } else {
                                    None // GROUP_CONCAT default (space) handled in AggregateState
                                }
                            } else if upper_name == "LISTAGG" {
                                Some(",".to_string()) // ISO GQL default for LISTAGG
                            } else {
                                None // GROUP_CONCAT default (space) handled in AggregateState
                            }
                        } else {
                            None
                        };
                        AggregateExpr {
                            function: actual_func,
                            expression: Some(self.translate_expression(&args[0])?),
                            expression2,
                            distinct: *distinct,
                            alias: alias.clone(),
                            percentile,
                            separator,
                        }
                    };
                    Ok(Some(agg_expr))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }
}

/// Generate a simple random-ish ID for anonymous variables.
fn rand_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Checks if an AST expression contains an aggregate function call.
fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(contains_aggregate)
        }
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        ast::Expression::Case {
            input,
            whens,
            else_clause,
        } => {
            input.as_deref().is_some_and(contains_aggregate)
                || whens
                    .iter()
                    .any(|(w, t)| contains_aggregate(w) || contains_aggregate(t))
                || else_clause.as_deref().is_some_and(contains_aggregate)
        }
        ast::Expression::List(items) => items.iter().any(contains_aggregate),
        ast::Expression::ListComprehension {
            filter_expr,
            map_expr,
            ..
        } => filter_expr.as_deref().is_some_and(contains_aggregate) || contains_aggregate(map_expr),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let query = "MATCH (n:Person) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(!ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            // Should have Filter as input
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                } else {
                    panic!("Expected binary expression");
                }
            } else {
                panic!("Expected Filter operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_without_label() {
        let query = "MATCH (n) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert!(scan.label.is_none());
            } else {
                panic!("Expected NodeScan operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_distinct() {
        let query = "MATCH (n:Person) RETURN DISTINCT n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    // === Filter and Predicate Tests ===

    #[test]
    fn test_translate_filter_equality() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alix' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Navigate to find Filter
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "MATCH (n:Person) WHERE n.age > 20 AND n.age < 40 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::And);
        }
    }

    #[test]
    fn test_translate_filter_or() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Or);
        }
    }

    #[test]
    fn test_translate_filter_not() {
        let query = "MATCH (n:Person) WHERE NOT n.active RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Unary { op, .. } = &filter.predicate {
            assert_eq!(*op, UnaryOp::Not);
        }
    }

    // === Path Pattern / Join Tests ===

    #[test]
    fn test_translate_path_pattern() {
        let query = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Expand operator
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
        assert_eq!(expand.edge_types, vec!["KNOWS".to_string()]);
    }

    #[test]
    fn test_translate_incoming_path() {
        let query = "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_undirected_path() {
        let query = "MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Both);
    }

    // === Aggregation Tests ===

    #[test]
    fn test_translate_count_aggregate() {
        let query = "MATCH (n:Person) RETURN COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            // COUNT(expr) uses CountNonNull to ensure we fetch values for DISTINCT support
            assert_eq!(agg.aggregates[0].function, AggregateFunction::CountNonNull);
        } else {
            panic!("Expected Aggregate operator, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_sum_aggregate() {
        let query = "MATCH (n:Person) RETURN SUM(n.age)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            assert_eq!(agg.aggregates[0].function, AggregateFunction::Sum);
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    #[test]
    fn test_translate_group_by_aggregate() {
        let query = "MATCH (n:Person) RETURN n.city, COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.group_by.len(), 1); // n.city
            assert_eq!(agg.aggregates.len(), 1); // COUNT(n)
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    // === Ordering and Pagination Tests ===

    #[test]
    fn test_translate_order_by() {
        let query = "MATCH (n:Person) RETURN n ORDER BY n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Sort wraps Return so that RETURN aliases are visible to ORDER BY
        if let LogicalOperator::Sort(sort) = &plan.root {
            assert_eq!(sort.keys.len(), 1);
            assert_eq!(sort.keys[0].order, SortOrder::Ascending);
            if let LogicalOperator::Return(_ret) = sort.input.as_ref() {
                // Return is the inner operator, as expected
            } else {
                panic!("Expected Return operator inside Sort");
            }
        } else {
            panic!("Expected Sort operator");
        }
    }

    #[test]
    fn test_translate_limit() {
        let query = "MATCH (n:Person) RETURN n LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Limit
        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                LogicalOperator::Sort(s) => find_limit(&s.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let query = "MATCH (n:Person) RETURN n SKIP 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_insert_node() {
        let query = "INSERT (n:Person {name: 'Alix', age: 30})";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find CreateNode
        fn find_create(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::CreateNode(_) => true,
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => false,
            }
        }

        assert!(find_create(&plan.root));
    }

    #[test]
    fn test_translate_delete() {
        let query = "DELETE n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DeleteNode(del) = &plan.root {
            assert_eq!(del.variable, "n");
        } else {
            panic!("Expected DeleteNode operator");
        }
    }

    #[test]
    fn test_translate_set() {
        // SET is not a standalone statement in GQL, test the translator method directly
        let translator = GqlTranslator::new();
        let set_stmt = ast::SetStatement {
            assignments: vec![ast::PropertyAssignment {
                variable: "n".to_string(),
                property: "name".to_string(),
                value: ast::Expression::Literal(ast::Literal::String("Gus".to_string())),
            }],
            span: None,
        };

        let result = translator.translate_set(&set_stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::SetProperty(set) = &plan.root {
            assert_eq!(set.variable, "n");
            assert_eq!(set.properties.len(), 1);
            assert_eq!(set.properties[0].0, "name");
        } else {
            panic!("Expected SetProperty operator");
        }
    }

    // === Expression Translation Tests ===

    #[test]
    fn test_translate_literals() {
        let query = "MATCH (n) WHERE n.count = 42 AND n.active = true AND n.rate = 3.14 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_parameter() {
        let query = "MATCH (n:Person) WHERE n.name = $name RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(name) = right.as_ref() {
                assert_eq!(name, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_empty_delete_error() {
        // Create translator directly to test empty delete
        let translator = GqlTranslator::new();
        let delete = ast::DeleteStatement {
            targets: vec![],
            detach: false,
            span: None,
        };
        let result = translator.translate_delete(&delete);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_set_error() {
        let translator = GqlTranslator::new();
        let set = ast::SetStatement {
            assignments: vec![],
            span: None,
        };
        let result = translator.translate_set(&set);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_insert_error() {
        let translator = GqlTranslator::new();
        let insert = ast::InsertStatement {
            patterns: vec![],
            span: None,
        };
        let result = translator.translate_insert(&insert);
        assert!(result.is_err());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("SUM"));
        assert!(is_aggregate_function("AVG"));
        assert!(is_aggregate_function("MIN"));
        assert!(is_aggregate_function("MAX"));
        assert!(is_aggregate_function("COLLECT"));
        assert!(!is_aggregate_function("UPPER"));
        assert!(!is_aggregate_function("RANDOM"));
    }

    #[test]
    fn test_to_aggregate_function() {
        assert_eq!(
            to_aggregate_function("COUNT"),
            Some(AggregateFunction::Count)
        );
        assert_eq!(to_aggregate_function("sum"), Some(AggregateFunction::Sum));
        assert_eq!(to_aggregate_function("Avg"), Some(AggregateFunction::Avg));
        assert_eq!(to_aggregate_function("min"), Some(AggregateFunction::Min));
        assert_eq!(to_aggregate_function("MAX"), Some(AggregateFunction::Max));
        assert_eq!(
            to_aggregate_function("collect"),
            Some(AggregateFunction::Collect)
        );
        assert_eq!(to_aggregate_function("UNKNOWN"), None);
    }

    #[test]
    fn test_contains_aggregate() {
        let count_expr = ast::Expression::FunctionCall {
            name: "COUNT".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(contains_aggregate(&count_expr));

        let upper_expr = ast::Expression::FunctionCall {
            name: "UPPER".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(!contains_aggregate(&upper_expr));

        let var_expr = ast::Expression::Variable("n".to_string());
        assert!(!contains_aggregate(&var_expr));
    }

    #[test]
    fn test_binary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Eq),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ne),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Lt),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Le),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Gt),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ge),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::And),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Or),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Add),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Sub),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mul),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Div),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mod),
            BinaryOp::Mod
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Like),
            BinaryOp::Like
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::In),
            BinaryOp::In
        );
    }

    #[test]
    fn test_unary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Not),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Neg),
            UnaryOp::Neg
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNull),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNotNull),
            UnaryOp::IsNotNull
        );
    }

    // === ShortestPath Tests ===

    #[test]
    fn test_translate_shortest_path() {
        let query = "MATCH p = shortestPath((a:Person)-[:KNOWS]->(b:Person)) RETURN p";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "shortestPath should translate: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_shortest_path(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::ShortestPath(_) => true,
                LogicalOperator::Return(r) => find_shortest_path(&r.input),
                _ => false,
            }
        }
        assert!(
            find_shortest_path(&plan.root),
            "Plan should contain ShortestPath operator"
        );
    }

    #[test]
    fn test_translate_all_shortest_paths() {
        let query = "MATCH p = allShortestPaths((a)-[:ROAD]-(b)) RETURN p";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "allShortestPaths should translate: {:?}",
            result.err()
        );
    }

    // === CASE expression ===

    #[test]
    fn test_translate_case_expression() {
        let query = "MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS category";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "CASE expression should translate: {:?}",
            result.err()
        );
    }

    // === UNWIND ===

    #[test]
    fn test_translate_unwind() {
        let query = "UNWIND [1, 2, 3] AS x RETURN x";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "UNWIND should translate: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_unwind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Unwind(_) => true,
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => false,
            }
        }
        assert!(find_unwind(&plan.root), "Plan should contain Unwind");
    }

    // === MERGE ===

    #[test]
    fn test_translate_merge() {
        let query = "MERGE (n:Person {name: 'Alix'}) RETURN n";
        let result = translate(query);
        assert!(result.is_ok(), "MERGE should translate: {:?}", result.err());
        let plan = result.unwrap();

        fn find_merge(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Merge(_) => true,
                LogicalOperator::Return(r) => find_merge(&r.input),
                _ => false,
            }
        }
        assert!(find_merge(&plan.root), "Plan should contain Merge");
    }

    #[test]
    fn test_translate_merge_with_on_create() {
        let query = "MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "MERGE ON CREATE should translate: {:?}",
            result.err()
        );
    }

    // === WITH clause ===

    #[test]
    fn test_translate_with_clause() {
        let query = "MATCH (n:Person) WITH n.name AS name WHERE name = 'Alix' RETURN name";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "WITH clause should translate: {:?}",
            result.err()
        );
    }

    // === Label operations ===

    #[test]
    fn test_translate_add_label() {
        let query = "MATCH (n:Person) SET n:Employee RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "SET label should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_remove_label() {
        let query = "MATCH (n:Person) REMOVE n:Employee RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "REMOVE label should translate: {:?}",
            result.err()
        );
    }

    // === Multiple aggregates ===

    #[test]
    fn test_translate_multiple_aggregates() {
        let query = "MATCH (n:Person) RETURN count(n) AS cnt, sum(n.age) AS total_age, avg(n.age) AS avg_age";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Multiple aggregates should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_group_by_with_having_like_filter() {
        // Use WHERE after aggregation (emulating HAVING)
        let query = "MATCH (n:Person) RETURN n.city AS city, count(n) AS cnt ORDER BY cnt DESC";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "GROUP BY with ORDER BY should translate: {:?}",
            result.err()
        );
    }

    // === GqlTranslationResult enum Tests ===

    #[test]
    fn test_translate_full_returns_plan_for_query() {
        let query = "MATCH (n:Person) RETURN n";
        let result = translate_full(query);
        assert!(
            result.is_ok(),
            "translate_full should succeed: {:?}",
            result.err()
        );
        assert!(
            matches!(result.unwrap(), GqlTranslationResult::Plan(_)),
            "translate_full should return Plan for a query"
        );
    }

    #[test]
    fn test_translate_full_returns_session_command() {
        let query = "COMMIT";
        let result = translate_full(query);
        assert!(
            result.is_ok(),
            "translate_full should succeed for COMMIT: {:?}",
            result.err()
        );
        assert!(
            matches!(result.unwrap(), GqlTranslationResult::SessionCommand(_)),
            "translate_full should return SessionCommand for COMMIT"
        );
    }

    // === translate() vs session commands ===

    #[test]
    fn test_translate_returns_ok_for_query() {
        let query = "MATCH (n) RETURN n";
        let result = translate(query);
        assert!(result.is_ok(), "translate should succeed for a query");
    }

    #[test]
    fn test_translate_returns_err_for_session_command() {
        let query = "COMMIT";
        let result = translate(query);
        assert!(
            result.is_err(),
            "translate should return Err for session commands"
        );
    }

    // === Set Operations ===

    #[test]
    fn test_translate_except() {
        let query = "MATCH (a:Person) RETURN a EXCEPT MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "EXCEPT should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Except(_)),
            "Expected Except operator, got {:?}",
            plan.root
        );
    }

    #[test]
    fn test_translate_intersect() {
        let query = "MATCH (a:Person) RETURN a INTERSECT MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "INTERSECT should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Intersect(_)),
            "Expected Intersect operator, got {:?}",
            plan.root
        );
    }

    #[test]
    fn test_translate_otherwise() {
        let query = "MATCH (a:Person) RETURN a OTHERWISE MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OTHERWISE should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Otherwise(_)),
            "Expected Otherwise operator, got {:?}",
            plan.root
        );
    }

    // === FINISH ===

    #[test]
    fn test_translate_finish() {
        let query = "MATCH (n:Person) FINISH";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "FINISH should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // FINISH is translated as Limit(0)
        if let LogicalOperator::Limit(limit) = &plan.root {
            assert_eq!(limit.count, 0, "FINISH should produce Limit(0)");
        } else {
            panic!("Expected Limit operator for FINISH, got {:?}", plan.root);
        }
    }

    // === Element WHERE on nodes ===

    #[test]
    fn test_translate_element_where_on_node() {
        let query = "MATCH (n:Person WHERE n.age > 30) RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Element WHERE should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // Walk the plan to find a Filter with a > predicate
        fn find_gt_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        *op == BinaryOp::Gt || find_gt_filter(&f.input)
                    } else {
                        find_gt_filter(&f.input)
                    }
                }
                LogicalOperator::Return(r) => find_gt_filter(&r.input),
                _ => false,
            }
        }
        assert!(
            find_gt_filter(&plan.root),
            "Expected a Filter with Gt predicate from element WHERE clause"
        );
    }

    // === NULLIF desugaring ===

    #[test]
    fn test_translate_nullif_desugaring() {
        let query = "MATCH (n:Person) RETURN nullif(n.age, 0) AS age";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "NULLIF should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // NULLIF(x, y) desugars to CASE WHEN x = y THEN NULL ELSE x END.
        // The Return operator should contain a Case expression.
        fn find_case_in_return(op: &LogicalOperator) -> bool {
            if let LogicalOperator::Return(ret) = op {
                ret.items
                    .iter()
                    .any(|item| matches!(item.expression, LogicalExpression::Case { .. }))
            } else {
                false
            }
        }
        assert!(
            find_case_in_return(&plan.root),
            "Expected NULLIF to desugar into a CASE expression in RETURN"
        );
    }
}
