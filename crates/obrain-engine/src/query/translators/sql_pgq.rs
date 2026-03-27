//! SQL/PGQ AST to Logical Plan translator.
//!
//! Translates parsed SQL/PGQ `GRAPH_TABLE` queries into the common logical plan
//! representation. The inner MATCH clause reuses GQL AST types, so pattern
//! translation follows the GQL translator pattern.

use super::common::{
    combine_with_and, is_aggregate_function, to_aggregate_function, wrap_filter, wrap_limit,
    wrap_return, wrap_skip, wrap_sort,
};
use crate::query::plan::{
    AggregateExpr, AggregateOp, BinaryOp, CallProcedureOp, CreatePropertyGraphOp, ExpandDirection,
    ExpandOp, LeftJoinOp, LogicalExpression, LogicalOperator, LogicalPlan, NodeScanOp, PathMode,
    ProcedureYield, PropertyGraphEdgeTable, PropertyGraphNodeTable, ReturnItem, SortKey, SortOrder,
    UnaryOp,
};
use obrain_adapters::query::sql_pgq::{self, ast};
use obrain_common::types::Value;
use obrain_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Translates a SQL/PGQ query string to a logical plan.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let statement = sql_pgq::parse(query)?;
    let translator = SqlPgqTranslator::new();
    translator.translate_statement(&statement)
}

/// SQL/PGQ AST to logical plan translator.
struct SqlPgqTranslator;

impl SqlPgqTranslator {
    fn new() -> Self {
        Self
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Select(select) => self.translate_select(select),
            ast::Statement::CreatePropertyGraph(cpg) => self.translate_create_property_graph(cpg),
            ast::Statement::Call(call) => self.translate_call(call),
        }
    }

    fn translate_select(&self, select: &ast::SelectStatement) -> Result<LogicalPlan> {
        // Build the column alias → original expression map for resolving SQL references.
        // SQL WHERE/ORDER BY reference output column aliases (e.g., `g.age`), which must
        // be resolved back to graph expressions (e.g., `a.age`) for the binder/planner.
        let column_map: hashbrown::HashMap<&str, &ast::Expression> = select
            .graph_table
            .columns
            .items
            .iter()
            .map(|col| (col.alias.as_str(), &col.expression))
            .collect();
        let table_alias = select.table_alias.as_deref();

        // Plan structure: Limit → Skip → Return → Sort → Filter → NodeScan/Expand
        //
        // SQL WHERE and ORDER BY operate on output column aliases, but the binder/planner
        // need graph-level expressions. We resolve aliases back to graph expressions and
        // place Filter/Sort *below* the Return (COLUMNS) projection.

        // 1. Translate MATCH patterns → NodeScan + Expand
        let mut plan = self.translate_match(&select.graph_table.match_clause)?;

        // 1b. Translate optional matches → LeftJoin
        for opt_match in &select.graph_table.optional_matches {
            let right = self.translate_match(opt_match)?;
            plan = LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(plan),
                right: Box::new(right),
                condition: None,
            });
        }

        // 2. Translate SQL WHERE → Filter (below Return)
        if let Some(where_expr) = &select.where_clause {
            let predicate = self.translate_sql_expression(where_expr, table_alias, &column_map)?;
            plan = wrap_filter(plan, predicate);
        }

        // 3. Translate ORDER BY → Sort (below Return)
        if let Some(order_by) = &select.order_by {
            let keys: Vec<SortKey> = order_by
                .iter()
                .map(|item| {
                    Ok(SortKey {
                        expression: self.translate_sql_expression(
                            &item.expression,
                            table_alias,
                            &column_map,
                        )?,
                        order: match item.direction {
                            ast::SortDirection::Asc => SortOrder::Ascending,
                            ast::SortDirection::Desc => SortOrder::Descending,
                        },
                        nulls: None,
                    })
                })
                .collect::<Result<_>>()?;

            plan = wrap_sort(plan, keys);
        }

        // 4. Translate OFFSET → Skip (below Return, after Sort)
        if let Some(offset) = select.offset {
            plan = wrap_skip(plan, offset as usize);
        }

        // 5. Translate LIMIT → Limit (below Return, after Skip)
        if let Some(limit) = select.limit {
            plan = wrap_limit(plan, limit as usize);
        }

        // 6. Translate COLUMNS clause → Return (outermost projection)
        plan = self.translate_columns(&select.graph_table.columns, plan)?;

        // 7. Handle outer SELECT list with aggregates (e.g., SELECT COUNT(*) AS total)
        if let ast::SelectList::Columns(items) = &select.select_list {
            let has_aggregates = items.iter().any(|item| {
                matches!(
                    &item.expression,
                    ast::Expression::FunctionCall { name, .. }
                    if is_aggregate_function(name)
                )
            });

            if has_aggregates {
                let mut aggregates = Vec::new();
                let mut group_by = Vec::new();

                for item in items {
                    let alias = item.alias.clone();
                    match &item.expression {
                        ast::Expression::FunctionCall {
                            name,
                            args,
                            distinct,
                        } if is_aggregate_function(name) => {
                            let agg_fn = to_aggregate_function(name).expect(
                                "aggregate function validated by is_aggregate_function guard",
                            );
                            let expr = if args.len() == 1 {
                                let arg = &args[0];
                                if matches!(arg, ast::Expression::Variable(v) if v == "*") {
                                    None // COUNT(*)
                                } else {
                                    Some(self.translate_expression(arg, None)?)
                                }
                            } else {
                                None
                            };
                            aggregates.push(AggregateExpr {
                                function: agg_fn,
                                expression: expr,
                                expression2: None,
                                distinct: *distinct,
                                alias,
                                percentile: None,
                                separator: None,
                            });
                        }
                        _ => {
                            // Non-aggregate in SELECT with aggregates → group by
                            let expr = self.translate_expression(&item.expression, None)?;
                            group_by.push(expr);
                        }
                    }
                }

                plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by,
                    aggregates,
                    input: Box::new(plan),
                    having: None,
                });

                // Wrap with Return for the aggregate result
                let return_items: Vec<ReturnItem> = items
                    .iter()
                    .map(|item| {
                        let alias = item.alias.clone().unwrap_or_else(|| "result".to_string());
                        ReturnItem {
                            expression: LogicalExpression::Variable(alias.clone()),
                            alias: Some(alias),
                        }
                    })
                    .collect();
                plan = wrap_return(plan, return_items, false);
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    // ==================== CALL Translation ====================

    fn translate_call(&self, call: &ast::CallStatement) -> Result<LogicalPlan> {
        let arguments = call
            .arguments
            .iter()
            .map(|a| self.translate_expression(a, None))
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
            let predicate = self.translate_expression(&where_clause.expression, None)?;
            plan = wrap_filter(plan, predicate);
        }

        // Apply RETURN clause (ORDER BY, SKIP, LIMIT, projection)
        if let Some(return_clause) = &call.return_clause {
            // Apply ORDER BY
            if let Some(order_by) = &return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression, None)?,
                            order: match item.order {
                                ast::GqlSortOrder::Asc => SortOrder::Ascending,
                                ast::GqlSortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: None,
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

            // Apply RETURN projection (only when explicit items are present)
            if !return_clause.items.is_empty() {
                let return_items = return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression, None)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_return(plan, return_items, return_clause.distinct);
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    // ==================== MATCH Translation ====================

    fn translate_match(&self, match_clause: &ast::MatchClause) -> Result<LogicalOperator> {
        let mut plan: Option<LogicalOperator> = None;

        for aliased in &match_clause.patterns {
            plan = Some(self.translate_pattern(&aliased.pattern, plan)?);
        }

        plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty MATCH pattern",
            ))
        })
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => self.translate_node_pattern(node, input),
            ast::Pattern::Path(path) => self.translate_path_pattern(path, input),
            ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SQL/PGQ does not support quantified or union patterns",
            ))),
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node.variable.clone().unwrap_or_else(|| "_anon".to_string());
        let label = node.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add filters for inline properties (e.g., {city: 'NYC'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    fn translate_path_pattern(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = self.translate_node_pattern(&path.source, input)?;

        for edge in &path.edges {
            plan = self.translate_edge_pattern(edge, plan)?;
        }

        Ok(plan)
    }

    fn translate_edge_pattern(
        &self,
        edge: &ast::EdgePattern,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let from_variable = Self::get_last_variable(&input)?;
        let edge_variable = edge.variable.clone();
        let edge_types = edge.types.clone();
        let to_variable = edge
            .target
            .variable
            .clone()
            .unwrap_or_else(|| "_anon".to_string());
        let target_label = edge.target.labels.first().cloned();

        let direction = match edge.direction {
            ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
            ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
            ast::EdgeDirection::Undirected => ExpandDirection::Both,
        };

        let (min_hops, max_hops) = (edge.min_hops.unwrap_or(1), edge.max_hops.or(Some(1)));

        // Set path_alias for variable-length patterns so path functions work
        let is_variable_length =
            min_hops != 1 || max_hops.is_none() || max_hops.is_some_and(|m| m != 1);
        let path_alias = if is_variable_length {
            edge_variable.clone()
        } else {
            None
        };

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable,
            to_variable: to_variable.clone(),
            edge_variable,
            direction,
            edge_types,
            min_hops,
            max_hops,
            input: Box::new(input),
            path_alias,
            path_mode: PathMode::Walk,
        });

        // Add label filter on the target node if present
        if let Some(label) = target_label {
            Ok(wrap_filter(
                expand,
                LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        LogicalExpression::Variable(to_variable),
                        LogicalExpression::Literal(Value::from(label)),
                    ],
                    distinct: false,
                },
            ))
        } else {
            Ok(expand)
        }
    }

    // ==================== COLUMNS Translation ====================

    fn translate_columns(
        &self,
        columns: &ast::ColumnsClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let items: Vec<ReturnItem> = columns
            .items
            .iter()
            .map(|col| {
                Ok(ReturnItem {
                    expression: self.translate_expression(&col.expression, None)?,
                    alias: Some(col.alias.clone()),
                })
            })
            .collect::<Result<_>>()?;

        Ok(wrap_return(input, items, false))
    }

    // ==================== Expression Translation ====================

    /// Translates a SQL-level expression (WHERE, ORDER BY) that references output columns.
    ///
    /// When the expression references `g.age` (table alias + column name), this resolves
    /// the column alias back to the original graph expression from the COLUMNS clause
    /// (e.g., `n.age`). This ensures the binder can validate the expression.
    fn translate_sql_expression(
        &self,
        expr: &ast::Expression,
        table_alias: Option<&str>,
        column_map: &hashbrown::HashMap<&str, &ast::Expression>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => {
                // Check if this variable name is a column alias
                if let Some(original_expr) = column_map.get(name.as_str()) {
                    return self.translate_expression(original_expr, None);
                }
                Ok(LogicalExpression::Variable(name.clone()))
            }
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { variable, property } => {
                // If `variable` is the table alias, resolve `property` as a column alias
                if let Some(alias) = table_alias
                    && variable == alias
                {
                    if let Some(original_expr) = column_map.get(property.as_str()) {
                        return self.translate_expression(original_expr, None);
                    }
                    // Column not found in COLUMNS clause - use as variable reference
                    return Ok(LogicalExpression::Variable(property.clone()));
                }
                Ok(LogicalExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_sql_expression(left, table_alias, column_map)?;
                let right_expr = self.translate_sql_expression(right, table_alias, column_map)?;
                let binary_op = self.translate_binary_op(*op)?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr =
                    self.translate_sql_expression(operand, table_alias, column_map)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand_expr);
                }
                let unary_op = self.translate_unary_op(*op)?;
                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // Special handling for path functions in SQL context
                if args.len() == 1
                    && let ast::Expression::Variable(var_name) = &args[0]
                {
                    match name.to_uppercase().as_str() {
                        "LENGTH" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_length_{var_name}"
                            )));
                        }
                        "NODES" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_nodes_{var_name}"
                            )));
                        }
                        "EDGES" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_edges_{var_name}"
                            )));
                        }
                        _ => {}
                    }
                }

                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_sql_expression(a, table_alias, column_map))
                    .collect::<Result<_>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: *distinct,
                })
            }
            _ => self.translate_expression(expr, table_alias),
        }
    }

    /// Translates a GQL AST expression to a logical expression.
    ///
    /// Used for COLUMNS clause expressions (graph-level, no table alias resolution).
    fn translate_expression(
        &self,
        expr: &ast::Expression,
        table_alias: Option<&str>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { variable, property } => {
                // Check if the variable is a table alias (SQL qualification).
                if let Some(alias) = table_alias
                    && variable == alias
                {
                    return Ok(LogicalExpression::Variable(property.clone()));
                }
                // Otherwise it's a graph property access (e.g., `a.name` inside COLUMNS)
                Ok(LogicalExpression::Property {
                    variable: variable.clone(),
                    property: property.clone(),
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_expression(left, table_alias)?;
                let right_expr = self.translate_expression(right, table_alias)?;
                let binary_op = self.translate_binary_op(*op)?;

                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr = self.translate_expression(operand, table_alias)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand_expr);
                }
                let unary_op = self.translate_unary_op(*op)?;

                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // Special handling for path functions: LENGTH(p), NODES(p), EDGES(p)
                // These reference path variables and are translated to internal column names
                if args.len() == 1
                    && let ast::Expression::Variable(var_name) = &args[0]
                {
                    match name.to_uppercase().as_str() {
                        "LENGTH" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_length_{var_name}"
                            )));
                        }
                        "NODES" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_nodes_{var_name}"
                            )));
                        }
                        "EDGES" => {
                            return Ok(LogicalExpression::Variable(format!(
                                "_path_edges_{var_name}"
                            )));
                        }
                        _ => {}
                    }
                }

                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_expression(a, table_alias))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: *distinct,
                })
            }
            ast::Expression::List(items) => {
                let translated: Vec<LogicalExpression> = items
                    .iter()
                    .map(|i| self.translate_expression(i, table_alias))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::List(translated))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = input
                    .as_ref()
                    .map(|e| self.translate_expression(e, table_alias))
                    .transpose()?
                    .map(Box::new);

                let when_clauses: Vec<(LogicalExpression, LogicalExpression)> = whens
                    .iter()
                    .map(|(w, t)| {
                        Ok((
                            self.translate_expression(w, table_alias)?,
                            self.translate_expression(t, table_alias)?,
                        ))
                    })
                    .collect::<Result<_>>()?;

                let else_result = else_clause
                    .as_ref()
                    .map(|e| self.translate_expression(e, table_alias))
                    .transpose()?
                    .map(Box::new);

                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause: else_result,
                })
            }
            ast::Expression::Map(entries) => {
                let entries = entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v, table_alias)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Map(entries))
            }
            ast::Expression::ExistsSubquery { .. } => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "EXISTS subquery not supported in SQL/PGQ",
            ))),
            ast::Expression::CountSubquery { .. } => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "COUNT subquery not supported in SQL/PGQ",
            ))),
            ast::Expression::ValueSubquery { .. } => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "VALUE subquery not supported in SQL/PGQ",
            ))),
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base, table_alias)?;
                let index_expr = self.translate_expression(index, table_alias)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::LetIn { .. }
            | ast::Expression::ListComprehension { .. }
            | ast::Expression::ListPredicate { .. }
            | ast::Expression::PatternComprehension { .. }
            | ast::Expression::Reduce { .. } => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "This expression type is not supported in SQL/PGQ",
            ))),
        }
    }

    fn translate_literal(&self, lit: &ast::Literal) -> Result<LogicalExpression> {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::from(s.as_str()),
            ast::Literal::Date(s) => obrain_common::types::Date::parse(s)
                .map_or_else(|| Value::from(s.as_str()), Value::Date),
            ast::Literal::Time(s) => obrain_common::types::Time::parse(s)
                .map_or_else(|| Value::from(s.as_str()), Value::Time),
            ast::Literal::Duration(s) => obrain_common::types::Duration::parse(s)
                .map_or_else(|| Value::from(s.as_str()), Value::Duration),
            ast::Literal::Datetime(s) => {
                if let Some(pos) = s.find('T') {
                    if let (Some(d), Some(t)) = (
                        obrain_common::types::Date::parse(&s[..pos]),
                        obrain_common::types::Time::parse(&s[pos + 1..]),
                    ) {
                        Value::Timestamp(obrain_common::types::Timestamp::from_date_time(d, t))
                    } else {
                        Value::from(s.as_str())
                    }
                } else {
                    Value::from(s.as_str())
                }
            }
            ast::Literal::ZonedDatetime(s) => obrain_common::types::ZonedDatetime::parse(s)
                .map_or_else(|| Value::from(s.as_str()), Value::ZonedDatetime),
            ast::Literal::ZonedTime(s) => {
                if let Some(t) = obrain_common::types::Time::parse(s)
                    && t.offset_seconds().is_some()
                {
                    Value::Time(t)
                } else {
                    Value::from(s.as_str())
                }
            }
        };
        Ok(LogicalExpression::Literal(value))
    }

    fn translate_binary_op(&self, op: ast::BinaryOp) -> Result<BinaryOp> {
        Ok(match op {
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
        })
    }

    fn translate_unary_op(&self, op: ast::UnaryOp) -> Result<UnaryOp> {
        Ok(match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            // Pos is handled as a no-op at the call site; this arm is unreachable.
            ast::UnaryOp::Pos => UnaryOp::Not,
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        })
    }

    // ==================== DDL Translation ====================

    fn translate_create_property_graph(
        &self,
        cpg: &ast::CreatePropertyGraphStatement,
    ) -> Result<LogicalPlan> {
        // Validate: edge table references must point to defined node tables
        let node_table_names: hashbrown::HashSet<&str> =
            cpg.node_tables.iter().map(|t| t.name.as_str()).collect();

        for edge_table in &cpg.edge_tables {
            if !edge_table.source_table.is_empty()
                && !node_table_names.contains(edge_table.source_table.as_str())
            {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "Edge table '{}' references unknown source table '{}'",
                        edge_table.name, edge_table.source_table
                    ),
                )));
            }
            if !edge_table.target_table.is_empty()
                && !node_table_names.contains(edge_table.target_table.as_str())
            {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "Edge table '{}' references unknown target table '{}'",
                        edge_table.name, edge_table.target_table
                    ),
                )));
            }
        }

        let node_tables = cpg
            .node_tables
            .iter()
            .map(|nt| PropertyGraphNodeTable {
                name: nt.name.clone(),
                columns: nt
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.data_type.to_string()))
                    .collect(),
            })
            .collect();

        let edge_tables = cpg
            .edge_tables
            .iter()
            .map(|et| PropertyGraphEdgeTable {
                name: et.name.clone(),
                columns: et
                    .columns
                    .iter()
                    .map(|c| (c.name.clone(), c.data_type.to_string()))
                    .collect(),
                source_table: et.source_table.clone(),
                target_table: et.target_table.clone(),
            })
            .collect();

        let op = LogicalOperator::CreatePropertyGraph(CreatePropertyGraphOp {
            name: cpg.name.clone(),
            node_tables,
            edge_tables,
        });

        Ok(LogicalPlan::new(op))
    }

    // ==================== Helpers ====================

    fn build_property_predicate(
        &self,
        variable: &str,
        properties: &[(String, ast::Expression)],
    ) -> Result<LogicalExpression> {
        let predicates = properties
            .iter()
            .map(|(key, value)| {
                let left = LogicalExpression::Property {
                    variable: variable.to_string(),
                    property: key.clone(),
                };
                let right = self.translate_expression(value, None)?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::Eq,
                    right: Box::new(right),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        combine_with_and(predicates)
    }

    fn get_last_variable(plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::NodeScan(scan) => Ok(scan.variable.clone()),
            LogicalOperator::Expand(expand) => Ok(expand.to_variable.clone()),
            LogicalOperator::Filter(filter) => Self::get_last_variable(&filter.input),
            LogicalOperator::Project(project) => Self::get_last_variable(&project.input),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Cannot get variable from operator",
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_basic_node_query() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            )",
        )
        .unwrap();

        // Return → NodeScan
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return, got {:?}", plan.root);
        };
        assert_eq!(ret.items.len(), 1);
        assert_eq!(ret.items[0].alias.as_deref(), Some("name"));

        let LogicalOperator::NodeScan(scan) = ret.input.as_ref() else {
            panic!("Expected NodeScan");
        };
        assert_eq!(scan.variable, "n");
        assert_eq!(scan.label.as_deref(), Some("Person"));
    }

    #[test]
    fn test_translate_relationship_pattern() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, b.name AS friend)
            )",
        )
        .unwrap();

        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return");
        };
        assert_eq!(ret.items.len(), 2);
    }

    #[test]
    fn test_translate_where_with_table_alias() {
        let plan = translate(
            "SELECT g.person FROM GRAPH_TABLE (
                MATCH (a:Person)
                COLUMNS (a.name AS person)
            ) AS g
            WHERE g.person = 'Alix'",
        )
        .unwrap();

        // Return → Filter → NodeScan (Filter is below Return)
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return at top, got {:?}", plan.root);
        };
        let LogicalOperator::Filter(filter) = ret.input.as_ref() else {
            panic!("Expected Filter below Return");
        };
        // g.person should resolve back to the original COLUMNS expression: a.name
        let LogicalExpression::Binary { left, op, right } = &filter.predicate else {
            panic!("Expected Binary predicate");
        };
        assert_eq!(*op, BinaryOp::Eq);
        assert!(
            matches!(left.as_ref(), LogicalExpression::Property { variable, property } if variable == "a" && property == "name")
        );
        assert!(
            matches!(right.as_ref(), LogicalExpression::Literal(Value::String(s)) if s.as_str() == "Alix")
        );
    }

    #[test]
    fn test_translate_order_limit_offset() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.age AS age)
            )
            ORDER BY n.age DESC
            LIMIT 10
            OFFSET 5",
        )
        .unwrap();

        // Return → Limit → Skip → Sort → NodeScan
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return at top");
        };

        let LogicalOperator::Limit(limit) = ret.input.as_ref() else {
            panic!("Expected Limit below Return");
        };
        assert_eq!(limit.count, 10);

        let LogicalOperator::Skip(skip) = limit.input.as_ref() else {
            panic!("Expected Skip");
        };
        assert_eq!(skip.count, 5);

        let LogicalOperator::Sort(sort) = skip.input.as_ref() else {
            panic!("Expected Sort");
        };
        assert_eq!(sort.keys.len(), 1);
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    #[test]
    fn test_translate_incoming_edge() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)<-[:FOLLOWS]-(b:User)
                COLUMNS (a.name AS person)
            )",
        )
        .unwrap();

        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return");
        };
        // Drill into the input to find the Expand
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }
        let expand = find_expand(ret.input.as_ref()).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_multiple_columns() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS person, e.since AS year, b.name AS friend)
            )",
        )
        .unwrap();

        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return");
        };
        assert_eq!(ret.items.len(), 3);
        assert_eq!(ret.items[0].alias.as_deref(), Some("person"));
        assert_eq!(ret.items[1].alias.as_deref(), Some("year"));
        assert_eq!(ret.items[2].alias.as_deref(), Some("friend"));
    }

    #[test]
    fn test_translate_error_on_empty_query() {
        let result = translate("SELECT FROM");
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_variable_length_with_path_alias() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..5]->(dst:Person)
                COLUMNS (src.name AS source, LENGTH(p) AS distance, dst.name AS target)
            )",
        )
        .unwrap();

        // Return → hasLabel Filter → Expand → NodeScan
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return at top");
        };
        assert_eq!(ret.items.len(), 3);
        assert_eq!(ret.items[0].alias.as_deref(), Some("source"));
        assert_eq!(ret.items[1].alias.as_deref(), Some("distance"));
        assert_eq!(ret.items[2].alias.as_deref(), Some("target"));

        // The LENGTH(p) column should be translated to _path_length_p variable
        assert!(
            matches!(&ret.items[1].expression, LogicalExpression::Variable(v) if v == "_path_length_p")
        );

        // Find the Expand operator and verify path_alias is set
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                LogicalOperator::Return(r) => find_expand(&r.input),
                _ => None,
            }
        }
        let expand = find_expand(ret.input.as_ref()).expect("Expected Expand");
        assert_eq!(expand.path_alias, Some("p".to_string()));
        assert_eq!(expand.min_hops, 1);
        assert_eq!(expand.max_hops, Some(5));
    }

    #[test]
    fn test_translate_nodes_and_edges_path_functions() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (src:Person)-[p:KNOWS*1..3]->(dst:Person)
                COLUMNS (NODES(p) AS path_nodes, EDGES(p) AS path_edges)
            )",
        )
        .unwrap();

        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return");
        };
        assert_eq!(ret.items.len(), 2);

        // NODES(p) → _path_nodes_p
        assert!(
            matches!(&ret.items[0].expression, LogicalExpression::Variable(v) if v == "_path_nodes_p")
        );
        // EDGES(p) → _path_edges_p
        assert!(
            matches!(&ret.items[1].expression, LogicalExpression::Variable(v) if v == "_path_edges_p")
        );
    }

    // === Outer Aggregate Tests ===

    #[test]
    fn test_translate_outer_aggregate_count() {
        use crate::query::plan::AggregateFunction;

        let plan = translate(
            "SELECT COUNT(*) AS total FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g",
        )
        .unwrap();

        // The outer SELECT COUNT(*) should produce an Aggregate wrapping the
        // GRAPH_TABLE result. Walk the tree to find it.
        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Return(r) => find_aggregate(&r.input),
                LogicalOperator::Filter(f) => find_aggregate(&f.input),
                LogicalOperator::Limit(l) => find_aggregate(&l.input),
                LogicalOperator::Skip(s) => find_aggregate(&s.input),
                _ => None,
            }
        }
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate operator for COUNT(*)");
        assert_eq!(agg.aggregates.len(), 1, "Should have exactly one aggregate");
        assert!(
            matches!(agg.aggregates[0].function, AggregateFunction::Count),
            "Aggregate function should be Count"
        );
        // COUNT(*) has no expression (None)
        assert!(
            agg.aggregates[0].expression.is_none(),
            "COUNT(*) should have None expression"
        );
    }

    // === Undirected Edge Tests ===

    #[test]
    fn test_translate_undirected_edge() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[:KNOWS]-(b:Person)
                COLUMNS (a.name AS from_name, b.name AS to_name)
            )",
        )
        .unwrap();
        // Should parse without error; undirected produces Both direction
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Project(p) => find_expand(&p.input),
                _ => None,
            }
        }
        let expand = find_expand(&plan.root).expect("Expected Expand for undirected edge");
        assert_eq!(expand.direction, ExpandDirection::Both);
    }

    // === Node Property Filter Tests ===

    #[test]
    fn test_translate_node_property_filter() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person {age: 30})
                COLUMNS (n.name AS name)
            )",
        )
        .unwrap();
        // Inline property filter should produce a Filter operator
        fn has_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Return(r) => has_filter(&r.input),
                LogicalOperator::Project(p) => has_filter(&p.input),
                _ => false,
            }
        }
        assert!(
            has_filter(&plan.root),
            "Property filter should produce Filter"
        );
    }

    // === CALL Statement Tests ===

    #[test]
    fn test_translate_call_statement() {
        // SQL/PGQ CALL uses ORDER BY directly (no RETURN keyword)
        let plan = translate("CALL db.labels() YIELD label ORDER BY label").unwrap();
        fn find_call(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::CallProcedure(_) => true,
                LogicalOperator::Sort(s) => find_call(&s.input),
                LogicalOperator::Filter(f) => find_call(&f.input),
                _ => false,
            }
        }
        assert!(find_call(&plan.root), "CALL should produce CallProcedure");
    }

    #[test]
    fn test_translate_call_with_where() {
        let plan = translate("CALL db.labels() YIELD label WHERE label <> 'Internal'").unwrap();
        fn find_filter(op: &LogicalOperator) -> bool {
            matches!(op, LogicalOperator::Filter(_))
        }
        assert!(find_filter(&plan.root), "CALL WHERE should produce Filter");
    }

    // === Aggregate Function Tests ===

    #[test]
    fn test_translate_outer_aggregate_sum() {
        use crate::query::plan::AggregateFunction;

        let plan = translate(
            "SELECT SUM(g.score) AS total FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.score AS score)
            ) AS g",
        )
        .unwrap();

        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Return(r) => find_aggregate(&r.input),
                LogicalOperator::Filter(f) => find_aggregate(&f.input),
                _ => None,
            }
        }
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate for SUM");
        assert_eq!(agg.aggregates.len(), 1);
        assert!(matches!(agg.aggregates[0].function, AggregateFunction::Sum));
    }

    #[test]
    fn test_translate_outer_aggregate_avg_min_max() {
        use crate::query::plan::AggregateFunction;

        for (func, expected) in [
            ("AVG", AggregateFunction::Avg),
            ("MIN", AggregateFunction::Min),
            ("MAX", AggregateFunction::Max),
        ] {
            let query = format!(
                "SELECT {}(g.val) AS result FROM GRAPH_TABLE (
                    MATCH (n:Item)
                    COLUMNS (n.value AS val)
                ) AS g",
                func
            );
            let plan = translate(&query).unwrap();

            fn find_agg(op: &LogicalOperator) -> Option<&AggregateOp> {
                match op {
                    LogicalOperator::Aggregate(a) => Some(a),
                    LogicalOperator::Return(r) => find_agg(&r.input),
                    LogicalOperator::Filter(f) => find_agg(&f.input),
                    _ => None,
                }
            }
            let agg =
                find_agg(&plan.root).unwrap_or_else(|| panic!("Expected Aggregate for {func}"));
            assert!(
                agg.aggregates[0].function == expected,
                "{func} should produce {expected:?}"
            );
        }
    }

    // === Multiple MATCH Patterns ===

    #[test]
    fn test_translate_multiple_patterns() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person), (b:Company)
                COLUMNS (a.name AS person, b.name AS company)
            )",
        )
        .unwrap();
        // Multiple patterns produce nested NodeScans (second scan has input from first)
        fn count_node_scans(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::NodeScan(scan) => {
                    1 + scan.input.as_ref().map_or(0, |i| count_node_scans(i))
                }
                LogicalOperator::Filter(f) => count_node_scans(&f.input),
                LogicalOperator::Return(r) => count_node_scans(&r.input),
                _ => 0,
            }
        }
        assert!(
            count_node_scans(&plan.root) >= 2,
            "Multiple patterns should produce at least 2 NodeScans"
        );
    }

    // === Parameter Tests ===

    #[test]
    fn test_translate_parameter_in_outer_where() {
        // Parameters go in the outer SELECT WHERE, not inside GRAPH_TABLE
        let plan = translate(
            "SELECT g.name FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name)
            ) AS g WHERE g.name = $name",
        )
        .unwrap();
        fn find_param(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    matches!(&f.predicate, LogicalExpression::Binary { right, .. }
                        if matches!(right.as_ref(), LogicalExpression::Parameter(_)))
                }
                LogicalOperator::Return(r) => find_param(&r.input),
                _ => false,
            }
        }
        assert!(
            find_param(&plan.root),
            "Parameter should appear in outer WHERE filter"
        );
    }

    // === Multiple COLUMNS with Expressions ===

    #[test]
    fn test_translate_three_columns_with_edge() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person)-[e:KNOWS]->(b:Person)
                COLUMNS (a.name AS from_name, b.name AS to_name, e.since AS since)
            )",
        )
        .unwrap();
        // Find the return and check 3 items
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 3, "Should have 3 return items");
        } else {
            panic!("Expected Return as root operator");
        }
    }

    // === Node Property Filter ===

    #[test]
    fn test_translate_node_inline_property() {
        let plan = translate(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person {age: 30})
                COLUMNS (n.name AS name)
            )",
        )
        .unwrap();
        fn has_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(_) => true,
                LogicalOperator::Return(r) => has_filter(&r.input),
                LogicalOperator::Project(p) => has_filter(&p.input),
                _ => false,
            }
        }
        assert!(
            has_filter(&plan.root),
            "Inline property should produce Filter"
        );
    }

    // === CALL with LIMIT ===

    #[test]
    fn test_translate_call_with_limit() {
        let plan = translate("CALL db.labels() YIELD label LIMIT 5").unwrap();
        fn find_limit(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Limit(_) => true,
                LogicalOperator::Sort(s) => find_limit(&s.input),
                _ => false,
            }
        }
        assert!(find_limit(&plan.root), "CALL LIMIT should produce Limit");
    }

    // === CALL with multiple YIELD items ===

    #[test]
    fn test_translate_call_multiple_yields() {
        let plan = translate("CALL db.stats() YIELD name, value").unwrap();
        fn find_call(op: &LogicalOperator) -> Option<&CallProcedureOp> {
            match op {
                LogicalOperator::CallProcedure(c) => Some(c),
                LogicalOperator::Filter(f) => find_call(&f.input),
                _ => None,
            }
        }
        let call = find_call(&plan.root).expect("Expected CallProcedure");
        let yields = call.yield_items.as_ref().expect("Expected yield items");
        assert_eq!(yields.len(), 2, "Should have 2 yield items");
        assert_eq!(yields[0].field_name, "name");
        assert_eq!(yields[1].field_name, "value");
    }
}
