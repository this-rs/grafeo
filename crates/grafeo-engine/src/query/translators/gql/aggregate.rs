//! Aggregate extraction from RETURN items.

#[allow(clippy::wildcard_imports)]
use super::*;

impl GqlTranslator {
    /// Extracts aggregate and group-by expressions from RETURN items.
    ///
    /// Returns `(aggregates, group_by, post_return)` where `post_return` is
    /// `Some(...)` when any return item wraps an aggregate in a binary/unary
    /// expression (e.g. `count(n) > 0 AS exists`).
    pub(super) fn extract_aggregates_and_groups(
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
                // Non-aggregate expression: group-by key.
                // The Aggregate operator names its output columns using
                // expression_to_string, so the post-Return must reference
                // those column names (not the raw property expression).
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr.clone());
                let col_name = crate::query::planner::common::expression_to_string(&expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(col_name),
                    alias: item.alias.clone(),
                });
            }
        }

        // Always produce a post-Return when any item has an alias, so that
        // output column names reflect the aliases and are visible to ORDER BY.
        let has_aliases = items.iter().any(|item| item.alias.is_some());
        if needs_post_return || has_aliases {
            Ok((aggregates, group_by, Some(post_return_items)))
        } else {
            Ok((aggregates, group_by, None))
        }
    }

    /// Extracts an aggregate from inside a wrapping expression.
    pub(super) fn extract_wrapped_aggregate(
        &self,
        expr: &ast::Expression,
        synthetic_alias: &str,
    ) -> Result<(AggregateExpr, LogicalExpression)> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                // First, check if this function itself is an aggregate
                if let Some(agg) =
                    self.try_extract_aggregate(expr, &Some(synthetic_alias.to_string()))?
                {
                    let substitute = LogicalExpression::Variable(synthetic_alias.to_string());
                    return Ok((agg, substitute));
                }
                // Not an aggregate (e.g. COALESCE, toString, etc.) — recurse
                // into arguments to find the nested aggregate.
                for (i, arg) in args.iter().enumerate() {
                    if contains_aggregate(arg) {
                        let (agg, arg_sub) =
                            self.extract_wrapped_aggregate(arg, synthetic_alias)?;
                        // Rebuild the function call with the substituted argument
                        let mut translated_args = Vec::with_capacity(args.len());
                        for (j, a) in args.iter().enumerate() {
                            if j == i {
                                translated_args.push(arg_sub.clone());
                            } else {
                                translated_args.push(self.translate_expression(a)?);
                            }
                        }
                        let substitute = LogicalExpression::FunctionCall {
                            name: name.clone(),
                            args: translated_args,
                            distinct: *distinct,
                        };
                        return Ok((agg, substitute));
                    }
                }
                Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!(
                        "contains_aggregate was true but no aggregate found in arguments of {name}"
                    ),
                )))
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
                let (agg, sub) = self.extract_wrapped_aggregate(operand, synthetic_alias)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok((agg, sub));
                }
                let unary_op = self.translate_unary_op(*op);
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

    /// Tries to extract an aggregate expression from an AST expression.
    pub(super) fn try_extract_aggregate(
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

/// Checks if an AST expression contains an aggregate function call.
pub(super) fn contains_aggregate(expr: &ast::Expression) -> bool {
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
