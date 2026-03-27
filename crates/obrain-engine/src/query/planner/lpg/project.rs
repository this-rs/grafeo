//! Projection, RETURN, sort, limit, and skip planning.

use super::{
    Arc, Error, GraphStore, HashMap, LimitOp, LogicalExpression, LogicalOperator, LogicalType,
    NullOrder, Operator, PhysicalSortKey, ProjectExpr, ProjectOperator, Result, ReturnOp, SkipOp,
    SortDirection, SortOp, SortOperator, SortOrder, common, expression_to_string,
    value_to_logical_type,
};

impl super::Planner {
    /// Plans a RETURN clause.
    pub(super) fn plan_return(&self, ret: &ReturnOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Handle Empty input (standalone RETURN like: RETURN 2 * 3 AS product)
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(ret.input.as_ref(), LogicalOperator::Empty) {
                let single_row_op: Box<dyn Operator> = Box::new(
                    obrain_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row_op, Vec::new())
            } else {
                self.plan_operator(&ret.input)?
            };

        self.plan_return_with_input(ret, input_op, input_columns)
    }

    /// Plans a RETURN operator with an already-planned input operator.
    /// This is used by `plan_sort` when ORDER BY needs pre-Return property projections.
    pub(super) fn plan_return_with_input(
        &self,
        ret: &ReturnOp,
        input_op: Box<dyn Operator>,
        input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (operator, columns) = self.plan_return_projection(ret, input_op, input_columns)?;

        // Apply DISTINCT if requested
        if ret.distinct {
            let schema = vec![LogicalType::Any; columns.len()];
            Ok(common::build_distinct(operator, columns, None, schema))
        } else {
            Ok((operator, columns))
        }
    }

    /// Plans the projection part of a RETURN clause (without DISTINCT).
    fn plan_return_projection(
        &self,
        ret: &ReturnOp,
        input_op: Box<dyn Operator>,
        input_columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Expand RETURN * wildcard: replace with all user-visible input columns
        let expanded_items;
        let items = if ret.items.len() == 1
            && matches!(&ret.items[0].expression, LogicalExpression::Variable(n) if n == "*")
        {
            expanded_items = input_columns
                .iter()
                .filter(|col| !col.starts_with('_')) // Skip internal columns
                .map(|col| crate::query::plan::ReturnItem {
                    expression: LogicalExpression::Variable(col.clone()),
                    alias: None,
                })
                .collect::<Vec<_>>();
            &expanded_items
        } else {
            &ret.items
        };

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Extract column names from return items
        let columns: Vec<String> = items
            .iter()
            .map(|item| {
                item.alias.clone().unwrap_or_else(|| {
                    // Generate a default name from the expression
                    expression_to_string(&item.expression)
                })
            })
            .collect();

        // Check if we need a project operator (for property access or expression evaluation)
        let needs_project = items
            .iter()
            .any(|item| !matches!(&item.expression, LogicalExpression::Variable(_)));

        if needs_project {
            // Build project expressions
            let mut projections = Vec::with_capacity(items.len());
            let mut output_types = Vec::with_capacity(items.len());

            for item in items {
                match &item.expression {
                    LogicalExpression::Variable(name) => {
                        let col_idx = *variable_columns.get(name).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", name))
                        })?;
                        // Path detail variables and UNWIND/FOR scalar variables pass through as-is
                        if name.starts_with("_path_nodes_")
                            || name.starts_with("_path_edges_")
                            || name.starts_with("_path_length_")
                            || self.scalar_columns.borrow().contains(name)
                        {
                            projections.push(ProjectExpr::Column(col_idx));
                            output_types.push(LogicalType::Any);
                        } else if self.edge_columns.borrow().contains(name) {
                            projections.push(ProjectExpr::EdgeResolve { column: col_idx });
                            output_types.push(LogicalType::Any);
                        } else {
                            projections.push(ProjectExpr::NodeResolve { column: col_idx });
                            output_types.push(LogicalType::Any);
                        }
                    }
                    LogicalExpression::Property { variable, property } => {
                        let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                            Error::Internal(format!("Variable '{}' not found in input", variable))
                        })?;
                        projections.push(ProjectExpr::PropertyAccess {
                            column: col_idx,
                            property: property.clone(),
                        });
                        // Property could be any type - use Any/Generic to preserve type
                        output_types.push(LogicalType::Any);
                    }
                    LogicalExpression::Literal(value) => {
                        projections.push(ProjectExpr::Constant(value.clone()));
                        output_types.push(value_to_logical_type(value));
                    }
                    LogicalExpression::FunctionCall { name, args, .. } => {
                        // Handle built-in functions
                        match name.to_lowercase().as_str() {
                            "type" => {
                                // type(r) returns the edge type string
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "type() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    let col_idx =
                                        *variable_columns.get(var_name).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::EdgeType { column: col_idx });
                                    output_types.push(LogicalType::String);
                                } else {
                                    return Err(Error::Internal(
                                        "type() argument must be a variable".to_string(),
                                    ));
                                }
                            }
                            "length" => {
                                // length(p) returns the path length for path variables,
                                // or delegates to the expression evaluator for other
                                // arguments (e.g. length(a.name) on strings/lists).
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "length() requires exactly one argument".to_string(),
                                    ));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    // Try direct column first, then path detail column
                                    let path_col = format!("_path_length_{var_name}");
                                    let col_idx = variable_columns
                                        .get(&path_col)
                                        .or_else(|| variable_columns.get(var_name))
                                        .ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{}' not found in input",
                                                var_name
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(*col_idx));
                                    output_types.push(LogicalType::Int64);
                                } else {
                                    // Non-variable argument (e.g. property access):
                                    // fall through to expression evaluation
                                    let filter_expr = self.convert_expression(&item.expression)?;
                                    projections.push(ProjectExpr::Expression {
                                        expr: filter_expr,
                                        variable_columns: variable_columns.clone(),
                                    });
                                    output_types.push(LogicalType::Any);
                                }
                            }
                            "nodes" | "edges" | "relationships" => {
                                // nodes(p) / edges(p) / relationships(p) returns path components
                                let func_name = name.to_lowercase();
                                if args.len() != 1 {
                                    return Err(Error::Internal(format!(
                                        "{}() requires exactly one argument",
                                        name
                                    )));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    // Map to internal column name
                                    let suffix = if func_name == "nodes" {
                                        "nodes"
                                    } else {
                                        "edges"
                                    };
                                    let path_col = format!("_path_{suffix}_{var_name}");
                                    let col_idx = variable_columns
                                        .get(&path_col)
                                        .or_else(|| variable_columns.get(var_name))
                                        .ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{var_name}' not found in input",
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(*col_idx));
                                    output_types.push(LogicalType::Any);
                                } else {
                                    return Err(Error::Internal(format!(
                                        "{}() argument must be a variable",
                                        name
                                    )));
                                }
                            }
                            // For other functions (head, tail, size, etc.), use expression evaluation
                            _ => {
                                let filter_expr = self.convert_expression(&item.expression)?;
                                projections.push(ProjectExpr::Expression {
                                    expr: filter_expr,
                                    variable_columns: variable_columns.clone(),
                                });
                                output_types.push(LogicalType::Any);
                            }
                        }
                    }
                    LogicalExpression::Case { .. } => {
                        // Convert CASE expression to FilterExpression for evaluation
                        let filter_expr = self.convert_expression(&item.expression)?;
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns: variable_columns.clone(),
                        });
                        // CASE can return any type - use Any
                        output_types.push(LogicalType::Any);
                    }
                    LogicalExpression::Binary { .. }
                    | LogicalExpression::Unary { .. }
                    | LogicalExpression::List(_)
                    | LogicalExpression::Map(_)
                    | LogicalExpression::IndexAccess { .. }
                    | LogicalExpression::SliceAccess { .. }
                    | LogicalExpression::CountSubquery(_)
                    | LogicalExpression::ValueSubquery(_)
                    | LogicalExpression::MapProjection { .. }
                    | LogicalExpression::Reduce { .. }
                    | LogicalExpression::PatternComprehension { .. }
                    | LogicalExpression::ListComprehension { .. }
                    | LogicalExpression::ListPredicate { .. } => {
                        // Convert complex expressions to FilterExpression for evaluation
                        let filter_expr = self.convert_expression(&item.expression)?;
                        projections.push(ProjectExpr::Expression {
                            expr: filter_expr,
                            variable_columns: variable_columns.clone(),
                        });
                        output_types.push(LogicalType::Any);
                    }
                    _ => {
                        return Err(Error::Internal(format!(
                            "Unsupported RETURN expression: {:?}",
                            item.expression
                        )));
                    }
                }
            }

            let operator = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );

            Ok((operator, columns))
        } else {
            // Simple case: all return items are bare variables
            // Emit resolve variants for entity variables
            let mut projections = Vec::with_capacity(items.len());
            let mut output_types = Vec::with_capacity(items.len());

            for item in items {
                if let LogicalExpression::Variable(name) = &item.expression {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    if self.scalar_columns.borrow().contains(name) {
                        projections.push(ProjectExpr::Column(col_idx));
                        output_types.push(LogicalType::Any);
                    } else if self.edge_columns.borrow().contains(name) {
                        projections.push(ProjectExpr::EdgeResolve { column: col_idx });
                        output_types.push(LogicalType::Any);
                    } else {
                        projections.push(ProjectExpr::NodeResolve { column: col_idx });
                        output_types.push(LogicalType::Any);
                    }
                }
            }

            // Skip ProjectOperator only when all projections are plain Column pass-throughs
            // (i.e., only scalar variables with no reordering). NodeResolve/EdgeResolve
            // always require a ProjectOperator with store access.
            if projections.len() == input_columns.len()
                && projections
                    .iter()
                    .enumerate()
                    .all(|(i, p)| matches!(p, ProjectExpr::Column(c) if *c == i))
            {
                // No reordering or resolution needed
                Ok((input_op, columns))
            } else {
                let operator = Box::new(
                    ProjectOperator::with_store(
                        input_op,
                        projections,
                        output_types,
                        Arc::clone(&self.store) as Arc<dyn GraphStore>,
                    )
                    .with_transaction_context(self.viewing_epoch, self.transaction_id)
                    .with_session_context(self.session_context.clone()),
                );
                Ok((operator, columns))
            }
        }
    }

    /// Plans a project operator (for WITH clause).
    pub(super) fn plan_project(
        &self,
        project: &crate::query::plan::ProjectOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Handle Empty input specially (standalone WITH like: WITH [1,2,3] AS nums)
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(project.input.as_ref(), LogicalOperator::Empty) {
                // Create a single-row operator for projecting literals
                let single_row_op: Box<dyn Operator> = Box::new(
                    obrain_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                (single_row_op, Vec::new())
            } else {
                self.plan_operator(&project.input)?
            };

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Build projections and new column names
        let capacity = if project.pass_through_input {
            input_columns.len() + project.projections.len()
        } else {
            project.projections.len()
        };
        let mut projections = Vec::with_capacity(capacity);
        let mut output_types = Vec::with_capacity(capacity);
        let mut output_columns = Vec::with_capacity(capacity);

        // When pass_through_input is set (e.g. LET clause), first pass through
        // all existing input columns so they remain accessible to downstream
        // operators. The explicit projections are then appended as new columns.
        if project.pass_through_input {
            for (idx, col_name) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(idx));
                output_types.push(LogicalType::Any);
                output_columns.push(col_name.clone());
            }
        }

        for projection in &project.projections {
            // Determine the output column name (alias or expression string)
            let col_name = projection
                .alias
                .clone()
                .unwrap_or_else(|| expression_to_string(&projection.expression));

            match &projection.expression {
                LogicalExpression::Variable(name) => {
                    let col_idx = *variable_columns.get(name).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", name))
                    })?;
                    projections.push(ProjectExpr::Column(col_idx));
                    output_types.push(LogicalType::Node);
                    // Propagate scalar/edge status from input variable to output alias
                    if self.scalar_columns.borrow().contains(name) {
                        self.scalar_columns.borrow_mut().insert(col_name.clone());
                    } else if self.edge_columns.borrow().contains(name) {
                        self.edge_columns.borrow_mut().insert(col_name.clone());
                    }
                }
                LogicalExpression::Property { variable, property } => {
                    let col_idx = *variable_columns.get(variable).ok_or_else(|| {
                        Error::Internal(format!("Variable '{}' not found in input", variable))
                    })?;
                    projections.push(ProjectExpr::PropertyAccess {
                        column: col_idx,
                        property: property.clone(),
                    });
                    output_types.push(LogicalType::Any);
                    // Property access produces a scalar value
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
                LogicalExpression::Literal(value) => {
                    projections.push(ProjectExpr::Constant(value.clone()));
                    output_types.push(value_to_logical_type(value));
                    // Literals are scalar values
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
                _ => {
                    // For complex expressions, use full expression evaluation
                    let filter_expr = self.convert_expression(&projection.expression)?;
                    projections.push(ProjectExpr::Expression {
                        expr: filter_expr,
                        variable_columns: variable_columns.clone(),
                    });
                    output_types.push(LogicalType::Any);
                    // Expression results are scalar values
                    self.scalar_columns.borrow_mut().insert(col_name.clone());
                }
            }

            output_columns.push(col_name);
        }

        let operator = Box::new(
            ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone()),
        );

        Ok((operator, output_columns))
    }

    /// Plans a LIMIT operator.
    pub(super) fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&limit.input)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(crate::query::planner::common::build_limit(
            input_op,
            columns,
            limit.count.value(),
            schema,
        ))
    }

    /// Plans a SKIP operator.
    pub(super) fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let schema = self.derive_schema_from_columns(&columns);
        Ok(crate::query::planner::common::build_skip(
            input_op,
            columns,
            skip.count.value(),
            schema,
        ))
    }

    /// Plans a SORT (ORDER BY) operator.
    ///
    /// When Sort wraps a Return (e.g. `RETURN p.name ORDER BY p.age`), ORDER BY
    /// may reference entity variables that Return has already projected away. In
    /// that case we inject a property projection BEFORE the Return so the sort
    /// key is available in the output columns.
    pub(super) fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check if we need pre-Return property projections. This is necessary
        // when ORDER BY references a property on an entity variable (e.g. p.age)
        // that is not included in the RETURN clause (e.g. RETURN p.name).
        let needs_pre_return = if let LogicalOperator::Return(ret) = sort.input.as_ref() {
            sort.keys.iter().any(|key| {
                if let LogicalExpression::Property { variable, .. } = &key.expression {
                    // Check if the Return items produce a column matching this variable
                    !ret.items.iter().any(|item| {
                        item.alias.as_deref() == Some(variable)
                            || matches!(
                                &item.expression,
                                LogicalExpression::Variable(v) if v == variable
                            )
                    })
                } else {
                    false
                }
            })
        } else {
            false
        };

        // Number of extra sort-key columns appended after Return items
        let mut sort_extra_count: usize = 0;

        let (mut input_op, input_columns) = if needs_pre_return {
            // Plan the Return's input, then build a combined projection that
            // outputs both RETURN items and ORDER BY sort-key properties.
            let LogicalOperator::Return(ret) = sort.input.as_ref() else {
                unreachable!()
            };
            let (inner_op, inner_columns) = self.plan_operator(&ret.input)?;
            let inner_vars: HashMap<String, usize> = inner_columns
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect();

            // Collect ORDER BY property keys that need extra projection
            let mut extra_items: Vec<(String, String, String)> = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for key in &sort.keys {
                if let LogicalExpression::Property { variable, property } = &key.expression
                    && inner_vars.contains_key(variable)
                {
                    let col_name = format!("{}_{}", variable, property);
                    if seen.insert(col_name.clone()) {
                        extra_items.push((variable.clone(), property.clone(), col_name));
                    }
                }
            }

            // Build an augmented ReturnOp that includes the extra sort-key items
            let mut augmented_items = ret.items.clone();
            let mut extra_columns = Vec::new();
            for (variable, property, col_name) in &extra_items {
                augmented_items.push(crate::query::plan::ReturnItem {
                    expression: LogicalExpression::Property {
                        variable: variable.clone(),
                        property: property.clone(),
                    },
                    alias: Some(col_name.clone()),
                });
                extra_columns.push(col_name.clone());
            }

            let augmented_ret = crate::query::plan::ReturnOp {
                items: augmented_items,
                distinct: ret.distinct,
                input: ret.input.clone(),
            };

            // Plan the augmented Return with the original inner operator
            let (op, columns) =
                self.plan_return_with_input(&augmented_ret, inner_op, inner_columns)?;
            sort_extra_count = extra_columns.len();
            (op, columns)
        } else {
            self.plan_operator(&sort.input)?
        };

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect property expressions that need to be projected before sorting
        let mut property_projections: Vec<(String, String, String)> = Vec::new();
        let mut next_col_idx = input_columns.len();

        // Collect cognitive UDF expressions that need projection
        #[cfg(feature = "cognitive")]
        let mut cognitive_projections: Vec<(String, LogicalExpression)> = Vec::new();

        for key in &sort.keys {
            if let LogicalExpression::Property { variable, property } = &key.expression {
                let col_name = format!("{}_{}", variable, property);
                if !variable_columns.contains_key(&col_name) {
                    property_projections.push((
                        variable.clone(),
                        property.clone(),
                        col_name.clone(),
                    ));
                    variable_columns.insert(col_name, next_col_idx);
                    next_col_idx += 1;
                }
            }

            // Detect cognitive UDF calls (e.g., obrain.energy(n)) and project them
            #[cfg(feature = "cognitive")]
            if let LogicalExpression::FunctionCall { name, .. } = &key.expression {
                let col_name = name.replace('.', "_");
                if !variable_columns.contains_key(&col_name) {
                    cognitive_projections.push((col_name.clone(), key.expression.clone()));
                    variable_columns.insert(col_name, next_col_idx);
                    next_col_idx += 1;
                }
            }
        }

        // Track output columns
        let mut output_columns = input_columns.clone();

        // If we have property expressions, add a projection to materialize them
        if !property_projections.is_empty() {
            let mut projections = Vec::new();
            let mut output_types = Vec::new();

            // First, pass through all existing columns (use Node type to preserve node IDs
            // for subsequent property access - nodes need VectorData::NodeId for get_node_id())
            for (i, _) in input_columns.iter().enumerate() {
                projections.push(ProjectExpr::Column(i));
                output_types.push(LogicalType::Node);
            }

            // Then add property access projections
            for (variable, property, col_name) in &property_projections {
                let source_col = *variable_columns.get(variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for ORDER BY property projection",
                        variable
                    ))
                })?;
                projections.push(ProjectExpr::PropertyAccess {
                    column: source_col,
                    property: property.clone(),
                });
                output_types.push(LogicalType::Any);
                output_columns.push(col_name.clone());
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
        }

        // Project cognitive UDF expressions (e.g., obrain.energy(n)) as extra columns
        #[cfg(feature = "cognitive")]
        if !cognitive_projections.is_empty() {
            let mut projections = Vec::new();
            let mut proj_types = Vec::new();

            // Pass through all existing columns
            for i in 0..output_columns.len() {
                projections.push(ProjectExpr::Column(i));
                proj_types.push(LogicalType::Any);
            }

            // Add cognitive expression columns
            for (col_name, expr) in &cognitive_projections {
                let filter_expr = self.convert_expression(expr)?;
                projections.push(ProjectExpr::Expression {
                    expr: filter_expr,
                    variable_columns: variable_columns.clone(),
                });
                proj_types.push(LogicalType::Float64);
                output_columns.push(col_name.clone());
            }

            input_op = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    projections,
                    proj_types,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
        }

        // Convert logical sort keys to physical sort keys
        let physical_keys: Vec<PhysicalSortKey> = sort
            .keys
            .iter()
            .map(|key| {
                let col_idx = self
                    .resolve_sort_expression_with_properties(&key.expression, &variable_columns)?;
                Ok(PhysicalSortKey {
                    column: col_idx,
                    direction: match key.order {
                        SortOrder::Ascending => SortDirection::Ascending,
                        SortOrder::Descending => SortDirection::Descending,
                    },
                    null_order: match key.nulls {
                        Some(crate::query::plan::NullsOrdering::First) => NullOrder::NullsFirst,
                        Some(crate::query::plan::NullsOrdering::Last) => NullOrder::NullsLast,
                        None => NullOrder::NullsLast, // default
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let mut operator: Box<dyn Operator> =
            Box::new(SortOperator::new(input_op, physical_keys, output_schema));

        // Strip extra sort-key columns that were injected for ORDER BY resolution
        if sort_extra_count > 0 {
            let keep_count = output_columns.len() - sort_extra_count;
            let strip_projections: Vec<ProjectExpr> =
                (0..keep_count).map(ProjectExpr::Column).collect();
            let strip_types: Vec<LogicalType> = (0..keep_count).map(|_| LogicalType::Any).collect();
            operator = Box::new(ProjectOperator::new(
                operator,
                strip_projections,
                strip_types,
            ));
            output_columns.truncate(keep_count);
        }

        Ok((operator, output_columns))
    }

    /// Resolves a sort expression to a column index, using projected property columns.
    pub(super) fn resolve_sort_expression_with_properties(
        &self,
        expr: &LogicalExpression,
        variable_columns: &HashMap<String, usize>,
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => {
                variable_columns.get(name).copied().ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for ORDER BY", name))
                })
            }
            LogicalExpression::Property { variable, property } => {
                // Look up the projected property column (e.g., "p_age" for p.age)
                let col_name = format!("{}_{}", variable, property);
                variable_columns.get(&col_name).copied().ok_or_else(|| {
                    Error::Internal(format!(
                        "Property column '{}' not found for ORDER BY (from {}.{})",
                        col_name, variable, property
                    ))
                })
            }
            LogicalExpression::FunctionCall { name, .. } => {
                // Look for a projected cognitive/function column
                let col_name = name.replace('.', "_");
                variable_columns.get(&col_name).copied().ok_or_else(|| {
                    Error::Internal(format!(
                        "Function column '{}' not found for ORDER BY (from {}()). \
                         Cognitive UDF columns are auto-projected when the cognitive feature is enabled.",
                        col_name, name
                    ))
                })
            }
            _ => Err(Error::Internal(format!(
                "Unsupported ORDER BY expression: {:?}",
                expr
            ))),
        }
    }

    /// Derives a schema from column names (uses Any type to handle all value types).
    pub(super) fn derive_schema_from_columns(&self, columns: &[String]) -> Vec<LogicalType> {
        columns.iter().map(|_| LogicalType::Any).collect()
    }
}
