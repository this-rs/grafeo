//! Projection, RETURN, sort, limit, and skip planning.

use super::*;

impl super::Planner {
    /// Plans a RETURN clause.
    pub(super) fn plan_return(&self, ret: &ReturnOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator
        let (input_op, input_columns) = self.plan_operator(&ret.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Extract column names from return items
        let columns: Vec<String> = ret
            .items
            .iter()
            .map(|item| {
                item.alias.clone().unwrap_or_else(|| {
                    // Generate a default name from the expression
                    expression_to_string(&item.expression)
                })
            })
            .collect();

        // Check if we need a project operator (for property access or expression evaluation)
        let needs_project = ret
            .items
            .iter()
            .any(|item| !matches!(&item.expression, LogicalExpression::Variable(_)));

        if needs_project {
            // Build project expressions
            let mut projections = Vec::with_capacity(ret.items.len());
            let mut output_types = Vec::with_capacity(ret.items.len());

            for item in &ret.items {
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
                                // length(p) returns the path length
                                // For shortestPath results, the path column already contains the length
                                if args.len() != 1 {
                                    return Err(Error::Internal(
                                        "length() requires exactly one argument".to_string(),
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
                                    // Pass through the column value directly
                                    projections.push(ProjectExpr::Column(col_idx));
                                    output_types.push(LogicalType::Int64);
                                } else {
                                    return Err(Error::Internal(
                                        "length() argument must be a variable".to_string(),
                                    ));
                                }
                            }
                            "nodes" | "edges" => {
                                // nodes(p) / edges(p) returns the list of nodes/edges in the path
                                if args.len() != 1 {
                                    return Err(Error::Internal(format!(
                                        "{}() requires exactly one argument",
                                        name
                                    )));
                                }
                                if let LogicalExpression::Variable(var_name) = &args[0] {
                                    let col_idx =
                                        *variable_columns.get(var_name).ok_or_else(|| {
                                            Error::Internal(format!(
                                                "Variable '{var_name}' not found in input",
                                            ))
                                        })?;
                                    projections.push(ProjectExpr::Column(col_idx));
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
                    | LogicalExpression::List(_) => {
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

            let operator = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store),
            ));

            Ok((operator, columns))
        } else {
            // Simple case: all return items are bare variables
            // Emit resolve variants for entity variables
            let mut projections = Vec::with_capacity(ret.items.len());
            let mut output_types = Vec::with_capacity(ret.items.len());

            for item in &ret.items {
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
                let operator = Box::new(ProjectOperator::with_store(
                    input_op,
                    projections,
                    output_types,
                    Arc::clone(&self.store),
                ));
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
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
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
        let mut projections = Vec::with_capacity(project.projections.len());
        let mut output_types = Vec::with_capacity(project.projections.len());
        let mut output_columns = Vec::with_capacity(project.projections.len());

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

        let operator = Box::new(ProjectOperator::with_store(
            input_op,
            projections,
            output_types,
            Arc::clone(&self.store),
        ));

        Ok((operator, output_columns))
    }

    /// Plans a LIMIT operator.
    pub(super) fn plan_limit(&self, limit: &LimitOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&limit.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(LimitOperator::new(input_op, limit.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SKIP operator.
    pub(super) fn plan_skip(&self, skip: &SkipOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&skip.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(SkipOperator::new(input_op, skip.count, output_schema));
        Ok((operator, columns))
    }

    /// Plans a SORT (ORDER BY) operator.
    pub(super) fn plan_sort(&self, sort: &SortOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (mut input_op, input_columns) = self.plan_operator(&sort.input)?;

        // Build variable to column index mapping
        let mut variable_columns: HashMap<String, usize> = input_columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Collect property expressions that need to be projected before sorting
        let mut property_projections: Vec<(String, String, String)> = Vec::new();
        let mut next_col_idx = input_columns.len();

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

            input_op = Box::new(ProjectOperator::with_store(
                input_op,
                projections,
                output_types,
                Arc::clone(&self.store),
            ));
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
                    null_order: NullOrder::NullsLast,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let operator = Box::new(SortOperator::new(input_op, physical_keys, output_schema));
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
