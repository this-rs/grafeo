//! Mutation planning (CREATE, DELETE, SET, MERGE, CALL, labels).

use super::*;

impl super::Planner {
    /// Plans a CREATE NODE operator.
    pub(super) fn plan_create_node(
        &self,
        create: &CreateNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan input if present
        let (input_op, mut columns) = if let Some(ref input) = create.input {
            let (op, cols) = self.plan_operator(input)?;
            (Some(op), cols)
        } else {
            (None, vec![])
        };

        // Output column for the created node
        let output_column = columns.len();
        columns.push(create.variable.clone());

        // Convert properties — resolve variables/property access from input columns
        let properties: Vec<(String, PropertySource)> = create
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = self.expression_to_property_source(expr, &columns)?;
                Ok((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator = Box::new(
            CreateNodeOperator::new(
                Arc::clone(&self.store),
                input_op,
                create.labels.clone(),
                properties,
                output_schema,
                output_column,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, columns))
    }

    /// Plans a CREATE EDGE operator.
    pub(super) fn plan_create_edge(
        &self,
        create: &CreateEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, mut columns) = self.plan_operator(&create.input)?;

        // Find source and target columns
        let from_column = columns
            .iter()
            .position(|c| c == &create.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found",
                    create.from_variable
                ))
            })?;

        let to_column = columns
            .iter()
            .position(|c| c == &create.to_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found",
                    create.to_variable
                ))
            })?;

        // Output column for the created edge (if named)
        let output_column = create.variable.as_ref().map(|v| {
            let idx = columns.len();
            columns.push(v.clone());
            idx
        });

        // Convert properties — resolve variables/property access from input columns
        let properties: Vec<(String, PropertySource)> = create
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = self.expression_to_property_source(expr, &columns)?;
                Ok((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&columns);

        let mut operator = CreateEdgeOperator::new(
            Arc::clone(&self.store),
            input_op,
            from_column,
            to_column,
            create.edge_type.clone(),
            output_schema,
        )
        .with_properties(properties)
        .with_tx_context(self.viewing_epoch, self.tx_id);

        if let Some(col) = output_column {
            operator = operator.with_output_column(col);
        }

        let operator = Box::new(operator);

        Ok((operator, columns))
    }

    /// Plans a DELETE NODE operator.
    pub(super) fn plan_delete_node(
        &self,
        delete: &DeleteNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let node_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Output schema for delete count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["deleted_count".to_string()];

        let operator = Box::new(
            DeleteNodeOperator::new(
                Arc::clone(&self.store),
                input_op,
                node_column,
                output_schema,
                delete.detach, // DETACH DELETE deletes connected edges first
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, output_columns))
    }

    /// Plans a DELETE EDGE operator.
    pub(super) fn plan_delete_edge(
        &self,
        delete: &DeleteEdgeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let edge_column = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Output schema for delete count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["deleted_count".to_string()];

        let operator = Box::new(
            DeleteEdgeOperator::new(
                Arc::clone(&self.store),
                input_op,
                edge_column,
                output_schema,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id),
        );

        Ok((operator, output_columns))
    }

    /// Plans a LEFT JOIN operator (for OPTIONAL MATCH).
    pub(super) fn plan_left_join(
        &self,
        left_join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&left_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&left_join.right)?;

        // Build combined output columns (left + right)
        let mut columns = left_columns.clone();
        columns.extend(right_columns.clone());

        // Find common variables between left and right for join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();

        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            PhysicalJoinType::Left,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Plans an ANTI JOIN operator (for WHERE NOT EXISTS patterns).
    pub(super) fn plan_anti_join(
        &self,
        anti_join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&anti_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&anti_join.right)?;

        // Anti-join only keeps left columns (filters out matching rows)
        let columns = left_columns.clone();

        // Find common variables between left and right for join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();

        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&columns);

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            PhysicalJoinType::Anti,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Plans an unwind operator.
    pub(super) fn plan_unwind(
        &self,
        unwind: &UnwindOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        // Handle Empty specially - use a single-row operator
        let (input_op, input_columns): (Box<dyn Operator>, Vec<String>) =
            if matches!(&*unwind.input, LogicalOperator::Empty) {
                // For UNWIND without prior MATCH, create a single-row input
                // We need an operator that produces one row with the list to unwind
                // For now, use EmptyScan which produces no rows - we'll handle the literal
                // list in the unwind operator itself
                let literal_list = self.convert_expression(&unwind.expression)?;

                // Create a project operator that produces a single row with the list
                let single_row_op: Box<dyn Operator> = Box::new(
                    grafeo_core::execution::operators::single_row::SingleRowOperator::new(),
                );
                let project_op: Box<dyn Operator> = Box::new(ProjectOperator::with_store(
                    single_row_op,
                    vec![ProjectExpr::Expression {
                        expr: literal_list,
                        variable_columns: HashMap::new(),
                    }],
                    vec![LogicalType::Any],
                    Arc::clone(&self.store),
                ));

                (project_op, vec!["__list__".to_string()])
            } else {
                self.plan_operator(&unwind.input)?
            };

        // The UNWIND expression should be a list - we need to find/evaluate it
        // Handle variable references, property access, and literal lists

        // Find if the expression references an existing column (like a list property)
        let list_col_idx = match &unwind.expression {
            LogicalExpression::Variable(var) => input_columns.iter().position(|c| c == var),
            LogicalExpression::Property { variable, .. } => {
                input_columns.iter().position(|c| c == variable)
            }
            LogicalExpression::List(_) | LogicalExpression::Literal(_) => {
                // Literal list expression - needs to be added as a column
                None
            }
            _ => None,
        };

        // When the expression is a literal list (including parameter-substituted lists)
        // and there's prior input, add the list as an extra column via ProjectOperator.
        let (final_input_op, final_input_columns, col_idx) = if let Some(idx) = list_col_idx {
            (input_op, input_columns, idx)
        } else if matches!(
            &unwind.expression,
            LogicalExpression::List(_)
                | LogicalExpression::Literal(Value::List(_))
                | LogicalExpression::Literal(Value::Vector(_))
        ) {
            // Wrap input in a ProjectOperator that adds the list as an extra column
            let literal_list = self.convert_expression(&unwind.expression)?;
            let mut proj_exprs: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            proj_exprs.push(ProjectExpr::Expression {
                expr: literal_list,
                variable_columns: HashMap::new(),
            });
            let mut proj_schema = self.derive_schema_from_columns(&input_columns);
            proj_schema.push(LogicalType::Any);
            let project_op: Box<dyn Operator> = Box::new(ProjectOperator::with_store(
                input_op,
                proj_exprs,
                proj_schema,
                Arc::clone(&self.store),
            ));
            let list_col = input_columns.len();
            let mut cols = input_columns;
            cols.push("__unwind_list__".to_string());
            (project_op, cols, list_col)
        } else {
            // Fallback: assume column 0 contains the list
            (input_op, input_columns, 0)
        };

        // Build output columns: all input columns plus the new variable
        let mut columns = final_input_columns.clone();
        columns.push(unwind.variable.clone());

        // Mark the UNWIND variable as scalar (not a node/edge ID) so that
        // plan_return uses LogicalType::Any instead of Node for it.
        self.scalar_columns
            .borrow_mut()
            .insert(unwind.variable.clone());

        // Build output schema
        let mut output_schema = self.derive_schema_from_columns(&final_input_columns);
        output_schema.push(LogicalType::Any); // The unwound element type is dynamic

        // Add ORDINALITY column (1-based index) if requested
        let emit_ordinality = unwind.ordinality_var.is_some();
        if let Some(ref ord_var) = unwind.ordinality_var {
            columns.push(ord_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(ord_var.clone());
        }

        // Add OFFSET column (0-based index) if requested
        let emit_offset = unwind.offset_var.is_some();
        if let Some(ref off_var) = unwind.offset_var {
            columns.push(off_var.clone());
            output_schema.push(LogicalType::Int64);
            self.scalar_columns.borrow_mut().insert(off_var.clone());
        }

        let operator: Box<dyn Operator> = Box::new(UnwindOperator::new(
            final_input_op,
            col_idx,
            unwind.variable.clone(),
            output_schema,
            emit_ordinality,
            emit_offset,
        ));

        Ok((operator, columns))
    }

    /// Plans a MERGE operator.
    pub(super) fn plan_merge(&self, merge: &MergeOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator if present (skip if Empty)
        let mut columns = if matches!(merge.input.as_ref(), LogicalOperator::Empty) {
            Vec::new()
        } else {
            let (_input_op, cols) = self.plan_operator(&merge.input)?;
            cols
        };

        // Convert match properties from LogicalExpression to Value
        let match_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .match_properties
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None // Skip non-literal expressions for now
                }
            })
            .collect();

        // Convert ON CREATE properties
        let on_create_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .on_create
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Convert ON MATCH properties
        let on_match_properties: Vec<(String, grafeo_common::types::Value)> = merge
            .on_match
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Add the merged node variable to output columns
        columns.push(merge.variable.clone());

        let operator: Box<dyn Operator> = Box::new(MergeOperator::new(
            Arc::clone(&self.store),
            merge.variable.clone(),
            merge.labels.clone(),
            match_properties,
            on_create_properties,
            on_match_properties,
        ));

        Ok((operator, columns))
    }

    /// Plans a SHORTEST PATH operator.
    pub(super) fn plan_shortest_path(
        &self,
        sp: &ShortestPathOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator
        let (input_op, mut columns) = self.plan_operator(&sp.input)?;

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &sp.source_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for shortestPath",
                    sp.source_var
                ))
            })?;

        let target_column = columns
            .iter()
            .position(|c| c == &sp.target_var)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found for shortestPath",
                    sp.target_var
                ))
            })?;

        // Convert direction
        let direction = match sp.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        // Create the shortest path operator
        let operator: Box<dyn Operator> = Box::new(
            ShortestPathOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                target_column,
                sp.edge_type.clone(),
                direction,
            )
            .with_all_paths(sp.all_paths),
        );

        // Add path length column with the expected naming convention
        // The translator expects _path_length_{alias} format for length(p) calls
        let path_col_name = format!("_path_length_{}", sp.path_alias);
        columns.push(path_col_name.clone());

        // Mark path length as scalar so plan_return uses LogicalType::Any, not Node
        self.scalar_columns.borrow_mut().insert(path_col_name);

        Ok((operator, columns))
    }

    /// Plans a CALL procedure operator.
    #[cfg(feature = "algos")]
    pub(super) fn plan_call_procedure(
        &self,
        call: &CallProcedureOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::procedures::{self, BuiltinProcedures};

        static PROCEDURES: std::sync::OnceLock<BuiltinProcedures> = std::sync::OnceLock::new();
        let registry = PROCEDURES.get_or_init(BuiltinProcedures::new);

        // Special case: grafeo.procedures() lists all procedures
        let resolved_name = call.name.join(".");
        if resolved_name == "grafeo.procedures" || resolved_name == "procedures" {
            let result = procedures::procedures_result(registry);
            return self.plan_static_result(result, &call.yield_items);
        }

        // Look up the algorithm
        let algorithm = registry.get(&call.name).ok_or_else(|| {
            Error::Internal(format!(
                "Unknown procedure: '{}'. Use CALL grafeo.procedures() to list available procedures.",
                call.name.join(".")
            ))
        })?;

        // Evaluate arguments to Parameters
        let params = procedures::evaluate_arguments(&call.arguments, algorithm.parameters());

        // Canonical column names for this algorithm (user-facing names)
        let canonical_columns = procedures::output_columns_for_name(algorithm.as_ref());

        // Determine output columns from YIELD or algorithm defaults
        let yield_columns = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| (item.field_name.clone(), item.alias.clone()))
                .collect::<Vec<_>>()
        });

        let output_columns = if let Some(yield_cols) = &yield_columns {
            yield_cols
                .iter()
                .map(|(name, alias)| alias.clone().unwrap_or_else(|| name.clone()))
                .collect()
        } else {
            canonical_columns.clone()
        };

        let operator = Box::new(
            crate::query::executor::procedure_call::ProcedureCallOperator::new(
                Arc::clone(&self.store),
                algorithm,
                params,
                yield_columns,
                canonical_columns,
            ),
        );

        Ok((operator, output_columns))
    }

    /// Plans a static result set (e.g., from `grafeo.procedures()`).
    #[cfg(feature = "algos")]
    pub(super) fn plan_static_result(
        &self,
        result: grafeo_adapters::plugins::AlgorithmResult,
        yield_items: &Option<Vec<crate::query::plan::ProcedureYield>>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Determine output columns and column indices
        let (output_columns, column_indices) = if let Some(items) = yield_items {
            let mut cols = Vec::new();
            let mut indices = Vec::new();
            for item in items {
                let idx = result
                    .columns
                    .iter()
                    .position(|c| c == &item.field_name)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "YIELD column '{}' not found (available: {})",
                            item.field_name,
                            result.columns.join(", ")
                        ))
                    })?;
                indices.push(idx);
                cols.push(
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone()),
                );
            }
            (cols, indices)
        } else {
            let indices: Vec<usize> = (0..result.columns.len()).collect();
            (result.columns.clone(), indices)
        };

        let operator = Box::new(StaticResultOperator {
            rows: result.rows,
            column_indices,
            row_index: 0,
        });

        Ok((operator, output_columns))
    }

    /// Plans an ADD LABEL operator.
    pub(super) fn plan_add_label(
        &self,
        add_label: &AddLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&add_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &add_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for ADD LABEL",
                    add_label.variable
                ))
            })?;

        // Output schema for update count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["labels_added".to_string()];

        let operator = Box::new(AddLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            add_label.labels.clone(),
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Plans a REMOVE LABEL operator.
    pub(super) fn plan_remove_label(
        &self,
        remove_label: &RemoveLabelOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&remove_label.input)?;

        // Find the node column
        let node_column = columns
            .iter()
            .position(|c| c == &remove_label.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for REMOVE LABEL",
                    remove_label.variable
                ))
            })?;

        // Output schema for update count
        let output_schema = vec![LogicalType::Int64];
        let output_columns = vec!["labels_removed".to_string()];

        let operator = Box::new(RemoveLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            remove_label.labels.clone(),
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Plans a SET PROPERTY operator.
    pub(super) fn plan_set_property(
        &self,
        set_prop: &SetPropertyOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&set_prop.input)?;

        // Find the entity column (node or edge variable)
        let entity_column = columns
            .iter()
            .position(|c| c == &set_prop.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for SET",
                    set_prop.variable
                ))
            })?;

        // Convert properties to PropertySource
        let properties: Vec<(String, PropertySource)> = set_prop
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = self.expression_to_property_source(expr, &columns)?;
                Ok((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        // Output schema preserves input schema (passes through)
        let output_schema: Vec<LogicalType> = columns.iter().map(|_| LogicalType::Node).collect();
        let output_columns = columns.clone();

        // Determine if this is a node or edge (for now assume node, edge detection can be added later)
        let operator = Box::new(SetPropertyOperator::new_for_node(
            Arc::clone(&self.store),
            input_op,
            entity_column,
            properties,
            output_schema,
        ));

        Ok((operator, output_columns))
    }

    /// Converts a logical expression to a PropertySource.
    pub(super) fn expression_to_property_source(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<PropertySource> {
        match expr {
            LogicalExpression::Literal(value) => Ok(PropertySource::Constant(value.clone())),
            LogicalExpression::Variable(name) => {
                let col_idx = columns.iter().position(|c| c == name).ok_or_else(|| {
                    Error::Internal(format!("Variable '{}' not found for property source", name))
                })?;
                Ok(PropertySource::Column(col_idx))
            }
            LogicalExpression::Property { variable, property } => {
                let col_idx = columns.iter().position(|c| c == variable).ok_or_else(|| {
                    Error::Internal(format!(
                        "Variable '{}' not found for property access '{}.{}'",
                        variable, variable, property
                    ))
                })?;
                Ok(PropertySource::PropertyAccess {
                    column: col_idx,
                    property: property.clone(),
                })
            }
            LogicalExpression::Parameter(name) => {
                // Parameters should be resolved before planning
                // For now, treat as a placeholder
                Ok(PropertySource::Constant(
                    grafeo_common::types::Value::String(format!("${}", name).into()),
                ))
            }
            _ => {
                if let Some(value) = Self::try_fold_expression(expr) {
                    Ok(PropertySource::Constant(value))
                } else {
                    Err(Error::Internal(format!(
                        "Unsupported expression type for property source: {:?}",
                        expr
                    )))
                }
            }
        }
    }

    /// Tries to evaluate a constant expression at plan time.
    ///
    /// Recursively folds literals, lists, and known function calls (like `vector()`)
    /// into concrete values. Returns `None` if the expression contains non-constant
    /// parts (variables, property accesses, etc.).
    pub(super) fn try_fold_expression(expr: &LogicalExpression) -> Option<Value> {
        match expr {
            LogicalExpression::Literal(v) => Some(v.clone()),
            LogicalExpression::List(items) => {
                let values: Option<Vec<Value>> =
                    items.iter().map(Self::try_fold_expression).collect();
                let values = values?;
                // All-numeric lists become vectors (matches Python list[float] behavior)
                let all_numeric = !values.is_empty()
                    && values
                        .iter()
                        .all(|v| matches!(v, Value::Float64(_) | Value::Int64(_)));
                if all_numeric {
                    let floats: Vec<f32> = values
                        .iter()
                        .filter_map(|v| match v {
                            Value::Float64(f) => Some(*f as f32),
                            Value::Int64(i) => Some(*i as f32),
                            _ => None,
                        })
                        .collect();
                    Some(Value::Vector(floats.into()))
                } else {
                    Some(Value::List(values.into()))
                }
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                match name.to_lowercase().as_str() {
                    "vector" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::List(items) => {
                                let floats: Vec<f32> = items
                                    .iter()
                                    .filter_map(|v| match v {
                                        Value::Float64(f) => Some(*f as f32),
                                        Value::Int64(i) => Some(*i as f32),
                                        _ => None,
                                    })
                                    .collect();
                                if floats.len() == items.len() {
                                    Some(Value::Vector(floats.into()))
                                } else {
                                    None
                                }
                            }
                            // Already a vector (from all-numeric list folding)
                            Value::Vector(v) => Some(Value::Vector(v)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}
