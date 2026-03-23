//! Mutation planning (CREATE, DELETE, SET, MERGE, CALL, labels).

use super::{
    AddLabelOp, AddLabelOperator, AntiJoinOp, Arc, CreateEdgeOp, CreateEdgeOperator, CreateNodeOp,
    CreateNodeOperator, DeleteEdgeOp, DeleteEdgeOperator, DeleteNodeOp, DeleteNodeOperator,
    Direction, Error, ExpandDirection, GraphStore, HashMap, LeftJoinOp, LogicalExpression,
    LogicalOperator, LogicalType, MergeConfig, MergeOp, MergeOperator, MergeRelationshipConfig,
    MergeRelationshipOp, MergeRelationshipOperator, Operator, ProjectExpr, ProjectOperator,
    PropertySource, RemoveLabelOp, RemoveLabelOperator, Result, SetPropertyOp, SetPropertyOperator,
    ShortestPathOp, ShortestPathOperator, UnaryOp, UnwindOp, UnwindOperator, Value,
};
#[cfg(feature = "algos")]
use super::{CallProcedureOp, GraphStoreMut, StaticResultOperator};

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

        // Convert properties: resolve variables/property access from input columns
        let properties: Vec<(String, PropertySource)> = create
            .properties
            .iter()
            .map(|(name, expr)| {
                let source = self.expression_to_property_source(expr, &columns)?;
                Ok((name.clone(), source))
            })
            .collect::<Result<Vec<_>>>()?;

        let output_schema = self.derive_schema_from_columns(&columns);

        let mut op = CreateNodeOperator::new(
            Arc::clone(&self.store),
            input_op,
            create.labels.clone(),
            properties,
            output_schema,
            output_column,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(ref validator) = self.validator {
            op = op.with_validator(Arc::clone(validator));
        }

        let operator = Box::new(op);
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
            self.edge_columns.borrow_mut().insert(v.clone());
            idx
        });

        // Convert properties: resolve variables/property access from input columns
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
        .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref tracker) = self.write_tracker {
            operator = operator.with_write_tracker(Arc::clone(tracker));
        }
        if let Some(col) = output_column {
            operator = operator.with_output_column(col);
        }
        if let Some(ref validator) = self.validator {
            operator = operator.with_validator(Arc::clone(validator));
        }

        let operator = Box::new(operator);

        Ok((operator, columns))
    }

    /// Plans a DELETE NODE operator.
    ///
    /// If the variable is tracked as an edge (via `edge_columns`), this
    /// automatically delegates to [`DeleteEdgeOperator`] instead.
    pub(super) fn plan_delete_node(
        &self,
        delete: &DeleteNodeOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&delete.input)?;

        let col_idx = columns
            .iter()
            .position(|c| c == &delete.variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Variable '{}' not found for delete",
                    delete.variable
                ))
            })?;

        // Preserve input columns so downstream RETURN/aggregate can reference
        // the deleted variable (e.g., DETACH DELETE n RETURN count(n)).
        let output_schema: Vec<LogicalType> = columns.iter().map(|_| LogicalType::Node).collect();
        let output_columns = columns.clone();

        // Auto-detect edge variables and use the correct operator
        let is_edge = self.edge_columns.borrow().contains(&delete.variable);

        if is_edge {
            let mut op =
                DeleteEdgeOperator::new(Arc::clone(&self.store), input_op, col_idx, output_schema)
                    .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            Ok((Box::new(op), output_columns))
        } else {
            let mut op = DeleteNodeOperator::new(
                Arc::clone(&self.store),
                input_op,
                col_idx,
                output_schema,
                delete.detach,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            Ok((Box::new(op), output_columns))
        }
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

        // Preserve input columns so downstream clauses can reference the
        // deleted variable (same pass-through pattern as delete_node).
        let output_schema: Vec<LogicalType> = columns.iter().map(|_| LogicalType::Node).collect();
        let output_columns = columns.clone();

        let mut op = DeleteEdgeOperator::new(
            Arc::clone(&self.store),
            input_op,
            edge_column,
            output_schema,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }

        Ok((Box::new(op), output_columns))
    }

    /// Plans a LEFT JOIN operator (for OPTIONAL MATCH).
    pub(super) fn plan_left_join(
        &self,
        left_join: &LeftJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&left_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&left_join.right)?;
        let schema_fn = |cols: &[String]| self.derive_schema_from_columns(cols);
        Ok(super::common::build_left_join(
            left_op,
            right_op,
            &left_columns,
            &right_columns,
            schema_fn,
        ))
    }

    /// Plans an ANTI JOIN operator (for WHERE NOT EXISTS patterns).
    pub(super) fn plan_anti_join(
        &self,
        anti_join: &AntiJoinOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&anti_join.left)?;
        let (right_op, right_columns) = self.plan_operator(&anti_join.right)?;
        let schema = self.derive_schema_from_columns(&left_columns);
        Ok(super::common::build_anti_join(
            left_op,
            right_op,
            left_columns,
            &right_columns,
            schema,
        ))
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
                let project_op: Box<dyn Operator> = Box::new(
                    ProjectOperator::with_store(
                        single_row_op,
                        vec![ProjectExpr::Expression {
                            expr: literal_list,
                            variable_columns: HashMap::new(),
                        }],
                        vec![LogicalType::Any],
                        Arc::clone(&self.store) as Arc<dyn GraphStore>,
                    )
                    .with_transaction_context(self.viewing_epoch, self.transaction_id)
                    .with_session_context(self.session_context.clone()),
                );

                (project_op, vec!["__list__".to_string()])
            } else {
                self.plan_operator(&unwind.input)?
            };

        // The UNWIND expression should be a list - we need to find/evaluate it
        // Handle variable references, property access, and literal lists

        // Find if the expression references an existing column that is itself a list
        let list_col_idx = match &unwind.expression {
            LogicalExpression::Variable(var) => input_columns.iter().position(|c| c == var),
            LogicalExpression::List(_) | LogicalExpression::Literal(_) => {
                // Literal list expression - needs to be added as a column
                None
            }
            _ => None,
        };

        // When the expression needs runtime evaluation (property access, literal list, etc.),
        // wrap input in a ProjectOperator that computes the list as an extra column.
        let (final_input_op, final_input_columns, col_idx) = if let Some(idx) = list_col_idx {
            (input_op, input_columns, idx)
        } else if matches!(
            &unwind.expression,
            LogicalExpression::List(_)
                | LogicalExpression::Literal(Value::List(_))
                | LogicalExpression::Literal(Value::Vector(_))
                | LogicalExpression::Property { .. }
        ) {
            // Wrap input in a ProjectOperator that adds the list as an extra column
            let literal_list = self.convert_expression(&unwind.expression)?;
            let mut proj_exprs: Vec<ProjectExpr> =
                (0..input_columns.len()).map(ProjectExpr::Column).collect();
            let var_cols: HashMap<String, usize> = input_columns
                .iter()
                .enumerate()
                .map(|(i, c)| (c.clone(), i))
                .collect();
            proj_exprs.push(ProjectExpr::Expression {
                expr: literal_list,
                variable_columns: var_cols,
            });
            let mut proj_schema = self.derive_schema_from_columns(&input_columns);
            proj_schema.push(LogicalType::Any);
            let project_op: Box<dyn Operator> = Box::new(
                ProjectOperator::with_store(
                    input_op,
                    proj_exprs,
                    proj_schema,
                    Arc::clone(&self.store) as Arc<dyn GraphStore>,
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()),
            );
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
        let (input_op, mut columns) = if matches!(merge.input.as_ref(), LogicalOperator::Empty) {
            (None, Vec::new())
        } else {
            let (op, cols) = self.plan_operator(&merge.input)?;
            (Some(op), cols)
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

        // Column index for the merged node ID in the output
        let output_column = columns.len();
        columns.push(merge.variable.clone());

        // Build output schema
        let output_schema: Vec<LogicalType> = columns.iter().map(|_| LogicalType::Node).collect();

        let mut merge_op = MergeOperator::new(
            Arc::clone(&self.store),
            input_op,
            MergeConfig {
                variable: merge.variable.clone(),
                labels: merge.labels.clone(),
                match_properties,
                on_create_properties,
                on_match_properties,
                output_schema,
                output_column,
            },
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref validator) = self.validator {
            merge_op = merge_op.with_validator(Arc::clone(validator));
        }

        let operator: Box<dyn Operator> = Box::new(merge_op);

        Ok((operator, columns))
    }

    /// Plans a MERGE RELATIONSHIP operator.
    pub(super) fn plan_merge_relationship(
        &self,
        merge_rel: &MergeRelationshipOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, mut columns) = self.plan_operator(&merge_rel.input)?;

        // Find source and target node columns
        let source_column = columns
            .iter()
            .position(|c| c == &merge_rel.source_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found for MERGE relationship",
                    merge_rel.source_variable
                ))
            })?;

        let target_column = columns
            .iter()
            .position(|c| c == &merge_rel.target_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Target variable '{}' not found for MERGE relationship",
                    merge_rel.target_variable
                ))
            })?;

        // Convert match properties from LogicalExpression to Value
        let match_properties: Vec<(String, grafeo_common::types::Value)> = merge_rel
            .match_properties
            .iter()
            .filter_map(|(name, expr)| {
                if let LogicalExpression::Literal(v) = expr {
                    Some((name.clone(), v.clone()))
                } else {
                    None
                }
            })
            .collect();

        let on_create_properties: Vec<(String, grafeo_common::types::Value)> = merge_rel
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

        let on_match_properties: Vec<(String, grafeo_common::types::Value)> = merge_rel
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

        // Add the edge variable to output columns and track it as an edge
        let edge_output_column = columns.len();
        columns.push(merge_rel.variable.clone());
        self.edge_columns
            .borrow_mut()
            .insert(merge_rel.variable.clone());

        // Build output schema: input columns + edge column
        let mut output_schema: Vec<LogicalType> =
            columns.iter().map(|_| LogicalType::Node).collect();
        output_schema[edge_output_column] = LogicalType::Edge;

        let config = MergeRelationshipConfig {
            source_column,
            target_column,
            edge_type: merge_rel.edge_type.clone(),
            match_properties,
            on_create_properties,
            on_match_properties,
            output_schema,
            edge_output_column,
        };

        let mut merge_rel_op =
            MergeRelationshipOperator::new(Arc::clone(&self.store), input_op, config)
                .with_transaction_context(self.viewing_epoch, self.transaction_id);

        if let Some(ref validator) = self.validator {
            merge_rel_op = merge_rel_op.with_validator(Arc::clone(validator));
        }

        let operator: Box<dyn Operator> = Box::new(merge_rel_op);

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
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                input_op,
                source_column,
                target_column,
                sp.edge_types.clone(),
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

        // Catalog introspection procedures
        if let Some(result) = self.plan_catalog_procedure(&resolved_name) {
            return self.plan_static_result(result, &call.yield_items);
        }

        // Check user-defined procedures first
        if let Some(catalog) = &self.catalog {
            let proc_name = if call.name.len() == 1 {
                &call.name[0]
            } else {
                // For dotted names, try the last segment as procedure name
                call.name.last().expect("name has at least one segment")
            };
            if let Some(proc_def) = catalog.get_procedure(proc_name) {
                return self.plan_user_procedure(call, &proc_def);
            }
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
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                algorithm,
                params,
                yield_columns,
                canonical_columns,
            ),
        );

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

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

        // Static result outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Plans a user-defined procedure call.
    #[cfg(feature = "algos")]
    fn plan_user_procedure(
        &self,
        call: &CallProcedureOp,
        proc_def: &crate::catalog::ProcedureDefinition,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        use crate::query::executor::user_procedure::{ProcedureContext, UserProcedureOperator};

        // Validate argument count
        if call.arguments.len() != proc_def.params.len() {
            return Err(Error::Internal(format!(
                "Procedure '{}' expects {} arguments, got {}",
                proc_def.name,
                proc_def.params.len(),
                call.arguments.len()
            )));
        }

        // Evaluate arguments to values
        let mut arg_values = Vec::new();
        for arg in &call.arguments {
            let val = crate::query::planner::eval_constant_expression(arg)?;
            arg_values.push(val);
        }

        // Build parameter map: param_name -> value
        let mut param_map = std::collections::HashMap::new();
        for (param, value) in proc_def.params.iter().zip(arg_values) {
            param_map.insert(param.0.clone(), value);
        }

        // Determine output columns
        let return_columns: Vec<String> = proc_def.returns.iter().map(|r| r.0.clone()).collect();

        let output_columns = if let Some(yield_items) = &call.yield_items {
            yield_items
                .iter()
                .map(|item| {
                    item.alias
                        .clone()
                        .unwrap_or_else(|| item.field_name.clone())
                })
                .collect()
        } else {
            return_columns.clone()
        };

        let yield_columns = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| item.field_name.clone())
                .collect::<Vec<_>>()
        });

        let operator = Box::new(UserProcedureOperator::new(
            proc_def.body.clone(),
            param_map,
            return_columns,
            yield_columns,
            ProcedureContext {
                store: Arc::clone(&self.store) as Arc<dyn GraphStoreMut>,
                transaction_manager: self.transaction_manager.clone(),
                transaction_id: self.transaction_id,
                viewing_epoch: self.viewing_epoch,
                catalog: self.catalog.clone(),
            },
        ));

        // Procedure outputs are scalar values, not node/edge IDs
        for col in &output_columns {
            self.scalar_columns.borrow_mut().insert(col.clone());
        }

        Ok((operator, output_columns))
    }

    /// Resolves a catalog introspection procedure by name.
    ///
    /// Supports `db.labels`, `db.relationshipTypes`, `db.propertyKeys`,
    /// `db.schema`, and `db.indexes` (also with `grafeo.` prefix).
    #[cfg(feature = "algos")]
    fn plan_catalog_procedure(
        &self,
        name: &str,
    ) -> Option<grafeo_adapters::plugins::AlgorithmResult> {
        use grafeo_adapters::plugins::AlgorithmResult;
        use grafeo_common::types::Value;

        match name {
            "db.labels" | "grafeo.labels" => {
                let labels = self.store.all_labels();
                let mut result = AlgorithmResult::new(vec!["label".to_string()]);
                for label in labels {
                    result.rows.push(vec![Value::String(label.into())]);
                }
                Some(result)
            }
            "db.relationshipTypes" | "grafeo.relationshipTypes" => {
                let types = self.store.all_edge_types();
                let mut result = AlgorithmResult::new(vec!["relationshipType".to_string()]);
                for t in types {
                    result.rows.push(vec![Value::String(t.into())]);
                }
                Some(result)
            }
            "db.propertyKeys" | "grafeo.propertyKeys" => {
                let keys = self.store.all_property_keys();
                let mut result = AlgorithmResult::new(vec!["propertyKey".to_string()]);
                for key in keys {
                    result.rows.push(vec![Value::String(key.into())]);
                }
                Some(result)
            }
            _ => None,
        }
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

        let mut op = AddLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            add_label.labels.clone(),
            output_schema,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }

        Ok((Box::new(op), output_columns))
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

        let mut op = RemoveLabelOperator::new(
            Arc::clone(&self.store),
            input_op,
            node_column,
            remove_label.labels.clone(),
            output_schema,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if let Some(ref tracker) = self.write_tracker {
            op = op.with_write_tracker(Arc::clone(tracker));
        }

        Ok((Box::new(op), output_columns))
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

        // Determine if this is a node or edge using tracked edge columns
        let is_edge = set_prop.is_edge || self.edge_columns.borrow().contains(&set_prop.variable);
        let operator: Box<dyn Operator> = if is_edge {
            let mut op = SetPropertyOperator::new_for_edge(
                Arc::clone(&self.store),
                input_op,
                entity_column,
                properties,
                output_schema,
            )
            .with_replace(set_prop.replace)
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            if let Some(ref validator) = self.validator {
                op = op.with_validator(Arc::clone(validator));
            }
            Box::new(op)
        } else {
            let mut op = SetPropertyOperator::new_for_node(
                Arc::clone(&self.store),
                input_op,
                entity_column,
                properties,
                output_schema,
            )
            .with_replace(set_prop.replace)
            .with_transaction_context(self.viewing_epoch, self.transaction_id);
            if let Some(ref tracker) = self.write_tracker {
                op = op.with_write_tracker(Arc::clone(tracker));
            }
            if let Some(ref validator) = self.validator {
                op = op.with_validator(Arc::clone(validator));
            }
            Box::new(op)
        };

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
    /// Recursively folds literals, unary operators, lists, and known function calls
    /// (like `vector()`) into concrete values. Returns `None` if the expression
    /// contains non-constant parts (variables, property accesses, etc.).
    pub(super) fn try_fold_expression(expr: &LogicalExpression) -> Option<Value> {
        match expr {
            LogicalExpression::Literal(v) => Some(v.clone()),
            LogicalExpression::List(items) => {
                let values: Option<Vec<Value>> =
                    items.iter().map(Self::try_fold_expression).collect();
                Some(Value::List(values?.into()))
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
                    "date" | "todate" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Date::parse(&s).map(Value::Date)
                            }
                            _ => None,
                        }
                    }
                    "time" | "totime" | "local_time" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                grafeo_common::types::Time::parse(&s).map(Value::Time)
                            }
                            _ => None,
                        }
                    }
                    "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                        if args.len() != 1 {
                            return None;
                        }
                        let val = Self::try_fold_expression(&args[0])?;
                        match val {
                            Value::String(s) => {
                                if let Some(d) = grafeo_common::types::Date::parse(&s) {
                                    return Some(Value::Timestamp(d.to_timestamp()));
                                }
                                if let Some(pos) = s.find('T') {
                                    let (date_part, time_part) = (&s[..pos], &s[pos + 1..]);
                                    if let (Some(d), Some(t)) = (
                                        grafeo_common::types::Date::parse(date_part),
                                        grafeo_common::types::Time::parse(time_part),
                                    ) {
                                        return Some(Value::Timestamp(
                                            grafeo_common::types::Timestamp::from_date_time(d, t),
                                        ));
                                    }
                                }
                                None
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            LogicalExpression::Map(entries) => {
                let folded: Option<Vec<(String, Value)>> = entries
                    .iter()
                    .map(|(k, v)| Self::try_fold_expression(v).map(|val| (k.clone(), val)))
                    .collect();
                let folded = folded?;
                let map: std::collections::BTreeMap<grafeo_common::types::PropertyKey, Value> =
                    folded
                        .into_iter()
                        .map(|(k, v)| (grafeo_common::types::PropertyKey::from(k), v))
                        .collect();
                Some(Value::Map(std::sync::Arc::new(map)))
            }
            LogicalExpression::Unary { op, operand } => {
                let value = Self::try_fold_expression(operand)?;
                match op {
                    UnaryOp::Neg => match value {
                        Value::Int64(n) => Some(Value::Int64(-n)),
                        Value::Float64(f) => Some(Value::Float64(-f)),
                        _ => None,
                    },
                    UnaryOp::Not => match value {
                        Value::Bool(b) => Some(Value::Bool(!b)),
                        _ => None,
                    },
                    UnaryOp::IsNull | UnaryOp::IsNotNull => None,
                }
            }
            _ => None,
        }
    }
}
