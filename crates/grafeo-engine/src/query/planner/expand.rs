//! Relationship expansion and factorized chain planning.

use super::*;

impl super::Planner {
    /// Plans an expand operator.
    pub(super) fn plan_expand(
        &self,
        expand: &ExpandOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Plan the input operator first
        let (input_op, input_columns) = self.plan_operator(&expand.input)?;

        // Find the source column index
        let source_column = input_columns
            .iter()
            .position(|c| c == &expand.from_variable)
            .ok_or_else(|| {
                Error::Internal(format!(
                    "Source variable '{}' not found in input columns",
                    expand.from_variable
                ))
            })?;

        // Convert expand direction
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };

        // Check if this is a variable-length path
        let is_variable_length =
            expand.min_hops != 1 || expand.max_hops.is_none() || expand.max_hops != Some(1);

        let operator: Box<dyn Operator> = if is_variable_length {
            // Use VariableLengthExpandOperator for multi-hop paths
            let max_hops = expand.max_hops.unwrap_or(expand.min_hops + 10); // Default max if unlimited
            let mut expand_op = VariableLengthExpandOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                direction,
                expand.edge_type.clone(),
                expand.min_hops,
                max_hops,
            )
            .with_tx_context(self.viewing_epoch, self.tx_id);

            // If a path alias is set, enable path length and detail output
            if expand.path_alias.is_some() {
                expand_op = expand_op
                    .with_path_length_output()
                    .with_path_detail_output();
            }

            Box::new(expand_op)
        } else {
            // Use simple ExpandOperator for single-hop paths
            let expand_op = ExpandOperator::new(
                Arc::clone(&self.store),
                input_op,
                source_column,
                direction,
                expand.edge_type.clone(),
            )
            .with_tx_context(self.viewing_epoch, self.tx_id);
            Box::new(expand_op)
        };

        // Build output columns: [input_columns..., edge, target, (path_length)?]
        // Preserve all input columns and add edge + target to match ExpandOperator output
        let mut columns = input_columns;

        // Generate edge column name - use provided name or generate anonymous name
        let edge_col_name = expand.edge_variable.clone().unwrap_or_else(|| {
            let count = self.anon_edge_counter.get();
            self.anon_edge_counter.set(count + 1);
            format!("_anon_edge_{}", count)
        });
        self.edge_columns.borrow_mut().insert(edge_col_name.clone());
        columns.push(edge_col_name);

        columns.push(expand.to_variable.clone());

        // If a path alias is set, add columns for path length, nodes, and edges
        if let Some(ref path_alias) = expand.path_alias {
            columns.push(format!("_path_length_{}", path_alias));
            columns.push(format!("_path_nodes_{}", path_alias));
            columns.push(format!("_path_edges_{}", path_alias));
        }

        Ok((operator, columns))
    }

    /// Plans a chain of consecutive expand operations using factorized execution.
    ///
    /// This avoids the Cartesian product explosion that occurs with separate expands.
    /// For a 2-hop query with degree d, this uses O(d) memory instead of O(d^2).
    ///
    /// The chain is executed lazily at query time, not during planning. This ensures
    /// that any filters applied above the expand chain are properly respected.
    pub(super) fn plan_expand_chain(
        &self,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let expands = Self::collect_expand_chain(op);
        if expands.is_empty() {
            return Err(Error::Internal("Empty expand chain".to_string()));
        }

        // Get the base operator (before first expand)
        let first_expand = expands[0];
        let (base_op, base_columns) = self.plan_operator(&first_expand.input)?;

        let mut columns = base_columns.clone();
        let mut steps = Vec::new();

        // Track the level-local source column for each expand
        // For the first expand, it's the column in the input (base_columns)
        // For subsequent expands, the target from the previous level is always at index 1
        // (each level adds [edge, target], so target is at index 1)
        let mut is_first = true;

        for expand in &expands {
            // Find source column for this expand
            let source_column = if is_first {
                // For first expand, find in base columns
                base_columns
                    .iter()
                    .position(|c| c == &expand.from_variable)
                    .ok_or_else(|| {
                        Error::Internal(format!(
                            "Source variable '{}' not found in base columns",
                            expand.from_variable
                        ))
                    })?
            } else {
                // For subsequent expands, the target from the previous level is at index 1
                // (each level adds [edge, target], so target is the second column)
                1
            };

            // Convert direction
            let direction = match expand.direction {
                ExpandDirection::Outgoing => Direction::Outgoing,
                ExpandDirection::Incoming => Direction::Incoming,
                ExpandDirection::Both => Direction::Both,
            };

            // Add expand step configuration
            steps.push(ExpandStep {
                source_column,
                direction,
                edge_type: expand.edge_type.clone(),
            });

            // Add edge and target columns
            let edge_col_name = expand.edge_variable.clone().unwrap_or_else(|| {
                let count = self.anon_edge_counter.get();
                self.anon_edge_counter.set(count + 1);
                format!("_anon_edge_{}", count)
            });
            columns.push(edge_col_name);
            columns.push(expand.to_variable.clone());

            is_first = false;
        }

        // Create lazy operator that executes at query time, not planning time
        let mut lazy_op = LazyFactorizedChainOperator::new(Arc::clone(&self.store), base_op, steps);

        if let Some(tx_id) = self.tx_id {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, Some(tx_id));
        } else {
            lazy_op = lazy_op.with_tx_context(self.viewing_epoch, None);
        }

        Ok((Box::new(lazy_op), columns))
    }
}
