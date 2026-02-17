//! Join, union, and distinct planning.

use super::*;

impl super::Planner {
    /// Plans a JOIN operator.
    pub(super) fn plan_join(&self, join: &JoinOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (left_op, left_columns) = self.plan_operator(&join.left)?;
        let (right_op, right_columns) = self.plan_operator(&join.right)?;

        // Build combined output columns
        let mut columns = left_columns.clone();
        columns.extend(right_columns.clone());

        // Convert join type
        let physical_join_type = match join.join_type {
            JoinType::Inner => PhysicalJoinType::Inner,
            JoinType::Left => PhysicalJoinType::Left,
            JoinType::Right => PhysicalJoinType::Right,
            JoinType::Full => PhysicalJoinType::Full,
            JoinType::Cross => PhysicalJoinType::Cross,
            JoinType::Semi => PhysicalJoinType::Semi,
            JoinType::Anti => PhysicalJoinType::Anti,
        };

        // Build key columns from join conditions
        let (probe_keys, build_keys): (Vec<usize>, Vec<usize>) = if join.conditions.is_empty() {
            // Cross join - no keys
            (vec![], vec![])
        } else {
            join.conditions
                .iter()
                .filter_map(|cond| {
                    // Try to extract column indices from expressions
                    let left_idx = self.expression_to_column(&cond.left, &left_columns).ok()?;
                    let right_idx = self
                        .expression_to_column(&cond.right, &right_columns)
                        .ok()?;
                    Some((left_idx, right_idx))
                })
                .unzip()
        };

        let output_schema = self.derive_schema_from_columns(&columns);

        // Check if we should use leapfrog join for cyclic patterns
        // Currently we use hash join by default; leapfrog is available but
        // requires explicit multi-way join detection which will be added
        // when we have proper cyclic pattern detection in the optimizer.
        // For now, LeapfrogJoinOperator is available for direct use.
        let _ = LeapfrogJoinOperator::new; // Suppress unused warning

        let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            physical_join_type,
            output_schema,
        ));

        Ok((operator, columns))
    }

    /// Extracts a column index from an expression.
    pub(super) fn expression_to_column(
        &self,
        expr: &LogicalExpression,
        columns: &[String],
    ) -> Result<usize> {
        match expr {
            LogicalExpression::Variable(name) => columns
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| Error::Internal(format!("Variable '{}' not found", name))),
            _ => Err(Error::Internal(
                "Only variables supported in join conditions".to_string(),
            )),
        }
    }

    /// Plans a UNION operator.
    pub(super) fn plan_union(&self, union: &UnionOp) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if union.inputs.is_empty() {
            return Err(Error::Internal(
                "Union requires at least one input".to_string(),
            ));
        }

        let mut inputs = Vec::with_capacity(union.inputs.len());
        let mut columns = Vec::new();

        for (i, input) in union.inputs.iter().enumerate() {
            let (op, cols) = self.plan_operator(input)?;
            if i == 0 {
                columns = cols;
            }
            inputs.push(op);
        }

        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(UnionOperator::new(inputs, output_schema));

        Ok((operator, columns))
    }

    /// Plans a DISTINCT operator.
    pub(super) fn plan_distinct(
        &self,
        distinct: &DistinctOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (input_op, columns) = self.plan_operator(&distinct.input)?;
        let output_schema = self.derive_schema_from_columns(&columns);
        let operator = Box::new(DistinctOperator::new(input_op, output_schema));
        Ok((operator, columns))
    }
}
