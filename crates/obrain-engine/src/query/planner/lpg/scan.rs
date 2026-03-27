//! Node scan planning.

use super::{
    Arc, GraphStore, LogicalType, NestedLoopJoinOperator, NodeScanOp, Operator, PhysicalJoinType,
    Result, ScanOperator,
};

impl super::Planner {
    /// Plans a node scan operator.
    pub(super) fn plan_node_scan(
        &self,
        scan: &NodeScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let scan_op = if let Some(label) = &scan.label {
            ScanOperator::with_label(Arc::clone(&self.store) as Arc<dyn GraphStore>, label)
        } else {
            ScanOperator::new(Arc::clone(&self.store) as Arc<dyn GraphStore>)
        };

        // Apply MVCC context if available
        let scan_operator: Box<dyn Operator> =
            Box::new(scan_op.with_transaction_context(self.viewing_epoch, self.transaction_id));

        // If there's an input, chain operators with a nested loop join (cross join)
        if let Some(input) = &scan.input {
            let (input_op, mut input_columns) = self.plan_operator(input)?;

            // If the scan variable already exists in the input (e.g., from a
            // correlated ParameterScan), skip the redundant scan and reuse the
            // bound value. This avoids a cross product in CALL { WITH var MATCH (var)... }.
            if input_columns.contains(&scan.variable) {
                return Ok((input_op, input_columns));
            }

            // Build output schema: input columns + scan column
            let mut output_schema: Vec<LogicalType> =
                input_columns.iter().map(|_| LogicalType::Any).collect();
            output_schema.push(LogicalType::Node);

            // Add scan column to input columns
            input_columns.push(scan.variable.clone());

            // Use nested loop join to combine input rows with scanned nodes
            let join_op = Box::new(NestedLoopJoinOperator::new(
                input_op,
                scan_operator,
                None, // No join condition (cross join)
                PhysicalJoinType::Cross,
                output_schema,
            ));

            Ok((join_op, input_columns))
        } else {
            let columns = vec![scan.variable.clone()];
            Ok((scan_operator, columns))
        }
    }
}
