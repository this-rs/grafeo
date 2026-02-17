//! Filter planning with zone map pre-filtering and index lookups.

use super::*;

impl super::Planner {
    /// Plans a filter operator.
    ///
    /// Uses zone map pre-filtering to potentially skip scans when predicates
    /// definitely won't match any data. Also uses property indexes when available
    /// for O(1) lookups instead of full scans.
    pub(super) fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Check zone maps for simple property predicates before scanning
        // If zone map says "definitely no matches", we can short-circuit
        if let Some(false) = self.check_zone_map_for_predicate(&filter.predicate) {
            // Zone map says no matches possible - return empty result
            let (_, columns) = self.plan_operator(&filter.input)?;
            let schema = self.derive_schema_from_columns(&columns);
            let empty_op = Box::new(EmptyOperator::new(schema));
            return Ok((empty_op, columns));
        }

        // Try to use property index for equality predicates on indexed properties
        if let Some(result) = self.try_plan_filter_with_property_index(filter)? {
            return Ok(result);
        }

        // Try to use range optimization for range predicates (>, <, >=, <=)
        if let Some(result) = self.try_plan_filter_with_range_index(filter)? {
            return Ok(result);
        }

        // Plan the input operator first
        let (input_op, columns) = self.plan_operator(&filter.input)?;

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = self.convert_expression(&filter.predicate)?;

        // Create the predicate
        let predicate =
            ExpressionPredicate::new(filter_expr, variable_columns, Arc::clone(&self.store));

        // Create the filter operator
        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));

        Ok((operator, columns))
    }

    /// Checks zone maps for a predicate to see if we can skip the scan entirely.
    ///
    /// Returns:
    /// - `Some(false)` if zone map proves no matches possible (can skip)
    /// - `Some(true)` if zone map says matches might exist
    /// - `None` if zone map check not applicable
    pub(super) fn check_zone_map_for_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<bool> {
        use grafeo_core::graph::lpg::CompareOp;

        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                // Check for AND/OR first (compound conditions)
                match op {
                    BinaryOp::And => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // If either side definitely won't match, the AND won't match
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            // If both might match, might match overall
                            (Some(true), Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    BinaryOp::Or => {
                        let left_result = self.check_zone_map_for_predicate(left);
                        let right_result = self.check_zone_map_for_predicate(right);

                        return match (left_result, right_result) {
                            // Both sides definitely won't match
                            (Some(false), Some(false)) => Some(false),
                            // At least one side might match
                            (Some(true), _) | (_, Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    _ => {}
                }

                // Simple property comparison: n.property op value
                let (property, compare_op, value) = match (left.as_ref(), right.as_ref()) {
                    (
                        LogicalExpression::Property { property, .. },
                        LogicalExpression::Literal(val),
                    ) => {
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Lt,
                            BinaryOp::Le => CompareOp::Le,
                            BinaryOp::Gt => CompareOp::Gt,
                            BinaryOp::Ge => CompareOp::Ge,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    (
                        LogicalExpression::Literal(val),
                        LogicalExpression::Property { property, .. },
                    ) => {
                        // Flip comparison for reversed operands
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Gt, // val < prop means prop > val
                            BinaryOp::Le => CompareOp::Ge,
                            BinaryOp::Gt => CompareOp::Lt,
                            BinaryOp::Ge => CompareOp::Le,
                            _ => return None,
                        };
                        (property.clone(), cmp, val.clone())
                    }
                    _ => return None,
                };

                // Check zone map for node properties
                let might_match =
                    self.store
                        .node_property_might_match(&property.into(), compare_op, &value);

                Some(might_match)
            }

            _ => None,
        }
    }

    /// Tries to use a property index for filter optimization.
    ///
    /// When a filter predicate is an equality check on an indexed property,
    /// and the input is a simple NodeScan, we can use the index to look up
    /// matching nodes directly instead of scanning all nodes.
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    pub(super) fn try_plan_filter_with_property_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Extract property equality conditions from the predicate
        // Handles both simple (n.prop = val) and compound (n.a = 1 AND n.b = 2)
        let conditions = self.extract_equality_conditions(&filter.predicate, &scan_variable);

        if conditions.is_empty() {
            return Ok(None);
        }

        // Check if at least one condition has an index
        let has_indexed_condition = conditions
            .iter()
            .any(|(prop, _)| self.store.has_property_index(prop));

        // Without an index we can still optimize when there's a label constraint:
        // label-first scan + property check avoids DataChunk/expression overhead.
        if !has_indexed_condition && scan_label.is_none() {
            return Ok(None);
        }

        let mut matching_nodes = if has_indexed_condition {
            // Use index-based batch lookup
            let conditions_ref: Vec<(&str, Value)> = conditions
                .iter()
                .map(|(p, v)| (p.as_str(), v.clone()))
                .collect();
            let mut nodes = self.store.find_nodes_by_properties(&conditions_ref);

            // Intersect with label if present
            if let Some(label) = &scan_label {
                let label_nodes: std::collections::HashSet<_> =
                    self.store.nodes_by_label(label).into_iter().collect();
                nodes.retain(|n| label_nodes.contains(n));
            }
            nodes
        } else {
            // No index but we have a label — scan label first, then check properties.
            // This is more efficient than ScanOperator → DataChunk → FilterOperator
            // because it avoids DataChunk materialization and expression evaluation.
            let label = scan_label.as_ref().expect("label checked above");
            let label_nodes = self.store.nodes_by_label(label);
            label_nodes
                .into_iter()
                .filter(|&node_id| {
                    conditions.iter().all(|(prop, val)| {
                        let key = grafeo_common::types::PropertyKey::new(prop);
                        self.store
                            .get_node_property(node_id, &key)
                            .is_some_and(|v| v == *val)
                    })
                })
                .collect()
        };

        // MVCC visibility: filter out nodes not visible at the current epoch/tx.
        // Without this, rolled-back or uncommitted nodes could leak through.
        let epoch = self.viewing_epoch;
        let tx = self.tx_id.unwrap_or(TxId::SYSTEM);
        matching_nodes.retain(|id| self.store.get_node_versioned(*id, epoch, tx).is_some());

        let columns = vec![scan_variable.clone()];
        let node_list_op: Box<dyn Operator> = Box::new(NodeListOperator::new(matching_nodes, 2048));

        // Check for remaining predicate parts that weren't pushed down
        // (e.g., range conditions in a compound predicate like `n.name = 'Alice' AND n.age > 30`)
        if let Some(remaining) =
            self.extract_remaining_predicate(&filter.predicate, &scan_variable, &conditions)
        {
            let variable_columns: HashMap<String, usize> = columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(&remaining)?;
            let predicate =
                ExpressionPredicate::new(filter_expr, variable_columns, Arc::clone(&self.store));
            let filtered = Box::new(FilterOperator::new(node_list_op, Box::new(predicate)));
            Ok(Some((filtered, columns)))
        } else {
            Ok(Some((node_list_op, columns)))
        }
    }

    /// Extracts the remaining predicate after removing pushed-down equality conditions.
    ///
    /// Given `n.name = 'Alice' AND n.age > 30` with pushed conditions `[("name", "Alice")]`,
    /// returns `Some(n.age > 30)`. Returns `None` when all conditions were pushed down.
    pub(super) fn extract_remaining_predicate(
        &self,
        predicate: &LogicalExpression,
        target_variable: &str,
        pushed_conditions: &[(String, Value)],
    ) -> Option<LogicalExpression> {
        match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                let left_remaining =
                    self.extract_remaining_predicate(left, target_variable, pushed_conditions);
                let right_remaining =
                    self.extract_remaining_predicate(right, target_variable, pushed_conditions);

                match (left_remaining, right_remaining) {
                    (Some(l), Some(r)) => Some(LogicalExpression::Binary {
                        left: Box::new(l),
                        op: BinaryOp::And,
                        right: Box::new(r),
                    }),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                // Check if this equality was pushed down
                if let Some((var, prop, val)) = self.extract_property_equality(left, right)
                    && var == target_variable
                    && pushed_conditions
                        .iter()
                        .any(|(p, v)| *p == prop && *v == val)
                {
                    None // Already handled at the store level
                } else {
                    Some(predicate.clone())
                }
            }
            _ => Some(predicate.clone()),
        }
    }

    /// Extracts equality conditions (property = literal) from a predicate.
    ///
    /// Handles both simple predicates and AND chains:
    /// - `n.name = "Alice"` → `[("name", "Alice")]`
    /// - `n.name = "Alice" AND n.age = 30` → `[("name", "Alice"), ("age", 30)]`
    pub(super) fn extract_equality_conditions(
        &self,
        predicate: &LogicalExpression,
        target_variable: &str,
    ) -> Vec<(String, Value)> {
        let mut conditions = Vec::new();
        self.collect_equality_conditions(predicate, target_variable, &mut conditions);
        conditions
    }

    /// Recursively collects equality conditions from AND expressions.
    pub(super) fn collect_equality_conditions(
        &self,
        expr: &LogicalExpression,
        target_variable: &str,
        conditions: &mut Vec<(String, Value)>,
    ) {
        match expr {
            // Handle AND: recurse into both sides
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                self.collect_equality_conditions(left, target_variable, conditions);
                self.collect_equality_conditions(right, target_variable, conditions);
            }

            // Handle equality: extract property and value
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                if let Some((var, prop, val)) = self.extract_property_equality(left, right)
                    && var == target_variable
                {
                    conditions.push((prop, val));
                }
            }

            _ => {}
        }
    }

    /// Extracts (variable, property, value) from a property equality expression.
    pub(super) fn extract_property_equality(
        &self,
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<(String, String, Value)> {
        match (left, right) {
            (
                LogicalExpression::Property { variable, property },
                LogicalExpression::Literal(val),
            ) => Some((variable.clone(), property.clone(), val.clone())),
            (
                LogicalExpression::Literal(val),
                LogicalExpression::Property { variable, property },
            ) => Some((variable.clone(), property.clone(), val.clone())),
            _ => None,
        }
    }

    /// Tries to optimize a filter using range queries on properties.
    ///
    /// This optimization is applied when:
    /// - The input is a simple NodeScan (no nested operations)
    /// - The predicate contains range comparisons (>, <, >=, <=)
    /// - The same variable and property are being filtered
    ///
    /// Handles both simple range predicates (`n.age > 30`) and BETWEEN patterns
    /// (`n.age >= 30 AND n.age <= 50`).
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` if not applicable, or `Err` on error.
    pub(super) fn try_plan_filter_with_range_index(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only optimize if input is a simple NodeScan (not nested)
        let (scan_variable, scan_label) = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
                (scan.variable.clone(), scan.label.clone())
            }
            _ => return Ok(None),
        };

        // Try to extract BETWEEN pattern first (more efficient)
        if let Some((variable, property, min, max, min_inc, max_inc)) =
            self.extract_between_predicate(&filter.predicate)
            && variable == scan_variable
        {
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: Some(&min),
                    max: Some(&max),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        // Try to extract simple range predicate
        if let Some((variable, property, op, value)) =
            self.extract_range_predicate(&filter.predicate)
            && variable == scan_variable
        {
            let (min, max, min_inc, max_inc) = match op {
                BinaryOp::Lt => (None, Some(value), false, false),
                BinaryOp::Le => (None, Some(value), false, true),
                BinaryOp::Gt => (Some(value), None, false, false),
                BinaryOp::Ge => (Some(value), None, true, false),
                _ => return Ok(None),
            };
            return self.plan_range_filter(
                &scan_variable,
                &scan_label,
                &property,
                RangeBounds {
                    min: min.as_ref(),
                    max: max.as_ref(),
                    min_inclusive: min_inc,
                    max_inclusive: max_inc,
                },
            );
        }

        Ok(None)
    }

    /// Plans a range filter using `find_nodes_in_range`.
    pub(super) fn plan_range_filter(
        &self,
        scan_variable: &str,
        scan_label: &Option<String>,
        property: &str,
        bounds: RangeBounds<'_>,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Use the store's range query method
        let mut matching_nodes = self.store.find_nodes_in_range(
            property,
            bounds.min,
            bounds.max,
            bounds.min_inclusive,
            bounds.max_inclusive,
        );

        // If there's a label filter, also filter by label
        if let Some(label) = scan_label {
            let label_nodes: std::collections::HashSet<_> =
                self.store.nodes_by_label(label).into_iter().collect();
            matching_nodes.retain(|n| label_nodes.contains(n));
        }

        // MVCC visibility: filter out nodes not visible at the current epoch/tx.
        let epoch = self.viewing_epoch;
        let tx = self.tx_id.unwrap_or(TxId::SYSTEM);
        matching_nodes.retain(|id| self.store.get_node_versioned(*id, epoch, tx).is_some());

        // Create a NodeListOperator with the matching nodes
        let node_list_op = Box::new(NodeListOperator::new(matching_nodes, 2048));
        let columns = vec![scan_variable.to_string()];

        Ok(Some((node_list_op, columns)))
    }

    /// Extracts a simple range predicate (>, <, >=, <=) from an expression.
    ///
    /// Returns `(variable, property, operator, value)` if found.
    pub(super) fn extract_range_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, BinaryOp, Value)> {
        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                match op {
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        // Try property on left: n.age > 30
                        if let (
                            LogicalExpression::Property { variable, property },
                            LogicalExpression::Literal(val),
                        ) = (left.as_ref(), right.as_ref())
                        {
                            return Some((variable.clone(), property.clone(), *op, val.clone()));
                        }

                        // Try property on right: 30 < n.age (flip operator)
                        if let (
                            LogicalExpression::Literal(val),
                            LogicalExpression::Property { variable, property },
                        ) = (left.as_ref(), right.as_ref())
                        {
                            let flipped_op = match op {
                                BinaryOp::Lt => BinaryOp::Gt,
                                BinaryOp::Le => BinaryOp::Ge,
                                BinaryOp::Gt => BinaryOp::Lt,
                                BinaryOp::Ge => BinaryOp::Le,
                                _ => return None,
                            };
                            return Some((
                                variable.clone(),
                                property.clone(),
                                flipped_op,
                                val.clone(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        None
    }

    /// Extracts a BETWEEN pattern from compound predicates.
    ///
    /// Recognizes patterns like:
    /// - `n.age >= 30 AND n.age <= 50`
    /// - `n.age > 30 AND n.age < 50`
    ///
    /// Returns `(variable, property, min_value, max_value, min_inclusive, max_inclusive)`.
    pub(super) fn extract_between_predicate(
        &self,
        predicate: &LogicalExpression,
    ) -> Option<(String, String, Value, Value, bool, bool)> {
        // Must be an AND expression
        let (left, right) = match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return None,
        };

        // Extract range predicates from both sides
        let left_range = self.extract_range_predicate(left);
        let right_range = self.extract_range_predicate(right);

        let (left_var, left_prop, left_op, left_val) = left_range?;
        let (right_var, right_prop, right_op, right_val) = right_range?;

        // Must be same variable and property
        if left_var != right_var || left_prop != right_prop {
            return None;
        }

        // Determine which is lower bound and which is upper bound
        let (min_val, max_val, min_inc, max_inc) = match (left_op, right_op) {
            // n.x >= min AND n.x <= max
            (BinaryOp::Ge, BinaryOp::Le) => (left_val, right_val, true, true),
            // n.x >= min AND n.x < max
            (BinaryOp::Ge, BinaryOp::Lt) => (left_val, right_val, true, false),
            // n.x > min AND n.x <= max
            (BinaryOp::Gt, BinaryOp::Le) => (left_val, right_val, false, true),
            // n.x > min AND n.x < max
            (BinaryOp::Gt, BinaryOp::Lt) => (left_val, right_val, false, false),
            // Reversed order: n.x <= max AND n.x >= min
            (BinaryOp::Le, BinaryOp::Ge) => (right_val, left_val, true, true),
            // n.x < max AND n.x >= min
            (BinaryOp::Lt, BinaryOp::Ge) => (right_val, left_val, true, false),
            // n.x <= max AND n.x > min
            (BinaryOp::Le, BinaryOp::Gt) => (right_val, left_val, false, true),
            // n.x < max AND n.x > min
            (BinaryOp::Lt, BinaryOp::Gt) => (right_val, left_val, false, false),
            _ => return None,
        };

        Some((left_var, left_prop, min_val, max_val, min_inc, max_inc))
    }
}
