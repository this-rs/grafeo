//! Expression conversion from logical to physical representations.

use super::*;

impl super::Planner {
    /// Converts a logical expression to a filter expression.
    pub(super) fn convert_expression(&self, expr: &LogicalExpression) -> Result<FilterExpression> {
        match expr {
            LogicalExpression::Literal(v) => Ok(FilterExpression::Literal(v.clone())),
            LogicalExpression::Variable(name) => Ok(FilterExpression::Variable(name.clone())),
            LogicalExpression::Property { variable, property } => Ok(FilterExpression::Property {
                variable: variable.clone(),
                property: property.clone(),
            }),
            LogicalExpression::Binary { left, op, right } => {
                let left_expr = self.convert_expression(left)?;
                let right_expr = self.convert_expression(right)?;
                let filter_op = convert_binary_op(*op)?;
                Ok(FilterExpression::Binary {
                    left: Box::new(left_expr),
                    op: filter_op,
                    right: Box::new(right_expr),
                })
            }
            LogicalExpression::Unary { op, operand } => {
                let operand_expr = self.convert_expression(operand)?;
                let filter_op = convert_unary_op(*op)?;
                Ok(FilterExpression::Unary {
                    op: filter_op,
                    operand: Box::new(operand_expr),
                })
            }
            LogicalExpression::FunctionCall { name, args, .. } => {
                let filter_args: Vec<FilterExpression> = args
                    .iter()
                    .map(|a| self.convert_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::FunctionCall {
                    name: name.clone(),
                    args: filter_args,
                })
            }
            LogicalExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                let filter_operand = operand
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                let filter_when_clauses: Vec<(FilterExpression, FilterExpression)> = when_clauses
                    .iter()
                    .map(|(cond, result)| {
                        Ok((
                            self.convert_expression(cond)?,
                            self.convert_expression(result)?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let filter_else = else_clause
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::Case {
                    operand: filter_operand,
                    when_clauses: filter_when_clauses,
                    else_clause: filter_else,
                })
            }
            LogicalExpression::List(items) => {
                let filter_items: Vec<FilterExpression> = items
                    .iter()
                    .map(|item| self.convert_expression(item))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::List(filter_items))
            }
            LogicalExpression::Map(pairs) => {
                let filter_pairs: Vec<(String, FilterExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.convert_expression(v)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(FilterExpression::Map(filter_pairs))
            }
            LogicalExpression::IndexAccess { base, index } => {
                let base_expr = self.convert_expression(base)?;
                let index_expr = self.convert_expression(index)?;
                Ok(FilterExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            LogicalExpression::SliceAccess { base, start, end } => {
                let base_expr = self.convert_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.convert_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.convert_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(FilterExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            LogicalExpression::Parameter(_) => Err(Error::Internal(
                "Parameters not yet supported in filters".to_string(),
            )),
            LogicalExpression::Labels(var) => Ok(FilterExpression::Labels(var.clone())),
            LogicalExpression::Type(var) => Ok(FilterExpression::Type(var.clone())),
            LogicalExpression::Id(var) => Ok(FilterExpression::Id(var.clone())),
            LogicalExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let list = self.convert_expression(list_expr)?;
                let filter = filter_expr
                    .as_ref()
                    .map(|f| self.convert_expression(f))
                    .transpose()?
                    .map(Box::new);
                let map = self.convert_expression(map_expr)?;
                Ok(FilterExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list),
                    filter_expr: filter,
                    map_expr: Box::new(map),
                })
            }
            LogicalExpression::ExistsSubquery(subplan) => {
                // Extract the pattern from the subplan
                // For EXISTS { MATCH (n)-[:TYPE]->() }, we extract start_var, direction, edge_type
                let (start_var, direction, edge_type, end_labels) =
                    self.extract_exists_pattern(subplan)?;

                Ok(FilterExpression::ExistsSubquery {
                    start_var,
                    direction,
                    edge_type,
                    end_labels,
                    min_hops: None,
                    max_hops: None,
                })
            }
            LogicalExpression::CountSubquery(_) => Err(Error::Internal(
                "COUNT subqueries not yet supported".to_string(),
            )),
        }
    }

    /// Extracts the pattern from an EXISTS subplan.
    /// Returns (start_variable, direction, edge_type, end_labels).
    pub(super) fn extract_exists_pattern(
        &self,
        subplan: &LogicalOperator,
    ) -> Result<(String, Direction, Option<String>, Option<Vec<String>>)> {
        match subplan {
            LogicalOperator::Expand(expand) => {
                // Get end node labels from the to_variable if there's a node scan input
                let end_labels = self.extract_end_labels_from_expand(expand);
                let direction = match expand.direction {
                    ExpandDirection::Outgoing => Direction::Outgoing,
                    ExpandDirection::Incoming => Direction::Incoming,
                    ExpandDirection::Both => Direction::Both,
                };
                Ok((
                    expand.from_variable.clone(),
                    direction,
                    expand.edge_type.clone(),
                    end_labels,
                ))
            }
            LogicalOperator::NodeScan(scan) => {
                if let Some(input) = &scan.input {
                    self.extract_exists_pattern(input)
                } else {
                    Err(Error::Internal(
                        "EXISTS subquery must contain an edge pattern".to_string(),
                    ))
                }
            }
            LogicalOperator::Filter(filter) => self.extract_exists_pattern(&filter.input),
            _ => Err(Error::Internal(
                "Unsupported EXISTS subquery pattern".to_string(),
            )),
        }
    }

    /// Extracts end node labels from an Expand operator if present.
    pub(super) fn extract_end_labels_from_expand(&self, expand: &ExpandOp) -> Option<Vec<String>> {
        // Check if the expand has a NodeScan input with a label filter
        match expand.input.as_ref() {
            LogicalOperator::NodeScan(scan) => scan.label.clone().map(|l| vec![l]),
            _ => None,
        }
    }
}
