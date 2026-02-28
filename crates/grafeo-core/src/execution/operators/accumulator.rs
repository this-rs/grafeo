//! Shared accumulator types for both pull-based and push-based aggregate operators.
//!
//! Provides the canonical definitions of [`AggregateFunction`], [`AggregateExpr`],
//! and [`HashableValue`] used by both `aggregate.rs` (pull) and `push/aggregate.rs`.

use grafeo_common::types::Value;

/// Aggregation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Count of rows (COUNT(*)).
    Count,
    /// Count of non-null values (COUNT(column)).
    CountNonNull,
    /// Sum of values.
    Sum,
    /// Average of values.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// First value in the group.
    First,
    /// Last value in the group.
    Last,
    /// Collect values into a list.
    Collect,
    /// Sample standard deviation (STDEV).
    StdDev,
    /// Population standard deviation (STDEVP).
    StdDevPop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
}

/// An aggregation expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// The aggregation function.
    pub function: AggregateFunction,
    /// Column index to aggregate (None for COUNT(*)).
    pub column: Option<usize>,
    /// Whether to aggregate distinct values only.
    pub distinct: bool,
    /// Output alias (for naming the result column).
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
}

impl AggregateExpr {
    /// Creates a COUNT(*) expression.
    pub fn count_star() -> Self {
        Self {
            function: AggregateFunction::Count,
            column: None,
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a COUNT(column) expression.
    pub fn count(column: usize) -> Self {
        Self {
            function: AggregateFunction::CountNonNull,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a SUM(column) expression.
    pub fn sum(column: usize) -> Self {
        Self {
            function: AggregateFunction::Sum,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates an AVG(column) expression.
    pub fn avg(column: usize) -> Self {
        Self {
            function: AggregateFunction::Avg,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a MIN(column) expression.
    pub fn min(column: usize) -> Self {
        Self {
            function: AggregateFunction::Min,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a MAX(column) expression.
    pub fn max(column: usize) -> Self {
        Self {
            function: AggregateFunction::Max,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a FIRST(column) expression.
    pub fn first(column: usize) -> Self {
        Self {
            function: AggregateFunction::First,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a LAST(column) expression.
    pub fn last(column: usize) -> Self {
        Self {
            function: AggregateFunction::Last,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a COLLECT(column) expression.
    pub fn collect(column: usize) -> Self {
        Self {
            function: AggregateFunction::Collect,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a STDEV(column) expression (sample standard deviation).
    pub fn stdev(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDev,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a STDEVP(column) expression (population standard deviation).
    pub fn stdev_pop(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDevPop,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a PERCENTILE_DISC(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_disc(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileDisc,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
        }
    }

    /// Creates a PERCENTILE_CONT(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_cont(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileCont,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
        }
    }

    /// Sets the distinct flag.
    pub fn with_distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Sets the output alias.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }
}

/// A wrapper for [`Value`] that can be hashed (for DISTINCT tracking).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HashableValue {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int64(i64),
    /// Float as raw bits (for deterministic hashing).
    Float64Bits(u64),
    /// String value.
    String(String),
    /// Fallback for other types (uses Debug representation).
    Other(String),
}

impl From<&Value> for HashableValue {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => HashableValue::Null,
            Value::Bool(b) => HashableValue::Bool(*b),
            Value::Int64(i) => HashableValue::Int64(*i),
            Value::Float64(f) => HashableValue::Float64Bits(f.to_bits()),
            Value::String(s) => HashableValue::String(s.to_string()),
            other => HashableValue::Other(format!("{other:?}")),
        }
    }
}

impl From<Value> for HashableValue {
    fn from(v: Value) -> Self {
        Self::from(&v)
    }
}
