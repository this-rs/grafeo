//! Database configuration.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

/// The graph data model for a database.
///
/// Each database uses exactly one model, chosen at creation time and immutable
/// after that. The engine initializes only the relevant store, saving memory.
///
/// Schema variants (OWL, RDFS, JSON Schema) are a server-level concern - from
/// the engine's perspective those map to either `Lpg` or `Rdf`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GraphModel {
    /// Labeled Property Graph (default). Supports GQL, Cypher, Gremlin, GraphQL.
    #[default]
    Lpg,
    /// RDF triple store. Supports SPARQL.
    Rdf,
}

impl fmt::Display for GraphModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lpg => write!(f, "LPG"),
            Self::Rdf => write!(f, "RDF"),
        }
    }
}

/// Access mode for opening a database.
///
/// Controls whether the database is opened for full read-write access
/// (the default) or read-only access. Read-only mode uses a shared file
/// lock, allowing multiple processes to read the same `.grafeo` file
/// concurrently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum AccessMode {
    /// Full read-write access (default). Acquires an exclusive file lock.
    #[default]
    ReadWrite,
    /// Read-only access. Acquires a shared file lock, allowing concurrent
    /// readers. The database loads the last checkpoint snapshot but does not
    /// replay the WAL or allow mutations.
    ReadOnly,
}

impl fmt::Display for AccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadWrite => write!(f, "read-write"),
            Self::ReadOnly => write!(f, "read-only"),
        }
    }
}

/// Storage format for persistent databases.
///
/// Controls whether the database uses a single `.grafeo` file or a legacy
/// WAL directory. The default (`Auto`) auto-detects based on the path:
/// files ending in `.grafeo` use single-file format, directories use WAL.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum StorageFormat {
    /// Auto-detect based on path: `.grafeo` extension = single file,
    /// existing directory = WAL directory, new path without extension = WAL directory.
    #[default]
    Auto,
    /// Legacy WAL directory format (directory with `wal/` subdirectory).
    WalDirectory,
    /// Single `.grafeo` file with a sidecar `.grafeo.wal/` directory during operation.
    /// At rest (after checkpoint), only the `.grafeo` file exists.
    SingleFile,
}

impl fmt::Display for StorageFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::WalDirectory => write!(f, "wal-directory"),
            Self::SingleFile => write!(f, "single-file"),
        }
    }
}

/// WAL durability mode controlling the trade-off between safety and speed.
///
/// This enum lives in config so that `Config` can always carry the desired
/// durability regardless of whether the `wal` feature is compiled in. When
/// WAL is enabled, the engine maps this to the adapter-level durability mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Fsync after every commit. Slowest but safest.
    Sync,
    /// Batch fsync periodically. Good balance of performance and durability.
    Batch {
        /// Maximum time between syncs in milliseconds.
        max_delay_ms: u64,
        /// Maximum records between syncs.
        max_records: u64,
    },
    /// Adaptive sync via a background flusher thread.
    Adaptive {
        /// Target interval between flushes in milliseconds.
        target_interval_ms: u64,
    },
    /// No sync - rely on OS buffer flushing. Fastest but may lose recent data.
    NoSync,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batch {
            max_delay_ms: 100,
            max_records: 1000,
        }
    }
}

/// Errors from [`Config::validate()`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Memory limit must be greater than zero.
    ZeroMemoryLimit,
    /// Thread count must be greater than zero.
    ZeroThreads,
    /// WAL flush interval must be greater than zero.
    ZeroWalFlushInterval,
    /// RDF graph model requires the `rdf` feature flag.
    RdfFeatureRequired,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMemoryLimit => write!(f, "memory_limit must be greater than zero"),
            Self::ZeroThreads => write!(f, "threads must be greater than zero"),
            Self::ZeroWalFlushInterval => {
                write!(f, "wal_flush_interval_ms must be greater than zero")
            }
            Self::RdfFeatureRequired => {
                write!(
                    f,
                    "RDF graph model requires the `rdf` feature flag to be enabled"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Database configuration.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
pub struct Config {
    /// Graph data model (LPG or RDF). Immutable after database creation.
    pub graph_model: GraphModel,
    /// Path to the database directory (None for in-memory only).
    pub path: Option<PathBuf>,

    /// Memory limit in bytes (None for unlimited).
    pub memory_limit: Option<usize>,

    /// Path for spilling data to disk under memory pressure.
    pub spill_path: Option<PathBuf>,

    /// Number of worker threads for query execution.
    pub threads: usize,

    /// Whether to enable WAL for durability.
    pub wal_enabled: bool,

    /// WAL flush interval in milliseconds.
    pub wal_flush_interval_ms: u64,

    /// Whether to maintain backward edges.
    pub backward_edges: bool,

    /// Whether to enable query logging.
    pub query_logging: bool,

    /// Adaptive execution configuration.
    pub adaptive: AdaptiveConfig,

    /// Whether to use factorized execution for multi-hop queries.
    ///
    /// When enabled, consecutive MATCH expansions are executed using factorized
    /// representation which avoids Cartesian product materialization. This provides
    /// 5-100x speedup for multi-hop queries with high fan-out.
    ///
    /// Enabled by default.
    pub factorized_execution: bool,

    /// WAL durability mode. Only used when `wal_enabled` is true.
    pub wal_durability: DurabilityMode,

    /// Storage format for persistent databases.
    ///
    /// `Auto` (default) detects the format from the path: `.grafeo` extension
    /// uses single-file format, directories use the legacy WAL directory.
    pub storage_format: StorageFormat,

    /// Whether to enable catalog schema constraint enforcement.
    ///
    /// When true, the catalog enforces label, edge type, and property constraints
    /// (e.g. required properties, uniqueness). The server sets this for JSON
    /// Schema databases and populates constraints after creation.
    pub schema_constraints: bool,

    /// Maximum time a single query may run before being cancelled.
    ///
    /// When set, the executor checks the deadline between operator batches and
    /// returns `QueryError::timeout()` if the wall-clock limit is exceeded.
    /// `None` means no timeout (queries may run indefinitely).
    pub query_timeout: Option<Duration>,

    /// Run MVCC version garbage collection every N commits.
    ///
    /// Old versions that are no longer visible to any active transaction are
    /// pruned to reclaim memory. Set to 0 to disable automatic GC.
    pub gc_interval: usize,

    /// Access mode: read-write (default) or read-only.
    ///
    /// Read-only mode uses a shared file lock, allowing multiple processes to
    /// read the same database concurrently. Mutations are rejected at the
    /// session level.
    pub access_mode: AccessMode,
}

/// Configuration for adaptive query execution.
///
/// Adaptive execution monitors actual row counts during query processing and
/// can trigger re-optimization when estimates are significantly wrong.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Whether adaptive execution is enabled.
    pub enabled: bool,

    /// Deviation threshold that triggers re-optimization.
    ///
    /// A value of 3.0 means re-optimization is triggered when actual cardinality
    /// is more than 3x or less than 1/3x the estimated value.
    pub threshold: f64,

    /// Minimum number of rows before considering re-optimization.
    ///
    /// Helps avoid thrashing on small result sets.
    pub min_rows: u64,

    /// Maximum number of re-optimizations allowed per query.
    pub max_reoptimizations: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 3.0,
            min_rows: 1000,
            max_reoptimizations: 3,
        }
    }
}

impl AdaptiveConfig {
    /// Creates a disabled adaptive config.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Sets the deviation threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the minimum rows before re-optimization.
    #[must_use]
    pub fn with_min_rows(mut self, min_rows: u64) -> Self {
        self.min_rows = min_rows;
        self
    }

    /// Sets the maximum number of re-optimizations.
    #[must_use]
    pub fn with_max_reoptimizations(mut self, max: usize) -> Self {
        self.max_reoptimizations = max;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            graph_model: GraphModel::default(),
            path: None,
            memory_limit: None,
            spill_path: None,
            threads: num_cpus::get(),
            wal_enabled: true,
            wal_flush_interval_ms: 100,
            backward_edges: true,
            query_logging: false,
            adaptive: AdaptiveConfig::default(),
            factorized_execution: true,
            wal_durability: DurabilityMode::default(),
            storage_format: StorageFormat::default(),
            schema_constraints: false,
            query_timeout: None,
            gc_interval: 100,
            access_mode: AccessMode::default(),
        }
    }
}

impl Config {
    /// Creates a new configuration for an in-memory database.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            path: None,
            wal_enabled: false,
            ..Default::default()
        }
    }

    /// Creates a new configuration for a persistent database.
    #[must_use]
    pub fn persistent(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            wal_enabled: true,
            ..Default::default()
        }
    }

    /// Sets the memory limit.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Sets the number of worker threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Disables backward edges.
    #[must_use]
    pub fn without_backward_edges(mut self) -> Self {
        self.backward_edges = false;
        self
    }

    /// Enables query logging.
    #[must_use]
    pub fn with_query_logging(mut self) -> Self {
        self.query_logging = true;
        self
    }

    /// Sets the memory budget as a fraction of system RAM.
    #[must_use]
    pub fn with_memory_fraction(mut self, fraction: f64) -> Self {
        use grafeo_common::memory::buffer::BufferManagerConfig;
        let system_memory = BufferManagerConfig::detect_system_memory();
        self.memory_limit = Some((system_memory as f64 * fraction) as usize);
        self
    }

    /// Sets the spill directory for out-of-core processing.
    #[must_use]
    pub fn with_spill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spill_path = Some(path.into());
        self
    }

    /// Sets the adaptive execution configuration.
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: AdaptiveConfig) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Disables adaptive execution.
    #[must_use]
    pub fn without_adaptive(mut self) -> Self {
        self.adaptive.enabled = false;
        self
    }

    /// Disables factorized execution for multi-hop queries.
    ///
    /// This reverts to the traditional flat execution model where each expansion
    /// creates a full Cartesian product. Only use this if you encounter issues
    /// with factorized execution.
    #[must_use]
    pub fn without_factorized_execution(mut self) -> Self {
        self.factorized_execution = false;
        self
    }

    /// Sets the graph data model.
    #[must_use]
    pub fn with_graph_model(mut self, model: GraphModel) -> Self {
        self.graph_model = model;
        self
    }

    /// Sets the WAL durability mode.
    #[must_use]
    pub fn with_wal_durability(mut self, mode: DurabilityMode) -> Self {
        self.wal_durability = mode;
        self
    }

    /// Sets the storage format for persistent databases.
    #[must_use]
    pub fn with_storage_format(mut self, format: StorageFormat) -> Self {
        self.storage_format = format;
        self
    }

    /// Enables catalog schema constraint enforcement.
    #[must_use]
    pub fn with_schema_constraints(mut self) -> Self {
        self.schema_constraints = true;
        self
    }

    /// Sets the maximum time a query may run before being cancelled.
    #[must_use]
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = Some(timeout);
        self
    }

    /// Sets the MVCC garbage collection interval (every N commits).
    ///
    /// Set to 0 to disable automatic GC.
    #[must_use]
    pub fn with_gc_interval(mut self, interval: usize) -> Self {
        self.gc_interval = interval;
        self
    }

    /// Sets the access mode (read-write or read-only).
    #[must_use]
    pub fn with_access_mode(mut self, mode: AccessMode) -> Self {
        self.access_mode = mode;
        self
    }

    /// Shorthand for opening a persistent database in read-only mode.
    ///
    /// Uses a shared file lock, allowing multiple processes to read the same
    /// `.grafeo` file concurrently. Mutations are rejected at the session level.
    #[must_use]
    pub fn read_only(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            wal_enabled: false,
            access_mode: AccessMode::ReadOnly,
            ..Default::default()
        }
    }

    /// Validates the configuration, returning an error for invalid combinations.
    ///
    /// Called automatically by [`GrafeoDB::with_config()`](crate::GrafeoDB::with_config).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any setting is invalid.
    pub fn validate(&self) -> std::result::Result<(), ConfigError> {
        if let Some(limit) = self.memory_limit
            && limit == 0
        {
            return Err(ConfigError::ZeroMemoryLimit);
        }

        if self.threads == 0 {
            return Err(ConfigError::ZeroThreads);
        }

        if self.wal_flush_interval_ms == 0 {
            return Err(ConfigError::ZeroWalFlushInterval);
        }

        #[cfg(not(feature = "rdf"))]
        if self.graph_model == GraphModel::Rdf {
            return Err(ConfigError::RdfFeatureRequired);
        }

        Ok(())
    }
}

/// Helper function to get CPU count (fallback implementation).
mod num_cpus {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn get() -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.graph_model, GraphModel::Lpg);
        assert!(config.path.is_none());
        assert!(config.memory_limit.is_none());
        assert!(config.spill_path.is_none());
        assert!(config.threads > 0);
        assert!(config.wal_enabled);
        assert_eq!(config.wal_flush_interval_ms, 100);
        assert!(config.backward_edges);
        assert!(!config.query_logging);
        assert!(config.factorized_execution);
        assert_eq!(config.wal_durability, DurabilityMode::default());
        assert!(!config.schema_constraints);
        assert!(config.query_timeout.is_none());
        assert_eq!(config.gc_interval, 100);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.path.is_none());
        assert!(!config.wal_enabled);
        assert!(config.backward_edges);
    }

    #[test]
    fn test_config_persistent() {
        let config = Config::persistent("/tmp/test_db");
        assert_eq!(
            config.path.as_deref(),
            Some(std::path::Path::new("/tmp/test_db"))
        );
        assert!(config.wal_enabled);
    }

    #[test]
    fn test_config_with_memory_limit() {
        let config = Config::in_memory().with_memory_limit(1024 * 1024);
        assert_eq!(config.memory_limit, Some(1024 * 1024));
    }

    #[test]
    fn test_config_with_threads() {
        let config = Config::in_memory().with_threads(8);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_config_without_backward_edges() {
        let config = Config::in_memory().without_backward_edges();
        assert!(!config.backward_edges);
    }

    #[test]
    fn test_config_with_query_logging() {
        let config = Config::in_memory().with_query_logging();
        assert!(config.query_logging);
    }

    #[test]
    fn test_config_with_spill_path() {
        let config = Config::in_memory().with_spill_path("/tmp/spill");
        assert_eq!(
            config.spill_path.as_deref(),
            Some(std::path::Path::new("/tmp/spill"))
        );
    }

    #[test]
    fn test_config_with_memory_fraction() {
        let config = Config::in_memory().with_memory_fraction(0.5);
        assert!(config.memory_limit.is_some());
        assert!(config.memory_limit.unwrap() > 0);
    }

    #[test]
    fn test_config_with_adaptive() {
        let adaptive = AdaptiveConfig::default().with_threshold(5.0);
        let config = Config::in_memory().with_adaptive(adaptive);
        assert!((config.adaptive.threshold - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_without_adaptive() {
        let config = Config::in_memory().without_adaptive();
        assert!(!config.adaptive.enabled);
    }

    #[test]
    fn test_config_without_factorized_execution() {
        let config = Config::in_memory().without_factorized_execution();
        assert!(!config.factorized_execution);
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_memory_limit(512 * 1024 * 1024)
            .with_threads(4)
            .with_query_logging()
            .without_backward_edges()
            .with_spill_path("/tmp/spill");

        assert!(config.path.is_some());
        assert_eq!(config.memory_limit, Some(512 * 1024 * 1024));
        assert_eq!(config.threads, 4);
        assert!(config.query_logging);
        assert!(!config.backward_edges);
        assert!(config.spill_path.is_some());
    }

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert!(config.enabled);
        assert!((config.threshold - 3.0).abs() < f64::EPSILON);
        assert_eq!(config.min_rows, 1000);
        assert_eq!(config.max_reoptimizations, 3);
    }

    #[test]
    fn test_adaptive_config_disabled() {
        let config = AdaptiveConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_adaptive_config_with_threshold() {
        let config = AdaptiveConfig::default().with_threshold(10.0);
        assert!((config.threshold - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_config_with_min_rows() {
        let config = AdaptiveConfig::default().with_min_rows(500);
        assert_eq!(config.min_rows, 500);
    }

    #[test]
    fn test_adaptive_config_with_max_reoptimizations() {
        let config = AdaptiveConfig::default().with_max_reoptimizations(5);
        assert_eq!(config.max_reoptimizations, 5);
    }

    #[test]
    fn test_adaptive_config_builder_chaining() {
        let config = AdaptiveConfig::default()
            .with_threshold(2.0)
            .with_min_rows(100)
            .with_max_reoptimizations(10);
        assert!((config.threshold - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.min_rows, 100);
        assert_eq!(config.max_reoptimizations, 10);
    }

    // --- GraphModel tests ---

    #[test]
    fn test_graph_model_default_is_lpg() {
        assert_eq!(GraphModel::default(), GraphModel::Lpg);
    }

    #[test]
    fn test_graph_model_display() {
        assert_eq!(GraphModel::Lpg.to_string(), "LPG");
        assert_eq!(GraphModel::Rdf.to_string(), "RDF");
    }

    #[test]
    fn test_config_with_graph_model() {
        let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
        assert_eq!(config.graph_model, GraphModel::Rdf);
    }

    // --- DurabilityMode tests ---

    #[test]
    fn test_durability_mode_default_is_batch() {
        let mode = DurabilityMode::default();
        assert_eq!(
            mode,
            DurabilityMode::Batch {
                max_delay_ms: 100,
                max_records: 1000
            }
        );
    }

    #[test]
    fn test_config_with_wal_durability() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Sync);
        assert_eq!(config.wal_durability, DurabilityMode::Sync);
    }

    #[test]
    fn test_config_with_wal_durability_nosync() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::NoSync);
        assert_eq!(config.wal_durability, DurabilityMode::NoSync);
    }

    #[test]
    fn test_config_with_wal_durability_adaptive() {
        let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Adaptive {
            target_interval_ms: 50,
        });
        assert_eq!(
            config.wal_durability,
            DurabilityMode::Adaptive {
                target_interval_ms: 50
            }
        );
    }

    // --- schema_constraints tests ---

    #[test]
    fn test_config_with_schema_constraints() {
        let config = Config::in_memory().with_schema_constraints();
        assert!(config.schema_constraints);
    }

    // --- query_timeout tests ---

    #[test]
    fn test_config_with_query_timeout() {
        let config = Config::in_memory().with_query_timeout(Duration::from_secs(30));
        assert_eq!(config.query_timeout, Some(Duration::from_secs(30)));
    }

    // --- gc_interval tests ---

    #[test]
    fn test_config_with_gc_interval() {
        let config = Config::in_memory().with_gc_interval(50);
        assert_eq!(config.gc_interval, 50);
    }

    #[test]
    fn test_config_gc_disabled() {
        let config = Config::in_memory().with_gc_interval(0);
        assert_eq!(config.gc_interval, 0);
    }

    // --- validate() tests ---

    #[test]
    fn test_validate_default_config() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn test_validate_in_memory_config() {
        assert!(Config::in_memory().validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_memory_limit() {
        let config = Config::in_memory().with_memory_limit(0);
        assert_eq!(config.validate(), Err(ConfigError::ZeroMemoryLimit));
    }

    #[test]
    fn test_validate_rejects_zero_threads() {
        let config = Config::in_memory().with_threads(0);
        assert_eq!(config.validate(), Err(ConfigError::ZeroThreads));
    }

    #[test]
    fn test_validate_rejects_zero_wal_flush_interval() {
        let mut config = Config::in_memory();
        config.wal_flush_interval_ms = 0;
        assert_eq!(config.validate(), Err(ConfigError::ZeroWalFlushInterval));
    }

    #[cfg(not(feature = "rdf"))]
    #[test]
    fn test_validate_rejects_rdf_without_feature() {
        let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
        assert_eq!(config.validate(), Err(ConfigError::RdfFeatureRequired));
    }

    #[test]
    fn test_config_error_display() {
        assert_eq!(
            ConfigError::ZeroMemoryLimit.to_string(),
            "memory_limit must be greater than zero"
        );
        assert_eq!(
            ConfigError::ZeroThreads.to_string(),
            "threads must be greater than zero"
        );
        assert_eq!(
            ConfigError::ZeroWalFlushInterval.to_string(),
            "wal_flush_interval_ms must be greater than zero"
        );
        assert_eq!(
            ConfigError::RdfFeatureRequired.to_string(),
            "RDF graph model requires the `rdf` feature flag to be enabled"
        );
    }

    // --- Builder chaining with new fields ---

    #[test]
    fn test_config_full_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_graph_model(GraphModel::Lpg)
            .with_memory_limit(512 * 1024 * 1024)
            .with_threads(4)
            .with_query_logging()
            .with_wal_durability(DurabilityMode::Sync)
            .with_schema_constraints()
            .without_backward_edges()
            .with_spill_path("/tmp/spill")
            .with_query_timeout(Duration::from_secs(60));

        assert_eq!(config.graph_model, GraphModel::Lpg);
        assert!(config.path.is_some());
        assert_eq!(config.memory_limit, Some(512 * 1024 * 1024));
        assert_eq!(config.threads, 4);
        assert!(config.query_logging);
        assert_eq!(config.wal_durability, DurabilityMode::Sync);
        assert!(config.schema_constraints);
        assert!(!config.backward_edges);
        assert!(config.spill_path.is_some());
        assert_eq!(config.query_timeout, Some(Duration::from_secs(60)));
        assert!(config.validate().is_ok());
    }

    // --- AccessMode tests ---

    #[test]
    fn test_access_mode_default_is_read_write() {
        assert_eq!(AccessMode::default(), AccessMode::ReadWrite);
    }

    #[test]
    fn test_access_mode_display() {
        assert_eq!(AccessMode::ReadWrite.to_string(), "read-write");
        assert_eq!(AccessMode::ReadOnly.to_string(), "read-only");
    }

    #[test]
    fn test_config_with_access_mode() {
        let config = Config::persistent("/tmp/db").with_access_mode(AccessMode::ReadOnly);
        assert_eq!(config.access_mode, AccessMode::ReadOnly);
    }

    #[test]
    fn test_config_read_only() {
        let config = Config::read_only("/tmp/db.grafeo");
        assert_eq!(config.access_mode, AccessMode::ReadOnly);
        assert!(config.path.is_some());
        assert!(!config.wal_enabled);
    }

    #[test]
    fn test_config_default_is_read_write() {
        let config = Config::default();
        assert_eq!(config.access_mode, AccessMode::ReadWrite);
    }
}
