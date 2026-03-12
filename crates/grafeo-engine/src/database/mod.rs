//! The main database struct and operations.
//!
//! Start here with [`GrafeoDB`] - it's your handle to everything.
//!
//! Operations are split across focused submodules:
//! - `query` - Query execution (execute, execute_cypher, etc.)
//! - `crud` - Node/edge CRUD operations
//! - `index` - Property, vector, and text index management
//! - `search` - Vector, text, and hybrid search
//! - `embed` - Embedding model management
//! - `persistence` - Save, load, snapshots, iteration
//! - `admin` - Stats, introspection, diagnostics, CDC

mod admin;
mod crud;
#[cfg(feature = "embed")]
mod embed;
mod index;
mod persistence;
mod query;
mod search;
#[cfg(feature = "wal")]
pub(crate) mod wal_store;

#[cfg(feature = "wal")]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;

#[cfg(feature = "grafeo-file")]
use grafeo_adapters::storage::file::GrafeoFileManager;
#[cfg(feature = "wal")]
use grafeo_adapters::storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgWal, WalConfig, WalRecord, WalRecovery,
};
use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStoreMut;
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use crate::catalog::Catalog;
use crate::config::Config;
use crate::query::cache::QueryCache;
use crate::session::Session;
use crate::transaction::TransactionManager;

/// Your handle to a Grafeo database.
///
/// Start here. Create one with [`new_in_memory()`](Self::new_in_memory) for
/// quick experiments, or [`open()`](Self::open) for persistent storage.
/// Then grab a [`session()`](Self::session) to start querying.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// // Quick in-memory database
/// let db = GrafeoDB::new_in_memory();
///
/// // Add some data
/// db.create_node(&["Person"]);
///
/// // Query it
/// let session = db.session();
/// let result = session.execute("MATCH (p:Person) RETURN p")?;
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
pub struct GrafeoDB {
    /// Database configuration.
    pub(super) config: Config,
    /// The underlying graph store.
    pub(super) store: Arc<LpgStore>,
    /// Schema and metadata catalog shared across sessions.
    pub(super) catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "rdf")]
    pub(super) rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    pub(super) transaction_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    pub(super) buffer_manager: Arc<BufferManager>,
    /// Write-ahead log manager (if durability is enabled).
    #[cfg(feature = "wal")]
    pub(super) wal: Option<Arc<LpgWal>>,
    /// Shared WAL graph context tracker. Tracks which named graph was last
    /// written to the WAL, so concurrent sessions can emit `SwitchGraph`
    /// records only when the context actually changes.
    #[cfg(feature = "wal")]
    pub(super) wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
    /// Query cache for parsed and optimized plans.
    pub(super) query_cache: Arc<QueryCache>,
    /// Shared commit counter for auto-GC across sessions.
    pub(super) commit_counter: Arc<AtomicUsize>,
    /// Whether the database is open.
    pub(super) is_open: RwLock<bool>,
    /// Change data capture log for tracking mutations.
    #[cfg(feature = "cdc")]
    pub(super) cdc_log: Arc<crate::cdc::CdcLog>,
    /// Registered embedding models for text-to-vector conversion.
    #[cfg(feature = "embed")]
    pub(super) embedding_models:
        RwLock<hashbrown::HashMap<String, Arc<dyn crate::embedding::EmbeddingModel>>>,
    /// Single-file database manager (when using `.grafeo` format).
    #[cfg(feature = "grafeo-file")]
    pub(super) file_manager: Option<Arc<GrafeoFileManager>>,
    /// External graph store (when using with_store()).
    /// When set, sessions route queries through this store instead of the built-in LpgStore.
    pub(super) external_store: Option<Arc<dyn GraphStoreMut>>,
    /// Persistent graph context for one-shot `execute()` calls.
    /// When set, each call to `session()` pre-configures the session to this graph.
    /// Updated after every one-shot `execute()` to reflect `USE GRAPH` / `SESSION RESET`.
    current_graph: RwLock<Option<String>>,
}

impl GrafeoDB {
    /// Creates an in-memory database, fast to create, gone when dropped.
    ///
    /// Use this for tests, experiments, or when you don't need persistence.
    /// For data that survives restarts, use [`open()`](Self::open) instead.
    ///
    /// # Panics
    ///
    /// Panics if the internal arena allocator cannot be initialized (out of memory).
    /// Use [`with_config()`](Self::with_config) for a fallible alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory()).expect("In-memory database creation should not fail")
    }

    /// Opens a database at the given path, creating it if it doesn't exist.
    ///
    /// If you've used this path before, Grafeo recovers your data from the
    /// write-ahead log automatically. First open on a new path creates an
    /// empty database.
    ///
    /// # Errors
    ///
    /// Returns an error if the path isn't writable or recovery fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open("./my_social_network")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "wal")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(Config::persistent(path.as_ref()))
    }

    /// Creates a database with custom configuration.
    ///
    /// Use this when you need fine-grained control over memory limits,
    /// thread counts, or persistence settings. For most cases,
    /// [`new_in_memory()`](Self::new_in_memory) or [`open()`](Self::open)
    /// are simpler.
    ///
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, Config};
    ///
    /// // In-memory with a 512MB limit
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024);
    ///
    /// let db = GrafeoDB::with_config(config)?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn with_config(config: Config) -> Result<Self> {
        // Validate configuration before proceeding
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let store = Arc::new(LpgStore::new()?);
        #[cfg(feature = "rdf")]
        let rdf_store = Arc::new(RdfStore::new());
        let transaction_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: config
                .spill_path
                .clone()
                .or_else(|| config.path.as_ref().map(|p| p.join("spill"))),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Create catalog early so WAL replay can restore schema definitions
        let catalog = Arc::new(Catalog::new());

        // --- Single-file format (.grafeo) ---
        #[cfg(feature = "grafeo-file")]
        let file_manager: Option<Arc<GrafeoFileManager>> = if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                if Self::should_use_single_file(db_path, config.storage_format) {
                    let fm = if db_path.exists() && db_path.is_file() {
                        GrafeoFileManager::open(db_path)?
                    } else if !db_path.exists() {
                        GrafeoFileManager::create(db_path)?
                    } else {
                        // Path exists but is not a file (directory, etc.)
                        return Err(grafeo_common::utils::error::Error::Internal(format!(
                            "path exists but is not a file: {}",
                            db_path.display()
                        )));
                    };

                    // Load snapshot data from the file
                    let snapshot_data = fm.read_snapshot()?;
                    if !snapshot_data.is_empty() {
                        Self::apply_snapshot_data(
                            &store,
                            &catalog,
                            #[cfg(feature = "rdf")]
                            &rdf_store,
                            &snapshot_data,
                        )?;
                    }

                    // Recover sidecar WAL if present
                    if fm.has_sidecar_wal() {
                        let recovery = WalRecovery::new(fm.sidecar_wal_path());
                        let records = recovery.recover()?;
                        Self::apply_wal_records(
                            &store,
                            &catalog,
                            #[cfg(feature = "rdf")]
                            &rdf_store,
                            &records,
                        )?;
                    }

                    Some(Arc::new(fm))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Determine whether to use the WAL directory path (legacy) or sidecar
        #[cfg(feature = "wal")]
        let wal = if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                // When using single-file format, the WAL is a sidecar directory
                #[cfg(feature = "grafeo-file")]
                let wal_path = if let Some(ref fm) = file_manager {
                    let p = fm.sidecar_wal_path();
                    std::fs::create_dir_all(&p)?;
                    p
                } else {
                    // Legacy: WAL inside the database directory
                    std::fs::create_dir_all(db_path)?;
                    db_path.join("wal")
                };

                #[cfg(not(feature = "grafeo-file"))]
                let wal_path = {
                    std::fs::create_dir_all(db_path)?;
                    db_path.join("wal")
                };

                // For legacy WAL directory format, check if WAL exists and recover
                #[cfg(feature = "grafeo-file")]
                let is_single_file = file_manager.is_some();
                #[cfg(not(feature = "grafeo-file"))]
                let is_single_file = false;

                if !is_single_file && wal_path.exists() {
                    let recovery = WalRecovery::new(&wal_path);
                    let records = recovery.recover()?;
                    Self::apply_wal_records(
                        &store,
                        &catalog,
                        #[cfg(feature = "rdf")]
                        &rdf_store,
                        &records,
                    )?;
                }

                // Open/create WAL manager with configured durability
                let wal_durability = match config.wal_durability {
                    crate::config::DurabilityMode::Sync => WalDurabilityMode::Sync,
                    crate::config::DurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    } => WalDurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    },
                    crate::config::DurabilityMode::Adaptive { target_interval_ms } => {
                        WalDurabilityMode::Adaptive { target_interval_ms }
                    }
                    crate::config::DurabilityMode::NoSync => WalDurabilityMode::NoSync,
                };
                let wal_config = WalConfig {
                    durability: wal_durability,
                    ..WalConfig::default()
                };
                let wal_manager = LpgWal::with_config(&wal_path, wal_config)?;
                Some(Arc::new(wal_manager))
            } else {
                None
            }
        } else {
            None
        };

        // Create query cache with default capacity (1000 queries)
        let query_cache = Arc::new(QueryCache::default());

        Ok(Self {
            config,
            store,
            catalog,
            #[cfg(feature = "rdf")]
            rdf_store,
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager,
            external_store: None,
            current_graph: RwLock::new(None),
        })
    }

    /// Creates a database backed by a custom [`GraphStoreMut`] implementation.
    ///
    /// The external store handles all data persistence. WAL, CDC, and index
    /// management are the responsibility of the store implementation.
    ///
    /// Query execution (all 6 languages, optimizer, planner) works through the
    /// provided store. Admin operations (schema introspection, persistence,
    /// vector/text indexes) are not available on external stores.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use grafeo_engine::{GrafeoDB, Config};
    /// use grafeo_core::graph::GraphStoreMut;
    ///
    /// fn example(store: Arc<dyn GraphStoreMut>) -> grafeo_common::utils::error::Result<()> {
    ///     let db = GrafeoDB::with_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`GraphStoreMut`]: grafeo_core::graph::GraphStoreMut
    pub fn with_store(store: Arc<dyn GraphStoreMut>, config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;

        let dummy_store = Arc::new(LpgStore::new()?);
        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());

        Ok(Self {
            config,
            store: dummy_store,
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            #[cfg(feature = "grafeo-file")]
            file_manager: None,
            external_store: Some(store),
            current_graph: RwLock::new(None),
        })
    }

    /// Applies WAL records to restore the database state.
    ///
    /// Data mutation records are routed through a graph cursor that tracks
    /// `SwitchGraph` context markers, replaying mutations into the correct
    /// named graph (or the default graph when cursor is `None`).
    #[cfg(feature = "wal")]
    fn apply_wal_records(
        store: &Arc<LpgStore>,
        catalog: &Catalog,
        #[cfg(feature = "rdf")] rdf_store: &Arc<RdfStore>,
        records: &[WalRecord],
    ) -> Result<()> {
        use crate::catalog::{
            EdgeTypeDefinition, NodeTypeDefinition, PropertyDataType, TypeConstraint, TypedProperty,
        };
        use grafeo_common::utils::error::Error;

        // Graph cursor: tracks which named graph receives data mutations.
        // `None` means the default graph.
        let mut current_graph: Option<String> = None;
        let mut target_store: Arc<LpgStore> = Arc::clone(store);

        for record in records {
            match record {
                // --- Named graph lifecycle ---
                WalRecord::CreateNamedGraph { name } => {
                    let _ = store.create_graph(name);
                }
                WalRecord::DropNamedGraph { name } => {
                    store.drop_graph(name);
                    // Reset cursor if the dropped graph was active
                    if current_graph.as_deref() == Some(name.as_str()) {
                        current_graph = None;
                        target_store = Arc::clone(store);
                    }
                }
                WalRecord::SwitchGraph { name } => {
                    current_graph.clone_from(name);
                    target_store = match &current_graph {
                        None => Arc::clone(store),
                        Some(graph_name) => store
                            .graph_or_create(graph_name)
                            .map_err(|e| Error::Internal(e.to_string()))?,
                    };
                }

                // --- Data mutations: routed through target_store ---
                WalRecord::CreateNode { id, labels } => {
                    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                    target_store.create_node_with_id(*id, &label_refs)?;
                }
                WalRecord::DeleteNode { id } => {
                    target_store.delete_node(*id);
                }
                WalRecord::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                } => {
                    target_store.create_edge_with_id(*id, *src, *dst, edge_type)?;
                }
                WalRecord::DeleteEdge { id } => {
                    target_store.delete_edge(*id);
                }
                WalRecord::SetNodeProperty { id, key, value } => {
                    target_store.set_node_property(*id, key, value.clone());
                }
                WalRecord::SetEdgeProperty { id, key, value } => {
                    target_store.set_edge_property(*id, key, value.clone());
                }
                WalRecord::AddNodeLabel { id, label } => {
                    target_store.add_label(*id, label);
                }
                WalRecord::RemoveNodeLabel { id, label } => {
                    target_store.remove_label(*id, label);
                }
                WalRecord::RemoveNodeProperty { id, key } => {
                    target_store.remove_node_property(*id, key);
                }
                WalRecord::RemoveEdgeProperty { id, key } => {
                    target_store.remove_edge_property(*id, key);
                }

                // --- Schema DDL replay (always on root catalog) ---
                WalRecord::CreateNodeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = NodeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        parent_types: Vec::new(),
                    };
                    let _ = catalog.register_node_type(def);
                }
                WalRecord::DropNodeType { name } => {
                    let _ = catalog.drop_node_type(name);
                }
                WalRecord::CreateEdgeType {
                    name,
                    properties,
                    constraints,
                } => {
                    let def = EdgeTypeDefinition {
                        name: name.clone(),
                        properties: properties
                            .iter()
                            .map(|(n, t, nullable)| TypedProperty {
                                name: n.clone(),
                                data_type: PropertyDataType::from_type_name(t),
                                nullable: *nullable,
                                default_value: None,
                            })
                            .collect(),
                        constraints: constraints
                            .iter()
                            .map(|(kind, props)| match kind.as_str() {
                                "unique" => TypeConstraint::Unique(props.clone()),
                                "primary_key" => TypeConstraint::PrimaryKey(props.clone()),
                                "not_null" if !props.is_empty() => {
                                    TypeConstraint::NotNull(props[0].clone())
                                }
                                _ => TypeConstraint::Unique(props.clone()),
                            })
                            .collect(),
                        source_node_types: Vec::new(),
                        target_node_types: Vec::new(),
                    };
                    let _ = catalog.register_edge_type_def(def);
                }
                WalRecord::DropEdgeType { name } => {
                    let _ = catalog.drop_edge_type_def(name);
                }
                WalRecord::CreateIndex { .. } | WalRecord::DropIndex { .. } => {
                    // Index recreation is handled by the store on startup
                    // (indexes are rebuilt from data, not WAL)
                }
                WalRecord::CreateConstraint { .. } | WalRecord::DropConstraint { .. } => {
                    // Constraint definitions are part of type definitions
                    // and replayed via CreateNodeType/CreateEdgeType
                }
                WalRecord::CreateGraphType {
                    name,
                    node_types,
                    edge_types,
                    open,
                } => {
                    use crate::catalog::GraphTypeDefinition;
                    let def = GraphTypeDefinition {
                        name: name.clone(),
                        allowed_node_types: node_types.clone(),
                        allowed_edge_types: edge_types.clone(),
                        open: *open,
                    };
                    let _ = catalog.register_graph_type(def);
                }
                WalRecord::DropGraphType { name } => {
                    let _ = catalog.drop_graph_type(name);
                }
                WalRecord::CreateSchema { name } => {
                    let _ = catalog.register_schema_namespace(name.clone());
                }
                WalRecord::DropSchema { name } => {
                    let _ = catalog.drop_schema_namespace(name);
                }

                WalRecord::AlterNodeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_node_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_node_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterEdgeType { name, alterations } => {
                    for (action, prop_name, type_name, nullable) in alterations {
                        match action.as_str() {
                            "add" => {
                                let prop = TypedProperty {
                                    name: prop_name.clone(),
                                    data_type: PropertyDataType::from_type_name(type_name),
                                    nullable: *nullable,
                                    default_value: None,
                                };
                                let _ = catalog.alter_edge_type_add_property(name, prop);
                            }
                            "drop" => {
                                let _ = catalog.alter_edge_type_drop_property(name, prop_name);
                            }
                            _ => {}
                        }
                    }
                }
                WalRecord::AlterGraphType { name, alterations } => {
                    for (action, type_name) in alterations {
                        match action.as_str() {
                            "add_node" => {
                                let _ =
                                    catalog.alter_graph_type_add_node_type(name, type_name.clone());
                            }
                            "drop_node" => {
                                let _ = catalog.alter_graph_type_drop_node_type(name, type_name);
                            }
                            "add_edge" => {
                                let _ =
                                    catalog.alter_graph_type_add_edge_type(name, type_name.clone());
                            }
                            "drop_edge" => {
                                let _ = catalog.alter_graph_type_drop_edge_type(name, type_name);
                            }
                            _ => {}
                        }
                    }
                }

                WalRecord::CreateProcedure {
                    name,
                    params,
                    returns,
                    body,
                } => {
                    use crate::catalog::ProcedureDefinition;
                    let def = ProcedureDefinition {
                        name: name.clone(),
                        params: params.clone(),
                        returns: returns.clone(),
                        body: body.clone(),
                    };
                    let _ = catalog.register_procedure(def);
                }
                WalRecord::DropProcedure { name } => {
                    let _ = catalog.drop_procedure(name);
                }

                // --- RDF triple replay ---
                #[cfg(feature = "rdf")]
                WalRecord::InsertRdfTriple {
                    subject,
                    predicate,
                    object,
                    graph,
                } => {
                    use grafeo_core::graph::rdf::Term;
                    if let (Some(s), Some(p), Some(o)) = (
                        Term::from_ntriples(subject),
                        Term::from_ntriples(predicate),
                        Term::from_ntriples(object),
                    ) {
                        let triple = grafeo_core::graph::rdf::Triple::new(s, p, o);
                        let target = match graph {
                            Some(name) => rdf_store.graph_or_create(name),
                            None => Arc::clone(rdf_store),
                        };
                        target.insert(triple);
                    }
                }
                #[cfg(feature = "rdf")]
                WalRecord::DeleteRdfTriple {
                    subject,
                    predicate,
                    object,
                    graph,
                } => {
                    use grafeo_core::graph::rdf::Term;
                    if let (Some(s), Some(p), Some(o)) = (
                        Term::from_ntriples(subject),
                        Term::from_ntriples(predicate),
                        Term::from_ntriples(object),
                    ) {
                        let triple = grafeo_core::graph::rdf::Triple::new(s, p, o);
                        let target = match graph {
                            Some(name) => rdf_store.graph_or_create(name),
                            None => Arc::clone(rdf_store),
                        };
                        target.remove(&triple);
                    }
                }
                #[cfg(feature = "rdf")]
                WalRecord::ClearRdfGraph { graph } => {
                    rdf_store.clear_graph(graph.as_deref());
                }
                #[cfg(feature = "rdf")]
                WalRecord::CreateRdfGraph { name } => {
                    let _ = rdf_store.create_graph(name);
                }
                #[cfg(feature = "rdf")]
                WalRecord::DropRdfGraph { name } => match name {
                    None => rdf_store.clear(),
                    Some(graph_name) => {
                        rdf_store.drop_graph(graph_name);
                    }
                },
                // Skip RDF records when rdf feature is disabled
                #[cfg(not(feature = "rdf"))]
                WalRecord::InsertRdfTriple { .. }
                | WalRecord::DeleteRdfTriple { .. }
                | WalRecord::ClearRdfGraph { .. }
                | WalRecord::CreateRdfGraph { .. }
                | WalRecord::DropRdfGraph { .. } => {}

                WalRecord::TransactionCommit { .. }
                | WalRecord::TransactionAbort { .. }
                | WalRecord::Checkpoint { .. } => {
                    // Transaction control records don't need replay action
                    // (recovery already filtered to only committed transactions)
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // Single-file format helpers
    // =========================================================================

    /// Returns `true` if the given path should use single-file format.
    #[cfg(feature = "grafeo-file")]
    fn should_use_single_file(
        path: &std::path::Path,
        configured: crate::config::StorageFormat,
    ) -> bool {
        use crate::config::StorageFormat;
        match configured {
            StorageFormat::SingleFile => true,
            StorageFormat::WalDirectory => false,
            StorageFormat::Auto => {
                // Existing file: check magic bytes
                if path.is_file() {
                    if let Ok(mut f) = std::fs::File::open(path) {
                        use std::io::Read;
                        let mut magic = [0u8; 4];
                        if f.read_exact(&mut magic).is_ok()
                            && magic == grafeo_adapters::storage::file::MAGIC
                        {
                            return true;
                        }
                    }
                    return false;
                }
                // Existing directory: legacy format
                if path.is_dir() {
                    return false;
                }
                // New path: check extension
                path.extension().is_some_and(|ext| ext == "grafeo")
            }
        }
    }

    /// Applies snapshot data (from a `.grafeo` file) to restore the store and catalog.
    #[cfg(feature = "grafeo-file")]
    fn apply_snapshot_data(
        store: &Arc<LpgStore>,
        catalog: &Arc<crate::catalog::Catalog>,
        #[cfg(feature = "rdf")] rdf_store: &Arc<RdfStore>,
        data: &[u8],
    ) -> Result<()> {
        persistence::load_snapshot_into_store(
            store,
            catalog,
            #[cfg(feature = "rdf")]
            rdf_store,
            data,
        )
    }

    // =========================================================================
    // Session & Configuration
    // =========================================================================

    /// Opens a new session for running queries.
    ///
    /// Sessions are cheap to create: spin up as many as you need. Each
    /// gets its own transaction context, so concurrent sessions won't
    /// block each other on reads.
    ///
    /// # Panics
    ///
    /// Panics if the database was configured with an external graph store and
    /// the internal arena allocator cannot be initialized (out of memory).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Run queries through the session
    /// let result = session.execute("MATCH (n) RETURN count(n)")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn session(&self) -> Session {
        let session_cfg = || crate::session::SessionConfig {
            transaction_manager: Arc::clone(&self.transaction_manager),
            query_cache: Arc::clone(&self.query_cache),
            catalog: Arc::clone(&self.catalog),
            adaptive_config: self.config.adaptive.clone(),
            factorized_execution: self.config.factorized_execution,
            graph_model: self.config.graph_model,
            query_timeout: self.config.query_timeout,
            commit_counter: Arc::clone(&self.commit_counter),
            gc_interval: self.config.gc_interval,
        };

        if let Some(ref ext_store) = self.external_store {
            return Session::with_external_store(Arc::clone(ext_store), session_cfg())
                .expect("arena allocation for external store session");
        }

        #[cfg(feature = "rdf")]
        let mut session = Session::with_rdf_store_and_adaptive(
            Arc::clone(&self.store),
            Arc::clone(&self.rdf_store),
            session_cfg(),
        );
        #[cfg(not(feature = "rdf"))]
        let mut session = Session::with_adaptive(Arc::clone(&self.store), session_cfg());

        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            session.set_wal(Arc::clone(wal), Arc::clone(&self.wal_graph_context));
        }

        #[cfg(feature = "cdc")]
        session.set_cdc_log(Arc::clone(&self.cdc_log));

        // Propagate persistent graph context to the new session
        if let Some(ref graph) = *self.current_graph.read() {
            session.use_graph(graph);
        }

        // Suppress unused_mut when cdc/wal are disabled
        let _ = &mut session;

        session
    }

    /// Returns the current graph name, if any.
    ///
    /// This is the persistent graph context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `USE GRAPH`, `SESSION SET GRAPH`,
    /// or `SESSION RESET`.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_graph.read().clone()
    }

    /// Sets the current graph context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `USE GRAPH <name>` but without creating a session.
    /// Pass `None` to reset to the default graph.
    pub fn set_current_graph(&self, name: Option<&str>) {
        *self.current_graph.write() = name.map(ToString::to_string);
    }

    /// Returns the adaptive execution configuration.
    #[must_use]
    pub fn adaptive_config(&self) -> &crate::config::AdaptiveConfig {
        &self.config.adaptive
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the graph data model of this database.
    #[must_use]
    pub fn graph_model(&self) -> crate::config::GraphModel {
        self.config.graph_model
    }

    /// Returns the configured memory limit in bytes, if any.
    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.config.memory_limit
    }

    /// Returns the underlying (default) store.
    ///
    /// This provides direct access to the LPG store for algorithm implementations
    /// and admin operations (index management, schema introspection, MVCC internals).
    ///
    /// For code that only needs read/write graph operations, prefer
    /// [`graph_store()`](Self::graph_store) which returns the trait interface.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }

    /// Returns the LPG store for the currently active graph.
    ///
    /// If [`current_graph`](Self::current_graph) is `None` or `"default"`, returns
    /// the default store. Otherwise looks up the named graph in the root store.
    /// Falls back to the default store if the named graph does not exist.
    #[allow(dead_code)] // Reserved for future graph-aware CRUD methods
    fn active_store(&self) -> Arc<LpgStore> {
        let graph_name = self.current_graph.read().clone();
        match graph_name {
            None => Arc::clone(&self.store),
            Some(ref name) if name.eq_ignore_ascii_case("default") => Arc::clone(&self.store),
            Some(ref name) => self
                .store
                .graph(name)
                .unwrap_or_else(|| Arc::clone(&self.store)),
        }
    }

    // === Named Graph Management ===

    /// Creates a named graph. Returns `true` if created, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if arena allocation fails.
    pub fn create_graph(&self, name: &str) -> Result<bool> {
        Ok(self.store.create_graph(name)?)
    }

    /// Drops a named graph. Returns `true` if dropped, `false` if it did not exist.
    pub fn drop_graph(&self, name: &str) -> bool {
        self.store.drop_graph(name)
    }

    /// Returns all named graph names.
    #[must_use]
    pub fn list_graphs(&self) -> Vec<String> {
        self.store.graph_names()
    }

    /// Returns the graph store as a trait object.
    ///
    /// This provides the [`GraphStoreMut`] interface for code that should work
    /// with any storage backend. Use this when you only need graph read/write
    /// operations and don't need admin methods like index management.
    ///
    /// [`GraphStoreMut`]: grafeo_core::graph::GraphStoreMut
    #[must_use]
    pub fn graph_store(&self) -> Arc<dyn GraphStoreMut> {
        if let Some(ref ext_store) = self.external_store {
            Arc::clone(ext_store)
        } else {
            Arc::clone(&self.store) as Arc<dyn GraphStoreMut>
        }
    }

    /// Garbage collects old MVCC versions that are no longer visible.
    ///
    /// Determines the minimum epoch required by active transactions and prunes
    /// version chains older than that threshold. Also cleans up completed
    /// transaction metadata in the transaction manager.
    pub fn gc(&self) {
        let min_epoch = self.transaction_manager.min_active_epoch();
        self.store.gc_versions(min_epoch);
        self.transaction_manager.gc();
    }

    /// Returns the buffer manager for memory-aware operations.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns the query cache.
    #[must_use]
    pub fn query_cache(&self) -> &Arc<QueryCache> {
        &self.query_cache
    }

    /// Clears all cached query plans.
    ///
    /// This is called automatically after DDL operations, but can also be
    /// invoked manually after external schema changes (e.g., WAL replay,
    /// import) or when you want to force re-optimization of all queries.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Closes the database, flushing all pending writes.
    ///
    /// For persistent databases, this ensures everything is safely on disk.
    /// Called automatically when the database is dropped, but you can call
    /// it explicitly if you need to guarantee durability at a specific point.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions).
    pub fn close(&self) -> Result<()> {
        let mut is_open = self.is_open.write();
        if !*is_open {
            return Ok(());
        }

        // For single-file format: checkpoint to .grafeo file, then clean up sidecar WAL.
        // We must do this BEFORE the WAL close path because checkpoint_to_file
        // removes the sidecar WAL directory.
        #[cfg(feature = "grafeo-file")]
        let is_single_file = self.file_manager.is_some();
        #[cfg(not(feature = "grafeo-file"))]
        let is_single_file = false;

        #[cfg(feature = "grafeo-file")]
        if let Some(ref fm) = self.file_manager {
            // Flush WAL first so all records are on disk before we snapshot
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                wal.sync()?;
            }
            self.checkpoint_to_file(fm)?;

            // Release WAL file handles before removing sidecar directory.
            // On Windows, open handles prevent directory deletion.
            #[cfg(feature = "wal")]
            if let Some(ref wal) = self.wal {
                wal.close_active_log();
            }

            fm.remove_sidecar_wal()?;
            fm.close()?;
        }

        // Commit and checkpoint WAL (legacy directory format only)
        #[cfg(feature = "wal")]
        if !is_single_file && let Some(ref wal) = self.wal {
            let epoch = self.store.current_epoch();

            // Use the last assigned transaction ID, or create a checkpoint-only tx
            let checkpoint_tx = self
                .transaction_manager
                .last_assigned_transaction_id()
                .unwrap_or_else(|| {
                    // No transactions have been started; begin one for checkpoint
                    self.transaction_manager.begin()
                });

            // Log a TransactionCommit to mark all pending records as committed
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: checkpoint_tx,
            })?;

            // Then checkpoint
            wal.checkpoint(checkpoint_tx, epoch)?;
            wal.sync()?;
        }

        *is_open = false;
        Ok(())
    }

    /// Returns the typed WAL if available.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn wal(&self) -> Option<&Arc<LpgWal>> {
        self.wal.as_ref()
    }

    /// Logs a WAL record if WAL is enabled.
    #[cfg(feature = "wal")]
    pub(super) fn log_wal(&self, record: &WalRecord) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.log(record)?;
        }
        Ok(())
    }

    /// Writes the current database snapshot to the `.grafeo` file.
    ///
    /// Does NOT remove the sidecar WAL: callers that want to clean up
    /// the sidecar (e.g. `close()`) should call `fm.remove_sidecar_wal()`
    /// separately after this returns.
    #[cfg(feature = "grafeo-file")]
    fn checkpoint_to_file(&self, fm: &GrafeoFileManager) -> Result<()> {
        use grafeo_core::testing::crash::maybe_crash;

        maybe_crash("checkpoint_to_file:before_export");
        let snapshot_data = self.export_snapshot()?;
        maybe_crash("checkpoint_to_file:after_export");

        let epoch = self.store.current_epoch();
        let transaction_id = self
            .transaction_manager
            .last_assigned_transaction_id()
            .map_or(0, |t| t.0);
        let node_count = self.store.node_count() as u64;
        let edge_count = self.store.edge_count() as u64;

        fm.write_snapshot(
            &snapshot_data,
            epoch.0,
            transaction_id,
            node_count,
            edge_count,
        )?;

        maybe_crash("checkpoint_to_file:after_write_snapshot");
        Ok(())
    }

    /// Returns the file manager if using single-file format.
    #[cfg(feature = "grafeo-file")]
    #[must_use]
    pub fn file_manager(&self) -> Option<&Arc<GrafeoFileManager>> {
        self.file_manager.as_ref()
    }
}

impl Drop for GrafeoDB {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            tracing::error!("Error closing database: {}", e);
        }
    }
}

impl crate::admin::AdminService for GrafeoDB {
    fn info(&self) -> crate::admin::DatabaseInfo {
        self.info()
    }

    fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        self.detailed_stats()
    }

    fn schema(&self) -> crate::admin::SchemaInfo {
        self.schema()
    }

    fn validate(&self) -> crate::admin::ValidationResult {
        self.validate()
    }

    fn wal_status(&self) -> crate::admin::WalStatus {
        self.wal_status()
    }

    fn wal_checkpoint(&self) -> Result<()> {
        self.wal_checkpoint()
    }
}

// =========================================================================
// Query Result Types
// =========================================================================

/// The result of running a query.
///
/// Contains rows and columns, like a table. Use [`iter()`](Self::iter) to
/// loop through rows, or [`scalar()`](Self::scalar) if you expect a single value.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// let db = GrafeoDB::new_in_memory();
/// db.create_node(&["Person"]);
///
/// let result = db.execute("MATCH (p:Person) RETURN count(p) AS total")?;
///
/// // Check what we got
/// println!("Columns: {:?}", result.columns);
/// println!("Rows: {}", result.row_count());
///
/// // Iterate through results
/// for row in result.iter() {
///     println!("{:?}", row);
/// }
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Debug)]
pub struct QueryResult {
    /// Column names from the RETURN clause.
    pub columns: Vec<String>,
    /// Column types - useful for distinguishing NodeId/EdgeId from plain integers.
    pub column_types: Vec<grafeo_common::types::LogicalType>,
    /// The actual result rows.
    pub rows: Vec<Vec<grafeo_common::types::Value>>,
    /// Query execution time in milliseconds (if timing was enabled).
    pub execution_time_ms: Option<f64>,
    /// Number of rows scanned during query execution (estimate).
    pub rows_scanned: Option<u64>,
    /// Status message for DDL and session commands (e.g., "Created node type 'Person'").
    pub status_message: Option<String>,
    /// GQLSTATUS code per ISO/IEC 39075:2024, sec 23.
    pub gql_status: grafeo_common::utils::GqlStatus,
}

impl QueryResult {
    /// Creates a fully empty query result (no columns, no rows).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a query result with only a status message (for DDL commands).
    #[must_use]
    pub fn status(msg: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: Some(msg.into()),
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result with column types.
    #[must_use]
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<grafeo_common::types::LogicalType>,
    ) -> Self {
        Self {
            columns,
            column_types,
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Sets the execution metrics on this result.
    pub fn with_metrics(mut self, execution_time_ms: f64, rows_scanned: u64) -> Self {
        self.execution_time_ms = Some(execution_time_ms);
        self.rows_scanned = Some(rows_scanned);
        self
    }

    /// Returns the execution time in milliseconds, if available.
    #[must_use]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Returns the number of rows scanned, if available.
    #[must_use]
    pub fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Extracts a single value from the result.
    ///
    /// Use this when your query returns exactly one row with one column,
    /// like `RETURN count(n)` or `RETURN sum(p.amount)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the result has multiple rows or columns.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.rows.len() != 1 || self.columns.len() != 1 {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows[0][0])
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<grafeo_common::types::Value>> {
        self.rows.iter()
    }
}

/// Converts a [`grafeo_common::types::Value`] to a concrete Rust type.
///
/// Implemented for common types like `i64`, `f64`, `String`, and `bool`.
/// Used by [`QueryResult::scalar()`] to extract typed values.
pub trait FromValue: Sized {
    /// Attempts the conversion, returning an error on type mismatch.
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            grafeo_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config = Config::in_memory().with_threads(4).with_query_logging();

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = GrafeoDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_persistent_database_recovery() {
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Create database and add some data
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));

            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));

            let _edge = db.create_edge(alix, gus, "KNOWS");

            // Explicitly close to flush WAL
            db.close().unwrap();
        }

        // Reopen and verify data was recovered
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            // Verify nodes exist
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node0.is_some());

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node1.is_some());
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]);
        db.delete_node(node);

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_recovery_multiple_sessions() {
        // Tests that WAL recovery works correctly across multiple open/close cycles
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_session_db");

        // Session 1: Create initial data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));
            db.close().unwrap();
        }

        // Session 2: Add more data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 1); // Previous data recovered
            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));
            db.close().unwrap();
        }

        // Session 3: Verify all data
        {
            let db = GrafeoDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 2);

            // Verify properties were recovered correctly
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0)).unwrap();
            assert!(node0.labels.iter().any(|l| l.as_str() == "Person"));

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1)).unwrap();
            assert!(node1.labels.iter().any(|l| l.as_str() == "Person"));
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_database_consistency_after_mutations() {
        // Tests that database remains consistent after a series of create/delete operations
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("consistency_db");

        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Create nodes
            let a = db.create_node(&["Node"]);
            let b = db.create_node(&["Node"]);
            let c = db.create_node(&["Node"]);

            // Create edges
            let e1 = db.create_edge(a, b, "LINKS");
            let _e2 = db.create_edge(b, c, "LINKS");

            // Delete middle node and its edge
            db.delete_edge(e1);
            db.delete_node(b);

            // Set properties on remaining nodes
            db.set_node_property(a, "value", Value::Int64(1));
            db.set_node_property(c, "value", Value::Int64(3));

            db.close().unwrap();
        }

        // Reopen and verify consistency
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            // Should have 2 nodes (a and c), b was deleted
            // Note: node_count includes deleted nodes in some implementations
            // What matters is that the non-deleted nodes are accessible
            let node_a = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node_a.is_some());

            let node_c = db.get_node(grafeo_common::types::NodeId::new(2));
            assert!(node_c.is_some());

            // Middle node should be deleted
            let node_b = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node_b.is_none());
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_close_is_idempotent() {
        // Calling close() multiple times should not cause errors
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("close_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();
        db.create_node(&["Test"]);

        // First close should succeed
        assert!(db.close().is_ok());

        // Second close should also succeed (idempotent)
        assert!(db.close().is_ok());
    }

    #[test]
    fn test_query_result_has_metrics() {
        // Verifies that query results include execution metrics
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            let result = db.execute("MATCH (n:Person) RETURN n").unwrap();

            // Metrics should be populated
            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert!(result.execution_time_ms.unwrap() >= 0.0);
            assert_eq!(result.rows_scanned.unwrap(), 2);
        }
    }

    #[test]
    fn test_empty_query_result_metrics() {
        // Verifies metrics are correct for queries returning no results
        let db = GrafeoDB::new_in_memory();
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            // Query that matches nothing
            let result = db.execute("MATCH (n:NonExistent) RETURN n").unwrap();

            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert_eq!(result.rows_scanned.unwrap(), 0);
        }
    }

    #[cfg(feature = "cdc")]
    mod cdc_integration {
        use super::*;

        #[test]
        fn test_node_lifecycle_history() {
            let db = GrafeoDB::new_in_memory();

            // Create
            let id = db.create_node(&["Person"]);
            // Update
            db.set_node_property(id, "name", "Alix".into());
            db.set_node_property(id, "name", "Gus".into());
            // Delete
            db.delete_node(id);

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 4); // create + 2 updates + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert!(history[1].before.is_none()); // first set_node_property has no prior value
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Update);
            assert!(history[2].before.is_some()); // second update has prior "Alix"
            assert_eq!(history[3].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_edge_lifecycle_history() {
            let db = GrafeoDB::new_in_memory();

            let alix = db.create_node(&["Person"]);
            let gus = db.create_node(&["Person"]);
            let edge = db.create_edge(alix, gus, "KNOWS");
            db.set_edge_property(edge, "since", 2024i64.into());
            db.delete_edge(edge);

            let history = db.history(edge).unwrap();
            assert_eq!(history.len(), 3); // create + update + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_create_node_with_props_cdc() {
            let db = GrafeoDB::new_in_memory();

            let id = db.create_node_with_props(
                &["Person"],
                vec![
                    ("name", grafeo_common::types::Value::from("Alix")),
                    ("age", grafeo_common::types::Value::from(30i64)),
                ],
            );

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1);
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            // Props should be captured
            let after = history[0].after.as_ref().unwrap();
            assert_eq!(after.len(), 2);
        }

        #[test]
        fn test_changes_between() {
            let db = GrafeoDB::new_in_memory();

            let id1 = db.create_node(&["A"]);
            let _id2 = db.create_node(&["B"]);
            db.set_node_property(id1, "x", 1i64.into());

            // All events should be at the same epoch (in-memory, epoch doesn't advance without tx)
            let changes = db
                .changes_between(
                    grafeo_common::types::EpochId(0),
                    grafeo_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert_eq!(changes.len(), 3); // 2 creates + 1 update
        }
    }

    #[test]
    fn test_with_store_basic() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", "Alix".into());

        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let result = db.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_session() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let session = db.session();
        let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_mutations() {
        use grafeo_core::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new().unwrap());
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(graph_store, Config::in_memory()).unwrap();

        let mut session = db.session();

        // Use an explicit transaction so INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);

        session.commit().unwrap();
    }
}
