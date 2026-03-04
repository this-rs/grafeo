//! Lightweight handles for database interaction.
//!
//! A session is your conversation with the database. Each session can have
//! its own transaction state, so concurrent sessions don't interfere with
//! each other. Sessions are cheap to create - spin up as many as you need.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use grafeo_common::types::{EdgeId, EpochId, NodeId, TxId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::Direction;
use grafeo_core::graph::GraphStoreMut;
use grafeo_core::graph::lpg::{Edge, LpgStore, Node};
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use crate::catalog::{Catalog, CatalogConstraintValidator};
use crate::config::{AdaptiveConfig, GraphModel};
use crate::database::QueryResult;
use crate::query::cache::QueryCache;
use crate::transaction::TransactionManager;

/// Your handle to the database - execute queries and manage transactions.
///
/// Get one from [`GrafeoDB::session()`](crate::GrafeoDB::session). Each session
/// tracks its own transaction state, so you can have multiple concurrent
/// sessions without them interfering.
pub struct Session {
    /// The underlying store.
    store: Arc<LpgStore>,
    /// Graph store trait object for pluggable storage backends.
    graph_store: Arc<dyn GraphStoreMut>,
    /// Schema and metadata catalog shared across sessions.
    catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "rdf")]
    rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Query cache shared across sessions.
    query_cache: Arc<QueryCache>,
    /// Current transaction ID (if any).
    current_tx: Option<TxId>,
    /// Whether the session is in auto-commit mode.
    auto_commit: bool,
    /// Adaptive execution configuration.
    #[allow(dead_code)]
    adaptive_config: AdaptiveConfig,
    /// Whether to use factorized execution for multi-hop queries.
    factorized_execution: bool,
    /// The graph data model this session operates on.
    graph_model: GraphModel,
    /// Maximum time a query may run before being cancelled.
    query_timeout: Option<Duration>,
    /// Shared commit counter for triggering auto-GC.
    commit_counter: Arc<AtomicUsize>,
    /// GC every N commits (0 = disabled).
    gc_interval: usize,
    /// Node count at the start of the current transaction (for PreparedCommit stats).
    tx_start_node_count: usize,
    /// Edge count at the start of the current transaction (for PreparedCommit stats).
    tx_start_edge_count: usize,
    /// WAL for logging schema changes.
    #[cfg(feature = "wal")]
    wal: Option<Arc<grafeo_adapters::storage::wal::LpgWal>>,
    /// CDC log for change tracking.
    #[cfg(feature = "cdc")]
    cdc_log: Arc<crate::cdc::CdcLog>,
    /// Current graph name (for multi-graph USE GRAPH support). None = default graph.
    current_graph: parking_lot::Mutex<Option<String>>,
    /// Session time zone override.
    time_zone: parking_lot::Mutex<Option<String>>,
    /// Session-level parameters (SET PARAMETER).
    session_params:
        parking_lot::Mutex<std::collections::HashMap<String, grafeo_common::types::Value>>,
    /// Override epoch for time-travel queries (None = use transaction/current epoch).
    viewing_epoch_override: parking_lot::Mutex<Option<EpochId>>,
}

impl Session {
    /// Creates a new session with adaptive execution configuration.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn with_adaptive(
        store: Arc<LpgStore>,
        tx_manager: Arc<TransactionManager>,
        query_cache: Arc<QueryCache>,
        catalog: Arc<Catalog>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
        graph_model: GraphModel,
        query_timeout: Option<Duration>,
        commit_counter: Arc<AtomicUsize>,
        gc_interval: usize,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        Self {
            store,
            graph_store,
            catalog,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            query_cache,
            current_tx: None,
            auto_commit: true,
            adaptive_config,
            factorized_execution,
            graph_model,
            query_timeout,
            commit_counter,
            gc_interval,
            tx_start_node_count: 0,
            tx_start_edge_count: 0,
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            current_graph: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
        }
    }

    /// Sets the WAL for this session (shared with the database).
    #[cfg(feature = "wal")]
    pub(crate) fn set_wal(&mut self, wal: Arc<grafeo_adapters::storage::wal::LpgWal>) {
        self.wal = Some(wal);
    }

    /// Sets the CDC log for this session (shared with the database).
    #[cfg(feature = "cdc")]
    pub(crate) fn set_cdc_log(&mut self, cdc_log: Arc<crate::cdc::CdcLog>) {
        self.cdc_log = cdc_log;
    }

    /// Creates a new session with RDF store and adaptive configuration.
    #[cfg(feature = "rdf")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_rdf_store_and_adaptive(
        store: Arc<LpgStore>,
        rdf_store: Arc<RdfStore>,
        tx_manager: Arc<TransactionManager>,
        query_cache: Arc<QueryCache>,
        catalog: Arc<Catalog>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
        graph_model: GraphModel,
        query_timeout: Option<Duration>,
        commit_counter: Arc<AtomicUsize>,
        gc_interval: usize,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        Self {
            store,
            graph_store,
            catalog,
            rdf_store,
            tx_manager,
            query_cache,
            current_tx: None,
            auto_commit: true,
            adaptive_config,
            factorized_execution,
            graph_model,
            query_timeout,
            commit_counter,
            gc_interval,
            tx_start_node_count: 0,
            tx_start_edge_count: 0,
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            current_graph: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
        }
    }

    /// Creates a session backed by an external graph store.
    ///
    /// The external store handles all data operations. Transaction management
    /// (begin/commit/rollback) is not supported for external stores.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn with_external_store(
        store: Arc<dyn GraphStoreMut>,
        tx_manager: Arc<TransactionManager>,
        query_cache: Arc<QueryCache>,
        catalog: Arc<Catalog>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
        graph_model: GraphModel,
        query_timeout: Option<Duration>,
        commit_counter: Arc<AtomicUsize>,
        gc_interval: usize,
    ) -> Self {
        Self {
            store: Arc::new(LpgStore::new()), // dummy for LpgStore-specific ops
            graph_store: store,
            catalog,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            query_cache,
            current_tx: None,
            auto_commit: true,
            adaptive_config,
            factorized_execution,
            graph_model,
            query_timeout,
            commit_counter,
            gc_interval,
            tx_start_node_count: 0,
            tx_start_edge_count: 0,
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            current_graph: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
        }
    }

    /// Returns the graph model this session operates on.
    #[must_use]
    pub fn graph_model(&self) -> GraphModel {
        self.graph_model
    }

    // === Session State Management ===

    /// Sets the current graph for this session (USE GRAPH).
    pub fn use_graph(&self, name: &str) {
        *self.current_graph.lock() = Some(name.to_string());
    }

    /// Returns the current graph name, if any.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_graph.lock().clone()
    }

    /// Sets the session time zone.
    pub fn set_time_zone(&self, tz: &str) {
        *self.time_zone.lock() = Some(tz.to_string());
    }

    /// Returns the session time zone, if set.
    #[must_use]
    pub fn time_zone(&self) -> Option<String> {
        self.time_zone.lock().clone()
    }

    /// Sets a session parameter.
    pub fn set_parameter(&self, key: &str, value: grafeo_common::types::Value) {
        self.session_params.lock().insert(key.to_string(), value);
    }

    /// Gets a session parameter by cloning it.
    #[must_use]
    pub fn get_parameter(&self, key: &str) -> Option<grafeo_common::types::Value> {
        self.session_params.lock().get(key).cloned()
    }

    /// Resets all session state to defaults.
    pub fn reset_session(&self) {
        *self.current_graph.lock() = None;
        *self.time_zone.lock() = None;
        self.session_params.lock().clear();
        *self.viewing_epoch_override.lock() = None;
    }

    // --- Time-travel API ---

    /// Sets a viewing epoch override for time-travel queries.
    ///
    /// While set, all queries on this session see the database as it existed
    /// at the given epoch. Use [`clear_viewing_epoch`](Self::clear_viewing_epoch)
    /// to return to normal behavior.
    pub fn set_viewing_epoch(&self, epoch: EpochId) {
        *self.viewing_epoch_override.lock() = Some(epoch);
    }

    /// Clears the viewing epoch override, returning to normal behavior.
    pub fn clear_viewing_epoch(&self) {
        *self.viewing_epoch_override.lock() = None;
    }

    /// Returns the current viewing epoch override, if any.
    #[must_use]
    pub fn viewing_epoch(&self) -> Option<EpochId> {
        *self.viewing_epoch_override.lock()
    }

    /// Returns all versions of a node with their creation/deletion epochs.
    ///
    /// Properties and labels reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.store.get_node_history(id)
    }

    /// Returns all versions of an edge with their creation/deletion epochs.
    ///
    /// Properties reflect the current state (not versioned per-epoch).
    #[must_use]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.store.get_edge_history(id)
    }

    /// Checks that the session's graph model supports LPG operations.
    fn require_lpg(&self, language: &str) -> Result<()> {
        if self.graph_model == GraphModel::Rdf {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "This is an RDF database. {language} queries require an LPG database."
            )));
        }
        Ok(())
    }

    /// Executes a session or transaction command, returning an empty result.
    #[cfg(feature = "gql")]
    fn execute_session_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SessionCommand,
    ) -> Result<QueryResult> {
        use grafeo_adapters::query::gql::ast::SessionCommand;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        match cmd {
            SessionCommand::CreateGraph {
                name,
                if_not_exists,
            } => {
                let created = self.store.create_graph(&name);
                if !created && !if_not_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' already exists"),
                    )));
                }
                Ok(QueryResult::empty())
            }
            SessionCommand::DropGraph { name, if_exists } => {
                let dropped = self.store.drop_graph(&name);
                if !dropped && !if_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                Ok(QueryResult::empty())
            }
            SessionCommand::UseGraph(name) => {
                // Verify graph exists (default graph is always valid)
                if !name.eq_ignore_ascii_case("default") && self.store.graph(&name).is_none() {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                self.use_graph(&name);
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetGraph(name) => {
                self.use_graph(&name);
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetTimeZone(tz) => {
                self.set_time_zone(&tz);
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetParameter(key, expr) => {
                if key.eq_ignore_ascii_case("viewing_epoch") {
                    match Self::eval_integer_literal(&expr) {
                        Some(n) if n >= 0 => {
                            self.set_viewing_epoch(EpochId::new(n as u64));
                            Ok(QueryResult::status(format!("Set viewing_epoch to {n}")))
                        }
                        _ => Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "viewing_epoch must be a non-negative integer literal",
                        ))),
                    }
                } else {
                    // For now, store parameter name with Null value.
                    // Full expression evaluation would require building and executing a plan.
                    self.set_parameter(&key, Value::Null);
                    Ok(QueryResult::empty())
                }
            }
            SessionCommand::SessionReset => {
                self.reset_session();
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionClose => {
                self.reset_session();
                Ok(QueryResult::empty())
            }
            SessionCommand::StartTransaction => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Use session.begin_tx() for transaction management",
            ))),
            SessionCommand::Commit => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Use session.commit() for transaction management",
            ))),
            SessionCommand::Rollback => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Use session.rollback() for transaction management",
            ))),
        }
    }

    /// Logs a WAL record for a schema change (no-op if WAL is not enabled).
    #[allow(unused_variables)]
    fn log_schema_wal(&self, record: &grafeo_adapters::storage::wal::WalRecord) {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal
            && let Err(e) = wal.log(record)
        {
            tracing::warn!("Failed to log schema change to WAL: {}", e);
        }
    }

    /// Executes a schema DDL command, returning a status result.
    #[cfg(feature = "gql")]
    fn execute_schema_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> Result<QueryResult> {
        use crate::catalog::{
            EdgeTypeDefinition, NodeTypeDefinition, PropertyDataType, TypedProperty,
        };
        use grafeo_adapters::query::gql::ast::SchemaStatement;
        use grafeo_adapters::storage::wal::WalRecord;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        match cmd {
            SchemaStatement::CreateNodeType(stmt) => {
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = NodeTypeDefinition {
                    name: stmt.name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: None,
                        })
                        .collect(),
                    constraints: Vec::new(),
                };
                let result = if stmt.or_replace {
                    let _ = self.catalog.drop_node_type(&stmt.name);
                    self.catalog.register_node_type(def)
                } else {
                    self.catalog.register_node_type(def)
                };
                match result {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::CreateNodeType {
                            name: stmt.name.clone(),
                            properties: props_for_wal,
                            constraints: Vec::new(),
                        });
                        Ok(QueryResult::status(format!(
                            "Created node type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateEdgeType(stmt) => {
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = EdgeTypeDefinition {
                    name: stmt.name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: None,
                        })
                        .collect(),
                    constraints: Vec::new(),
                };
                let result = if stmt.or_replace {
                    let _ = self.catalog.drop_edge_type_def(&stmt.name);
                    self.catalog.register_edge_type_def(def)
                } else {
                    self.catalog.register_edge_type_def(def)
                };
                match result {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::CreateEdgeType {
                            name: stmt.name.clone(),
                            properties: props_for_wal,
                            constraints: Vec::new(),
                        });
                        Ok(QueryResult::status(format!(
                            "Created edge type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateVectorIndex(stmt) => {
                Self::create_vector_index_on_store(
                    &self.store,
                    &stmt.node_label,
                    &stmt.property,
                    stmt.dimensions,
                    stmt.metric.as_deref(),
                )?;
                self.log_schema_wal(&WalRecord::CreateIndex {
                    name: stmt.name.clone(),
                    label: stmt.node_label.clone(),
                    property: stmt.property.clone(),
                    index_type: "vector".to_string(),
                });
                Ok(QueryResult::status(format!(
                    "Created vector index '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::DropNodeType { name, if_exists } => {
                match self.catalog.drop_node_type(&name) {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::DropNodeType { name: name.clone() });
                        Ok(QueryResult::status(format!("Dropped node type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropEdgeType { name, if_exists } => {
                match self.catalog.drop_edge_type_def(&name) {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::DropEdgeType { name: name.clone() });
                        Ok(QueryResult::status(format!("Dropped edge type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateIndex(stmt) => {
                use grafeo_adapters::query::gql::ast::IndexKind;
                let index_type_str = match stmt.index_kind {
                    IndexKind::Property => "property",
                    IndexKind::BTree => "btree",
                    IndexKind::Text => "text",
                    IndexKind::Vector => "vector",
                };
                match stmt.index_kind {
                    IndexKind::Property | IndexKind::BTree => {
                        for prop in &stmt.properties {
                            self.store.create_property_index(prop);
                        }
                    }
                    IndexKind::Text => {
                        for prop in &stmt.properties {
                            Self::create_text_index_on_store(&self.store, &stmt.label, prop)?;
                        }
                    }
                    IndexKind::Vector => {
                        for prop in &stmt.properties {
                            Self::create_vector_index_on_store(
                                &self.store,
                                &stmt.label,
                                prop,
                                stmt.options.dimensions,
                                stmt.options.metric.as_deref(),
                            )?;
                        }
                    }
                }
                for prop in &stmt.properties {
                    self.log_schema_wal(&WalRecord::CreateIndex {
                        name: stmt.name.clone(),
                        label: stmt.label.clone(),
                        property: prop.clone(),
                        index_type: index_type_str.to_string(),
                    });
                }
                Ok(QueryResult::status(format!(
                    "Created {} index '{}'",
                    index_type_str, stmt.name
                )))
            }
            SchemaStatement::DropIndex { name, if_exists } => {
                // Try to drop property index by name
                let dropped = self.store.drop_property_index(&name);
                if dropped || if_exists {
                    if dropped {
                        self.log_schema_wal(&WalRecord::DropIndex { name: name.clone() });
                    }
                    Ok(QueryResult::status(if dropped {
                        format!("Dropped index '{name}'")
                    } else {
                        "No change".to_string()
                    }))
                } else {
                    Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Index '{name}' does not exist"),
                    )))
                }
            }
            SchemaStatement::CreateConstraint(stmt) => {
                use grafeo_adapters::query::gql::ast::ConstraintKind;
                let kind_str = match stmt.constraint_kind {
                    ConstraintKind::Unique => "unique",
                    ConstraintKind::NodeKey => "node_key",
                    ConstraintKind::NotNull => "not_null",
                    ConstraintKind::Exists => "exists",
                };
                let constraint_name = stmt
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{}_{kind_str}", stmt.label));
                self.log_schema_wal(&WalRecord::CreateConstraint {
                    name: constraint_name.clone(),
                    label: stmt.label.clone(),
                    properties: stmt.properties.clone(),
                    kind: kind_str.to_string(),
                });
                Ok(QueryResult::status(format!(
                    "Created {kind_str} constraint '{constraint_name}'"
                )))
            }
            SchemaStatement::DropConstraint { name, if_exists } => {
                let _ = if_exists;
                self.log_schema_wal(&WalRecord::DropConstraint { name: name.clone() });
                Ok(QueryResult::status(format!("Dropped constraint '{name}'")))
            }
            SchemaStatement::CreateGraphType(stmt) => {
                use crate::catalog::GraphTypeDefinition;
                let def = GraphTypeDefinition {
                    name: stmt.name.clone(),
                    allowed_node_types: stmt.node_types.clone(),
                    allowed_edge_types: stmt.edge_types.clone(),
                    open: stmt.open,
                };
                let result = if stmt.or_replace {
                    // Drop existing first, ignore error if not found
                    let _ = self.catalog.drop_graph_type(&stmt.name);
                    self.catalog.register_graph_type(def)
                } else {
                    self.catalog.register_graph_type(def)
                };
                match result {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::CreateGraphType {
                            name: stmt.name.clone(),
                            node_types: stmt.node_types,
                            edge_types: stmt.edge_types,
                            open: stmt.open,
                        });
                        Ok(QueryResult::status(format!(
                            "Created graph type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropGraphType { name, if_exists } => {
                match self.catalog.drop_graph_type(&name) {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::DropGraphType { name: name.clone() });
                        Ok(QueryResult::status(format!("Dropped graph type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateSchema {
                name,
                if_not_exists,
            } => match self.catalog.register_schema_namespace(name.clone()) {
                Ok(()) => {
                    self.log_schema_wal(&WalRecord::CreateSchema { name: name.clone() });
                    Ok(QueryResult::status(format!("Created schema '{name}'")))
                }
                Err(e) if if_not_exists => {
                    let _ = e;
                    Ok(QueryResult::status("No change"))
                }
                Err(e) => Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    e.to_string(),
                ))),
            },
            SchemaStatement::DropSchema { name, if_exists } => {
                match self.catalog.drop_schema_namespace(&name) {
                    Ok(()) => {
                        self.log_schema_wal(&WalRecord::DropSchema { name: name.clone() });
                        Ok(QueryResult::status(format!("Dropped schema '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
        }
    }

    /// Creates a vector index on the store by scanning existing nodes.
    #[cfg(all(feature = "gql", feature = "vector-index"))]
    fn create_vector_index_on_store(
        store: &LpgStore,
        label: &str,
        property: &str,
        dimensions: Option<usize>,
        metric: Option<&str>,
    ) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_common::utils::error::Error;
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex};

        let metric = match metric {
            Some(m) => DistanceMetric::from_str(m).ok_or_else(|| {
                Error::Internal(format!(
                    "Unknown distance metric '{m}'. Use: cosine, euclidean, dot_product, manhattan"
                ))
            })?,
            None => DistanceMetric::Cosine,
        };

        let prop_key = PropertyKey::new(property);
        let mut found_dims: Option<usize> = dimensions;
        let mut vectors: Vec<(grafeo_common::types::NodeId, Vec<f32>)> = Vec::new();

        for node in store.nodes_with_label(label) {
            if let Some(Value::Vector(v)) = node.properties.get(&prop_key) {
                if let Some(expected) = found_dims {
                    if v.len() != expected {
                        return Err(Error::Internal(format!(
                            "Vector dimension mismatch: expected {expected}, found {} on node {}",
                            v.len(),
                            node.id.0
                        )));
                    }
                } else {
                    found_dims = Some(v.len());
                }
                vectors.push((node.id, v.to_vec()));
            }
        }

        let Some(dims) = found_dims else {
            return Err(Error::Internal(format!(
                "No vector properties found on :{label}({property}) and no dimensions specified"
            )));
        };

        let config = HnswConfig::new(dims, metric);
        let index = HnswIndex::with_capacity(config, vectors.len());
        let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(store, property);
        for (node_id, vec) in &vectors {
            index.insert(*node_id, vec, &accessor);
        }

        store.add_vector_index(label, property, Arc::new(index));
        Ok(())
    }

    /// Stub for when vector-index feature is not enabled.
    #[cfg(all(feature = "gql", not(feature = "vector-index")))]
    fn create_vector_index_on_store(
        _store: &LpgStore,
        _label: &str,
        _property: &str,
        _dimensions: Option<usize>,
        _metric: Option<&str>,
    ) -> Result<()> {
        Err(grafeo_common::utils::error::Error::Internal(
            "Vector index support requires the 'vector-index' feature".to_string(),
        ))
    }

    /// Creates a text index on the store by scanning existing nodes.
    #[cfg(all(feature = "gql", feature = "text-index"))]
    fn create_text_index_on_store(store: &LpgStore, label: &str, property: &str) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};

        let mut index = InvertedIndex::new(BM25Config::default());
        let prop_key = PropertyKey::new(property);

        let nodes = store.nodes_by_label(label);
        for node_id in nodes {
            if let Some(Value::String(text)) = store.get_node_property(node_id, &prop_key) {
                index.insert(node_id, text.as_str());
            }
        }

        store.add_text_index(label, property, Arc::new(parking_lot::RwLock::new(index)));
        Ok(())
    }

    /// Stub for when text-index feature is not enabled.
    #[cfg(all(feature = "gql", not(feature = "text-index")))]
    fn create_text_index_on_store(_store: &LpgStore, _label: &str, _property: &str) -> Result<()> {
        Err(grafeo_common::utils::error::Error::Internal(
            "Text index support requires the 'text-index' feature".to_string(),
        ))
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create a node
    /// session.execute("INSERT (:Person {name: 'Alice', age: 30})")?;
    ///
    /// // Query nodes
    /// let result = session.execute("MATCH (n:Person) RETURN n.name, n.age")?;
    /// for row in &result.rows {
    ///     println!("{:?}", row);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gql")]
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.require_lpg("GQL")?;

        use crate::query::{
            Executor, binder::Binder, cache::CacheKey, gql_translator, optimizer::Optimizer,
            processor::QueryLanguage,
        };

        let start_time = std::time::Instant::now();

        // Parse and translate, checking for session/schema commands first
        let translation = gql_translator::translate_full(query)?;
        let logical_plan = match translation {
            gql_translator::GqlTranslationResult::SessionCommand(cmd) => {
                return self.execute_session_command(cmd);
            }
            gql_translator::GqlTranslationResult::SchemaCommand(cmd) => {
                return self.execute_schema_command(cmd);
            }
            gql_translator::GqlTranslationResult::Plan(plan) => plan,
        };

        // Create cache key for this query
        let cache_key = CacheKey::new(query, QueryLanguage::Gql);

        // Try to get cached optimized plan, or use the plan we just translated
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let optimizer = Optimizer::from_graph_store(&*self.graph_store);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan for future use
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        // (Physical planning cannot be cached as it depends on transaction state)
        let planner = self.create_planner(viewing_epoch, tx_id);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let mut result = executor.execute(physical_plan.operator.as_mut())?;

        // Add execution metrics
        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let rows_scanned = result.rows.len() as u64;
        result.execution_time_ms = Some(elapsed_ms);
        result.rows_scanned = Some(rows_scanned);

        Ok(result)
    }

    /// Executes a GQL query with visibility at the specified epoch.
    ///
    /// This enables time-travel queries: the query sees the database
    /// as it existed at the given epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch(&self, query: &str, epoch: EpochId) -> Result<QueryResult> {
        let previous = self.viewing_epoch_override.lock().replace(epoch);
        let result = self.execute(query);
        *self.viewing_epoch_override.lock() = previous;
        result
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gql")]
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        self.require_lpg("GQL")?;

        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor = QueryProcessor::for_graph_store_with_tx(
            Arc::clone(&self.graph_store),
            Arc::clone(&self.tx_manager),
        );

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::Gql, Some(&params))
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute_with_params(
        &self,
        _query: &str,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        Err(grafeo_common::utils::error::Error::Internal(
            "No query language enabled".to_string(),
        ))
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute(&self, _query: &str) -> Result<QueryResult> {
        Err(grafeo_common::utils::error::Error::Internal(
            "No query language enabled".to_string(),
        ))
    }

    /// Executes a Cypher query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, binder::Binder, cache::CacheKey, cypher_translator, optimizer::Optimizer,
            processor::QueryLanguage,
        };

        // Create cache key for this query
        let cache_key = CacheKey::new(query, QueryLanguage::Cypher);

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Parse and translate the query to a logical plan
            let logical_plan = cypher_translator::translate(query)?;

            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let optimizer = Optimizer::from_graph_store(&*self.graph_store);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = self.create_planner(viewing_epoch, tx_id);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut())?;
        Ok(result)
    }

    /// Executes a Gremlin query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["Person"]);
    ///
    /// // Query using Gremlin
    /// let result = session.execute_gremlin("g.V().hasLabel('Person')")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{Executor, binder::Binder, gremlin_translator, optimizer::Optimizer};

        // Parse and translate the query to a logical plan
        let logical_plan = gremlin_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::from_graph_store(&*self.graph_store);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = self.create_planner(viewing_epoch, tx_id);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut())?;
        Ok(result)
    }

    /// Executes a Gremlin query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor = QueryProcessor::for_graph_store_with_tx(
            Arc::clone(&self.graph_store),
            Arc::clone(&self.tx_manager),
        );

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::Gremlin, Some(&params))
    }

    /// Executes a GraphQL query against the LPG store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["User"]);
    ///
    /// // Query using GraphQL
    /// let result = session.execute_graphql("query { user { id name } }")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{Executor, binder::Binder, graphql_translator, optimizer::Optimizer};

        // Parse and translate the query to a logical plan
        let logical_plan = graphql_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::from_graph_store(&*self.graph_store);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = self.create_planner(viewing_epoch, tx_id);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut())?;
        Ok(result)
    }

    /// Executes a GraphQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor = QueryProcessor::for_graph_store_with_tx(
            Arc::clone(&self.graph_store),
            Arc::clone(&self.tx_manager),
        );

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::GraphQL, Some(&params))
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// let result = session.execute_sql(
    ///     "SELECT * FROM GRAPH_TABLE (
    ///         MATCH (n:Person)
    ///         COLUMNS (n.name AS name)
    ///     )"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, binder::Binder, cache::CacheKey, optimizer::Optimizer, plan::LogicalOperator,
            processor::QueryLanguage, sql_pgq_translator,
        };

        // Parse and translate (always needed to check for DDL)
        let logical_plan = sql_pgq_translator::translate(query)?;

        // Handle DDL statements directly (they don't go through the query pipeline)
        if let LogicalOperator::CreatePropertyGraph(ref cpg) = logical_plan.root {
            return Ok(QueryResult {
                columns: vec!["status".into()],
                column_types: vec![grafeo_common::types::LogicalType::String],
                rows: vec![vec![Value::from(format!(
                    "Property graph '{}' created ({} node tables, {} edge tables)",
                    cpg.name,
                    cpg.node_tables.len(),
                    cpg.edge_tables.len()
                ))]],
                execution_time_ms: None,
                rows_scanned: None,
                status_message: None,
            });
        }

        // Create cache key for query plans
        let cache_key = CacheKey::new(query, QueryLanguage::SqlPgq);

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let optimizer = Optimizer::from_graph_store(&*self.graph_store);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = self.create_planner(viewing_epoch, tx_id);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut())?;
        Ok(result)
    }

    /// Executes a SQL/PGQ query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor = QueryProcessor::for_graph_store_with_tx(
            Arc::clone(&self.graph_store),
            Arc::clone(&self.tx_manager),
        );

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::SqlPgq, Some(&params))
    }

    /// Executes a SPARQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(all(feature = "sparql", feature = "rdf"))]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner_rdf::RdfPlanner, sparql_translator,
        };

        // Parse and translate the SPARQL query to a logical plan
        let logical_plan = sparql_translator::translate(query)?;

        // Optimize the plan
        let optimizer = Optimizer::from_graph_store(&*self.graph_store);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan using RDF planner
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store)).with_tx_id(self.current_tx);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a SPARQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(all(feature = "sparql", feature = "rdf"))]
    pub fn execute_sparql_with_params(
        &self,
        query: &str,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        // TODO: Implement parameter substitution for SPARQL
        // For now, just execute the query without parameters
        self.execute_sparql(query)
    }

    /// Executes a query in the specified language by name.
    ///
    /// Supported language names: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
    /// `"sparql"`, `"sql"`. Each requires the corresponding feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the language is unknown/disabled or the query fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        match language {
            "gql" => {
                if let Some(p) = params {
                    self.execute_with_params(query, p)
                } else {
                    self.execute(query)
                }
            }
            #[cfg(feature = "cypher")]
            "cypher" => {
                if let Some(p) = params {
                    use crate::query::processor::{QueryLanguage, QueryProcessor};
                    let processor = QueryProcessor::for_graph_store_with_tx(
                        Arc::clone(&self.graph_store),
                        Arc::clone(&self.tx_manager),
                    );
                    let (viewing_epoch, tx_id) = self.get_transaction_context();
                    let processor = if let Some(tx_id) = tx_id {
                        processor.with_tx_context(viewing_epoch, tx_id)
                    } else {
                        processor
                    };
                    processor.process(query, QueryLanguage::Cypher, Some(&p))
                } else {
                    self.execute_cypher(query)
                }
            }
            #[cfg(feature = "gremlin")]
            "gremlin" => {
                if let Some(p) = params {
                    self.execute_gremlin_with_params(query, p)
                } else {
                    self.execute_gremlin(query)
                }
            }
            #[cfg(feature = "graphql")]
            "graphql" => {
                if let Some(p) = params {
                    self.execute_graphql_with_params(query, p)
                } else {
                    self.execute_graphql(query)
                }
            }
            #[cfg(feature = "sql-pgq")]
            "sql" | "sql-pgq" => {
                if let Some(p) = params {
                    self.execute_sql_with_params(query, p)
                } else {
                    self.execute_sql(query)
                }
            }
            #[cfg(all(feature = "sparql", feature = "rdf"))]
            "sparql" => {
                if let Some(p) = params {
                    self.execute_sparql_with_params(query, p)
                } else {
                    self.execute_sparql(query)
                }
            }
            other => Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    format!("Unknown query language: '{other}'"),
                ),
            )),
        }
    }

    /// Begins a new transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.execute("INSERT (:Person {name: 'Bob'})")?;
    /// session.commit()?; // Both inserts committed atomically
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin_tx(&mut self) -> Result<()> {
        if self.current_tx.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "Transaction already active".to_string(),
                ),
            ));
        }

        self.tx_start_node_count = self.store.node_count();
        self.tx_start_edge_count = self.store.edge_count();
        let tx_id = self.tx_manager.begin();
        self.current_tx = Some(tx_id);
        Ok(())
    }

    /// Begins a transaction with a specific isolation level.
    ///
    /// See [`begin_tx`](Self::begin_tx) for the default (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    pub fn begin_tx_with_isolation(
        &mut self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        if self.current_tx.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "Transaction already active".to_string(),
                ),
            ));
        }

        self.tx_start_node_count = self.store.node_count();
        self.tx_start_edge_count = self.store.edge_count();
        let tx_id = self.tx_manager.begin_with_isolation(isolation_level);
        self.current_tx = Some(tx_id);
        Ok(())
    }

    /// Commits the current transaction.
    ///
    /// Makes all changes since [`begin_tx`](Self::begin_tx) permanent.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    pub fn commit(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Commit RDF store pending operations
        #[cfg(feature = "rdf")]
        self.rdf_store.commit_tx(tx_id);

        self.tx_manager.commit(tx_id)?;

        // Sync the LpgStore epoch with the TxManager so that
        // convenience lookups (edge_type, get_edge, get_node) that use
        // store.current_epoch() can see versions created at the latest epoch.
        self.store.sync_epoch(self.tx_manager.current_epoch());

        // Auto-GC: periodically prune old MVCC versions
        if self.gc_interval > 0 {
            let count = self.commit_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(self.gc_interval) {
                let min_epoch = self.tx_manager.min_active_epoch();
                self.store.gc_versions(min_epoch);
                self.tx_manager.gc();
            }
        }

        Ok(())
    }

    /// Aborts the current transaction.
    ///
    /// Discards all changes since [`begin_tx`](Self::begin_tx).
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.rollback()?; // Insert is discarded
    /// # Ok(())
    /// # }
    /// ```
    pub fn rollback(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Discard uncommitted versions in the LPG store
        self.store.discard_uncommitted_versions(tx_id);

        // Discard pending operations in the RDF store
        #[cfg(feature = "rdf")]
        self.rdf_store.rollback_tx(tx_id);

        // Mark transaction as aborted in the manager
        self.tx_manager.abort(tx_id)
    }

    /// Returns whether a transaction is active.
    #[must_use]
    pub fn in_transaction(&self) -> bool {
        self.current_tx.is_some()
    }

    /// Returns the current transaction ID, if any.
    #[must_use]
    pub(crate) fn current_tx_id(&self) -> Option<TxId> {
        self.current_tx
    }

    /// Returns a reference to the transaction manager.
    #[must_use]
    pub(crate) fn tx_manager(&self) -> &TransactionManager {
        &self.tx_manager
    }

    /// Returns the store's current node count and the count at transaction start.
    #[must_use]
    pub(crate) fn node_count_delta(&self) -> (usize, usize) {
        (self.tx_start_node_count, self.store.node_count())
    }

    /// Returns the store's current edge count and the count at transaction start.
    #[must_use]
    pub(crate) fn edge_count_delta(&self) -> (usize, usize) {
        (self.tx_start_edge_count, self.store.edge_count())
    }

    /// Prepares the current transaction for a two-phase commit.
    ///
    /// Returns a [`PreparedCommit`](crate::transaction::PreparedCommit) that
    /// lets you inspect pending changes and attach metadata before finalizing.
    /// The mutable borrow prevents concurrent operations while the commit is
    /// pending.
    ///
    /// If the `PreparedCommit` is dropped without calling `commit()` or
    /// `abort()`, the transaction is automatically rolled back.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    ///
    /// let mut prepared = session.prepare_commit()?;
    /// println!("Nodes written: {}", prepared.info().nodes_written);
    /// prepared.set_metadata("audit_user", "admin");
    /// prepared.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn prepare_commit(&mut self) -> Result<crate::transaction::PreparedCommit<'_>> {
        crate::transaction::PreparedCommit::new(self)
    }

    /// Sets auto-commit mode.
    pub fn set_auto_commit(&mut self, auto_commit: bool) {
        self.auto_commit = auto_commit;
    }

    /// Returns whether auto-commit is enabled.
    #[must_use]
    pub fn auto_commit(&self) -> bool {
        self.auto_commit
    }

    /// Computes the wall-clock deadline for query execution.
    #[must_use]
    fn query_deadline(&self) -> Option<Instant> {
        self.query_timeout.map(|d| Instant::now() + d)
    }

    /// Evaluates a simple integer literal from a session parameter expression.
    fn eval_integer_literal(expr: &grafeo_adapters::query::gql::ast::Expression) -> Option<i64> {
        use grafeo_adapters::query::gql::ast::{Expression, Literal};
        match expr {
            Expression::Literal(Literal::Integer(n)) => Some(*n),
            _ => None,
        }
    }

    /// Returns the current transaction context for MVCC visibility.
    ///
    /// Returns `(viewing_epoch, tx_id)` where:
    /// - `viewing_epoch` is the epoch at which to check version visibility
    /// - `tx_id` is the current transaction ID (if in a transaction)
    #[must_use]
    fn get_transaction_context(&self) -> (EpochId, Option<TxId>) {
        // Time-travel override takes precedence (read-only, no tx context)
        if let Some(epoch) = *self.viewing_epoch_override.lock() {
            return (epoch, None);
        }

        if let Some(tx_id) = self.current_tx {
            // In a transaction: use the transaction's start epoch
            let epoch = self
                .tx_manager
                .start_epoch(tx_id)
                .unwrap_or_else(|| self.tx_manager.current_epoch());
            (epoch, Some(tx_id))
        } else {
            // No transaction: use current epoch
            (self.tx_manager.current_epoch(), None)
        }
    }

    /// Creates a planner with transaction context and constraint validator.
    fn create_planner(&self, viewing_epoch: EpochId, tx_id: Option<TxId>) -> crate::query::Planner {
        use crate::query::Planner;

        let mut planner = Planner::with_context(
            Arc::clone(&self.graph_store),
            Arc::clone(&self.tx_manager),
            tx_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution);

        // Attach the constraint validator for schema enforcement
        let validator = CatalogConstraintValidator::new(Arc::clone(&self.catalog));
        planner = planner.with_validator(Arc::new(validator));

        planner
    }

    /// Creates a node directly (bypassing query execution).
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the node will be versioned with the transaction ID.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .create_node_versioned(labels, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Creates a node with properties.
    ///
    /// If a transaction is active, the node will be versioned with the transaction ID.
    pub fn create_node_with_props<'a>(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> NodeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store.create_node_with_props_versioned(
            labels,
            properties.into_iter().map(|(k, v)| (k, v)),
            epoch,
            tx_id.unwrap_or(TxId::SYSTEM),
        )
    }

    /// Creates an edge between two nodes.
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the edge will be versioned with the transaction ID.
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .create_edge_versioned(src, dst, edge_type, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    // =========================================================================
    // Direct Lookup APIs (bypass query planning for O(1) point reads)
    // =========================================================================

    /// Gets a node by ID directly, bypassing query planning.
    ///
    /// This is the fastest way to retrieve a single node when you know its ID.
    /// Skips parsing, binding, optimization, and physical planning entirely.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention (uses DashMap internally)
    /// - ~20-30x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let node_id = session.create_node(&["Person"]);
    ///
    /// // Direct lookup - O(1), no query planning
    /// let node = session.get_node(node_id);
    /// assert!(node.is_some());
    /// ```
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .get_node_versioned(id, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Gets a single property from a node by ID, bypassing query planning.
    ///
    /// More efficient than `get_node()` when you only need one property,
    /// as it avoids loading the full node with all properties.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No query planning overhead
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let id = session.create_node_with_props(&["Person"], [("name", "Alice".into())]);
    ///
    /// // Direct property access - O(1)
    /// let name = session.get_node_property(id, "name");
    /// assert_eq!(name, Some(Value::String("Alice".into())));
    /// ```
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.get_node(id)
            .and_then(|node| node.get_property(key).cloned())
    }

    /// Gets an edge by ID directly, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .get_edge_versioned(id, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Gets outgoing neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all outgoing edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of outgoing edges
    /// - Uses adjacency index for direct access
    /// - ~10-20x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let alice = session.create_node(&["Person"]);
    /// let bob = session.create_node(&["Person"]);
    /// session.create_edge(alice, bob, "KNOWS");
    ///
    /// // Direct neighbor lookup - O(degree)
    /// let neighbors = session.get_neighbors_outgoing(alice);
    /// assert_eq!(neighbors.len(), 1);
    /// assert_eq!(neighbors[0].0, bob);
    /// ```
    #[must_use]
    pub fn get_neighbors_outgoing(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.store.edges_from(node, Direction::Outgoing).collect()
    }

    /// Gets incoming neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all incoming edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of incoming edges
    /// - Uses backward adjacency index for direct access
    #[must_use]
    pub fn get_neighbors_incoming(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.store.edges_from(node, Direction::Incoming).collect()
    }

    /// Gets outgoing neighbors filtered by edge type, bypassing query planning.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// # let session = db.session();
    /// # let alice = session.create_node(&["Person"]);
    /// let neighbors = session.get_neighbors_outgoing_by_type(alice, "KNOWS");
    /// ```
    #[must_use]
    pub fn get_neighbors_outgoing_by_type(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        self.store
            .edges_from(node, Direction::Outgoing)
            .filter(|(_, edge_id)| {
                self.get_edge(*edge_id)
                    .is_some_and(|e| e.edge_type.as_str() == edge_type)
            })
            .collect()
    }

    /// Checks if a node exists, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1)
    /// - Fastest existence check available
    #[must_use]
    pub fn node_exists(&self, id: NodeId) -> bool {
        self.get_node(id).is_some()
    }

    /// Checks if an edge exists, bypassing query planning.
    #[must_use]
    pub fn edge_exists(&self, id: EdgeId) -> bool {
        self.get_edge(id).is_some()
    }

    /// Gets the degree (number of edges) of a node.
    ///
    /// Returns (outgoing_degree, incoming_degree).
    #[must_use]
    pub fn get_degree(&self, node: NodeId) -> (usize, usize) {
        let out = self.store.out_degree(node);
        let in_degree = self.store.in_degree(node);
        (out, in_degree)
    }

    /// Batch lookup of multiple nodes by ID.
    ///
    /// More efficient than calling `get_node()` in a loop because it
    /// amortizes overhead.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n) where n is the number of IDs
    /// - Better cache utilization than individual lookups
    #[must_use]
    pub fn get_nodes_batch(&self, ids: &[NodeId]) -> Vec<Option<Node>> {
        let (epoch, tx_id) = self.get_transaction_context();
        let tx = tx_id.unwrap_or(TxId::SYSTEM);
        ids.iter()
            .map(|&id| self.store.get_node_versioned(id, epoch, tx))
            .collect()
    }

    // ── Change Data Capture ─────────────────────────────────────────────

    /// Returns the full change history for an entity (node or edge).
    #[cfg(feature = "cdc")]
    pub fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history(entity_id.into()))
    }

    /// Returns change events for an entity since the given epoch.
    #[cfg(feature = "cdc")]
    pub fn history_since(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        since_epoch: EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history_since(entity_id.into(), since_epoch))
    }

    /// Returns all change events across all entities in an epoch range.
    #[cfg(feature = "cdc")]
    pub fn changes_between(
        &self,
        start_epoch: EpochId,
        end_epoch: EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}

#[cfg(test)]
mod tests {
    use crate::database::GrafeoDB;

    #[test]
    fn test_session_create_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let id = session.create_node(&["Person"]);
        assert!(id.is_valid());
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_session_transaction() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        assert!(!session.in_transaction());

        session.begin_tx().unwrap();
        assert!(session.in_transaction());

        session.commit().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_transaction_context() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Without transaction - context should have current epoch and no tx_id
        let (_epoch1, tx_id1) = session.get_transaction_context();
        assert!(tx_id1.is_none());

        // Start a transaction
        session.begin_tx().unwrap();
        let (epoch2, tx_id2) = session.get_transaction_context();
        assert!(tx_id2.is_some());
        // Transaction should have a valid epoch
        let _ = epoch2; // Use the variable

        // Commit and verify
        session.commit().unwrap();
        let (epoch3, tx_id3) = session.get_transaction_context();
        assert!(tx_id3.is_none());
        // Epoch should have advanced after commit
        assert!(epoch3.as_u64() >= epoch2.as_u64());
    }

    #[test]
    fn test_session_rollback() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_tx().unwrap();
        session.rollback().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_rollback_discards_versions() {
        use grafeo_common::types::TxId;

        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction (at system level)
        let node_before = db.store().create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction
        let mut session = db.session();
        session.begin_tx().unwrap();
        let tx_id = session.current_tx.unwrap();

        // Create a node versioned with the transaction's ID
        let epoch = db.store().current_epoch();
        let node_in_tx = db.store().create_node_versioned(&["Person"], epoch, tx_id);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes at this point
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();
        assert!(!session.in_transaction());

        // The node created in the transaction should be discarded
        // Only the first node should remain visible
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard uncommitted node, but got {count_after}"
        );

        // The original node should still be accessible
        let current_epoch = db.store().current_epoch();
        assert!(
            db.store()
                .get_node_versioned(node_before, current_epoch, TxId::SYSTEM)
                .is_some(),
            "Original node should still exist"
        );

        // The node created in the transaction should not be accessible
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, current_epoch, TxId::SYSTEM)
                .is_none(),
            "Transaction node should be gone"
        );
    }

    #[test]
    fn test_session_create_node_in_transaction() {
        // Test that session.create_node() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        let node_before = db.create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node through the session
        let mut session = db.session();
        session.begin_tx().unwrap();

        // Create a node through session.create_node() - should be versioned with tx
        let node_in_tx = session.create_node(&["Person"]);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes at this point
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();

        // The node created via session.create_node() should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node(), but got {count_after}"
        );
    }

    #[test]
    fn test_session_create_node_with_props_in_transaction() {
        use grafeo_common::types::Value;

        // Test that session.create_node_with_props() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        db.create_node(&["Person"]);
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node with properties
        let mut session = db.session();
        session.begin_tx().unwrap();

        let node_in_tx =
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();

        // The node should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node_with_props()"
        );
    }

    #[cfg(feature = "gql")]
    mod gql_tests {
        use super::*;

        #[test]
        fn test_gql_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a GQL query
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_gql_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_gql_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid GQL syntax
            let result = session.execute("MATCH (n RETURN n");

            assert!(result.is_err());
        }

        #[test]
        fn test_gql_relationship_traversal() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alice -> Bob, Alice -> Charlie
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let charlie = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, charlie, "KNOWS");

            // Execute a path query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return 2 rows (Alice->Bob, Alice->Charlie)
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "a");
            assert_eq!(result.columns[1], "b");
        }

        #[test]
        fn test_gql_relationship_with_type_filter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alice -KNOWS-> Bob, Alice -WORKS_WITH-> Charlie
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let charlie = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, charlie, "WORKS_WITH");

            // Query only KNOWS relationships
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return only 1 row (Alice->Bob)
            assert_eq!(result.row_count(), 1);
        }

        #[test]
        fn test_gql_semantic_error_undefined_variable() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Reference undefined variable 'x' in RETURN
            let result = session.execute("MATCH (n:Person) RETURN x");

            // Should fail with semantic error
            assert!(result.is_err());
            let Err(err) = result else {
                panic!("Expected error")
            };
            assert!(
                err.to_string().contains("Undefined variable"),
                "Expected undefined variable error, got: {}",
                err
            );
        }

        #[test]
        fn test_gql_where_clause_property_filter() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with ages
            session.create_node_with_props(&["Person"], [("age", Value::Int64(25))]);
            session.create_node_with_props(&["Person"], [("age", Value::Int64(35))]);
            session.create_node_with_props(&["Person"], [("age", Value::Int64(45))]);

            // Query with WHERE clause: age > 30
            let result = session
                .execute("MATCH (n:Person) WHERE n.age > 30 RETURN n")
                .unwrap();

            // Should return 2 people (ages 35 and 45)
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_where_clause_equality() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Bob".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            // Query with WHERE clause: name = "Alice"
            let result = session
                .execute("MATCH (n:Person) WHERE n.name = \"Alice\" RETURN n")
                .unwrap();

            // Should return 2 people named Alice
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_return_property_access() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names and ages
            session.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String("Alice".into())),
                    ("age", Value::Int64(30)),
                ],
            );
            session.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String("Bob".into())),
                    ("age", Value::Int64(25)),
                ],
            );

            // Query returning properties
            let result = session
                .execute("MATCH (n:Person) RETURN n.name, n.age")
                .unwrap();

            // Should return 2 rows with name and age columns
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n.name");
            assert_eq!(result.columns[1], "n.age");

            // Check that we get actual values
            let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
            assert!(names.contains(&&Value::String("Alice".into())));
            assert!(names.contains(&&Value::String("Bob".into())));
        }

        #[test]
        fn test_gql_return_mixed_expressions() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a person
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            // Query returning both node and property
            let result = session
                .execute("MATCH (n:Person) RETURN n, n.name")
                .unwrap();

            assert_eq!(result.row_count(), 1);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n");
            assert_eq!(result.columns[1], "n.name");

            // Second column should be the name
            assert_eq!(result.rows[0][1], Value::String("Alice".into()));
        }
    }

    #[cfg(feature = "cypher")]
    mod cypher_tests {
        use super::*;

        #[test]
        fn test_cypher_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a Cypher query
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_cypher_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_cypher_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid Cypher syntax
            let result = session.execute_cypher("MATCH (n RETURN n");

            assert!(result.is_err());
        }
    }

    // ==================== Direct Lookup API Tests ====================

    mod direct_lookup_tests {
        use super::*;
        use grafeo_common::types::Value;

        #[test]
        fn test_get_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);
            let node = session.get_node(id);

            assert!(node.is_some());
            let node = node.unwrap();
            assert_eq!(node.id, id);
        }

        #[test]
        fn test_get_node_not_found() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Try to get a non-existent node
            let node = session.get_node(NodeId::new(9999));
            assert!(node.is_none());
        }

        #[test]
        fn test_get_node_property() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session
                .create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            let name = session.get_node_property(id, "name");
            assert_eq!(name, Some(Value::String("Alice".into())));

            // Non-existent property
            let missing = session.get_node_property(id, "missing");
            assert!(missing.is_none());
        }

        #[test]
        fn test_get_edge() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            let edge = session.get_edge(edge_id);
            assert!(edge.is_some());
            let edge = edge.unwrap();
            assert_eq!(edge.id, edge_id);
            assert_eq!(edge.src, alice);
            assert_eq!(edge.dst, bob);
        }

        #[test]
        fn test_get_edge_not_found() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let edge = session.get_edge(EdgeId::new(9999));
            assert!(edge.is_none());
        }

        #[test]
        fn test_get_neighbors_outgoing() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, carol, "KNOWS");

            let neighbors = session.get_neighbors_outgoing(alice);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&bob));
            assert!(neighbor_ids.contains(&carol));
        }

        #[test]
        fn test_get_neighbors_incoming() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            session.create_edge(bob, alice, "KNOWS");
            session.create_edge(carol, alice, "KNOWS");

            let neighbors = session.get_neighbors_incoming(alice);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&bob));
            assert!(neighbor_ids.contains(&carol));
        }

        #[test]
        fn test_get_neighbors_outgoing_by_type() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let company = session.create_node(&["Company"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, company, "WORKS_AT");

            let knows_neighbors = session.get_neighbors_outgoing_by_type(alice, "KNOWS");
            assert_eq!(knows_neighbors.len(), 1);
            assert_eq!(knows_neighbors[0].0, bob);

            let works_neighbors = session.get_neighbors_outgoing_by_type(alice, "WORKS_AT");
            assert_eq!(works_neighbors.len(), 1);
            assert_eq!(works_neighbors[0].0, company);

            // No edges of this type
            let no_neighbors = session.get_neighbors_outgoing_by_type(alice, "LIKES");
            assert!(no_neighbors.is_empty());
        }

        #[test]
        fn test_node_exists() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);

            assert!(session.node_exists(id));
            assert!(!session.node_exists(NodeId::new(9999)));
        }

        #[test]
        fn test_edge_exists() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            assert!(session.edge_exists(edge_id));
            assert!(!session.edge_exists(EdgeId::new(9999)));
        }

        #[test]
        fn test_get_degree() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            // Alice knows Bob and Carol (2 outgoing)
            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, carol, "KNOWS");
            // Bob knows Alice (1 incoming for Alice)
            session.create_edge(bob, alice, "KNOWS");

            let (out_degree, in_degree) = session.get_degree(alice);
            assert_eq!(out_degree, 2);
            assert_eq!(in_degree, 1);

            // Node with no edges
            let lonely = session.create_node(&["Person"]);
            let (out, in_deg) = session.get_degree(lonely);
            assert_eq!(out, 0);
            assert_eq!(in_deg, 0);
        }

        #[test]
        fn test_get_nodes_batch() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            let nodes = session.get_nodes_batch(&[alice, bob, carol]);
            assert_eq!(nodes.len(), 3);
            assert!(nodes[0].is_some());
            assert!(nodes[1].is_some());
            assert!(nodes[2].is_some());

            // With non-existent node
            use grafeo_common::types::NodeId;
            let nodes_with_missing = session.get_nodes_batch(&[alice, NodeId::new(9999), carol]);
            assert_eq!(nodes_with_missing.len(), 3);
            assert!(nodes_with_missing[0].is_some());
            assert!(nodes_with_missing[1].is_none()); // Missing node
            assert!(nodes_with_missing[2].is_some());
        }

        #[test]
        fn test_auto_commit_setting() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Default is auto-commit enabled
            assert!(session.auto_commit());

            session.set_auto_commit(false);
            assert!(!session.auto_commit());

            session.set_auto_commit(true);
            assert!(session.auto_commit());
        }

        #[test]
        fn test_transaction_double_begin_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            session.begin_tx().unwrap();
            let result = session.begin_tx();

            assert!(result.is_err());
            // Clean up
            session.rollback().unwrap();
        }

        #[test]
        fn test_commit_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.commit();
            assert!(result.is_err());
        }

        #[test]
        fn test_rollback_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.rollback();
            assert!(result.is_err());
        }

        #[test]
        fn test_create_edge_in_transaction() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Create nodes outside transaction
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);

            // Create edge in transaction
            session.begin_tx().unwrap();
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            // Edge should be visible in the transaction
            assert!(session.edge_exists(edge_id));

            // Commit
            session.commit().unwrap();

            // Edge should still be visible
            assert!(session.edge_exists(edge_id));
        }

        #[test]
        fn test_neighbors_empty_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let lonely = session.create_node(&["Person"]);

            assert!(session.get_neighbors_outgoing(lonely).is_empty());
            assert!(session.get_neighbors_incoming(lonely).is_empty());
            assert!(
                session
                    .get_neighbors_outgoing_by_type(lonely, "KNOWS")
                    .is_empty()
            );
        }
    }

    #[test]
    fn test_auto_gc_triggers_on_commit_interval() {
        use crate::config::Config;

        let config = Config::in_memory().with_gc_interval(2);
        let db = GrafeoDB::with_config(config).unwrap();
        let mut session = db.session();

        // First commit: counter = 1, no GC (not a multiple of 2)
        session.begin_tx().unwrap();
        session.create_node(&["A"]);
        session.commit().unwrap();

        // Second commit: counter = 2, GC should trigger (multiple of 2)
        session.begin_tx().unwrap();
        session.create_node(&["B"]);
        session.commit().unwrap();

        // Verify the database is still functional after GC
        assert_eq!(db.node_count(), 2);
    }

    #[test]
    fn test_query_timeout_config_propagates_to_session() {
        use crate::config::Config;
        use std::time::Duration;

        let config = Config::in_memory().with_query_timeout(Duration::from_secs(5));
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        // Verify the session has a query deadline (timeout was set)
        assert!(session.query_deadline().is_some());
    }

    #[test]
    fn test_no_query_timeout_returns_no_deadline() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Default config has no timeout
        assert!(session.query_deadline().is_none());
    }

    #[test]
    fn test_graph_model_accessor() {
        use crate::config::GraphModel;

        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        assert_eq!(session.graph_model(), GraphModel::Lpg);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_external_store_session() {
        use grafeo_core::graph::GraphStoreMut;
        use std::sync::Arc;

        let config = crate::config::Config::in_memory();
        let store = Arc::new(grafeo_core::graph::lpg::LpgStore::new()) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(store, config).unwrap();

        let session = db.session();

        // Create data through a query (goes through the external graph_store)
        session.execute("INSERT (:Test {name: 'hello'})").unwrap();

        // Verify we can query through it
        let result = session.execute("MATCH (n:Test) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    // ==================== Session Command Tests ====================

    #[cfg(feature = "gql")]
    mod session_command_tests {
        use super::*;

        #[test]
        fn test_use_graph_sets_current_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create the graph first, then USE it
            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("USE GRAPH mydb").unwrap();

            assert_eq!(session.current_graph(), Some("mydb".to_string()));
        }

        #[test]
        fn test_use_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("USE GRAPH doesnotexist");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_use_graph_default_always_valid() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // "default" is always valid, even without CREATE GRAPH
            session.execute("USE GRAPH default").unwrap();
            assert_eq!(session.current_graph(), Some("default".to_string()));
        }

        #[test]
        fn test_session_set_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // SESSION SET GRAPH does not verify existence
            session.execute("SESSION SET GRAPH analytics").unwrap();
            assert_eq!(session.current_graph(), Some("analytics".to_string()));
        }

        #[test]
        fn test_session_set_time_zone() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);

            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            assert_eq!(session.time_zone(), Some("UTC".to_string()));

            session
                .execute("SESSION SET TIME ZONE 'America/New_York'")
                .unwrap();
            assert_eq!(session.time_zone(), Some("America/New_York".to_string()));
        }

        #[test]
        fn test_session_set_parameter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("SESSION SET PARAMETER $timeout = 30")
                .unwrap();

            // Parameter is stored (value is Null for now, since expression
            // evaluation is not yet wired up)
            assert!(session.get_parameter("timeout").is_some());
        }

        #[test]
        fn test_session_reset_clears_all_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Set various session state
            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            session
                .execute("SESSION SET PARAMETER $limit = 100")
                .unwrap();

            // Verify state was set
            assert!(session.current_graph().is_some());
            assert!(session.time_zone().is_some());
            assert!(session.get_parameter("limit").is_some());

            // Reset everything
            session.execute("SESSION RESET").unwrap();

            assert_eq!(session.current_graph(), None);
            assert_eq!(session.time_zone(), None);
            assert!(session.get_parameter("limit").is_none());
        }

        #[test]
        fn test_session_close_clears_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();

            session.execute("SESSION CLOSE").unwrap();

            assert_eq!(session.current_graph(), None);
            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_create_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();

            // Should be able to USE it now
            session.execute("USE GRAPH mydb").unwrap();
            assert_eq!(session.current_graph(), Some("mydb".to_string()));
        }

        #[test]
        fn test_create_graph_duplicate_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            let result = session.execute("CREATE GRAPH mydb");

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("already exists"),
                "Expected 'already exists' error, got: {err}"
            );
        }

        #[test]
        fn test_create_graph_if_not_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            // Should succeed silently with IF NOT EXISTS
            session.execute("CREATE GRAPH IF NOT EXISTS mydb").unwrap();
        }

        #[test]
        fn test_drop_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("DROP GRAPH mydb").unwrap();

            // Should no longer be usable
            let result = session.execute("USE GRAPH mydb");
            assert!(result.is_err());
        }

        #[test]
        fn test_drop_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("DROP GRAPH nosuchgraph");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_drop_graph_if_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Should succeed silently with IF EXISTS
            session.execute("DROP GRAPH IF EXISTS nosuchgraph").unwrap();
        }

        #[test]
        fn test_start_transaction_returns_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("START TRANSACTION");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("begin_tx"),
                "Expected guidance to use begin_tx(), got: {err}"
            );
        }

        #[test]
        fn test_commit_via_gql_returns_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("COMMIT");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("commit()"),
                "Expected guidance to use commit(), got: {err}"
            );
        }

        #[test]
        fn test_rollback_via_gql_returns_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("ROLLBACK");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("rollback()"),
                "Expected guidance to use rollback(), got: {err}"
            );
        }

        #[test]
        fn test_session_commands_return_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SESSION SET GRAPH test").unwrap();
            assert_eq!(result.row_count(), 0);
            assert_eq!(result.column_count(), 0);
        }

        #[test]
        fn test_current_graph_default_is_none() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.current_graph(), None);
        }

        #[test]
        fn test_time_zone_default_is_none() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_session_state_independent_across_sessions() {
            let db = GrafeoDB::new_in_memory();
            let session1 = db.session();
            let session2 = db.session();

            session1.execute("SESSION SET GRAPH first").unwrap();
            session2.execute("SESSION SET GRAPH second").unwrap();

            assert_eq!(session1.current_graph(), Some("first".to_string()));
            assert_eq!(session2.current_graph(), Some("second".to_string()));
        }
    }
}
