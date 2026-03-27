//! RDF-specific session methods.
//!
//! This module consolidates all RDF functionality from the session layer.
//! The entire module is gated behind `#[cfg(feature = "rdf")]` in the parent.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
#[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
use std::time::Instant;

use grafeo_common::types::{TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStoreMut;
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::rdf::RdfStore;

use crate::database::QueryResult;

use super::{Session, SessionConfig};

impl Session {
    /// Creates a new session with RDF store and adaptive configuration.
    pub(crate) fn with_rdf_store_and_adaptive(
        store: Arc<LpgStore>,
        rdf_store: Arc<RdfStore>,
        cfg: SessionConfig,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        Self {
            store,
            graph_store,
            catalog: cfg.catalog,
            rdf_store,
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            current_transaction: parking_lot::Mutex::new(None),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            transaction_start_node_count: AtomicUsize::new(0),
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "wal")]
            wal_graph_context: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            current_graph: parking_lot::Mutex::new(None),
            current_schema: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "reactive")]
            post_commit_hook: None,
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
        }
    }

    /// Executes a GraphQL query against the RDF store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner::rdf::RdfPlanner, translators::graphql_rdf,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let logical_plan = graphql_rdf::translate(query, "http://example.org/")?;
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut());

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a GraphQL query against the RDF store with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let has_mutations = Self::query_looks_like_mutation(query);
        let active = self.active_store();

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let processor = QueryProcessor::for_graph_store_with_transaction(
                Arc::clone(&active),
                Arc::clone(&self.transaction_manager),
            )?;
            let processor = if let Some(transaction_id) = transaction_id {
                processor.with_transaction_context(viewing_epoch, transaction_id)
            } else {
                processor
            };
            processor.process(query, QueryLanguage::GraphQLRdf, Some(&params))
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SPARQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner::rdf::RdfPlanner, translators::sparql,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let logical_plan = sparql::translate(query)?;
        let rdf_stats = self.rdf_store.collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics(rdf_stats);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(&optimized_plan));
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut());

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SPARQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner::rdf::RdfPlanner, processor::substitute_params,
            translators::sparql,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let mut logical_plan = sparql::translate(query)?;
        substitute_params(&mut logical_plan, &params)?;

        let rdf_stats = self.rdf_store.collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics(rdf_stats);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(&optimized_plan));
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_deadline(self.query_deadline());
        let result = executor.execute(physical_plan.operator.as_mut());

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        result
    }

    /// Commits RDF transaction state.
    ///
    /// Called from the main commit path to finalize RDF changes.
    pub(super) fn commit_rdf_transaction(&self, transaction_id: TransactionId) {
        self.rdf_store.commit_transaction(transaction_id);
    }

    /// Rolls back RDF transaction state.
    ///
    /// Called from the main commit-conflict and rollback paths to discard RDF changes.
    pub(super) fn rollback_rdf_transaction(&self, transaction_id: TransactionId) {
        self.rdf_store.rollback_transaction(transaction_id);
    }
}
