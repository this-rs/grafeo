//! Query execution methods for GrafeoDB.

#[cfg(any(feature = "cypher", feature = "sql-pgq", feature = "rdf"))]
use std::sync::Arc;

use grafeo_common::utils::error::Result;
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use super::{FromValue, QueryResult};

impl super::GrafeoDB {
    /// Runs a query directly on the database.
    ///
    /// A convenience method that creates a temporary session behind the
    /// scenes. If you're running multiple queries, grab a
    /// [`session()`](Self::session) instead to avoid the overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute(query)
    }

    /// Executes a GQL query with visibility at the specified epoch.
    ///
    /// This enables time-travel queries: the query sees the database
    /// as it existed at the given epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    pub fn execute_at_epoch(
        &self,
        query: &str,
        epoch: grafeo_common::types::EpochId,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_at_epoch(query, epoch)
    }

    /// Executes a query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_with_params(query, params)
    }

    /// Executes a Cypher query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_cypher(query)
    }

    /// Executes a Cypher query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Create processor
        let processor = QueryProcessor::for_lpg(Arc::clone(&self.store));
        processor.process(query, QueryLanguage::Cypher, Some(&params))
    }

    /// Executes a Gremlin query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_gremlin(query)
    }

    /// Executes a Gremlin query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_gremlin_with_params(query, params)
    }

    /// Executes a GraphQL query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_graphql(query)
    }

    /// Executes a GraphQL query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_graphql_with_params(query, params)
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE) and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_sql(query)
    }

    /// Executes a SQL/PGQ query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Create processor
        let processor = QueryProcessor::for_lpg(Arc::clone(&self.store));
        processor.process(query, QueryLanguage::SqlPgq, Some(&params))
    }

    /// Executes a SPARQL query and returns the result.
    ///
    /// SPARQL queries operate on the RDF triple store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(feature = "sparql", feature = "rdf"))]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner_rdf::RdfPlanner, sparql_translator,
        };

        // Parse and translate the SPARQL query to a logical plan
        let logical_plan = sparql_translator::translate(query)?;

        // Optimize the plan
        let optimizer = Optimizer::from_store(&self.store);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan using RDF planner
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a query in the specified language by name.
    ///
    /// Supported language names: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
    /// `"sparql"`, `"sql"`. Each requires the corresponding feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the language is unknown/disabled, or if the query
    /// fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<std::collections::HashMap<String, grafeo_common::types::Value>>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_language(query, language, params)
    }

    /// Returns the RDF store.
    ///
    /// This provides direct access to the RDF store for triple operations.
    #[cfg(feature = "rdf")]
    #[must_use]
    pub fn rdf_store(&self) -> &Arc<RdfStore> {
        &self.rdf_store
    }

    /// Executes a query and returns a single scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or doesn't return exactly one row.
    pub fn query_scalar<T: FromValue>(&self, query: &str) -> Result<T> {
        let result = self.execute(query)?;
        result.scalar()
    }
}
