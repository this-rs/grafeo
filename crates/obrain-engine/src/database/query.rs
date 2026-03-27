//! Query execution methods for GrafeoDB.

use grafeo_common::utils::error::Result;

use super::{FromValue, QueryResult};

impl super::GrafeoDB {
    /// Executes a closure with a one-shot session, syncing graph context back
    /// to the database afterward. This ensures `USE GRAPH`, `SESSION SET GRAPH`,
    /// and `SESSION RESET` persist across one-shot `execute()` calls.
    fn with_session<F>(&self, func: F) -> Result<QueryResult>
    where
        F: FnOnce(&crate::session::Session) -> Result<QueryResult>,
    {
        let session = self.session();
        let result = func(&session);
        // Sync graph state back, even on error (USE GRAPH may have succeeded
        // before a subsequent query failed in the same session).
        *self.current_graph.write() = session.current_graph();
        result
    }

    /// Runs a query directly on the database.
    ///
    /// A convenience method that creates a temporary session behind the
    /// scenes. If you're running multiple queries, grab a
    /// [`session()`](Self::session) instead to avoid the overhead.
    ///
    /// Graph context commands (`USE GRAPH`, `SESSION SET GRAPH`, `SESSION RESET`)
    /// persist across calls: running `execute("USE GRAPH analytics")` followed
    /// by `execute("MATCH (n) RETURN n")` routes the second query to the
    /// analytics graph.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute(query))
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
        self.with_session(|s| s.execute_at_epoch(query, epoch))
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
        self.with_session(|s| s.execute_with_params(query, params))
    }

    /// Executes a Cypher query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_cypher(query))
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
        self.with_session(|s| s.execute_language(query, "cypher", Some(params)))
    }

    /// Executes a Gremlin query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_gremlin(query))
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
        self.with_session(|s| s.execute_gremlin_with_params(query, params))
    }

    /// Executes a GraphQL query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_graphql(query))
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
        self.with_session(|s| s.execute_graphql_with_params(query, params))
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE) and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        self.with_session(|s| s.execute_sql(query))
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
        self.with_session(|s| s.execute_sql_with_params(query, params))
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
        self.with_session(|s| s.execute_language(query, language, params))
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
