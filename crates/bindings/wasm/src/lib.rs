//! WebAssembly bindings for Grafeo graph database.
//!
//! Use Grafeo from JavaScript in the browser, Deno, or Cloudflare Workers.
//!
//! ```js
//! import init, { Database } from '@grafeo-db/wasm';
//!
//! await init();
//! const db = new Database();
//! db.execute("INSERT (:Person {name: 'Alice', age: 30})");
//! const result = db.execute("MATCH (p:Person) RETURN p.name, p.age");
//! console.log(result); // [{name: "Alice", age: 30}]
//! ```

mod types;
mod utils;

use std::collections::HashMap;

use js_sys::Array;
use wasm_bindgen::prelude::*;

use grafeo_bindings_common::json::json_params_to_map;
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A Grafeo graph database instance running in WebAssembly.
///
/// All data is held in memory within the WASM heap. For persistence,
/// use `exportSnapshot()` / `importSnapshot()` with IndexedDB or
/// the higher-level `@grafeo-db/web` package.
#[wasm_bindgen]
pub struct Database {
    inner: GrafeoDB,
}

#[wasm_bindgen]
impl Database {
    /// Creates a new in-memory database.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Database, JsError> {
        utils::set_panic_hook();
        Ok(Database {
            inner: GrafeoDB::new_in_memory(),
        })
    }

    /// Executes a GQL query and returns results as an array of objects.
    ///
    /// Each row becomes a JavaScript object with column names as keys.
    ///
    /// ```js
    /// const results = db.execute("MATCH (p:Person) RETURN p.name, p.age");
    /// // [{name: "Alice", age: 30}, {name: "Bob", age: 25}]
    /// ```
    pub fn execute(&self, query: &str) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute(query)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
    }

    /// Executes a GQL query and returns raw columns, rows, and metadata.
    ///
    /// Returns `{ columns: string[], rows: any[][], executionTimeMs?: number }`.
    #[wasm_bindgen(js_name = "executeRaw")]
    pub fn execute_raw(&self, query: &str) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute(query)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let obj = js_sys::Object::new();

        // columns: string[]
        let cols = Array::new_with_length(result.columns.len() as u32);
        for (i, col) in result.columns.iter().enumerate() {
            cols.set(i as u32, JsValue::from_str(col));
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("columns"), &cols);

        // rows: any[][]
        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            let js_row = Array::new_with_length(row.len() as u32);
            for (j, val) in row.iter().enumerate() {
                js_row.set(j as u32, types::value_to_js(val));
            }
            rows.set(i as u32, js_row.into());
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("rows"), &rows);

        // executionTimeMs?: number
        if let Some(ms) = result.execution_time_ms {
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("executionTimeMs"),
                &JsValue::from_f64(ms),
            );
        }

        Ok(obj.into())
    }

    /// Returns the number of nodes in the database.
    #[wasm_bindgen(js_name = "nodeCount")]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of edges in the database.
    #[wasm_bindgen(js_name = "edgeCount")]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Executes a query using a specific query language.
    ///
    /// Supported languages: `"gql"`, `"cypher"`, `"sparql"`, `"gremlin"`, `"graphql"`, `"sql"`.
    /// Languages require their corresponding feature flag to be enabled.
    ///
    /// ```js
    /// const results = db.executeWithLanguage(
    ///   "MATCH (p:Person) RETURN p.name",
    ///   "cypher"
    /// );
    /// ```
    #[wasm_bindgen(js_name = "executeWithLanguage")]
    pub fn execute_with_language(&self, query: &str, language: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, language, None)
    }

    /// Exports the database to a binary snapshot.
    ///
    /// Returns a `Uint8Array` that can be stored in IndexedDB, localStorage,
    /// or sent over the network. Restore with `Database.importSnapshot()`.
    ///
    /// ```js
    /// const bytes = db.exportSnapshot();
    /// // Store in IndexedDB, download as file, etc.
    /// ```
    #[wasm_bindgen(js_name = "exportSnapshot")]
    pub fn export_snapshot(&self) -> Result<Vec<u8>, JsError> {
        self.inner
            .export_snapshot()
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Creates a database from a binary snapshot.
    ///
    /// The `data` must have been produced by `exportSnapshot()`.
    ///
    /// ```js
    /// const db = Database.importSnapshot(bytes);
    /// ```
    #[wasm_bindgen(js_name = "importSnapshot")]
    pub fn import_snapshot(data: &[u8]) -> Result<Database, JsError> {
        utils::set_panic_hook();
        let inner = GrafeoDB::import_snapshot(data).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Database { inner })
    }

    /// Returns schema information about the database.
    ///
    /// Returns an object describing labels, edge types, and property keys.
    ///
    /// ```js
    /// const schema = db.schema();
    /// // { lpg: { labels: [...], edgeTypes: [...], propertyKeys: [...] } }
    /// ```
    pub fn schema(&self) -> Result<JsValue, JsError> {
        let info = self.inner.schema();
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Creates a text index on a label+property pair for full-text (BM25) search.
    ///
    /// Indexes all existing nodes with matching label and string property values.
    ///
    /// ```js
    /// db.createTextIndex("Article", "content");
    /// ```
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "createTextIndex")]
    pub fn create_text_index(&self, label: &str, property: &str) -> Result<(), JsError> {
        self.inner
            .create_text_index(label, property)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Drops a text index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "dropTextIndex")]
    pub fn drop_text_index(&self, label: &str, property: &str) -> bool {
        self.inner.drop_text_index(label, property)
    }

    /// Rebuilds a text index by re-scanning all matching nodes.
    ///
    /// Use after bulk imports to refresh the index.
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "rebuildTextIndex")]
    pub fn rebuild_text_index(&self, label: &str, property: &str) -> Result<(), JsError> {
        self.inner
            .rebuild_text_index(label, property)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Performs full-text search using BM25 ranking.
    ///
    /// Returns an array of `{id, score}` objects, ordered by relevance.
    ///
    /// ```js
    /// db.createTextIndex("Article", "content");
    /// const results = db.textSearch("Article", "content", "graph database", 10);
    /// // [{id: 42, score: 2.5}, {id: 17, score: 1.8}]
    /// ```
    #[cfg(feature = "text-index")]
    #[wasm_bindgen(js_name = "textSearch")]
    pub fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let results = self
            .inner
            .text_search(label, property, query, k)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let arr = Array::new_with_length(results.len() as u32);
        for (i, (id, score)) in results.iter().enumerate() {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("id"),
                &JsValue::from_f64(id.0 as f64),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("score"),
                &JsValue::from_f64(*score),
            );
            arr.set(i as u32, obj.into());
        }
        Ok(arr.into())
    }

    /// Performs hybrid search combining text (BM25) and vector similarity.
    ///
    /// Uses Reciprocal Rank Fusion to combine results from both indexes.
    /// Returns an array of `{id, score}` objects.
    ///
    /// ```js
    /// const results = db.hybridSearch("Article", "content", "embedding", "graph databases", 10);
    /// ```
    #[cfg(feature = "hybrid-search")]
    #[wasm_bindgen(js_name = "hybridSearch")]
    pub fn hybrid_search(
        &self,
        label: &str,
        text_property: &str,
        vector_property: &str,
        query_text: &str,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let results = self
            .inner
            .hybrid_search(
                label,
                text_property,
                vector_property,
                query_text,
                None,
                k,
                None,
            )
            .map_err(|e| JsError::new(&e.to_string()))?;

        let arr = Array::new_with_length(results.len() as u32);
        for (i, (id, score)) in results.iter().enumerate() {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("id"),
                &JsValue::from_f64(id.0 as f64),
            );
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("score"),
                &JsValue::from_f64(*score),
            );
            arr.set(i as u32, obj.into());
        }
        Ok(arr.into())
    }

    /// Executes a GQL query with parameters and returns results as an array of objects.
    ///
    /// Parameters are passed as a JavaScript object with string keys.
    /// Use `$name` syntax in the query to reference parameters.
    ///
    /// ```js
    /// const results = db.executeWithParams(
    ///   "MATCH (p:Person {name: $name}) RETURN p.name, p.age",
    ///   { name: "Alice" }
    /// );
    /// ```
    #[wasm_bindgen(js_name = "executeWithParams")]
    pub fn execute_with_params(&self, query: &str, params: JsValue) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "gql", Some(params))
    }

    /// Executes a query using a specific language with parameters.
    ///
    /// Combines language selection with parameterised queries.
    ///
    /// ```js
    /// const results = db.executeWithLanguageAndParams(
    ///   "MATCH (p:Person {name: $name}) RETURN p.name",
    ///   "cypher",
    ///   { name: "Alice" }
    /// );
    /// ```
    #[wasm_bindgen(js_name = "executeWithLanguageAndParams")]
    pub fn execute_with_language_and_params(
        &self,
        query: &str,
        language: &str,
        params: JsValue,
    ) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, language, Some(params))
    }

    /// Executes a Cypher query and returns results as an array of objects.
    ///
    /// Requires the `cypher` feature flag.
    ///
    /// ```js
    /// const results = db.executeCypher("MATCH (p:Person) RETURN p.name");
    /// ```
    #[cfg(feature = "cypher")]
    #[wasm_bindgen(js_name = "executeCypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "cypher", None)
    }

    /// Executes a Gremlin query and returns results as an array of objects.
    ///
    /// Requires the `gremlin` feature flag.
    ///
    /// ```js
    /// const results = db.executeGremlin("g.V().hasLabel('Person').values('name')");
    /// ```
    #[cfg(feature = "gremlin")]
    #[wasm_bindgen(js_name = "executeGremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "gremlin", None)
    }

    /// Executes a GraphQL query and returns results as an array of objects.
    ///
    /// Requires the `graphql` feature flag.
    ///
    /// ```js
    /// const results = db.executeGraphql("{ Person { name age } }");
    /// ```
    #[cfg(feature = "graphql")]
    #[wasm_bindgen(js_name = "executeGraphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "graphql", None)
    }

    /// Executes a SPARQL query and returns results as an array of objects.
    ///
    /// Requires the `sparql` feature flag.
    ///
    /// ```js
    /// const results = db.executeSparql("SELECT ?name WHERE { ?p a :Person ; :name ?name }");
    /// ```
    #[cfg(feature = "sparql")]
    #[wasm_bindgen(js_name = "executeSparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "sparql", None)
    }

    /// Executes a SQL/PGQ query and returns results as an array of objects.
    ///
    /// Requires the `sql-pgq` feature flag.
    ///
    /// ```js
    /// const results = db.executeSql("SELECT * FROM GRAPH_TABLE (...)");
    /// ```
    #[cfg(feature = "sql-pgq")]
    #[wasm_bindgen(js_name = "executeSql")]
    pub fn execute_sql(&self, query: &str) -> Result<JsValue, JsError> {
        self.execute_language_impl(query, "sql", None)
    }

    /// Executes a query in a specific language and returns raw columns, rows, and metadata.
    ///
    /// Returns `{ columns: string[], rows: any[][], executionTimeMs?: number }`.
    ///
    /// ```js
    /// const raw = db.executeRawWithLanguage("MATCH (p:Person) RETURN p.name", "cypher");
    /// // { columns: ["p.name"], rows: [["Alice"], ["Bob"]], executionTimeMs: 0.5 }
    /// ```
    #[wasm_bindgen(js_name = "executeRawWithLanguage")]
    pub fn execute_raw_with_language(
        &self,
        query: &str,
        language: &str,
    ) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .execute_language(query, language, None)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let obj = js_sys::Object::new();

        // columns: string[]
        let cols = Array::new_with_length(result.columns.len() as u32);
        for (i, col) in result.columns.iter().enumerate() {
            cols.set(i as u32, JsValue::from_str(col));
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("columns"), &cols);

        // rows: any[][]
        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            let js_row = Array::new_with_length(row.len() as u32);
            for (j, val) in row.iter().enumerate() {
                js_row.set(j as u32, types::value_to_js(val));
            }
            rows.set(i as u32, js_row.into());
        }
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("rows"), &rows);

        // executionTimeMs?: number
        if let Some(ms) = result.execution_time_ms {
            let _ = js_sys::Reflect::set(
                &obj,
                &JsValue::from_str("executionTimeMs"),
                &JsValue::from_f64(ms),
            );
        }

        Ok(obj.into())
    }

    /// Returns the Grafeo version.
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

// ---------------------------------------------------------------------------
// Private helpers (not exported to JS)
// ---------------------------------------------------------------------------

impl Database {
    /// Shared implementation for all language-specific execute methods.
    ///
    /// Converts an optional JS params object to the internal
    /// `HashMap<String, Value>` representation and delegates to
    /// `GrafeoDB::execute_language`.
    fn execute_language_impl(
        &self,
        query: &str,
        language: &str,
        params: Option<JsValue>,
    ) -> Result<JsValue, JsError> {
        let param_map = Self::convert_params(params)?;

        let result = self
            .inner
            .execute_language(query, language, param_map)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
    }

    /// Converts a JS params value (object or null/undefined) to an optional
    /// `HashMap<String, Value>` suitable for `execute_language`.
    fn convert_params(params: Option<JsValue>) -> Result<Option<HashMap<String, Value>>, JsError> {
        let Some(js_val) = params else {
            return Ok(None);
        };
        if js_val.is_null() || js_val.is_undefined() {
            return Ok(None);
        }
        let json_val: serde_json::Value =
            serde_wasm_bindgen::from_value(js_val).map_err(|e| JsError::new(&e.to_string()))?;
        json_params_to_map(Some(&json_val)).map_err(|e| JsError::new(&e))
    }
}
