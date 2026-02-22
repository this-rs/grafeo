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

use js_sys::Array;
use wasm_bindgen::prelude::*;

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
    /// Supported languages: `"gql"`, `"cypher"`, `"sparql"`, `"gremlin"`, `"graphql"`.
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
        let result = self
            .inner
            .execute_language(query, language, None)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let rows = Array::new_with_length(result.rows.len() as u32);
        for (i, row) in result.rows.iter().enumerate() {
            rows.set(i as u32, types::row_to_js_object(&result.columns, row));
        }
        Ok(rows.into())
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

    /// Returns the Grafeo version.
    pub fn version() -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}
