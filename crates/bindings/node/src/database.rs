//! Main entry point for using Grafeo from Node.js.
//!
//! [`JsGrafeoDB`] wraps the Rust database engine and gives you a JavaScript API.

use std::collections::HashMap;
use std::sync::Arc;

use napi::JsString;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_engine::config::Config;
use grafeo_engine::database::{GrafeoDB, QueryResult as EngineQueryResult};

use crate::error::NodeGrafeoError;
use crate::graph::{JsEdge, JsNode};
use crate::query::QueryResult;
use crate::transaction::Transaction;
use crate::types;

/// Converts a serde_json filter map to a Grafeo filter map.
fn convert_json_filters(
    filters: Option<HashMap<String, serde_json::Value>>,
) -> Result<Option<HashMap<String, Value>>> {
    let Some(map) = filters else {
        return Ok(None);
    };
    let mut result = HashMap::new();
    for (key, val) in &map {
        let grafeo_val = json_to_value(val)?;
        result.insert(key.clone(), grafeo_val);
    }
    Ok(Some(result))
}

/// Validate a JavaScript number as a safe node ID.
///
/// JavaScript numbers are f64, but entity IDs are u64. This rejects
/// negative values, NaN, Infinity, and values beyond `Number.MAX_SAFE_INTEGER`.
fn validate_node_id(id: f64) -> Result<NodeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid node ID: {id}")).into());
    }
    Ok(NodeId(id as u64))
}

/// Validate a JavaScript number as a safe edge ID.
fn validate_edge_id(id: f64) -> Result<EdgeId> {
    if !(0.0..=9_007_199_254_740_991.0).contains(&id) {
        return Err(NodeGrafeoError::InvalidArgument(format!("Invalid edge ID: {id}")).into());
    }
    Ok(EdgeId(id as u64))
}

/// Your connection to a Grafeo database.
#[napi(js_name = "GrafeoDB")]
pub struct JsGrafeoDB {
    inner: Arc<RwLock<GrafeoDB>>,
}

#[napi]
impl JsGrafeoDB {
    /// Create a database. Pass a path for persistence, or omit for in-memory.
    #[napi(factory)]
    pub fn create(path: Option<String>) -> Result<Self> {
        let config = match path {
            Some(p) => Config::persistent(p),
            None => Config::in_memory(),
        };
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Open an existing database at the given path.
    #[napi(factory)]
    pub fn open(path: String) -> Result<Self> {
        let config = Config::persistent(path);
        let db = GrafeoDB::with_config(config).map_err(NodeGrafeoError::from)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Shared implementation for all language-specific execute methods.
    async fn execute_language_impl(
        &self,
        language: &'static str,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        let db = self.inner.clone();
        let result = tokio::task::spawn_blocking(move || {
            let db = db.read();
            execute_language_query(&db, &query, language, params.as_ref())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.inner.read();
        let (nodes, edges) = extract_entities(&result, &db);

        Ok(QueryResult::with_metrics(
            result.columns,
            result.rows,
            nodes,
            edges,
            result.execution_time_ms,
            result.rows_scanned,
        ))
    }

    /// Execute a GQL query. Returns a Promise<QueryResult>.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gql", query, params).await
    }

    /// Execute a Cypher query.
    #[cfg(feature = "cypher")]
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("cypher", query, params).await
    }

    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    #[cfg(feature = "sql-pgq")]
    #[napi(js_name = "executeSql")]
    pub async fn execute_sql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sql", query, params).await
    }

    /// Execute a Gremlin query.
    #[cfg(feature = "gremlin")]
    #[napi(js_name = "executeGremlin")]
    pub async fn execute_gremlin(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gremlin", query, params).await
    }

    /// Execute a GraphQL query.
    #[cfg(feature = "graphql")]
    #[napi(js_name = "executeGraphql")]
    pub async fn execute_graphql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("graphql", query, params).await
    }

    /// Execute a SPARQL query against the RDF triple store.
    #[cfg(feature = "sparql")]
    #[napi(js_name = "executeSparql")]
    pub async fn execute_sparql(&self, query: String) -> Result<QueryResult> {
        self.execute_language_impl("sparql", query, None).await
    }

    /// Create a node with labels and optional properties.
    #[napi(js_name = "createNode")]
    pub fn create_node(
        &self,
        env: Env,
        labels: Vec<String>,
        properties: Option<Object<'_>>,
    ) -> Result<JsNode> {
        let db = self.inner.read();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

        let id = if let Some(props_obj) = properties {
            let mut props = Vec::new();
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_node_with_props(&label_refs, props)
        } else {
            db.create_node(&label_refs)
        };

        fetch_node(&db, id)
    }

    /// Create an edge between two nodes.
    #[napi(js_name = "createEdge")]
    pub fn create_edge(
        &self,
        env: Env,
        source_id: f64,
        target_id: f64,
        edge_type: String,
        properties: Option<Object<'_>>,
    ) -> Result<JsEdge> {
        let db = self.inner.read();
        let src = validate_node_id(source_id)?;
        let dst = validate_node_id(target_id)?;

        let id = if let Some(props_obj) = properties {
            let mut props = Vec::new();
            let keys = props_obj.get_property_names()?;
            let len = keys.get_array_length()?;
            for i in 0..len {
                let key: JsString = keys.get_element(i)?;
                let key_str = key.into_utf8()?.into_owned()?;
                let value: Unknown<'_> = props_obj.get_named_property(&key_str)?;
                let val = types::js_to_value(&env, value)?;
                props.push((grafeo_common::types::PropertyKey::new(key_str), val));
            }
            db.create_edge_with_props(src, dst, &edge_type, props)
        } else {
            db.create_edge(src, dst, &edge_type)
        };

        fetch_edge(&db, id)
    }

    /// Get a node by ID.
    #[napi(js_name = "getNode")]
    pub fn get_node(&self, id: f64) -> Result<Option<JsNode>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node(node_id).map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsNode::new(node_id, labels, properties)
        }))
    }

    /// Get an edge by ID.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(&self, id: f64) -> Result<Option<JsEdge>> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.get_edge(edge_id).map(|edge| {
            let properties = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsEdge::new(
                edge_id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        }))
    }

    /// Delete a node by ID. Returns true if the node existed.
    #[napi(js_name = "deleteNode")]
    pub fn delete_node(&self, id: f64) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_node(node_id))
    }

    /// Delete an edge by ID. Returns true if the edge existed.
    #[napi(js_name = "deleteEdge")]
    pub fn delete_edge(&self, id: f64) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.delete_edge(edge_id))
    }

    /// Set a property on a node.
    #[napi(js_name = "setNodeProperty")]
    pub fn set_node_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_node_property(node_id, &key, val);
        Ok(())
    }

    /// Set a property on an edge.
    #[napi(js_name = "setEdgeProperty")]
    pub fn set_edge_property(
        &self,
        env: Env,
        id: f64,
        key: String,
        value: Unknown<'_>,
    ) -> Result<()> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        let val = types::js_to_value(&env, value)?;
        db.set_edge_property(edge_id, &key, val);
        Ok(())
    }

    /// Get the number of nodes.
    #[napi(getter, js_name = "nodeCount")]
    pub fn node_count(&self) -> u32 {
        self.inner.read().node_count() as u32
    }

    /// Get the number of edges.
    #[napi(getter, js_name = "edgeCount")]
    pub fn edge_count(&self) -> u32 {
        self.inner.read().edge_count() as u32
    }

    /// Begin a transaction with an optional isolation level.
    ///
    /// Isolation levels: "read_committed", "snapshot" (default), "serializable".
    #[napi(js_name = "beginTransaction")]
    pub fn begin_transaction(&self, isolation_level: Option<String>) -> Result<Transaction> {
        Transaction::new(self.inner.clone(), isolation_level.as_deref())
    }

    /// Create a vector similarity index on a node property.
    #[napi(js_name = "createVectorIndex")]
    pub async fn create_vector_index(
        &self,
        label: String,
        property: String,
        dimensions: Option<u32>,
        metric: Option<String>,
        m: Option<u32>,
        ef_construction: Option<u32>,
    ) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.create_vector_index(
                &label,
                &property,
                dimensions.map(|d| d as usize),
                metric.as_deref(),
                m.map(|v| v as usize),
                ef_construction.map(|v| v as usize),
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Drop a vector index for the given label and property.
    /// Returns true if the index existed and was removed.
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "dropVectorIndex")]
    pub async fn drop_vector_index(&self, label: String, property: String) -> Result<bool> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            Ok(db.drop_vector_index(&label, &property))
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Rebuild a vector index by rescanning all matching nodes.
    /// Preserves the original index configuration.
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "rebuildVectorIndex")]
    pub async fn rebuild_vector_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.rebuild_vector_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for the k nearest neighbors of a query vector.
    #[napi(js_name = "vectorSearch")]
    pub async fn vector_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .vector_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            // Return as [[nodeId, distance], ...] since napi doesn't have tuples
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Bulk-insert nodes with vector properties.
    #[napi(js_name = "batchCreateNodes")]
    pub async fn batch_create_nodes(
        &self,
        label: String,
        property: String,
        vectors: Vec<Vec<f64>>,
    ) -> Result<Vec<f64>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let vecs_f32: Vec<Vec<f32>> = vectors
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let ids = db.batch_create_nodes(&label, &property, vecs_f32);
            Ok(ids
                .into_iter()
                .map(|id| id.as_u64() as f64)
                .collect::<Vec<f64>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Batch search for nearest neighbors of multiple query vectors.
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "batchVectorSearch")]
    pub async fn batch_vector_search(
        &self,
        label: String,
        property: String,
        queries: Vec<Vec<f64>>,
        k: u32,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let queries_f32: Vec<Vec<f32>> = queries
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .collect();
            let results = db
                .batch_vector_search(
                    &label,
                    &property,
                    &queries_f32,
                    k as usize,
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|inner| {
                    inner
                        .into_iter()
                        .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                        .collect::<Vec<Vec<f64>>>()
                })
                .collect::<Vec<Vec<Vec<f64>>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "mmrSearch")]
    #[allow(clippy::too_many_arguments)]
    pub async fn mmr_search(
        &self,
        label: String,
        property: String,
        query: Vec<f64>,
        k: u32,
        fetch_k: Option<u32>,
        lambda_mult: Option<f64>,
        ef: Option<u32>,
        filters: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f64>>> {
        let filter_map = convert_json_filters(filters)?;
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
            let results = db
                .mmr_search(
                    &label,
                    &property,
                    &query_f32,
                    k as usize,
                    fetch_k.map(|v| v as usize),
                    lambda_mult.map(|v| v as f32),
                    ef.map(|v| v as usize),
                    filter_map.as_ref(),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, dist as f64])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    // ── Text Search ──────────────────────────────────────────────

    /// Create a BM25 text index on a node property for full-text search.
    #[cfg(feature = "text-index")]
    #[napi(js_name = "createTextIndex")]
    pub async fn create_text_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.create_text_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Drop a text index for the given label and property.
    #[cfg(feature = "text-index")]
    #[napi(js_name = "dropTextIndex")]
    pub async fn drop_text_index(&self, label: String, property: String) -> Result<bool> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            Ok(db.drop_text_index(&label, &property))
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Rebuild a text index by rescanning all matching nodes.
    #[cfg(feature = "text-index")]
    #[napi(js_name = "rebuildTextIndex")]
    pub async fn rebuild_text_index(&self, label: String, property: String) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            db.rebuild_text_index(&label, &property)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search a text index using BM25 scoring.
    ///
    /// Returns an array of [nodeId, score] pairs sorted by descending relevance.
    #[cfg(feature = "text-index")]
    #[napi(js_name = "textSearch")]
    pub async fn text_search(
        &self,
        label: String,
        property: String,
        query: String,
        k: u32,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .text_search(&label, &property, &query, k as usize)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Perform hybrid search combining text (BM25) and vector similarity.
    ///
    /// Returns an array of [nodeId, score] pairs.
    #[cfg(feature = "hybrid-search")]
    #[napi(js_name = "hybridSearch")]
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        label: String,
        text_property: String,
        vector_property: String,
        query_text: String,
        k: u32,
        query_vector: Option<Vec<f64>>,
        fusion: Option<String>,
        weights: Option<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let fusion_method = match fusion.as_deref() {
                Some("weighted") => {
                    let w = weights.unwrap_or_else(|| vec![0.5, 0.5]);
                    Some(grafeo_core::index::text::FusionMethod::Weighted { weights: w })
                }
                _ => None,
            };

            let query_vec_f32: Option<Vec<f32>> =
                query_vector.map(|v| v.iter().map(|&x| x as f32).collect());

            let db = db.read();
            let results = db
                .hybrid_search(
                    &label,
                    &text_property,
                    &vector_property,
                    &query_text,
                    query_vec_f32.as_deref(),
                    k as usize,
                    fusion_method,
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, score)| vec![id.as_u64() as f64, score])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Remove a property from a node. Returns true if the property existed.
    #[napi(js_name = "removeNodeProperty")]
    pub fn remove_node_property(&self, id: f64, key: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_node_property(node_id, &key))
    }

    /// Remove a property from an edge. Returns true if the property existed.
    #[napi(js_name = "removeEdgeProperty")]
    pub fn remove_edge_property(&self, id: f64, key: String) -> Result<bool> {
        let edge_id = validate_edge_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_edge_property(edge_id, &key))
    }

    /// Add a label to an existing node. Returns true if the label was added.
    #[napi(js_name = "addNodeLabel")]
    pub fn add_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.add_node_label(node_id, &label))
    }

    /// Remove a label from a node. Returns true if the label was removed.
    #[napi(js_name = "removeNodeLabel")]
    pub fn remove_node_label(&self, id: f64, label: String) -> Result<bool> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.remove_node_label(node_id, &label))
    }

    /// Get all labels for a node. Returns null if the node doesn't exist.
    #[napi(js_name = "getNodeLabels")]
    pub fn get_node_labels(&self, id: f64) -> Result<Option<Vec<String>>> {
        let node_id = validate_node_id(id)?;
        let db = self.inner.read();
        Ok(db.get_node_labels(node_id))
    }

    /// Returns high-level database information as a JSON object.
    #[napi]
    pub fn info(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let info = db.info();
        serde_json::to_value(&info).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }

    /// Returns schema information (labels, edge types, property keys) as a JSON object.
    #[napi]
    pub fn schema(&self) -> Result<serde_json::Value> {
        let db = self.inner.read();
        let schema = db.schema();
        serde_json::to_value(&schema).map_err(|e| NodeGrafeoError::Database(e.to_string()).into())
    }

    /// Returns the Grafeo engine version string.
    #[napi]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    /// Close the database.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner
            .read()
            .close()
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)
    }

    // ── Change Data Capture ─────────────────────────────────────────────

    /// Returns the full change history for a node.
    #[cfg(feature = "cdc")]
    #[napi(js_name = "nodeHistory")]
    pub async fn node_history(&self, node_id: f64) -> Result<Vec<serde_json::Value>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let id = grafeo_common::types::NodeId::new(node_id as u64);
            let events = db
                .history(id)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns the full change history for an edge.
    #[cfg(feature = "cdc")]
    #[napi(js_name = "edgeHistory")]
    pub async fn edge_history(&self, edge_id: f64) -> Result<Vec<serde_json::Value>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let id = grafeo_common::types::EdgeId::new(edge_id as u64);
            let events = db
                .history(id)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns change events for a node since a given epoch.
    #[cfg(feature = "cdc")]
    #[napi(js_name = "nodeHistorySince")]
    pub async fn node_history_since(
        &self,
        node_id: f64,
        since_epoch: f64,
    ) -> Result<Vec<serde_json::Value>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let id = grafeo_common::types::NodeId::new(node_id as u64);
            let events = db
                .history_since(id, grafeo_common::types::EpochId(since_epoch as u64))
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Returns all change events across entities in an epoch range.
    #[cfg(feature = "cdc")]
    #[napi(js_name = "changesBetween")]
    pub async fn changes_between(
        &self,
        start_epoch: f64,
        end_epoch: f64,
    ) -> Result<Vec<serde_json::Value>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let events = db
                .changes_between(
                    grafeo_common::types::EpochId(start_epoch as u64),
                    grafeo_common::types::EpochId(end_epoch as u64),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(events.iter().map(change_event_to_json).collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

// Embed methods live in a separate impl block so the entire block can be
// conditionally compiled.  napi-rs generates callback registrations for every
// method inside a `#[napi]` impl, so a per-method `#[cfg]` doesn't work.
#[cfg(feature = "embed")]
#[napi]
impl JsGrafeoDB {
    /// Register an ONNX embedding model for text-to-vector conversion.
    ///
    /// Once registered, use embedText() and vectorSearchText() with the model name.
    #[napi(js_name = "registerEmbeddingModel")]
    pub async fn register_embedding_model(
        &self,
        name: String,
        model_path: String,
        tokenizer_path: String,
        batch_size: Option<u32>,
    ) -> Result<()> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = grafeo_engine::embedding::OnnxEmbeddingModel::from_files(
                &name,
                &model_path,
                &tokenizer_path,
            )
            .map_err(NodeGrafeoError::from)
            .map_err(napi::Error::from)?;
            if let Some(bs) = batch_size {
                model = model.with_batch_size(bs as usize);
            }
            let db = db.read();
            db.register_embedding_model(&name, std::sync::Arc::new(model));
            Ok(())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Generate embeddings for a list of texts using a registered model.
    ///
    /// Returns an array of float arrays, one per input text.
    #[napi(js_name = "embedText")]
    pub async fn embed_text(
        &self,
        model_name: String,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let results = db
                .embed_text(&model_name, &text_refs)
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            // Convert f32 → f64 for JavaScript number compatibility
            Ok(results
                .into_iter()
                .map(|v| v.into_iter().map(f64::from).collect())
                .collect())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }

    /// Search a vector index using a text query, generating the embedding on-the-fly.
    ///
    /// Returns an array of [nodeId, distance] pairs.
    #[cfg(feature = "vector-index")]
    #[napi(js_name = "vectorSearchText")]
    pub async fn vector_search_text(
        &self,
        label: String,
        property: String,
        model_name: String,
        query_text: String,
        k: u32,
        ef: Option<u32>,
    ) -> Result<Vec<Vec<f64>>> {
        let db = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = db.read();
            let results = db
                .vector_search_text(
                    &label,
                    &property,
                    &model_name,
                    &query_text,
                    k as usize,
                    ef.map(|e| e as usize),
                )
                .map_err(NodeGrafeoError::from)
                .map_err(napi::Error::from)?;
            Ok(results
                .into_iter()
                .map(|(id, dist)| vec![id.as_u64() as f64, f64::from(dist)])
                .collect::<Vec<Vec<f64>>>())
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
    }
}

/// Execute a query in a given language with optional JSON params.
fn execute_language_query(
    db: &GrafeoDB,
    query: &str,
    language: &str,
    params: Option<&serde_json::Value>,
) -> std::result::Result<EngineQueryResult, napi::Error> {
    let param_map = convert_json_params(params)?;
    db.execute_language(query, language, param_map)
        .map_err(NodeGrafeoError::from)
        .map_err(napi::Error::from)
}

/// Convert JSON params to a HashMap<String, Value>.
fn convert_json_params(
    params: Option<&serde_json::Value>,
) -> std::result::Result<Option<HashMap<String, Value>>, napi::Error> {
    grafeo_bindings_common::json::json_params_to_map(params)
        .map_err(|msg| NodeGrafeoError::InvalidArgument(msg).into())
}

/// Convert a serde_json::Value to a Grafeo Value.
pub(crate) fn json_to_value(v: &serde_json::Value) -> std::result::Result<Value, napi::Error> {
    Ok(grafeo_bindings_common::json::json_to_value(v))
}

/// Fetch a node from the database and wrap it as JsNode.
fn fetch_node(db: &GrafeoDB, id: NodeId) -> Result<JsNode> {
    db.get_node(id)
        .map(|node| {
            let labels: Vec<String> = node.labels.iter().map(|s| s.to_string()).collect();
            let properties = node
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsNode::new(id, labels, properties)
        })
        .ok_or_else(|| NodeGrafeoError::Database("Failed to fetch created node".into()).into())
}

/// Fetch an edge from the database and wrap it as JsEdge.
fn fetch_edge(db: &GrafeoDB, id: EdgeId) -> Result<JsEdge> {
    db.get_edge(id)
        .map(|edge| {
            let properties = edge
                .properties
                .into_iter()
                .map(|(k, v)| (k.as_str().to_string(), v))
                .collect();
            JsEdge::new(
                id,
                edge.edge_type.to_string(),
                edge.src,
                edge.dst,
                properties,
            )
        })
        .ok_or_else(|| NodeGrafeoError::Database("Failed to fetch created edge".into()).into())
}

/// Extract nodes and edges from query results based on column types.
pub(crate) fn extract_entities(
    result: &EngineQueryResult,
    _db: &GrafeoDB,
) -> (Vec<JsNode>, Vec<JsEdge>) {
    let (raw_nodes, raw_edges) = grafeo_bindings_common::entity::extract_entities(result);
    let nodes = raw_nodes
        .into_iter()
        .map(|n| JsNode::new(n.id, n.labels, n.properties))
        .collect();
    let edges = raw_edges
        .into_iter()
        .map(|e| JsEdge::new(e.id, e.edge_type, e.source_id, e.target_id, e.properties))
        .collect();
    (nodes, edges)
}

/// Convert a Grafeo Value to serde_json::Value.
fn grafeo_value_to_json(v: &Value) -> serde_json::Value {
    grafeo_bindings_common::json::value_to_json(v)
}

/// Convert a CDC ChangeEvent to a JSON object.
#[cfg(feature = "cdc")]
fn change_event_to_json(event: &grafeo_engine::cdc::ChangeEvent) -> serde_json::Value {
    let entity_type = if event.entity_id.is_node() {
        "node"
    } else {
        "edge"
    };
    let kind = match event.kind {
        grafeo_engine::cdc::ChangeKind::Create => "create",
        grafeo_engine::cdc::ChangeKind::Update => "update",
        grafeo_engine::cdc::ChangeKind::Delete => "delete",
    };

    let before = match &event.before {
        Some(props) => {
            let obj: serde_json::Map<String, serde_json::Value> = props
                .iter()
                .map(|(k, v)| (k.clone(), grafeo_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        None => serde_json::Value::Null,
    };

    let after = match &event.after {
        Some(props) => {
            let obj: serde_json::Map<String, serde_json::Value> = props
                .iter()
                .map(|(k, v)| (k.clone(), grafeo_value_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        None => serde_json::Value::Null,
    };

    serde_json::json!({
        "entity_id": event.entity_id.as_u64(),
        "entity_type": entity_type,
        "kind": kind,
        "epoch": event.epoch.0,
        "timestamp": event.timestamp,
        "before": before,
        "after": after,
    })
}
