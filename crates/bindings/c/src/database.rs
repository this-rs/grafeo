//! All `#[no_mangle] extern "C"` functions exposed by the Obrain C API.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use obrain_common::types::{EdgeId, NodeId};
use obrain_engine::config::Config;
use obrain_engine::database::ObrainDB;

use crate::error::{ObrainStatus, set_error, set_last_error, str_from_ptr};
use crate::types::{ObrainDatabase, ObrainEdge, ObrainNode, ObrainResult, ObrainTransaction};

// ===== Helpers =====

/// Dereference an opaque database pointer, returning an error status on null.
macro_rules! db_ref {
    ($ptr:expr) => {{
        if $ptr.is_null() {
            set_last_error("Null database pointer");
            return ObrainStatus::ErrorNullPointer;
        }
        // SAFETY: Caller guarantees ptr from obrain_open* and not yet freed.
        unsafe { &*$ptr }
    }};
}

/// Same as `db_ref!` but returns a null pointer on error (for functions that
/// return pointers).
macro_rules! db_ref_or_null {
    ($ptr:expr) => {{
        if $ptr.is_null() {
            set_last_error("Null database pointer");
            return std::ptr::null_mut();
        }
        // SAFETY: Caller guarantees ptr from obrain_open* and not yet freed.
        unsafe { &*$ptr }
    }};
}

/// Serialize a `QueryResult` into a `ObrainResult`.
fn build_result(result: &obrain_engine::database::QueryResult) -> *mut ObrainResult {
    let json_rows: Vec<serde_json::Value> = result
        .rows
        .iter()
        .map(|row| {
            let obj: serde_json::Map<String, serde_json::Value> = result
                .columns
                .iter()
                .zip(row.iter())
                .map(|(col, val)| (col.clone(), crate::types::value_to_json(val)))
                .collect();
            serde_json::Value::Object(obj)
        })
        .collect();

    let json_str = serde_json::to_string(&json_rows).unwrap_or_default();
    let c_json = CString::new(json_str).unwrap_or_default();

    Box::into_raw(Box::new(ObrainResult {
        json: c_json,
        row_count: result.rows.len(),
        execution_time_ms: result.execution_time_ms.unwrap_or(0.0),
        rows_scanned: result.rows_scanned.unwrap_or(0),
    }))
}

// =========================================================================
// Lifecycle
// =========================================================================

/// Create a new in-memory database.
///
/// Returns an opaque pointer, or null on error (check `obrain_last_error()`).
#[unsafe(no_mangle)]
pub extern "C" fn obrain_open_memory() -> *mut ObrainDatabase {
    let db = ObrainDB::new_in_memory();
    Box::into_raw(Box::new(ObrainDatabase {
        inner: Arc::new(RwLock::new(db)),
    }))
}

/// Open or create a persistent database at `path`.
///
/// Returns an opaque pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_open(path: *const c_char) -> *mut ObrainDatabase {
    let Ok(path_str) = str_from_ptr(path) else {
        return std::ptr::null_mut();
    };
    match ObrainDB::with_config(Config::persistent(path_str)) {
        Ok(db) => Box::into_raw(Box::new(ObrainDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Open an existing database in read-only mode.
///
/// Uses a shared file lock, so multiple processes can read the same
/// .obrain file concurrently. Mutations will return an error.
///
/// Returns an opaque pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_open_read_only(path: *const c_char) -> *mut ObrainDatabase {
    let Ok(path_str) = str_from_ptr(path) else {
        return std::ptr::null_mut();
    };
    match ObrainDB::with_config(Config::read_only(path_str)) {
        Ok(db) => Box::into_raw(Box::new(ObrainDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Close the database, flushing pending writes.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_close(db: *mut ObrainDatabase) -> ObrainStatus {
    if db.is_null() {
        return ObrainStatus::Ok;
    }
    // SAFETY: Caller guarantees valid pointer from obrain_open*.
    let db = unsafe { &*db };
    match db.inner.read().close() {
        Ok(()) => ObrainStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Free a database handle. Must be called after `obrain_close`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_database(db: *mut ObrainDatabase) {
    if !db.is_null() {
        // SAFETY: We take ownership back and drop it.
        unsafe { drop(Box::from_raw(db)) };
    }
}

/// Returns the library version string. The pointer is static and must NOT be freed.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_version() -> *const c_char {
    // Include a trailing NUL in the byte literal.
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr().cast::<c_char>()
}

// =========================================================================
// Query Execution
// =========================================================================

/// Execute a GQL query. Returns a result pointer, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = db.inner.read();
    match guard.execute_language(query_str, "gql", None) {
        Ok(result) => build_result(&result),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a GQL query with JSON-encoded parameters.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_with_params(
    db: *mut ObrainDatabase,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = db.inner.read();
    let params = crate::types::parse_params(params_json);
    match guard.execute_language(query_str, "gql", params) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a Cypher query.
#[cfg(feature = "cypher")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_cypher(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "cypher", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a Gremlin query.
#[cfg(feature = "gremlin")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_gremlin(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "gremlin", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a GraphQL query.
#[cfg(feature = "graphql")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_graphql(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "graphql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SPARQL query.
#[cfg(feature = "sparql")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_sparql(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "sparql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
#[cfg(feature = "sql-pgq")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_execute_sql(
    db: *mut ObrainDatabase,
    query: *const c_char,
) -> *mut ObrainResult {
    let db = db_ref_or_null!(db);
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    match db.inner.read().execute_language(query_str, "sql", None) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

// =========================================================================
// Result Access
// =========================================================================

/// Get the JSON-encoded result rows. Returns null if result is null.
/// The pointer is valid until `obrain_free_result` is called.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_result_json(result: *const ObrainResult) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer from obrain_execute*.
    unsafe { &*result }.json.as_ptr()
}

/// Get the number of rows in the result.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_result_row_count(result: *const ObrainResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.row_count
}

/// Get execution time in milliseconds.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_result_execution_time_ms(result: *const ObrainResult) -> f64 {
    if result.is_null() {
        return 0.0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.execution_time_ms
}

/// Get estimated rows scanned.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_result_rows_scanned(result: *const ObrainResult) -> u64 {
    if result.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*result }.rows_scanned
}

/// Free a query result.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_result(result: *mut ObrainResult) {
    if !result.is_null() {
        // SAFETY: We take ownership back and drop it.
        unsafe { drop(Box::from_raw(result)) };
    }
}

// =========================================================================
// Node CRUD
// =========================================================================

/// Create a node with labels (JSON array) and optional properties (JSON object).
/// Returns the new node ID, or `u64::MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_create_node(
    db: *mut ObrainDatabase,
    labels_json: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let labels = crate::types::parse_labels(labels_json).unwrap_or_default();
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    let guard = db.inner.read();
    let id = if let Some(props) = crate::types::parse_properties(properties_json) {
        guard.create_node_with_props(&label_refs, props)
    } else {
        guard.create_node(&label_refs)
    };
    id.as_u64()
}

/// Get a node by ID. Writes into `out`. Returns `Ok` or an error status.
/// On success, `out` must be freed with `obrain_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_get_node(
    db: *mut ObrainDatabase,
    id: u64,
    out: *mut *mut ObrainNode,
) -> ObrainStatus {
    let db = db_ref!(db);
    if out.is_null() {
        set_last_error("Null output pointer");
        return ObrainStatus::ErrorNullPointer;
    }
    let guard = db.inner.read();
    match guard.get_node(NodeId::new(id)) {
        Some(node) => {
            let labels: Vec<serde_json::Value> = node
                .labels
                .iter()
                .map(|l| serde_json::Value::String(l.to_string()))
                .collect();
            let labels_json = CString::new(serde_json::to_string(&labels).unwrap_or_default())
                .unwrap_or_default();
            let properties_json = crate::types::properties_to_json(&node.properties);

            let gnode = Box::new(ObrainNode {
                id: node.id.as_u64(),
                labels_json,
                properties_json,
            });
            // SAFETY: We checked out is not null above.
            unsafe { *out = Box::into_raw(gnode) };
            ObrainStatus::Ok
        }
        None => {
            set_last_error(&format!("Node not found: {id}"));
            // SAFETY: We checked out is not null above.
            unsafe { *out = std::ptr::null_mut() };
            ObrainStatus::ErrorDatabase
        }
    }
}

/// Delete a node by ID. Returns 1 if deleted, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_delete_node(db: *mut ObrainDatabase, id: u64) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    i32::from(db.inner.read().delete_node(NodeId::new(id)))
}

/// Set a property on a node. `value_json` is a JSON-encoded value.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_set_node_property(
    db: *mut ObrainDatabase,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> ObrainStatus {
    let db = db_ref!(db);
    let key_str = match str_from_ptr(key) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return ObrainStatus::ErrorSerialization;
    };
    db.inner
        .read()
        .set_node_property(NodeId::new(id), key_str, value);
    ObrainStatus::Ok
}

/// Remove a property from a node. Returns 1 if removed, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_remove_node_property(
    db: *mut ObrainDatabase,
    id: u64,
    key: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(key_str) = str_from_ptr(key) else {
        return -1;
    };
    i32::from(
        db.inner
            .read()
            .remove_node_property(NodeId::new(id), key_str),
    )
}

/// Add a label to a node. Returns 1 if added, 0 if already present.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_add_node_label(
    db: *mut ObrainDatabase,
    id: u64,
    label: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return -1;
    };
    i32::from(db.inner.read().add_node_label(NodeId::new(id), label_str))
}

/// Remove a label from a node. Returns 1 if removed, 0 if not present.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_remove_node_label(
    db: *mut ObrainDatabase,
    id: u64,
    label: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return -1;
    };
    i32::from(
        db.inner
            .read()
            .remove_node_label(NodeId::new(id), label_str),
    )
}

/// Get labels for a node as a JSON array string.
/// Returns null if node not found. Caller must free with `obrain_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_get_node_labels(db: *mut ObrainDatabase, id: u64) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    match db.inner.read().get_node_labels(NodeId::new(id)) {
        Some(labels) => {
            let json = serde_json::to_string(&labels).unwrap_or_default();
            CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
        }
        None => std::ptr::null_mut(),
    }
}

/// Free a `ObrainNode` returned by `obrain_get_node`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_node(node: *mut ObrainNode) {
    if !node.is_null() {
        // SAFETY: We take ownership back.
        unsafe { drop(Box::from_raw(node)) };
    }
}

/// Access the node ID from a `ObrainNode`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_node_id(node: *const ObrainNode) -> u64 {
    if node.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.id
}

/// Access labels JSON from a `ObrainNode`. Valid until `obrain_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_node_labels_json(node: *const ObrainNode) -> *const c_char {
    if node.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.labels_json.as_ptr()
}

/// Access properties JSON from a `ObrainNode`. Valid until `obrain_free_node`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_node_properties_json(node: *const ObrainNode) -> *const c_char {
    if node.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*node }.properties_json.as_ptr()
}

// =========================================================================
// Edge CRUD
// =========================================================================

/// Create an edge. `properties_json` may be null for no properties.
/// Returns the new edge ID, or `u64::MAX` on error.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_create_edge(
    db: *mut ObrainDatabase,
    source_id: u64,
    target_id: u64,
    edge_type: *const c_char,
    properties_json: *const c_char,
) -> u64 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(type_str) = str_from_ptr(edge_type) else {
        return u64::MAX;
    };
    let src = NodeId::new(source_id);
    let dst = NodeId::new(target_id);
    let guard = db.inner.read();

    let id = if let Some(props) = crate::types::parse_properties(properties_json) {
        guard.create_edge_with_props(src, dst, type_str, props)
    } else {
        guard.create_edge(src, dst, type_str)
    };
    id.as_u64()
}

/// Get an edge by ID. Writes into `out`. Returns `Ok` or error status.
/// On success, `out` must be freed with `obrain_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_get_edge(
    db: *mut ObrainDatabase,
    id: u64,
    out: *mut *mut ObrainEdge,
) -> ObrainStatus {
    let db = db_ref!(db);
    if out.is_null() {
        set_last_error("Null output pointer");
        return ObrainStatus::ErrorNullPointer;
    }
    let guard = db.inner.read();
    match guard.get_edge(EdgeId(id)) {
        Some(edge) => {
            let edge_type = CString::new(edge.edge_type.to_string()).unwrap_or_default();
            let properties_json = crate::types::properties_to_json(&edge.properties);

            let gedge = Box::new(ObrainEdge {
                id: edge.id.as_u64(),
                source_id: edge.src.as_u64(),
                target_id: edge.dst.as_u64(),
                edge_type,
                properties_json,
            });
            // SAFETY: We checked out is not null above.
            unsafe { *out = Box::into_raw(gedge) };
            ObrainStatus::Ok
        }
        None => {
            set_last_error(&format!("Edge not found: {id}"));
            // SAFETY: We checked out is not null above.
            unsafe { *out = std::ptr::null_mut() };
            ObrainStatus::ErrorDatabase
        }
    }
}

/// Delete an edge by ID. Returns 1 if deleted, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_delete_edge(db: *mut ObrainDatabase, id: u64) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    i32::from(db.inner.read().delete_edge(EdgeId(id)))
}

/// Set a property on an edge.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_set_edge_property(
    db: *mut ObrainDatabase,
    id: u64,
    key: *const c_char,
    value_json: *const c_char,
) -> ObrainStatus {
    let db = db_ref!(db);
    let key_str = match str_from_ptr(key) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return ObrainStatus::ErrorSerialization;
    };
    db.inner
        .read()
        .set_edge_property(EdgeId(id), key_str, value);
    ObrainStatus::Ok
}

/// Remove a property from an edge. Returns 1 if removed, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_remove_edge_property(
    db: *mut ObrainDatabase,
    id: u64,
    key: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(key_str) = str_from_ptr(key) else {
        return -1;
    };
    i32::from(db.inner.read().remove_edge_property(EdgeId(id), key_str))
}

/// Free a `ObrainEdge` returned by `obrain_get_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_edge(edge: *mut ObrainEdge) {
    if !edge.is_null() {
        // SAFETY: We take ownership back.
        unsafe { drop(Box::from_raw(edge)) };
    }
}

/// Access the edge ID.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_id(edge: *const ObrainEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.id
}

/// Access source node ID.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_source_id(edge: *const ObrainEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.source_id
}

/// Access target node ID.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_target_id(edge: *const ObrainEdge) -> u64 {
    if edge.is_null() {
        return u64::MAX;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.target_id
}

/// Access edge type string. Valid until `obrain_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_type(edge: *const ObrainEdge) -> *const c_char {
    if edge.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.edge_type.as_ptr()
}

/// Access edge properties JSON. Valid until `obrain_free_edge`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_properties_json(edge: *const ObrainEdge) -> *const c_char {
    if edge.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*edge }.properties_json.as_ptr()
}

// =========================================================================
// Property Indexes
// =========================================================================

/// Create a property index.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_create_property_index(
    db: *mut ObrainDatabase,
    property: *const c_char,
) -> ObrainStatus {
    let db = db_ref!(db);
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    db.inner.read().create_property_index(prop_str);
    ObrainStatus::Ok
}

/// Drop a property index. Returns 1 if dropped, 0 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_drop_property_index(
    db: *mut ObrainDatabase,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return -1;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(prop_str) = str_from_ptr(property) else {
        return -1;
    };
    i32::from(db.inner.read().drop_property_index(prop_str))
}

/// Check if a property index exists. Returns 1 if exists, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_has_property_index(
    db: *mut ObrainDatabase,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let Ok(prop_str) = str_from_ptr(property) else {
        return 0;
    };
    i32::from(db.inner.read().has_property_index(prop_str))
}

/// Find nodes by property value. Writes node IDs into `out_ids` and count
/// into `out_count`. Caller must free `*out_ids` with `obrain_free_node_ids`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_find_nodes_by_property(
    db: *mut ObrainDatabase,
    property: *const c_char,
    value_json: *const c_char,
    out_ids: *mut *mut u64,
    out_count: *mut usize,
) -> ObrainStatus {
    let db = db_ref!(db);
    if out_ids.is_null() || out_count.is_null() {
        set_last_error("Null output pointer");
        return ObrainStatus::ErrorNullPointer;
    }
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(value) = crate::types::parse_value(value_json) else {
        set_last_error("Invalid JSON value");
        return ObrainStatus::ErrorSerialization;
    };
    let ids = db.inner.read().find_nodes_by_property(prop_str, &value);
    let count = ids.len();
    let mut raw_ids: Vec<u64> = ids.iter().map(|id| id.as_u64()).collect();
    raw_ids.shrink_to_fit();
    let ptr = raw_ids.as_mut_ptr();
    std::mem::forget(raw_ids);

    // SAFETY: We checked these are non-null above.
    unsafe {
        *out_ids = ptr;
        *out_count = count;
    }
    ObrainStatus::Ok
}

/// Free a node ID array from `obrain_find_nodes_by_property`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_node_ids(ids: *mut u64, count: usize) {
    if !ids.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in find_nodes_by_property.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(ids, count);
            drop(Box::from_raw(slice));
        }
    }
}

// =========================================================================
// Vector Operations
// =========================================================================

/// Create a vector similarity index on a node property.
/// `dimensions`, `m`, and `ef_construction` use -1 for default.
/// `metric` may be null for default (cosine).
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_create_vector_index(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
    dimensions: i32,
    metric: *const c_char,
    m: i32,
    ef_construction: i32,
) -> ObrainStatus {
    let db = db_ref!(db);
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let dims = if dimensions > 0 {
        Some(dimensions as usize)
    } else {
        None
    };
    let metric_str = if metric.is_null() {
        None
    } else {
        str_from_ptr(metric).ok()
    };
    let m_val = if m > 0 { Some(m as usize) } else { None };
    let ef_val = if ef_construction > 0 {
        Some(ef_construction as usize)
    } else {
        None
    };

    match db
        .inner
        .read()
        .create_vector_index(label_str, prop_str, dims, metric_str, m_val, ef_val)
    {
        Ok(()) => ObrainStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Drop a vector index for the given label and property.
/// Returns 1 if removed, 0 if not found.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_drop_vector_index(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
) -> i32 {
    if db.is_null() {
        set_last_error("Null database pointer");
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer from obrain_open*.
    let db = unsafe { &*db };
    let Ok(label_str) = str_from_ptr(label) else {
        return 0;
    };
    let Ok(prop_str) = str_from_ptr(property) else {
        return 0;
    };
    i32::from(db.inner.read().drop_vector_index(label_str, prop_str))
}

/// Rebuild a vector index by rescanning all matching nodes.
/// Preserves original configuration.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_rebuild_vector_index(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
) -> ObrainStatus {
    let db = db_ref!(db);
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().rebuild_vector_index(label_str, prop_str) {
        Ok(()) => ObrainStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Search for k nearest neighbors of a query vector.
/// Results written to `out_ids` and `out_distances` arrays of length `*out_count`.
/// Caller must free with `obrain_free_vector_results`.
/// `ef` uses -1 for default.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_vector_search(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
    query: *const f32,
    query_len: usize,
    k: usize,
    ef: i32,
    out_ids: *mut *mut u64,
    out_distances: *mut *mut f32,
    out_count: *mut usize,
) -> ObrainStatus {
    let db = db_ref!(db);
    if query.is_null() || out_ids.is_null() || out_distances.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return ObrainStatus::ErrorNullPointer;
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees query points to query_len f32s.
    let query_slice = unsafe { std::slice::from_raw_parts(query, query_len) };
    let ef_val = if ef > 0 { Some(ef as usize) } else { None };

    match db
        .inner
        .read()
        .vector_search(label_str, prop_str, query_slice, k, ef_val, None)
    {
        Ok(results) => {
            let count = results.len();
            let mut ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
            let mut dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
            ids.shrink_to_fit();
            dists.shrink_to_fit();
            let ids_ptr = ids.as_mut_ptr();
            let dists_ptr = dists.as_mut_ptr();
            std::mem::forget(ids);
            std::mem::forget(dists);

            // SAFETY: We checked these are non-null above.
            unsafe {
                *out_ids = ids_ptr;
                *out_distances = dists_ptr;
                *out_count = count;
            }
            ObrainStatus::Ok
        }
        Err(e) => set_error(&e),
    }
}

/// Search for diverse nearest neighbors using Maximal Marginal Relevance (MMR).
/// Results written to `out_ids` and `out_distances` arrays of length `*out_count`.
/// Caller must free with `obrain_free_vector_results`.
/// `fetch_k` and `ef` use -1 for defaults. `lambda` uses negative for default (0.5).
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_mmr_search(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
    query: *const f32,
    query_len: usize,
    k: usize,
    fetch_k: i32,
    lambda: f32,
    ef: i32,
    out_ids: *mut *mut u64,
    out_distances: *mut *mut f32,
    out_count: *mut usize,
) -> ObrainStatus {
    let db = db_ref!(db);
    if query.is_null() || out_ids.is_null() || out_distances.is_null() || out_count.is_null() {
        set_last_error("Null pointer argument");
        return ObrainStatus::ErrorNullPointer;
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees query points to query_len f32s.
    let query_slice = unsafe { std::slice::from_raw_parts(query, query_len) };
    let fetch_k_val = if fetch_k > 0 {
        Some(fetch_k as usize)
    } else {
        None
    };
    let lambda_val = if lambda >= 0.0 { Some(lambda) } else { None };
    let ef_val = if ef > 0 { Some(ef as usize) } else { None };

    match db.inner.read().mmr_search(
        label_str,
        prop_str,
        query_slice,
        k,
        fetch_k_val,
        lambda_val,
        ef_val,
        None,
    ) {
        Ok(results) => {
            let count = results.len();
            let mut ids: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
            let mut dists: Vec<f32> = results.iter().map(|(_, d)| *d).collect();
            ids.shrink_to_fit();
            dists.shrink_to_fit();
            let ids_ptr = ids.as_mut_ptr();
            let dists_ptr = dists.as_mut_ptr();
            std::mem::forget(ids);
            std::mem::forget(dists);

            // SAFETY: We checked these are non-null above.
            unsafe {
                *out_ids = ids_ptr;
                *out_distances = dists_ptr;
                *out_count = count;
            }
            ObrainStatus::Ok
        }
        Err(e) => set_error(&e),
    }
}

/// Bulk-insert nodes with vector properties. Returns node IDs in `out_ids`.
/// `vectors` is a flat array of `vector_count * dimensions` f32 values.
/// Caller must free `*out_ids` with `obrain_free_node_ids`.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_batch_create_nodes(
    db: *mut ObrainDatabase,
    label: *const c_char,
    property: *const c_char,
    vectors: *const f32,
    vector_count: usize,
    dimensions: usize,
    out_ids: *mut *mut u64,
) -> ObrainStatus {
    let db = db_ref!(db);
    if vectors.is_null() || out_ids.is_null() {
        set_last_error("Null pointer argument");
        return ObrainStatus::ErrorNullPointer;
    }
    let label_str = match str_from_ptr(label) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let prop_str = match str_from_ptr(property) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // SAFETY: Caller guarantees vectors points to vector_count * dimensions f32s.
    let flat = unsafe { std::slice::from_raw_parts(vectors, vector_count * dimensions) };
    let vecs: Vec<Vec<f32>> = flat.chunks(dimensions).map(|c| c.to_vec()).collect();

    let node_ids = db
        .inner
        .read()
        .batch_create_nodes(label_str, prop_str, vecs);
    let mut raw_ids: Vec<u64> = node_ids.iter().map(|id| id.as_u64()).collect();
    raw_ids.shrink_to_fit();
    let ptr = raw_ids.as_mut_ptr();
    let _count = raw_ids.len();
    std::mem::forget(raw_ids);

    // SAFETY: We checked out_ids is not null.
    unsafe { *out_ids = ptr };
    ObrainStatus::Ok
}

/// Free vector search result arrays.
#[cfg(feature = "vector-index")]
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_vector_results(ids: *mut u64, distances: *mut f32, count: usize) {
    if !ids.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(ids, count);
            drop(Box::from_raw(slice));
        }
    }
    if !distances.is_null() && count > 0 {
        // SAFETY: Reconstructs the Vec that was forgotten in vector_search.
        unsafe {
            let slice = std::ptr::slice_from_raw_parts_mut(distances, count);
            drop(Box::from_raw(slice));
        }
    }
}

// =========================================================================
// Statistics
// =========================================================================

/// Get the number of nodes.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_node_count(db: *mut ObrainDatabase) -> usize {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*db }.inner.read().node_count()
}

/// Get the number of edges.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_edge_count(db: *mut ObrainDatabase) -> usize {
    if db.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees valid pointer.
    unsafe { &*db }.inner.read().edge_count()
}

// =========================================================================
// Transactions
// =========================================================================

/// Begin a transaction with default isolation (snapshot isolation).
#[unsafe(no_mangle)]
pub extern "C" fn obrain_begin_transaction(db: *mut ObrainDatabase) -> *mut ObrainTransaction {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let mut session = db.inner.read().session();
    match session.begin_transaction() {
        Ok(()) => Box::into_raw(Box::new(ObrainTransaction {
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Begin a transaction with a specific isolation level.
/// Levels: 0 = ReadCommitted, 1 = SnapshotIsolation, 2 = Serializable.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_begin_transaction_with_isolation(
    db: *mut ObrainDatabase,
    isolation: i32,
) -> *mut ObrainTransaction {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };

    let level = match isolation {
        0 => obrain_engine::transaction::IsolationLevel::ReadCommitted,
        1 => obrain_engine::transaction::IsolationLevel::SnapshotIsolation,
        2 => obrain_engine::transaction::IsolationLevel::Serializable,
        _ => {
            set_last_error("Invalid isolation level (expected 0, 1, or 2)");
            return std::ptr::null_mut();
        }
    };

    let mut session = db.inner.read().session();
    match session.begin_transaction_with_isolation(level) {
        Ok(()) => Box::into_raw(Box::new(ObrainTransaction {
            session: parking_lot::Mutex::new(Some(session)),
            committed: false,
            rolled_back: false,
        })),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a query within a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_transaction_execute(
    tx: *mut ObrainTransaction,
    query: *const c_char,
) -> *mut ObrainResult {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &*tx };
    if tx.committed || tx.rolled_back {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    }
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = tx.session.lock();
    let Some(session) = guard.as_ref() else {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    };
    match session.execute(query_str) {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Execute a query with params within a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_transaction_execute_with_params(
    tx: *mut ObrainTransaction,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut ObrainResult {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &*tx };
    if tx.committed || tx.rolled_back {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    }
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let guard = tx.session.lock();
    let Some(session) = guard.as_ref() else {
        set_last_error("Transaction is no longer active");
        return std::ptr::null_mut();
    };
    let result = if let Some(params) = crate::types::parse_params(params_json) {
        session.execute_with_params(query_str, params)
    } else {
        session.execute(query_str)
    };
    match result {
        Ok(r) => build_result(&r),
        Err(e) => {
            set_error(&e);
            std::ptr::null_mut()
        }
    }
}

/// Commit a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_commit(tx: *mut ObrainTransaction) -> ObrainStatus {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return ObrainStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &mut *tx };
    if tx.committed {
        set_last_error("Transaction already committed");
        return ObrainStatus::ErrorTransaction;
    }
    if tx.rolled_back {
        set_last_error("Transaction already rolled back");
        return ObrainStatus::ErrorTransaction;
    }
    let mut guard = tx.session.lock();
    if let Some(ref mut session) = *guard {
        match session.commit() {
            Ok(()) => {
                tx.committed = true;
                ObrainStatus::Ok
            }
            Err(e) => set_error(&e),
        }
    } else {
        set_last_error("Transaction is no longer active");
        ObrainStatus::ErrorTransaction
    }
}

/// Rollback a transaction.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_rollback(tx: *mut ObrainTransaction) -> ObrainStatus {
    if tx.is_null() {
        set_last_error("Null transaction pointer");
        return ObrainStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees valid pointer.
    let tx = unsafe { &mut *tx };
    if tx.committed {
        set_last_error("Transaction already committed");
        return ObrainStatus::ErrorTransaction;
    }
    if tx.rolled_back {
        set_last_error("Transaction already rolled back");
        return ObrainStatus::ErrorTransaction;
    }
    let mut guard = tx.session.lock();
    if let Some(ref mut session) = *guard {
        match session.rollback() {
            Ok(()) => {
                tx.rolled_back = true;
                ObrainStatus::Ok
            }
            Err(e) => set_error(&e),
        }
    } else {
        set_last_error("Transaction is no longer active");
        ObrainStatus::ErrorTransaction
    }
}

/// Free a transaction handle.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_transaction(tx: *mut ObrainTransaction) {
    if !tx.is_null() {
        // SAFETY: We take ownership back. Drop impl handles auto-rollback.
        unsafe { drop(Box::from_raw(tx)) };
    }
}

// =========================================================================
// Admin
// =========================================================================

/// Clear all cached query plans.
///
/// Forces re-parsing and re-optimization on next execution.
/// Called automatically after DDL operations, but can be invoked manually.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_clear_plan_cache(db: *mut ObrainDatabase) -> ObrainStatus {
    let db = db_ref!(db);
    db.inner.read().clear_plan_cache();
    ObrainStatus::Ok
}

/// Get database info as a JSON string. Caller must free with `obrain_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_info(db: *mut ObrainDatabase) -> *mut c_char {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees valid pointer.
    let db = unsafe { &*db };
    let info = db.inner.read().info();
    let json = serde_json::json!({
        "node_count": info.node_count,
        "edge_count": info.edge_count,
        "is_persistent": info.is_persistent,
        "path": info.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
        "wal_enabled": info.wal_enabled,
        "version": info.version,
    });
    let s = serde_json::to_string(&json).unwrap_or_default();
    CString::new(s).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Save database to a path. Caller provides a path string.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_save(db: *mut ObrainDatabase, path: *const c_char) -> ObrainStatus {
    let db = db_ref!(db);
    let path_str = match str_from_ptr(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match db.inner.read().save(path_str) {
        Ok(()) => ObrainStatus::Ok,
        Err(e) => set_error(&e),
    }
}

/// Trigger a WAL checkpoint.
#[unsafe(no_mangle)]
pub extern "C" fn obrain_wal_checkpoint(db: *mut ObrainDatabase) -> ObrainStatus {
    let db = db_ref!(db);
    match db.inner.read().wal_checkpoint() {
        Ok(()) => ObrainStatus::Ok,
        Err(e) => set_error(&e),
    }
}

// =========================================================================
// Memory Management
// =========================================================================

/// Free a string returned by any `obrain_*` function that documents
/// "caller must free with `obrain_free_string`".
#[unsafe(no_mangle)]
pub extern "C" fn obrain_free_string(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: Reconstructs the CString that was turned into a raw pointer.
        unsafe { drop(CString::from_raw(s)) };
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_open_memory_and_close() {
        let db = obrain_open_memory();
        assert!(!db.is_null());
        assert_eq!(obrain_node_count(db), 0);
        assert_eq!(obrain_edge_count(db), 0);
        let status = obrain_close(db);
        assert_eq!(status, ObrainStatus::Ok);
        obrain_free_database(db);
    }

    #[test]
    fn test_create_node_and_get() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix","age":30}"#).unwrap();
        let id = obrain_create_node(db, labels.as_ptr(), props.as_ptr());
        assert_ne!(id, u64::MAX);
        assert_eq!(obrain_node_count(db), 1);

        let mut node_ptr: *mut ObrainNode = std::ptr::null_mut();
        let status = obrain_get_node(db, id, &raw mut node_ptr);
        assert_eq!(status, ObrainStatus::Ok);
        assert!(!node_ptr.is_null());

        // SAFETY: We just verified it's not null.
        let node = unsafe { &*node_ptr };
        assert_eq!(node.id, id);

        obrain_free_node(node_ptr);
        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_create_edge_and_get() {
        let db = obrain_open_memory();
        let labels_a = CString::new(r#"["Person"]"#).unwrap();
        let labels_b = CString::new(r#"["Person"]"#).unwrap();
        let id_a = obrain_create_node(db, labels_a.as_ptr(), std::ptr::null());
        let id_b = obrain_create_node(db, labels_b.as_ptr(), std::ptr::null());

        let edge_type = CString::new("KNOWS").unwrap();
        let edge_id = obrain_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());
        assert_ne!(edge_id, u64::MAX);
        assert_eq!(obrain_edge_count(db), 1);

        let mut edge_ptr: *mut ObrainEdge = std::ptr::null_mut();
        let status = obrain_get_edge(db, edge_id, &raw mut edge_ptr);
        assert_eq!(status, ObrainStatus::Ok);
        assert!(!edge_ptr.is_null());

        // SAFETY: We just verified it's not null.
        let edge = unsafe { &*edge_ptr };
        assert_eq!(edge.source_id, id_a);
        assert_eq!(edge.target_id, id_b);

        obrain_free_edge(edge_ptr);
        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_execute_query() {
        let db = obrain_open_memory();
        let create = CString::new("CREATE (:Person {name: 'Alix', age: 30})").unwrap();
        let result = obrain_execute(db, create.as_ptr());
        // CREATE returns a result (possibly empty rows).
        if !result.is_null() {
            obrain_free_result(result);
        }

        let query = CString::new("MATCH (p:Person) RETURN p.name, p.age").unwrap();
        let result = obrain_execute(db, query.as_ptr());
        assert!(!result.is_null());
        assert_eq!(obrain_result_row_count(result), 1);

        let json_ptr = obrain_result_json(result);
        assert!(!json_ptr.is_null());
        // SAFETY: Valid C string from our API.
        let json_str = unsafe { std::ffi::CStr::from_ptr(json_ptr) }
            .to_str()
            .unwrap();
        assert!(json_str.contains("Alix"));

        obrain_free_result(result);
        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_null_pointer_safety() {
        // All functions should handle null gracefully.
        let result = obrain_execute(std::ptr::null_mut(), std::ptr::null());
        assert!(result.is_null());
        let err = crate::error::obrain_last_error();
        assert!(!err.is_null());

        assert_eq!(obrain_node_count(std::ptr::null_mut()), 0);
        assert_eq!(obrain_edge_count(std::ptr::null_mut()), 0);
        assert_eq!(obrain_delete_node(std::ptr::null_mut(), 0), -1);
        assert_eq!(obrain_delete_edge(std::ptr::null_mut(), 0), -1);
    }

    #[test]
    fn test_transaction_commit() {
        let db = obrain_open_memory();
        let tx = obrain_begin_transaction(db);
        assert!(!tx.is_null());

        let query = CString::new("CREATE (:Tx {val: 1})").unwrap();
        let result = obrain_transaction_execute(tx, query.as_ptr());
        if !result.is_null() {
            obrain_free_result(result);
        }

        let status = obrain_commit(tx);
        assert_eq!(status, ObrainStatus::Ok);
        obrain_free_transaction(tx);

        // Verify committed data is visible.
        assert_eq!(obrain_node_count(db), 1);

        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_transaction_rollback() {
        let db = obrain_open_memory();

        // Create a baseline node.
        let labels = CString::new(r#"["Base"]"#).unwrap();
        obrain_create_node(db, labels.as_ptr(), std::ptr::null());
        assert_eq!(obrain_node_count(db), 1);

        let tx = obrain_begin_transaction(db);
        let query = CString::new("CREATE (:Rolled {val: 2})").unwrap();
        let result = obrain_transaction_execute(tx, query.as_ptr());
        if !result.is_null() {
            obrain_free_result(result);
        }

        let status = obrain_rollback(tx);
        assert_eq!(status, ObrainStatus::Ok);
        obrain_free_transaction(tx);

        // Rolled-back node should not be visible.
        assert_eq!(obrain_node_count(db), 1);

        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_node_property_crud() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let id = obrain_create_node(db, labels.as_ptr(), std::ptr::null());

        let key = CString::new("city").unwrap();
        let value = CString::new(r#""Berlin""#).unwrap();
        let status = obrain_set_node_property(db, id, key.as_ptr(), value.as_ptr());
        assert_eq!(status, ObrainStatus::Ok);

        let removed = obrain_remove_node_property(db, id, key.as_ptr());
        assert_eq!(removed, 1);

        obrain_close(db);
        obrain_free_database(db);
    }

    #[test]
    fn test_version() {
        let v = obrain_version();
        assert!(!v.is_null());
        // SAFETY: Static string, always valid.
        let version_str = unsafe { std::ffi::CStr::from_ptr(v) }.to_str().unwrap();
        assert!(!version_str.is_empty());
    }

    // ── Edge property CRUD ──

    #[test]
    fn test_edge_property_crud() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = obrain_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = obrain_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("R").unwrap();
        let eid = obrain_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());
        assert_ne!(eid, u64::MAX);

        // Set edge property
        let key = CString::new("weight").unwrap();
        let value = CString::new("1.5").unwrap();
        let status = obrain_set_edge_property(db, eid, key.as_ptr(), value.as_ptr());
        assert_eq!(status, ObrainStatus::Ok);

        // Remove edge property
        let removed = obrain_remove_edge_property(db, eid, key.as_ptr());
        assert_eq!(removed, 1);

        // Removing again returns 0
        let removed2 = obrain_remove_edge_property(db, eid, key.as_ptr());
        assert_eq!(removed2, 0);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Delete operations ──

    #[test]
    fn test_delete_node_and_edge() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = obrain_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = obrain_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("R").unwrap();
        let eid = obrain_create_edge(db, id_a, id_b, edge_type.as_ptr(), std::ptr::null());

        assert_eq!(obrain_node_count(db), 2);
        assert_eq!(obrain_edge_count(db), 1);

        // Delete edge
        assert_eq!(obrain_delete_edge(db, eid), 1);
        assert_eq!(obrain_edge_count(db), 0);
        // Second delete returns 0
        assert_eq!(obrain_delete_edge(db, eid), 0);

        // Delete node
        assert_eq!(obrain_delete_node(db, id_a), 1);
        assert_eq!(obrain_node_count(db), 1);
        assert_eq!(obrain_delete_node(db, id_a), 0);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Label operations ──

    #[test]
    fn test_label_operations() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let id = obrain_create_node(db, labels.as_ptr(), std::ptr::null());

        // Add label
        let label = CString::new("Employee").unwrap();
        assert_eq!(obrain_add_node_label(db, id, label.as_ptr()), 1);
        // Adding same label returns 0
        assert_eq!(obrain_add_node_label(db, id, label.as_ptr()), 0);

        // Get labels
        let labels_json = obrain_get_node_labels(db, id);
        assert!(!labels_json.is_null());
        // SAFETY: We just verified the pointer is not null.
        let labels_str = unsafe { std::ffi::CStr::from_ptr(labels_json) }
            .to_str()
            .unwrap();
        assert!(labels_str.contains("Person"));
        assert!(labels_str.contains("Employee"));
        obrain_free_string(labels_json);

        // Remove label
        assert_eq!(obrain_remove_node_label(db, id, label.as_ptr()), 1);
        assert_eq!(obrain_remove_node_label(db, id, label.as_ptr()), 0);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Execute with parameters ──

    #[test]
    fn test_execute_with_params() {
        let db = obrain_open_memory();

        // Create a node first
        let create = CString::new("CREATE (:Person {name: 'Alix', age: 30})").unwrap();
        let result = obrain_execute(db, create.as_ptr());
        if !result.is_null() {
            obrain_free_result(result);
        }

        // Query with parameters
        let query = CString::new("MATCH (n:Person) WHERE n.name = $name RETURN n.age").unwrap();
        let params = CString::new(r#"{"name":"Alix"}"#).unwrap();
        let result = obrain_execute_with_params(db, query.as_ptr(), params.as_ptr());
        assert!(!result.is_null());
        assert_eq!(obrain_result_row_count(result), 1);
        obrain_free_result(result);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Property index operations ──

    #[test]
    fn test_property_index_lifecycle() {
        let db = obrain_open_memory();
        let prop = CString::new("name").unwrap();

        // No index initially
        assert_eq!(obrain_has_property_index(db, prop.as_ptr()), 0);

        // Create index
        let status = obrain_create_property_index(db, prop.as_ptr());
        assert_eq!(status, ObrainStatus::Ok);
        assert_eq!(obrain_has_property_index(db, prop.as_ptr()), 1);

        // Create node with the property
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix"}"#).unwrap();
        obrain_create_node(db, labels.as_ptr(), props.as_ptr());

        // Find by property
        let value = CString::new(r#""Alix""#).unwrap();
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_count: usize = 0;
        let status = obrain_find_nodes_by_property(
            db,
            prop.as_ptr(),
            value.as_ptr(),
            &raw mut out_ids,
            &raw mut out_count,
        );
        assert_eq!(status, ObrainStatus::Ok);
        assert_eq!(out_count, 1);
        if !out_ids.is_null() {
            obrain_free_node_ids(out_ids, out_count);
        }

        // Drop index
        assert_eq!(obrain_drop_property_index(db, prop.as_ptr()), 1);
        assert_eq!(obrain_has_property_index(db, prop.as_ptr()), 0);
        assert_eq!(obrain_drop_property_index(db, prop.as_ptr()), 0);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Database info ──

    #[test]
    fn test_database_info() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["Person"]"#).unwrap();
        let props = CString::new(r#"{"name":"Alix"}"#).unwrap();
        obrain_create_node(db, labels.as_ptr(), props.as_ptr());

        let info = obrain_info(db);
        assert!(!info.is_null());
        // SAFETY: We just verified the pointer is not null.
        let info_str = unsafe { std::ffi::CStr::from_ptr(info) }.to_str().unwrap();
        assert!(info_str.contains("node_count"));
        obrain_free_string(info);

        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Result metadata ──

    #[test]
    fn test_result_metadata() {
        let db = obrain_open_memory();
        let create = CString::new("CREATE (:N {x: 1}), (:N {x: 2}), (:N {x: 3})").unwrap();
        let result = obrain_execute(db, create.as_ptr());
        if !result.is_null() {
            obrain_free_result(result);
        }

        let query = CString::new("MATCH (n:N) RETURN n.x").unwrap();
        let result = obrain_execute(db, query.as_ptr());
        assert!(!result.is_null());

        assert_eq!(obrain_result_row_count(result), 3);
        let time = obrain_result_execution_time_ms(result);
        assert!(time >= 0.0);

        obrain_free_result(result);
        obrain_close(db);
        obrain_free_database(db);
    }

    // ── Edge accessor functions ──

    #[test]
    fn test_edge_accessors() {
        let db = obrain_open_memory();
        let labels = CString::new(r#"["N"]"#).unwrap();
        let id_a = obrain_create_node(db, labels.as_ptr(), std::ptr::null());
        let id_b = obrain_create_node(db, labels.as_ptr(), std::ptr::null());

        let edge_type = CString::new("KNOWS").unwrap();
        let edge_props = CString::new(r#"{"since":2020}"#).unwrap();
        let eid = obrain_create_edge(db, id_a, id_b, edge_type.as_ptr(), edge_props.as_ptr());

        let mut edge_ptr: *mut ObrainEdge = std::ptr::null_mut();
        let status = obrain_get_edge(db, eid, &raw mut edge_ptr);
        assert_eq!(status, ObrainStatus::Ok);
        assert!(!edge_ptr.is_null());

        // Test all accessor functions
        assert_eq!(obrain_edge_id(edge_ptr), eid);
        assert_eq!(obrain_edge_source_id(edge_ptr), id_a);
        assert_eq!(obrain_edge_target_id(edge_ptr), id_b);

        let type_ptr = obrain_edge_type(edge_ptr);
        assert!(!type_ptr.is_null());
        // SAFETY: We just verified the pointer is not null.
        let type_str = unsafe { std::ffi::CStr::from_ptr(type_ptr) }
            .to_str()
            .unwrap();
        assert_eq!(type_str, "KNOWS");

        let props_ptr = obrain_edge_properties_json(edge_ptr);
        assert!(!props_ptr.is_null());
        // SAFETY: We just verified the pointer is not null.
        let props_str = unsafe { std::ffi::CStr::from_ptr(props_ptr) }
            .to_str()
            .unwrap();
        assert!(props_str.contains("2020"));

        obrain_free_edge(edge_ptr);
        obrain_close(db);
        obrain_free_database(db);
    }
}
