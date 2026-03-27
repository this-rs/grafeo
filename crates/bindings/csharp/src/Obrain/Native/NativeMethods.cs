// P/Invoke declarations for all obrain-c exported functions.
// Uses .NET 8 LibraryImport source generation for AOT-safe marshalling.

using System.Runtime.InteropServices;

namespace Obrain.Native;

internal static partial class NativeMethods
{
    private const string LibName = "obrain_c";

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// <summary>Create a new in-memory database. Returns null on error.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_open_memory();

    /// <summary>Open or create a persistent database at path. Returns null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_open(string path);

    /// <summary>Close the database, flushing pending writes. Returns status code.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_close(nint db);

    /// <summary>Free a database handle. Must be called after obrain_close.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_database(nint db);

    /// <summary>Returns the library version string. The pointer is static, do NOT free.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_version();

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query. Returns result pointer, or null on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute(nint db, string query);

    /// <summary>Execute a GQL query with JSON-encoded parameters.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_with_params(nint db, string query, string paramsJson);

    /// <summary>Execute a Cypher query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_cypher(nint db, string query);

    /// <summary>Execute a Gremlin query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_gremlin(nint db, string query);

    /// <summary>Execute a GraphQL query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_graphql(nint db, string query);

    /// <summary>Execute a SPARQL query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_sparql(nint db, string query);

    /// <summary>Execute a SQL/PGQ query.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_execute_sql(nint db, string query);

    // =========================================================================
    // Result Access
    // =========================================================================

    /// <summary>Get the JSON string from a result. Valid until obrain_free_result.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_result_json(nint result);

    /// <summary>Get the row count from a result.</summary>
    [LibraryImport(LibName)]
    internal static partial nuint obrain_result_row_count(nint result);

    /// <summary>Get the execution time in milliseconds.</summary>
    [LibraryImport(LibName)]
    internal static partial double obrain_result_execution_time_ms(nint result);

    /// <summary>Get the number of rows scanned.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong obrain_result_rows_scanned(nint result);

    /// <summary>Free a query result.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_result(nint result);

    // =========================================================================
    // Node CRUD
    // =========================================================================

    /// <summary>Create a node with labels (JSON array) and properties (JSON object). Returns node ID or u64.MaxValue on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong obrain_create_node(nint db, string labelsJson, string? propertiesJson);

    /// <summary>Get a node by ID. Writes into out pointer. Returns status.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_get_node(nint db, ulong id, out nint nodeOut);

    /// <summary>Delete a node by ID. Returns 1 if deleted, 0 if not found, -1 on error.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_delete_node(nint db, ulong id);

    /// <summary>Set a property on a node.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_set_node_property(nint db, ulong id, string key, string valueJson);

    /// <summary>Remove a property from a node. Returns 1 if removed, 0 if not found.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_remove_node_property(nint db, ulong id, string key);

    /// <summary>Add a label to a node. Returns 1 if added, 0 if already present.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_add_node_label(nint db, ulong id, string label);

    /// <summary>Remove a label from a node. Returns 1 if removed, 0 if not present.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_remove_node_label(nint db, ulong id, string label);

    /// <summary>Get labels for a node as JSON array. Caller must free with obrain_free_string.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_get_node_labels(nint db, ulong id);

    /// <summary>Free a ObrainNode.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_node(nint node);

    /// <summary>Access node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong obrain_node_id(nint node);

    /// <summary>Access labels JSON. Valid until obrain_free_node.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_node_labels_json(nint node);

    /// <summary>Access properties JSON. Valid until obrain_free_node.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_node_properties_json(nint node);

    // =========================================================================
    // Edge CRUD
    // =========================================================================

    /// <summary>Create an edge. Returns edge ID or u64.MaxValue on error.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial ulong obrain_create_edge(nint db, ulong sourceId, ulong targetId, string edgeType, string? propertiesJson);

    /// <summary>Get an edge by ID. Writes into out pointer. Returns status.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_get_edge(nint db, ulong id, out nint edgeOut);

    /// <summary>Delete an edge by ID. Returns 1 if deleted, 0 if not found, -1 on error.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_delete_edge(nint db, ulong id);

    /// <summary>Set a property on an edge.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_set_edge_property(nint db, ulong id, string key, string valueJson);

    /// <summary>Remove a property from an edge. Returns 1 if removed, 0 if not found.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_remove_edge_property(nint db, ulong id, string key);

    /// <summary>Free a ObrainEdge.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_edge(nint edge);

    /// <summary>Access edge ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong obrain_edge_id(nint edge);

    /// <summary>Access source node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong obrain_edge_source_id(nint edge);

    /// <summary>Access target node ID.</summary>
    [LibraryImport(LibName)]
    internal static partial ulong obrain_edge_target_id(nint edge);

    /// <summary>Access edge type string. Valid until obrain_free_edge.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_edge_type(nint edge);

    /// <summary>Access edge properties JSON. Valid until obrain_free_edge.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_edge_properties_json(nint edge);

    // =========================================================================
    // Indexes
    // =========================================================================

    /// <summary>Create a property index on a label and property.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_create_property_index(nint db, string label, string property);

    /// <summary>Drop a property index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_drop_property_index(nint db, string label, string property);

    /// <summary>Check if a property index exists. Returns 1 if exists, 0 if not.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_has_property_index(nint db, string label, string property);

    /// <summary>Find nodes by property value. Writes IDs and count to out pointers.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_find_nodes_by_property(
        nint db, string label, string property, string valueJson,
        out nint idsOut, out nuint countOut);

    /// <summary>Free node IDs returned by obrain_find_nodes_by_property.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_node_ids(nint ids, nuint count);

    // =========================================================================
    // Vector Search
    // =========================================================================

    /// <summary>Create a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_create_vector_index(
        nint db, string label, string property,
        uint dimensions, uint m, uint efConstruction, uint ef);

    /// <summary>Drop a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_drop_vector_index(nint db, string label, string property);

    /// <summary>Rebuild a vector index.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_rebuild_vector_index(nint db, string label, string property);

    /// <summary>Perform a vector search.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int obrain_vector_search(
        nint db, string label, string property,
        float* query, nuint queryLen, nuint k, uint ef,
        out nint idsOut, out nint distsOut, out nuint countOut);

    /// <summary>Perform an MMR search.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static unsafe partial int obrain_mmr_search(
        nint db, string label, string property,
        float* query, nuint queryLen, nuint k,
        int fetchK, float lambda, int ef,
        out nint idsOut, out nint distsOut, out nuint countOut);

    /// <summary>Free vector search results.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_vector_results(nint ids, nint dists, nuint count);

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /// <summary>Batch create nodes. Returns number of nodes created.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial long obrain_batch_create_nodes(nint db, string batchJson);

    // =========================================================================
    // Admin
    // =========================================================================

    /// <summary>Get the number of nodes.</summary>
    [LibraryImport(LibName)]
    internal static partial nuint obrain_node_count(nint db);

    /// <summary>Get the number of edges.</summary>
    [LibraryImport(LibName)]
    internal static partial nuint obrain_edge_count(nint db);

    /// <summary>Get database info as JSON. Caller must free with obrain_free_string.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_info(nint db);

    /// <summary>Save database to path.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int obrain_save(nint db, string path);

    /// <summary>Checkpoint the WAL.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_wal_checkpoint(nint db);

    // =========================================================================
    // Transactions
    // =========================================================================

    /// <summary>Begin a new transaction. Returns null on error.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_begin_transaction(nint db);

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_begin_transaction_with_isolation(nint db, string isolationLevel);

    /// <summary>Execute a query within a transaction.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_transaction_execute(nint tx, string query);

    /// <summary>Execute a query with parameters within a transaction.</summary>
    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint obrain_transaction_execute_with_params(nint tx, string query, string paramsJson);

    /// <summary>Commit a transaction.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_commit(nint tx);

    /// <summary>Rollback a transaction.</summary>
    [LibraryImport(LibName)]
    internal static partial int obrain_rollback(nint tx);

    /// <summary>Free a transaction handle.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_transaction(nint tx);

    // =========================================================================
    // Error Handling
    // =========================================================================

    /// <summary>Get the last error message. Valid until next FFI call on this thread. Do NOT free.</summary>
    [LibraryImport(LibName)]
    internal static partial nint obrain_last_error();

    /// <summary>Clear the last error.</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_clear_error();

    /// <summary>Free a string returned by the API (info, labels, etc.).</summary>
    [LibraryImport(LibName)]
    internal static partial void obrain_free_string(nint str);
}
