// Primary database handle for the Grafeo graph database.

using System.Runtime.InteropServices;

using Grafeo.Native;

namespace Grafeo;

/// <summary>
/// Primary handle to a Grafeo graph database.
/// Thread-safe: the underlying engine uses <c>Arc&lt;RwLock&gt;</c>.
/// Implements <see cref="IDisposable"/> and <see cref="IAsyncDisposable"/>
/// for deterministic cleanup via <c>using</c>/<c>await using</c>.
/// </summary>
public sealed class GrafeoDB : IDisposable, IAsyncDisposable
{
    private readonly DatabaseHandle _handle;
    private volatile bool _disposed;

    private GrafeoDB(DatabaseHandle handle) => _handle = handle;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// <summary>Create a new in-memory database.</summary>
    public static GrafeoDB Memory()
    {
        var ptr = NativeMethods.grafeo_open_memory();
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <summary>Open or create a persistent database at <paramref name="path"/>.</summary>
    public static GrafeoDB Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var ptr = NativeMethods.grafeo_open(path);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();

        var handle = new DatabaseHandle();
        Marshal.InitHandle(handle, ptr);
        return new GrafeoDB(handle);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _handle.Dispose();
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    // =========================================================================
    // Query Execution
    // =========================================================================

    /// <summary>Execute a GQL query synchronously.</summary>
    public QueryResult Execute(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteAsync(string query, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_execute(h, query);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Execute a GQL query with parameters.</summary>
    public QueryResult ExecuteWithParams(string query, Dictionary<string, object?> parameters)
    {
        ThrowIfDisposed();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var resultPtr = NativeMethods.grafeo_execute_with_params(Handle, query, paramsJson);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GQL query with parameters on the thread pool.</summary>
    public Task<QueryResult> ExecuteWithParamsAsync(
        string query,
        Dictionary<string, object?> parameters,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var paramsJson = ValueConverter.EncodeParams(parameters);
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = NativeMethods.grafeo_execute_with_params(h, query, paramsJson);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Execute a Cypher query.</summary>
    public QueryResult ExecuteCypher(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_cypher(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a Cypher query on the thread pool.</summary>
    public Task<QueryResult> ExecuteCypherAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_cypher, query, ct);

    /// <summary>Execute a SPARQL query.</summary>
    public QueryResult ExecuteSparql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_sparql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a SPARQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSparqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_sparql, query, ct);

    /// <summary>Execute a Gremlin query.</summary>
    public QueryResult ExecuteGremlin(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_gremlin(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a Gremlin query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGremlinAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_gremlin, query, ct);

    /// <summary>Execute a GraphQL query.</summary>
    public QueryResult ExecuteGraphql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_graphql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a GraphQL query on the thread pool.</summary>
    public Task<QueryResult> ExecuteGraphqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_graphql, query, ct);

    /// <summary>Execute a SQL/PGQ query.</summary>
    public QueryResult ExecuteSql(string query)
    {
        ThrowIfDisposed();
        var resultPtr = NativeMethods.grafeo_execute_sql(Handle, query);
        if (resultPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Query);
        return BuildResult(resultPtr);
    }

    /// <summary>Execute a SQL/PGQ query on the thread pool.</summary>
    public Task<QueryResult> ExecuteSqlAsync(string query, CancellationToken ct = default)
        => ExecuteLanguageAsync(NativeMethods.grafeo_execute_sql, query, ct);

    // =========================================================================
    // Transactions
    // =========================================================================

    /// <summary>Begin a new ACID transaction with the default isolation level.</summary>
    public Transaction BeginTransaction()
    {
        ThrowIfDisposed();
        var txPtr = NativeMethods.grafeo_begin_transaction(Handle);
        if (txPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        return new Transaction(txPtr);
    }

    /// <summary>Begin a transaction with a specific isolation level.</summary>
    /// <param name="isolationLevel">E.g. "read_committed", "repeatable_read", "serializable".</param>
    public Transaction BeginTransaction(string isolationLevel)
    {
        ThrowIfDisposed();
        var txPtr = NativeMethods.grafeo_begin_transaction_with_isolation(Handle, isolationLevel);
        if (txPtr == nint.Zero)
            throw GrafeoException.FromLastError(GrafeoStatus.Transaction);
        return new Transaction(txPtr);
    }

    // =========================================================================
    // Node CRUD
    // =========================================================================

    /// <summary>Create a node with labels and optional properties. Returns the new node ID.</summary>
    public long CreateNode(IEnumerable<string> labels, Dictionary<string, object?>? properties = null)
    {
        ThrowIfDisposed();
        var labelsJson = System.Text.Json.JsonSerializer.Serialize(labels);
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_node(Handle, labelsJson, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get a node by ID. Returns null if not found.</summary>
    public Node? GetNode(long id)
    {
        ThrowIfDisposed();
        var status = NativeMethods.grafeo_get_node(Handle, (ulong)id, out var nodePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadNode(nodePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_node(nodePtr);
        }
    }

    /// <summary>Delete a node by ID. Returns true if deleted.</summary>
    public bool DeleteNode(long id)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_delete_node(Handle, (ulong)id) == 1;
    }

    /// <summary>Set a property on a node.</summary>
    public void SetNodeProperty(long id, string key, object? value)
    {
        ThrowIfDisposed();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_node_property(Handle, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from a node. Returns true if removed.</summary>
    public bool RemoveNodeProperty(long id, string key)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_node_property(Handle, (ulong)id, key) == 1;
    }

    /// <summary>Add a label to a node. Returns true if added.</summary>
    public bool AddNodeLabel(long id, string label)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_add_node_label(Handle, (ulong)id, label) == 1;
    }

    /// <summary>Remove a label from a node. Returns true if removed.</summary>
    public bool RemoveNodeLabel(long id, string label)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_node_label(Handle, (ulong)id, label) == 1;
    }

    // =========================================================================
    // Edge CRUD
    // =========================================================================

    /// <summary>Create an edge between two nodes. Returns the new edge ID.</summary>
    public long CreateEdge(
        long sourceId,
        long targetId,
        string edgeType,
        Dictionary<string, object?>? properties = null)
    {
        ThrowIfDisposed();
        var propsJson = properties is not null ? ValueConverter.EncodeParams(properties) : null;
        var id = NativeMethods.grafeo_create_edge(
            Handle, (ulong)sourceId, (ulong)targetId, edgeType, propsJson);
        if (id == ulong.MaxValue)
            throw GrafeoException.FromLastError();
        return (long)id;
    }

    /// <summary>Get an edge by ID. Returns null if not found.</summary>
    public Edge? GetEdge(long id)
    {
        ThrowIfDisposed();
        var status = NativeMethods.grafeo_get_edge(Handle, (ulong)id, out var edgePtr);
        if (status != (int)GrafeoStatus.Ok)
            return null;
        try
        {
            return ReadEdge(edgePtr);
        }
        finally
        {
            NativeMethods.grafeo_free_edge(edgePtr);
        }
    }

    /// <summary>Delete an edge by ID. Returns true if deleted.</summary>
    public bool DeleteEdge(long id)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_delete_edge(Handle, (ulong)id) == 1;
    }

    /// <summary>Set a property on an edge.</summary>
    public void SetEdgeProperty(long id, string key, object? value)
    {
        ThrowIfDisposed();
        var valueJson = ValueConverter.EncodeValue(value);
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_set_edge_property(Handle, (ulong)id, key, valueJson));
    }

    /// <summary>Remove a property from an edge. Returns true if removed.</summary>
    public bool RemoveEdgeProperty(long id, string key)
    {
        ThrowIfDisposed();
        return NativeMethods.grafeo_remove_edge_property(Handle, (ulong)id, key) == 1;
    }

    // =========================================================================
    // Admin
    // =========================================================================

    /// <summary>Number of nodes in the database.</summary>
    public long NodeCount
    {
        get
        {
            ThrowIfDisposed();
            return (long)NativeMethods.grafeo_node_count(Handle);
        }
    }

    /// <summary>Number of edges in the database.</summary>
    public long EdgeCount
    {
        get
        {
            ThrowIfDisposed();
            return (long)NativeMethods.grafeo_edge_count(Handle);
        }
    }

    /// <summary>Get database info as a dictionary (version, node count, edge count, etc.).</summary>
    public IReadOnlyDictionary<string, object?> Info()
    {
        ThrowIfDisposed();
        var ptr = NativeMethods.grafeo_info(Handle);
        if (ptr == nint.Zero)
            throw GrafeoException.FromLastError();
        try
        {
            var json = Marshal.PtrToStringUTF8(ptr)!;
            return ValueConverter.ParseObject(json);
        }
        finally
        {
            NativeMethods.grafeo_free_string(ptr);
        }
    }

    /// <summary>Get the Grafeo library version string.</summary>
    public static string Version
    {
        get
        {
            var ptr = NativeMethods.grafeo_version();
            return Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        }
    }

    /// <summary>Save the database to a file path.</summary>
    public void Save(string path)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(NativeMethods.grafeo_save(Handle, path));
    }

    // =========================================================================
    // Vector Search
    // =========================================================================

    /// <summary>Drop a vector index on the given label and property.</summary>
    public void DropVectorIndex(string label, string property)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_drop_vector_index(Handle, label, property));
    }

    /// <summary>Rebuild a vector index on the given label and property.</summary>
    public void RebuildVectorIndex(string label, string property)
    {
        ThrowIfDisposed();
        GrafeoException.ThrowIfFailed(
            NativeMethods.grafeo_rebuild_vector_index(Handle, label, property));
    }

    /// <summary>Perform a vector similarity search.</summary>
    /// <returns>List of (nodeId, distance) results ordered by distance.</returns>
    public IReadOnlyList<VectorResult> VectorSearch(
        string label, string property, float[] query, int k, uint ef = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_vector_search(
                    Handle, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    /// <summary>Perform a Maximal Marginal Relevance (MMR) search.</summary>
    public IReadOnlyList<VectorResult> MmrSearch(
        string label, string property, float[] query,
        int k, int fetchK, float lambda, int ef = 0)
    {
        ThrowIfDisposed();
        unsafe
        {
            fixed (float* queryPtr = query)
            {
                var status = NativeMethods.grafeo_mmr_search(
                    Handle, label, property,
                    queryPtr, (nuint)query.Length, (nuint)k,
                    fetchK, lambda, ef,
                    out var idsPtr, out var distsPtr, out var count);

                GrafeoException.ThrowIfFailed(status);
                return ReadVectorResults(idsPtr, distsPtr, count);
            }
        }
    }

    // =========================================================================
    // Internals
    // =========================================================================

    private Task<QueryResult> ExecuteLanguageAsync(
        Func<nint, string, nint> nativeMethod,
        string query,
        CancellationToken ct)
    {
        ThrowIfDisposed();
        var h = Handle;
        return Task.Run(() =>
        {
            ct.ThrowIfCancellationRequested();
            var resultPtr = nativeMethod(h, query);
            if (resultPtr == nint.Zero)
                throw GrafeoException.FromLastError(GrafeoStatus.Query);
            return BuildResult(resultPtr);
        }, ct);
    }

    /// <summary>Get the raw handle, checking for disposal first.</summary>
    private nint Handle
    {
        get
        {
            ThrowIfDisposed();
            return _handle.DangerousGetHandle();
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>Parse a native result pointer into a QueryResult, then free the native result.</summary>
    private static QueryResult BuildResult(nint resultPtr)
    {
        try
        {
            var jsonPtr = NativeMethods.grafeo_result_json(resultPtr);
            var json = Marshal.PtrToStringUTF8(jsonPtr) ?? "[]";
            var executionTimeMs = NativeMethods.grafeo_result_execution_time_ms(resultPtr);
            var rowsScanned = (long)NativeMethods.grafeo_result_rows_scanned(resultPtr);

            var rows = ValueConverter.ParseRows(json);
            var columns = ValueConverter.ExtractColumns(rows);
            var (nodes, edges) = ValueConverter.ExtractEntities(rows);

            return new QueryResult(columns, rows, nodes, edges, executionTimeMs, rowsScanned);
        }
        finally
        {
            NativeMethods.grafeo_free_result(resultPtr);
        }
    }

    /// <summary>Read a node from a native GrafeoNode pointer.</summary>
    private static Node ReadNode(nint nodePtr)
    {
        var id = (long)NativeMethods.grafeo_node_id(nodePtr);
        var labelsJsonPtr = NativeMethods.grafeo_node_labels_json(nodePtr);
        var propsJsonPtr = NativeMethods.grafeo_node_properties_json(nodePtr);

        var labelsJson = Marshal.PtrToStringUTF8(labelsJsonPtr) ?? "[]";
        var propsJson = Marshal.PtrToStringUTF8(propsJsonPtr) ?? "{}";

        var labels = ValueConverter.ParseStringArray(labelsJson);
        var properties = ValueConverter.ParseObject(propsJson);
        return new Node(id, labels, properties);
    }

    /// <summary>Read an edge from a native GrafeoEdge pointer.</summary>
    private static Edge ReadEdge(nint edgePtr)
    {
        var id = (long)NativeMethods.grafeo_edge_id(edgePtr);
        var sourceId = (long)NativeMethods.grafeo_edge_source_id(edgePtr);
        var targetId = (long)NativeMethods.grafeo_edge_target_id(edgePtr);
        var typePtr = NativeMethods.grafeo_edge_type(edgePtr);
        var propsPtr = NativeMethods.grafeo_edge_properties_json(edgePtr);

        var edgeType = Marshal.PtrToStringUTF8(typePtr) ?? "";
        var propsJson = Marshal.PtrToStringUTF8(propsPtr) ?? "{}";
        var properties = ValueConverter.ParseObject(propsJson);

        return new Edge(id, edgeType, sourceId, targetId, properties);
    }

    /// <summary>Read vector search results from native pointers, then free them.</summary>
    private static IReadOnlyList<VectorResult> ReadVectorResults(
        nint idsPtr, nint distsPtr, nuint count)
    {
        if (count == 0)
            return Array.Empty<VectorResult>();

        var results = new VectorResult[(int)count];
        unsafe
        {
            var ids = (ulong*)idsPtr;
            var dists = (float*)distsPtr;
            for (var i = 0; i < (int)count; i++)
            {
                results[i] = new VectorResult((long)ids[i], dists[i]);
            }
        }
        NativeMethods.grafeo_free_vector_results(idsPtr, distsPtr, count);
        return results;
    }
}
