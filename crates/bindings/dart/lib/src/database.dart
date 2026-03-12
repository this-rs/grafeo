/// The main GrafeoDB database class.
///
/// Wraps the grafeo-c shared library via FFI. Uses [NativeFinalizer] to
/// prevent leaks if [close] is not called explicitly.
library;

import 'dart:convert';
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'ffi/loader.dart';
import 'transaction.dart';
import 'types.dart';
import 'value.dart';

/// A Grafeo graph database instance.
///
/// Create with [GrafeoDB.memory] (in-memory) or [GrafeoDB.open] (persistent).
/// Always call [close] when done, or rely on [NativeFinalizer] as a safety net.
class GrafeoDB implements Finalizable {
  final GrafeoBindings _bindings;
  Pointer<Void> _handle;
  bool _closed = false;

  static NativeFinalizer? _finalizer;

  GrafeoDB._(this._handle, this._bindings) {
    // Lazily create a finalizer that calls grafeo_free_database on the handle.
    // The Rust Drop impl for the inner Arc<RwLock<GrafeoDB>> flushes writes.
    _finalizer ??= NativeFinalizer(
      _bindings.library
          .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'grafeo_free_database',
          ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /// Create a new in-memory database.
  static GrafeoDB memory({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoOpenMemory();
    if (ptr == nullptr) throwLastError(bindings);
    return GrafeoDB._(ptr, bindings);
  }

  /// Open a persistent database at [path].
  static GrafeoDB open(String path, {String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final ptr = bindings.grafeoOpen(pathPtr);
      if (ptr == nullptr) throwLastError(bindings);
      return GrafeoDB._(ptr, bindings);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Close the database, flushing all writes.
  ///
  /// Safe to call multiple times. After close, all other methods throw.
  void close() {
    if (_closed) return;
    _closed = true;
    _finalizer!.detach(this);
    final status = _bindings.grafeoClose(_handle);
    _bindings.grafeoFreeDatabase(_handle);
    _handle = nullptr;
    if (status != GrafeoStatus.ok.code) {
      throw classifyError(status, lastError(_bindings));
    }
  }

  /// Returns the grafeo-c library version.
  ///
  /// The C function returns a pointer to a static string that must NOT
  /// be freed (it lives in the binary's read-only data segment).
  static String version({String? libraryPath}) {
    final lib = loadNativeLibrary(libraryPath);
    final bindings = GrafeoBindings(lib);
    final ptr = bindings.grafeoVersion();
    return ptr.toDartString();
  }

  void _checkOpen() {
    if (_closed) {
      throw DatabaseException('Database is closed', GrafeoStatus.database);
    }
  }

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a GQL query and return a [QueryResult].
  QueryResult execute(String query) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoExecute(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  /// Execute a GQL query with parameters.
  ///
  /// Parameters are JSON-encoded using the grafeo-bindings-common wire format.
  /// Temporal types (DateTime, Duration) are automatically encoded as
  /// `$timestamp_us`, `$duration`, etc.
  QueryResult executeWithParams(
    String query,
    Map<String, dynamic> params,
  ) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    final paramsJson = encodeParams(params);
    final paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.grafeoExecuteWithParams(
        _handle,
        queryPtr,
        paramsPtr,
      );
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
      malloc.free(paramsPtr);
    }
  }

  /// Execute a Cypher query (requires `cypher` feature in grafeo-c).
  QueryResult executeCypher(String query) => _executeLanguage(
    query,
    _bindings.grafeoExecuteCypher,
  );

  /// Execute a Gremlin query (requires `gremlin` feature in grafeo-c).
  QueryResult executeGremlin(String query) => _executeLanguage(
    query,
    _bindings.grafeoExecuteGremlin,
  );

  /// Execute a GraphQL query (requires `graphql` feature in grafeo-c).
  QueryResult executeGraphql(String query) => _executeLanguage(
    query,
    _bindings.grafeoExecuteGraphql,
  );

  /// Execute a SPARQL query (requires `sparql` feature in grafeo-c).
  QueryResult executeSparql(String query) => _executeLanguage(
    query,
    _bindings.grafeoExecuteSparql,
  );

  QueryResult _executeLanguage(
    String query,
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>) fn,
  ) {
    _checkOpen();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = fn(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  // ===========================================================================
  // Statistics and info
  // ===========================================================================

  /// Get the number of nodes in the database (O(1), synchronous).
  int get nodeCount {
    _checkOpen();
    return _bindings.grafeoNodeCount(_handle);
  }

  /// Get the number of edges in the database (O(1), synchronous).
  int get edgeCount {
    _checkOpen();
    return _bindings.grafeoEdgeCount(_handle);
  }

  /// Get database info as a parsed JSON map.
  ///
  /// The C function allocates a string that must be freed with
  /// grafeo_free_string.
  Map<String, dynamic> info() {
    _checkOpen();
    final ptr = _bindings.grafeoInfo(_handle);
    if (ptr == nullptr) throwLastError(_bindings);
    try {
      return parseObject(ptr.toDartString());
    } finally {
      _bindings.grafeoFreeString(ptr);
    }
  }

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /// Begin a new transaction with the default isolation level.
  Transaction beginTransaction() {
    _checkOpen();
    final txPtr = _bindings.grafeoBeginTransaction(_handle);
    if (txPtr == nullptr) throwLastError(_bindings);
    return Transaction(txPtr, _bindings);
  }

  /// Begin a transaction with a specific [isolationLevel].
  Transaction beginTransactionWithIsolation(IsolationLevel isolationLevel) {
    _checkOpen();
    final txPtr = _bindings.grafeoBeginTransactionWithIsolation(
      _handle,
      isolationLevel.code,
    );
    if (txPtr == nullptr) throwLastError(_bindings);
    return Transaction(txPtr, _bindings);
  }

  // ===========================================================================
  // Node CRUD
  // ===========================================================================

  /// Create a node with [labels] and [properties]. Returns the new node ID.
  int createNode(List<String> labels, Map<String, dynamic> properties) {
    _checkOpen();
    final labelsJson = jsonEncode(labels);
    final propsJson = jsonEncode(
      properties.map((k, v) => MapEntry(k, _encodeValue(v))),
    );
    final labelsPtr = labelsJson.toNativeUtf8(allocator: malloc);
    final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
    try {
      final id = _bindings.grafeoCreateNode(_handle, labelsPtr, propsPtr);
      if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
      return id;
    } finally {
      malloc.free(labelsPtr);
      malloc.free(propsPtr);
    }
  }

  /// Get a node by [id]. Returns a [Node] or throws on error.
  Node getNode(int id) {
    _checkOpen();
    final outPtr = malloc<Pointer<Void>>();
    try {
      final status = _bindings.grafeoGetNode(_handle, id, outPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final nodePtr = outPtr.value;
      try {
        final nodeId = _bindings.grafeoNodeId(nodePtr);
        final labelsJson =
            _bindings.grafeoNodeLabelsJson(nodePtr).toDartString();
        final propsJson =
            _bindings.grafeoNodePropertiesJson(nodePtr).toDartString();
        return Node(
          nodeId,
          parseStringArray(labelsJson),
          parseObject(propsJson),
        );
      } finally {
        _bindings.grafeoFreeNode(nodePtr);
      }
    } finally {
      malloc.free(outPtr);
    }
  }

  /// Delete a node by [id]. Returns true on success.
  bool deleteNode(int id) {
    _checkOpen();
    final result = _bindings.grafeoDeleteNode(_handle, id);
    if (result < 0) throwLastError(_bindings);
    return result == 1;
  }

  /// Set a property on node [id].
  void setNodeProperty(int id, String key, dynamic value) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    final valueJson = encodeValue(value);
    final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSetNodeProperty(
        _handle,
        id,
        keyPtr,
        valuePtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(keyPtr);
      malloc.free(valuePtr);
    }
  }

  /// Remove a property from node [id].
  void removeNodeProperty(int id, String key) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveNodeProperty(_handle, id, keyPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(keyPtr);
    }
  }

  /// Add a label to node [id].
  void addNodeLabel(int id, String label) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoAddNodeLabel(_handle, id, labelPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(labelPtr);
    }
  }

  /// Remove a label from node [id].
  void removeNodeLabel(int id, String label) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveNodeLabel(_handle, id, labelPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(labelPtr);
    }
  }

  // ===========================================================================
  // Edge CRUD
  // ===========================================================================

  /// Create an edge from [sourceId] to [targetId] with the given [type] and
  /// [properties]. Returns the new edge ID.
  int createEdge(
    int sourceId,
    int targetId,
    String type,
    Map<String, dynamic> properties,
  ) {
    _checkOpen();
    final typePtr = type.toNativeUtf8(allocator: malloc);
    final propsJson = jsonEncode(
      properties.map((k, v) => MapEntry(k, _encodeValue(v))),
    );
    final propsPtr = propsJson.toNativeUtf8(allocator: malloc);
    try {
      final id = _bindings.grafeoCreateEdge(
        _handle,
        sourceId,
        targetId,
        typePtr,
        propsPtr,
      );
      if (id == -1) throwLastError(_bindings); // C returns u64::MAX on error
      return id;
    } finally {
      malloc.free(typePtr);
      malloc.free(propsPtr);
    }
  }

  /// Get an edge by [id]. Returns an [Edge] or throws on error.
  Edge getEdge(int id) {
    _checkOpen();
    final outPtr = malloc<Pointer<Void>>();
    try {
      final status = _bindings.grafeoGetEdge(_handle, id, outPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
      final edgePtr = outPtr.value;
      try {
        final edgeId = _bindings.grafeoEdgeId(edgePtr);
        final sourceId = _bindings.grafeoEdgeSourceId(edgePtr);
        final targetId = _bindings.grafeoEdgeTargetId(edgePtr);
        final edgeType = _bindings.grafeoEdgeType(edgePtr).toDartString();
        final propsJson =
            _bindings.grafeoEdgePropertiesJson(edgePtr).toDartString();
        return Edge(
          edgeId,
          edgeType,
          sourceId,
          targetId,
          parseObject(propsJson),
        );
      } finally {
        _bindings.grafeoFreeEdge(edgePtr);
      }
    } finally {
      malloc.free(outPtr);
    }
  }

  /// Delete an edge by [id]. Returns true on success.
  bool deleteEdge(int id) {
    _checkOpen();
    final result = _bindings.grafeoDeleteEdge(_handle, id);
    if (result < 0) throwLastError(_bindings);
    return result == 1;
  }

  /// Set a property on edge [id].
  void setEdgeProperty(int id, String key, dynamic value) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    final valueJson = encodeValue(value);
    final valuePtr = valueJson.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSetEdgeProperty(
        _handle,
        id,
        keyPtr,
        valuePtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(keyPtr);
      malloc.free(valuePtr);
    }
  }

  /// Remove a property from edge [id].
  void removeEdgeProperty(int id, String key) {
    _checkOpen();
    final keyPtr = key.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoRemoveEdgeProperty(_handle, id, keyPtr);
      if (result < 0) throwLastError(_bindings);
    } finally {
      malloc.free(keyPtr);
    }
  }

  // ===========================================================================
  // Vector operations
  // ===========================================================================

  /// Drop a vector index. Returns true if the index existed.
  bool dropVectorIndex(String label, String property) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    try {
      final result = _bindings.grafeoDropVectorIndex(
        _handle,
        labelPtr,
        propertyPtr,
      );
      return result != 0;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
    }
  }

  /// Rebuild a vector index.
  void rebuildVectorIndex(String label, String property) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoRebuildVectorIndex(
        _handle,
        labelPtr,
        propertyPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
    }
  }

  /// Perform an MMR (Maximal Marginal Relevance) vector search.
  List<VectorResult> mmrSearch(
    String label,
    String property,
    List<double> query, {
    required int k,
    required int fetchK,
    required double lambda,
    required int ef,
  }) {
    _checkOpen();
    final labelPtr = label.toNativeUtf8(allocator: malloc);
    final propertyPtr = property.toNativeUtf8(allocator: malloc);
    final queryPtr = malloc<Float>(query.length);
    for (var i = 0; i < query.length; i++) {
      queryPtr[i] = query[i];
    }
    final outIdsPtr = malloc<Pointer<Uint64>>();
    final outDistsPtr = malloc<Pointer<Float>>();
    final outCountPtr = malloc<IntPtr>();

    try {
      final status = _bindings.grafeoMmrSearch(
        _handle,
        labelPtr,
        propertyPtr,
        queryPtr,
        query.length,
        k,
        fetchK,
        lambda,
        ef,
        outIdsPtr,
        outDistsPtr,
        outCountPtr,
      );
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);

      final count = outCountPtr.value;
      if (count == 0) return [];

      final ids = outIdsPtr.value;
      final dists = outDistsPtr.value;
      final results = <VectorResult>[
        for (var i = 0; i < count; i++) VectorResult(ids[i], dists[i]),
      ];

      _bindings.grafeoFreeVectorResults(ids, dists, count);
      return results;
    } finally {
      malloc.free(labelPtr);
      malloc.free(propertyPtr);
      malloc.free(queryPtr);
      malloc.free(outIdsPtr);
      malloc.free(outDistsPtr);
      malloc.free(outCountPtr);
    }
  }

  // ===========================================================================
  // Admin
  // ===========================================================================

  /// Save a database snapshot to [path].
  void save(String path) {
    _checkOpen();
    final pathPtr = path.toNativeUtf8(allocator: malloc);
    try {
      final status = _bindings.grafeoSave(_handle, pathPtr);
      if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
    } finally {
      malloc.free(pathPtr);
    }
  }

  /// Force a WAL checkpoint.
  void walCheckpoint() {
    _checkOpen();
    final status = _bindings.grafeoWalCheckpoint(_handle);
    if (status != GrafeoStatus.ok.code) throwStatus(_bindings, status);
  }

  // ===========================================================================
  // Internal helpers
  // ===========================================================================

  /// Extract a [QueryResult] from a native result pointer, then free it.
  QueryResult _buildResult(Pointer<Void> resultPtr) {
    try {
      final jsonPtr = _bindings.grafeoResultJson(resultPtr);
      final jsonString = jsonPtr.toDartString();
      final executionTimeMs =
          _bindings.grafeoResultExecutionTimeMs(resultPtr);
      final rowsScanned = _bindings.grafeoResultRowsScanned(resultPtr);

      final rows = parseRows(jsonString);
      final columns = extractColumns(rows);
      final (nodes, edges) = extractEntities(rows);

      return QueryResult(
        columns: columns,
        rows: rows,
        nodes: nodes,
        edges: edges,
        executionTimeMs: executionTimeMs,
        rowsScanned: rowsScanned,
      );
    } finally {
      _bindings.grafeoFreeResult(resultPtr);
    }
  }

  /// Encode a single value (used by CRUD methods for property values).
  static dynamic _encodeValue(dynamic value) {
    return switch (value) {
      null => null,
      bool b => b,
      int i => i,
      double d => d,
      String s => s,
      DateTime dt => {r'$timestamp_us': dt.toUtc().microsecondsSinceEpoch},
      _ => value.toString(),
    };
  }
}
