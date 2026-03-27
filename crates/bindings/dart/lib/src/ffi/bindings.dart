/// Raw FFI bindings for every obrain-c function.
///
/// All lookups use `late final` so the symbol resolution happens once per
/// binding instance, not on every call.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

/// Statically-typed FFI bindings for the obrain-c shared library.
final class ObrainBindings {
  /// The underlying [DynamicLibrary] used for symbol lookups.
  final DynamicLibrary library;

  ObrainBindings(this.library);

  // ===========================================================================
  // Error handling
  // ===========================================================================

  /// Returns the last error message (thread-local). Do NOT free the pointer.
  late final obrainLastError = library
      .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
        'obrain_last_error',
      );

  /// Clears the thread-local error state.
  late final obrainClearError =
      library.lookupFunction<Void Function(), void Function()>(
        'obrain_clear_error',
      );

  /// Free a heap-allocated string returned by obrain-c (e.g. obrain_info).
  late final obrainFreeString = library
      .lookupFunction<Void Function(Pointer<Utf8>), void Function(
        Pointer<Utf8>,
      )>('obrain_free_string');

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  /// Create a new in-memory database. Returns null on error.
  late final obrainOpenMemory = library.lookupFunction<
    Pointer<Void> Function(),
    Pointer<Void> Function()
  >('obrain_open_memory');

  /// Open a persistent database at [path]. Returns null on error.
  late final obrainOpen = library.lookupFunction<
    Pointer<Void> Function(Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Utf8>)
  >('obrain_open');

  /// Close a database, flushing writes. Returns ObrainStatus.
  late final obrainClose = library.lookupFunction<
    Int32 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_close');

  /// Free the opaque database handle.
  late final obrainFreeDatabase = library.lookupFunction<
    Void Function(Pointer<Void>),
    void Function(Pointer<Void>)
  >('obrain_free_database');

  /// Returns the library version string. Static pointer, do NOT free.
  late final obrainVersion = library
      .lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>(
        'obrain_version',
      );

  // ===========================================================================
  // Query execution
  // ===========================================================================

  /// Execute a GQL query. Returns result pointer or null on error.
  late final obrainExecute = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_execute');

  /// Execute a GQL query with JSON-encoded parameters.
  late final obrainExecuteWithParams = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_execute_with_params');

  /// Execute a Cypher query.
  late final obrainExecuteCypher = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_execute_cypher');

  /// Execute a Gremlin query.
  late final obrainExecuteGremlin = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_execute_gremlin');

  /// Execute a GraphQL query.
  late final obrainExecuteGraphql = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_execute_graphql');

  /// Execute a SPARQL query.
  late final obrainExecuteSparql = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_execute_sparql');

  // ===========================================================================
  // Result access
  // ===========================================================================

  /// Get JSON string from a result. Pointer valid until obrain_free_result.
  late final obrainResultJson = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_result_json');

  /// Get the number of rows in a result.
  late final obrainResultRowCount = library.lookupFunction<
    IntPtr Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_result_row_count');

  /// Get execution time in milliseconds.
  late final obrainResultExecutionTimeMs = library.lookupFunction<
    Double Function(Pointer<Void>),
    double Function(Pointer<Void>)
  >('obrain_result_execution_time_ms');

  /// Get the number of rows scanned.
  late final obrainResultRowsScanned = library.lookupFunction<
    Uint64 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_result_rows_scanned');

  /// Free a result handle.
  late final obrainFreeResult = library.lookupFunction<
    Void Function(Pointer<Void>),
    void Function(Pointer<Void>)
  >('obrain_free_result');

  // ===========================================================================
  // Node CRUD
  // ===========================================================================

  /// Create a node with JSON labels and properties. Returns node ID (0 = error).
  late final obrainCreateNode = library.lookupFunction<
    Uint64 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_create_node');

  /// Get a node by ID. Writes to [out]. Returns ObrainStatus.
  late final obrainGetNode = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Pointer<Void>>),
    int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  >('obrain_get_node');

  /// Delete a node by ID. Returns 0 on success, -1 on error.
  late final obrainDeleteNode = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64),
    int Function(Pointer<Void>, int)
  >('obrain_delete_node');

  /// Set a property on a node. Returns ObrainStatus.
  late final obrainSetNodeProperty = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Uint64,
      Pointer<Utf8>,
      Pointer<Utf8>,
    ),
    int Function(Pointer<Void>, int, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_set_node_property');

  /// Remove a property from a node. Returns 0 on success, -1 on error.
  late final obrainRemoveNodeProperty = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
    int Function(Pointer<Void>, int, Pointer<Utf8>)
  >('obrain_remove_node_property');

  /// Add a label to a node. Returns 0 on success, -1 on error.
  late final obrainAddNodeLabel = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
    int Function(Pointer<Void>, int, Pointer<Utf8>)
  >('obrain_add_node_label');

  /// Remove a label from a node. Returns 0 on success, -1 on error.
  late final obrainRemoveNodeLabel = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
    int Function(Pointer<Void>, int, Pointer<Utf8>)
  >('obrain_remove_node_label');

  /// Get labels for a node as JSON. Caller must free with obrainFreeString.
  late final obrainGetNodeLabels = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>, Uint64),
    Pointer<Utf8> Function(Pointer<Void>, int)
  >('obrain_get_node_labels');

  /// Get node ID from an opaque node pointer.
  late final obrainNodeId = library.lookupFunction<
    Uint64 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_node_id');

  /// Get node labels JSON. Valid until obrain_free_node.
  late final obrainNodeLabelsJson = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_node_labels_json');

  /// Get node properties JSON. Valid until obrain_free_node.
  late final obrainNodePropertiesJson = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_node_properties_json');

  /// Free an opaque node handle.
  late final obrainFreeNode = library.lookupFunction<
    Void Function(Pointer<Void>),
    void Function(Pointer<Void>)
  >('obrain_free_node');

  // ===========================================================================
  // Edge CRUD
  // ===========================================================================

  /// Create an edge. Returns edge ID (0 = error).
  late final obrainCreateEdge = library.lookupFunction<
    Uint64 Function(
      Pointer<Void>,
      Uint64,
      Uint64,
      Pointer<Utf8>,
      Pointer<Utf8>,
    ),
    int Function(Pointer<Void>, int, int, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_create_edge');

  /// Get an edge by ID. Writes to [out]. Returns ObrainStatus.
  late final obrainGetEdge = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Pointer<Void>>),
    int Function(Pointer<Void>, int, Pointer<Pointer<Void>>)
  >('obrain_get_edge');

  /// Delete an edge by ID. Returns 0 on success, -1 on error.
  late final obrainDeleteEdge = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64),
    int Function(Pointer<Void>, int)
  >('obrain_delete_edge');

  /// Set a property on an edge. Returns ObrainStatus.
  late final obrainSetEdgeProperty = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Uint64,
      Pointer<Utf8>,
      Pointer<Utf8>,
    ),
    int Function(Pointer<Void>, int, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_set_edge_property');

  /// Remove a property from an edge. Returns 0 on success, -1 on error.
  late final obrainRemoveEdgeProperty = library.lookupFunction<
    Int32 Function(Pointer<Void>, Uint64, Pointer<Utf8>),
    int Function(Pointer<Void>, int, Pointer<Utf8>)
  >('obrain_remove_edge_property');

  /// Get edge ID from an opaque edge pointer.
  late final obrainEdgeId = library.lookupFunction<
    Uint64 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_edge_id');

  /// Get source node ID from an edge pointer.
  late final obrainEdgeSourceId = library.lookupFunction<
    Uint64 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_edge_source_id');

  /// Get target node ID from an edge pointer.
  late final obrainEdgeTargetId = library.lookupFunction<
    Uint64 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_edge_target_id');

  /// Get edge type string. Valid until obrain_free_edge.
  late final obrainEdgeType = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_edge_type');

  /// Get edge properties JSON. Valid until obrain_free_edge.
  late final obrainEdgePropertiesJson = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_edge_properties_json');

  /// Free an opaque edge handle.
  late final obrainFreeEdge = library.lookupFunction<
    Void Function(Pointer<Void>),
    void Function(Pointer<Void>)
  >('obrain_free_edge');

  // ===========================================================================
  // Property indexes
  // ===========================================================================

  /// Create a property index. Returns ObrainStatus.
  late final obrainCreatePropertyIndex = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_create_property_index');

  /// Drop a property index. Returns 0 on success, -1 on error.
  late final obrainDropPropertyIndex = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_drop_property_index');

  /// Check if a property index exists. Returns 1 if exists, 0 if not.
  late final obrainHasPropertyIndex = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_has_property_index');

  /// Find nodes by property value. Writes IDs to [outIds], count to [outCount].
  late final obrainFindNodesByProperty = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Pointer<Uint64>>,
      Pointer<IntPtr>,
    ),
    int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Pointer<Uint64>>,
      Pointer<IntPtr>,
    )
  >('obrain_find_nodes_by_property');

  /// Free a node ID array returned by obrainFindNodesByProperty.
  late final obrainFreeNodeIds = library.lookupFunction<
    Void Function(Pointer<Uint64>, IntPtr),
    void Function(Pointer<Uint64>, int)
  >('obrain_free_node_ids');

  // ===========================================================================
  // Vector operations
  // ===========================================================================

  /// Create a vector index. Returns ObrainStatus.
  late final obrainCreateVectorIndex = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Int32,
      Pointer<Utf8>,
      Int32,
      Int32,
    ),
    int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      int,
      Pointer<Utf8>,
      int,
      int,
    )
  >('obrain_create_vector_index');

  /// Drop a vector index. Returns 0 on success, -1 on error.
  late final obrainDropVectorIndex = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_drop_vector_index');

  /// Rebuild a vector index. Returns ObrainStatus.
  late final obrainRebuildVectorIndex = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_rebuild_vector_index');

  /// Vector similarity search.
  late final obrainVectorSearch = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      IntPtr,
      IntPtr,
      Int32,
      Pointer<Pointer<Uint64>>,
      Pointer<Pointer<Float>>,
      Pointer<IntPtr>,
    ),
    int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      int,
      int,
      int,
      Pointer<Pointer<Uint64>>,
      Pointer<Pointer<Float>>,
      Pointer<IntPtr>,
    )
  >('obrain_vector_search');

  /// MMR (Maximal Marginal Relevance) search.
  late final obrainMmrSearch = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      IntPtr,
      IntPtr,
      Int32,
      Float,
      Int32,
      Pointer<Pointer<Uint64>>,
      Pointer<Pointer<Float>>,
      Pointer<IntPtr>,
    ),
    int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      int,
      int,
      int,
      double,
      int,
      Pointer<Pointer<Uint64>>,
      Pointer<Pointer<Float>>,
      Pointer<IntPtr>,
    )
  >('obrain_mmr_search');

  /// Batch-create nodes with vector embeddings.
  late final obrainBatchCreateNodes = library.lookupFunction<
    Int32 Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      IntPtr,
      IntPtr,
      Pointer<Pointer<Uint64>>,
    ),
    int Function(
      Pointer<Void>,
      Pointer<Utf8>,
      Pointer<Utf8>,
      Pointer<Float>,
      int,
      int,
      Pointer<Pointer<Uint64>>,
    )
  >('obrain_batch_create_nodes');

  /// Free vector search results.
  late final obrainFreeVectorResults = library.lookupFunction<
    Void Function(Pointer<Uint64>, Pointer<Float>, IntPtr),
    void Function(Pointer<Uint64>, Pointer<Float>, int)
  >('obrain_free_vector_results');

  // ===========================================================================
  // Statistics
  // ===========================================================================

  /// Get the number of nodes.
  late final obrainNodeCount = library.lookupFunction<
    IntPtr Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_node_count');

  /// Get the number of edges.
  late final obrainEdgeCount = library.lookupFunction<
    IntPtr Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_edge_count');

  // ===========================================================================
  // Transactions
  // ===========================================================================

  /// Begin a transaction. Returns null on error.
  late final obrainBeginTransaction = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>),
    Pointer<Void> Function(Pointer<Void>)
  >('obrain_begin_transaction');

  /// Begin a transaction with a specific isolation level.
  late final obrainBeginTransactionWithIsolation = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Int32),
    Pointer<Void> Function(Pointer<Void>, int)
  >('obrain_begin_transaction_with_isolation');

  /// Execute a query within a transaction.
  late final obrainTransactionExecute = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_transaction_execute');

  /// Execute a parameterized query within a transaction.
  late final obrainTransactionExecuteWithParams = library.lookupFunction<
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>),
    Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>, Pointer<Utf8>)
  >('obrain_transaction_execute_with_params');

  /// Commit a transaction. Returns ObrainStatus.
  late final obrainCommit = library.lookupFunction<
    Int32 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_commit');

  /// Rollback a transaction. Returns ObrainStatus.
  late final obrainRollback = library.lookupFunction<
    Int32 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_rollback');

  /// Free a transaction handle.
  late final obrainFreeTransaction = library.lookupFunction<
    Void Function(Pointer<Void>),
    void Function(Pointer<Void>)
  >('obrain_free_transaction');

  // ===========================================================================
  // Admin
  // ===========================================================================

  /// Get database info as JSON. Caller must free with obrainFreeString.
  late final obrainInfo = library.lookupFunction<
    Pointer<Utf8> Function(Pointer<Void>),
    Pointer<Utf8> Function(Pointer<Void>)
  >('obrain_info');

  /// Save a snapshot to the given path. Returns ObrainStatus.
  late final obrainSave = library.lookupFunction<
    Int32 Function(Pointer<Void>, Pointer<Utf8>),
    int Function(Pointer<Void>, Pointer<Utf8>)
  >('obrain_save');

  /// Force a WAL checkpoint. Returns ObrainStatus.
  late final obrainWalCheckpoint = library.lookupFunction<
    Int32 Function(Pointer<Void>),
    int Function(Pointer<Void>)
  >('obrain_wal_checkpoint');
}
