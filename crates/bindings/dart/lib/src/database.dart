/// Database operations for Grafeo Dart bindings.

import 'dart:ffi';
import 'dart:convert';

import 'package:ffi/ffi.dart';

import 'bindings.dart';
import 'error.dart';
import 'types.dart';

/// Native types for FFI.
typedef Size = IntPtr;

/// Grafeo database handle.
class GrafeoDatabase {
  /// The underlying database pointer.
  final GrafeoDatabasePtr _ptr;

  /// Whether the database is closed.
  bool _closed = false;

  /// Create a new GrafeoDatabase instance from a pointer.
  GrafeoDatabase._(this._ptr);

  /// Open an in-memory database.
  static Future<GrafeoDatabase> openMemory() async {
    final ptr = bindings.grafeo_open_memory();
    if (ptr == nullptr) {
      final error = getLastError();
      throw GrafeoError(error, GrafeoStatus.error);
    }
    return GrafeoDatabase._(ptr);
  }

  /// Open a persistent database at the given path.
  static Future<GrafeoDatabase> open(String path) async {
    final pathPtr = path.toNativeUtf8();
    try {
      final ptr = bindings.grafeo_open(pathPtr);
      if (ptr == nullptr) {
        final error = getLastError();
        throw GrafeoError(error, GrafeoStatus.error);
      }
      return GrafeoDatabase._(ptr);
    } finally {
      calloc.free(pathPtr);
    }
  }

  /// Close the database and free its resources.
  Future<void> close() async {
    if (_closed) {
      return;
    }

    final status = bindings.grafeo_close(_ptr);
    bindings.grafeo_free_database(_ptr);
    _closed = true;

    final grafeoStatus = GrafeoStatus.fromValue(status);
    if (grafeoStatus != GrafeoStatus.ok) {
      final error = getLastError();
      throw GrafeoError(error, grafeoStatus);
    }
  }

  /// Execute a GQL query.
  Future<dynamic> execute(String query) async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    final queryPtr = query.toNativeUtf8();
    try {
      final resultPtr = bindings.grafeo_execute(_ptr, queryPtr);
      if (resultPtr == nullptr) {
        final error = getLastError();
        throw GrafeoError(error, GrafeoStatus.error);
      }

      // Get the JSON result
      final jsonPtr = bindings.grafeo_result_json(resultPtr);
      final jsonString = jsonPtr.cast<Utf8>().toDartString();

      // Parse the JSON result
      final result = _parseQueryResult(jsonString);

      bindings.grafeo_free_result(resultPtr);
      return result;
    } finally {
      calloc.free(queryPtr);
    }
  }

  /// Execute a GQL query with parameters.
  Future<dynamic> executeWithParams(
      String query, Map<String, dynamic> params) async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    final queryPtr = query.toNativeUtf8();
    final paramsJson = _encodeParams(params);
    final paramsPtr = paramsJson.toNativeUtf8();

    try {
      final resultPtr =
          bindings.grafeo_execute_with_params(_ptr, queryPtr, paramsPtr);
      if (resultPtr == nullptr) {
        final error = getLastError();
        throw GrafeoError(error, GrafeoStatus.error);
      }

      // Get the JSON result
      final jsonPtr = bindings.grafeo_result_json(resultPtr);
      final jsonString = jsonPtr.cast<Utf8>().toDartString();

      // Parse the JSON result
      final result = _parseQueryResult(jsonString);

      bindings.grafeo_free_result(resultPtr);
      return result;
    } finally {
      calloc.free(queryPtr);
      calloc.free(paramsPtr);
    }
  }

  /// Get the number of nodes in the database.
  Future<int> nodeCount() async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    return bindings.grafeo_node_count(_ptr);
  }

  /// Get the number of edges in the database.
  Future<int> edgeCount() async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    return bindings.grafeo_edge_count(_ptr);
  }

  /// Drop a vector index.
  Future<bool> dropVectorIndex(String label, String property) async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    final labelPtr = label.toNativeUtf8();
    final propertyPtr = property.toNativeUtf8();

    try {
      final result =
          bindings.grafeo_drop_vector_index(_ptr, labelPtr, propertyPtr);
      return result.toDartBool();
    } finally {
      calloc.free(labelPtr);
      calloc.free(propertyPtr);
    }
  }

  /// Rebuild a vector index.
  Future<void> rebuildVectorIndex(String label, String property) async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    final labelPtr = label.toNativeUtf8();
    final propertyPtr = property.toNativeUtf8();

    try {
      final status =
          bindings.grafeo_rebuild_vector_index(_ptr, labelPtr, propertyPtr);
      final grafeoStatus = GrafeoStatus.fromValue(status);
      if (grafeoStatus != GrafeoStatus.ok) {
        final error = getLastError();
        throw GrafeoError(error, grafeoStatus);
      }
    } finally {
      calloc.free(labelPtr);
      calloc.free(propertyPtr);
    }
  }

  /// Perform an MMR search.
  Future<List<VectorResult>> mmrSearch(
    String label,
    String property,
    List<double> query,
    int k,
    int fetchK,
    double lambda,
    int ef,
  ) async {
    if (_closed) {
      throw GrafeoError('Database is closed', GrafeoStatus.error);
    }

    final labelPtr = label.toNativeUtf8();
    final propertyPtr = property.toNativeUtf8();
    final queryPtr = calloc<Float>(query.length);

    for (var i = 0; i < query.length; i++) {
      queryPtr[i] = query[i];
    }

    final outIdsPtr = calloc<Pointer<Uint64>>();
    final outDistsPtr = calloc<Pointer<Float>>();
    final outCountPtr = calloc<Size>();

    try {
      // Use dynamic lookup to avoid type errors
      final result = await _executeDynamic(
        'grafeo_mmr_search',
        [
          _ptr.cast<Void>(),
          labelPtr,
          propertyPtr,
          queryPtr.cast<Void>(),
          query.length,
          k,
          fetchK,
          lambda,
          ef,
          outIdsPtr.cast<Void>(),
          outDistsPtr.cast<Void>(),
          outCountPtr.cast<Pointer<IntPtr>>(),
        ],
      );

      final status = result as int;
      final grafeoStatus = GrafeoStatus.fromValue(status);
      if (grafeoStatus != GrafeoStatus.ok) {
        final error = getLastError();
        throw GrafeoError(error, grafeoStatus);
      }

      final count = outCountPtr[0];
      if (count == 0) {
        return [];
      }

      final ids = outIdsPtr[0];
      final dists = outDistsPtr[0];

      final results = <VectorResult>[];
      for (var i = 0; i < count; i++) {
        results.add(VectorResult(ids[i], dists[i]));
      }

      // Free the results
      await _executeDynamic(
        'grafeo_free_vector_results',
        [
          ids.cast<Void>(),
          dists.cast<Void>(),
          count,
        ],
      );

      return results;
    } finally {
      calloc.free(labelPtr);
      calloc.free(propertyPtr);
      calloc.free(queryPtr);
      calloc.free(outIdsPtr);
      calloc.free(outDistsPtr);
      calloc.free(outCountPtr);
    }
  }

  /// Execute a dynamic FFI function to avoid type errors
  Future<dynamic> _executeDynamic(
      String functionName, List<dynamic> args) async {
    final library = GrafeoBindings.loadLibrary();

    if (functionName == 'grafeo_mmr_search') {
      // Lookup the function with the correct signature
      final fn = library.lookup<
          NativeFunction<
              Int32 Function(
                Pointer<Void>,
                Pointer<Utf8>,
                Pointer<Utf8>,
                Pointer<Void>,
                IntPtr,
                IntPtr,
                Int32,
                Float,
                Int32,
                Pointer<Void>,
                Pointer<Void>,
                Pointer<IntPtr>,
              )>>(functionName);

      // Convert to Dart function
      final dartFn = fn.asFunction<
          int Function(
            Pointer<Void>,
            Pointer<Utf8>,
            Pointer<Utf8>,
            Pointer<Void>,
            int,
            int,
            int,
            double,
            int,
            Pointer<Void>,
            Pointer<Void>,
            Pointer<IntPtr>,
          )>();

      // Call the function with the provided arguments
      return dartFn(
        args[0] as Pointer<Void>,
        args[1] as Pointer<Utf8>,
        args[2] as Pointer<Utf8>,
        args[3] as Pointer<Void>,
        args[4] as int,
        args[5] as int,
        args[6] as int,
        args[7] as double,
        args[8] as int,
        args[9] as Pointer<Void>,
        args[10] as Pointer<Void>,
        args[11] as Pointer<IntPtr>,
      );
    } else if (functionName == 'grafeo_free_vector_results') {
      // Lookup the function with the correct signature
      final fn = library.lookup<
          NativeFunction<
              Void Function(
                Pointer<Void>,
                Pointer<Void>,
                IntPtr,
              )>>(functionName);

      // Convert to Dart function
      final dartFn = fn.asFunction<
          void Function(
            Pointer<Void>,
            Pointer<Void>,
            int,
          )>();

      // Call the function with the provided arguments
      dartFn(
        args[0] as Pointer<Void>,
        args[1] as Pointer<Void>,
        args[2] as int,
      );

      return null;
    } else {
      throw ArgumentError('Unknown function: $functionName');
    }
  }

  /// Get the Grafeo library version.
  static String version() {
    final versionPtr = bindings.grafeo_version();
    final version = versionPtr.cast<Utf8>().toDartString();
    // freeString(versionPtr.cast<Utf8>()); // Temporarily disabled to avoid crash
    return version;
  }

  /// Encode parameters as JSON string.
  String _encodeParams(Map<String, dynamic> params) {
    // For now, we'll just return an empty string
    // In a real implementation, we would encode the params as JSON
    return '{}';
  }

  /// Parse the query result from JSON string.
  dynamic _parseQueryResult(String jsonString) {
    final dynamic json = jsonDecode(jsonString);
    return json;
  }
}
