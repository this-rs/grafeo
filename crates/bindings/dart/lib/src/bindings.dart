/// FFI bindings for Grafeo graph database.

import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'types.dart';

/// Native types for FFI.
typedef Size = IntPtr;

/// FFI bindings for Grafeo graph database.
class GrafeoBindings {
  /// Load the Grafeo shared library.
  static DynamicLibrary loadLibrary() {
    final String libraryName;
    if (Platform.isWindows) {
      libraryName = 'grafeo_c.dll';
    } else if (Platform.isMacOS) {
      libraryName = 'libgrafeo_c.dylib';
    } else if (Platform.isLinux) {
      libraryName = 'libgrafeo_c.so';
    } else {
      throw UnsupportedError('Unsupported platform');
    }

    return DynamicLibrary.open(libraryName);
  }

  /// The underlying dynamic library.
  final DynamicLibrary library;

  /// Create a new GrafeoBindings instance.
  GrafeoBindings(this.library);

  /// Open an in-memory database.
  GrafeoDatabasePtr Function() get grafeo_open_memory => library
      .lookup<NativeFunction<GrafeoDatabasePtr Function()>>(
          'grafeo_open_memory')
      .asFunction();

  /// Open a persistent database at the given path.
  GrafeoDatabasePtr Function(Pointer<Utf8>) get grafeo_open => library
      .lookup<NativeFunction<GrafeoDatabasePtr Function(Pointer<Utf8>)>>(
          'grafeo_open')
      .asFunction();

  /// Get database info
  Pointer<Utf8> Function(GrafeoDatabasePtr) get grafeo_info => library
      .lookup<NativeFunction<Pointer<Utf8> Function(GrafeoDatabasePtr)>>(
          'grafeo_info')
      .asFunction();

  /// Close a database and free its resources.
  int Function(GrafeoDatabasePtr) get grafeo_close => library
      .lookup<NativeFunction<Int32 Function(GrafeoDatabasePtr)>>('grafeo_close')
      .asFunction();

  /// Free a database handle.
  void Function(GrafeoDatabasePtr) get grafeo_free_database => library
      .lookup<NativeFunction<Void Function(GrafeoDatabasePtr)>>(
          'grafeo_free_database')
      .asFunction();

  /// Execute a GQL query.
  GrafeoResultPtr Function(GrafeoDatabasePtr, Pointer<Utf8>)
      get grafeo_execute => library
          .lookup<
              NativeFunction<
                  GrafeoResultPtr Function(
                      GrafeoDatabasePtr, Pointer<Utf8>)>>('grafeo_execute')
          .asFunction();

  /// Execute a GQL query with parameters.
  GrafeoResultPtr Function(GrafeoDatabasePtr, Pointer<Utf8>, Pointer<Utf8>)
      get grafeo_execute_with_params => library
          .lookup<
              NativeFunction<
                  GrafeoResultPtr Function(GrafeoDatabasePtr, Pointer<Utf8>,
                      Pointer<Utf8>)>>('grafeo_execute_with_params')
          .asFunction();

  /// Free a query result.
  void Function(GrafeoResultPtr) get grafeo_free_result => library
      .lookup<NativeFunction<Void Function(GrafeoResultPtr)>>(
          'grafeo_free_result')
      .asFunction();

  /// Get the JSON string representation of a query result.
  Pointer<Utf8> Function(GrafeoResultPtr) get grafeo_result_json => library
      .lookup<NativeFunction<Pointer<Utf8> Function(GrafeoResultPtr)>>(
          'grafeo_result_json')
      .asFunction();

  /// Get the number of nodes in the database.
  int Function(GrafeoDatabasePtr) get grafeo_node_count => library
      .lookup<NativeFunction<Uint64 Function(GrafeoDatabasePtr)>>(
          'grafeo_node_count')
      .asFunction();

  /// Get the number of edges in the database.
  int Function(GrafeoDatabasePtr) get grafeo_edge_count => library
      .lookup<NativeFunction<Uint64 Function(GrafeoDatabasePtr)>>(
          'grafeo_edge_count')
      .asFunction();

  /// Drop a vector index.
  int Function(GrafeoDatabasePtr, Pointer<Utf8>, Pointer<Utf8>)
      get grafeo_drop_vector_index => library
          .lookup<
              NativeFunction<
                  Int32 Function(GrafeoDatabasePtr, Pointer<Utf8>,
                      Pointer<Utf8>)>>('grafeo_drop_vector_index')
          .asFunction();

  /// Rebuild a vector index.
  int Function(GrafeoDatabasePtr, Pointer<Utf8>, Pointer<Utf8>)
      get grafeo_rebuild_vector_index => library
          .lookup<
              NativeFunction<
                  Int32 Function(GrafeoDatabasePtr, Pointer<Utf8>,
                      Pointer<Utf8>)>>('grafeo_rebuild_vector_index')
          .asFunction();

  // MMR search functions are defined directly in database.dart to avoid type issues

  /// Get the Grafeo library version.
  Pointer<Utf8> Function() get grafeo_version => library
      .lookup<NativeFunction<Pointer<Utf8> Function()>>('grafeo_version')
      .asFunction();

  /// Get the last error message.
  Pointer<Utf8> Function() get grafeo_last_error => library
      .lookup<NativeFunction<Pointer<Utf8> Function()>>('grafeo_last_error')
      .asFunction();

  /// Free a string returned by the Grafeo API.
  void Function(Pointer<Void>) get grafeo_free_string => library
      .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
          'grafeo_free_string')
      .asFunction();
}

/// Extension to convert C bool (int32) to Dart bool.
extension Int32ToBool on int {
  /// Convert C bool (int32) to Dart bool.
  bool toDartBool() => this != 0;
}

/// Global Grafeo bindings instance.
final bindings = GrafeoBindings(GrafeoBindings.loadLibrary());
