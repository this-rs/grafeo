/// Error handling for Grafeo Dart bindings.

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'bindings.dart';

/// Status codes returned by Grafeo FFI functions.
enum GrafeoStatus {
  /// Success
  ok(0),

  /// Generic error
  error(1),

  /// Invalid argument
  invalidArgument(2),

  /// Database not found
  notFound(3),

  /// Database already exists
  alreadyExists(4),

  /// Operation not supported
  notSupported(5),

  /// Out of memory
  outOfMemory(6),

  /// IO error
  ioError(7),

  /// Query parsing error
  parseError(8),

  /// Transaction error
  transactionError(9);

  /// The integer value of the status code.
  final int value;

  /// Create a new GrafeoStatus.
  const GrafeoStatus(this.value);

  /// Get the GrafeoStatus from an integer value.
  static GrafeoStatus fromValue(int value) {
    return values.firstWhere((status) => status.value == value,
        orElse: () => error);
  }
}

/// Grafeo database error.
class GrafeoError implements Exception {
  /// The error message.
  final String message;

  /// The status code.
  final GrafeoStatus status;

  /// Create a new GrafeoError.
  const GrafeoError(this.message, this.status);

  @override
  String toString() => 'GrafeoError($status): $message';
}

/// Get the last error message from the Grafeo FFI.
String getLastError() {
  final errorPtr = bindings.grafeo_last_error();
  if (errorPtr == nullptr) {
    return 'Unknown error';
  }

  final errorString = errorPtr.cast<Utf8>().toDartString();
  // freeString(errorPtr.cast<Utf8>()); // Temporarily disabled to avoid crash
  return errorString;
}

/// Free a string returned by the Grafeo API.
void freeString(Pointer<Utf8> string) {
  bindings.grafeo_free_string(string.cast<Void>());
}
