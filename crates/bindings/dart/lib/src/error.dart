/// Error handling for the Obrain Dart binding.
///
/// Status codes match the C enum in `obrain-c/src/error.rs` exactly.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'ffi/bindings.dart';

/// Status codes returned by obrain-c FFI functions.
///
/// Values must match the C `ObrainStatus` enum:
///   Ok=0, ErrorDatabase=1, ErrorQuery=2, ErrorTransaction=3, ErrorStorage=4,
///   ErrorIo=5, ErrorSerialization=6, ErrorInternal=7, ErrorNullPointer=8,
///   ErrorInvalidUtf8=9.
enum ObrainStatus {
  ok(0),
  database(1),
  query(2),
  transaction(3),
  storage(4),
  io(5),
  serialization(6),
  internal(7),
  nullPointer(8),
  invalidUtf8(9);

  final int code;
  const ObrainStatus(this.code);

  static ObrainStatus fromCode(int code) =>
      values.firstWhere((s) => s.code == code, orElse: () => internal);
}

/// Base exception for all Obrain errors.
sealed class ObrainException implements Exception {
  final String message;
  final ObrainStatus status;
  const ObrainException(this.message, this.status);

  @override
  String toString() => '$runtimeType(${status.name}): $message';
}

/// A query parsing or execution error (status 2).
class QueryException extends ObrainException {
  const QueryException(super.message, super.status);
}

/// A transaction error such as conflict or invalid state (status 3).
class TransactionException extends ObrainException {
  const TransactionException(super.message, super.status);
}

/// A storage or IO error (status 4, 5).
class StorageException extends ObrainException {
  const StorageException(super.message, super.status);
}

/// A serialization error (status 6).
class SerializationException extends ObrainException {
  const SerializationException(super.message, super.status);
}

/// A generic database error (status 1, 7, 8, 9, or unknown).
class DatabaseException extends ObrainException {
  /// Creates a [DatabaseException] with [message] and [status].
  const DatabaseException(super.message, super.status);
}

/// Map a C status code and error message to a typed Dart exception.
///
/// Mirrors `obrain-bindings-common::error::classify_error`.
ObrainException classifyError(int statusCode, String message) {
  final status = ObrainStatus.fromCode(statusCode);
  return switch (status) {
    ObrainStatus.query => QueryException(message, status),
    ObrainStatus.transaction => TransactionException(message, status),
    ObrainStatus.storage || ObrainStatus.io =>
      StorageException(message, status),
    ObrainStatus.serialization => SerializationException(message, status),
    _ => DatabaseException(message, status),
  };
}

/// Read the last error message from the obrain-c thread-local error slot.
///
/// The returned pointer is owned by the C library (thread-local storage) and
/// is valid until the next FFI call on this thread. We copy it to a Dart
/// string immediately and do NOT free it.
String lastError(ObrainBindings bindings) {
  final ptr = bindings.obrainLastError();
  if (ptr == nullptr) return 'Unknown error';
  return ptr.toDartString();
}

/// Throw a [ObrainException] for a failed FFI call that returned a status code.
Never throwStatus(ObrainBindings bindings, int statusCode) {
  throw classifyError(statusCode, lastError(bindings));
}

/// Throw a [ObrainException] for a failed FFI call that returned null.
Never throwLastError(ObrainBindings bindings) {
  throw DatabaseException(
    lastError(bindings),
    ObrainStatus.database,
  );
}
