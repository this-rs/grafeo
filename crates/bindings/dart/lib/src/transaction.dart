/// ACID transaction support for Obrain.
///
/// Uses [NativeFinalizer] to auto-rollback (via Rust Drop) if neither
/// [commit] nor [rollback] is called.
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'error.dart';
import 'ffi/bindings.dart';
import 'types.dart';
import 'value.dart';

/// An ACID transaction on a Obrain database.
///
/// Obtain via [ObrainDB.beginTransaction]. Must be explicitly committed
/// or rolled back. If dropped without either, the Rust Drop impl
/// performs an automatic rollback.
class Transaction implements Finalizable {
  final ObrainBindings _bindings;
  Pointer<Void> _handle;
  bool _finished = false;

  static NativeFinalizer? _finalizer;

  /// Create a transaction wrapper around a native handle.
  ///
  /// Typically called by [ObrainDB.beginTransaction], not directly.
  Transaction(this._handle, this._bindings) {
    _finalizer ??= NativeFinalizer(
      _bindings.library
          .lookup<NativeFunction<Void Function(Pointer<Void>)>>(
            'obrain_free_transaction',
          ),
    );
    _finalizer!.attach(this, _handle.cast(), detach: this);
  }

  void _checkActive() {
    if (_finished) {
      throw TransactionException(
        'Transaction already finished',
        ObrainStatus.transaction,
      );
    }
  }

  /// Execute a GQL query within this transaction.
  QueryResult execute(String query) {
    _checkActive();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.obrainTransactionExecute(_handle, queryPtr);
      if (resultPtr == nullptr) throwLastError(_bindings);
      return _buildResult(resultPtr);
    } finally {
      malloc.free(queryPtr);
    }
  }

  /// Execute a parameterized GQL query within this transaction.
  QueryResult executeWithParams(
    String query,
    Map<String, dynamic> params,
  ) {
    _checkActive();
    final queryPtr = query.toNativeUtf8(allocator: malloc);
    final paramsJson = encodeParams(params);
    final paramsPtr = paramsJson.toNativeUtf8(allocator: malloc);
    try {
      final resultPtr = _bindings.obrainTransactionExecuteWithParams(
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

  /// Commit the transaction, making all changes permanent.
  void commit() {
    _checkActive();
    _finished = true;
    _finalizer!.detach(this);
    final status = _bindings.obrainCommit(_handle);
    _bindings.obrainFreeTransaction(_handle);
    _handle = nullptr;
    if (status != ObrainStatus.ok.code) {
      throw classifyError(status, lastError(_bindings));
    }
  }

  /// Rollback the transaction, discarding all changes.
  void rollback() {
    _checkActive();
    _finished = true;
    _finalizer!.detach(this);
    final status = _bindings.obrainRollback(_handle);
    _bindings.obrainFreeTransaction(_handle);
    _handle = nullptr;
    if (status != ObrainStatus.ok.code) {
      throw classifyError(status, lastError(_bindings));
    }
  }

  QueryResult _buildResult(Pointer<Void> resultPtr) {
    try {
      final jsonPtr = _bindings.obrainResultJson(resultPtr);
      final jsonString = jsonPtr.toDartString();
      final executionTimeMs =
          _bindings.obrainResultExecutionTimeMs(resultPtr);
      final rowsScanned = _bindings.obrainResultRowsScanned(resultPtr);

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
      _bindings.obrainFreeResult(resultPtr);
    }
  }
}
