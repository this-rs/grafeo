
/// Types for Grafeo Dart bindings.

import 'dart:ffi';

/// Opaque pointer to a Grafeo database.
typedef GrafeoDatabasePtr = Pointer<Void>;

/// Opaque pointer to a Grafeo query result.
typedef GrafeoResultPtr = Pointer<Void>;

/// Opaque pointer to a Grafeo transaction.
typedef GrafeoTransactionPtr = Pointer<Void>;

/// Opaque pointer to a Grafeo session.
typedef GrafeoSessionPtr = Pointer<Void>;

/// Vector search result.
class VectorResult {
  /// Node ID
  final int nodeId;
  
  /// Distance
  final double distance;
  
  /// Create a new VectorResult.
  const VectorResult(this.nodeId, this.distance);
  
  @override
  String toString() => 'VectorResult(nodeId: $nodeId, distance: $distance)';
}
