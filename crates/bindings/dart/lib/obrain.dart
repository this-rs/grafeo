/// Dart bindings for the Obrain graph database.
///
/// Provides a native Dart API backed by the obrain-c shared library via FFI.
///
/// ## Quick Start
///
/// ```dart
/// import 'package:obrain/obrain.dart';
///
/// void main() {
///   final db = ObrainDB.memory();
///
///   db.execute("INSERT (:Person {name: 'Alix', age: 30})");
///   final result = db.execute(
///     "MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.name, p.age",
///   );
///   print(result); // QueryResult(1 rows, ...)
///
///   db.close();
/// }
/// ```
library;

export 'src/database.dart';
export 'src/error.dart';
export 'src/transaction.dart';
export 'src/types.dart';
export 'src/value.dart' show encodeParams, encodeValue;
