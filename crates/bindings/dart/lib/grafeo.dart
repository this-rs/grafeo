/// Dart bindings for Grafeo graph database.
///
/// This package provides a Dart API for interacting with the Grafeo graph database.
/// It uses FFI to communicate with the underlying Rust implementation.
///
/// ## Quick Start
///
/// ```dart
/// import 'package:grafeo/grafeo.dart';
///
/// void main() async {
///   // Create an in-memory database
///   final db = await GrafeoDatabase.openMemory();
///   
///   // Execute a query
///   final result = await db.execute('CREATE (:Person {name: "Alix", age: 30})');
///   
///   // Query the database
///   final queryResult = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
///   
///   // Process the result
///   print('Query result: $queryResult');
///   
///   // Close the database
///   await db.close();
/// }
/// ```

library grafeo;

export 'src/database.dart';
export 'src/error.dart';
export 'src/types.dart';
