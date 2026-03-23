# Grafeo Dart Bindings

Dart FFI bindings for the [Grafeo](https://grafeo.dev) graph database. Wraps the `grafeo-c` shared library for native performance with a Dart-idiomatic API.

## Installation

```yaml
dependencies:
  grafeo: ^0.5.24
```

You also need the `grafeo-c` native library for your platform. See [Building from Source](#building-from-source) below.

## Quick Start

```dart
import 'package:grafeo/grafeo.dart';

void main() {
  final db = GrafeoDB.memory();

  // Insert data
  db.execute("INSERT (:Person {name: 'Alix', age: 30})");
  db.execute("INSERT (:Person {name: 'Gus', age: 28})");

  // Query with parameters
  final result = db.executeWithParams(
    r'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name, p.age',
    {'minAge': 25},
  );

  for (final row in result.rows) {
    print('${row['p.name']}: ${row['p.age']}');
  }

  // Transactions
  final tx = db.beginTransaction();
  tx.execute("INSERT (:City {name: 'Amsterdam'})");
  tx.execute("INSERT (:City {name: 'Berlin'})");
  tx.commit();

  // CRUD operations
  final nodeId = db.createNode(['Person'], {'name': 'Vincent'});
  db.setNodeProperty(nodeId, 'role', 'hitman');
  final node = db.getNode(nodeId);
  print(node); // Node(id, [Person], {name: Vincent, role: hitman})

  db.close();
}
```

## API Reference

### GrafeoDB

| Method | Description |
|--------|-------------|
| `GrafeoDB.memory()` | Open an in-memory database |
| `GrafeoDB.open(path)` | Open a persistent database |
| `GrafeoDB.version()` | Get the library version string |
| `execute(query)` | Execute a GQL query |
| `executeWithParams(query, params)` | Execute with parameters |
| `executeCypher(query)` | Execute a Cypher query |
| `executeSparql(query)` | Execute a SPARQL query |
| `beginTransaction()` | Start an ACID transaction |
| `createNode(labels, properties)` | Create a node, returns ID |
| `getNode(id)` | Get a node by ID |
| `deleteNode(id)` | Delete a node |
| `createEdge(src, dst, type, props)` | Create an edge, returns ID |
| `getEdge(id)` | Get an edge by ID |
| `deleteEdge(id)` | Delete an edge |
| `setNodeProperty(id, key, value)` | Set a node property |
| `setEdgeProperty(id, key, value)` | Set an edge property |
| `nodeCount` | Number of nodes |
| `edgeCount` | Number of edges |
| `info()` | Database metadata as JSON map |
| `close()` | Close and flush |

### Transaction

| Method | Description |
|--------|-------------|
| `execute(query)` | Execute within transaction |
| `executeWithParams(query, params)` | Execute with parameters |
| `commit()` | Make changes permanent |
| `rollback()` | Discard changes |

### Types

- **`QueryResult`**: rows, columns, nodes, edges, executionTimeMs, rowsScanned
- **`Node`**: id, labels, properties
- **`Edge`**: id, type, sourceId, targetId, properties
- **`VectorResult`**: nodeId, distance

## Building from Source

```bash
# Clone and build the native library
git clone https://github.com/GrafeoDB/grafeo.git
cd grafeo
cargo build --release -p grafeo-c

# Copy to the Dart package (or your project)
# Linux:   cp target/release/libgrafeo_c.so crates/bindings/dart/
# macOS:   cp target/release/libgrafeo_c.dylib crates/bindings/dart/
# Windows: copy target\release\grafeo_c.dll crates\bindings\dart\

# Run tests
cd crates/bindings/dart
dart pub get
dart test
```

## License

Apache-2.0. See [LICENSE](../../LICENSE) for details.
