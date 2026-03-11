# Grafeo Dart Bindings

Dart bindings for the Grafeo graph database.

## Installation

To use Grafeo in your Dart project, add the following dependency to your `pubspec.yaml` file:

```yaml
dependencies:
  grafeo: ^0.1.0
```

## Quick Start

```dart
import 'package:grafeo/grafeo.dart';

void main() async {
  // Create an in-memory database
  final db = await GrafeoDatabase.openMemory();
  
  // Execute a query
  await db.execute('CREATE (:Person {name: "Alix", age: 30})');
  
  // Query the database
  final result = await db.execute('MATCH (p:Person) RETURN p.name, p.age');
  
  // Process the result
  print('Query result: $result');
  
  // Close the database
  await db.close();
}
```

## API Reference

### GrafeoDatabase

#### Static Methods

- `openMemory()`: Open an in-memory database.
- `open(String path)`: Open a persistent database at the given path.
- `version()`: Get the Grafeo library version.

#### Instance Methods

- `execute(String query)`: Execute a GQL query.
- `executeWithParams(String query, Map<String, dynamic> params)`: Execute a GQL query with parameters.
- `nodeCount()`: Get the number of nodes in the database.
- `edgeCount()`: Get the number of edges in the database.
- `dropVectorIndex(String label, String property)`: Drop a vector index.
- `rebuildVectorIndex(String label, String property)`: Rebuild a vector index.
- `mmrSearch(String label, String property, List<double> query, int k, int fetchK, double lambda, int ef)`: Perform an MMR search.
- `close()`: Close the database and free its resources.

## Building from Source

To build the Grafeo Dart bindings from source, follow these steps:

1. Clone the Grafeo repository:
   ```bash
   git clone https://github.com/GrafeoDB/grafeo.git
   cd grafeo
   ```

2. Build the Rust library:
   ```bash
   cargo build --release
   ```

3. Copy the built library to the Dart package directory:
   - On Windows: `copy target\release\grafeo_c.dll crates\bindings\dart\`
   - On macOS: `cp target/release/libgrafeo_c.dylib crates/bindings/dart/`
   - On Linux: `cp target/release/libgrafeo_c.so crates/bindings/dart/`

4. Build the Dart package:
   ```bash
   cd dart
   dart pub get
   ```

## License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.
