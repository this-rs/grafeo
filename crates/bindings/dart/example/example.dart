// ignore_for_file: avoid_print

import 'package:obrain/obrain.dart';

void main() {
  // Open an in-memory database (no file I/O, great for quick experiments).
  final db = ObrainDB.memory();

  try {
    // Create two nodes and a relationship between them.
    db.execute('''
      CREATE (:Person {name: "Alix", age: 30}),
             (:Person {name: "Gus",  age: 28})
    ''');

    db.execute('''
      MATCH (a:Person {name: "Alix"}), (b:Person {name: "Gus"})
      CREATE (a)-[:KNOWS]->(b)
    ''');

    // Query the graph and print results.
    final result = db.execute('''
      MATCH (a:Person)-[r:KNOWS]->(b:Person)
      RETURN a.name AS from, b.name AS to
    ''');

    for (final row in result.rows) {
      print('${row['from']} knows ${row['to']}');
    }
    // Output: Alix knows Gus

    // Transactions: atomic multi-step writes.
    final tx = db.beginTransaction();
    try {
      tx.execute('CREATE (:Person {name: "Vincent"})');
      tx.commit();
    } catch (_) {
      tx.rollback();
      rethrow;
    }
  } finally {
    db.close();
  }
}
