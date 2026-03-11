import 'package:grafeo/grafeo.dart';

void main() async {
  print('Testing Grafeo Dart bindings...');

  // Test 1: Open an in-memory database
  print('1. Opening local storage');
  final db = await GrafeoDatabase.open("../../../../test_isolate");

  // query db [info](https://grafeo.dev/api/node/database/?h=schema#info)
  print(await db.execute('RETURN info() as db_info'));

  // query db [schema](https://grafeo.dev/api/node/database/?h=schema#schema)
  print(await db.execute('RETURN schema() as db_schema'));

  // create a empty schema to be switch.
  await db.execute('CREATE SCHEMA IF NOT EXISTS default');
  print('2. ✓ Created a empty schema to be switch.');

  // create a data container schema
  await db.execute('CREATE SCHEMA IF NOT EXISTS reporting');
  print('3. ✓ Created a data container schema.');
  // switch to data container
  await db.execute('SESSION  SET SCHEMA reporting');
  print('4. ✓ Switched to the data container schema.');
  await db.execute('''
    CREATE GRAPH TYPE IF NOT EXISTS social_network (
      NODE TYPE Person (name STRING NOT NULL, age INTEGER),
      EDGE TYPE KNOWS (since INTEGER)
  )
  ''');
  await db.execute('''
    CREATE GRAPH IF NOT EXISTS my_social TYPED social_network
  ''');
  await db.execute('SESSION SET GRAPH social');
  await db.execute("INSERT (:Person {name: 'Dave'})");
  await db.execute("INSERT (:Person {name: 'Eve'})");
  print('5. ✓ Load data into the data container schema.');

  print('6. Getting persons');
  var persons = await db.executeWithParams('MATCH (n:Person) RETURN n', {});
  print('✓ Persons: $persons');

  await db.execute('SESSION SET SCHEMA default');
  print('7. ✓ Switched to the empty schema.');
  print('8. Getting persons');
  persons = await db.executeWithParams('MATCH (n:Person) RETURN n', {});
  print('✓ Persons: $persons');

  // Test 6: Close the database
  print('6. Closing database...');
  await db.close();
  print('✓ Database closed successfully');

  print('All tests passed!');
}
