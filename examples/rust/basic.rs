//! Basic Obrain example: create a social graph and query it.
//!
//! Run with: `cargo run -p obrain-examples --bin basic`

use obrain::ObrainDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an in-memory database (no persistence, no setup required)
    let db = ObrainDB::new_in_memory();

    // Sessions are lightweight handles for executing queries.
    // You can create as many as you need.
    let session = db.session();

    // ── Insert nodes ──────────────────────────────────────────────
    // Each INSERT creates a node with a label and properties.
    // GQL uses the (:Label {key: value}) syntax for node patterns.
    session.execute("INSERT (:Person {name: 'Alix', age: 30, city: 'Utrecht'})")?;
    session.execute("INSERT (:Person {name: 'Gus', age: 28, city: 'Leiden'})")?;
    session.execute("INSERT (:Person {name: 'Vincent', age: 35, city: 'Paris'})")?;

    // ── Insert edges ──────────────────────────────────────────────
    // MATCH two nodes, then INSERT an edge between them.
    // Edges have a type (KNOWS) and can carry properties too.
    session.execute(
        "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
         INSERT (a)-[:KNOWS {since: 2020}]->(b)",
    )?;
    session.execute(
        "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'})
         INSERT (a)-[:KNOWS {since: 2022}]->(b)",
    )?;

    // ── Query: find all people ────────────────────────────────────
    // MATCH finds nodes matching the pattern, RETURN selects which
    // properties to include in the result.
    let result = session.execute(
        "MATCH (p:Person)
         RETURN p.name, p.age, p.city
         ORDER BY p.name",
    )?;

    println!("All people:");
    println!("{:<10} {:<5} City", "Name", "Age");
    println!("{}", "-".repeat(30));
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let age = row[1].as_int64().unwrap_or(0);
        let city = row[2].as_str().unwrap_or("?");
        println!("{:<10} {:<5} {}", name, age, city);
    }

    // ── Query: who does Gus know? ─────────────────────────────────
    // Edges are matched with -[:TYPE]-> syntax. The arrow indicates
    // direction: Gus KNOWS someone, not the other way around.
    let result = session.execute(
        "MATCH (a:Person {name: 'Gus'})-[:KNOWS]->(b:Person)
         RETURN b.name, b.city",
    )?;

    println!("\nGus knows:");
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let city = row[1].as_str().unwrap_or("?");
        println!("  {} ({})", name, city);
    }

    // ── Query: find people in Utrecht ───────────────────────────
    // WHERE adds a filter condition to the MATCH pattern.
    let result = session.execute(
        "MATCH (p:Person)
         WHERE p.city = 'Utrecht'
         RETURN p.name
         ORDER BY p.name",
    )?;

    println!("\nPeople in Utrecht:");
    for row in result.iter() {
        println!("  {}", row[0].as_str().unwrap_or("?"));
    }

    // ── Scalar query: count nodes ─────────────────────────────────
    // scalar() extracts a single value from a single-row, single-column result.
    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("\nTotal people: {count}");

    // ── Query: friends-of-friends ─────────────────────────────────
    // Multi-hop traversal: find people reachable through two KNOWS edges.
    let result = session.execute(
        "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->(c)
         RETURN a.name AS from_person, c.name AS to_person",
    )?;

    println!("\nFriends of friends of Alix:");
    for row in result.iter() {
        let from = row[0].as_str().unwrap_or("?");
        let to = row[1].as_str().unwrap_or("?");
        println!("  {} -> {}", from, to);
    }

    println!("\nDone!");
    Ok(())
}
