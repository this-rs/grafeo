//! Transaction example: commit, rollback, and savepoints.
//!
//! Run with: `cargo run -p obrain-examples --bin transactions`

use obrain::ObrainDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();

    // ── Act 1: Committed transaction ──────────────────────────────
    // Begin a transaction. All changes are isolated until commit.
    // Note: begin_transaction() requires &mut self.
    session.begin_transaction()?;

    // Create several people inside the transaction
    session.execute("INSERT (:Person {name: 'Alix', city: 'Utrecht'})")?;
    session.execute("INSERT (:Person {name: 'Gus', city: 'Leiden'})")?;
    session.execute(
        "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
         INSERT (a)-[:KNOWS]->(b)",
    )?;

    // Before commit: data is visible within this transaction
    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("Inside transaction (before commit): {count} people");

    // Commit makes the changes permanent
    session.commit()?;

    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("After commit: {count} people");

    // ── Act 2: Rolled-back transaction ────────────────────────────
    // Start another transaction, but this time undo everything.
    session.begin_transaction()?;

    // Insert more data that we'll discard
    session.execute("INSERT (:Person {name: 'Vincent', city: 'Paris'})")?;
    session.execute("INSERT (:Person {name: 'Jules', city: 'Berlin'})")?;

    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("\nInside transaction (before rollback): {count} people");

    // Rollback discards all changes in this transaction
    session.rollback()?;

    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("After rollback: {count} people (unchanged)");

    // ── Act 3: Savepoints for partial rollback ────────────────────
    // Savepoints let you undo part of a transaction while keeping
    // the rest.
    session.begin_transaction()?;

    // Step 1: create a person (will be kept)
    session.execute("INSERT (:Person {name: 'Mia', city: 'Barcelona'})")?;

    // Mark this point so we can come back to it
    session.savepoint("after_mia")?;

    // Step 2: create another person (will be undone)
    session.execute("INSERT (:Person {name: 'Butch', city: 'Prague'})")?;

    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("\nInside transaction (both Mia and Butch): {count} people");

    // Undo step 2, but keep step 1
    session.rollback_to_savepoint("after_mia")?;

    // Commit: only Mia's node persists
    session.commit()?;

    // Verify: Mia exists, Butch does not
    let mia: i64 = session
        .execute("MATCH (p:Person {name: 'Mia'}) RETURN COUNT(p)")?
        .scalar()?;
    let butch: i64 = session
        .execute("MATCH (p:Person {name: 'Butch'}) RETURN COUNT(p)")?
        .scalar()?;

    println!("After savepoint rollback and commit:");
    println!("  Mia exists:   {}", mia == 1);
    println!("  Butch exists: {}", butch == 1);

    // Final count should be 3: Alix, Gus (act 1) + Mia (act 3)
    let total: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("\nTotal people: {total}");

    println!("\nDone!");
    Ok(())
}
