//! GQL smoke test — end-to-end query execution on substrate backend
//! (T17 cutover validation).

use obrain_engine::ObrainDB;

fn main() {
    println!("[gql] Building in-memory DB (substrate tempfile)...");
    let db = ObrainDB::new_in_memory();
    let session = db.session();

    let r1 = session
        .execute("INSERT (:Person {name: 'Alice', age: 30})")
        .unwrap();
    println!(
        "[gql] INSERT 1: {} rows scanned",
        r1.rows_scanned.unwrap_or(0)
    );

    let r2 = session
        .execute("INSERT (:Person {name: 'Bob', age: 25})")
        .unwrap();
    println!(
        "[gql] INSERT 2: {} rows scanned",
        r2.rows_scanned.unwrap_or(0)
    );

    let r3 = session.execute("MATCH (p:Person) RETURN p").unwrap();
    println!("[gql] MATCH all Person: {} rows", r3.rows.len());

    let r4 = session
        .execute("MATCH (p:Person) WHERE p.age > 28 RETURN p.name")
        .unwrap();
    println!("[gql] Filtered (age > 28): {} rows", r4.rows.len());
    for row in &r4.rows {
        println!("[gql]   row = {:?}", row);
    }

    println!("[gql] ALL QUERIES PASSED ✓");
}
