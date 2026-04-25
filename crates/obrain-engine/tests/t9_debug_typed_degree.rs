//! T17h T9c debug — compare slow-path count vs T8 typed_degree per
//! node to find where the divergence comes from.

#![cfg(feature = "cypher")]

use obrain_core::graph::GraphStore;
use obrain_engine::ObrainDB;

#[test]
fn debug_total_imports_file_degree() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        return;
    }
    let db = ObrainDB::open(&po_path).expect("open PO db");
    let session = db.session();

    // 1. Total IMPORTS edges in the graph.
    let r1 = session
        .execute_cypher("MATCH ()-[r:IMPORTS]->() RETURN count(r) AS total")
        .unwrap();
    println!("Total IMPORTS edges : {:?}", r1.rows);

    // 2. IMPORTS edges where the SOURCE is :File.
    let r2 = session
        .execute_cypher("MATCH (f:File)-[r:IMPORTS]->() RETURN count(r) AS from_file")
        .unwrap();
    println!("IMPORTS edges sourced on :File : {:?}", r2.rows);

    // 3. IMPORTS edges where the TARGET is :File.
    let r3 = session
        .execute_cypher("MATCH ()-[r:IMPORTS]->(f:File) RETURN count(r) AS to_file")
        .unwrap();
    println!("IMPORTS edges targeting :File  : {:?}", r3.rows);

    // 4. Max out_degree IMPORTS per File (slow path via COUNT).
    let r4 = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported) \
             WITH f, count(DISTINCT imported) AS c \
             RETURN max(c) AS max_out",
        )
        .unwrap();
    println!(
        "Max OPTIONAL MATCH count(DISTINCT imported) for :File : {:?}",
        r4.rows
    );

    // 5. Sum out_degree IMPORTS across all Files (slow path total).
    let r5 = session
        .execute_cypher(
            "MATCH (f:File) \
             OPTIONAL MATCH (f)-[:IMPORTS]->(imported) \
             WITH f, count(DISTINCT imported) AS c \
             RETURN sum(c) AS sum_out",
        )
        .unwrap();
    println!("Sum over :File of count(DISTINCT imported) : {:?}", r5.rows);

    // 6. Sample a File node and ask the store directly for its
    // typed out_degree vs manual walk.
    let store = db.store();
    let files = store.nodes_by_label("File");
    println!(
        "Total File nodes via store.nodes_by_label : {}",
        files.len()
    );
    let mut deg_sum_direct = 0usize;
    let mut deg_sum_typed = 0usize;
    let mut max_direct = 0usize;
    let mut max_typed = 0usize;
    for f in files.iter().take(100) {
        let direct = store
            .edges_from(*f, obrain_core::graph::Direction::Outgoing)
            .into_iter()
            .filter(|(_, eid)| {
                store
                    .edge_type(*eid)
                    .map(|t| &*t == "IMPORTS")
                    .unwrap_or(false)
            })
            .count();
        let typed = store.out_degree_by_type(*f, Some("IMPORTS"));
        deg_sum_direct += direct;
        deg_sum_typed += typed;
        max_direct = max_direct.max(direct);
        max_typed = max_typed.max(typed);
        if direct != typed {
            println!("  NodeId({}) : direct={} typed={}", f.0, direct, typed);
        }
    }
    println!(
        "First 100 File nodes : direct sum={}, typed sum={}, max_direct={}, max_typed={}",
        deg_sum_direct, deg_sum_typed, max_direct, max_typed
    );
}
