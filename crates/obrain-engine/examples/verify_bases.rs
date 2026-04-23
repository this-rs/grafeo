use obrain_engine::ObrainDB;
use obrain_core::graph::GraphStore;

fn main() {
    let bases = ["/Users/triviere/.obrain/db/po", "/Users/triviere/.obrain/db/megalaw", "/Users/triviere/.obrain/db/wikipedia"];
    for path in bases.iter() {
        print!("Opening {path}... ");
        match ObrainDB::open(path) {
            Ok(db) => {
                let store = db.store();
                let n = store.node_count();
                let e = store.edge_count();
                println!("OK nodes={n} edges={e}");
            }
            Err(err) => println!("FAIL: {err}"),
        }
    }
}
