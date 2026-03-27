**Obrain** — The graph database that learns. A cognitive graph database with a Rust core: data forms **engrams** with decaying energy, connections become **Hebbian synapses** that strengthen through co-activation, and **spreading activation** surfaces insights across the topology.

Built on 7 biomimetic layers, Obrain supports both **Labeled Property Graph (LPG)** and **RDF** data models, with zero required C dependencies. Optional allocators (jemalloc/mimalloc) and TLS use C libraries for performance.

## Cognitive Features

- **Engrams**: Nodes carry activation energy that decays exponentially — the graph remembers what matters
- **Hebbian Synapses**: Edges strengthen with co-activation, compete, and self-prune
- **Spreading Activation**: Query one node, discover related hot zones via energy propagation
- **Scars**: Failed transactions leave persistent memory that influences risk scoring
- **Knowledge Fabric**: Continuous risk, staleness, and centrality scoring for every node
- **RAG-native**: HNSW vector search + BM25 full-text + hybrid retrieval with cognitive scoring

## Graph Foundation

- **Dual data model support**: LPG and RDF with optimized storage for each
- **Multi-language queries**: GQL, Cypher, Gremlin, GraphQL, SPARQL and SQL/PGQ
- Embeddable with zero external dependencies
- Multi-language bindings: Python (PyO3), Node.js (napi-rs), Go (CGO), C (FFI), C# (.NET P/Invoke), Dart (dart:ffi), WebAssembly (wasm-bindgen)
- In-memory and persistent storage modes
- MVCC transactions with snapshot isolation

## Query Language & Data Model Support

| Query Language | LPG | RDF | Status |
|----------------|-----|-----|--------|
| GQL (ISO/IEC 39075) | ✅ | - | Default |
| Cypher (openCypher 9.0) | ✅ | - | Default |
| Gremlin (Apache TinkerPop) | ✅ | - | Default |
| GraphQL | ✅ | ✅ | Default |
| SPARQL (W3C 1.1) | - | ✅ | Default |
| SQL/PGQ (SQL:2023) | ✅ | - | Default |

Obrain uses a modular translator architecture where query languages are parsed into ASTs, then translated to a unified logical plan that executes against the appropriate storage backend (LPG or RDF).

## Installation

```bash
cargo add obrain
```

By default, the `embedded` profile is enabled: GQL, AI features (vector/text/hybrid search, CDC), graph algorithms and parallel execution. Customize with feature groups:

```bash
cargo add obrain                                             # Default (embedded profile)
cargo add obrain --no-default-features --features full       # All languages + AI + storage + RDF
cargo add obrain --no-default-features --features languages  # All languages, no AI
cargo add obrain --no-default-features --features gql,ai     # GQL + AI features
cargo add obrain --no-default-features --features gql        # Minimal: GQL only
cargo add obrain --features embed                            # Add ONNX embeddings (opt-in)
```

## Quick Start

```rust
use obrain::ObrainDB;

fn main() -> Result<(), obrain_common::utils::error::Error> {
    let db = ObrainDB::new_in_memory();
    let mut session = db.session();

    // Create nodes
    session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    session.execute("INSERT (:Person {name: 'Gus', age: 25})")?;

    // Query
    let result = session.execute("MATCH (p:Person) RETURN p.name, p.age")?;
    for row in result.rows {
        println!("{:?}", row);
    }

    Ok(())
}
```

## License

Apache-2.0
