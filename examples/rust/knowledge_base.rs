//! Knowledge Base — Cognitive Graph Database Example
//!
//! Demonstrates synapse reinforcement between concepts, multi-signal search
//! combining graph structure with relevance scoring, and consolidation of
//! stale nodes.
//!
//! Run with: `cargo run -p obrain-examples --bin knowledge_base`

use obrain::{ObrainDB, NodeId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Knowledge Base — Cognitive Graph Example ===\n");

    let db = ObrainDB::new_in_memory();
    let session = db.session();

    // ── Build knowledge graph ─────────────────────────────────────
    // Concepts with energy (activation) and access_count (mutation frequency).
    // In a cognitive graph, these are tracked automatically by the
    // EnergyStore and FabricStore via reactive MutationListeners.

    let concepts = [
        ("GraphDB", 0.90, 42),
        ("Cypher", 0.85, 38),
        ("GQL", 0.80, 35),
        ("SPARQL", 0.50, 15),
        ("RDF", 0.45, 12),
        ("PropertyGraph", 0.75, 28),
        ("Vector Search", 0.70, 25),
        ("PageRank", 0.60, 20),
        ("Community Detection", 0.55, 18),
        ("Neural Network", 0.40, 10),
        ("Embedding", 0.65, 22),
        ("ACID", 0.30, 8),
        ("B-Tree", 0.20, 5),
        ("WAL", 0.15, 3),
        ("Linked Data", 0.10, 2),
    ];

    for (name, energy, access_count) in &concepts {
        session.execute(&format!(
            "INSERT (:Concept {{name: '{name}', energy: {energy}, access_count: {access_count}}})"
        ))?;
    }

    // Relationships between concepts with synaptic weights.
    // In a cognitive graph, synapses form via Hebbian learning:
    // when two concepts are co-accessed, the edge strengthens.
    // Weight decays exponentially: W(t) = W₀ × 2^(-Δt / half_life)
    let relations = [
        ("GraphDB", "Cypher", "USES", 0.9),
        ("GraphDB", "GQL", "USES", 0.85),
        ("GraphDB", "SPARQL", "USES", 0.5),
        ("GraphDB", "PropertyGraph", "IMPLEMENTS", 0.8),
        ("GraphDB", "RDF", "IMPLEMENTS", 0.4),
        ("GraphDB", "ACID", "REQUIRES", 0.6),
        ("GraphDB", "Vector Search", "INTEGRATES", 0.7),
        ("PropertyGraph", "Cypher", "QUERIED_BY", 0.85),
        ("PropertyGraph", "GQL", "QUERIED_BY", 0.80),
        ("RDF", "SPARQL", "QUERIED_BY", 0.9),
        ("RDF", "Linked Data", "ENABLES", 0.5),
        ("Vector Search", "Embedding", "USES", 0.8),
        ("Vector Search", "Neural Network", "POWERED_BY", 0.6),
        ("PageRank", "GraphDB", "RUNS_ON", 0.7),
        ("Community Detection", "GraphDB", "RUNS_ON", 0.65),
        ("ACID", "WAL", "USES", 0.7),
        ("ACID", "B-Tree", "USES", 0.5),
    ];

    for (from, to, rel_type, weight) in &relations {
        session.execute(&format!(
            "MATCH (a:Concept {{name: '{from}'}}), (b:Concept {{name: '{to}'}})
             INSERT (a)-[:{rel_type} {{weight: {weight}}}]->(b)"
        ))?;
    }

    println!(
        "Created knowledge graph: {} concepts, {} relations\n",
        concepts.len(),
        relations.len()
    );

    // ── Synapse Reinforcement ─────────────────────────────────────
    // Simulate co-access: when a user reads about GraphDB and then
    // Cypher, the synapse between them strengthens.
    // In production, SynapseListener detects co-activation in
    // mutation batches and calls synapse_store.reinforce() automatically.

    println!("── Synapse Reinforcement ──");
    println!("  Simulating co-access pattern: GraphDB → Cypher → GQL\n");

    let co_accessed = [("GraphDB", "Cypher"), ("Cypher", "GQL"), ("GraphDB", "GQL")];

    for (a, b) in &co_accessed {
        // Read current weight
        let result = session.execute(&format!(
            "MATCH (a:Concept {{name: '{a}'}})-[r]->(b:Concept {{name: '{b}'}})
             RETURN r.weight"
        ))?;

        if let Some(row) = result.iter().next() {
            let old_weight = row[0].as_float64().unwrap_or(0.0);
            // Reinforce: weight += reinforce_amount (clamped to 1.0)
            let new_weight = (old_weight + 0.05).min(1.0);
            session.execute(&format!(
                "MATCH (a:Concept {{name: '{a}'}})-[r]->(b:Concept {{name: '{b}'}})
                 SET r.weight = {new_weight}"
            ))?;
            println!("  {a} → {b}: {old_weight:.2} → {new_weight:.2} (reinforced)");
        }
    }

    // ── Multi-Signal Search ───────────────────────────────────────
    // Combine multiple signals to find the most relevant concepts:
    //   1. Graph structure (PageRank) — structural importance
    //   2. Energy level — recency of access
    //   3. Synapse weight — strength of connections to the query topic
    //
    // In a cognitive graph, the Fabric provides a composite risk_score
    // and the UDF obrain.energy() gives real-time activation levels.

    println!("\n── Multi-Signal Search: 'query languages' ──");

    // Signal 1: PageRank for structural importance
    let pr_result = session.execute("CALL obrain.pagerank({damping: 0.85, max_iterations: 20})")?;
    let pageranks: Vec<(i64, f64)> = pr_result
        .iter()
        .map(|row| {
            (
                row[0].as_int64().unwrap_or(0),
                row[1].as_float64().unwrap_or(0.0),
            )
        })
        .collect();

    // Signal 2: Direct connections to query-related concepts
    let query_concepts = ["Cypher", "GQL", "SPARQL"];

    let result = session.execute(
        "MATCH (c:Concept)
         RETURN c.name, c.energy, c.access_count
         ORDER BY c.energy DESC",
    )?;

    println!("  Ranking by: 0.4×energy + 0.3×pagerank + 0.3×relevance\n");
    println!(
        "  {:<22} {:<8} {:<10} {:<10} Combined",
        "Concept", "Energy", "PageRank", "Relevance"
    );
    println!("  {}", "-".repeat(65));

    let mut search_results: Vec<(String, f64)> = Vec::new();

    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?").to_string();
        let energy = row[1].as_float64().unwrap_or(0.0);

        // Find pagerank for this concept
        let pr = find_score_by_name(&db, &pageranks, &name);

        // Compute relevance: is this concept related to query languages?
        let relevance = if query_concepts.contains(&name.as_str()) {
            1.0
        } else {
            // Check if connected to any query concept
            let mut rel: f64 = 0.0;
            for qc in &query_concepts {
                let connected = session.execute(&format!(
                    "MATCH (a:Concept {{name: '{name}'}})-[r]-(b:Concept {{name: '{qc}'}})
                     RETURN r.weight"
                ))?;
                if let Some(r) = connected.iter().next() {
                    rel = rel.max(r[0].as_float64().unwrap_or(0.0));
                }
            }
            rel
        };

        // Composite score
        let combined = 0.4 * energy + 0.3 * (pr * 10.0).min(1.0) + 0.3 * relevance;
        search_results.push((name.clone(), combined));

        println!(
            "  {:<22} {:<8.2} {:<10.4} {:<10.2} {:.3}",
            name, energy, pr, relevance, combined
        );
    }

    search_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n  Top results for 'query languages':");
    for (name, score) in search_results.iter().take(5) {
        println!("    {:<22} score={:.3}", name, score);
    }

    // ── Consolidation of Stale Nodes ──────────────────────────────
    // In a cognitive graph, the MemoryManager identifies nodes with
    // energy below a threshold and marks them for consolidation
    // (archival or pruning). The Fabric's staleness metric drives this.

    println!("\n── Stale Node Consolidation ──");
    let stale_threshold = 0.25;

    let stale = session.execute(&format!(
        "MATCH (c:Concept)
         WHERE c.energy < {stale_threshold}
         RETURN c.name, c.energy, c.access_count
         ORDER BY c.energy ASC"
    ))?;

    println!("  Nodes below energy threshold ({stale_threshold}):");
    println!(
        "  {:<20} {:<8} {:<8} Action",
        "Concept", "Energy", "Accesses"
    );
    println!("  {}", "-".repeat(50));

    for row in stale.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let energy = row[1].as_float64().unwrap_or(0.0);
        let accesses = row[2].as_int64().unwrap_or(0);
        let action = if accesses < 5 {
            "ARCHIVE (rarely used)"
        } else {
            "KEEP (historically active)"
        };
        println!("  {:<20} {:<8.2} {:<8} {}", name, energy, accesses, action);
    }

    // ── Knowledge Clusters ────────────────────────────────────────
    // Louvain community detection reveals natural topic clusters.

    println!("\n── Knowledge Clusters (Louvain) ──");
    let result = session.execute("CALL obrain.louvain()")?;

    let mut clusters: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();

    for row in result.iter() {
        let node_id = row[0].as_int64().unwrap_or(0);
        let community = row[1].as_int64().unwrap_or(0);
        let name = get_concept_name(&db, node_id);
        clusters.entry(community).or_default().push(name);
    }

    for (cluster_id, members) in &clusters {
        println!("  Cluster {}: {}", cluster_id, members.join(", "));
    }

    println!("\nDone!");
    Ok(())
}

/// Look up a concept's name by raw node ID.
fn get_concept_name(db: &ObrainDB, raw_id: i64) -> String {
    let node_id = NodeId::from(raw_id as u64);
    db.get_node(node_id)
        .and_then(|n| {
            n.get_property("name")
                .and_then(|v| v.as_str().map(String::from))
        })
        .unwrap_or_else(|| "?".to_string())
}

/// Find the PageRank score for a concept by name.
fn find_score_by_name(db: &ObrainDB, scores: &[(i64, f64)], name: &str) -> f64 {
    scores
        .iter()
        .find(|(nid, _)| get_concept_name(db, *nid) == name)
        .map(|(_, s)| *s)
        .unwrap_or(0.0)
}
