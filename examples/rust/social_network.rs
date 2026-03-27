//! Social Network — Cognitive Graph Database Example
//!
//! Demonstrates energy decay on interactions, community detection via Louvain,
//! and spreading activation for influence measurement in a social graph.
//!
//! Run with: `cargo run -p obrain-examples --bin social_network`

use obrain::{NodeId, ObrainDB};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Social Network — Cognitive Graph Example ===\n");

    let db = ObrainDB::new_in_memory();
    let session = db.session();

    // ── Build social graph ────────────────────────────────────────
    // Users with an `energy` property representing activation level.
    // In a cognitive graph DB, energy decays exponentially over time:
    //   E(t) = E₀ × 2^(-Δt / half_life)
    // Here we simulate snapshots at different activity levels.

    let users = [
        ("Alix", 0.95),      // Very active — just interacted
        ("Gus", 0.80),       // Active recently
        ("Mia", 0.60),       // Moderate activity
        ("Jules", 0.30),     // Starting to go stale
        ("Vincent", 0.15),   // Low energy — hasn't interacted in a while
        ("Butch", 0.05),     // Nearly dormant
        ("Esme", 0.70),      // Active in a different community
        ("Zed", 0.55),       // Moderate, same community as Esme
        ("Marsellus", 0.40), // Bridge between communities
    ];

    for (name, energy) in &users {
        session.execute(&format!(
            "INSERT (:User {{name: '{name}', energy: {energy}}})"
        ))?;
    }

    // Interactions with a `weight` representing frequency/strength.
    // In a cognitive graph, these are Hebbian synapses: edges that
    // strengthen when both endpoints are co-activated.
    let interactions = [
        // Community 1: Alix-Gus-Mia-Jules tight cluster
        ("Alix", "Gus", 5),
        ("Alix", "Mia", 3),
        ("Gus", "Mia", 4),
        ("Gus", "Jules", 2),
        ("Mia", "Jules", 1),
        // Community 2: Esme-Zed cluster
        ("Esme", "Zed", 6),
        // Bridge: Marsellus connects communities
        ("Marsellus", "Jules", 2),
        ("Marsellus", "Esme", 3),
        // Weak ties
        ("Vincent", "Jules", 1),
        ("Butch", "Marsellus", 1),
    ];

    for (from, to, weight) in &interactions {
        session.execute(&format!(
            "MATCH (a:User {{name: '{from}'}}), (b:User {{name: '{to}'}})
             INSERT (a)-[:INTERACTS {{weight: {weight}}}]->(b)"
        ))?;
    }

    println!(
        "Created social graph: {} users, {} interactions\n",
        users.len(),
        interactions.len()
    );

    // ── Energy Decay Simulation ───────────────────────────────────
    // Show how energy levels naturally separate active from stale users.
    // In production, Obrain's EnergyStore handles this automatically
    // via exponential decay with configurable half-life.

    println!("── Energy Levels (activation state) ──");
    println!("{:<12} {:<8} Status", "User", "Energy");
    println!("{}", "-".repeat(35));

    let result = session.execute(
        "MATCH (u:User)
         RETURN u.name, u.energy
         ORDER BY u.energy DESC",
    )?;

    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let energy = row[1].as_float64().unwrap_or(0.0);
        let status = match energy {
            e if e >= 0.7 => "active",
            e if e >= 0.3 => "cooling",
            e if e >= 0.1 => "stale",
            _ => "dormant",
        };
        println!("  {:<12} {:<8.2} {}", name, energy, status);
    }

    // ── Energy Boost on Interaction ───────────────────────────────
    // When a user interacts, their energy is boosted. Simulate this
    // by updating Vincent's energy after an interaction.

    println!("\n── Energy Boost (Vincent interacts) ──");
    let before: f64 = session
        .execute("MATCH (u:User {name: 'Vincent'}) RETURN u.energy")?
        .scalar()?;

    // Simulate boost: energy += boost_amount, clamped to max
    let boost = 0.3;
    let new_energy = (before + boost).min(1.0);
    session.execute(&format!(
        "MATCH (u:User {{name: 'Vincent'}})
         SET u.energy = {new_energy}"
    ))?;

    let after: f64 = session
        .execute("MATCH (u:User {name: 'Vincent'}) RETURN u.energy")?
        .scalar()?;

    println!(
        "  Vincent: {:.2} → {:.2} (boosted by {:.2})",
        before, after, boost
    );

    // ── Community Detection (Louvain) ─────────────────────────────
    // Louvain optimizes modularity to discover communities.
    // In a cognitive graph, communities feed into the Fabric score
    // and enable stagnation detection per community.

    println!("\n── Community Detection (Louvain) ──");

    let result = session.execute("CALL obrain.louvain()")?;

    // Group nodes by community
    let mut communities: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();

    for row in result.iter() {
        let node_id = row[0].as_int64().unwrap_or(0);
        let community = row[1].as_int64().unwrap_or(0);
        let name = get_user_name(&db, node_id);
        communities.entry(community).or_default().push(name);
    }

    for (community_id, members) in &communities {
        println!("  Community {}: {}", community_id, members.join(", "));
    }

    // ── Spreading Activation for Influence ────────────────────────
    // Starting from a seed node, energy propagates through the graph
    // via weighted edges (synapses). Each hop attenuates the signal.
    // This reveals which nodes are "reachable" and how strongly.

    println!("\n── Spreading Activation (seed: Alix) ──");
    println!("  Simulating energy propagation through synaptic connections...\n");

    // Use PageRank as a proxy for influence — it captures how
    // energy flows through the network structure.
    let result = session.execute("CALL obrain.pagerank({damping: 0.85, max_iterations: 20})")?;

    let mut scores: Vec<_> = result
        .iter()
        .map(|row| {
            let node_id = row[0].as_int64().unwrap_or(0);
            let score = row[1].as_float64().unwrap_or(0.0);
            (node_id, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  {:<12} {:<10} Influence", "User", "PageRank");
    println!("  {}", "-".repeat(38));
    for (node_id, score) in &scores {
        let name = get_user_name(&db, *node_id);
        let influence = match *score {
            s if s >= 0.15 => "high",
            s if s >= 0.10 => "medium",
            _ => "low",
        };
        println!("  {:<12} {:<10.4} {}", name, score, influence);
    }

    // ── Betweenness Centrality (bridge detection) ─────────────────
    // Identifies bridge nodes — users who connect otherwise separate
    // communities. In a cognitive graph, high-betweenness nodes get
    // higher risk scores because their failure disconnects the graph.

    println!("\n── Bridge Detection (Betweenness Centrality) ──");
    let result = session.execute("CALL obrain.betweenness_centrality()")?;

    let mut bridges: Vec<_> = result
        .iter()
        .map(|row| {
            let node_id = row[0].as_int64().unwrap_or(0);
            let score = row[1].as_float64().unwrap_or(0.0);
            (node_id, score)
        })
        .collect();
    bridges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (node_id, score) in bridges.iter().take(5) {
        let name = get_user_name(&db, *node_id);
        let role = if *score > 0.1 { "BRIDGE" } else { "leaf" };
        println!("  {:<12} betweenness={:.4}  [{}]", name, score, role);
    }

    // ── Composite: Energy × Influence ─────────────────────────────
    // In a cognitive graph, the Fabric combines energy, PageRank,
    // betweenness, and scars into a single risk score per node.
    // Here we manually compute a simplified version.

    println!("\n── Fabric Score (energy × influence) ──");

    let result = session.execute(
        "MATCH (u:User)
         RETURN u.name, u.energy
         ORDER BY u.name",
    )?;

    let mut fabric_scores: Vec<(String, f64)> = Vec::new();
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?").to_string();
        let energy = row[1].as_float64().unwrap_or(0.0);
        // Find this user's PageRank
        let pagerank = scores
            .iter()
            .find(|(nid, _)| get_user_name(&db, *nid) == name)
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        // Simplified fabric score: energy × normalized_pagerank
        let fabric = energy * (pagerank * 10.0).min(1.0);
        fabric_scores.push((name, fabric));
    }

    fabric_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  {:<12} {:<8} Verdict", "User", "Score");
    println!("  {}", "-".repeat(35));
    for (name, score) in &fabric_scores {
        let verdict = match *score {
            s if s >= 0.5 => "active + important",
            s if s >= 0.2 => "moderate",
            s if s >= 0.05 => "fading",
            _ => "consolidation candidate",
        };
        println!("  {:<12} {:<8.3} {}", name, score, verdict);
    }

    println!("\nDone!");
    Ok(())
}

/// Look up a user's name by their raw node ID from CALL procedure results.
fn get_user_name(db: &ObrainDB, raw_id: i64) -> String {
    let node_id = NodeId::from(raw_id as u64);
    db.get_node(node_id)
        .and_then(|n| {
            n.get_property("name")
                .and_then(|v| v.as_str().map(String::from))
        })
        .unwrap_or_else(|| "?".to_string())
}
