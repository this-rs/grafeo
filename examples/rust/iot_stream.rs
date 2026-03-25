//! IoT / Event Stream — Cognitive Graph Database Example
//!
//! Demonstrates co-change detection on sensor events and stagnation
//! alerting on inactive sensors using a cognitive graph model.
//!
//! Run with: `cargo run -p grafeo-examples --bin iot_stream`

use grafeo::{GrafeoDB, NodeId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== IoT Event Stream — Cognitive Graph Example ===\n");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // ── Build sensor network ──────────────────────────────────────
    // Sensors with energy (last activity), a zone, and reading count.
    // In a cognitive graph, energy decays automatically — sensors that
    // stop reporting will naturally drop to low energy levels.

    let sensors = [
        ("temp_floor1", "Floor1", "temperature", 0.95, 150),
        ("temp_floor2", "Floor2", "temperature", 0.90, 140),
        ("humid_floor1", "Floor1", "humidity", 0.85, 130),
        ("humid_floor2", "Floor2", "humidity", 0.80, 120),
        ("motion_lobby", "Lobby", "motion", 0.70, 80),
        ("motion_floor1", "Floor1", "motion", 0.60, 60),
        ("co2_floor1", "Floor1", "co2", 0.50, 40),
        ("co2_floor2", "Floor2", "co2", 0.45, 35),
        ("smoke_floor1", "Floor1", "smoke", 0.10, 5), // Almost dormant
        ("smoke_floor2", "Floor2", "smoke", 0.08, 3), // Almost dormant
        ("power_main", "Utility", "power", 0.92, 200),
        ("water_main", "Utility", "water", 0.05, 1), // Stagnant
    ];

    for (id, zone, sensor_type, energy, readings) in &sensors {
        session.execute(&format!(
            "INSERT (:Sensor {{sensor_id: '{id}', zone: '{zone}', type: '{sensor_type}', energy: {energy}, readings: {readings}}})"
        ))?;
    }

    // Zone nodes
    for zone in &["Floor1", "Floor2", "Lobby", "Utility"] {
        session.execute(&format!("INSERT (:Zone {{name: '{zone}'}})"))?;
    }

    // Sensor-to-zone relationships
    for (id, zone, _, _, _) in &sensors {
        session.execute(&format!(
            "MATCH (s:Sensor {{sensor_id: '{id}'}}), (z:Zone {{name: '{zone}'}})
             INSERT (s)-[:IN_ZONE]->(z)"
        ))?;
    }

    println!(
        "Created sensor network: {} sensors, 4 zones\n",
        sensors.len()
    );

    // ── Event Stream Simulation ───────────────────────────────────
    // Simulate a batch of sensor events. In a cognitive graph, each
    // mutation triggers reactive listeners that update energy, detect
    // co-changes, and refresh fabric scores.

    println!("── Event Stream (batch of readings) ──");

    let events = [
        ("temp_floor1", 22.5),
        ("humid_floor1", 45.0),
        ("co2_floor1", 800.0),
        ("temp_floor1", 22.7), // temp and humid co-change
        ("humid_floor1", 46.0),
        ("temp_floor2", 21.0),
        ("humid_floor2", 43.0),
        ("motion_lobby", 1.0),
        ("power_main", 230.5),
    ];

    for (sensor_id, value) in &events {
        // Simulate energy boost on each reading
        session.execute(&format!(
            "MATCH (s:Sensor {{sensor_id: '{sensor_id}'}})
             SET s.energy = CASE WHEN s.energy + 0.02 > 1.0 THEN 1.0 ELSE s.energy + 0.02 END,
                 s.readings = s.readings + 1,
                 s.last_value = {value}"
        ))?;
    }

    println!(
        "  Processed {} events. Energy boosted for active sensors.\n",
        events.len()
    );

    // ── Co-Change Detection ───────────────────────────────────────
    // Sensors that change together within a time window form co-change
    // relationships. In a cognitive graph, CoChangeDetector tracks
    // this automatically via the MutationListener.
    //
    // Here we detect sensors that were updated in the same batch.

    println!("── Co-Change Detection ──");
    println!("  Sensors that frequently change together:\n");

    // Build co-change edges based on same-zone, same-batch patterns
    let co_change_pairs = [
        ("temp_floor1", "humid_floor1", 15), // Temperature and humidity correlate
        ("temp_floor2", "humid_floor2", 12),
        ("temp_floor1", "co2_floor1", 8), // Temperature and CO2 in same zone
        ("motion_lobby", "power_main", 5), // Motion triggers power usage
    ];

    for (a, b, frequency) in &co_change_pairs {
        session.execute(&format!(
            "MATCH (a:Sensor {{sensor_id: '{a}'}}), (b:Sensor {{sensor_id: '{b}'}})
             INSERT (a)-[:CO_CHANGES_WITH {{frequency: {frequency}}}]->(b)"
        ))?;
        println!("  {a} ↔ {b} (co-change frequency: {frequency})");
    }

    // ── Stagnation Alerting ───────────────────────────────────────
    // Identify sensors with low energy (inactive) — these are stagnation
    // candidates. In a cognitive graph, StagnationDetector computes a
    // composite score per community:
    //   stagnation = 0.4×(1-energy) + 0.3×age + 0.3×(1-synapse_activity)

    println!("\n── Stagnation Alert ──");
    let stagnation_threshold = 0.20;

    let stagnant = session.execute(&format!(
        "MATCH (s:Sensor)
         WHERE s.energy < {stagnation_threshold}
         RETURN s.sensor_id, s.zone, s.type, s.energy, s.readings
         ORDER BY s.energy ASC"
    ))?;

    if stagnant.iter().next().is_some() {
        println!(
            "  ALERT: {} sensors below energy threshold ({stagnation_threshold}):\n",
            stagnant.iter().count()
        );
        println!(
            "  {:<18} {:<10} {:<12} {:<8} {:<8} Status",
            "Sensor", "Zone", "Type", "Energy", "Reads"
        );
        println!("  {}", "-".repeat(65));

        for row in stagnant.iter() {
            let id = row[0].as_str().unwrap_or("?");
            let zone = row[1].as_str().unwrap_or("?");
            let stype = row[2].as_str().unwrap_or("?");
            let energy = row[3].as_float64().unwrap_or(0.0);
            let readings = row[4].as_int64().unwrap_or(0);
            let status = if energy < 0.05 { "CRITICAL" } else { "WARNING" };
            println!(
                "  {:<18} {:<10} {:<12} {:<8.2} {:<8} {}",
                id, zone, stype, energy, readings, status
            );
        }
    }

    // ── Zone Health Analysis ──────────────────────────────────────
    // Aggregate sensor energy per zone to identify unhealthy zones.

    println!("\n── Zone Health Analysis ──");

    let zones = session.execute(
        "MATCH (s:Sensor)-[:IN_ZONE]->(z:Zone)
         RETURN z.name, COUNT(s), AVG(s.energy)
         ORDER BY AVG(s.energy) DESC",
    )?;

    println!(
        "  {:<12} {:<10} {:<12} Status",
        "Zone", "Sensors", "Avg Energy"
    );
    println!("  {}", "-".repeat(45));

    for row in zones.iter() {
        let zone = row[0].as_str().unwrap_or("?");
        let count = row[1].as_int64().unwrap_or(0);
        let avg_energy = row[2].as_float64().unwrap_or(0.0);
        let status = match avg_energy {
            e if e >= 0.7 => "HEALTHY",
            e if e >= 0.4 => "DEGRADED",
            _ => "CRITICAL",
        };
        println!(
            "  {:<12} {:<10} {:<12.2} {}",
            zone, count, avg_energy, status
        );
    }

    // ── Network Topology: Bridge Sensors ──────────────────────────
    // Sensors with high betweenness centrality are critical — if they
    // fail, monitoring coverage is disrupted.

    println!("\n── Critical Sensors (Betweenness Centrality) ──");

    let result = session.execute("CALL grafeo.betweenness_centrality()")?;
    let mut centrality: Vec<_> = result
        .iter()
        .map(|row| {
            let node_id = row[0].as_int64().unwrap_or(0);
            let score = row[1].as_float64().unwrap_or(0.0);
            (node_id, score)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();
    centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if centrality.is_empty() {
        println!("  No critical bridge sensors detected (healthy topology).");
    } else {
        for (node_id, score) in centrality.iter().take(5) {
            let name = get_sensor_id(&db, *node_id);
            let criticality = if *score > 0.2 { "HIGH" } else { "MODERATE" };
            println!("  {:<18} betweenness={:.4}  [{}]", name, score, criticality);
        }
    }

    // ── Community Detection (co-change clusters) ──────────────────
    // Louvain reveals groups of sensors that form natural monitoring
    // clusters — useful for automated alert correlation.

    println!("\n── Sensor Communities (Louvain) ──");
    let result = session.execute("CALL grafeo.louvain()")?;

    let mut communities: std::collections::HashMap<i64, Vec<String>> =
        std::collections::HashMap::new();

    for row in result.iter() {
        let node_id = row[0].as_int64().unwrap_or(0);
        let community = row[1].as_int64().unwrap_or(0);
        let name = get_sensor_id(&db, node_id);
        // Only include sensor nodes (skip zone nodes)
        if !name.starts_with('?')
            && !["Floor1", "Floor2", "Lobby", "Utility"].contains(&name.as_str())
        {
            communities.entry(community).or_default().push(name);
        }
    }

    for (community_id, members) in &communities {
        if !members.is_empty() {
            println!("  Cluster {}: {}", community_id, members.join(", "));
        }
    }

    // ── Composite Risk Score ──────────────────────────────────────
    // Combine energy, centrality, and stagnation into a risk score.
    // In production, the Fabric computes this automatically:
    //   risk = w₁×pagerank + w₂×mutation_freq + w₃×annotation_gap
    //        + w₄×betweenness + w₅×scar_intensity

    println!("\n── Risk Assessment ──");

    let all_sensors = session.execute(
        "MATCH (s:Sensor)
         RETURN s.sensor_id, s.energy, s.readings
         ORDER BY s.energy ASC",
    )?;

    println!(
        "  {:<18} {:<8} {:<8} {:<8} Action",
        "Sensor", "Energy", "Reads", "Risk"
    );
    println!("  {}", "-".repeat(55));

    for row in all_sensors.iter() {
        let id = row[0].as_str().unwrap_or("?");
        let energy = row[1].as_float64().unwrap_or(0.0);
        let readings = row[2].as_int64().unwrap_or(0);

        // Simplified risk: high risk = low energy + low readings
        let energy_risk = 1.0 - energy;
        let activity_risk = 1.0 - ((readings as f64) / 200.0).min(1.0);
        let risk = 0.6 * energy_risk + 0.4 * activity_risk;

        let action = match risk {
            r if r >= 0.8 => "REPLACE sensor",
            r if r >= 0.6 => "INVESTIGATE",
            r if r >= 0.4 => "MONITOR",
            _ => "OK",
        };

        println!(
            "  {:<18} {:<8.2} {:<8} {:<8.2} {}",
            id, energy, readings, risk, action
        );
    }

    println!("\nDone!");
    Ok(())
}

/// Look up a sensor's ID by raw node ID from CALL procedure results.
fn get_sensor_id(db: &GrafeoDB, raw_id: i64) -> String {
    let node_id = NodeId::from(raw_id as u64);
    db.get_node(node_id)
        .and_then(|n| {
            n.get_property("sensor_id")
                .and_then(|v| v.as_str().map(String::from))
                .or_else(|| {
                    n.get_property("name")
                        .and_then(|v| v.as_str().map(String::from))
                })
        })
        .unwrap_or_else(|| "?".to_string())
}
