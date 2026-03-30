//! Generate a test graph for B5-bis benchmark.
//! Uses GQL sessions to ensure WAL persistence.

use obrain::ObrainDB;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/obrain-b5bis-graph".to_string());

    let _ = std::fs::remove_dir_all(&path);

    let db = ObrainDB::open(&path)?;
    let mut session = db.session();

    // === People ===
    let people = vec![
        (
            "Thomas Rivière",
            "développeur Rust senior",
            "Lyon",
            "34",
            "escalade et photographie",
            "Mozilla",
            "Rust",
            "Arch Linux",
        ),
        (
            "Sophie Martin",
            "data scientist",
            "Paris",
            "29",
            "piano et randonnée",
            "Datadog",
            "Python",
            "macOS",
        ),
        (
            "Marc Dupont",
            "architecte logiciel",
            "Lyon",
            "41",
            "cuisine et vélo",
            "OVHcloud",
            "Go",
            "Ubuntu",
        ),
        (
            "Alice Chen",
            "chercheuse en IA",
            "Grenoble",
            "31",
            "dessin et yoga",
            "INRIA",
            "Python",
            "Ubuntu",
        ),
        (
            "Pierre Bernard",
            "DevOps engineer",
            "Toulouse",
            "37",
            "running et jeux vidéo",
            "Airbus",
            "Bash",
            "CentOS",
        ),
    ];

    for (name, prof, city, age, hobby, company, lang, os) in &people {
        let q = format!(
            "INSERT (:Person {{name: '{name}', profession: '{prof}', ville: '{city}', age: '{age}', hobby: '{hobby}', entreprise: '{company}', langage_favori: '{lang}', os: '{os}'}})"
        );
        session.execute(&q)?;
        eprintln!("  Person: {name}");
    }

    // === Projects ===
    let projects = vec![
        (
            "Obrain",
            "Moteur de graphe embarqué en Rust",
            "Rust",
            "open-source",
        ),
        (
            "Grafeo",
            "Base de données graphe native",
            "Rust",
            "open-source",
        ),
        (
            "DataPipeline",
            "ETL pipeline temps réel",
            "Python",
            "interne Datadog",
        ),
        (
            "CloudInfra",
            "Infrastructure cloud hybride",
            "Go",
            "interne OVHcloud",
        ),
        (
            "NeuralSearch",
            "Recherche neurale multimodale",
            "Python",
            "recherche INRIA",
        ),
        (
            "SkyNet-CI",
            "CI/CD pour systèmes embarqués avioniques",
            "Bash/Python",
            "interne Airbus",
        ),
    ];

    for (name, desc, lang, scope) in &projects {
        let q = format!(
            "INSERT (:Project {{name: '{name}', description: '{desc}', language: '{lang}', scope: '{scope}'}})"
        );
        session.execute(&q)?;
        eprintln!("  Project: {name}");
    }

    // === Cities ===
    let cities = vec![
        ("Lyon", "Rhône-Alpes, France", "500000"),
        ("Paris", "Île-de-France, France", "2100000"),
        ("Grenoble", "Isère, France", "160000"),
        ("Toulouse", "Occitanie, France", "470000"),
    ];

    for (name, region, pop) in &cities {
        let q =
            format!("INSERT (:City {{name: '{name}', region: '{region}', population: '{pop}'}})");
        session.execute(&q)?;
        eprintln!("  City: {name}");
    }

    // === Technologies ===
    let techs = vec![
        ("Rust", "Langage système performant et memory-safe"),
        (
            "Python",
            "Langage polyvalent pour la data science et le scripting",
        ),
        ("Go", "Langage compilé pour les microservices et le cloud"),
        ("GGML", "Framework de tenseurs pour inférence LLM"),
        ("PyTorch", "Framework apprentissage profond"),
        ("Kubernetes", "Orchestrateur de conteneurs"),
        ("llama.cpp", "Inférence LLM en C++ optimisée"),
        ("GraphQL", "Langage de requête pour APIs"),
    ];

    for (name, desc) in &techs {
        let q = format!("INSERT (:Technology {{name: '{name}', description: '{desc}'}})");
        session.execute(&q)?;
        eprintln!("  Tech: {name}");
    }

    // === Events ===
    let events = vec![
        (
            "RustConf 2025",
            "Lyon",
            "2025-09-15",
            "Conférence annuelle Rust",
        ),
        (
            "PyData Paris 2025",
            "Paris",
            "2025-06-20",
            "Data science en Python",
        ),
        (
            "KubeCon EU 2025",
            "Paris",
            "2025-03-18",
            "Cloud native summit",
        ),
        ("ICML 2025", "Grenoble", "2025-07-10", "Machine learning"),
    ];

    for (name, city, date, desc) in &events {
        let q = format!(
            "INSERT (:Event {{name: '{name}', city: '{city}', date: '{date}', description: '{desc}'}})"
        );
        session.execute(&q)?;
        eprintln!("  Event: {name}");
    }

    // === Edges ===
    // LIVES_IN
    for (person, city) in &[
        ("Thomas Rivière", "Lyon"),
        ("Sophie Martin", "Paris"),
        ("Marc Dupont", "Lyon"),
        ("Alice Chen", "Grenoble"),
        ("Pierre Bernard", "Toulouse"),
    ] {
        let q = format!(
            "MATCH (p:Person {{name: '{person}'}}), (c:City {{name: '{city}'}}) INSERT (p)-[:LIVES_IN]->(c)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: LIVES_IN");

    // WORKS_ON
    for (person, proj) in &[
        ("Thomas Rivière", "Obrain"),
        ("Thomas Rivière", "Grafeo"),
        ("Sophie Martin", "DataPipeline"),
        ("Marc Dupont", "CloudInfra"),
        ("Alice Chen", "NeuralSearch"),
        ("Pierre Bernard", "SkyNet-CI"),
    ] {
        let q = format!(
            "MATCH (p:Person {{name: '{person}'}}), (pr:Project {{name: '{proj}'}}) INSERT (p)-[:WORKS_ON]->(pr)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: WORKS_ON");

    // USES
    for (person, tech) in &[
        ("Thomas Rivière", "Rust"),
        ("Thomas Rivière", "GGML"),
        ("Thomas Rivière", "llama.cpp"),
        ("Sophie Martin", "Python"),
        ("Sophie Martin", "PyTorch"),
        ("Marc Dupont", "Go"),
        ("Marc Dupont", "Kubernetes"),
        ("Alice Chen", "Python"),
        ("Alice Chen", "PyTorch"),
        ("Pierre Bernard", "Kubernetes"),
    ] {
        let q = format!(
            "MATCH (p:Person {{name: '{person}'}}), (t:Technology {{name: '{tech}'}}) INSERT (p)-[:USES]->(t)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: USES");

    // KNOWS (bidirectional)
    for (a, b) in &[
        ("Thomas Rivière", "Marc Dupont"),
        ("Thomas Rivière", "Alice Chen"),
        ("Sophie Martin", "Alice Chen"),
        ("Marc Dupont", "Pierre Bernard"),
    ] {
        let q = format!(
            "MATCH (a:Person {{name: '{a}'}}), (b:Person {{name: '{b}'}}) INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: KNOWS");

    // ATTENDED
    for (person, event) in &[
        ("Thomas Rivière", "RustConf 2025"),
        ("Marc Dupont", "RustConf 2025"),
        ("Sophie Martin", "PyData Paris 2025"),
        ("Alice Chen", "ICML 2025"),
        ("Pierre Bernard", "KubeCon EU 2025"),
        ("Marc Dupont", "KubeCon EU 2025"),
    ] {
        let q = format!(
            "MATCH (p:Person {{name: '{person}'}}), (e:Event {{name: '{event}'}}) INSERT (p)-[:ATTENDED]->(e)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: ATTENDED");

    // BUILT_WITH
    for (proj, tech) in &[
        ("Obrain", "Rust"),
        ("Obrain", "GGML"),
        ("Obrain", "llama.cpp"),
        ("Grafeo", "Rust"),
        ("DataPipeline", "Python"),
        ("DataPipeline", "PyTorch"),
        ("CloudInfra", "Go"),
        ("CloudInfra", "Kubernetes"),
        ("NeuralSearch", "Python"),
        ("NeuralSearch", "PyTorch"),
        ("SkyNet-CI", "Kubernetes"),
    ] {
        let q = format!(
            "MATCH (pr:Project {{name: '{proj}'}}), (t:Technology {{name: '{tech}'}}) INSERT (pr)-[:BUILT_WITH]->(t)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: BUILT_WITH");

    // HELD_IN
    for (event, city) in &[
        ("RustConf 2025", "Lyon"),
        ("PyData Paris 2025", "Paris"),
        ("KubeCon EU 2025", "Paris"),
        ("ICML 2025", "Grenoble"),
    ] {
        let q = format!(
            "MATCH (e:Event {{name: '{event}'}}), (c:City {{name: '{city}'}}) INSERT (e)-[:HELD_IN]->(c)"
        );
        session.execute(&q)?;
    }
    eprintln!("  Edges: HELD_IN");

    // Verify
    let r = session.execute("MATCH (n) RETURN count(n)")?;
    eprintln!("\n=== Graph created at {} ===", path);
    eprintln!("  Query result: {:?}", r);

    let store = db.store();
    eprintln!(
        "  Store: {} nodes, {} edges",
        store.node_count(),
        store.edge_count()
    );

    drop(session);
    drop(db);
    eprintln!("  Database closed");

    // Verify reopen
    let db2 = ObrainDB::open(&path)?;
    let store2 = db2.store();
    eprintln!(
        "  Reopened: {} nodes, {} edges",
        store2.node_count(),
        store2.edge_count()
    );
    drop(db2);

    Ok(())
}
