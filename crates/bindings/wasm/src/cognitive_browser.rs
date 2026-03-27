//! Cognitive graph database in the browser — WASM bindings.
//!
//! Demonstrates creating a graph, applying cognitive features (energy boost,
//! synapse creation, spreading activation search), and displaying results
//! via `console.log`.
//!
//! Enabled by the `cognitive` feature flag on `obrain-wasm`.

use obrain_cognitive::energy::{EnergyConfig, EnergyStore};
use obrain_cognitive::search::{SearchConfig, SearchPipeline, SearchWeights};
use obrain_cognitive::synapse::{SynapseConfig, SynapseStore};
use obrain_common::types::Value;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Helper macro that logs to browser console.
macro_rules! console_log {
    ($($t:tt)*) => (log(&format!($($t)*)))
}

/// Run a cognitive graph demo in the browser.
///
/// Creates a small knowledge graph, boosts energy on nodes, creates synapses
/// between co-accessed nodes, and performs a cognitive search — all running
/// in WebAssembly.
#[wasm_bindgen]
pub fn cognitive_demo() -> Result<JsValue, JsError> {
    crate::utils::set_panic_hook();

    console_log!("=== Obrain Cognitive Browser Demo ===");
    console_log!("");

    // --- 1. Create an in-memory graph ---
    let db = obrain_engine::ObrainDB::new_in_memory();

    console_log!("[1/5] Creating knowledge graph...");

    let alice = db.store().create_node(&["Person"]);
    db.store()
        .set_node_property(alice, "name", Value::String("Alice".into()));

    let bob = db.store().create_node(&["Person"]);
    db.store()
        .set_node_property(bob, "name", Value::String("Bob".into()));

    let rust_lang = db.store().create_node(&["Topic"]);
    db.store()
        .set_node_property(rust_lang, "name", Value::String("Rust".into()));

    let wasm_topic = db.store().create_node(&["Topic"]);
    db.store()
        .set_node_property(wasm_topic, "name", Value::String("WebAssembly".into()));

    let obrain_topic = db.store().create_node(&["Topic"]);
    db.store()
        .set_node_property(obrain_topic, "name", Value::String("Obrain".into()));

    db.store().create_edge(alice, rust_lang, "KNOWS");
    db.store().create_edge(alice, wasm_topic, "KNOWS");
    db.store().create_edge(bob, rust_lang, "KNOWS");
    db.store().create_edge(bob, obrain_topic, "KNOWS");
    db.store().create_edge(rust_lang, wasm_topic, "RELATED_TO");
    db.store()
        .create_edge(wasm_topic, obrain_topic, "RELATED_TO");

    console_log!(
        "  Created {} nodes and {} edges",
        db.store().node_count(),
        db.store().edge_count()
    );

    // --- 2. Initialize cognitive stores ---
    console_log!("[2/5] Initializing cognitive stores (energy + synapses)...");

    let energy_store = EnergyStore::new(EnergyConfig::default());
    let synapse_store = SynapseStore::new(SynapseConfig::default());

    // --- 3. Boost energy on accessed nodes ---
    console_log!("[3/5] Simulating node access (energy boosts)...");

    // Simulate accessing Alice's topics (boost takes node_id + amount)
    energy_store.boost(alice, 1.0);
    energy_store.boost(rust_lang, 1.0);
    energy_store.boost(rust_lang, 1.0); // double access
    energy_store.boost(wasm_topic, 1.0);
    energy_store.boost(obrain_topic, 1.0);

    for (name, nid) in [
        ("Alice", alice),
        ("Bob", bob),
        ("Rust", rust_lang),
        ("WebAssembly", wasm_topic),
        ("Obrain", obrain_topic),
    ] {
        let energy = energy_store.get_energy(nid);
        console_log!("  {name}: energy = {energy:.3}");
    }

    // --- 4. Create synapses between co-accessed nodes ---
    console_log!("[4/5] Creating Hebbian synapses (co-activation)...");

    // Simulate co-access patterns (reinforce takes source, target, amount)
    synapse_store.reinforce(alice, rust_lang, 1.0);
    synapse_store.reinforce(alice, wasm_topic, 1.0);
    synapse_store.reinforce(rust_lang, wasm_topic, 1.0);
    synapse_store.reinforce(bob, rust_lang, 1.0);
    synapse_store.reinforce(bob, obrain_topic, 1.0);

    console_log!("  Created {} synapses", synapse_store.len());

    for (a_name, a_id, b_name, b_id) in [
        ("Alice", alice, "Rust", rust_lang),
        ("Alice", alice, "WebAssembly", wasm_topic),
        ("Rust", rust_lang, "WebAssembly", wasm_topic),
        ("Bob", bob, "Obrain", obrain_topic),
    ] {
        if let Some(syn) = synapse_store.get_synapse(a_id, b_id) {
            console_log!(
                "  {} <-> {}: weight = {:.3}",
                a_name,
                b_name,
                syn.current_weight()
            );
        }
    }

    // --- 5. Cognitive search ---
    console_log!("[5/5] Running cognitive search pipeline...");

    let pipeline = SearchPipeline::new();
    let config = SearchConfig {
        weights: SearchWeights::new(1.0, 0.5, 0.3, 0.0),
        limit: 5,
        ..Default::default()
    };

    let candidates = vec![
        (alice, 0.9),
        (bob, 0.3),
        (rust_lang, 0.8),
        (wasm_topic, 0.7),
        (obrain_topic, 0.6),
    ];

    let results = pipeline.search(&candidates, &config);

    console_log!("  Search results (top {}):", config.limit);
    for (i, result) in results.iter().enumerate() {
        let name = db
            .store()
            .get_node_property(result.node_id, &"name".into())
            .map_or_else(|| "?".to_string(), |v| v.to_string());
        console_log!("    {}. {} (score: {:.3})", i + 1, name, result.score);
    }

    console_log!("");
    console_log!("=== Demo complete! ===");

    Ok(JsValue::from_str("cognitive_demo completed successfully"))
}
