//! T17j T5 — check chain↔rec.src coherence on PO / Wikipedia / Megalaw.
//! Read-only (no repair) — just diagnoses whether the bug is
//! universal or specific to PO.
//!
//! ```bash
//! cargo test -p obrain-engine --release --features cypher \
//!   --test t17j_t5_check_all_bases -- --nocapture
//! ```

#![cfg(feature = "cypher")]

use obrain_core::graph::GraphStore;
use obrain_substrate::SubstrateStore;

fn check(name: &str, path: std::path::PathBuf) {
    if !path.exists() {
        eprintln!("⏭  {name}: not present");
        return;
    }
    let subs = SubstrateStore::open(&path).expect("open");
    println!("\n═══ {name} ═══");
    for label in ["File", "Function", "Article", "Decision", "Concept"] {
        let nodes = subs.nodes_by_label(label);
        if nodes.is_empty() {
            continue;
        }
        let sample = nodes.len().min(50);
        let mut walked = 0u64;
        let mut mismatches = 0u64;
        for nid in nodes.iter().take(sample) {
            let (w, _m, mm, _) = subs.diagnose_chain_vs_rec_src(*nid);
            walked += w;
            mismatches += mm;
        }
        let pct = if walked > 0 {
            (mismatches as f64 / walked as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  :{:<12}  sample={:<3}  walked={:>6}  mismatches={:>6}  ({:.1}% broken)",
            label, sample, walked, mismatches, pct
        );
    }
}

#[test]
fn check_chain_consistency_three_corpora() {
    let home = std::env::var("HOME").unwrap_or_default();
    check(
        "PO (repaired)",
        std::path::PathBuf::from(&home).join(".obrain/db/po"),
    );
    check(
        "Wikipedia",
        std::path::PathBuf::from(&home).join(".obrain/db/wikipedia"),
    );
    check(
        "Megalaw",
        std::path::PathBuf::from(&home).join(".obrain/db/megalaw"),
    );
}
