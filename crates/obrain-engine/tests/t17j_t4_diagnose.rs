//! T17j T4 — diagnose IMPORTS histogram gap on PO using
//! `SubstrateStore::diagnose_edge_endpoints`. Classifies every live
//! IMPORTS edge by whether its src/dst NodeRecord exists + has a
//! non-zero label bitset.

#![cfg(feature = "cypher")]

use obrain_core::graph::GraphStore;
use obrain_substrate::SubstrateStore;

#[test]
fn diagnose_imports_endpoints_on_po() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        eprintln!("⏭ PO not present");
        return;
    }
    let subs = SubstrateStore::open(&po_path).expect("open");

    // T17j T5 critical diagnostic : walk chains of labelled nodes and
    // verify each edge's rec.src matches the chain owner.
    println!("\n=== Chain vs rec.src consistency (sample 50 nodes per label) ===");
    for label in ["File", "Function", "Impl", "Struct"] {
        let nodes = subs.nodes_by_label(label);
        let sample_n = nodes.len().min(50);
        let mut total_walked = 0u64;
        let mut total_matches = 0u64;
        let mut total_mismatches = 0u64;
        let mut first_mismatch_sample: Option<(u64, u64, u32)> = None;
        for nid in nodes.iter().take(sample_n) {
            let (w, m, mm, fm) = subs.diagnose_chain_vs_rec_src(*nid);
            total_walked += w;
            total_matches += m;
            total_mismatches += mm;
            if first_mismatch_sample.is_none()
                && let Some((eid, rsrc)) = fm
            {
                first_mismatch_sample = Some((nid.0, eid, rsrc));
            }
        }
        println!(
            "  label :{:<10}  sample={:<3}  walked={:>6}  matches={:>6}  mismatches={:>6}",
            label, sample_n, total_walked, total_matches, total_mismatches
        );
        if let Some((owner, eid, rsrc)) = first_mismatch_sample {
            println!(
                "    first mismatch : chain_owner=NodeId({owner})  edge_slot={eid}  rec.src=slot({rsrc})"
            );
        }
    }

    // T17j T5 diagnostic : check label_bitset consistency for key labels
    println!("\n=== Label bitset sanity (first 100 nodes per label) ===");
    let all_labels_list = subs.all_labels();
    // Sample 100K — enough to see ALL the variability (PO has ≤ 150K per label).
    for label in ["File", "Function", "Impl", "Struct", "Trait"] {
        let (n, union_bits) = subs
            .diagnose_label_bitsets_for(label, 200_000)
            .expect("diagnose label");
        // Decode union bits to names
        let names: Vec<_> = (0..64)
            .filter(|b| union_bits & (1u64 << b) != 0)
            .filter_map(|b| all_labels_list.get(b).cloned())
            .collect();
        println!(
            "  label :{:<10} → sampled={:<4}  union_bits={:#018x}  decoded={:?}",
            label, n, union_bits, names
        );
    }

    for et in ["IMPORTS", "CONTAINS", "CALLS", "IMPLEMENTS", "EXTENDS"] {
        let (total, svw, svn, sm, dvw, dvn, dm) =
            subs.diagnose_edge_endpoints(et).expect("diagnose");
        if total == 0 {
            continue;
        }
        println!(
            "\n{et} live edges = {total}\n  SRC : valid+labels={svw}  valid+NO_labels={svn}  missing={sm}\n  DST : valid+labels={dvw}  valid+NO_labels={dvn}  missing={dm}"
        );

        let (_, src_bits, dst_bits) = subs
            .diagnose_edge_endpoint_bits(et)
            .expect("diagnose bits");
        println!(
            "  union_src_bits = {:064b} ({:#018x})",
            src_bits, src_bits
        );
        println!(
            "  union_dst_bits = {:064b} ({:#018x})",
            dst_bits, dst_bits
        );
        // Decode bit positions to label names.
        let labels = subs.all_labels();
        let decode = |bits: u64| {
            (0..64)
                .filter(|b| bits & (1u64 << b) != 0)
                .filter_map(|b| labels.get(b).cloned())
                .collect::<Vec<_>>()
        };
        println!("  src_labels = {:?}", decode(src_bits));
        println!("  dst_labels = {:?}", decode(dst_bits));
    }
}

/// T17j T5 — apply repair to Wikipedia + Megalaw. Same as PO test
/// but larger (Wiki 119M edges / Megalaw 14.68M edges). MUTATES.
#[test]
#[ignore = "mutates Wiki + Megalaw — run manually with --include-ignored"]
fn repair_wiki_and_megalaw_outgoing_chains() {
    let home = std::env::var("HOME").unwrap_or_default();
    for (name, label) in [("wikipedia", "Article"), ("megalaw", "Concept")] {
        let path = std::path::PathBuf::from(&home).join(format!(".obrain/db/{name}"));
        if !path.exists() {
            eprintln!("⏭  {name} not present");
            continue;
        }
        println!("\n=== {name} ===");
        let subs = SubstrateStore::open(&path).expect("open");

        let nodes = subs.nodes_by_label(label);
        let mut walked_before = 0u64;
        let mut mismatches_before = 0u64;
        for nid in nodes.iter().take(50) {
            let (w, _, mm, _) = subs.diagnose_chain_vs_rec_src(*nid);
            walked_before += w;
            mismatches_before += mm;
        }
        println!(
            "  BEFORE : :{label} (50) walked={} mismatches={}",
            walked_before, mismatches_before
        );

        let t0 = std::time::Instant::now();
        let (reset, rewired) = subs.repair_outgoing_chains().expect("repair");
        println!(
            "  REPAIR : reset {} nodes + rewired {} edges in {:.2} s",
            reset,
            rewired,
            t0.elapsed().as_secs_f64()
        );

        let mut walked_after = 0u64;
        let mut mismatches_after = 0u64;
        for nid in nodes.iter().take(50) {
            let (w, _, mm, _) = subs.diagnose_chain_vs_rec_src(*nid);
            walked_after += w;
            mismatches_after += mm;
        }
        println!(
            "  AFTER  : :{label} (50) walked={} mismatches={}",
            walked_after, mismatches_after
        );
        assert_eq!(mismatches_after, 0, "{name} still inconsistent");
    }
}

/// T17j T5 — apply repair_outgoing_chains on PO, verify chains now
/// match rec.src. `#[ignore]` by default because it MUTATES PO —
/// run manually with `cargo test -- repair_po --include-ignored`.
#[test]
#[ignore = "mutates PO — run manually with --include-ignored"]
fn repair_po_outgoing_chains() {
    let home = std::env::var("HOME").unwrap_or_default();
    let po_path = std::path::PathBuf::from(&home).join(".obrain/db/po");
    if !po_path.exists() {
        return;
    }
    let subs = SubstrateStore::open(&po_path).expect("open");

    println!("\n=== T17j T5 : BEFORE repair ===");
    let nodes = subs.nodes_by_label("File");
    let mut walked_before = 0u64;
    let mut mismatches_before = 0u64;
    for nid in nodes.iter().take(50) {
        let (w, _m, mm, _) = subs.diagnose_chain_vs_rec_src(*nid);
        walked_before += w;
        mismatches_before += mm;
    }
    println!(
        "  File (50 sample) : walked={} mismatches={}",
        walked_before, mismatches_before
    );

    println!("\n=== Running repair_outgoing_chains ... ===");
    let t0 = std::time::Instant::now();
    let (reset, rewired) = subs.repair_outgoing_chains().expect("repair");
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "  reset {} node chain heads + rewired {} edges in {:.2} s",
        reset, rewired, elapsed
    );

    println!("\n=== AFTER repair ===");
    let mut walked_after = 0u64;
    let mut matches_after = 0u64;
    let mut mismatches_after = 0u64;
    for nid in nodes.iter().take(50) {
        let (w, m, mm, _) = subs.diagnose_chain_vs_rec_src(*nid);
        walked_after += w;
        matches_after += m;
        mismatches_after += mm;
    }
    println!(
        "  File (50 sample) : walked={} matches={} mismatches={}",
        walked_after, matches_after, mismatches_after
    );
    assert_eq!(mismatches_after, 0, "chains still inconsistent after repair");
}
