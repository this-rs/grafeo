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
