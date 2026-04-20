//! Throughput benchmarks for the SIMD-accelerated column decay.
//!
//! ## Acceptance criterion (T6 Step 7)
//!
//! `decay_all` on **10⁶ slots** must complete in ≤ 1 ms wall time on the
//! reference hardware (Apple M-series / modern x86_64). That budget is
//! roughly the DDR4/DDR5 streaming bandwidth bound for a 32 MB working set
//! — i.e. memory-bound, not compute-bound. The SIMD kernel in
//! `obrain_substrate::simd` keeps the arithmetic off the critical path so
//! the loop is free to run at peak store-bandwidth.
//!
//! Run with:
//!
//! ```bash
//! cargo bench -p obrain-substrate --bench decay_all
//! ```

use std::hint::black_box;

use bytemuck::Zeroable;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use obrain_substrate::{SubstrateFile, SyncMode, Writer};

fn seed_nodes(w: &Writer, n: u32) {
    use obrain_substrate::record::NodeRecord;
    // Fill every slot with a fixed energy and a null-label bitset. The
    // exact values don't matter for throughput; we just need `n` live slots.
    let mut rec = NodeRecord::zeroed();
    rec.energy = 0xC000; // 0.75 in Q1.15
    for i in 1..=n {
        w.write_node(i, rec).unwrap();
    }
    w.commit().unwrap();
}

fn seed_edges(w: &Writer, n: u64) {
    use obrain_substrate::record::{EdgeRecord, U48};
    let mut rec = EdgeRecord {
        src: 0,
        dst: 0,
        edge_type: 1,
        weight_u16: 0xC000,
        next_from: U48::from_u64(0),
        next_to: U48::from_u64(0),
        ricci_u8: 0,
        flags: 0,
        engram_tag: 0,
        _pad: [0; 4],
    };
    for i in 1..=n {
        rec.src = (i & 0xFFFF_FFFF) as u32;
        rec.dst = ((i >> 32) & 0xFFFF_FFFF) as u32;
        w.write_edge(i, rec).unwrap();
    }
    w.commit().unwrap();
}

fn bench_decay_energy(c: &mut Criterion) {
    let mut g = c.benchmark_group("decay_all_energy");

    for n in [10_000u32, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n as u64));
        let id = format!("nodes={}", n);
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        seed_nodes(&w, n);
        g.bench_function(id, |b| {
            b.iter(|| {
                // factor_q16=0xFFCC ≈ 0.9995 — a non-trivial multiplier.
                w.decay_all_energy(black_box(0xFFCC), black_box(n + 1))
                    .unwrap();
            });
        });
    }

    g.finish();
}

fn bench_decay_synapse(c: &mut Criterion) {
    let mut g = c.benchmark_group("decay_all_synapse");

    for n in [10_000u64, 100_000, 1_000_000] {
        g.throughput(Throughput::Elements(n));
        let id = format!("edges={}", n);
        let sub = SubstrateFile::open_tempfile().unwrap();
        let w = Writer::new(sub, SyncMode::Never).unwrap();
        seed_edges(&w, n);
        g.bench_function(id, |b| {
            b.iter(|| {
                w.decay_all_synapse(black_box(0xFFCC), black_box(n + 1))
                    .unwrap();
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_decay_energy, bench_decay_synapse);
criterion_main!(benches);
