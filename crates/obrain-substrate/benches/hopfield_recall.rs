//! T7 Step 3 acceptance bench — Hopfield recall on 10⁶ nodes.
//!
//! Verifies that the bitset-filtered recall pipeline meets the ≤ 500 µs
//! budget for a column of 10⁶ nodes on reference hardware (Apple
//! M-series / modern x86_64 with POPCNT).
//!
//! Run with:
//! ```bash
//! cargo bench -p obrain-substrate --bench hopfield_recall
//! ```
//!
//! The bench is deliberately in-memory — it exercises the raw
//! [`scan_overlap`] and [`top_k_by_overlap`] kernels on a heap-allocated
//! `Vec<u64>`, without the `ZoneFile` mmap plumbing. That isolates the
//! kernel throughput from I/O effects; the `Writer::hopfield_recall`
//! path (covered in a smaller bench below) is dominated by the same
//! kernel once the page cache is warm.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use obrain_substrate::{scan_overlap, scan_overlap_scalar, top_k_by_overlap};

/// Deterministic xorshift64 — no `rand` dependency at bench time.
struct Xorshift64(u64);
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn build_column(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Xorshift64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

fn bench_scan_overlap(c: &mut Criterion) {
    let n = 1_000_000;
    let column = build_column(n, 0xA5A5_5A5A_DEAD_BEEF);
    let query: u64 = 0xFEED_FACE_CAFE_BABE;

    // Reusable output buffer — allocated once, re-used per iteration.
    let mut out = vec![0u32; n];

    let mut g = c.benchmark_group("hopfield_recall_1M");
    g.throughput(Throughput::Bytes((n * 8) as u64));

    g.bench_function("scan_overlap_dispatched", |b| {
        b.iter(|| {
            scan_overlap(black_box(&column), black_box(query), black_box(&mut out));
        });
    });

    g.bench_function("scan_overlap_scalar", |b| {
        b.iter(|| {
            scan_overlap_scalar(black_box(&column), black_box(query), black_box(&mut out));
        });
    });

    g.bench_function("top_k_32_fused", |b| {
        b.iter(|| {
            let r = top_k_by_overlap(black_box(&column), black_box(query), 32);
            black_box(r);
        });
    });

    g.bench_function("top_k_128_fused", |b| {
        b.iter(|| {
            let r = top_k_by_overlap(black_box(&column), black_box(query), 128);
            black_box(r);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_scan_overlap);
criterion_main!(benches);
