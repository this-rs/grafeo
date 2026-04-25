//! Criterion bench — activation spreading / engram recall / cognitive kernels.
//!
//! Wired in T11 (CoactivationMap) + T12 (Hopfield recall).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_activation_stub(c: &mut Criterion) {
    let mut g = c.benchmark_group("activation_stub");
    g.bench_function("noop", |b| {
        b.iter(|| black_box(1u64));
    });
    g.finish();
}

criterion_group!(benches, bench_activation_stub);
criterion_main!(benches);
