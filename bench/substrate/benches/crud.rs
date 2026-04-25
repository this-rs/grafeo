//! Criterion bench — CRUD (create / read / update / delete) latencies.
//!
//! Wired in T3 (WAL roundtrip) and T6 (GraphStore trait impl).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_crud_stub(c: &mut Criterion) {
    let mut g = c.benchmark_group("crud_stub");
    g.bench_function("noop", |b| {
        b.iter(|| black_box(0u64));
    });
    g.finish();
}

criterion_group!(benches, bench_crud_stub);
criterion_main!(benches);
