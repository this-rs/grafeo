//! Benchmark: MutationBus publish overhead when there are no subscribers.
//!
//! Acceptance criterion: < 5µs per mutation with zero subscribers.

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_reactive::{MutationBatch, MutationBus, MutationEvent, NodeSnapshot};
use smallvec::smallvec;

fn make_node_event(id: u64) -> MutationEvent {
    MutationEvent::NodeCreated {
        node: NodeSnapshot {
            id: NodeId::new(id),
            labels: smallvec![arcstr::literal!("Person")],
            properties: vec![],
        },
    }
}

fn bench_publish_no_subscribers(c: &mut Criterion) {
    let bus = MutationBus::new();

    c.bench_function("publish_single_no_sub", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });

    c.bench_function("publish_batch_10_no_sub", |b| {
        b.iter(|| {
            let events: Vec<_> = (0..10).map(make_node_event).collect();
            bus.publish_batch(MutationBatch::new(events));
        });
    });
}

fn bench_publish_with_subscriber(c: &mut Criterion) {
    let bus = MutationBus::new();
    let _rx = bus.subscribe(); // One subscriber, never read

    c.bench_function("publish_single_1_sub", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });
}

criterion_group!(
    benches,
    bench_publish_no_subscribers,
    bench_publish_with_subscriber
);
criterion_main!(benches);
