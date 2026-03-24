//! Benchmarks for the cognitive subsystem overhead.
//!
//! Measures the per-mutation cost of cognitive features (energy tracking, synapse
//! reinforcement) to ensure the reactive pipeline stays within acceptable bounds.
//!
//! Run with: cargo bench -p grafeo-cognitive --features cognitive

use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use grafeo_common::types::NodeId;
use grafeo_reactive::{
    BatchConfig, MutationBatch, MutationBus, MutationEvent, NodeSnapshot, Scheduler,
};
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

// ============================================================================
// Scheduler dispatch benchmarks
// ============================================================================

/// Measures scheduler dispatch throughput: publish events through the full
/// bus → scheduler → listener pipeline.
fn bench_scheduler_dispatch(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Scheduler with a no-op listener that counts events
    let bus = MutationBus::new();
    let config = BatchConfig::new(1000, Duration::from_secs(60));
    let scheduler = Scheduler::new(&bus, config);

    struct CountingListener {
        count: std::sync::atomic::AtomicU64,
    }

    #[async_trait::async_trait]
    impl grafeo_reactive::MutationListener for CountingListener {
        fn name(&self) -> &str {
            "counter"
        }
        async fn on_event(&self, _event: &MutationEvent) {}
        async fn on_batch(&self, events: &[MutationEvent]) {
            self.count
                .fetch_add(events.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    scheduler.register_listener(Arc::new(CountingListener {
        count: std::sync::atomic::AtomicU64::new(0),
    }));

    c.bench_function("scheduler_dispatch_single", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });

    c.bench_function("scheduler_dispatch_batch_100", |b| {
        b.iter(|| {
            let events: Vec<_> = (0..100).map(make_node_event).collect();
            bus.publish_batch(MutationBatch::new(events));
        });
    });

    // Flush remaining events
    rt.block_on(async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        scheduler.shutdown().await;
    });
}

// ============================================================================
// Energy subsystem benchmarks
// ============================================================================

#[cfg(feature = "energy")]
fn bench_energy_tracking(c: &mut Criterion) {
    use grafeo_cognitive::energy::{EnergyConfig, EnergyListener, EnergyStore};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(50));
    let scheduler = Scheduler::new(&bus, config);

    let energy_store = Arc::new(EnergyStore::new(EnergyConfig::default()));
    let listener = Arc::new(EnergyListener::new(Arc::clone(&energy_store)));
    scheduler.register_listener(listener);

    c.bench_function("energy_track_single_mutation", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });

    // Measure energy decay computation (cold path, no events)
    c.bench_function("energy_decay_lookup_1000_nodes", |b| {
        // Pre-populate the store with 1000 nodes
        rt.block_on(async {
            let events: Vec<_> = (0..1000).map(make_node_event).collect();
            bus.publish_batch(MutationBatch::new(events));
            tokio::time::sleep(Duration::from_millis(200)).await;
        });

        b.iter(|| {
            for id in 0..1000u64 {
                std::hint::black_box(energy_store.get_energy(NodeId::new(id)));
            }
        });
    });

    rt.block_on(scheduler.shutdown());
}

// ============================================================================
// Synapse subsystem benchmarks
// ============================================================================

#[cfg(feature = "synapse")]
fn bench_synapse_reinforcement(c: &mut Criterion) {
    use grafeo_cognitive::synapse::{SynapseConfig, SynapseListener, SynapseStore};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(50));
    let scheduler = Scheduler::new(&bus, config);

    let synapse_store = Arc::new(SynapseStore::new(SynapseConfig::default()));
    let listener = Arc::new(SynapseListener::new(Arc::clone(&synapse_store)));
    scheduler.register_listener(listener);

    c.bench_function("synapse_reinforce_single_mutation", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });

    rt.block_on(scheduler.shutdown());
}

// ============================================================================
// Full cognitive pipeline benchmark (energy + synapse combined)
// ============================================================================

fn bench_full_cognitive_pipeline(c: &mut Criterion) {
    use grafeo_cognitive::{CognitiveConfig, CognitiveEngineBuilder};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let bus = MutationBus::new();
    let config = BatchConfig::new(100, Duration::from_millis(50));
    let scheduler = Scheduler::new(&bus, config);

    // Build the full cognitive engine (registers all enabled listeners)
    let cognitive_config = CognitiveConfig::default(); // energy + synapse
    let _engine = CognitiveEngineBuilder::from_config(&cognitive_config).build(&scheduler);

    c.bench_function("cognitive_pipeline_single_mutation", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            bus.publish(make_node_event(i));
        });
    });

    c.bench_function("cognitive_pipeline_batch_100", |b| {
        b.iter(|| {
            let events: Vec<_> = (0..100).map(make_node_event).collect();
            bus.publish_batch(MutationBatch::new(events));
        });
    });

    rt.block_on(async {
        tokio::time::sleep(Duration::from_millis(200)).await;
        scheduler.shutdown().await;
    });
}

// ============================================================================
// Lazy scheduler overhead benchmark (no listeners = zero cost)
// ============================================================================

fn bench_scheduler_lazy_no_listeners(c: &mut Criterion) {
    c.bench_function("scheduler_create_no_listener", |b| {
        b.iter(|| {
            let bus = MutationBus::new();
            let config = BatchConfig::new(100, Duration::from_millis(50));
            let scheduler = Scheduler::new(&bus, config);
            std::hint::black_box(&scheduler);
            // Scheduler should NOT spawn a task (lazy)
            assert!(!scheduler.is_running());
        });
    });

    c.bench_function("publish_with_lazy_scheduler_no_listener", |b| {
        let bus = MutationBus::new();
        let config = BatchConfig::new(100, Duration::from_millis(50));
        let _scheduler = Scheduler::new(&bus, config);

        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            // Should be zero-cost: no subscriber on the bus
            bus.publish(make_node_event(i));
        });
    });
}

criterion_group!(
    scheduler_benches,
    bench_scheduler_dispatch,
    bench_scheduler_lazy_no_listeners,
);

#[cfg(feature = "energy")]
criterion_group!(energy_benches, bench_energy_tracking);

#[cfg(feature = "synapse")]
criterion_group!(synapse_benches, bench_synapse_reinforcement);

criterion_group!(pipeline_benches, bench_full_cognitive_pipeline,);

#[cfg(all(feature = "energy", feature = "synapse"))]
criterion_main!(
    scheduler_benches,
    energy_benches,
    synapse_benches,
    pipeline_benches
);

#[cfg(all(feature = "energy", not(feature = "synapse")))]
criterion_main!(scheduler_benches, energy_benches, pipeline_benches);

#[cfg(all(not(feature = "energy"), feature = "synapse"))]
criterion_main!(scheduler_benches, synapse_benches, pipeline_benches);

#[cfg(not(any(feature = "energy", feature = "synapse")))]
criterion_main!(scheduler_benches, pipeline_benches);
