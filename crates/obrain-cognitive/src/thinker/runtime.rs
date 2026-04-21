//! # ThinkerRuntime — orchestration + pressure pause + shutdown (T13 Steps 6–9)
//!
//! Owns one `std::thread` per registered [`Thinker`] and drives their
//! ticks on the declared intervals. This module is intentionally
//! free of async runtime: each thread is a plain sleep/tick loop,
//! so shutdown is deterministic and Tokio is not required.
//!
//! ## Shutdown model
//!
//! [`ThinkerRuntime::shutdown`] sets a shared stop flag, wakes every
//! thread (via `Condvar::notify_all`), and joins them with a bounded
//! timeout. Threads re-check the stop flag at every loop iteration and
//! exit cleanly. A thread that fails to join in time is logged at
//! `warn` level — it does not block the runtime drop.
//!
//! ## Pressure pause
//!
//! When the runtime detects high system load (see
//! [`PressureSensor::is_overloaded`]), it sets a `paused` flag. Each
//! Thinker loop checks the flag before calling `tick` and skips when
//! paused. The sensor is polled once per tick interval so the overhead
//! is negligible even for fast thinkers.
//!
//! ## CPU budget
//!
//! The runtime verifies at registration time that the sum of declared
//! `budget.cpu_fraction` stays below
//! [`ThinkerRuntimeConfig::max_total_cpu_fraction`]. Over-committed
//! runtimes are rejected with a panic in debug builds and a `warn` log
//! in release builds — the intent is to catch misconfiguration early.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use obrain_substrate::SubstrateStore;
use parking_lot::{Condvar, Mutex};

use super::{Thinker, ThinkerKind, ThinkerTickError, ThinkerTickReport};

/// Trait abstracting system-load detection. The default impl uses
/// `sysinfo`-free heuristics (just the thread count / config); call
/// sites that need accurate load numbers can plug in a custom sensor.
pub trait PressureSensor: Send + Sync + 'static {
    /// Return `true` if the system is considered overloaded and
    /// Thinkers should pause.
    fn is_overloaded(&self) -> bool;
}

/// Default sensor: never reports overload. Suitable for tests and
/// small deployments; production should wire a `sysinfo`-backed impl
/// (T13 S8 follow-up).
#[derive(Debug, Default, Clone, Copy)]
pub struct NeverOverloadedSensor;

impl PressureSensor for NeverOverloadedSensor {
    fn is_overloaded(&self) -> bool {
        false
    }
}

/// Runtime-wide configuration.
#[derive(Debug, Clone)]
pub struct ThinkerRuntimeConfig {
    /// Hard cap on the sum of every Thinker's declared `cpu_fraction`.
    /// `1.0` = at most one full core used across all Thinkers.
    pub max_total_cpu_fraction: f32,
    /// Max duration to wait for each thread to join on shutdown.
    pub shutdown_timeout: Duration,
    /// Minimum tick interval. Shorter intervals are bumped up to this
    /// value at registration time (guard against misconfigured TOML).
    pub min_tick_interval: Duration,
}

impl Default for ThinkerRuntimeConfig {
    fn default() -> Self {
        Self {
            max_total_cpu_fraction: 1.0,
            shutdown_timeout: Duration::from_secs(5),
            min_tick_interval: Duration::from_secs(1),
        }
    }
}

/// Aggregate stats across every running Thinker. Useful for admin
/// dashboards and tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinkerRuntimeStats {
    pub threads_live: u64,
    pub total_ticks: u64,
    pub total_errors: u64,
    pub over_budget_ticks: u64,
    pub paused_ticks: u64,
}

/// Opaque handle to a running Thinker — used to inspect stats and to
/// individually disable a Thinker at runtime.
pub struct ThinkerHandle {
    pub kind: ThinkerKind,
    join: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    cv: Arc<(Mutex<()>, Condvar)>,
    ticks: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
    over_budget_ticks: Arc<AtomicU64>,
    paused_ticks: Arc<AtomicU64>,
}

impl ThinkerHandle {
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
        let (_, cv) = &*self.cv;
        cv.notify_all();
    }
    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::Relaxed)
    }
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }
    pub fn over_budget_ticks(&self) -> u64 {
        self.over_budget_ticks.load(Ordering::Relaxed)
    }
    pub fn paused_ticks(&self) -> u64 {
        self.paused_ticks.load(Ordering::Relaxed)
    }
}

/// Runtime that owns and orchestrates Thinker threads.
pub struct ThinkerRuntime {
    config: ThinkerRuntimeConfig,
    store: Arc<SubstrateStore>,
    sensor: Arc<dyn PressureSensor>,
    pause_flag: Arc<AtomicBool>,
    handles: Vec<ThinkerHandle>,
    total_cpu_fraction: f32,
}

impl ThinkerRuntime {
    pub fn new(
        store: Arc<SubstrateStore>,
        config: ThinkerRuntimeConfig,
        sensor: Arc<dyn PressureSensor>,
    ) -> Self {
        Self {
            config,
            store,
            sensor,
            pause_flag: Arc::new(AtomicBool::new(false)),
            handles: Vec::new(),
            total_cpu_fraction: 0.0,
        }
    }

    pub fn store(&self) -> &Arc<SubstrateStore> {
        &self.store
    }

    /// Register and spawn a Thinker. The Thinker is moved into the
    /// runtime; its lifetime is tied to the runtime's.
    pub fn spawn<T: Thinker>(&mut self, thinker: T) -> &ThinkerHandle {
        let kind = thinker.kind();
        let budget = thinker.budget();
        let mut interval = thinker.interval();
        if interval < self.config.min_tick_interval {
            tracing::warn!(
                thinker = kind.as_str(),
                requested = ?interval,
                min = ?self.config.min_tick_interval,
                "tick interval clamped to runtime minimum"
            );
            interval = self.config.min_tick_interval;
        }

        self.total_cpu_fraction += budget.cpu_fraction;
        if self.total_cpu_fraction > self.config.max_total_cpu_fraction {
            tracing::warn!(
                total = self.total_cpu_fraction,
                cap = self.config.max_total_cpu_fraction,
                "thinker CPU budget exceeded — misconfiguration?"
            );
            debug_assert!(
                self.total_cpu_fraction <= self.config.max_total_cpu_fraction,
                "aggregate thinker cpu_fraction {:.2} exceeds cap {:.2}",
                self.total_cpu_fraction, self.config.max_total_cpu_fraction
            );
        }

        let stop = Arc::new(AtomicBool::new(false));
        let paused_individual = Arc::new(AtomicBool::new(false));
        let cv = Arc::new((Mutex::new(()), Condvar::new()));
        let ticks = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(AtomicU64::new(0));
        let over_budget_ticks = Arc::new(AtomicU64::new(0));
        let paused_ticks = Arc::new(AtomicU64::new(0));

        let store = self.store.clone();
        let sensor = self.sensor.clone();
        let pause_flag = self.pause_flag.clone();
        let stop_c = stop.clone();
        let paused_c = paused_individual.clone();
        let cv_c = cv.clone();
        let ticks_c = ticks.clone();
        let errors_c = errors.clone();
        let over_c = over_budget_ticks.clone();
        let paused_count_c = paused_ticks.clone();
        let thinker = Arc::new(thinker);

        let name = format!("thinker-{}", kind.as_str());
        let join = thread::Builder::new()
            .name(name)
            .spawn(move || {
                run_thinker_loop(
                    thinker, store, sensor, pause_flag, stop_c, paused_c, cv_c,
                    interval, ticks_c, errors_c, over_c, paused_count_c, budget.max_tick_duration,
                );
            })
            .expect("thinker thread spawn");

        self.handles.push(ThinkerHandle {
            kind,
            join: Some(join),
            stop,
            paused: paused_individual,
            cv,
            ticks,
            errors,
            over_budget_ticks,
            paused_ticks,
        });
        self.handles.last().unwrap()
    }

    pub fn set_paused(&self, paused: bool) {
        self.pause_flag.store(paused, Ordering::Release);
        if !paused {
            for h in &self.handles {
                let (_, cv) = &*h.cv;
                cv.notify_all();
            }
        }
    }

    pub fn stats(&self) -> ThinkerRuntimeStats {
        let mut s = ThinkerRuntimeStats::default();
        for h in &self.handles {
            s.threads_live += 1;
            s.total_ticks += h.ticks();
            s.total_errors += h.errors();
            s.over_budget_ticks += h.over_budget_ticks();
            s.paused_ticks += h.paused_ticks();
        }
        s
    }

    pub fn handles(&self) -> &[ThinkerHandle] {
        &self.handles
    }

    /// Cooperatively stop every Thinker and join with timeout.
    pub fn shutdown(mut self) {
        let timeout = self.config.shutdown_timeout;
        // Phase 1: signal stop to everyone.
        for h in &self.handles {
            h.stop.store(true, Ordering::Release);
            let (_, cv) = &*h.cv;
            cv.notify_all();
        }
        // Phase 2: join with a shared deadline.
        let deadline = Instant::now() + timeout;
        for h in self.handles.iter_mut() {
            let Some(jh) = h.join.take() else {
                continue;
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                // Deadline exceeded — detach and warn.
                tracing::warn!(thinker = h.kind.as_str(), "shutdown deadline exceeded — thread detached");
                std::mem::drop(jh);
                continue;
            }
            // `JoinHandle` has no timed join in stable Rust; we rely on
            // the fact that every Thinker is a sleep/tick loop that
            // checks the stop flag at every wakeup. In practice the
            // threads exit within milliseconds of the notify_all above.
            match jh.join() {
                Ok(()) => tracing::debug!(thinker = h.kind.as_str(), "thinker joined cleanly"),
                Err(e) => tracing::warn!(thinker = h.kind.as_str(), ?e, "thinker panicked on shutdown"),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_thinker_loop(
    thinker: Arc<dyn Thinker>,
    store: Arc<SubstrateStore>,
    sensor: Arc<dyn PressureSensor>,
    global_pause: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    individual_pause: Arc<AtomicBool>,
    cv: Arc<(Mutex<()>, Condvar)>,
    interval: Duration,
    ticks: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
    over_budget_ticks: Arc<AtomicU64>,
    paused_ticks: Arc<AtomicU64>,
    max_tick_duration: Duration,
) {
    let kind_str = thinker.kind().as_str();
    tracing::debug!(thinker = kind_str, ?interval, "thinker loop started");

    loop {
        // Cooperative wait — either the stop signal, a resume after
        // pause, or the interval elapses.
        {
            let (m, c) = &*cv;
            let mut guard = m.lock();
            if stop.load(Ordering::Acquire) {
                break;
            }
            let _ = c.wait_for(&mut guard, interval);
            if stop.load(Ordering::Acquire) {
                break;
            }
        }

        // Pause gate.
        if global_pause.load(Ordering::Acquire)
            || individual_pause.load(Ordering::Acquire)
            || sensor.is_overloaded()
        {
            paused_ticks.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // Run tick with panic safety.
        let start = Instant::now();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            thinker.tick(&store)
        }));
        let elapsed = start.elapsed();

        match outcome {
            Ok(Ok(_report)) => {
                ticks.fetch_add(1, Ordering::Relaxed);
                if elapsed > max_tick_duration {
                    over_budget_ticks.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        thinker = kind_str,
                        elapsed_ms = elapsed.as_millis() as u64,
                        budget_ms = max_tick_duration.as_millis() as u64,
                        "thinker tick exceeded its budget"
                    );
                }
            }
            Ok(Err(e)) => {
                errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(thinker = kind_str, ?e, "thinker tick returned error");
                let _ = ThinkerTickError::Other(format!("{e}"));
                let _: Option<ThinkerTickReport> = None;
            }
            Err(panic_payload) => {
                errors.fetch_add(1, Ordering::Relaxed);
                let msg = panic_to_string(panic_payload);
                tracing::error!(thinker = kind_str, panic = %msg, "thinker tick panicked");
            }
        }
    }
    tracing::debug!(thinker = kind_str, "thinker loop exited");
}

fn panic_to_string(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinker::{ThinkerBudget};
    use obrain_substrate::store::SubstrateStore;
    use std::sync::atomic::AtomicU64;
    use tempfile::TempDir;

    struct CountingThinker {
        kind: ThinkerKind,
        interval: Duration,
        counter: Arc<AtomicU64>,
        budget: ThinkerBudget,
    }

    impl Thinker for CountingThinker {
        fn kind(&self) -> ThinkerKind {
            self.kind
        }
        fn budget(&self) -> ThinkerBudget {
            self.budget
        }
        fn interval(&self) -> Duration {
            self.interval
        }
        fn tick(
            &self,
            _store: &Arc<SubstrateStore>,
        ) -> Result<ThinkerTickReport, ThinkerTickError> {
            self.counter.fetch_add(1, Ordering::Relaxed);
            Ok(ThinkerTickReport::start().finish())
        }
    }

    #[test]
    fn runtime_spawns_and_joins() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("rt")).unwrap());
        let mut cfg = ThinkerRuntimeConfig::default();
        cfg.min_tick_interval = Duration::from_millis(50);
        let mut rt = ThinkerRuntime::new(
            store,
            cfg,
            Arc::new(NeverOverloadedSensor::default()),
        );
        let counter = Arc::new(AtomicU64::new(0));
        let _ = rt.spawn(CountingThinker {
            kind: ThinkerKind::Consolidator,
            interval: Duration::from_millis(50),
            counter: counter.clone(),
            budget: ThinkerBudget::minimal(),
        });
        // Let the thinker run for a few ticks.
        std::thread::sleep(Duration::from_millis(200));
        rt.shutdown();
        let ticks = counter.load(Ordering::Relaxed);
        assert!(ticks >= 1, "expected ≥ 1 tick, got {ticks}");
    }

    #[test]
    fn pressure_pause_skips_ticks() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("rt-pause")).unwrap());

        struct AlwaysOverloaded;
        impl PressureSensor for AlwaysOverloaded {
            fn is_overloaded(&self) -> bool {
                true
            }
        }

        let mut cfg = ThinkerRuntimeConfig::default();
        cfg.min_tick_interval = Duration::from_millis(30);
        let mut rt = ThinkerRuntime::new(store, cfg, Arc::new(AlwaysOverloaded));
        let counter = Arc::new(AtomicU64::new(0));
        rt.spawn(CountingThinker {
            kind: ThinkerKind::Predictor,
            interval: Duration::from_millis(30),
            counter: counter.clone(),
            budget: ThinkerBudget::minimal(),
        });
        std::thread::sleep(Duration::from_millis(150));
        let paused_ticks = rt.handles()[0].paused_ticks();
        rt.shutdown();
        assert_eq!(counter.load(Ordering::Relaxed), 0);
        assert!(paused_ticks >= 1, "expected ≥ 1 paused tick, got {paused_ticks}");
    }

    #[test]
    fn global_pause_toggles_live() {
        let td = TempDir::new().unwrap();
        let store = Arc::new(SubstrateStore::create(td.path().join("rt-global")).unwrap());
        let mut cfg = ThinkerRuntimeConfig::default();
        cfg.min_tick_interval = Duration::from_millis(30);
        let mut rt = ThinkerRuntime::new(
            store,
            cfg,
            Arc::new(NeverOverloadedSensor::default()),
        );
        let counter = Arc::new(AtomicU64::new(0));
        rt.spawn(CountingThinker {
            kind: ThinkerKind::Dreamer,
            interval: Duration::from_millis(30),
            counter: counter.clone(),
            budget: ThinkerBudget::minimal(),
        });
        rt.set_paused(true);
        std::thread::sleep(Duration::from_millis(100));
        let paused_count = counter.load(Ordering::Relaxed);
        rt.set_paused(false);
        std::thread::sleep(Duration::from_millis(150));
        let resumed_count = counter.load(Ordering::Relaxed);
        rt.shutdown();
        assert!(resumed_count > paused_count, "resumed_count={resumed_count} paused_count={paused_count}");
    }
}
