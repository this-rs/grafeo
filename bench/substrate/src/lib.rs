//! Shared harness for Substrate benchmarks.
//!
//! Produces JSON results matching [`BenchResult`] for downstream comparison in CI.

use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::{Pid, ProcessesToUpdate, System};

/// Target database for a benchmark run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Target {
    /// PO database (~500 k nodes, 4 M edges).
    Po,
    /// megalaw database (~12 M nodes, 90 M edges).
    Megalaw,
    /// A custom path.
    Custom,
}

/// Which backend are we benchmarking?
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Backend {
    /// Current LpgStore-backed engine (baseline).
    Lpg,
    /// New Substrate-backed engine (target).
    Substrate,
}

/// A single measurement point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub unit: String,
    pub value: f64,
    /// Optional percentile label ("p50", "p95", "p99", "mean", "max", …).
    pub stat: Option<String>,
}

impl Measurement {
    pub fn new(name: impl Into<String>, unit: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            unit: unit.into(),
            value,
            stat: None,
        }
    }

    pub fn with_stat(mut self, stat: impl Into<String>) -> Self {
        self.stat = Some(stat.into());
        self
    }
}

/// Output document — one per backend × target pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub schema_version: u32,
    pub backend: Backend,
    pub target: Target,
    pub db_path: String,
    pub timestamp: i64,
    pub host: HostInfo,
    pub measurements: Vec<Measurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub cores: usize,
    pub total_ram_bytes: u64,
    pub rustc_version: String,
}

impl HostInfo {
    pub fn collect() -> Self {
        let sys = System::new_all();
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu: sys
                .cpus()
                .first()
                .map(|c| c.brand().to_string())
                .unwrap_or_else(|| "unknown".into()),
            cores: sys.cpus().len(),
            // sysinfo ≥ 0.30 returns bytes (not KiB as in pre-0.30).
            total_ram_bytes: sys.total_memory(),
            // Capture rustc version via `rustc --version` at build time if env is set;
            // otherwise fall back to a runtime placeholder.
            rustc_version: option_env!("RUSTC_VERSION")
                .unwrap_or("unknown")
                .to_string(),
        }
    }
}

/// Samples RSS (resident set size) of the current process, in bytes.
///
/// Implementation: read `/proc/self/status` on Linux, `mach_task_basic_info` via sysinfo on macOS.
pub struct RssSampler {
    sys: System,
    pid: Pid,
    anon_peak: u64,
    total_peak: u64,
}

impl RssSampler {
    pub fn new() -> Self {
        let sys = System::new_all();
        let pid = Pid::from_u32(std::process::id());
        Self {
            sys,
            pid,
            anon_peak: 0,
            total_peak: 0,
        }
    }

    /// Take a sample. Returns `(anon_bytes, total_bytes)`.
    ///
    /// `anon` is heap + stacks (counts toward malloc pressure).
    /// `total` includes file-backed mmap (cheap — OS can evict under pressure).
    pub fn sample(&mut self) -> (u64, u64) {
        self.sys
            .refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        // sysinfo ≥ 0.30 reports memory in bytes.
        // `memory()` is the resident set size (RSS) of the process.
        // `virtual_memory()` is VSIZE — much larger on macOS due to shared pages.
        // For our purposes, `memory()` ≈ resident (includes both anon + file-backed mmap on Linux;
        // on macOS, excludes compressed pages). We report it as `total` and leave `anon` as
        // a best-effort OS-specific split to be refined per platform in T1-step4.
        let (anon, total) = match self.sys.process(self.pid) {
            Some(p) => {
                let total = p.memory();
                // TODO(T1-step4): Linux — parse /proc/self/status (VmRSS - VmLib - file-backed).
                //                 macOS — mach_task_basic_info to split phys_footprint vs virtual.
                // Placeholder: report `memory()` as both `anon` and `total`; this over-reports
                // anon on file-backed-mmap-heavy workloads (exactly what Substrate wants to show).
                let anon = p.memory();
                (anon, total)
            }
            None => (0, 0),
        };
        self.anon_peak = self.anon_peak.max(anon);
        self.total_peak = self.total_peak.max(total);
        (anon, total)
    }

    pub fn peaks(&self) -> (u64, u64) {
        (self.anon_peak, self.total_peak)
    }
}

impl Default for RssSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: time a closure.
pub fn time_it<F, T>(f: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}
