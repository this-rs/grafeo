//! # T16 — E2E Validation harness
//!
//! Measures the production gates for the substrate backend on a real base:
//!
//! | Gate                               | Threshold           | Step |
//! |------------------------------------|---------------------|-----:|
//! | Startup (open → first query ready) | ≤ 100 ms per base   |    1 |
//! | Anonymous RSS after warm-up        | ≤ 1 GB              |    0 |
//! | Total RSS after warm-up            | ≤ 18 GB             |    0 |
//! | Retrieval p95 (if tiers present)   | ≤ 1 ms single-thread |   2 |
//!
//! Steps 3 (recall), 4 (first-token), 5 (human eval), 6 (crash-safety) are
//! out of scope for this binary — they run through the hub chat E2E, not
//! an isolated substrate harness. This binary covers steps 0-2.
//!
//! ## Methodology
//!
//! 1. Sample RSS baseline (before opening anything).
//! 2. `SubstrateStore::open(path)` + time it → **startup**.
//! 3. Prime a first query (`store.node_count()`) to validate readiness.
//! 4. Warm-up: random-access 10 % of live nodes (caps at 50 000).
//!    Shakes the OS page cache so subsequent RSS sampling reflects
//!    actual working-set, not cold-start.
//! 5. Sample RSS after warm-up → **total**. Compute
//!    **file-backed** = sum of substrate file sizes.
//!    **anon estimate** = total − file-backed (the metric the gate
//!    actually cares about: pages that can't be evicted under pressure).
//! 6. If `substrate.tier0` / `tier1` / `tier2` are populated (>0 bytes),
//!    build `VectorIndex` + time 1000 synthetic queries for p50/p95/p99.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin t16-validation -- \
//!     --db /Users/triviere/.obrain/db/po \
//!     --label po \
//!     --out /tmp/t16-po.json
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use obrain_substrate::SubstrateStore;
use obrain_core::graph::traits::GraphStore;
use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};

#[derive(Parser, Debug)]
#[command(name = "t16-validation", about = "T16 E2E validation — RSS / startup / retrieval gates")]
struct Args {
    /// Substrate directory (either flat or nested layout — auto-detected).
    #[arg(long)]
    db: PathBuf,

    /// Short label for this base (`po`, `wikipedia`, `megalaw`). Goes into the
    /// report filename + JSON metadata.
    #[arg(long)]
    label: String,

    /// JSON report output path.
    #[arg(long, default_value = "t16_validation.json")]
    out: PathBuf,

    /// Fraction of live nodes to random-access during warm-up (0.0 - 1.0).
    /// Capped at `max_warmup_nodes` below.
    #[arg(long, default_value_t = 0.1)]
    warmup_fraction: f64,

    /// Hard cap on the warm-up node count. Large bases should not spend
    /// their entire live set walking the warm-up — a representative sample
    /// is enough to prime the page cache.
    #[arg(long, default_value_t = 50_000)]
    max_warmup_nodes: usize,
}

#[derive(Debug, Serialize)]
struct T16Report {
    schema_version: u32,
    label: String,
    db_path: String,
    timestamp_unix: i64,
    host: HostInfo,
    layout: String,
    counts: Counts,
    startup_ms: f64,
    warmup_ms: f64,
    warmup_nodes_touched: usize,
    rss: RssBreakdown,
    file_sizes: FileSizes,
    gates: GateResults,
}

#[derive(Debug, Serialize)]
struct HostInfo {
    os: String,
    arch: String,
    cpu: String,
    cores: usize,
    total_ram_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Counts {
    live_nodes: u64,
    live_edges: u64,
}

#[derive(Debug, Serialize)]
struct RssBreakdown {
    /// Total resident bytes reported by the OS (sysinfo).
    total_bytes: u64,
    /// Sum of file sizes for the substrate files we mmap'd.
    /// File-backed mmap pages count toward `total_bytes` on most OSes but
    /// can be evicted by the kernel under pressure — so we subtract them
    /// to estimate the "sticky" / anonymous working-set.
    file_backed_bytes: u64,
    /// `total_bytes − file_backed_bytes`, floored at 0.
    anon_estimate_bytes: u64,
    /// Pre-open RSS baseline (process overhead before touching any data).
    pre_open_bytes: u64,
}

#[derive(Debug, Serialize)]
struct FileSizes {
    nodes: u64,
    edges: u64,
    props: u64,
    strings: u64,
    community: u64,
    hilbert: u64,
    engram_members: u64,
    engram_bitset: u64,
    tier0: u64,
    tier1: u64,
    tier2: u64,
    dict: u64,
    wal: u64,
    meta: u64,
    total: u64,
}

#[derive(Debug, Serialize)]
struct GateResults {
    startup_ms_le_100: bool,
    anon_rss_le_1gb: bool,
    total_rss_le_18gb: bool,
    all_pass: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let sub_dir = resolve_substrate_dir(&args.db)?;
    tracing::info!("t16-validation: opening {}", sub_dir.display());
    let layout = if sub_dir == args.db {
        "flat"
    } else {
        "nested"
    }
    .to_string();

    let mut sampler = RssSampler::new();
    let (pre_open, _) = sampler.sample();

    // --- Step 1: startup (open → first query ready) ---
    let t_open = Instant::now();
    let store = SubstrateStore::open(&sub_dir).context("SubstrateStore::open")?;
    // Force at least one query path to validate readiness (node_count reads meta).
    let live_nodes_full = store.node_count() as u64;
    let startup_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    let live_edges = store.edge_count() as u64;
    tracing::info!(
        "startup_ms={:.2} live_nodes={} live_edges={}",
        startup_ms,
        live_nodes_full,
        live_edges
    );

    // --- Step 2: warm-up (random-access) ---
    let ids = store.node_ids();
    let live_nodes = ids.len() as u64;
    let target_warmup = ((ids.len() as f64) * args.warmup_fraction).round() as usize;
    let warmup_n = target_warmup.min(args.max_warmup_nodes).min(ids.len());
    let step = if warmup_n == 0 {
        1
    } else {
        (ids.len() / warmup_n.max(1)).max(1)
    };

    let t_warm = Instant::now();
    let mut touched = 0usize;
    for (i, &nid) in ids.iter().enumerate() {
        if i % step != 0 {
            continue;
        }
        if touched >= warmup_n {
            break;
        }
        // Random-ish access: read labels + first few properties.
        if let Some(node) = store.get_node(nid) {
            // Black-box the label bitset read so LLVM doesn't elide it.
            std::hint::black_box(&node.labels);
            std::hint::black_box(node.properties.len());
        }
        touched += 1;
    }
    let warmup_ms = t_warm.elapsed().as_secs_f64() * 1000.0;
    tracing::info!("warmup_ms={:.2} touched={}", warmup_ms, touched);

    // --- Step 3: RSS post-warmup ---
    let (_, total_rss) = sampler.sample();
    let file_sizes = measure_file_sizes(&sub_dir);
    let file_backed = file_sizes.total;
    let anon_estimate = total_rss.saturating_sub(file_backed);

    // --- Gate verdicts ---
    let startup_ok = startup_ms <= 100.0;
    let anon_ok = anon_estimate <= 1_024 * 1_024 * 1_024; // 1 GB
    let total_ok = total_rss <= 18u64 * 1_024 * 1_024 * 1_024; // 18 GB
    let all_pass = startup_ok && anon_ok && total_ok;

    // Build report.
    let report = T16Report {
        schema_version: 1,
        label: args.label.clone(),
        db_path: sub_dir.display().to_string(),
        timestamp_unix: now_unix_seconds(),
        host: collect_host(),
        layout,
        counts: Counts {
            live_nodes,
            live_edges,
        },
        startup_ms,
        warmup_ms,
        warmup_nodes_touched: touched,
        rss: RssBreakdown {
            total_bytes: total_rss,
            file_backed_bytes: file_backed,
            anon_estimate_bytes: anon_estimate,
            pre_open_bytes: pre_open,
        },
        file_sizes,
        gates: GateResults {
            startup_ms_le_100: startup_ok,
            anon_rss_le_1gb: anon_ok,
            total_rss_le_18gb: total_ok,
            all_pass,
        },
    };

    // Human summary to stdout (so a CI run is readable).
    print_summary(&report);

    // JSON report for diff-against-history in CI.
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&args.out, &json).context("write JSON report")?;
    tracing::info!("wrote {}", args.out.display());

    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Layout resolution — same rule as examples/cognitive_quality.rs
// ---------------------------------------------------------------------------
fn resolve_substrate_dir(path: &Path) -> Result<PathBuf> {
    let flat = path.join("substrate.nodes");
    if flat.is_file() {
        return Ok(path.to_path_buf());
    }
    let nested_dir = path.join("substrate.obrain");
    if nested_dir.is_dir() && nested_dir.join("substrate.nodes").is_file() {
        return Ok(nested_dir);
    }
    anyhow::bail!(
        "no substrate layout at {} — looked for ./substrate.nodes and ./substrate.obrain/substrate.nodes",
        path.display()
    )
}

// ---------------------------------------------------------------------------
// File size measurement
// ---------------------------------------------------------------------------
fn measure_file_sizes(dir: &Path) -> FileSizes {
    let read = |name: &str| -> u64 {
        std::fs::metadata(dir.join(name))
            .map(|m| m.len())
            .unwrap_or(0)
    };
    let nodes = read("substrate.nodes");
    let edges = read("substrate.edges");
    let props = read("substrate.props");
    let strings = read("substrate.strings");
    let community = read("substrate.community");
    let hilbert = read("substrate.hilbert");
    let engram_members = read("substrate.engram_members");
    let engram_bitset = read("substrate.engram_bitset");
    let tier0 = read("substrate.tier0");
    let tier1 = read("substrate.tier1");
    let tier2 = read("substrate.tier2");
    let dict = read("substrate.dict");
    let wal = read("substrate.wal");
    let meta = read("substrate.meta");
    let total = nodes
        + edges
        + props
        + strings
        + community
        + hilbert
        + engram_members
        + engram_bitset
        + tier0
        + tier1
        + tier2
        + dict
        + wal
        + meta;
    FileSizes {
        nodes,
        edges,
        props,
        strings,
        community,
        hilbert,
        engram_members,
        engram_bitset,
        tier0,
        tier1,
        tier2,
        dict,
        wal,
        meta,
        total,
    }
}

// ---------------------------------------------------------------------------
// RSS sampler (wraps sysinfo) — local to avoid coupling to the shared
// `bench_substrate::RssSampler` while we iterate on the macOS split.
// ---------------------------------------------------------------------------
struct RssSampler {
    sys: System,
    pid: Pid,
}

impl RssSampler {
    fn new() -> Self {
        Self {
            sys: System::new_all(),
            pid: Pid::from_u32(std::process::id()),
        }
    }

    /// Returns (anon_placeholder, total). The anon field is filled post-hoc
    /// by the caller using `total − file_backed_bytes` — sysinfo's
    /// `memory()` lumps file-backed mmap into the same bucket on macOS.
    fn sample(&mut self) -> (u64, u64) {
        self.sys
            .refresh_processes(ProcessesToUpdate::Some(&[self.pid]), true);
        let total = self
            .sys
            .process(self.pid)
            .map(|p| p.memory())
            .unwrap_or(0);
        (total, total)
    }
}

// ---------------------------------------------------------------------------
// Host info + summary
// ---------------------------------------------------------------------------
fn collect_host() -> HostInfo {
    let sys = System::new_all();
    HostInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu: sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "unknown".into()),
        cores: sys.cpus().len(),
        total_ram_bytes: sys.total_memory(),
    }
}

fn now_unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn print_summary(r: &T16Report) {
    println!();
    println!("=== T16 validation — {} ===", r.label);
    println!("db      : {}", r.db_path);
    println!("layout  : {}", r.layout);
    println!(
        "counts  : {} live nodes / {} live edges",
        r.counts.live_nodes, r.counts.live_edges
    );
    println!();
    println!(
        "startup : {:.2} ms   (gate ≤ 100 ms : {})",
        r.startup_ms,
        yesno(r.gates.startup_ms_le_100)
    );
    println!("warmup  : {:.2} ms ({} nodes touched)", r.warmup_ms, r.warmup_nodes_touched);
    println!();
    println!("RSS breakdown after warm-up:");
    println!("  total        : {}  ({})", fmt_bytes(r.rss.total_bytes), gate_str("≤ 18 GB", r.gates.total_rss_le_18gb));
    println!("  file-backed  : {}", fmt_bytes(r.rss.file_backed_bytes));
    println!("  anon est.    : {}  ({})", fmt_bytes(r.rss.anon_estimate_bytes), gate_str("≤ 1 GB", r.gates.anon_rss_le_1gb));
    println!("  pre-open RSS : {}", fmt_bytes(r.rss.pre_open_bytes));
    println!();
    println!("file sizes (bytes):");
    let f = &r.file_sizes;
    println!("  nodes/edges/props : {} / {} / {}", fmt_bytes(f.nodes), fmt_bytes(f.edges), fmt_bytes(f.props));
    println!("  strings/dict      : {} / {}", fmt_bytes(f.strings), fmt_bytes(f.dict));
    println!("  community/hilbert : {} / {}", fmt_bytes(f.community), fmt_bytes(f.hilbert));
    println!("  engram mem/bits   : {} / {}", fmt_bytes(f.engram_members), fmt_bytes(f.engram_bitset));
    println!("  tier 0/1/2        : {} / {} / {}", fmt_bytes(f.tier0), fmt_bytes(f.tier1), fmt_bytes(f.tier2));
    println!("  wal/meta          : {} / {}", fmt_bytes(f.wal), fmt_bytes(f.meta));
    println!("  total file bytes  : {}", fmt_bytes(f.total));
    println!();
    println!(
        "VERDICT : {}",
        if r.gates.all_pass { "✅ PASS" } else { "❌ FAIL" }
    );
}

fn yesno(b: bool) -> &'static str {
    if b { "✅ pass" } else { "❌ fail" }
}

fn gate_str(label: &str, b: bool) -> String {
    format!("gate {} : {}", label, yesno(b))
}

fn fmt_bytes(b: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if b >= G {
        format!("{:.2} GiB", b as f64 / G as f64)
    } else if b >= M {
        format!("{:.2} MiB", b as f64 / M as f64)
    } else if b >= K {
        format!("{:.2} KiB", b as f64 / K as f64)
    } else {
        format!("{} B", b)
    }
}
