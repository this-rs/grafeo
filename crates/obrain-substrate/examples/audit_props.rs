//! # `audit_props` — per-key byte histogram of `substrate.props`
//!
//! Answers: "what's *actually* in the 2.89 GB / 7.72 GB `substrate.props`
//! sidecar?" — without materialising the whole snapshot in RAM.
//!
//! `PropertiesSnapshotV1::load` does a full `std::fs::read +
//! bincode::decode_from_slice`, which is precisely what blows anon RSS
//! to 2.7 × / 57 × the gate on PO / Wiki (T16.5 measurement). This
//! binary sidesteps that path: it reads the file one `PropEntry` at a
//! time through a `BufReader`, updates an in-place histogram, and
//! drops the decoded entry before moving on. Peak anon ≈ 1 entry +
//! 1 MB BufReader buffer = well under 10 MB regardless of file size.
//!
//! ## Output
//!
//! Two tables, nodes and edges, each sorted by `total_bytes desc`:
//!
//! ```text
//! key                  occurrences    total_bytes    avg_bytes    max_bytes
//! content              1_355_567      1_200_000_000  885          12_400_000
//! _kernel_embedding    1_355_567        432_000_000  320          320
//! ...
//! ```
//!
//! ## Usage
//! ```text
//! cargo run --release -p obrain-substrate --example audit_props -- \
//!     /Users/me/.obrain/db/po-t16.5
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;

use bincode::config::{Configuration, Fixint, LittleEndian};
use obrain_common::types::Value;
use obrain_substrate::PROPS_FILENAME;
use serde::{Deserialize, Serialize};

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

const MAGIC: u32 = 0x5052_4F50;
const FORMAT_VERSION: u32 = 1;

/// Mirror of `PropEntry` — decoded one at a time then dropped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PropEntry {
    id: u64,
    props: Vec<(String, Value)>,
}

#[derive(Default, Clone)]
struct KeyStats {
    count: u64,
    total_bytes: u64,
    max_bytes: u64,
    sum_bytes_sq: u128, // for variance; overkill but cheap
}

impl KeyStats {
    fn observe(&mut self, bytes: u64) {
        self.count += 1;
        self.total_bytes += bytes;
        if bytes > self.max_bytes {
            self.max_bytes = bytes;
        }
        self.sum_bytes_sq += (bytes as u128) * (bytes as u128);
    }
}

fn bincode_config() -> Configuration<LittleEndian, Fixint> {
    bincode::config::standard()
        .with_little_endian()
        .with_fixed_int_encoding()
}

fn value_bytes(v: &Value) -> u64 {
    // Approx on-disk footprint under bincode fixint LE. Tag byte for the
    // enum variant + payload. The goal isn't byte-exact accounting (bincode
    // serialises enum tags as u32 fixint, for example) — it's to identify
    // the heavy keys, which is dominated by payload.
    4 + match v {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Int64(_) => 8,
        Value::Float64(_) => 8,
        Value::String(s) => {
            let s_ref: &str = s.as_ref();
            8 + s_ref.len() as u64
        }
        Value::Bytes(b) => 8 + b.len() as u64,
        Value::Vector(vec) => 8 + (vec.len() as u64) * 4,
        // Catch-all for any variant we don't explicitly size.
        _ => 8,
    }
}

fn resolve_props_path(base: &std::path::Path) -> PathBuf {
    let nested = base.join("substrate.obrain").join(PROPS_FILENAME);
    if nested.exists() {
        return nested;
    }
    let flat = base.join(PROPS_FILENAME);
    if flat.exists() {
        return flat;
    }
    // Maybe the user passed the nested dir directly.
    let direct = base.join(PROPS_FILENAME);
    direct
}

fn main() -> Result<(), BoxErr> {
    let base: PathBuf = std::env::args()
        .nth(1)
        .ok_or_else(|| -> BoxErr { "usage: audit_props <substrate-dir>".into() })?
        .into();
    let path = resolve_props_path(&base);
    if !path.exists() {
        return Err(format!("no substrate.props at {}", path.display()).into());
    }

    let size = std::fs::metadata(&path)?.len();
    eprintln!(
        "audit_props: {} ({:.2} GiB)",
        path.display(),
        size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let mut f = File::open(&path)?;
    let mut hdr = [0u8; 4];
    f.read_exact(&mut hdr)?;
    let magic = u32::from_le_bytes(hdr);
    if magic != MAGIC {
        return Err(format!("bad magic {magic:#x}").into());
    }
    f.read_exact(&mut hdr)?;
    let version = u32::from_le_bytes(hdr);
    if version != FORMAT_VERSION {
        return Err(format!("unsupported version {version}").into());
    }

    let mut reader = BufReader::with_capacity(1 << 20, f);
    let cfg = bincode_config();

    // ---- Nodes ----
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let nodes_len = u64::from_le_bytes(len_buf);
    eprintln!("audit_props: {} node entries", nodes_len);

    let mut node_hist: HashMap<String, KeyStats> = HashMap::new();
    let mut node_entries_with_any_prop = 0u64;
    let mut node_props_total = 0u64;

    let t_nodes = std::time::Instant::now();
    for i in 0..nodes_len {
        if i > 0 && i % 250_000 == 0 {
            eprintln!(
                "audit_props: nodes {}/{} ({:.1}s elapsed)",
                i,
                nodes_len,
                t_nodes.elapsed().as_secs_f64()
            );
        }
        let entry: PropEntry = bincode::serde::decode_from_std_read(&mut reader, cfg)
            .map_err(|e| -> BoxErr { format!("node decode at {i}: {e}").into() })?;
        if !entry.props.is_empty() {
            node_entries_with_any_prop += 1;
        }
        for (k, v) in &entry.props {
            node_props_total += 1;
            let vb = value_bytes(v);
            node_hist.entry(k.clone()).or_default().observe(vb);
        }
        // entry dropped here
    }
    eprintln!(
        "audit_props: nodes done in {:.2}s",
        t_nodes.elapsed().as_secs_f64()
    );

    // ---- Edges ----
    reader.read_exact(&mut len_buf)?;
    let edges_len = u64::from_le_bytes(len_buf);
    eprintln!("audit_props: {} edge entries", edges_len);

    let mut edge_hist: HashMap<String, KeyStats> = HashMap::new();
    let mut edge_entries_with_any_prop = 0u64;
    let mut edge_props_total = 0u64;

    let t_edges = std::time::Instant::now();
    for i in 0..edges_len {
        if i > 0 && i % 500_000 == 0 {
            eprintln!(
                "audit_props: edges {}/{} ({:.1}s elapsed)",
                i,
                edges_len,
                t_edges.elapsed().as_secs_f64()
            );
        }
        let entry: PropEntry = bincode::serde::decode_from_std_read(&mut reader, cfg)
            .map_err(|e| -> BoxErr { format!("edge decode at {i}: {e}").into() })?;
        if !entry.props.is_empty() {
            edge_entries_with_any_prop += 1;
        }
        for (k, v) in &entry.props {
            edge_props_total += 1;
            let vb = value_bytes(v);
            edge_hist.entry(k.clone()).or_default().observe(vb);
        }
    }
    eprintln!(
        "audit_props: edges done in {:.2}s",
        t_edges.elapsed().as_secs_f64()
    );

    // ---- Report ----
    println!();
    println!("=== audit_props: {} ===", path.display());
    println!(
        "file_size_bytes       : {} ({:.2} GiB)",
        size,
        size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("node_entries          : {}", nodes_len);
    println!("  with >=1 prop       : {}", node_entries_with_any_prop);
    println!("  total (key,val)     : {}", node_props_total);
    println!("edge_entries          : {}", edges_len);
    println!("  with >=1 prop       : {}", edge_entries_with_any_prop);
    println!("  total (key,val)     : {}", edge_props_total);

    print_hist("node keys", &node_hist);
    print_hist("edge keys", &edge_hist);

    // Coverage sanity: sum of per-key total_bytes should approximate
    // the file size (less overhead for entries + header + CRC).
    let sum_node: u64 = node_hist.values().map(|s| s.total_bytes).sum();
    let sum_edge: u64 = edge_hist.values().map(|s| s.total_bytes).sum();
    println!();
    println!(
        "sum_node_value_bytes  : {} ({:.2} GiB)",
        sum_node,
        sum_node as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "sum_edge_value_bytes  : {} ({:.2} GiB)",
        sum_edge,
        sum_edge as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let accounted = sum_node + sum_edge;
    println!(
        "accounted_vs_file     : {:.1}%",
        accounted as f64 / size as f64 * 100.0
    );
    Ok(())
}

fn print_hist(title: &str, hist: &HashMap<String, KeyStats>) {
    println!();
    println!("=== {} (sorted by total_bytes desc) ===", title);
    println!(
        "{:<28}  {:>12}  {:>16}  {:>10}  {:>12}",
        "key", "count", "total_bytes", "avg", "max_bytes"
    );
    let mut rows: Vec<_> = hist.iter().collect();
    rows.sort_by(|a, b| b.1.total_bytes.cmp(&a.1.total_bytes));
    for (k, s) in rows.iter().take(30) {
        let avg = if s.count > 0 {
            s.total_bytes / s.count
        } else {
            0
        };
        println!(
            "{:<28}  {:>12}  {:>16}  {:>10}  {:>12}",
            truncate(k, 28),
            s.count,
            s.total_bytes,
            avg,
            s.max_bytes
        );
    }
    if rows.len() > 30 {
        println!("  ... +{} more keys", rows.len() - 30);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max - 1;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}
