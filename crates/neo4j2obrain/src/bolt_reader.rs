//! # Stage 1 — Neo4j bolt streaming reader
//!
//! Compiled only with `--features neo4j-bolt`. Pulls nodes + edges off
//! a live Neo4j database via [`neo4rs`] and writes them into the
//! [`SubstrateStore`] one batch at a time. The intent is to keep peak
//! memory roughly bounded by `batch_size × (avg node payload)` — the
//! batch is flushed to substrate + WAL before the next one is fetched.
//!
//! ## Query shape
//!
//! The reader issues two straightforward `CALL apoc.periodic.iterate`
//! equivalents using `SKIP / LIMIT` windows:
//!
//! ```cypher
//! MATCH (n) RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props
//!   SKIP $skip LIMIT $limit
//! ```
//!
//! ```cypher
//! MATCH (a)-[r]->(b)
//!   RETURN id(a) AS src, id(b) AS dst, type(r) AS ty, properties(r) AS props
//!   SKIP $skip LIMIT $limit
//! ```
//!
//! Properties are translated 1:1 to [`obrain_common::types::Value`] via
//! [`value_from_bolt`]; unsupported kinds (Geo, Date types) are warned
//! once and dropped.
//!
//! ## Scope
//!
//! This is a structural reader — it does **not** preserve the Neo4j
//! internal node id (Neo4j ids are dense but not contractually stable
//! across DB restarts). Edges reference substrate [`NodeId`]s via a
//! `HashMap<i64, NodeId>` populated during the node pass.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use obrain_common::NodeId;
use obrain_common::types::{PropertyKey, Value};
use obrain_core::graph::traits::GraphStoreMut;
use obrain_substrate::SubstrateStore;

use crate::pipeline::PipelineOptions;

/// Reader output — plain counters suitable for `PipelineStats`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReadCounts {
    pub nodes: u64,
    pub edges: u64,
}

/// Stream nodes + edges from Neo4j into the given substrate store.
///
/// Opens a single bolt connection, walks the full graph in
/// `opts.batch_size` windows, and commits each batch to substrate
/// before requesting the next.
pub async fn stream_into_substrate(
    opts: &PipelineOptions,
    substrate: &Arc<SubstrateStore>,
) -> Result<ReadCounts> {
    let graph = neo4rs::Graph::new(
        &opts.neo4j_url,
        opts.neo4j_user.clone(),
        opts.neo4j_password.clone(),
    )
    .await
    .with_context(|| format!("connect to Neo4j at {}", opts.neo4j_url))?;

    let mut counts = ReadCounts::default();
    let mut id_map: HashMap<i64, NodeId> = HashMap::new();

    // ---- Node pass ----------------------------------------------------
    let mut skip: i64 = 0;
    let limit: i64 = opts.batch_size as i64;
    loop {
        let q = neo4rs::query(
            "MATCH (n) RETURN id(n) AS nid, labels(n) AS labels, \
             properties(n) AS props SKIP $skip LIMIT $limit",
        )
        .param("skip", skip)
        .param("limit", limit);
        let mut res = graph.execute(q).await.context("exec node query")?;
        let mut seen = 0i64;
        while let Ok(Some(row)) = res.next().await {
            let nid_raw: i64 = row.get("nid").context("row missing nid")?;
            let labels_raw: Vec<String> = row.get("labels").unwrap_or_default();
            let props_raw: neo4rs::BoltMap = row.get("props").unwrap_or_default();

            let label_refs: Vec<&str> = labels_raw.iter().map(String::as_str).collect();
            let props = bolt_map_to_props(&props_raw);

            let new_id = substrate.create_node_with_props(&label_refs, &props);
            id_map.insert(nid_raw, new_id);
            counts.nodes += 1;
            seen += 1;
        }
        substrate.flush().ok();
        if seen < limit {
            break;
        }
        skip += limit;
    }
    tracing::info!("Stage 1 (nodes): ingested {} nodes", counts.nodes);

    // ---- Edge pass ----------------------------------------------------
    let mut skip: i64 = 0;
    loop {
        let q = neo4rs::query(
            "MATCH (a)-[r]->(b) RETURN id(a) AS src, id(b) AS dst, \
             type(r) AS ty, properties(r) AS props SKIP $skip LIMIT $limit",
        )
        .param("skip", skip)
        .param("limit", limit);
        let mut res = graph.execute(q).await.context("exec edge query")?;
        let mut seen = 0i64;
        while let Ok(Some(row)) = res.next().await {
            let src_raw: i64 = row.get("src").context("row missing src")?;
            let dst_raw: i64 = row.get("dst").context("row missing dst")?;
            let ty: String = row.get("ty").unwrap_or_else(|_| "REL".into());
            let props_raw: neo4rs::BoltMap = row.get("props").unwrap_or_default();

            let (Some(&src), Some(&dst)) = (id_map.get(&src_raw), id_map.get(&dst_raw)) else {
                // Dangling reference — Neo4j returned an edge whose
                // endpoints were not in the node scan (very rare but
                // possible if the DB is mutating during the import).
                continue;
            };

            let props = bolt_map_to_props(&props_raw);
            let _ = substrate.create_edge_with_props(src, dst, &ty, &props);
            counts.edges += 1;
            seen += 1;
        }
        substrate.flush().ok();
        if seen < limit {
            break;
        }
        skip += limit;
    }
    tracing::info!("Stage 1 (edges): ingested {} edges", counts.edges);

    Ok(counts)
}

/// Translate a Neo4j bolt map into substrate [`Value`]s.
///
/// Supported kinds:
/// * scalars — Null, Boolean, Integer, Float, String, Bytes
/// * temporal — Date, DateTime, LocalDateTime, DateTimeZoneId
///   (collapsed to [`Value::Timestamp`] in seconds)
/// * containers — List (→ `Vector<f32>` if all-numeric, else recursive `List`)
///   and Map (recursive)
///
/// Dropped kinds (with a single warn-level trace per occurrence): Node,
/// Relation, UnboundedRelation, Path, Point2D, Point3D, Duration, Time,
/// LocalTime. These are either embedded-graph constructs that do not
/// translate 1:1 to properties, or geo / duration types the cognitive
/// pipeline does not consume.
fn bolt_map_to_props(map: &neo4rs::BoltMap) -> Vec<(PropertyKey, Value)> {
    let mut out = Vec::with_capacity(map.value.len());
    for (k, v) in map.value.iter() {
        if let Some(val) = value_from_bolt(v) {
            out.push((PropertyKey::new(k.value.as_str()), val));
        }
    }
    out
}

/// Best-effort translation of a single bolt value.
///
/// Returns `None` when the value is Null or of a kind we intentionally
/// drop. The first two cases are separated in the match so Null does
/// not trip the warn branch (Null is semantically a valid "absent"
/// marker, not an unsupported type).
fn value_from_bolt(v: &neo4rs::BoltType) -> Option<Value> {
    use neo4rs::BoltType::*;
    use obrain_common::types::Timestamp;

    Some(match v {
        Null(_) => return None,
        Boolean(b) => Value::Bool(b.value),
        Integer(i) => Value::Int64(i.value),
        Float(f) => Value::Float64(f.value),
        String(s) => Value::String(s.value.clone().into()),
        Bytes(b) => Value::Bytes(b.value.as_ref().into()),

        // -------- Lists --------------------------------------------------
        //
        // Neo4j embedding vectors are almost always `List<Float>` on the
        // bolt wire. Detect that shape and fast-path to `Value::Vector`
        // (Arc<[f32]>) so the cognitive pipeline sees them natively.
        // Fallback: recursive translation into `Value::List` for
        // heterogeneous or non-numeric lists.
        List(list) => translate_list(list)?,

        // -------- Maps ---------------------------------------------------
        Map(m) => {
            use std::collections::BTreeMap;
            use std::sync::Arc;
            let mut bt = BTreeMap::new();
            for (k, v) in m.value.iter() {
                if let Some(val) = value_from_bolt(v) {
                    bt.insert(PropertyKey::new(k.value.as_str()), val);
                }
            }
            if bt.is_empty() {
                return None;
            }
            Value::Map(Arc::new(bt))
        }

        // -------- Temporal ----------------------------------------------
        //
        // We collapse every time-zoned / local / date-only variant to
        // a single UTC-seconds `Timestamp`. The cognitive pipeline does
        // not distinguish; preserving the original kind would force us
        // to fan out into several obrain Value variants for no benefit.
        DateTime(dt) => {
            use chrono::DateTime as ChronoDT;
            use chrono::FixedOffset;
            match ChronoDT::<FixedOffset>::try_from(dt) {
                Ok(t) => Value::Timestamp(Timestamp::from_secs(t.timestamp())),
                Err(_) => return None,
            }
        }
        LocalDateTime(dt) => {
            use chrono::NaiveDateTime;
            match NaiveDateTime::try_from(dt) {
                Ok(t) => Value::Timestamp(Timestamp::from_secs(t.and_utc().timestamp())),
                Err(_) => return None,
            }
        }
        DateTimeZoneId(dt) => {
            use chrono::DateTime as ChronoDT;
            use chrono::FixedOffset;
            match ChronoDT::<FixedOffset>::try_from(dt) {
                Ok(t) => Value::Timestamp(Timestamp::from_secs(t.timestamp())),
                Err(_) => return None,
            }
        }
        Date(d) => {
            use chrono::NaiveDate;
            match NaiveDate::try_from(d) {
                Ok(nd) => {
                    // Midnight UTC on that date. Keeps ordering semantics
                    // without introducing a distinct Date variant.
                    let ts = nd
                        .and_hms_opt(0, 0, 0)
                        .map(|ndt| ndt.and_utc().timestamp())
                        .unwrap_or(0);
                    Value::Timestamp(Timestamp::from_secs(ts))
                }
                Err(_) => return None,
            }
        }

        // -------- Dropped kinds -----------------------------------------
        _ => {
            tracing::warn!(
                "bolt_reader: dropping unsupported Neo4j value kind ({:?})",
                std::mem::discriminant(v),
            );
            return None;
        }
    })
}

/// Translate a `BoltList` into either a dense `Value::Vector` (when
/// every element is numeric — the embedding case) or a heterogeneous
/// `Value::List`.
///
/// The numeric fast-path inspects every element: as soon as a non-numeric
/// one is seen we fall back to the recursive path. Integers are cast to
/// f32 so a `List<Integer>` that happens to encode a dense vector (rare,
/// but possible in legacy schemas) still lands as `Value::Vector`.
fn translate_list(list: &neo4rs::BoltList) -> Option<Value> {
    use neo4rs::BoltType::*;
    use std::sync::Arc;

    if list.value.is_empty() {
        return Some(Value::List(Arc::from(Vec::<Value>::new())));
    }

    // Numeric detection — Float and/or Integer only.
    let all_numeric = list
        .value
        .iter()
        .all(|e| matches!(e, Float(_) | Integer(_)));

    if all_numeric {
        let mut buf: Vec<f32> = Vec::with_capacity(list.value.len());
        for e in list.value.iter() {
            match e {
                Float(f) => buf.push(f.value as f32),
                Integer(i) => buf.push(i.value as f32),
                _ => unreachable!("numeric-only path already checked"),
            }
        }
        return Some(Value::Vector(Arc::from(buf)));
    }

    // Heterogeneous / non-numeric: recurse per element, skip Nulls and
    // drops cleanly. An all-dropped list collapses to None so we don't
    // write an empty `List` property — matches the Map handling above.
    let mut items: Vec<Value> = Vec::with_capacity(list.value.len());
    for e in list.value.iter() {
        if let Some(v) = value_from_bolt(e) {
            items.push(v);
        }
    }
    if items.is_empty() {
        return None;
    }
    Some(Value::List(Arc::from(items)))
}
