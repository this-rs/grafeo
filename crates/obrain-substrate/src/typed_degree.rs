//! # Per-edge-type degree registry (T17h T8)
//!
//! Maintains one [`DegreeColumn`](crate::degree_column::DegreeColumn)
//! per `edge_type_id: u16`, persisted as
//! `substrate.degrees.node.<edge_type_name>.u32`.
//!
//! ## Why
//!
//! T5 gave us a **total** out/in degree per node. That's enough for
//! `ORDER BY total_degree` queries, but **not** for Cypher patterns
//! that filter by edge type :
//!
//! ```cypher
//! MATCH (f:File)
//! OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
//! WITH f, count(*) AS imports ORDER BY imports DESC LIMIT 50
//! ```
//!
//! Using the total degree here would inflate the count with non-IMPORTS
//! edges (CONTAINS, SYNAPSE, etc.). This module keeps per-type counters
//! so Cypher can replace the `walk + count per edge type` with an
//! O(1) lookup.
//!
//! ## Design (T17h T8)
//!
//! - **One file per edge type** : `substrate.degrees.node.<type_name>.u32`
//!   where `<type_name>` is the ASCII edge-type name from the registry
//!   (with a small filename-safe encoding for the few special chars).
//! - **Columns stored in a `DashMap<u16, Arc<RwLock<DegreeColumn>>>`** :
//!   lazy creation on first `incr_*` for a type; existing types picked
//!   up via directory scan at `open_all`.
//! - **Reuses [`DegreeColumn`]** verbatim — same header, same CRC, same
//!   atomic accessors. Each column is independent (no cross-type lock).
//! - **Total degree T5 is preserved** : writer hooks call BOTH total
//!   (T5) and typed (T8) increments. `sum_over_types(out_by_type[n]) ==
//!   total_out[n]` is an invariant verified by tests.
//!
//! ## Crash-safety
//!
//! Each column has its own CRC header. On open, corrupt columns return
//! `None` → the caller rebuilds via `rebuild_from_scan` (walks edges,
//! partitions by type). One corrupt type doesn't invalidate the
//! others.
//!
//! ## Memory / disk cost
//!
//! Per-type column = 64 B header + 8 B × n_slots. For a store with K
//! edge types and N nodes :
//!
//! - PO (1.36M nodes × 63 types) : ~630 MB total
//! - Wikipedia (4.56M × 5 types)  : ~185 MB
//! - Megalaw (8.13M × 10 types)   : ~650 MB
//!
//! Lazy creation mitigates this : rare edge types (< 1% of nodes) can
//! be left uncolumned (fallback walk). For now the simplest policy is
//! "create column on first edge of that type".

#![allow(unsafe_code)]

use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::degree_column::DegreeColumn;
use crate::error::SubstrateResult;
use crate::file::SubstrateFile;

/// Filename prefix — full pattern is
/// `substrate.degrees.node.<type_name>.u32`. Strips the u32 suffix from
/// the T5 base filename to insert the type name mid-path.
pub const TYPED_DEGREE_FILENAME_PREFIX: &str = "substrate.degrees.node.";
pub const TYPED_DEGREE_FILENAME_SUFFIX: &str = ".u32";

/// Filename-safe encoding for a type name. Rejects characters that
/// would confuse the filesystem (path separators, `..`). In practice
/// edge type names in Obrain are ASCII identifiers
/// (`IMPORTS`, `HAS_FILE`, etc.) — this function mainly guards against
/// future additions.
pub fn filename_for_type(type_name: &str) -> String {
    let safe: String = type_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{TYPED_DEGREE_FILENAME_PREFIX}{safe}{TYPED_DEGREE_FILENAME_SUFFIX}")
}

/// Registry mapping `edge_type_id` → DegreeColumn. Created once per
/// store; internal DashMap supports concurrent lazy insertion.
pub struct TypedDegreeRegistry {
    /// `edge_type_id → Arc<RwLock<DegreeColumn>>`. `Arc` so callers can
    /// `.clone()` the column handle for short-lived read-lock scopes.
    columns: DashMap<u16, Arc<RwLock<DegreeColumn>>>,
    /// Needed for lazy file creation.
    substrate: Arc<parking_lot::Mutex<SubstrateFile>>,
    /// Edge-type name registry for filename resolution at persist /
    /// rebuild time. Same registry as `SubstrateStore::edge_types`.
    edge_type_names: Arc<RwLock<crate::store::EdgeTypeRegistry>>,
}

impl TypedDegreeRegistry {
    /// Construct an empty registry. Columns are created lazily on
    /// first `incr_*` for a type, or all-at-once via `rebuild_from_scan`.
    pub(crate) fn new(
        substrate: Arc<parking_lot::Mutex<SubstrateFile>>,
        edge_type_names: Arc<RwLock<crate::store::EdgeTypeRegistry>>,
    ) -> Self {
        Self {
            columns: DashMap::new(),
            substrate,
            edge_type_names,
        }
    }

    /// Retrieve (or lazily create) the column for `edge_type_id`.
    /// Returns a clone of the Arc so callers can drop the registry
    /// guard before holding the column lock.
    pub(crate) fn get_or_create(
        &self,
        edge_type_id: u16,
        n_slots_hint: u32,
    ) -> SubstrateResult<Arc<RwLock<DegreeColumn>>> {
        // Fast path: column exists.
        if let Some(col) = self.columns.get(&edge_type_id) {
            return Ok(col.clone());
        }
        // Cold path: create. Use entry() or_insert_with to avoid race
        // (two threads both trying to create).
        let type_name = self
            .edge_type_names
            .read()
            .name_for(edge_type_id)
            .ok_or_else(|| {
                crate::error::SubstrateError::Internal(format!(
                    "typed_degree: unknown edge_type_id {edge_type_id}"
                ))
            })?
            .to_string();
        let entry = self.columns.entry(edge_type_id).or_try_insert_with(|| {
            let sub = self.substrate.lock();
            // Try open existing first (for bases where the column was
            // persisted previously). Fall back to create.
            let col = match DegreeColumn::open_by_filename(
                &sub,
                &filename_for_type(&type_name),
            )? {
                Some(c) => c,
                None => DegreeColumn::create_by_filename(
                    &sub,
                    &filename_for_type(&type_name),
                    n_slots_hint,
                )?,
            };
            Ok::<_, crate::error::SubstrateError>(Arc::new(RwLock::new(col)))
        })?;
        Ok(entry.value().clone())
    }

    /// Increment out-degree for `(edge_type_id, src_slot)` by `delta`.
    /// Creates the column lazily if absent. Safe under concurrent
    /// writers (atomic fetch_add inside read lock; grow via upgrade).
    pub fn incr_out(
        &self,
        edge_type_id: u16,
        slot: u32,
        delta: i32,
        n_slots_hint: u32,
    ) -> SubstrateResult<()> {
        let col = self.get_or_create(edge_type_id, n_slots_hint)?;
        {
            let rl = col.read();
            if slot < rl.n_slots() {
                rl.incr_out(slot, delta);
                return Ok(());
            }
        }
        let mut wl = col.write();
        wl.ensure_slot(slot)?;
        wl.incr_out(slot, delta);
        Ok(())
    }

    /// Increment in-degree for `(edge_type_id, dst_slot)`. Same as
    /// `incr_out` but for the second u32 of each record.
    pub fn incr_in(
        &self,
        edge_type_id: u16,
        slot: u32,
        delta: i32,
        n_slots_hint: u32,
    ) -> SubstrateResult<()> {
        let col = self.get_or_create(edge_type_id, n_slots_hint)?;
        {
            let rl = col.read();
            if slot < rl.n_slots() {
                rl.incr_in(slot, delta);
                return Ok(());
            }
        }
        let mut wl = col.write();
        wl.ensure_slot(slot)?;
        wl.incr_in(slot, delta);
        Ok(())
    }

    /// Read out-degree for `(edge_type_id, slot)`. Returns 0 if no
    /// column exists for that type (no edges of that type ever
    /// created).
    pub fn out_degree(&self, edge_type_id: u16, slot: u32) -> u32 {
        self.columns
            .get(&edge_type_id)
            .map(|c| c.read().out_degree(slot))
            .unwrap_or(0)
    }

    /// Read in-degree for `(edge_type_id, slot)`. Returns 0 if no
    /// column exists for that type.
    pub fn in_degree(&self, edge_type_id: u16, slot: u32) -> u32 {
        self.columns
            .get(&edge_type_id)
            .map(|c| c.read().in_degree(slot))
            .unwrap_or(0)
    }

    /// Persist all open columns : compute CRC, msync. Called from
    /// `SubstrateStore::flush`.
    pub fn flush(&self) -> SubstrateResult<()> {
        for entry in self.columns.iter() {
            let mut col = entry.value().write();
            col.persist_header_crc();
            col.msync()?;
        }
        Ok(())
    }

    /// Returns the number of edge-type columns currently materialized.
    /// For observability / tests.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if no column has been created yet.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Returns all (edge_type_id, edge_type_name) pairs for which a
    /// column exists. For observability / debug.
    pub fn edge_type_ids(&self) -> Vec<u16> {
        let mut ids: Vec<u16> = self.columns.iter().map(|e| *e.key()).collect();
        ids.sort_unstable();
        ids
    }

    /// Scan the substrate directory for existing
    /// `substrate.degrees.node.<type>.u32` files and open each one into
    /// the registry. Idempotent — a type whose column is already
    /// materialised is left untouched.
    ///
    /// Returns the number of columns opened (freshly picked up from
    /// disk this call — already-materialised ones aren't counted).
    ///
    /// Used at open time after restore_counters + init_degrees (T5) to
    /// hydrate typed columns for bases that persisted them in a prior
    /// session. Bases with no prior typed-column sidecars return 0,
    /// and the caller is expected to invoke the rebuild path.
    pub(crate) fn open_existing_columns(&self) -> SubstrateResult<usize> {
        let dir_path = {
            let sub = self.substrate.lock();
            sub.path().to_path_buf()
        };
        let rd = match std::fs::read_dir(&dir_path) {
            Ok(rd) => rd,
            Err(_) => return Ok(0),
        };
        let mut opened = 0usize;
        for entry in rd.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            // Match `substrate.degrees.node.<type>.u32`, but reject the
            // plain-total T5 sidecar `substrate.degrees.node.u32` —
            // that one has no embedded type name.
            let Some(rest) = name_str.strip_prefix(TYPED_DEGREE_FILENAME_PREFIX) else {
                continue;
            };
            let Some(type_name) = rest.strip_suffix(TYPED_DEGREE_FILENAME_SUFFIX) else {
                continue;
            };
            if type_name.is_empty() {
                continue;
            }
            let reg = self.edge_type_names.read();
            let type_id = match reg.id_for(type_name) {
                Some(id) => id,
                None => {
                    // Orphan column — type name not in registry. Skip
                    // silently (could be from a pre-pruning compaction).
                    drop(reg);
                    continue;
                }
            };
            drop(reg);
            // Fast path — column already materialised (e.g. after a
            // prior incr). Skip re-open.
            if self.columns.contains_key(&type_id) {
                continue;
            }
            // Direct open path : bypasses get_or_create so a CRC-bad
            // file returns None cleanly (get_or_create would fall
            // through to create_by_filename with n_slots_hint=0, which
            // silently replaces the bad column with an empty one —
            // silent data loss). Here we skip bad columns ; the caller
            // is expected to trigger a rebuild for orphaned types.
            let sub = self.substrate.lock();
            let col_opt = DegreeColumn::open_by_filename(&sub, &filename_for_type(type_name))?;
            drop(sub);
            let Some(col) = col_opt else {
                // Corrupt CRC or unexpected shape — leave to rebuild.
                continue;
            };
            self.columns
                .insert(type_id, Arc::new(RwLock::new(col)));
            opened += 1;
        }
        Ok(opened)
    }
}
