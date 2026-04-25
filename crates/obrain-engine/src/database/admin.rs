//! Admin, introspection, and diagnostic operations for ObrainDB.

use std::path::Path;

use obrain_common::utils::error::Result;

impl super::ObrainDB {
    // =========================================================================
    // ADMIN API: Counts
    // =========================================================================

    /// Returns the number of nodes in the database.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.data_store().node_count()
    }

    /// Returns the number of edges in the database.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.data_store().edge_count()
    }

    /// Returns the number of distinct labels in the database.
    ///
    /// T17 W3c slice 5b: derived from the trait-safe `all_labels()` so
    /// substrate-backed databases report real counts (previously routed
    /// through the LpgStore-inherent `label_count()` on the dummy store).
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.data_store().all_labels().len()
    }

    /// Returns the number of distinct property keys in the database.
    ///
    /// T17 W3c slice 5b: see `label_count` — routes through the trait's
    /// `all_property_keys()` for substrate parity.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.data_store().all_property_keys().len()
    }

    /// Returns the number of distinct edge types in the database.
    ///
    /// T17 W3c slice 5b: see `label_count` — routes through the trait's
    /// `all_edge_types()` for substrate parity.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.data_store().all_edge_types().len()
    }

    // =========================================================================
    // ADMIN API: Introspection
    // =========================================================================

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return false.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.config.path.is_some()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.config.path.as_deref()
    }

    /// Returns high-level database information.
    ///
    /// Includes node/edge counts, persistence status, and mode (LPG/RDF).
    #[must_use]
    pub fn info(&self) -> crate::admin::DatabaseInfo {
        crate::admin::DatabaseInfo {
            mode: crate::admin::DatabaseMode::Lpg,
            node_count: self.data_store().node_count(),
            edge_count: self.data_store().edge_count(),
            is_persistent: self.is_persistent(),
            path: self.config.path.clone(),
            wal_enabled: self.config.wal_enabled,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Returns a hierarchical memory usage breakdown.
    ///
    /// Walks all internal structures (store, indexes, MVCC chains, caches,
    /// string pools, buffer manager) and returns estimated heap bytes for each.
    /// Safe to call concurrently with queries.
    #[must_use]
    pub fn memory_usage(&self) -> crate::memory_usage::MemoryUsage {
        use crate::memory_usage::{BufferManagerMemory, CacheMemory, MemoryUsage};
        use obrain_common::memory::MemoryRegion;
        use obrain_common::memory::usage::{
            IndexMemory, MvccMemory, StoreMemory, StringPoolMemory,
        };

        // T17 W3c slice 5b: `LpgStore::memory_breakdown()` walks LpgStore-specific
        // in-memory structures (node/edge VersionChain maps, MVCC chains, property
        // column DashMaps). In substrate mode storage is mmap-backed zones on disk
        // plus small in-RAM metadata; those categories don't map cleanly to
        // StoreMemory/MvccMemory. Until substrate grows a parallel breakdown,
        // return zeroed sub-totals so admin callers see "unknown" rather than
        // a silently-wrong empty-LpgStore report.
        //
        // TODO(T17 follow-up): add `SubstrateStore::memory_breakdown()` that
        // reports zone footprints (NodesZone, EdgesZone, PropsZone, WAL tail)
        // and plumb it here.
        let (store, indexes, mvcc, string_pool) = if self.substrate_store.is_some() {
            (
                StoreMemory::default(),
                IndexMemory::default(),
                MvccMemory::default(),
                StringPoolMemory::default(),
            )
        } else {
            self.store.memory_breakdown()
        };

        let (parsed_bytes, optimized_bytes, cached_plan_count) =
            self.query_cache.heap_memory_bytes();
        let mut caches = CacheMemory {
            parsed_plan_cache_bytes: parsed_bytes,
            optimized_plan_cache_bytes: optimized_bytes,
            cached_plan_count,
            ..Default::default()
        };
        caches.compute_total();

        let bm_stats = self.buffer_manager.stats();
        let buffer_manager = BufferManagerMemory {
            budget_bytes: bm_stats.budget,
            allocated_bytes: bm_stats.total_allocated,
            graph_storage_bytes: bm_stats.region_usage(MemoryRegion::GraphStorage),
            index_buffers_bytes: bm_stats.region_usage(MemoryRegion::IndexBuffers),
            execution_buffers_bytes: bm_stats.region_usage(MemoryRegion::ExecutionBuffers),
            spill_staging_bytes: bm_stats.region_usage(MemoryRegion::SpillStaging),
        };

        let mut usage = MemoryUsage {
            store,
            indexes,
            mvcc,
            caches,
            string_pool,
            buffer_manager,
            ..Default::default()
        };
        usage.compute_total();
        usage
    }

    /// Returns detailed database statistics.
    ///
    /// Includes counts, memory usage, and index information.
    #[must_use]
    pub fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        #[cfg(feature = "wal")]
        let disk_bytes = self.config.path.as_ref().and_then(|p| {
            if p.exists() {
                Self::calculate_disk_usage(p).ok()
            } else {
                None
            }
        });
        #[cfg(not(feature = "wal"))]
        let disk_bytes: Option<usize> = None;

        // T17 W3c slice 5b: route schema-cardinality reads through data_store()
        // so substrate mode reports real counts (LpgStore-inherent methods
        // target the dummy empty store in substrate mode).
        let data = self.data_store();
        crate::admin::DatabaseStats {
            node_count: data.node_count(),
            edge_count: data.edge_count(),
            label_count: data.all_labels().len(),
            edge_type_count: data.all_edge_types().len(),
            property_key_count: data.all_property_keys().len(),
            index_count: self.catalog.index_count(),
            memory_bytes: self.buffer_manager.allocated(),
            disk_bytes,
        }
    }

    /// Calculates total disk usage for the database directory.
    #[cfg(feature = "wal")]
    fn calculate_disk_usage(path: &Path) -> Result<usize> {
        let mut total = 0usize;
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total += metadata.len() as usize;
                } else if metadata.is_dir() {
                    total += Self::calculate_disk_usage(&entry.path())?;
                }
            }
        }
        Ok(total)
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// For LPG mode, returns label and edge type information.
    /// For RDF mode, returns predicate and named graph information.
    #[must_use]
    pub fn schema(&self) -> crate::admin::SchemaInfo {
        // T17 W3c slice 5a: trait reads route through data_store() so substrate
        // mode returns real labels/types/keys (previously self.store was the
        // dummy LpgStore → empty results). Per-label counts use the trait's
        // node_count_by_label (substrate overrides with O(1) label-index).
        // Edge-type counts still go through self.store.edges_with_type() —
        // no trait equivalent yet; slice 5b adds substrate parity.
        let data = self.data_store();
        let labels = data
            .all_labels()
            .into_iter()
            .map(|name| crate::admin::LabelInfo {
                count: data.node_count_by_label(&name),
                name,
            })
            .collect();

        let edge_types = data
            .all_edge_types()
            .into_iter()
            .map(|name| crate::admin::EdgeTypeInfo {
                // TODO(T17 W3c slice 5b): add edge_count_by_type to the trait
                // so substrate-backed ObrainDB can report non-zero counts here.
                count: self.store.edges_with_type(&name).len(),
                name,
            })
            .collect();

        let property_keys = data.all_property_keys();

        crate::admin::SchemaInfo::Lpg(crate::admin::LpgSchemaInfo {
            labels,
            edge_types,
            property_keys,
        })
    }

    /// Returns detailed information about all indexes.
    #[must_use]
    pub fn list_indexes(&self) -> Vec<crate::admin::IndexInfo> {
        self.catalog
            .all_indexes()
            .into_iter()
            .map(|def| {
                let label_name = self
                    .catalog
                    .get_label_name(def.label)
                    .unwrap_or_else(|| "?".into());
                let prop_name = self
                    .catalog
                    .get_property_key_name(def.property_key)
                    .unwrap_or_else(|| "?".into());
                crate::admin::IndexInfo {
                    name: format!("idx_{}_{}", label_name, prop_name),
                    index_type: format!("{:?}", def.index_type),
                    target: format!("{}:{}", label_name, prop_name),
                    unique: false,
                    cardinality: None,
                    size_bytes: None,
                }
            })
            .collect()
    }

    /// Validates database integrity.
    ///
    /// Checks for:
    /// - Dangling edge references (edges pointing to non-existent nodes)
    /// - Internal index consistency
    ///
    /// Returns a list of errors and warnings. Empty errors = valid.
    #[must_use]
    pub fn validate(&self) -> crate::admin::ValidationResult {
        let mut result = crate::admin::ValidationResult::default();

        // T17 W3c slice 5b: dangling-edge scan relies on LpgStore-inherent
        // `all_edges()` iterator. Substrate's edge topology lives in mmap'd
        // zones accessed by slot index; there is no trait-level iterator yet.
        // In substrate mode we bail cleanly — substrate's replay+checksum path
        // already validates on-disk edge integrity, so the admin validator is
        // informational rather than a correctness primitive there.
        if self.substrate_store.is_some() {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "VALIDATE_SUBSTRATE_SKIPPED".to_string(),
                message: "Dangling-edge validator not implemented for substrate backend; \
                          on-disk integrity is verified by WAL replay + zone checksums"
                    .to_string(),
                context: None,
            });
            return result;
        }

        // Check for dangling edge references
        for edge in self.store.all_edges() {
            if self.data_store().get_node(edge.src).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_SRC".to_string(),
                    message: format!(
                        "Edge {} references non-existent source node {}",
                        edge.id.0, edge.src.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
            if self.data_store().get_node(edge.dst).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_DST".to_string(),
                    message: format!(
                        "Edge {} references non-existent destination node {}",
                        edge.id.0, edge.dst.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
        }

        // Add warnings for potential issues
        if self.data_store().node_count() > 0 && self.data_store().edge_count() == 0 {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "NO_EDGES".to_string(),
                message: "Database has nodes but no edges".to_string(),
                context: None,
            });
        }

        result
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns None if WAL is not enabled.
    #[must_use]
    pub fn wal_status(&self) -> crate::admin::WalStatus {
        // W3a: route through data_store() so substrate-backed DBs report the
        // real substrate epoch rather than the dummy LpgStore's zero epoch.
        // See gotcha note 0b9fcabe-a780-4149-8709-ed32ee9ed82e.
        let current_epoch = self.data_store().current_epoch().as_u64();
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            return crate::admin::WalStatus {
                enabled: true,
                path: self.config.path.as_ref().map(|p| p.join("wal")),
                size_bytes: wal.size_bytes(),
                record_count: wal.record_count() as usize,
                last_checkpoint: wal.last_checkpoint_timestamp(),
                current_epoch,
            };
        }

        crate::admin::WalStatus {
            enabled: false,
            path: None,
            size_bytes: 0,
            record_count: 0,
            last_checkpoint: None,
            current_epoch,
        }
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails.
    pub fn wal_checkpoint(&self) -> Result<()> {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            // W3a: route through data_store() — substrate epoch in substrate mode.
            let epoch = self.data_store().current_epoch();
            let transaction_id = self
                .transaction_manager
                .last_assigned_transaction_id()
                .unwrap_or_else(|| self.transaction_manager.begin());
            wal.checkpoint(transaction_id, epoch)?;
            wal.sync()?;
        }

        // For single-file format: flush snapshot to .obrain file
        #[cfg(feature = "obrain-file")]
        if let Some(ref fm) = self.file_manager {
            self.checkpoint_to_file(fm)?;
        }

        Ok(())
    }

    /// Prunes old WAL log files that have been fully checkpointed.
    ///
    /// This removes WAL files whose sequence number is below the last
    /// checkpoint, freeing disk space. The current active log file is
    /// always preserved.
    ///
    /// Returns the number of bytes freed (approximate, based on size before prune).
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL is not enabled or pruning fails.
    pub fn prune_wal(&self) -> Result<usize> {
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            let size_before = wal.size_bytes();
            wal.prune_old_logs()?;
            let size_after = wal.size_bytes();
            return Ok(size_before.saturating_sub(size_after));
        }

        Ok(0)
    }

    // =========================================================================
    // ADMIN API: Change Data Capture
    // =========================================================================

    /// Returns the full change history for an entity (node or edge).
    ///
    /// Events are ordered chronologically by epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if the CDC feature is not enabled.
    #[cfg(feature = "cdc")]
    pub fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history(entity_id.into()))
    }

    /// Returns change events for an entity since the given epoch.
    #[cfg(feature = "cdc")]
    pub fn history_since(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        since_epoch: obrain_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history_since(entity_id.into(), since_epoch))
    }

    /// Returns all change events across all entities in an epoch range.
    #[cfg(feature = "cdc")]
    pub fn changes_between(
        &self,
        start_epoch: obrain_common::types::EpochId,
        end_epoch: obrain_common::types::EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}
