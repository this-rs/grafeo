use super::LpgStore;
use crate::graph::lpg::{EdgeRecord, NodeRecord};
use obrain_common::memory::AllocError;
use obrain_common::types::{EdgeId, EpochId, NodeId, TransactionId};
#[cfg(feature = "tiered-storage")]
use obrain_common::utils::hash::FxHashMap;
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use obrain_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use obrain_common::mvcc::{ColdVersionRef, HotVersionRef, VersionIndex};

impl LpgStore {
    /// Discards all uncommitted versions created by a transaction.
    ///
    /// This is called during transaction rollback to clean up uncommitted changes.
    /// The method removes version chain entries created by the specified transaction,
    /// and replays the property undo log to restore property values.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        // Remove uncommitted node versions
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.remove_versions_by(transaction_id);
            }
            // Remove completely empty chains (no versions left)
            nodes.retain(|_, chain| !chain.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.remove_versions_by(transaction_id);
            }
            // Remove completely empty chains (no versions left)
            edges.retain(|_, chain| !chain.is_empty());
        }

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities created by a transaction.
    ///
    /// Used for savepoint rollback: only reverts the entities written after
    /// the savepoint, keeping earlier writes intact.
    #[doc(hidden)]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        if !node_ids.is_empty() {
            let mut nodes = self.nodes.write();
            for &nid in node_ids {
                if let Some(chain) = nodes.get_mut(&nid) {
                    chain.remove_versions_by(transaction_id);
                    if chain.is_empty() {
                        nodes.remove(&nid);
                    }
                }
            }
        }

        if !edge_ids.is_empty() {
            let mut edges = self.edges.write();
            for &eid in edge_ids {
                if let Some(chain) = edges.get_mut(&eid) {
                    chain.remove_versions_by(transaction_id);
                    if chain.is_empty() {
                        edges.remove(&eid);
                    }
                }
            }
        }

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards all uncommitted versions created by a transaction.
    /// (Tiered storage version)
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_uncommitted_versions(&self, transaction_id: TransactionId) {
        // Remove uncommitted node versions
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(transaction_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(transaction_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }

        // Replay property undo log to restore pre-transaction property values
        self.rollback_transaction_properties(transaction_id);

        // Counters may be out of sync after rollback: force full recompute
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Discards uncommitted versions for specific entities (tiered storage version).
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    pub fn discard_entities_by_id(
        &self,
        transaction_id: TransactionId,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
    ) {
        if !node_ids.is_empty() {
            let mut versions = self.node_versions.write();
            for &nid in node_ids {
                if let Some(index) = versions.get_mut(&nid) {
                    index.remove_versions_by(transaction_id);
                    if index.is_empty() {
                        versions.remove(&nid);
                    }
                }
            }
        }

        if !edge_ids.is_empty() {
            let mut versions = self.edge_versions.write();
            for &eid in edge_ids {
                if let Some(index) = versions.get_mut(&eid) {
                    index.remove_versions_by(transaction_id);
                    if index.is_empty() {
                        versions.remove(&eid);
                    }
                }
            }
        }

        self.needs_stats_recompute.store(true, Ordering::Relaxed);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    ///
    /// Called at commit time: updates `created_epoch` from `EpochId::PENDING`
    /// to the real `commit_epoch`, making the versions visible to other sessions.
    /// Also advances the store's epoch so non-transactional reads can see the
    /// newly committed versions.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        #[cfg(feature = "temporal")]
        {
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Finalizes PENDING epochs for all versions created by a transaction.
    /// (Tiered storage version, also syncs the store epoch.)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn finalize_version_epochs(&self, transaction_id: TransactionId, commit_epoch: EpochId) {
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.finalize_epochs(transaction_id, commit_epoch);
            }
        }

        // Finalize PENDING epochs in property and label version logs
        #[cfg(feature = "temporal")]
        {
            self.node_properties.finalize_pending(commit_epoch);
            self.edge_properties.finalize_pending(commit_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.finalize_pending(commit_epoch);
            }
        }

        self.sync_epoch(commit_epoch);
    }

    /// Garbage collects old versions that are no longer visible to any transaction.
    ///
    /// Versions older than `min_epoch` are pruned from version chains, keeping
    /// at most one old version per entity as a baseline. Empty chains are removed.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.gc(min_epoch);
            }
            nodes.retain(|_, chain| !chain.is_empty());
        }
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.gc(min_epoch);
            }
            edges.retain(|_, chain| !chain.is_empty());
        }

        // GC old property and label versions
        #[cfg(feature = "temporal")]
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.gc(min_epoch);
            }
            labels.retain(|_, log| !log.is_empty());
        }
    }

    /// Garbage collects old versions (tiered storage variant).
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn gc_versions(&self, min_epoch: EpochId) {
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.gc(min_epoch);
            }
            versions.retain(|_, index| !index.is_empty());
        }
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.gc(min_epoch);
            }
            versions.retain(|_, index| !index.is_empty());
        }

        // GC old property and label versions
        #[cfg(feature = "temporal")]
        {
            self.node_properties.gc(min_epoch);
            self.edge_properties.gc(min_epoch);
            let mut labels = self.node_labels.write();
            for log in labels.values_mut() {
                log.gc(min_epoch);
            }
            labels.retain(|_, log| !log.is_empty());
        }
    }

    /// Freezes an epoch from hot (arena) storage to cold (compressed) storage.
    ///
    /// This is called by the transaction manager when an epoch becomes eligible
    /// for freezing (no active transactions can see it). The freeze process:
    ///
    /// 1. Collects all hot version refs for the epoch
    /// 2. Reads the corresponding records from arena
    /// 3. Compresses them into a `CompressedEpochBlock`
    /// 4. Updates `VersionIndex` entries to point to cold storage
    /// 5. The arena can be deallocated after all epochs in it are frozen
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch to freeze
    ///
    /// # Returns
    ///
    /// The number of records frozen (nodes + edges).
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub fn freeze_epoch(&self, epoch: EpochId) -> usize {
        // Collect node records to freeze
        let mut node_records: Vec<(u64, NodeRecord)> = Vec::new();
        let mut node_hot_refs: Vec<(NodeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.node_versions.read();
            for (node_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                    let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    node_records.push((node_id.as_u64(), *record));
                    node_hot_refs.push((*node_id, *hot_ref));
                }
            }
        }

        // Collect edge records to freeze
        let mut edge_records: Vec<(u64, EdgeRecord)> = Vec::new();
        let mut edge_hot_refs: Vec<(EdgeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.edge_versions.read();
            for (edge_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist for hot version ref");
                    // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                    let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    edge_records.push((edge_id.as_u64(), *record));
                    edge_hot_refs.push((*edge_id, *hot_ref));
                }
            }
        }

        let total_frozen = node_records.len() + edge_records.len();

        if total_frozen == 0 {
            return 0;
        }

        // Freeze to compressed storage
        let (node_entries, edge_entries) =
            self.epoch_store
                .freeze_epoch(epoch, node_records, edge_records);

        // Build lookup maps for index entries
        let node_entry_map: FxHashMap<u64, _> = node_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();
        let edge_entry_map: FxHashMap<u64, _> = edge_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();

        // Update version indexes to use cold refs
        {
            let mut versions = self.node_versions.write();
            for (node_id, hot_ref) in &node_hot_refs {
                if let Some(index) = versions.get_mut(node_id)
                    && let Some(&(offset, length)) = node_entry_map.get(&node_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        {
            let mut versions = self.edge_versions.write();
            for (edge_id, hot_ref) in &edge_hot_refs {
                if let Some(index) = versions.get_mut(edge_id)
                    && let Some(&(offset, length)) = edge_entry_map.get(&edge_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                        deleted_by: hot_ref.deleted_by,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        total_frozen
    }

    /// Returns the epoch store for cold storage statistics.
    #[doc(hidden)]
    #[cfg(feature = "tiered-storage")]
    #[must_use]
    pub fn epoch_store(&self) -> &crate::storage::EpochStore {
        &self.epoch_store
    }

    // === Recovery Support ===

    /// Creates a node with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore nodes with their original IDs.
    /// The caller must ensure IDs don't conflict with existing nodes.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot allocate space
    /// (only possible with the `tiered-storage` feature).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, epoch);

        // Create version chain with initial version (using SYSTEM tx for recovery)
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        self.nodes.write().insert(id, chain);
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates a node with a specific ID during recovery.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot create an epoch
    /// or allocate space for the node record.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        #[cfg(not(feature = "temporal"))]
        self.register_node_labels(id, labels);
        #[cfg(feature = "temporal")]
        self.register_node_labels(id, labels, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let mut versions = self.node_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));
        self.live_node_count.fetch_add(1, Ordering::Relaxed);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates an edge with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore edges with their original IDs.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot allocate space
    /// (only possible with the `tiered-storage` feature).
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        self.edges.write().insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Creates an edge with a specific ID during recovery.
    /// (Tiered storage version)
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot create an epoch
    /// or allocate space for the edge record.
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_edge_with_id(
        &self,
        id: EdgeId,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> Result<(), AllocError> {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch)?;
        let (offset, _stored) = arena.alloc_value_with_offset(record)?;

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, epoch, offset, TransactionId::SYSTEM);
        let mut versions = self.edge_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        self.live_edge_count.fetch_add(1, Ordering::Relaxed);
        self.increment_edge_type_count(type_id);

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
        Ok(())
    }

    /// Sets the current epoch during recovery.
    #[doc(hidden)]
    pub fn set_epoch(&self, epoch: EpochId) {
        self.current_epoch.store(epoch.as_u64(), Ordering::SeqCst);
    }

    /// Snapshots ALL current store data into epoch files on disk.
    ///
    /// This is the core of the `compact` operation: it reads all nodes, edges,
    /// properties, labels, and adjacency from the in-memory store and writes
    /// them to epoch files that can be mmap'd on next startup.
    ///
    /// The persist directory must have been set on the epoch store before calling.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails or persist directory is not configured.
    #[cfg(feature = "tiered-storage")]
    pub fn snapshot_to_epoch_file(
        &self,
        wal_sequence: u64,
    ) -> std::io::Result<std::path::PathBuf> {
        use crate::storage::epoch_store::IndexEntry;
        use crate::storage::mmap_epoch::EpochFileData;

        let epoch = self.current_epoch();
        let config = bincode::config::standard();

        // =====================================================================
        // 1. Collect node records from version indexes
        // =====================================================================
        let mut node_records: Vec<(u64, NodeRecord)> = Vec::new();
        {
            let versions = self.node_versions.read();
            for (node_id, index) in versions.iter() {
                // Get the latest version from hot refs
                if let Some(hot_ref) = index.latest_hot() {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist");
                    // SAFETY: read_at for a NodeRecord previously allocated
                    #[allow(unsafe_code)]
                    let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    if !record.is_deleted() {
                        node_records.push((node_id.as_u64(), *record));
                    }
                }
            }
        }
        node_records.sort_unstable_by_key(|(id, _)| *id);

        // =====================================================================
        // 2. Collect edge records from version indexes
        // =====================================================================
        let mut edge_records: Vec<(u64, EdgeRecord)> = Vec::new();
        {
            let versions = self.edge_versions.read();
            for (edge_id, index) in versions.iter() {
                if let Some(hot_ref) = index.latest_hot() {
                    let arena = self
                        .arena_allocator
                        .arena(hot_ref.arena_epoch)
                        .expect("arena epoch must exist");
                    #[allow(unsafe_code)]
                    let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    if !record.is_deleted() {
                        edge_records.push((edge_id.as_u64(), *record));
                    }
                }
            }
        }
        edge_records.sort_unstable_by_key(|(id, _)| *id);

        // =====================================================================
        // 3. Serialize node/edge data + build index entries
        // =====================================================================
        let mut node_data = Vec::new();
        let mut node_index = Vec::with_capacity(node_records.len());
        for (id, record) in &node_records {
            let offset = node_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config)
                .expect("NodeRecord serialization should not fail");
            let length = serialized.len() as u16;
            node_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
                _pad: 0,
            });
            node_data.extend_from_slice(&serialized);
        }

        let mut edge_data = Vec::new();
        let mut edge_index = Vec::with_capacity(edge_records.len());
        for (id, record) in &edge_records {
            let offset = edge_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config)
                .expect("EdgeRecord serialization should not fail");
            let length = serialized.len() as u16;
            edge_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
                _pad: 0,
            });
            edge_data.extend_from_slice(&serialized);
        }

        // =====================================================================
        // 4. Collect properties
        // =====================================================================
        // Serialize as Vec<(entity_id, Vec<(key_str, value)>)> via bincode
        let mut property_entries: Vec<(u64, Vec<(String, obrain_common::types::Value)>)> =
            Vec::new();

        // Node properties
        for (id, _) in &node_records {
            let node_id = NodeId::new(*id);
            let props = self.node_properties.get_all(node_id);
            if !props.is_empty() {
                let entries: Vec<(String, obrain_common::types::Value)> = props
                    .into_iter()
                    .map(|(k, v)| (k.as_str().to_string(), v))
                    .collect();
                property_entries.push((*id, entries));
            }
        }

        // Edge properties (tagged with high bit to distinguish from node IDs)
        // We use a separate section marker: first all node props, then a separator,
        // then edge props. Or we can just serialize them as two separate arrays.
        let mut edge_property_entries: Vec<(u64, Vec<(String, obrain_common::types::Value)>)> =
            Vec::new();
        for (id, _) in &edge_records {
            let edge_id = EdgeId::new(*id);
            let props = self.edge_properties.get_all(edge_id);
            if !props.is_empty() {
                let entries: Vec<(String, obrain_common::types::Value)> = props
                    .into_iter()
                    .map(|(k, v)| (k.as_str().to_string(), v))
                    .collect();
                edge_property_entries.push((*id, entries));
            }
        }

        // Serialize both as a single blob: (node_props, edge_props)
        let prop_blob = bincode::serde::encode_to_vec(
            &(&property_entries, &edge_property_entries),
            config,
        )
        .expect("property serialization should not fail");

        // =====================================================================
        // 5. Collect labels
        // =====================================================================
        // Serialize as Vec<(node_id, Vec<String>)>
        let mut label_entries: Vec<(u64, Vec<String>)> = Vec::new();
        {
            let node_labels = self.node_labels.read();
            let id_to_label = self.id_to_label.read();
            for (id, _) in &node_records {
                let node_id = NodeId::new(*id);
                #[cfg(not(feature = "temporal"))]
                if let Some(label_ids) = node_labels.get(&node_id) {
                    let labels: Vec<String> = label_ids
                        .iter()
                        .filter_map(|&lid| id_to_label.get(lid as usize).map(|s| s.to_string()))
                        .collect();
                    if !labels.is_empty() {
                        label_entries.push((*id, labels));
                    }
                }
                #[cfg(feature = "temporal")]
                if let Some(version_log) = node_labels.get(&node_id) {
                    if let Some(label_ids) = version_log.latest() {
                        let labels: Vec<String> = label_ids
                            .iter()
                            .filter_map(|&lid| {
                                id_to_label.get(lid as usize).map(|s| s.to_string())
                            })
                            .collect();
                        if !labels.is_empty() {
                            label_entries.push((*id, labels));
                        }
                    }
                }
            }
        }

        // Also serialize the edge type table (needed for reconstructing edges)
        let edge_types: Vec<String> = self
            .id_to_edge_type
            .read()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let label_blob = bincode::serde::encode_to_vec(
            &(&label_entries, &edge_types),
            config,
        )
        .expect("label serialization should not fail");

        // =====================================================================
        // 6. Collect adjacency
        // =====================================================================
        // Serialize as Vec<(node_id, Vec<(dest_node_id, edge_id)>)>
        let mut adj_entries: Vec<(u64, Vec<(u64, u64)>)> = Vec::new();
        for (id, _) in &node_records {
            let node_id = NodeId::new(*id);
            let edges = self.forward_adj.edges_from(node_id);
            if !edges.is_empty() {
                let adj: Vec<(u64, u64)> = edges
                    .iter()
                    .map(|(dst, eid)| (dst.as_u64(), eid.as_u64()))
                    .collect();
                adj_entries.push((*id, adj));
            }
        }

        // Backward adjacency if available
        let backward_entries: Vec<(u64, Vec<(u64, u64)>)> = if let Some(ref bwd) = self.backward_adj
        {
            let mut entries = Vec::new();
            for (id, _) in &node_records {
                let node_id = NodeId::new(*id);
                let edges = bwd.edges_from(node_id);
                if !edges.is_empty() {
                    let adj: Vec<(u64, u64)> = edges
                        .iter()
                        .map(|(dst, eid)| (dst.as_u64(), eid.as_u64()))
                        .collect();
                    entries.push((*id, adj));
                }
            }
            entries
        } else {
            Vec::new()
        };

        let adj_blob = bincode::serde::encode_to_vec(
            &(&adj_entries, &backward_entries),
            config,
        )
        .expect("adjacency serialization should not fail");

        // =====================================================================
        // 7. Build zone map and write epoch file
        // =====================================================================
        let zone_map = crate::storage::epoch_store::ZoneMap {
            min_node_id: node_records.first().map_or(0, |(id, _)| *id),
            max_node_id: node_records.last().map_or(0, |(id, _)| *id),
            min_edge_id: edge_records.first().map_or(0, |(id, _)| *id),
            max_edge_id: edge_records.last().map_or(0, |(id, _)| *id),
            node_count: node_records.len() as u32,
            edge_count: edge_records.len() as u32,
            min_epoch: epoch.as_u64(),
            max_epoch: epoch.as_u64(),
        };

        let file_data = EpochFileData {
            epoch,
            node_index: &node_index,
            node_data: &node_data,
            edge_index: &edge_index,
            edge_data: &edge_data,
            zone_map: &zone_map,
            property_data: Some(&prop_blob),
            label_data: Some(&label_blob),
            adjacency_data: Some(&adj_blob),
        };

        self.epoch_store.persist_epoch_direct(&file_data, wal_sequence)
    }

    /// Restores the store from mmap'd epoch files.
    ///
    /// This is called during database startup after loading epoch files.
    /// It rebuilds the in-memory indexes (version indexes, labels, adjacency,
    /// ID counters) from the mmap'd data.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    #[cfg(feature = "tiered-storage")]
    pub fn restore_from_epoch_files(&self) -> std::io::Result<()> {
        let mmap_blocks = self.epoch_store.mmap_blocks().read();

        let mut max_node_id: u64 = 0;
        let mut max_edge_id: u64 = 0;

        for (epoch_id, block) in mmap_blocks.iter() {
            let config = bincode::config::standard();

            // =====================================================================
            // 1. Rebuild version indexes from node/edge index entries
            // =====================================================================
            {
                let mut versions = self.node_versions.write();
                for entry in block.node_index() {
                    let node_id = NodeId::new(entry.entity_id);
                    if entry.entity_id > max_node_id {
                        max_node_id = entry.entity_id;
                    }
                    let cold_ref = ColdVersionRef {
                        epoch: *epoch_id,
                        block_offset: entry.offset,
                        length: entry.length,
                        created_by: TransactionId::SYSTEM,
                        deleted_epoch: obrain_common::mvcc::OptionalEpochId::NONE,
                        deleted_by: None,
                    };
                    let index = versions
                        .entry(node_id)
                        .or_insert_with(VersionIndex::new);
                    index.add_cold(cold_ref);
                }
            }

            {
                let mut versions = self.edge_versions.write();
                for entry in block.edge_index() {
                    let edge_id = EdgeId::new(entry.entity_id);
                    if entry.entity_id > max_edge_id {
                        max_edge_id = entry.entity_id;
                    }
                    let cold_ref = ColdVersionRef {
                        epoch: *epoch_id,
                        block_offset: entry.offset,
                        length: entry.length,
                        created_by: TransactionId::SYSTEM,
                        deleted_epoch: obrain_common::mvcc::OptionalEpochId::NONE,
                        deleted_by: None,
                    };
                    let index = versions
                        .entry(edge_id)
                        .or_insert_with(VersionIndex::new);
                    index.add_cold(cold_ref);
                }
            }

            // =====================================================================
            // 2. Restore labels
            // =====================================================================
            if let Some(label_bytes) = block.label_data_section() {
                let result: Result<
                    (Vec<(u64, Vec<String>)>, Vec<String>),
                    _,
                > = bincode::serde::decode_from_slice(label_bytes, config).map(|(v, _)| v);

                if let Ok((label_entries, edge_types)) = result {
                    // Restore edge type table
                    {
                        let mut id_to_edge_type = self.id_to_edge_type.write();
                        let mut edge_type_to_id = self.edge_type_to_id.write();
                        for (idx, etype) in edge_types.into_iter().enumerate() {
                            let arc = arcstr::ArcStr::from(etype);
                            if idx >= id_to_edge_type.len() {
                                edge_type_to_id.insert(arc.clone(), idx as u32);
                                id_to_edge_type.push(arc);
                            }
                        }
                    }

                    // Restore node labels
                    {
                        let mut label_to_id = self.label_to_id.write();
                        let mut id_to_label = self.id_to_label.write();
                        let mut node_labels = self.node_labels.write();
                        let mut label_index = self.label_index.write();

                        for (node_id_raw, labels) in label_entries {
                            let node_id = NodeId::new(node_id_raw);
                            let mut label_ids =
                                obrain_common::utils::hash::FxHashSet::default();

                            for label_str in labels {
                                let label_id = if let Some(&id) = label_to_id.get(label_str.as_str())
                                {
                                    id
                                } else {
                                    let id = id_to_label.len() as u32;
                                    let arc = arcstr::ArcStr::from(label_str);
                                    label_to_id.insert(arc.clone(), id);
                                    id_to_label.push(arc);
                                    // Ensure label_index has enough slots
                                    while label_index.len() <= id as usize {
                                        label_index
                                            .push(obrain_common::utils::hash::FxHashMap::default());
                                    }
                                    id
                                };
                                label_ids.insert(label_id);

                                // Update label index
                                if let Some(node_set) = label_index.get_mut(label_id as usize) {
                                    node_set.insert(node_id, ());
                                }
                            }

                            #[cfg(not(feature = "temporal"))]
                            {
                                node_labels.insert(node_id, label_ids);
                            }
                            #[cfg(feature = "temporal")]
                            {
                                let mut log = obrain_common::temporal::VersionLog::new();
                                log.append(*epoch_id, label_ids);
                                node_labels.insert(node_id, log);
                            }
                        }
                    }
                }
            }

            // =====================================================================
            // 3. Restore properties
            // =====================================================================
            if let Some(prop_bytes) = block.property_data() {
                type PropEntries = Vec<(u64, Vec<(String, obrain_common::types::Value)>)>;
                let result: Result<(PropEntries, PropEntries), _> =
                    bincode::serde::decode_from_slice(prop_bytes, config).map(|(v, _)| v);

                if let Ok((node_props, edge_props)) = result {
                    // Restore node properties
                    for (node_id_raw, props) in node_props {
                        let node_id = NodeId::new(node_id_raw);
                        for (key_str, value) in props {
                            self.node_properties.set(
                                node_id,
                                obrain_common::types::PropertyKey::new(key_str),
                                value,
                            );
                        }
                    }

                    // Restore edge properties
                    for (edge_id_raw, props) in edge_props {
                        let edge_id = EdgeId::new(edge_id_raw);
                        for (key_str, value) in props {
                            self.edge_properties.set(
                                edge_id,
                                obrain_common::types::PropertyKey::new(key_str),
                                value,
                            );
                        }
                    }
                }
            }

            // =====================================================================
            // 4. Restore adjacency
            // =====================================================================
            if let Some(adj_bytes) = block.adjacency_data() {
                type AdjEntries = Vec<(u64, Vec<(u64, u64)>)>;
                let result: Result<(AdjEntries, AdjEntries), _> =
                    bincode::serde::decode_from_slice(adj_bytes, config).map(|(v, _)| v);

                if let Ok((forward_adj, backward_adj)) = result {
                    // Restore forward adjacency
                    for (src_raw, edges) in forward_adj {
                        let src = NodeId::new(src_raw);
                        for (dst_raw, eid_raw) in edges {
                            self.forward_adj.add_edge(
                                src,
                                NodeId::new(dst_raw),
                                EdgeId::new(eid_raw),
                            );
                        }
                    }

                    // Restore backward adjacency
                    if let Some(ref bwd) = self.backward_adj {
                        for (src_raw, edges) in backward_adj {
                            let src = NodeId::new(src_raw);
                            for (dst_raw, eid_raw) in edges {
                                bwd.add_edge(
                                    src,
                                    NodeId::new(dst_raw),
                                    EdgeId::new(eid_raw),
                                );
                            }
                        }
                    }
                }
            }
        }

        // Update ID counters
        if max_node_id > 0 {
            self.next_node_id
                .fetch_max(max_node_id + 1, Ordering::SeqCst);
        }
        if max_edge_id > 0 {
            self.next_edge_id
                .fetch_max(max_edge_id + 1, Ordering::SeqCst);
        }

        // Update live counts
        let node_count = self.node_versions.read().len() as i64;
        let edge_count = self.edge_versions.read().len() as i64;
        self.live_node_count.store(node_count, Ordering::SeqCst);
        self.live_edge_count.store(edge_count, Ordering::SeqCst);

        Ok(())
    }
}
