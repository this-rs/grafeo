use super::LpgStore;
use crate::graph::lpg::{Node, NodeRecord};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::sync::atomic::Ordering;

#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;

#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{HotVersionRef, VersionIndex, VersionRef};

impl LpgStore {
    /// Creates a new node with the given labels.
    ///
    /// Uses the system transaction for non-transactional operations.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.create_node_versioned(labels, self.current_epoch(), TransactionId::SYSTEM)
    }

    /// Registers labels for a node: builds the label ID set, updates the
    /// label index (single lock acquisition), and stores the node-to-labels
    /// mapping.
    pub(super) fn register_node_labels(&self, id: NodeId, labels: &[&str]) {
        let mut node_label_set = FxHashSet::default();
        let mut label_ids = Vec::with_capacity(labels.len());
        for label in labels {
            let label_id = self.get_or_create_label_id(label);
            node_label_set.insert(label_id);
            label_ids.push(label_id);
        }

        // Update label index with a single lock acquisition
        let mut index = self.label_index.write();
        for label_id in label_ids {
            if index.len() <= label_id as usize {
                index.resize_with(label_id as usize + 1, FxHashMap::default);
            }
            index[label_id as usize].insert(id, ());
        }
        drop(index);

        self.node_labels.write().insert(id, node_label_set);
    }

    /// Builds a `Node` populated with labels and properties for the given ID.
    fn build_node(&self, id: NodeId) -> Node {
        let mut node = Node::new(id);

        let id_to_label = self.id_to_label.read();
        let node_labels = self.node_labels.read();
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = id_to_label.get(label_id as usize) {
                    node.labels.push(label.clone());
                }
            }
        }

        node.properties = self.node_properties.get_all(id).into_iter().collect();
        node
    }

    /// Creates a new node with the given labels within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        self.register_node_labels(id, labels);

        let chain = VersionChain::with_initial(record, epoch, transaction_id);
        self.nodes.write().insert(id, chain);
        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Creates a new node with the given labels within a transaction context.
    /// (Tiered storage version: stores data in arena, metadata in VersionIndex)
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        self.register_node_labels(id, labels);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch");
        let (offset, _stored) = arena
            .alloc_value_with_offset(record)
            .expect("arena allocation failed for node record");

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(epoch, offset, transaction_id);

        // Create or update version index
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }

        self.live_node_count.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> NodeId {
        self.create_node_with_props_versioned(
            labels,
            properties,
            self.current_epoch(),
            TransactionId::SYSTEM,
        )
    }

    /// Creates a new node with labels and properties within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, transaction_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            self.node_properties.set(id, prop_key, prop_value);
        }

        // Update props_count in record
        let count = self.node_properties.get_all(id).len() as u16;
        if let Some(chain) = self.nodes.write().get_mut(&id)
            && let Some(record) = chain.latest_mut()
        {
            record.props_count = count;
        }

        id
    }

    /// Creates a new node with labels and properties within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, transaction_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            self.node_properties.set(id, prop_key, prop_value);
        }

        // Note: props_count in record is not updated for tiered storage.
        // The record is immutable once allocated in the arena.

        id
    }

    /// Gets a node by ID (latest visible version).
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.get_node_at_epoch(id, self.current_epoch())
    }

    /// Gets a node by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_at(epoch)?;
        if record.is_deleted() {
            return None;
        }
        drop(nodes);
        Some(self.build_node(id))
    }

    /// Gets a node by ID at a specific epoch.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);
        Some(self.build_node(id))
    }

    /// Gets a node visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    #[doc(hidden)]
    pub fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_to(epoch, transaction_id)?;
        if record.is_deleted() {
            return None;
        }
        drop(nodes);
        Some(self.build_node(id))
    }

    /// Gets a node visible to a specific transaction.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    #[doc(hidden)]
    pub fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, transaction_id)?;
        let record = self.read_node_record(&version_ref)?;
        if record.is_deleted() {
            return None;
        }
        drop(versions);
        Some(self.build_node(id))
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    ///
    /// Each entry is `(created_epoch, deleted_epoch, Node)`. Note that labels and
    /// properties reflect the current state (they are not versioned per-epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let nodes = self.nodes.read();
        let Some(chain) = nodes.get(&id) else {
            return Vec::new();
        };

        // Cache labels and properties once, clone per version entry
        let template = self.build_node(id);

        chain
            .history()
            .map(|(info, _record)| (info.created_epoch, info.deleted_epoch, template.clone()))
            .collect()
    }

    /// Returns all versions of a node with their creation/deletion epochs, newest first.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        let versions = self.node_versions.read();
        let Some(index) = versions.get(&id) else {
            return Vec::new();
        };

        // Cache labels and properties once, clone per version entry
        let template = self.build_node(id);

        index
            .version_history()
            .into_iter()
            .map(|(created, deleted, _vref)| (created, deleted, template.clone()))
            .collect()
    }

    /// Reads a NodeRecord from arena (hot) or epoch store (cold) using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub(super) fn read_node_record(&self, version_ref: &VersionRef) -> Option<NodeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self
                    .arena_allocator
                    .arena(hot_ref.epoch)
                    .expect("epoch must exist for hot version ref");
                // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_node(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
        }
    }

    /// Deletes a node and all its edges (using latest epoch).
    pub fn delete_node(&self, id: NodeId) -> bool {
        self.delete_node_at_epoch(id, self.current_epoch())
    }

    /// Deletes a node at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            // Check if visible at this epoch (not already deleted)
            if let Some(record) = chain.visible_at(epoch) {
                if record.is_deleted() {
                    return false;
                }
            } else {
                // Not visible at this epoch (already deleted or doesn't exist)
                return false;
            }

            // Mark the version chain as deleted at this epoch
            chain.mark_deleted(epoch);

            // Remove from label index using node_labels map
            let mut index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(label_ids) = node_labels.remove(&id) {
                for label_id in label_ids {
                    if let Some(set) = index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove from text indexes before removing properties
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties
            drop(nodes); // Release lock before removing properties
            drop(index);
            drop(node_labels);
            self.node_properties.remove_all(id);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes a node at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub(crate) fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Check if visible at this epoch
            if let Some(version_ref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&version_ref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }

            // Mark as deleted in version index
            index.mark_deleted(epoch);

            // Remove from label index using node_labels map
            let mut label_index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(label_ids) = node_labels.remove(&id) {
                for label_id in label_ids {
                    if let Some(set) = label_index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove from text indexes before removing properties
            #[cfg(feature = "text-index")]
            self.remove_from_all_text_indexes(id);

            // Remove properties
            drop(versions);
            drop(label_index);
            drop(node_labels);
            self.node_properties.remove_all(id);

            self.live_node_count.fetch_sub(1, Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    ///
    /// Call this before `delete_node()` if you want to remove a node that
    /// has edges. Grafeo doesn't auto-delete edges - you have to be explicit.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        // Get outgoing edges
        let outgoing: Vec<EdgeId> = self
            .forward_adj
            .edges_from(node_id)
            .into_iter()
            .map(|(_, edge_id)| edge_id)
            .collect();

        // Get incoming edges
        let incoming: Vec<EdgeId> = if let Some(ref backward) = self.backward_adj {
            backward
                .edges_from(node_id)
                .into_iter()
                .map(|(_, edge_id)| edge_id)
                .collect()
        } else {
            // No backward adjacency - scan all edges
            let epoch = self.current_epoch();
            self.edges
                .read()
                .iter()
                .filter_map(|(id, chain)| {
                    chain.visible_at(epoch).and_then(|r| {
                        if !r.is_deleted() && r.dst == node_id {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        };

        // Delete all edges
        for edge_id in outgoing.into_iter().chain(incoming) {
            self.delete_edge(edge_id);
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        // Get outgoing edges
        let outgoing: Vec<EdgeId> = self
            .forward_adj
            .edges_from(node_id)
            .into_iter()
            .map(|(_, edge_id)| edge_id)
            .collect();

        // Get incoming edges
        let incoming: Vec<EdgeId> = if let Some(ref backward) = self.backward_adj {
            backward
                .edges_from(node_id)
                .into_iter()
                .map(|(_, edge_id)| edge_id)
                .collect()
        } else {
            // No backward adjacency - scan all edges
            let epoch = self.current_epoch();
            let versions = self.edge_versions.read();
            versions
                .iter()
                .filter_map(|(id, index)| {
                    index.visible_at(epoch).and_then(|vref| {
                        self.read_edge_record(&vref).and_then(|r| {
                            if !r.is_deleted() && r.dst == node_id {
                                Some(*id)
                            } else {
                                None
                            }
                        })
                    })
                })
                .collect()
        };

        // Delete all edges
        for edge_id in outgoing.into_iter().chain(incoming) {
            self.delete_edge(edge_id);
        }
    }

    // --- Visibility checks (no label/property loading) ---

    /// Checks if a node is visible at the given epoch.
    ///
    /// Only checks the version chain, skips label and property loading.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let nodes = self.nodes.read();
        nodes
            .get(&id)
            .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
    }

    /// Checks if a node is visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let versions = self.node_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_at(epoch).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Checks if a node is visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let nodes = self.nodes.read();
        nodes.get(&id).is_some_and(|chain| {
            chain
                .visible_to(epoch, transaction_id)
                .is_some_and(|r| !r.is_deleted())
        })
    }

    /// Checks if a node is visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        let versions = self.node_versions.read();
        versions.get(&id).is_some_and(|index| {
            index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                self.read_node_record(&vref)
                    .is_some_and(|r| !r.is_deleted())
            })
        })
    }

    /// Filters node IDs to only those visible at the given epoch.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        ids.iter()
            .copied()
            .filter(|id| {
                nodes
                    .get(id)
                    .is_some_and(|chain| chain.visible_at(epoch).is_some_and(|r| !r.is_deleted()))
            })
            .collect()
    }

    /// Filters node IDs to only those visible at the given epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        ids.iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_at(epoch).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect()
    }

    /// Filters node IDs to only those visible to a specific transaction.
    ///
    /// Holds a single lock for the entire batch instead of per-node locking.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let nodes = self.nodes.read();
        ids.iter()
            .copied()
            .filter(|id| {
                nodes.get(id).is_some_and(|chain| {
                    chain
                        .visible_to(epoch, transaction_id)
                        .is_some_and(|r| !r.is_deleted())
                })
            })
            .collect()
    }

    /// Filters node IDs to only those visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let versions = self.node_versions.read();
        ids.iter()
            .copied()
            .filter(|id| {
                versions.get(id).is_some_and(|index| {
                    index.visible_to(epoch, transaction_id).is_some_and(|vref| {
                        self.read_node_record(&vref)
                            .is_some_and(|r| !r.is_deleted())
                    })
                })
            })
            .collect()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        self.nodes
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_node_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Returns all node IDs in the store.
    ///
    /// This returns a snapshot of current node IDs. The returned vector
    /// excludes deleted nodes. Results are sorted by NodeId for deterministic
    /// iteration order.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let epoch = self.current_epoch();
        let mut ids: Vec<NodeId> = self
            .nodes
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs in the store.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        let mut ids: Vec<NodeId> = versions
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_node_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();
        ids.sort_unstable();
        ids
    }
}
