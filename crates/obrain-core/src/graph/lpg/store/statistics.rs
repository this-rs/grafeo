use super::LpgStore;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl LpgStore {
    // === Statistics ===

    /// Returns the current statistics (cheap `Arc` clone, no deep copy).
    #[must_use]
    pub fn statistics(&self) -> Arc<Statistics> {
        Arc::clone(&self.statistics.read())
    }

    /// Recomputes statistics if they are stale (i.e., after mutations).
    ///
    /// Call this before reading statistics for query optimization.
    /// Avoids redundant recomputation if no mutations occurred.
    #[doc(hidden)]
    pub fn ensure_statistics_fresh(&self) {
        if self.needs_stats_recompute.swap(false, Ordering::Relaxed) {
            self.recompute_statistics_full();
        } else {
            self.compute_statistics();
        }
    }

    /// Recomputes statistics from incremental counters.
    ///
    /// Reads live node/edge counts from atomic counters and per-label counts
    /// from the label index. This is O(|labels| + |edge_types|) instead of
    /// O(n + m) for a full scan.
    pub(crate) fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Read total counts from atomic counters
        stats.total_nodes = self.live_node_count.load(Ordering::Relaxed).max(0) as u64;
        stats.total_edges = self.live_edge_count.load(Ordering::Relaxed).max(0) as u64;

        // Compute per-label statistics from label_index (each is O(1) via .len())
        let id_to_label = self.id_to_label.read();
        let label_index = self.label_index.read();

        for (label_id, label_name) in id_to_label.iter().enumerate() {
            let node_count = label_index.get(label_id).map_or(0, |set| set.len() as u64);

            if node_count > 0 {
                let avg_out_degree = if stats.total_nodes > 0 {
                    stats.total_edges as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let label_stats =
                    LabelStatistics::new(node_count).with_degrees(avg_out_degree, avg_out_degree);

                stats.update_label(label_name.as_ref(), label_stats);
            }
        }

        // Compute per-edge-type statistics from incremental counts
        let id_to_edge_type = self.id_to_edge_type.read();
        let edge_type_counts = self.edge_type_live_counts.read();

        for (type_id, type_name) in id_to_edge_type.iter().enumerate() {
            let count = edge_type_counts.get(type_id).copied().unwrap_or(0).max(0) as u64;

            if count > 0 {
                let avg_degree = if stats.total_nodes > 0 {
                    count as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let edge_stats = EdgeTypeStatistics::new(count, avg_degree, avg_degree);
                stats.update_edge_type(type_name.as_ref(), edge_stats);
            }
        }

        *self.statistics.write() = Arc::new(stats);
    }

    /// Full recomputation from storage: used after rollback when counters
    /// may be out of sync. Also resyncs the atomic counters.
    #[cfg(not(feature = "tiered-storage"))]
    fn recompute_statistics_full(&self) {
        let epoch = self.current_epoch();

        // Full-scan node count
        let total_nodes = self
            .nodes
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count();

        // Full-scan edge count and per-type counts
        let edges = self.edges.read();
        let mut total_edges: i64 = 0;
        let id_to_edge_type = self.id_to_edge_type.read();
        let mut type_counts = vec![0i64; id_to_edge_type.len()];

        for chain in edges.values() {
            if let Some(record) = chain.visible_at(epoch)
                && !record.is_deleted()
            {
                total_edges += 1;
                if (record.type_id as usize) < type_counts.len() {
                    type_counts[record.type_id as usize] += 1;
                }
            }
        }

        // Resync the atomic counters
        self.live_node_count
            .store(total_nodes as i64, Ordering::Relaxed);
        self.live_edge_count.store(total_edges, Ordering::Relaxed);
        *self.edge_type_live_counts.write() = type_counts;

        drop(edges);
        drop(id_to_edge_type);

        // Now use the normal incremental path to build statistics
        self.compute_statistics();
    }

    /// Full recomputation from storage: used after rollback when counters
    /// may be out of sync. Also resyncs the atomic counters.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    fn recompute_statistics_full(&self) {
        let epoch = self.current_epoch();

        // Full-scan node count
        let versions = self.node_versions.read();
        let total_nodes = versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_node_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count();
        drop(versions);

        // Full-scan edge count and per-type counts
        let edge_versions = self.edge_versions.read();
        let id_to_edge_type = self.id_to_edge_type.read();
        let mut total_edges: i64 = 0;
        let mut type_counts = vec![0i64; id_to_edge_type.len()];

        for index in edge_versions.values() {
            if let Some(vref) = index.visible_at(epoch)
                && let Some(record) = self.read_edge_record(&vref)
                && !record.is_deleted()
            {
                total_edges += 1;
                if (record.type_id as usize) < type_counts.len() {
                    type_counts[record.type_id as usize] += 1;
                }
            }
        }

        // Resync the atomic counters
        self.live_node_count
            .store(total_nodes as i64, Ordering::Relaxed);
        self.live_edge_count.store(total_edges, Ordering::Relaxed);
        *self.edge_type_live_counts.write() = type_counts;

        drop(edge_versions);
        drop(id_to_edge_type);

        // Now use the normal incremental path to build statistics
        self.compute_statistics();
    }

    /// Estimates cardinality for a label scan.
    #[must_use]
    pub fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.statistics.read().estimate_label_cardinality(label)
    }

    /// Estimates average degree for an edge type.
    #[must_use]
    pub fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.statistics
            .read()
            .estimate_avg_degree(edge_type, outgoing)
    }
}
