use super::LpgStore;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use grafeo_common::utils::hash::FxHashMap;
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
    pub fn ensure_statistics_fresh(&self) {
        if self.needs_stats_recompute.swap(false, Ordering::Relaxed) {
            self.compute_statistics();
        }
    }

    /// Recomputes statistics from current data.
    ///
    /// Scans all labels and edge types to build cardinality estimates for the
    /// query optimizer. Call this periodically or after bulk data loads.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Compute total counts
        stats.total_nodes = self.node_count() as u64;
        stats.total_edges = self.edge_count() as u64;

        // Compute per-label statistics
        let id_to_label = self.id_to_label.read();
        let label_index = self.label_index.read();

        for (label_id, label_name) in id_to_label.iter().enumerate() {
            let node_count = label_index.get(label_id).map_or(0, |set| set.len() as u64);

            if node_count > 0 {
                // Estimate average degree
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

        // Compute per-edge-type statistics
        let id_to_edge_type = self.id_to_edge_type.read();
        let edges = self.edges.read();
        let epoch = self.current_epoch();

        let mut edge_type_counts: FxHashMap<u32, u64> = FxHashMap::default();
        for chain in edges.values() {
            if let Some(record) = chain.visible_at(epoch)
                && !record.is_deleted()
            {
                *edge_type_counts.entry(record.type_id).or_default() += 1;
            }
        }

        for (type_id, count) in edge_type_counts {
            if let Some(type_name) = id_to_edge_type.get(type_id as usize) {
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

    /// Recomputes statistics from current data.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Compute total counts
        stats.total_nodes = self.node_count() as u64;
        stats.total_edges = self.edge_count() as u64;

        // Compute per-label statistics
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

        // Compute per-edge-type statistics
        let id_to_edge_type = self.id_to_edge_type.read();
        let versions = self.edge_versions.read();
        let epoch = self.current_epoch();

        let mut edge_type_counts: FxHashMap<u32, u64> = FxHashMap::default();
        for index in versions.values() {
            if let Some(vref) = index.visible_at(epoch)
                && let Some(record) = self.read_edge_record(&vref)
                && !record.is_deleted()
            {
                *edge_type_counts.entry(record.type_id).or_default() += 1;
            }
        }

        for (type_id, count) in edge_type_counts {
            if let Some(type_name) = id_to_edge_type.get(type_id as usize) {
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
