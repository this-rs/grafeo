//! EdgeAnnotator implementation for ObrainDB.
//!
//! Stores f64 annotations on edges using a dedicated property namespace (`__ann:`).
//! Annotations are persisted through the existing WAL and snapshot infrastructure
//! by mapping them to regular edge properties with a reserved prefix.
//!
//! This keeps the engine agnostic — it knows edges can carry f64 metadata,
//! not that they're pheromones.

use obrain_cognitive::EdgeAnnotator;
use obrain_common::types::{EdgeId, Value};

/// Prefix for annotation properties. Keeps them separate from user properties.
const ANNOTATION_PREFIX: &str = "__ann:";

impl EdgeAnnotator for super::ObrainDB {
    /// Write an annotation on an edge. Persisted to WAL via `set_edge_property`.
    fn annotate(&self, edge: EdgeId, key: &str, value: f64) {
        let prop_key = format!("{ANNOTATION_PREFIX}{key}");
        self.set_edge_property(edge, &prop_key, Value::Float64(value));
    }

    /// Read an annotation from an edge. Returns `None` if the edge or key
    /// doesn't exist.
    fn get_annotation(&self, edge: EdgeId, key: &str) -> Option<f64> {
        let prop_key = format!("{ANNOTATION_PREFIX}{key}");
        // Route via the real backend so annotation reads see substrate edge
        // properties (not the dummy LpgStore) when T17 substrate mode is on.
        let edge = self.data_store().get_edge(edge)?;
        edge.get_property(&prop_key)?.as_float64()
    }

    /// Remove an annotation from an edge.
    fn remove_annotation(&self, edge: EdgeId, key: &str) {
        let prop_key = format!("{ANNOTATION_PREFIX}{key}");
        self.remove_edge_property(edge, &prop_key);
    }
}

/// Returns `true` if the property key is a cognitive annotation (starts with `__ann:`).
///
/// Useful for filtering annotations out of user-facing property listings.
#[must_use]
#[allow(dead_code)]
pub fn is_annotation_key(key: &str) -> bool {
    key.starts_with(ANNOTATION_PREFIX)
}

/// Strips the `__ann:` prefix from an annotation property key.
///
/// Returns `None` if the key doesn't start with the annotation prefix.
#[must_use]
#[allow(dead_code)]
pub fn strip_annotation_prefix(key: &str) -> Option<&str> {
    key.strip_prefix(ANNOTATION_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObrainDB;

    #[test]
    fn test_edge_annotator_basic() {
        let db = ObrainDB::new_in_memory();
        let a = db.create_node(&["A"]);
        let b = db.create_node(&["B"]);
        let edge = db.create_edge(a, b, "LINK");

        // No annotation yet
        assert_eq!(db.get_annotation(edge, "pheromone_query"), None);

        // Write and read back
        db.annotate(edge, "pheromone_query", 0.75);
        assert_eq!(db.get_annotation(edge, "pheromone_query"), Some(0.75));

        // Overwrite
        db.annotate(edge, "pheromone_query", 1.5);
        assert_eq!(db.get_annotation(edge, "pheromone_query"), Some(1.5));

        // Multiple keys on same edge
        db.annotate(edge, "pheromone_error", 0.3);
        assert_eq!(db.get_annotation(edge, "pheromone_query"), Some(1.5));
        assert_eq!(db.get_annotation(edge, "pheromone_error"), Some(0.3));

        // Remove
        db.remove_annotation(edge, "pheromone_query");
        assert_eq!(db.get_annotation(edge, "pheromone_query"), None);
        assert_eq!(db.get_annotation(edge, "pheromone_error"), Some(0.3));
    }

    #[test]
    fn test_edge_annotator_nonexistent_edge() {
        let db = ObrainDB::new_in_memory();
        assert_eq!(db.get_annotation(EdgeId::new(999), "key"), None);
    }

    #[test]
    fn test_annotation_key_helpers() {
        assert!(is_annotation_key("__ann:pheromone_query"));
        assert!(!is_annotation_key("weight"));
        assert!(!is_annotation_key("__ann")); // no colon

        assert_eq!(
            strip_annotation_prefix("__ann:pheromone_query"),
            Some("pheromone_query")
        );
        assert_eq!(strip_annotation_prefix("weight"), None);
    }
}
