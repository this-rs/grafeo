//! Index management for ObrainDB (property, vector, and text indexes).

#[cfg(any(feature = "vector-index", feature = "text-index"))]
use std::sync::Arc;

#[cfg(feature = "text-index")]
use parking_lot::RwLock;

use obrain_common::utils::error::Result;

impl super::ObrainDB {
    // =========================================================================
    // PROPERTY INDEX API
    // =========================================================================

    /// Creates an index on a node property for O(1) lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use obrain_engine::ObrainDB;
    /// # use obrain_common::types::Value;
    /// # let db = ObrainDB::new_in_memory();
    /// // Create an index on the 'email' property
    /// db.create_property_index("email");
    ///
    /// // Now lookups by email are O(1)
    /// let nodes = db.find_nodes_by_property("email", &Value::from("alix@example.com"));
    /// ```
    pub fn create_property_index(&self, property: &str) {
        // T17 W3c slice 5d: property-index registry lives on the dummy
        // LpgStore in substrate mode (`self.store`). Writing an index there
        // would succeed silently yet never be consulted — reads route through
        // `data_store()` / substrate tiers. Make the no-op explicit.
        if self.substrate_store.is_some() {
            tracing::warn!(
                property,
                "create_property_index is a no-op in substrate mode — the \
                 LpgStore index registry is not consulted by substrate reads. \
                 A substrate-native property index is tracked as T17b."
            );
            return;
        }
        self.store.create_property_index(property);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        if self.substrate_store.is_some() {
            return false; // Substrate has no LpgStore index registry.
        }
        self.store.drop_property_index(property)
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        if self.substrate_store.is_some() {
            return false; // Substrate has no LpgStore index registry.
        }
        self.store.has_property_index(property)
    }

    /// Finds all nodes that have a specific property value.
    ///
    /// If the property is indexed, this is O(1). Otherwise, it scans all nodes
    /// which is O(n). Use [`Self::create_property_index`] for frequently queried properties.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use obrain_engine::ObrainDB;
    /// # use obrain_common::types::Value;
    /// # let db = ObrainDB::new_in_memory();
    /// // Create index for fast lookups (optional but recommended)
    /// db.create_property_index("city");
    ///
    /// // Find all nodes where city = "NYC"
    /// let nyc_nodes = db.find_nodes_by_property("city", &Value::from("NYC"));
    /// ```
    #[must_use]
    pub fn find_nodes_by_property(
        &self,
        property: &str,
        value: &obrain_common::types::Value,
    ) -> Vec<obrain_common::types::NodeId> {
        self.data_store().find_nodes_by_property(property, value)
    }

    // =========================================================================
    // VECTOR INDEX API
    // =========================================================================

    /// Creates a vector similarity index on a node property.
    ///
    /// This enables efficient approximate nearest-neighbor search on vector
    /// properties. Currently validates the index parameters and scans existing
    /// nodes to verify the property contains vectors of the expected dimensions.
    ///
    /// # Arguments
    ///
    /// * `label` - Node label to index (e.g., `"Doc"`)
    /// * `property` - Property containing vector embeddings (e.g., `"embedding"`)
    /// * `dimensions` - Expected vector dimensions (inferred from data if `None`)
    /// * `metric` - Distance metric: `"cosine"` (default), `"euclidean"`, `"dot_product"`, `"manhattan"`
    /// * `m` - HNSW links per node (default: 16). Higher = better recall, more memory.
    /// * `ef_construction` - Construction beam width (default: 128). Higher = better index quality, slower build.
    ///
    /// # Errors
    ///
    /// Returns an error if the metric is invalid, no vectors are found, or
    /// dimensions don't match.
    pub fn create_vector_index(
        &self,
        label: &str,
        property: &str,
        dimensions: Option<usize>,
        metric: Option<&str>,
        m: Option<usize>,
        ef_construction: Option<usize>,
    ) -> Result<()> {
        use obrain_common::types::{PropertyKey, Value};
        use obrain_core::index::vector::DistanceMetric;

        // T17 W3c slice 5d: the vector-index registry is LpgStore-inherent.
        // Substrate mode exposes retrieval via SubstrateTieredIndex
        // (see `ObrainDB::substrate_handle()`) and does not consult this
        // registry — creating an index on the dummy store would silently
        // succeed yet never be read. Refuse loudly for symmetry with
        // `vector_search()` (gated in slice 5c).
        if self.substrate_store.is_some() {
            let _ = (label, property, dimensions, metric, m, ef_construction);
            return Err(obrain_common::utils::error::Error::Internal(
                "ObrainDB::create_vector_index() is LpgStore-only — substrate \
                 mode uses SubstrateTieredIndex built at migration time. A \
                 unified retrieval API is tracked as T17b."
                    .to_string(),
            ));
        }

        let metric = match metric {
            Some(m) => DistanceMetric::from_str(m).ok_or_else(|| {
                obrain_common::utils::error::Error::Internal(format!(
                    "Unknown distance metric '{}'. Use: cosine, euclidean, dot_product, manhattan",
                    m
                ))
            })?,
            None => DistanceMetric::Cosine,
        };

        // Pass 1: count vectors + detect dimensions (no cloning).
        let prop_key = PropertyKey::new(property);
        let mut found_dims: Option<usize> = dimensions;
        let mut vector_count = 0usize;

        for node in self.store.nodes_with_label(label) {
            if let Some(Value::Vector(v)) = node.properties.get(&prop_key) {
                if let Some(expected) = found_dims {
                    if v.len() != expected {
                        return Err(obrain_common::utils::error::Error::Internal(format!(
                            "Vector dimension mismatch: expected {}, found {} on node {}",
                            expected,
                            v.len(),
                            node.id.0
                        )));
                    }
                } else {
                    found_dims = Some(v.len());
                }
                vector_count += 1;
            }
        }

        let Some(dims) = found_dims else {
            // No vectors found yet: caller must have supplied explicit dimensions
            // so we can create an empty index that auto-populates via set_node_property.
            return if let Some(d) = dimensions {
                #[cfg(feature = "vector-index")]
                {
                    use obrain_core::index::vector::{HnswConfig, HnswIndex};

                    let mut config = HnswConfig::new(d, metric);
                    if let Some(m_val) = m {
                        config = config.with_m(m_val);
                    }
                    if let Some(ef_c) = ef_construction {
                        config = config.with_ef_construction(ef_c);
                    }

                    let index = HnswIndex::new(config);
                    self.store
                        .add_vector_index(label, property, Arc::new(index));
                }

                let _ = (m, ef_construction);
                tracing::info!(
                    "Empty vector index created: :{label}({property}) - 0 vectors, {d} dimensions, metric={metric_name}",
                    metric_name = metric.name()
                );
                Ok(())
            } else {
                Err(obrain_common::utils::error::Error::Internal(format!(
                    "No vector properties found on :{label}({property}) and no dimensions specified"
                )))
            };
        };

        // Build and populate the HNSW index
        #[cfg(feature = "vector-index")]
        {
            use obrain_core::index::vector::{HnswConfig, HnswIndex};

            let mut config = HnswConfig::new(dims, metric);
            if let Some(m_val) = m {
                config = config.with_m(m_val);
            }
            if let Some(ef_c) = ef_construction {
                config = config.with_ef_construction(ef_c);
            }

            let index = HnswIndex::with_capacity(config, vector_count);
            let accessor =
                obrain_core::index::vector::PropertyVectorAccessor::new(&*self.store, property);
            // Pass 2: iterate nodes again, insert directly without cloning vectors.
            for node in self.store.nodes_with_label(label) {
                if let Some(Value::Vector(v)) = node.properties.get(&prop_key) {
                    index.insert(node.id, v, &accessor);
                }
            }

            self.store
                .add_vector_index(label, property, Arc::new(index));
        }

        // Suppress unused variable warnings when vector-index is off
        let _ = (m, ef_construction);

        tracing::info!(
            "Vector index created: :{label}({property}) - {vector_count} vectors, {dims} dimensions, metric={metric_name}",
            metric_name = metric.name()
        );

        Ok(())
    }

    /// Drops a vector index for the given label and property.
    ///
    /// Returns `true` if the index existed and was removed, `false` if no
    /// index was found.
    ///
    /// After dropping, [`vector_search`](Self::vector_search) for this
    /// label+property pair will return an error.
    #[cfg(feature = "vector-index")]
    pub fn drop_vector_index(&self, label: &str, property: &str) -> bool {
        if self.substrate_store.is_some() {
            return false; // Substrate has no LpgStore vector-index registry.
        }
        let removed = self.store.remove_vector_index(label, property);
        if removed {
            tracing::info!("Vector index dropped: :{label}({property})");
        }
        removed
    }

    /// Drops and recreates a vector index, rescanning all matching nodes.
    ///
    /// This is useful after bulk inserts or when the index may be out of sync.
    /// When the index still exists, the previous configuration (dimensions,
    /// metric, M, ef\_construction) is preserved. When it has already been
    /// dropped, dimensions are inferred from existing data and default
    /// parameters are used.
    ///
    /// # Errors
    ///
    /// Returns an error if the rebuild fails (e.g., no matching vectors found
    /// and no dimensions can be inferred).
    #[cfg(feature = "vector-index")]
    pub fn rebuild_vector_index(&self, label: &str, property: &str) -> Result<()> {
        if self.substrate_store.is_some() {
            return Err(obrain_common::utils::error::Error::Internal(
                "ObrainDB::rebuild_vector_index() is LpgStore-only — substrate \
                 mode rebuilds SubstrateTieredIndex via obrain-migrate. A \
                 unified retrieval API is tracked as T17b."
                    .to_string(),
            ));
        }
        // Preserve config from existing index if available
        let config = self
            .store
            .get_vector_index(label, property)
            .map(|idx| idx.config().clone());

        self.store.remove_vector_index(label, property);

        if let Some(config) = config {
            self.create_vector_index(
                label,
                property,
                Some(config.dimensions),
                Some(config.metric.name()),
                Some(config.m),
                Some(config.ef_construction),
            )
        } else {
            // Index was already dropped: infer dimensions from data
            self.create_vector_index(label, property, None, None, None, None)
        }
    }

    // =========================================================================
    // TEXT INDEX API
    // =========================================================================

    /// Creates a BM25 text index on a node property for full-text search.
    ///
    /// Indexes all existing nodes with the given label and property.
    /// The index stays in sync automatically as nodes are created, updated,
    /// or deleted. Use [`rebuild_text_index`](Self::rebuild_text_index) only
    /// if the index was created before existing data was loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the label has no nodes or the property contains no text values.
    #[cfg(feature = "text-index")]
    pub fn create_text_index(&self, label: &str, property: &str) -> Result<()> {
        use obrain_common::types::{PropertyKey, Value};
        use obrain_core::index::text::{BM25Config, InvertedIndex};

        if self.substrate_store.is_some() {
            let _ = (label, property);
            return Err(obrain_common::utils::error::Error::Internal(
                "ObrainDB::create_text_index() is LpgStore-only — substrate \
                 mode exposes BM25 retrieval via SubstrateTieredIndex. A \
                 unified retrieval API is tracked as T17b."
                    .to_string(),
            ));
        }

        let mut index = InvertedIndex::new(BM25Config::default());
        let prop_key = PropertyKey::new(property);

        // Stream through nodes one-by-one instead of batch-fetching all
        // property values at once, which would allocate millions of strings
        // simultaneously for large labels (e.g. 8M+ Document nodes).
        // Each string is dropped after BM25 insert, keeping memory bounded.
        // Iterate nodes via the real backend (substrate in T17 mode, else
        // the LpgStore data store). The text-index registry itself still
        // lives on `self.store` (LpgStore-only API until the substrate BM25
        // rewrite lands — see W3b/W4).
        let data = self.data_store();
        let nodes = data.nodes_by_label(label);
        for &node_id in &nodes {
            if let Some(Value::String(text)) = data.get_node_property(node_id, &prop_key) {
                index.insert(node_id, text.as_str());
            }
        }

        self.store
            .add_text_index(label, property, Arc::new(RwLock::new(index)));
        Ok(())
    }

    /// Drops a text index on a label+property pair.
    ///
    /// Returns `true` if the index existed and was removed.
    #[cfg(feature = "text-index")]
    pub fn drop_text_index(&self, label: &str, property: &str) -> bool {
        if self.substrate_store.is_some() {
            return false; // Substrate has no LpgStore text-index registry.
        }
        self.store.remove_text_index(label, property)
    }

    /// Rebuilds a text index by re-scanning all matching nodes.
    ///
    /// Use after bulk property updates to keep the index current.
    ///
    /// # Errors
    ///
    /// Returns an error if no text index exists for this label+property.
    #[cfg(feature = "text-index")]
    pub fn rebuild_text_index(&self, label: &str, property: &str) -> Result<()> {
        if self.substrate_store.is_some() {
            return Err(obrain_common::utils::error::Error::Internal(
                "ObrainDB::rebuild_text_index() is LpgStore-only — substrate \
                 mode rebuilds BM25 via obrain-migrate. A unified retrieval \
                 API is tracked as T17b."
                    .to_string(),
            ));
        }
        self.store.remove_text_index(label, property);
        self.create_text_index(label, property)
    }
}
