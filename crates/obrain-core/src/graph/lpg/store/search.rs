use super::{LpgStore, value_in_range};
use crate::graph::lpg::property::CompareOp;
use crate::index::zone_map::ZoneMapEntry;
use obrain_common::types::{HashableValue, NodeId, PropertyKey, Value};

impl LpgStore {
    /// Finds all nodes whose property value falls within a range.
    ///
    /// Uses zone maps to skip the scan entirely when no values could possibly
    /// match. This is the primary building block for range predicates in query
    /// execution.
    ///
    /// # Arguments
    ///
    /// * `property` - The property key to check
    /// * `min` - Optional lower bound
    /// * `max` - Optional upper bound
    /// * `min_inclusive` - Whether the lower bound is inclusive
    /// * `max_inclusive` - Whether the upper bound is inclusive
    ///
    /// # Example
    ///
    /// ```
    /// use obrain_core::graph::lpg::LpgStore;
    /// use obrain_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "age", Value::from(25i64));
    /// store.set_node_property(n2, "age", Value::from(35i64));
    ///
    /// // Find nodes where age > 30
    /// let result = store.find_nodes_in_range(
    ///     "age",
    ///     Some(&Value::from(30i64)),
    ///     None,
    ///     false, // exclusive lower bound
    ///     true,  // inclusive upper bound (doesn't matter since None)
    /// );
    /// assert_eq!(result.len(), 1); // Only n2 matches
    /// ```
    #[must_use]
    pub fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let key = PropertyKey::new(property);

        // Check zone map first - if no values could match, return empty
        if !self
            .node_properties
            .might_match_range(&key, min, max, min_inclusive, max_inclusive)
        {
            return Vec::new();
        }

        // Scan all nodes and filter by range
        self.node_ids()
            .into_iter()
            .filter(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| value_in_range(&v, min, max, min_inclusive, max_inclusive))
            })
            .collect()
    }

    /// Finds nodes matching multiple property equality conditions.
    ///
    /// This is more efficient than intersecting multiple single-property lookups
    /// because it can use indexes when available and short-circuits on the first
    /// miss.
    ///
    /// # Example
    ///
    /// ```
    /// use obrain_core::graph::lpg::LpgStore;
    /// use obrain_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let alix = store.create_node(&["Person"]);
    /// store.set_node_property(alix, "name", Value::from("Alix"));
    /// store.set_node_property(alix, "city", Value::from("NYC"));
    ///
    /// // Find nodes where name = "Alix" AND city = "NYC"
    /// let matches = store.find_nodes_by_properties(&[
    ///     ("name", Value::from("Alix")),
    ///     ("city", Value::from("NYC")),
    /// ]);
    /// assert!(matches.contains(&alix));
    /// ```
    #[must_use]
    pub fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }

        // Find the most selective condition (smallest result set) to start
        // If any condition has an index, use that first
        let mut best_start: Option<(usize, Vec<NodeId>)> = None;
        let indexes = self.property_indexes.read();

        for (i, (prop, value)) in conditions.iter().enumerate() {
            let key = PropertyKey::new(*prop);
            let hv = HashableValue::new(value.clone());

            if let Some(index) = indexes.get(&key) {
                let matches: Vec<NodeId> = index
                    .get(&hv)
                    .map(|nodes| nodes.iter().copied().collect())
                    .unwrap_or_default();

                // Short-circuit if any indexed condition has no matches
                if matches.is_empty() {
                    return Vec::new();
                }

                // Use smallest indexed result as starting point
                if best_start
                    .as_ref()
                    .is_none_or(|(_, best)| matches.len() < best.len())
                {
                    best_start = Some((i, matches));
                }
            }
        }
        drop(indexes);

        // Start from best indexed result or fall back to full node scan
        let (start_idx, mut candidates) = best_start.unwrap_or_else(|| {
            // No indexes available, start with first condition via full scan
            let (prop, value) = &conditions[0];
            (0, self.find_nodes_by_property(prop, value))
        });

        // Filter candidates through remaining conditions
        for (i, (prop, value)) in conditions.iter().enumerate() {
            if i == start_idx {
                continue;
            }

            let key = PropertyKey::new(*prop);
            candidates.retain(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| v == *value)
            });

            // Short-circuit if no candidates remain
            if candidates.is_empty() {
                return Vec::new();
            }
        }

        candidates
    }

    /// Finds all nodes that have a specific property value.
    ///
    /// If the property is indexed, this is O(1). Otherwise, it scans all nodes
    /// which is O(n). Use [`Self::create_property_index`] for frequently queried properties.
    ///
    /// # Example
    ///
    /// ```
    /// use obrain_core::graph::lpg::LpgStore;
    /// use obrain_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// store.create_property_index("city"); // Optional but makes lookups fast
    ///
    /// let alix = store.create_node(&["Person"]);
    /// let gus = store.create_node(&["Person"]);
    /// store.set_node_property(alix, "city", Value::from("NYC"));
    /// store.set_node_property(gus, "city", Value::from("NYC"));
    ///
    /// let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    /// assert_eq!(nyc_people.len(), 2);
    /// ```
    #[must_use]
    pub fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let hv = HashableValue::new(value.clone());

        // Try indexed lookup first
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(&key) {
            if let Some(nodes) = index.get(&hv) {
                return nodes.iter().copied().collect();
            }
            return Vec::new();
        }
        drop(indexes);

        // Fall back to full scan
        self.node_ids()
            .into_iter()
            .filter(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| v == *value)
            })
            .collect()
    }

    /// Finds nodes whose property matches an operator filter.
    ///
    /// The `filter_value` is either a scalar (equality) or a `Value::Map` with
    /// `$`-prefixed operator keys like `$gt`, `$lt`, `$gte`, `$lte`, `$in`,
    /// `$nin`, `$ne`, `$contains`.
    pub fn find_nodes_matching_filter(&self, property: &str, filter_value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        self.node_ids()
            .into_iter()
            .filter(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| Self::matches_filter(&v, filter_value))
            })
            .collect()
    }

    /// Checks if a node property value matches a filter value.
    ///
    /// - Scalar filter: equality check
    /// - Map filter with `$`-prefixed keys: operator evaluation
    fn matches_filter(node_value: &Value, filter_value: &Value) -> bool {
        match filter_value {
            Value::Map(ops) if ops.keys().any(|k| k.as_str().starts_with('$')) => {
                ops.iter().all(|(op_key, op_val)| {
                    match op_key.as_str() {
                        "$gt" => {
                            Self::compare_values(node_value, op_val)
                                == Some(std::cmp::Ordering::Greater)
                        }
                        "$gte" => matches!(
                            Self::compare_values(node_value, op_val),
                            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                        ),
                        "$lt" => {
                            Self::compare_values(node_value, op_val)
                                == Some(std::cmp::Ordering::Less)
                        }
                        "$lte" => matches!(
                            Self::compare_values(node_value, op_val),
                            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                        ),
                        "$ne" => node_value != op_val,
                        "$in" => match op_val {
                            Value::List(items) => items.iter().any(|v| v == node_value),
                            _ => false,
                        },
                        "$nin" => match op_val {
                            Value::List(items) => !items.iter().any(|v| v == node_value),
                            _ => true,
                        },
                        "$contains" => match (node_value, op_val) {
                            (Value::String(a), Value::String(b)) => a.contains(b.as_str()),
                            _ => false,
                        },
                        _ => false, // Unknown operator: no match
                    }
                })
            }
            _ => node_value == filter_value, // Equality (backward compatible)
        }
    }

    /// Compares two values for ordering (cross-type numeric comparison supported).
    fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
        match (a, b) {
            (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
            (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
            (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
            (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
            (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    // === Zone Map Support ===

    /// Checks if a node property predicate might match any nodes.
    ///
    /// Uses zone maps for early filtering. Returns `true` if there might be
    /// matching nodes, `false` if there definitely aren't.
    #[must_use]
    pub fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.node_properties.might_match(property, op, value)
    }

    /// Checks if an edge property predicate might match any edges.
    #[must_use]
    pub fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.edge_properties.might_match(property, op, value)
    }

    /// Gets the zone map for a node property.
    #[must_use]
    pub fn node_property_zone_map(&self, property: &PropertyKey) -> Option<ZoneMapEntry> {
        self.node_properties.zone_map(property)
    }

    /// Gets the zone map for an edge property.
    #[must_use]
    pub fn edge_property_zone_map(&self, property: &PropertyKey) -> Option<ZoneMapEntry> {
        self.edge_properties.zone_map(property)
    }

    /// Rebuilds zone maps for all properties.
    #[doc(hidden)]
    pub fn rebuild_zone_maps(&self) {
        self.node_properties.rebuild_zone_maps();
        self.edge_properties.rebuild_zone_maps();
    }

    // === Label-scoped text search ===

    /// Finds nodes by label with a case-insensitive substring match on a property.
    ///
    /// This is dramatically faster than scanning all nodes when the label index
    /// restricts the candidate set. Supports early termination via `limit`.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label filter (if `None`, searches all nodes)
    /// * `property` - Property key to search in (e.g., "name", "title")
    /// * `substring` - Case-insensitive substring to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```
    /// use obrain_core::graph::lpg::LpgStore;
    /// use obrain_common::types::Value;
    ///
    /// let store = LpgStore::new().expect("arena allocation");
    /// let n1 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "name", Value::from("Alice Johnson"));
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n2, "name", Value::from("Bob Smith"));
    ///
    /// let results = store.find_nodes_by_label_property_contains(
    ///     Some("Person"), "name", "alice", 10
    /// );
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0], n1);
    /// ```
    #[must_use]
    pub fn find_nodes_by_label_property_contains(
        &self,
        label: Option<&str>,
        property: &str,
        substring: &str,
        limit: usize,
    ) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let needle = substring.to_lowercase();
        let mut results = Vec::with_capacity(limit.min(1024));

        // Get candidate node IDs: label-filtered or all
        if let Some(label) = label {
            let label_to_id = self.label_to_id.read();
            let Some(&label_id) = label_to_id.get(label) else {
                return Vec::new();
            };
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                for &nid in set.keys() {
                    if let Some(val) = self.node_properties.get(nid, &key)
                        && let Some(s) = val.as_str()
                        && s.to_lowercase().contains(&needle)
                    {
                        results.push(nid);
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        } else {
            for &nid in &self.node_ids() {
                if let Some(val) = self.node_properties.get(nid, &key)
                    && let Some(s) = val.as_str()
                    && s.to_lowercase().contains(&needle)
                {
                    results.push(nid);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        results
    }

    /// Like [`Self::find_nodes_by_label_property_contains`] but with an additional
    /// `max_scan` cap on the total number of nodes inspected. This prevents
    /// worst-case O(N) full scans when the search term matches nothing (e.g.
    /// English terms in a French-only database with 8M nodes).
    ///
    /// Returns early when either `limit` results are found **or** `max_scan`
    /// nodes have been inspected, whichever comes first.
    #[must_use]
    pub fn find_nodes_by_label_property_contains_bounded(
        &self,
        label: Option<&str>,
        property: &str,
        substring: &str,
        limit: usize,
        max_scan: usize,
    ) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let needle = substring.to_lowercase();
        let mut results = Vec::with_capacity(limit.min(1024));
        let mut scanned = 0usize;

        if let Some(label) = label {
            let label_to_id = self.label_to_id.read();
            let Some(&label_id) = label_to_id.get(label) else {
                return Vec::new();
            };
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                for &nid in set.keys() {
                    scanned += 1;
                    if scanned > max_scan {
                        break;
                    }
                    if let Some(val) = self.node_properties.get(nid, &key)
                        && let Some(s) = val.as_str()
                        && s.to_lowercase().contains(&needle)
                    {
                        results.push(nid);
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        } else {
            for &nid in &self.node_ids() {
                scanned += 1;
                if scanned > max_scan {
                    break;
                }
                if let Some(val) = self.node_properties.get(nid, &key)
                    && let Some(s) = val.as_str()
                    && s.to_lowercase().contains(&needle)
                {
                    results.push(nid);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        results
    }

    /// Finds nodes by label with a case-insensitive substring match across
    /// multiple properties. Returns as soon as any property matches.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional label filter
    /// * `properties` - Property keys to search across (checks in order)
    /// * `substring` - Case-insensitive substring to search for
    /// * `limit` - Maximum number of results to return
    #[must_use]
    pub fn find_nodes_by_label_properties_contains(
        &self,
        label: Option<&str>,
        properties: &[&str],
        substring: &str,
        limit: usize,
    ) -> Vec<NodeId> {
        let keys: Vec<PropertyKey> = properties.iter().map(|p| PropertyKey::new(*p)).collect();
        let needle = substring.to_lowercase();
        let mut results = Vec::with_capacity(limit.min(1024));

        let check_node = |nid: NodeId| -> bool {
            for key in &keys {
                if let Some(val) = self.node_properties.get(nid, key)
                    && let Some(s) = val.as_str()
                    && s.to_lowercase().contains(&needle)
                {
                    return true;
                }
            }
            false
        };

        if let Some(label) = label {
            let label_to_id = self.label_to_id.read();
            let Some(&label_id) = label_to_id.get(label) else {
                return Vec::new();
            };
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                for &nid in set.keys() {
                    if check_node(nid) {
                        results.push(nid);
                        if results.len() >= limit {
                            break;
                        }
                    }
                }
            }
        } else {
            for &nid in &self.node_ids() {
                if check_node(nid) {
                    results.push(nid);
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        }

        results
    }
}
