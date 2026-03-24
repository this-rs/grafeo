//! Per-tenant isolation via named graphs.
//!
//! Each tenant maps to a named graph. All cognitive stores (energy, synapse,
//! scar, fabric) are scoped to the active tenant graph. Isolation is structural
//! (separate DashMaps per tenant), not a property filter.
//!
//! # Usage
//!
//! ```ignore
//! let tm = TenantManager::new();
//! tm.create_tenant("tenant_123").unwrap();
//! tm.switch_tenant("tenant_123").unwrap();
//! // All cognitive operations now scoped to tenant_123
//! ```

use dashmap::DashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "energy")]
use crate::energy::{EnergyConfig, EnergyStore};

#[cfg(feature = "synapse")]
use crate::synapse::{SynapseConfig, SynapseStore};

#[cfg(feature = "fabric")]
use crate::fabric::FabricStore;

#[cfg(feature = "scar")]
use crate::scar::{ScarConfig, ScarStore};

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from tenant operations.
#[derive(Debug, thiserror::Error)]
pub enum TenantError {
    /// Tenant already exists.
    #[error("tenant already exists: {0}")]
    AlreadyExists(String),

    /// Tenant not found.
    #[error("tenant not found: {0}")]
    NotFound(String),

    /// Invalid tenant name.
    #[error("invalid tenant name: {0}")]
    InvalidName(String),
}

// ---------------------------------------------------------------------------
// TenantGraph — per-tenant cognitive state
// ---------------------------------------------------------------------------

/// Holds all cognitive stores for a single tenant (named graph).
///
/// This is the ACID isolation boundary: each tenant has completely
/// independent DashMaps, not just filtered views of a shared map.
pub struct TenantGraph {
    /// Tenant identifier (= named graph name).
    pub name: String,

    /// When this tenant graph was created.
    pub created_at: Instant,

    /// Energy store scoped to this tenant.
    #[cfg(feature = "energy")]
    pub energy_store: Arc<EnergyStore>,

    /// Synapse store scoped to this tenant.
    #[cfg(feature = "synapse")]
    pub synapse_store: Arc<SynapseStore>,

    /// Fabric store scoped to this tenant.
    #[cfg(feature = "fabric")]
    pub fabric_store: Arc<FabricStore>,

    /// Scar store scoped to this tenant.
    #[cfg(feature = "scar")]
    pub scar_store: Arc<ScarStore>,
}

impl TenantGraph {
    /// Creates a new tenant graph with default configs.
    fn new(name: String) -> Self {
        Self {
            name,
            created_at: Instant::now(),
            #[cfg(feature = "energy")]
            energy_store: Arc::new(EnergyStore::new(EnergyConfig::default())),
            #[cfg(feature = "synapse")]
            synapse_store: Arc::new(SynapseStore::new(SynapseConfig::default())),
            #[cfg(feature = "fabric")]
            fabric_store: Arc::new(FabricStore::new()),
            #[cfg(feature = "scar")]
            scar_store: Arc::new(ScarStore::new(ScarConfig::default())),
        }
    }

    /// Creates a new tenant graph with custom energy/synapse configs.
    #[cfg(all(feature = "energy", feature = "synapse"))]
    fn with_configs(name: String, energy_config: EnergyConfig, synapse_config: SynapseConfig) -> Self {
        Self {
            name,
            created_at: Instant::now(),
            energy_store: Arc::new(EnergyStore::new(energy_config)),
            synapse_store: Arc::new(SynapseStore::new(synapse_config)),
            #[cfg(feature = "fabric")]
            fabric_store: Arc::new(FabricStore::new()),
            #[cfg(feature = "scar")]
            scar_store: Arc::new(ScarStore::new(ScarConfig::default())),
        }
    }
}

impl fmt::Debug for TenantGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("TenantGraph");
        d.field("name", &self.name);

        #[cfg(feature = "energy")]
        d.field("energy_nodes", &self.energy_store.len());

        #[cfg(feature = "synapse")]
        d.field("synapse_count", &self.synapse_store.len());

        d.finish()
    }
}

// ---------------------------------------------------------------------------
// TenantInfo — metadata returned by list_tenants
// ---------------------------------------------------------------------------

/// Summary info about a tenant (returned by `list_tenants`).
#[derive(Debug, Clone)]
pub struct TenantInfo {
    /// Tenant name / graph id.
    pub name: String,
    /// When the tenant was created.
    pub created_at: Instant,
    /// Number of energy nodes (0 if feature disabled).
    pub energy_node_count: usize,
    /// Number of synapses (0 if feature disabled).
    pub synapse_count: usize,
}

// ---------------------------------------------------------------------------
// TenantManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of per-tenant named graphs.
///
/// Each tenant = one named graph with its own set of cognitive stores.
/// The active tenant determines which stores are used for all cognitive
/// operations. Switching tenant is O(1) — just swaps the active pointer.
///
/// Thread-safe: all operations use `DashMap` or `RwLock`.
pub struct TenantManager {
    /// All tenant graphs, indexed by name.
    tenants: DashMap<String, Arc<TenantGraph>>,
    /// Currently active tenant name. Empty string = no active tenant.
    active_tenant: RwLock<Option<String>>,
}

impl TenantManager {
    /// Creates a new TenantManager with no tenants.
    pub fn new() -> Self {
        Self {
            tenants: DashMap::new(),
            active_tenant: RwLock::new(None),
        }
    }

    /// Creates a new tenant (named graph) with default cognitive configs.
    ///
    /// Returns an error if the tenant already exists or the name is invalid.
    pub fn create_tenant(&self, name: &str) -> Result<(), TenantError> {
        Self::validate_name(name)?;

        if self.tenants.contains_key(name) {
            return Err(TenantError::AlreadyExists(name.to_string()));
        }

        let graph = Arc::new(TenantGraph::new(name.to_string()));
        // Use entry API to avoid TOCTOU
        match self.tenants.entry(name.to_string()) {
            dashmap::mapref::entry::Entry::Occupied(_) => {
                Err(TenantError::AlreadyExists(name.to_string()))
            }
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert(graph);
                tracing::info!(tenant = name, "tenant graph created");
                Ok(())
            }
        }
    }

    /// Deletes a tenant and all its cognitive state.
    ///
    /// If the deleted tenant was the active tenant, active is set to None.
    pub fn delete_tenant(&self, name: &str) -> Result<(), TenantError> {
        if self.tenants.remove(name).is_none() {
            return Err(TenantError::NotFound(name.to_string()));
        }

        // Clear active if it was this tenant
        let mut active = self.active_tenant.write();
        if active.as_deref() == Some(name) {
            *active = None;
        }

        tracing::info!(tenant = name, "tenant graph deleted");
        Ok(())
    }

    /// Lists all tenants with summary info.
    pub fn list_tenants(&self) -> Vec<TenantInfo> {
        self.tenants
            .iter()
            .map(|entry| {
                let g = entry.value();
                TenantInfo {
                    name: g.name.clone(),
                    created_at: g.created_at,
                    #[cfg(feature = "energy")]
                    energy_node_count: g.energy_store.len(),
                    #[cfg(not(feature = "energy"))]
                    energy_node_count: 0,
                    #[cfg(feature = "synapse")]
                    synapse_count: g.synapse_store.len(),
                    #[cfg(not(feature = "synapse"))]
                    synapse_count: 0,
                }
            })
            .collect()
    }

    /// Switches the active tenant (named graph scope).
    ///
    /// Equivalent to `USE GRAPH <name>`. All subsequent cognitive operations
    /// will use this tenant's stores.
    pub fn switch_tenant(&self, name: &str) -> Result<(), TenantError> {
        if !self.tenants.contains_key(name) {
            return Err(TenantError::NotFound(name.to_string()));
        }

        let mut active = self.active_tenant.write();
        *active = Some(name.to_string());
        tracing::debug!(tenant = name, "switched active tenant graph");
        Ok(())
    }

    /// Returns the currently active tenant name, if any.
    pub fn active_tenant_name(&self) -> Option<String> {
        self.active_tenant.read().clone()
    }

    /// Returns the active tenant's graph, if a tenant is active.
    pub fn active_graph(&self) -> Option<Arc<TenantGraph>> {
        let active = self.active_tenant.read();
        let name = active.as_deref()?;
        self.tenants.get(name).map(|e| Arc::clone(e.value()))
    }

    /// Returns a specific tenant's graph by name.
    pub fn get_tenant(&self, name: &str) -> Option<Arc<TenantGraph>> {
        self.tenants.get(name).map(|e| Arc::clone(e.value()))
    }

    /// Returns the number of tenants.
    pub fn tenant_count(&self) -> usize {
        self.tenants.len()
    }

    /// Returns whether the given tenant exists.
    pub fn tenant_exists(&self, name: &str) -> bool {
        self.tenants.contains_key(name)
    }

    // -- Active tenant store accessors --

    /// Returns the active tenant's energy store.
    #[cfg(feature = "energy")]
    pub fn energy_store(&self) -> Option<Arc<EnergyStore>> {
        self.active_graph()
            .map(|g| Arc::clone(&g.energy_store))
    }

    /// Returns the active tenant's synapse store.
    #[cfg(feature = "synapse")]
    pub fn synapse_store(&self) -> Option<Arc<SynapseStore>> {
        self.active_graph()
            .map(|g| Arc::clone(&g.synapse_store))
    }

    /// Returns the active tenant's fabric store.
    #[cfg(feature = "fabric")]
    pub fn fabric_store(&self) -> Option<Arc<FabricStore>> {
        self.active_graph()
            .map(|g| Arc::clone(&g.fabric_store))
    }

    /// Returns the active tenant's scar store.
    #[cfg(feature = "scar")]
    pub fn scar_store(&self) -> Option<Arc<ScarStore>> {
        self.active_graph()
            .map(|g| Arc::clone(&g.scar_store))
    }

    // -- Validation --

    fn validate_name(name: &str) -> Result<(), TenantError> {
        if name.is_empty() {
            return Err(TenantError::InvalidName(
                "tenant name cannot be empty".into(),
            ));
        }
        if name.len() > 256 {
            return Err(TenantError::InvalidName(
                "tenant name too long (max 256 chars)".into(),
            ));
        }
        // Only allow alphanumeric, underscore, hyphen, dot
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return Err(TenantError::InvalidName(format!(
                "tenant name contains invalid characters: {name}"
            )));
        }
        Ok(())
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TenantManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TenantManager")
            .field("tenant_count", &self.tenants.len())
            .field("active_tenant", &*self.active_tenant.read())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_list_tenants() {
        let tm = TenantManager::new();
        assert_eq!(tm.tenant_count(), 0);

        tm.create_tenant("tenant_a").unwrap();
        tm.create_tenant("tenant_b").unwrap();
        assert_eq!(tm.tenant_count(), 2);

        let list = tm.list_tenants();
        assert_eq!(list.len(), 2);
        let names: Vec<&str> = list.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"tenant_a"));
        assert!(names.contains(&"tenant_b"));
    }

    #[test]
    fn test_create_duplicate_tenant() {
        let tm = TenantManager::new();
        tm.create_tenant("dup").unwrap();
        assert!(matches!(
            tm.create_tenant("dup"),
            Err(TenantError::AlreadyExists(_))
        ));
    }

    #[test]
    fn test_delete_tenant() {
        let tm = TenantManager::new();
        tm.create_tenant("to_delete").unwrap();
        tm.switch_tenant("to_delete").unwrap();
        assert!(tm.active_tenant_name().is_some());

        tm.delete_tenant("to_delete").unwrap();
        assert_eq!(tm.tenant_count(), 0);
        assert!(tm.active_tenant_name().is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let tm = TenantManager::new();
        assert!(matches!(
            tm.delete_tenant("nope"),
            Err(TenantError::NotFound(_))
        ));
    }

    #[test]
    fn test_switch_tenant() {
        let tm = TenantManager::new();
        tm.create_tenant("a").unwrap();
        tm.create_tenant("b").unwrap();

        assert!(tm.active_tenant_name().is_none());

        tm.switch_tenant("a").unwrap();
        assert_eq!(tm.active_tenant_name().as_deref(), Some("a"));

        tm.switch_tenant("b").unwrap();
        assert_eq!(tm.active_tenant_name().as_deref(), Some("b"));
    }

    #[test]
    fn test_switch_nonexistent() {
        let tm = TenantManager::new();
        assert!(matches!(
            tm.switch_tenant("nope"),
            Err(TenantError::NotFound(_))
        ));
    }

    #[test]
    fn test_invalid_names() {
        let tm = TenantManager::new();
        assert!(matches!(
            tm.create_tenant(""),
            Err(TenantError::InvalidName(_))
        ));
        assert!(matches!(
            tm.create_tenant("bad name spaces"),
            Err(TenantError::InvalidName(_))
        ));
        assert!(matches!(
            tm.create_tenant("bad/slash"),
            Err(TenantError::InvalidName(_))
        ));
        // Valid names
        assert!(tm.create_tenant("good-name").is_ok());
        assert!(tm.create_tenant("good_name.123").is_ok());
    }

    #[cfg(feature = "energy")]
    #[test]
    fn test_tenant_isolation_energy() {
        use grafeo_common::types::NodeId;

        let tm = TenantManager::new();
        tm.create_tenant("tenant_1").unwrap();
        tm.create_tenant("tenant_2").unwrap();

        let node = NodeId(42);

        // Boost in tenant_1
        {
            let g1 = tm.get_tenant("tenant_1").unwrap();
            g1.energy_store.boost(node, 5.0);
        }

        // Boost in tenant_2 with different value
        {
            let g2 = tm.get_tenant("tenant_2").unwrap();
            g2.energy_store.boost(node, 10.0);
        }

        // Verify isolation: same node, different energies
        let e1 = tm.get_tenant("tenant_1").unwrap().energy_store.get_energy(node);
        let e2 = tm.get_tenant("tenant_2").unwrap().energy_store.get_energy(node);

        assert!((e1 - 5.0).abs() < 0.1, "tenant_1 energy should be ~5.0, got {e1}");
        assert!((e2 - 10.0).abs() < 0.1, "tenant_2 energy should be ~10.0, got {e2}");
    }

    #[cfg(feature = "synapse")]
    #[test]
    fn test_tenant_isolation_synapse() {
        use grafeo_common::types::NodeId;

        let tm = TenantManager::new();
        tm.create_tenant("tenant_a").unwrap();
        tm.create_tenant("tenant_b").unwrap();

        let n1 = NodeId(1);
        let n2 = NodeId(2);

        // Create synapse in tenant_a
        {
            let ga = tm.get_tenant("tenant_a").unwrap();
            ga.synapse_store.reinforce(n1, n2, 1.0);
        }

        // tenant_b should have no synapse
        {
            let gb = tm.get_tenant("tenant_b").unwrap();
            assert!(gb.synapse_store.get_synapse(n1, n2).is_none());
        }

        // tenant_a should have the synapse
        {
            let ga = tm.get_tenant("tenant_a").unwrap();
            assert!(ga.synapse_store.get_synapse(n1, n2).is_some());
        }
    }

    #[cfg(feature = "energy")]
    #[test]
    fn test_active_graph_store_access() {
        use grafeo_common::types::NodeId;

        let tm = TenantManager::new();
        tm.create_tenant("active_test").unwrap();
        tm.switch_tenant("active_test").unwrap();

        let store = tm.energy_store().expect("should have active energy store");
        store.boost(NodeId(1), 3.0);

        let energy = store.get_energy(NodeId(1));
        assert!((energy - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_delete_cleans_all_state() {
        let tm = TenantManager::new();
        tm.create_tenant("cleanup").unwrap();

        // Add some state
        #[cfg(feature = "energy")]
        {
            let g = tm.get_tenant("cleanup").unwrap();
            g.energy_store.boost(grafeo_common::types::NodeId(1), 5.0);
            assert!(!g.energy_store.is_empty());
        }

        // Delete
        tm.delete_tenant("cleanup").unwrap();

        // Tenant is gone — no way to access old state
        assert!(tm.get_tenant("cleanup").is_none());
        assert_eq!(tm.tenant_count(), 0);
    }
}
