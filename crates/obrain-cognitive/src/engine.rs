//! CognitiveEngine trait and `DefaultCognitiveEngine` implementation.
//!
//! The [`CognitiveEngine`] trait provides unified access to all cognitive
//! subsystems. [`CognitiveEngineBuilder`] constructs a [`DefaultCognitiveEngine`]
//! with selected subsystems and registers their listeners with the reactive
//! [`Scheduler`].

use crate::config::{CognitiveConfig, KernelConfigToml};
use crate::provenance::ProvenanceRecorder;
#[allow(unused_imports)]
use std::sync::Arc;

#[cfg(feature = "engram")]
use crate::engram::traits::VectorIndex;
#[cfg(feature = "engram")]
use crate::engram::{EngramMetricsCollector, EngramStore};

// Conditional imports based on feature flags
#[cfg(feature = "energy")]
use crate::energy::{EnergyConfig, EnergyListener, EnergyStore};

#[cfg(feature = "synapse")]
use crate::synapse::{SynapseConfig, SynapseListener, SynapseStore};

#[cfg(feature = "fabric")]
use crate::fabric::{FabricListener, FabricStore};

#[cfg(feature = "co-change")]
use crate::co_change::{CoChangeConfig, CoChangeDetector, CoChangeStore};

#[cfg(feature = "kernel")]
use crate::kernel::KernelListener;
#[cfg(feature = "kernel")]
use obrain_adapters::plugins::algorithms::KernelManager;
#[cfg(feature = "kernel")]
use obrain_core::graph::lpg::LpgStore;

use crate::tenant::TenantManager;
use obrain_core::graph::GraphStoreMut;
use obrain_reactive::Scheduler;

// ---------------------------------------------------------------------------
// CognitiveEngine trait
// ---------------------------------------------------------------------------

/// Unified access to all cognitive subsystems.
///
/// Each method returns `Option<&T>` — `None` if the subsystem was not enabled
/// at construction time or its feature flag is not compiled in.
///
/// This trait enables polymorphic access to the cognitive engine without
/// tying consumers to a specific implementation.
pub trait CognitiveEngine: Send + Sync + std::fmt::Debug {
    /// Returns the energy store, if the energy subsystem is active.
    #[cfg(feature = "energy")]
    fn energy_store(&self) -> Option<&Arc<EnergyStore>>;

    /// Returns the synapse store, if the synapse subsystem is active.
    #[cfg(feature = "synapse")]
    fn synapse_store(&self) -> Option<&Arc<SynapseStore>>;

    /// Returns the fabric store, if the fabric subsystem is active.
    #[cfg(feature = "fabric")]
    fn fabric_store(&self) -> Option<&Arc<FabricStore>>;

    /// Returns the co-change store, if the co-change subsystem is active.
    #[cfg(feature = "co-change")]
    fn co_change_store(&self) -> Option<&Arc<CoChangeStore>>;

    /// Returns the engram store, if the engram subsystem is active.
    #[cfg(feature = "engram")]
    fn engram_store(&self) -> Option<&Arc<EngramStore>>;

    /// Returns the engram metrics collector, if the engram subsystem is active.
    #[cfg(feature = "engram")]
    fn engram_metrics(&self) -> Option<&Arc<EngramMetricsCollector>>;

    /// Returns the vector index used for spectral signature search, if available.
    #[cfg(feature = "engram")]
    fn vector_index(&self) -> Option<&Arc<dyn VectorIndex>>;

    /// Returns the number of active subsystems.
    fn active_subsystem_count(&self) -> usize;

    /// Returns the tenant manager for per-tenant graph isolation.
    fn tenant_manager(&self) -> &TenantManager;

    /// Returns the provenance recorder for cognitive event tracking.
    fn provenance(&self) -> &Arc<ProvenanceRecorder>;
}

// ---------------------------------------------------------------------------
// DefaultCognitiveEngine
// ---------------------------------------------------------------------------

/// Default implementation of [`CognitiveEngine`] constructed via the builder.
///
/// Holds `Arc` references to the stores for each enabled subsystem.
/// The corresponding listeners are registered with the scheduler at build time.
pub struct DefaultCognitiveEngine {
    #[cfg(feature = "energy")]
    energy: Option<Arc<EnergyStore>>,

    #[cfg(feature = "synapse")]
    synapse: Option<Arc<SynapseStore>>,

    #[cfg(feature = "fabric")]
    fabric: Option<Arc<FabricStore>>,

    #[cfg(feature = "co-change")]
    co_change: Option<Arc<CoChangeStore>>,

    #[cfg(feature = "kernel")]
    kernel: Option<Arc<KernelManager>>,

    #[cfg(feature = "engram")]
    engram_store: Option<Arc<EngramStore>>,

    #[cfg(feature = "engram")]
    engram_metrics: Option<Arc<EngramMetricsCollector>>,

    #[cfg(feature = "engram")]
    vector_index: Option<Arc<dyn VectorIndex>>,

    /// Count of active subsystems.
    active_count: usize,

    /// Per-tenant named graph isolation manager.
    tenant_manager: TenantManager,

    /// Provenance recorder for automatic cognitive event tracking.
    provenance: Arc<ProvenanceRecorder>,
}

impl CognitiveEngine for DefaultCognitiveEngine {
    #[cfg(feature = "energy")]
    fn energy_store(&self) -> Option<&Arc<EnergyStore>> {
        self.energy.as_ref()
    }

    #[cfg(feature = "synapse")]
    fn synapse_store(&self) -> Option<&Arc<SynapseStore>> {
        self.synapse.as_ref()
    }

    #[cfg(feature = "fabric")]
    fn fabric_store(&self) -> Option<&Arc<FabricStore>> {
        self.fabric.as_ref()
    }

    #[cfg(feature = "co-change")]
    fn co_change_store(&self) -> Option<&Arc<CoChangeStore>> {
        self.co_change.as_ref()
    }

    #[cfg(feature = "engram")]
    fn engram_store(&self) -> Option<&Arc<EngramStore>> {
        self.engram_store.as_ref()
    }

    #[cfg(feature = "engram")]
    fn engram_metrics(&self) -> Option<&Arc<EngramMetricsCollector>> {
        self.engram_metrics.as_ref()
    }

    #[cfg(feature = "engram")]
    fn vector_index(&self) -> Option<&Arc<dyn VectorIndex>> {
        self.vector_index.as_ref()
    }

    fn active_subsystem_count(&self) -> usize {
        self.active_count
    }

    fn tenant_manager(&self) -> &TenantManager {
        &self.tenant_manager
    }

    fn provenance(&self) -> &Arc<ProvenanceRecorder> {
        &self.provenance
    }
}

impl DefaultCognitiveEngine {
    /// Returns the tenant manager for per-tenant named graph isolation.
    ///
    /// Use this to create, delete, list, and switch tenants.
    /// When a tenant is active, cognitive operations should use the
    /// tenant-scoped stores via `tenant_manager().energy_store()` etc.
    pub fn tenants(&self) -> &TenantManager {
        &self.tenant_manager
    }

    /// Returns the provenance recorder for querying cognitive event history.
    pub fn provenance_recorder(&self) -> &Arc<ProvenanceRecorder> {
        &self.provenance
    }

    /// Returns the kernel manager, if the kernel subsystem is active.
    #[cfg(feature = "kernel")]
    pub fn kernel_manager(&self) -> Option<&Arc<KernelManager>> {
        self.kernel.as_ref()
    }

    /// Sets the engram store, metrics collector, and vector index.
    ///
    /// Called by higher-level coordinators (e.g., EngramManager) to wire
    /// the engram subsystem into the cognitive engine for procedure access.
    #[cfg(feature = "engram")]
    pub fn set_engram_subsystem(
        &mut self,
        store: Arc<EngramStore>,
        metrics: Arc<EngramMetricsCollector>,
        vector_index: Arc<dyn VectorIndex>,
    ) {
        self.engram_store = Some(store);
        self.engram_metrics = Some(metrics);
        self.vector_index = Some(vector_index);
    }
}

impl std::fmt::Debug for DefaultCognitiveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("DefaultCognitiveEngine");
        d.field("active_subsystems", &self.active_count);

        #[cfg(feature = "energy")]
        d.field("energy", &self.energy.is_some());

        #[cfg(feature = "synapse")]
        d.field("synapse", &self.synapse.is_some());

        #[cfg(feature = "fabric")]
        d.field("fabric", &self.fabric.is_some());

        #[cfg(feature = "co-change")]
        d.field("co_change", &self.co_change.is_some());

        #[cfg(feature = "kernel")]
        d.field("kernel", &self.kernel.is_some());

        d.field("tenant_count", &self.tenant_manager.tenant_count());
        d.field("provenance_events", &self.provenance.total_events());

        d.finish()
    }
}

// ---------------------------------------------------------------------------
// CognitiveEngineBuilder
// ---------------------------------------------------------------------------

/// Builder for [`DefaultCognitiveEngine`].
///
/// Allows selective activation of cognitive subsystems. When [`build`](Self::build)
/// is called, the corresponding [`MutationListener`](obrain_reactive::MutationListener)s
/// are registered with the provided [`Scheduler`].
///
/// # Example
///
/// ```ignore
/// let engine = CognitiveEngineBuilder::new()
///     .with_energy(EnergyConfig::default())
///     .with_synapses(SynapseConfig::default())
///     .build(&scheduler);
/// ```
pub struct CognitiveEngineBuilder {
    #[cfg(feature = "energy")]
    energy_config: Option<EnergyConfig>,

    #[cfg(feature = "synapse")]
    synapse_config: Option<SynapseConfig>,

    #[cfg(feature = "fabric")]
    fabric_enabled: bool,

    #[cfg(feature = "co-change")]
    co_change_config: Option<CoChangeConfig>,

    #[cfg(feature = "kernel")]
    kernel_config: Option<KernelConfigToml>,

    /// Optional backing graph store for write-through persistence.
    graph_store: Option<Arc<dyn GraphStoreMut>>,

    /// Optional LpgStore reference for kernel embedding manager.
    /// KernelManager requires `Arc<LpgStore>` specifically.
    #[cfg(feature = "kernel")]
    lpg_store: Option<Arc<LpgStore>>,

    /// Optional substrate handle for column-backed cognitive stores
    /// (T17 cutover — preferred over [`graph_store`] when present).
    ///
    /// When set, `EnergyStore::with_substrate` / `SynapseStore::with_substrate`
    /// / analogous constructors route boost/decay/reinforce through the
    /// substrate Q1.15 / Q0.16 columns + dedicated WAL records instead of
    /// the generic property API. This is the supported path since the T17
    /// substrate cutover.
    #[cfg(feature = "substrate")]
    substrate: Option<Arc<obrain_substrate::SubstrateStore>>,
}

impl CognitiveEngineBuilder {
    /// Creates a new builder with all subsystems disabled.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "energy")]
            energy_config: None,

            #[cfg(feature = "synapse")]
            synapse_config: None,

            #[cfg(feature = "fabric")]
            fabric_enabled: false,

            #[cfg(feature = "co-change")]
            co_change_config: None,

            #[cfg(feature = "kernel")]
            kernel_config: None,

            graph_store: None,

            #[cfg(feature = "kernel")]
            lpg_store: None,

            #[cfg(feature = "substrate")]
            substrate: None,
        }
    }

    /// Creates a builder from a [`CognitiveConfig`], enabling subsystems
    /// based on their `enabled` flag (and compile-time feature flags).
    pub fn from_config(#[allow(unused_variables)] config: &CognitiveConfig) -> Self {
        #[allow(unused_mut)]
        let mut builder = Self::new();

        #[cfg(feature = "energy")]
        if config.energy.enabled {
            builder.energy_config = Some(config.energy.to_runtime());
        }

        #[cfg(feature = "synapse")]
        if config.synapse.enabled {
            builder.synapse_config = Some(config.synapse.to_runtime());
        }

        #[cfg(feature = "fabric")]
        if config.fabric.enabled {
            builder.fabric_enabled = true;
        }

        #[cfg(feature = "co-change")]
        if config.co_change.enabled {
            builder.co_change_config = Some(config.co_change.to_runtime());
        }

        #[cfg(feature = "kernel")]
        if config.kernel.enabled {
            builder.kernel_config = Some(config.kernel.clone());
        }

        builder
    }

    /// Sets the backing graph store for write-through persistence.
    ///
    /// When set, all cognitive stores will persist their scores as
    /// node/edge properties on the graph. This enables lazy reconstruction
    /// of the hot caches on restart.
    pub fn with_graph_store(mut self, store: Arc<dyn GraphStoreMut>) -> Self {
        self.graph_store = Some(store);
        self
    }

    /// Sets the substrate handle — the T17-preferred backing for cognitive
    /// stores that expose column-native writes (energy / synapse / scar /
    /// utility / affinity).
    ///
    /// When both [`with_graph_store`](Self::with_graph_store) and
    /// [`with_substrate`](Self::with_substrate) are set, the substrate handle
    /// takes precedence for stores that implement `with_substrate`; fabric
    /// and co-change stay on the trait-object path (no substrate variant
    /// yet). For `SynapseStore::with_substrate`, the graph store is still
    /// required for structural edge creation — pass both.
    #[cfg(feature = "substrate")]
    pub fn with_substrate(mut self, store: Arc<obrain_substrate::SubstrateStore>) -> Self {
        self.substrate = Some(store);
        self
    }

    /// Enables the energy subsystem with the given configuration.
    #[cfg(feature = "energy")]
    pub fn with_energy(mut self, config: EnergyConfig) -> Self {
        self.energy_config = Some(config);
        self
    }

    /// Enables the synapse subsystem with the given configuration.
    #[cfg(feature = "synapse")]
    pub fn with_synapses(mut self, config: SynapseConfig) -> Self {
        self.synapse_config = Some(config);
        self
    }

    /// Enables the fabric subsystem.
    #[cfg(feature = "fabric")]
    pub fn with_fabric(mut self) -> Self {
        self.fabric_enabled = true;
        self
    }

    /// Enables the co-change detection subsystem with the given configuration.
    #[cfg(feature = "co-change")]
    pub fn with_co_change(mut self, config: CoChangeConfig) -> Self {
        self.co_change_config = Some(config);
        self
    }

    /// Enables the kernel embedding subsystem with the given configuration.
    ///
    /// Requires [`with_lpg_store`](Self::with_lpg_store) to also be called,
    /// since [`KernelManager`] needs a direct `Arc<LpgStore>` reference.
    #[cfg(feature = "kernel")]
    pub fn with_kernel(mut self, config: KernelConfigToml) -> Self {
        self.kernel_config = Some(config);
        self
    }

    /// Sets the `Arc<LpgStore>` needed by the kernel embedding manager.
    ///
    /// The kernel subsystem requires direct access to the `LpgStore` (not
    /// the `dyn GraphStoreMut` trait object) for subscription-based incremental
    /// updates. Call this alongside [`with_kernel`](Self::with_kernel).
    #[cfg(feature = "kernel")]
    pub fn with_lpg_store(mut self, store: Arc<LpgStore>) -> Self {
        self.lpg_store = Some(store);
        self
    }

    /// Builds the cognitive engine and registers listeners with the scheduler.
    ///
    /// Each enabled subsystem:
    /// 1. Creates a shared store (`Arc<XxxStore>`) — with write-through if a
    ///    graph store was configured via [`with_graph_store`](Self::with_graph_store)
    /// 2. Creates a listener wrapping the store
    /// 3. Registers the listener with the scheduler
    ///
    /// Returns a [`DefaultCognitiveEngine`] holding references to all stores.
    pub fn build(self, #[allow(unused_variables)] scheduler: &Scheduler) -> DefaultCognitiveEngine {
        #[allow(unused_mut)]
        let mut active_count = 0;

        #[allow(unused_variables)]
        let gs = self.graph_store;

        #[cfg(feature = "substrate")]
        #[allow(unused_variables)]
        let sub = self.substrate;

        // Energy — prefer substrate (column-native WAL writes) over
        // graph_store (generic property API) over legacy in-memory.
        #[cfg(feature = "energy")]
        let energy = self.energy_config.map(|config| {
            let store = {
                #[cfg(feature = "substrate")]
                {
                    match (&sub, &gs) {
                        (Some(substrate), _) => Arc::new(EnergyStore::with_substrate(
                            config,
                            Arc::clone(substrate),
                        )),
                        (None, Some(graph_store)) => Arc::new(EnergyStore::with_graph_store(
                            config,
                            Arc::clone(graph_store),
                        )),
                        (None, None) => Arc::new(EnergyStore::new(config)),
                    }
                }
                #[cfg(not(feature = "substrate"))]
                {
                    match &gs {
                        Some(graph_store) => Arc::new(EnergyStore::with_graph_store(
                            config,
                            Arc::clone(graph_store),
                        )),
                        None => Arc::new(EnergyStore::new(config)),
                    }
                }
            };
            let listener = Arc::new(EnergyListener::new(Arc::clone(&store)));
            scheduler.register_listener(listener);
            active_count += 1;
            #[cfg(feature = "substrate")]
            let routed = if store.is_substrate_backed() {
                "substrate"
            } else if gs.is_some() {
                "graph_store"
            } else {
                "in_memory"
            };
            #[cfg(not(feature = "substrate"))]
            let routed = if gs.is_some() { "graph_store" } else { "in_memory" };
            tracing::info!(routed, "cognitive: energy subsystem activated");
            store
        });

        // Synapses — substrate variant requires BOTH substrate (weight
        // column) + graph_store (structural edges).
        #[cfg(feature = "synapse")]
        let synapse = self.synapse_config.map(|config| {
            let store = {
                #[cfg(feature = "substrate")]
                {
                    match (&sub, &gs) {
                        (Some(substrate), Some(graph_store)) => {
                            Arc::new(SynapseStore::with_substrate(
                                config,
                                Arc::clone(graph_store),
                                Arc::clone(substrate),
                            ))
                        }
                        (_, Some(graph_store)) => Arc::new(SynapseStore::with_graph_store(
                            config,
                            Arc::clone(graph_store),
                        )),
                        (_, None) => Arc::new(SynapseStore::new(config)),
                    }
                }
                #[cfg(not(feature = "substrate"))]
                {
                    match &gs {
                        Some(graph_store) => Arc::new(SynapseStore::with_graph_store(
                            config,
                            Arc::clone(graph_store),
                        )),
                        None => Arc::new(SynapseStore::new(config)),
                    }
                }
            };
            let listener = Arc::new(SynapseListener::new(Arc::clone(&store)));
            scheduler.register_listener(listener);
            active_count += 1;
            #[cfg(feature = "substrate")]
            let routed = if sub.is_some() && gs.is_some() {
                "substrate+graph"
            } else if gs.is_some() {
                "graph_store"
            } else {
                "in_memory"
            };
            #[cfg(not(feature = "substrate"))]
            let routed = if gs.is_some() { "graph_store" } else { "in_memory" };
            tracing::info!(routed, "cognitive: synapse subsystem activated");
            store
        });

        // Fabric
        #[cfg(feature = "fabric")]
        let fabric = if self.fabric_enabled {
            let store = match &gs {
                Some(graph_store) => {
                    Arc::new(FabricStore::with_graph_store(Arc::clone(graph_store)))
                }
                None => Arc::new(FabricStore::new()),
            };
            let listener = Arc::new(FabricListener::new(Arc::clone(&store)));
            scheduler.register_listener(listener);
            active_count += 1;
            tracing::info!("cognitive: fabric subsystem activated");
            Some(store)
        } else {
            None
        };

        // Co-change
        #[cfg(feature = "co-change")]
        let co_change = self.co_change_config.map(|config| {
            let store = match &gs {
                Some(graph_store) => Arc::new(CoChangeStore::with_graph_store(
                    config,
                    Arc::clone(graph_store),
                )),
                None => Arc::new(CoChangeStore::new(config)),
            };
            let detector = Arc::new(CoChangeDetector::new(Arc::clone(&store)));
            scheduler.register_listener(detector);
            active_count += 1;
            tracing::info!("cognitive: co-change subsystem activated");
            store
        });

        // Kernel embeddings
        #[cfg(feature = "kernel")]
        let kernel = self.kernel_config.and_then(|config| {
            let Some(lpg) = self.lpg_store else {
                tracing::warn!(
                    "cognitive: kernel subsystem enabled but no LpgStore provided \
                     (call with_lpg_store). Kernel embeddings will not be active."
                );
                return None;
            };
            let mut manager = KernelManager::new_untrained(Arc::clone(&lpg), config.seed);
            manager.set_alpha(config.alpha);
            manager.set_max_neighbors(config.max_neighbors);
            manager.debounce_threshold = config.debounce_threshold;
            // Attempt to restore Phi_0 from a previous run persisted in the store.
            // Falls back to random initialization if no weights are found.
            manager.load_phi();
            let manager = Arc::new(manager);
            let listener = Arc::new(KernelListener::new(Arc::clone(&manager)));
            scheduler.register_listener(listener);
            active_count += 1;
            tracing::info!("cognitive: kernel embedding subsystem activated");
            Some(manager)
        });

        let provenance = Arc::new(ProvenanceRecorder::new());

        tracing::info!(
            active = active_count,
            "cognitive engine built with {} active subsystem(s)",
            active_count
        );

        DefaultCognitiveEngine {
            #[cfg(feature = "energy")]
            energy,
            #[cfg(feature = "synapse")]
            synapse,
            #[cfg(feature = "fabric")]
            fabric,
            #[cfg(feature = "co-change")]
            co_change,
            #[cfg(feature = "kernel")]
            kernel,
            #[cfg(feature = "engram")]
            engram_store: None,
            #[cfg(feature = "engram")]
            engram_metrics: None,
            #[cfg(feature = "engram")]
            vector_index: None,
            active_count,
            tenant_manager: TenantManager::new(),
            provenance,
        }
    }
}

impl Default for CognitiveEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CognitiveEngineBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("CognitiveEngineBuilder");

        #[cfg(feature = "energy")]
        d.field("energy", &self.energy_config.is_some());

        #[cfg(feature = "synapse")]
        d.field("synapse", &self.synapse_config.is_some());

        #[cfg(feature = "fabric")]
        d.field("fabric", &self.fabric_enabled);

        #[cfg(feature = "co-change")]
        d.field("co_change", &self.co_change_config.is_some());

        #[cfg(feature = "kernel")]
        d.field("kernel", &self.kernel_config.is_some());

        d.field("graph_store", &self.graph_store.is_some());

        #[cfg(feature = "kernel")]
        d.field("lpg_store", &self.lpg_store.is_some());

        #[cfg(feature = "substrate")]
        d.field("substrate", &self.substrate.is_some());

        d.finish()
    }
}
