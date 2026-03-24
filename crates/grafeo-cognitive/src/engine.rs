//! CognitiveEngine trait and `DefaultCognitiveEngine` implementation.
//!
//! The [`CognitiveEngine`] trait provides unified access to all cognitive
//! subsystems. [`CognitiveEngineBuilder`] constructs a [`DefaultCognitiveEngine`]
//! with selected subsystems and registers their listeners with the reactive
//! [`Scheduler`].

use crate::config::CognitiveConfig;
#[allow(unused_imports)]
use std::sync::Arc;

// Conditional imports based on feature flags
#[cfg(feature = "energy")]
use crate::energy::{EnergyConfig, EnergyListener, EnergyStore};

#[cfg(feature = "synapse")]
use crate::synapse::{SynapseConfig, SynapseListener, SynapseStore};

#[cfg(feature = "fabric")]
use crate::fabric::{FabricListener, FabricStore};

#[cfg(feature = "co-change")]
use crate::co_change::{CoChangeConfig, CoChangeDetector, CoChangeStore};

use grafeo_reactive::Scheduler;

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

    /// Returns the number of active subsystems.
    fn active_subsystem_count(&self) -> usize;
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

    /// Count of active subsystems.
    active_count: usize,
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

    fn active_subsystem_count(&self) -> usize {
        self.active_count
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

        d.finish()
    }
}

// ---------------------------------------------------------------------------
// CognitiveEngineBuilder
// ---------------------------------------------------------------------------

/// Builder for [`DefaultCognitiveEngine`].
///
/// Allows selective activation of cognitive subsystems. When [`build`](Self::build)
/// is called, the corresponding [`MutationListener`](grafeo_reactive::MutationListener)s
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

        builder
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

    /// Builds the cognitive engine and registers listeners with the scheduler.
    ///
    /// Each enabled subsystem:
    /// 1. Creates a shared store (`Arc<XxxStore>`)
    /// 2. Creates a listener wrapping the store
    /// 3. Registers the listener with the scheduler
    ///
    /// Returns a [`DefaultCognitiveEngine`] holding references to all stores.
    pub fn build(self, #[allow(unused_variables)] scheduler: &Scheduler) -> DefaultCognitiveEngine {
        #[allow(unused_mut)]
        let mut active_count = 0;

        // Energy
        #[cfg(feature = "energy")]
        let energy = self.energy_config.map(|config| {
            let store = Arc::new(EnergyStore::new(config));
            let listener = Arc::new(EnergyListener::new(Arc::clone(&store)));
            scheduler.register_listener(listener);
            active_count += 1;
            tracing::info!("cognitive: energy subsystem activated");
            store
        });

        // Synapses
        #[cfg(feature = "synapse")]
        let synapse = self.synapse_config.map(|config| {
            let store = Arc::new(SynapseStore::new(config));
            let listener = Arc::new(SynapseListener::new(Arc::clone(&store)));
            scheduler.register_listener(listener);
            active_count += 1;
            tracing::info!("cognitive: synapse subsystem activated");
            store
        });

        // Fabric
        #[cfg(feature = "fabric")]
        let fabric = if self.fabric_enabled {
            let store = Arc::new(FabricStore::new());
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
            let store = Arc::new(CoChangeStore::new(config));
            let detector = Arc::new(CoChangeDetector::new(Arc::clone(&store)));
            scheduler.register_listener(detector);
            active_count += 1;
            tracing::info!("cognitive: co-change subsystem activated");
            store
        });

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
            active_count,
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

        d.finish()
    }
}
