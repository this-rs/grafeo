//! The main database struct and operations.
//!
//! Start here with [`ObrainDB`] - it's your handle to everything.
//!
//! Operations are split across focused submodules:
//! - `query` - Query execution (execute, execute_cypher, etc.)
//! - `crud` - Node/edge CRUD operations
//! - `index` - Property, vector, and text index management
//! - `search` - Vector, text, and hybrid search
//! - `embed` - Embedding model management
//! - `persistence` - Save, load, snapshots, iteration
//! - `admin` - Stats, introspection, diagnostics, CDC

mod admin;
#[cfg(feature = "cognitive")]
pub(crate) mod annotator;
mod crud;
#[cfg(feature = "embed")]
mod embed;
mod index;
// T17 final cutover (2026-04-23): the `persistence`, `mmap_store`,
// `nocache_reader`, and `native_writer` modules were deleted. They
// implemented the legacy single-file `.obrain` v1 (bincode snapshot)
// and v2 (mmap native) formats, which are retired in favor of the
// directory-based substrate backend. Users with a pre-substrate
// single-file database must use obrain-migrate ≤ v0.0.1 to convert
// to the directory-based LpgStore layout first, then a current
// release to convert that into substrate. Search archaeology:
// persistence.rs (1840 LOC) + mmap_store.rs (1315 LOC) +
// nocache_reader.rs (104 LOC) + native_writer.rs (133 LOC) +
// 26 feature-gated blocks in this file removed on 2026-04-23.
mod query;
#[cfg(feature = "rdf")]
mod rdf_ops;
mod search;
#[cfg(feature = "wal")]
pub mod wal_store;

#[cfg(feature = "wal")]
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use parking_lot::RwLock;

#[cfg(all(feature = "wal", feature = "tiered-storage"))]
use obrain_adapters::storage::wal::CheckpointMetadata;
#[cfg(feature = "wal")]
use obrain_adapters::storage::wal::{
    DurabilityMode as WalDurabilityMode, LpgWal, WalConfig, WalRecord, WalRecovery,
};
use obrain_common::memory::buffer::{BufferManager, BufferManagerConfig};
#[cfg(all(feature = "wal", feature = "tiered-storage"))]
use obrain_common::types::TransactionId;
use obrain_common::utils::error::Result;
use obrain_core::graph::GraphStoreMut;

#[cfg(feature = "rdf")]
use obrain_core::graph::rdf::RdfStore;

use crate::catalog::Catalog;
use crate::config::Config;
use crate::query::cache::QueryCache;
use crate::session::Session;
use crate::transaction::TransactionManager;

/// Your handle to a Obrain database.
///
/// Start here. Create one with [`new_in_memory()`](Self::new_in_memory) for
/// quick experiments, or [`open()`](Self::open) for persistent storage.
/// Then grab a [`session()`](Self::session) to start querying.
///
/// # Examples
///
/// ```
/// use obrain_engine::ObrainDB;
///
/// // Quick in-memory database
/// let db = ObrainDB::new_in_memory();
///
/// // Add some data
/// db.create_node(&["Person"]);
///
/// // Query it
/// let session = db.session();
/// let result = session.execute("MATCH (p:Person) RETURN p")?;
/// # Ok::<(), obrain_common::utils::error::Error>(())
/// ```
pub struct ObrainDB {
    /// Database configuration.
    pub(super) config: Config,
    /// The underlying graph store.
    ///
    /// T17 Step 24 (2026-04-23): retyped from `Arc<LpgStore>` to the
    /// erased trait object. In substrate mode this holds a
    /// substrate-backed store; in the legacy path it holds an
    /// in-memory `LpgStore`. Both variants dispatch through
    /// `GraphStoreMut` — MVCC, named-graph, index-management, and
    /// introspection hooks all live on the trait with
    /// substrate-compatible defaults (LpgStore overrides delegate to
    /// its inherent impls).
    pub(super) store: Arc<dyn GraphStoreMut>,
    /// Schema and metadata catalog shared across sessions.
    pub(super) catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "rdf")]
    pub(super) rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    pub(super) transaction_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    pub(super) buffer_manager: Arc<BufferManager>,
    /// Write-ahead log manager (if durability is enabled).
    #[cfg(feature = "wal")]
    pub(super) wal: Option<Arc<LpgWal>>,
    /// Shared WAL graph context tracker. Tracks which named graph was last
    /// written to the WAL, so concurrent sessions can emit `SwitchGraph`
    /// records only when the context actually changes.
    #[cfg(feature = "wal")]
    pub(super) wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
    /// Query cache for parsed and optimized plans.
    pub(super) query_cache: Arc<QueryCache>,
    /// Shared commit counter for auto-GC across sessions.
    pub(super) commit_counter: Arc<AtomicUsize>,
    /// Whether the database is open.
    pub(super) is_open: RwLock<bool>,
    /// Change data capture log for tracking mutations.
    #[cfg(feature = "cdc")]
    pub(super) cdc_log: Arc<crate::cdc::CdcLog>,
    /// Registered embedding models for text-to-vector conversion.
    #[cfg(feature = "embed")]
    pub(super) embedding_models:
        RwLock<hashbrown::HashMap<String, Arc<dyn crate::embedding::EmbeddingModel>>>,
    /// External graph store (when using with_store()).
    /// When set, sessions route queries through this store instead of the built-in LpgStore.
    pub(super) external_store: Option<Arc<dyn GraphStoreMut>>,
    /// Typed [`SubstrateStore`] handle, retained alongside the erased
    /// `external_store` when the database is opened via `open_substrate`.
    ///
    /// Cognitive stores (energy/scar/utility/affinity/synapse) need the
    /// concrete substrate handle to call its column-view APIs
    /// (`reinforce_edge_synapse_f32`, `decay_all_edge_synapse`, etc.) which
    /// are not part of the `GraphStoreMut` trait. `None` when the database
    /// is backed by `LpgStore` (legacy path).
    ///
    /// [`SubstrateStore`]: obrain_substrate::SubstrateStore
    pub(super) substrate_store: Option<Arc<obrain_substrate::SubstrateStore>>,
    /// Metrics registry shared across all sessions.
    #[cfg(feature = "metrics")]
    pub(crate) metrics: Option<Arc<crate::metrics::MetricsRegistry>>,
    /// Persistent graph context for one-shot `execute()` calls.
    /// When set, each call to `session()` pre-configures the session to this graph.
    /// Updated after every one-shot `execute()` to reflect `USE GRAPH` / `SESSION RESET`.
    current_graph: RwLock<Option<String>>,
    /// Whether this database is open in read-only mode.
    /// When true, sessions automatically enforce read-only transactions.
    read_only: bool,
    /// Cognitive engine — unified access to energy, synapses, fabric, etc.
    /// Only present when the `cognitive` feature flag is enabled at compile time.
    #[cfg(feature = "cognitive")]
    cognitive_engine: Option<Arc<dyn obrain_cognitive::CognitiveEngine>>,
    /// Reactive scheduler (keeps the background task alive).
    #[cfg(feature = "cognitive")]
    _cognitive_scheduler: Option<obrain_reactive::Scheduler>,
}

impl ObrainDB {
    /// Creates an in-memory database, fast to create, gone when dropped.
    ///
    /// Use this for tests, experiments, or when you don't need persistence.
    /// For data that survives restarts, use [`open()`](Self::open) instead.
    ///
    /// # Panics
    ///
    /// Panics if the internal arena allocator cannot be initialized (out of memory).
    /// Use [`with_config()`](Self::with_config) for a fallible alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// use obrain_engine::ObrainDB;
    ///
    /// let db = ObrainDB::new_in_memory();
    /// let session = db.session();
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// # Ok::<(), obrain_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory()).expect("In-memory database creation should not fail")
    }

    /// Opens a database at the given path, creating it if it doesn't exist.
    ///
    /// If you've used this path before, Obrain recovers your data from the
    /// write-ahead log automatically. First open on a new path creates an
    /// empty database.
    ///
    /// # Errors
    ///
    /// Returns an error if the path isn't writable or recovery fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use obrain_engine::ObrainDB;
    ///
    /// let db = ObrainDB::open("./my_social_network")?;
    /// # Ok::<(), obrain_common::utils::error::Error>(())
    /// ```
    #[cfg(feature = "wal")]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        // T17 cutover: substrate is the single storage backend. The former
        // LpgStore-backed path (`Config::persistent` → `with_config`) and the
        // `OBRAIN_BACKEND=substrate` runtime knob are gone. Any lingering
        // `OBRAIN_BACKEND` env-var setting is ignored on purpose.
        Self::open_substrate(path)
    }

    /// Opens a database backed by the `SubstrateStore` (mmap + WAL native).
    ///
    /// This is the T5 migration path. Internally, it constructs a
    /// [`obrain_substrate::SubstrateStore`] at the given path and routes all
    /// queries and mutations through it via the [`with_store`] constructor.
    ///
    /// Since T17 cutover, [`ObrainDB::open`] delegates unconditionally to
    /// this method — substrate is the single storage backend. Kept as a
    /// public API for callers that want to express the substrate intent
    /// explicitly in tests or tooling.
    ///
    /// # Errors
    ///
    /// Returns an error if the substrate store cannot be created/opened at
    /// `path`.
    ///
    /// [`with_store`]: Self::with_store
    pub fn open_substrate(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        // Detect existing substrate: if `substrate.meta` is present, open;
        // otherwise create. This mirrors the semantics of ObrainDB::open(path)
        // on the legacy path (open-or-create idempotent).
        let meta_path = path_ref.join("substrate.meta");
        let store = if meta_path.is_file() {
            obrain_substrate::SubstrateStore::open(path_ref).map_err(|e| {
                obrain_common::utils::error::Error::Internal(format!(
                    "substrate: failed to open existing store at {}: {e}",
                    path_ref.display()
                ))
            })?
        } else {
            obrain_substrate::SubstrateStore::create(path_ref).map_err(|e| {
                obrain_common::utils::error::Error::Internal(format!(
                    "substrate: failed to create store at {}: {e}",
                    path_ref.display()
                ))
            })?
        };
        // Keep a typed handle in addition to the erased `GraphStoreMut` one:
        // cognitive stores reach for the substrate column APIs (reinforce /
        // decay_all / iter_live_synapse_weights) which are not part of the
        // `GraphStoreMut` trait.
        let typed: Arc<obrain_substrate::SubstrateStore> = Arc::new(store);
        let erased: Arc<dyn GraphStoreMut> = Arc::clone(&typed) as Arc<dyn GraphStoreMut>;
        let mut db = Self::with_store(erased, Config::in_memory())?;
        db.substrate_store = Some(Arc::clone(&typed));

        // T17 Step 3 W1 — wire the cognitive engine to substrate column
        // routing. `with_store` leaves `cognitive_engine: None`; substrate-
        // backed databases deserve the same reactive bus + scheduler + built
        // engine as the legacy `with_config` path, but with the substrate
        // handle threaded through `CognitiveEngineBuilder::with_substrate` so
        // Energy/Synapse stores pick up the column-native constructors.
        #[cfg(feature = "cognitive")]
        {
            let bus = obrain_reactive::MutationBus::new();
            let scheduler =
                obrain_reactive::Scheduler::new(&bus, obrain_reactive::BatchConfig::default());
            let config = obrain_cognitive::CognitiveConfig::default();
            #[allow(unused_mut)]
            let mut builder = obrain_cognitive::CognitiveEngineBuilder::from_config(&config)
                .with_graph_store(db.graph_store())
                .with_substrate(Arc::clone(&typed));
            // T17 Step 19 W2c — wire the kernel subsystem to the substrate-routed
            // store (NOT the dummy `db.store()` handle; see gotcha W4.p4.D1.s9).
            // `db.graph_store()` resolves to the real substrate backend in
            // substrate mode, giving KernelManager the populated graph instead
            // of the empty dummy LpgStore. Fixes the latent wiring gap where
            // the kernel subsystem was never attached in substrate mode.
            #[cfg(feature = "kernel")]
            {
                builder = builder.with_kernel_store(db.graph_store());
            }
            let engine = builder.build(&scheduler);
            db.cognitive_engine =
                Some(Arc::new(engine) as Arc<dyn obrain_cognitive::CognitiveEngine>);
            db._cognitive_scheduler = Some(scheduler);
        }

        Ok(db)
    }

    /// Opens an existing database in read-only mode.
    ///
    /// Uses a shared file lock, so multiple processes can read the same
    /// `.obrain` file concurrently. The database loads the last checkpoint
    /// snapshot but does **not** replay the WAL or allow mutations.
    ///
    /// Currently only supports the single-file (`.obrain`) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist or can't be read.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use obrain_engine::ObrainDB;
    ///
    /// let db = ObrainDB::open_read_only("./my_graph.obrain")?;
    /// let session = db.session();
    /// let result = session.execute("MATCH (n) RETURN n LIMIT 10")?;
    /// // Mutations will return an error:
    /// // session.execute("INSERT (:Person)") => Err(ReadOnly)
    /// # Ok::<(), obrain_common::utils::error::Error>(())
    /// ```
    /// Creates a database with custom configuration.
    ///
    /// Use this when you need fine-grained control over memory limits,
    /// thread counts, or persistence settings. For most cases,
    /// [`new_in_memory()`](Self::new_in_memory) or [`open()`](Self::open)
    /// are simpler.
    ///
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use obrain_engine::{ObrainDB, Config};
    ///
    /// // In-memory with a 512MB limit
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024);
    ///
    /// let db = ObrainDB::with_config(config)?;
    /// # Ok::<(), obrain_common::utils::error::Error>(())
    /// ```
    pub fn with_config(config: Config) -> Result<Self> {
        // Validate configuration before proceeding
        config
            .validate()
            .map_err(|e| obrain_common::utils::error::Error::Internal(e.to_string()))?;

        // T17 final cutover (2026-04-23): `with_config` no longer builds
        // an in-memory LpgStore. Substrate is the single backend — an
        // in-memory config opens a substrate tempfile (auto-cleaned on
        // drop), a persistent config delegates to
        // `ObrainDB::open_substrate(path)`. Tests that rely on LpgStore-
        // inherent MVCC / named-graph semantics (~17 tests in session/
        // mod.rs + transaction/prepared.rs) are marked `#[ignore]` — the
        // substrate backend has its own transaction + persistence model
        // and the legacy LpgStore MVCC suite is no longer covered by
        // this entry point.
        if let Some(ref db_path) = config.path {
            // Persistent: delegate straight to the substrate open path.
            return Self::open_substrate(db_path);
        }

        let store = obrain_substrate::SubstrateStore::open_tempfile().map_err(|e| {
            obrain_common::utils::error::Error::Internal(format!(
                "substrate tempfile creation failed: {e}"
            ))
        })?;
        let typed: Arc<obrain_substrate::SubstrateStore> = Arc::new(store);
        let erased: Arc<dyn GraphStoreMut> = Arc::clone(&typed) as Arc<dyn GraphStoreMut>;
        let store = erased;
        #[cfg(feature = "rdf")]
        let rdf_store = Arc::new(RdfStore::new());
        let transaction_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: config
                .spill_path
                .clone()
                .or_else(|| config.path.as_ref().map(|p| p.join("spill"))),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Create catalog early so WAL replay can restore schema definitions
        let catalog = Arc::new(Catalog::new());

        let is_read_only = config.access_mode == crate::config::AccessMode::ReadOnly;

        // T17 final cutover (2026-04-23): the single-file `.obrain`
        // format handling that lived here (v1 bincode snapshot + v2
        // mmap native, ~140 LOC) was feature-gated on `obrain-file`
        // which was retired in commit `b0f5bd11`. The block was
        // `let file_manager: Option<Arc<ObrainFileManager>>` plus
        // `let mut native_mmap_store: Option<Arc<mmap_store::MmapStore>>`,
        // both unreachable once the feature was removed. Substrate is
        // now the single backend; legacy `.obrain` files are
        // converted via `obrain-migrate` before opening.

        // Determine whether to use the WAL directory path (legacy) or sidecar
        // Read-only mode skips WAL entirely (no recovery, no creation).
        #[cfg(feature = "wal")]
        let wal = if is_read_only {
            None
        } else if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                // T17 final cutover: WAL is always a directory inside
                // the database path. The pre-cutover sidecar layout
                // (when the store was a single `.obrain` file) no
                // longer applies — substrate is directory-based from
                // the start.
                std::fs::create_dir_all(db_path)?;
                let wal_path = db_path.join("wal");
                let is_single_file = false;

                // T17 final cutover (2026-04-23): legacy tiered-storage
                // epoch-file recovery + LpgWal replay are gone.
                // Substrate's `open_substrate()` path handles its own WAL
                // recovery natively. `with_config` always builds a fresh
                // substrate tempfile (see above), so there is no prior
                // state to replay here.
                let _ = &wal_path;
                let _ = &catalog;

                // Open/create WAL manager with configured durability
                let wal_durability = match config.wal_durability {
                    crate::config::DurabilityMode::Sync => WalDurabilityMode::Sync,
                    crate::config::DurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    } => WalDurabilityMode::Batch {
                        max_delay_ms,
                        max_records,
                    },
                    crate::config::DurabilityMode::Adaptive { target_interval_ms } => {
                        WalDurabilityMode::Adaptive { target_interval_ms }
                    }
                    crate::config::DurabilityMode::NoSync => WalDurabilityMode::NoSync,
                };
                let wal_config = WalConfig {
                    durability: wal_durability,
                    ..WalConfig::default()
                };
                let wal_manager = LpgWal::with_config(&wal_path, wal_config)?;
                Some(Arc::new(wal_manager))
            } else {
                None
            }
        } else {
            None
        };

        // Create query cache with default capacity (1000 queries)
        let query_cache = Arc::new(QueryCache::default());

        // After all snapshot/WAL recovery, sync TransactionManager epoch
        // with the store so queries use the correct viewing epoch.
        #[cfg(feature = "temporal")]
        transaction_manager.sync_epoch(store.current_epoch());

        // --- Cognitive engine initialization ---
        // Creates the reactive bus + scheduler + cognitive engine with default config.
        // Zero-cost: this entire block is compiled out when `cognitive` feature is off.
        #[cfg(feature = "cognitive")]
        let (cognitive_engine, _cognitive_scheduler) = {
            let bus = obrain_reactive::MutationBus::new();
            let scheduler =
                obrain_reactive::Scheduler::new(&bus, obrain_reactive::BatchConfig::default());
            let config = obrain_cognitive::CognitiveConfig::default();
            #[allow(unused_mut)]
            let mut builder = obrain_cognitive::CognitiveEngineBuilder::from_config(&config);
            #[cfg(feature = "kernel")]
            {
                // T17 Step 19 W2c + Step 24 `with_config` retype: the
                // local `store` is now `Arc<dyn GraphStoreMut>` (substrate
                // tempfile). Pass it straight through to the kernel
                // subsystem — no concrete LpgStore cast needed.
                builder = builder.with_kernel_store(Arc::clone(&store));
            }
            let engine = builder.build(&scheduler);
            (
                Some(Arc::new(engine) as Arc<dyn obrain_cognitive::CognitiveEngine>),
                Some(scheduler),
            )
        };

        Ok(Self {
            config,
            store,
            catalog,
            #[cfg(feature = "rdf")]
            rdf_store,
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal,
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            external_store: None,
            substrate_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            read_only: is_read_only,
            #[cfg(feature = "cognitive")]
            cognitive_engine,
            #[cfg(feature = "cognitive")]
            _cognitive_scheduler,
        })
    }

    /// Creates a database backed by a custom [`GraphStoreMut`] implementation.
    ///
    /// The external store handles all data persistence. WAL, CDC, and index
    /// management are the responsibility of the store implementation.
    ///
    /// Query execution (all 6 languages, optimizer, planner) works through the
    /// provided store. Admin operations (schema introspection, persistence,
    /// vector/text indexes) are not available on external stores.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::sync::Arc;
    /// use obrain_engine::{ObrainDB, Config};
    /// use obrain_core::graph::GraphStoreMut;
    ///
    /// fn example(store: Arc<dyn GraphStoreMut>) -> obrain_common::utils::error::Result<()> {
    ///     let db = ObrainDB::with_store(store, Config::in_memory())?;
    ///     let result = db.execute("MATCH (n) RETURN count(n)")?;
    ///     Ok(())
    /// }
    /// ```
    ///
    /// [`GraphStoreMut`]: obrain_core::graph::GraphStoreMut
    pub fn with_store(store: Arc<dyn GraphStoreMut>, config: Config) -> Result<Self> {
        config
            .validate()
            .map_err(|e| obrain_common::utils::error::Error::Internal(e.to_string()))?;

        // T17 Step 24 (2026-04-23): the dummy `LpgStore::new()` that used
        // to shadow the real store in this constructor is gone — the
        // `store` field is now `Arc<dyn GraphStoreMut>`, so it can hold
        // the real store directly (legacy or substrate). The legacy
        // Session path still needs a separate `external_store` handle
        // for MVCC dispatch, which is kept below.
        let transaction_manager = Arc::new(TransactionManager::new());

        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: None,
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        let query_cache = Arc::new(QueryCache::default());

        Ok(Self {
            config,
            store: Arc::clone(&store),
            catalog: Arc::new(Catalog::new()),
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager,
            buffer_manager,
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "wal")]
            wal_graph_context: Arc::new(parking_lot::Mutex::new(None)),
            query_cache,
            commit_counter: Arc::new(AtomicUsize::new(0)),
            is_open: RwLock::new(true),
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "embed")]
            embedding_models: RwLock::new(hashbrown::HashMap::new()),
            external_store: Some(store),
            substrate_store: None,
            #[cfg(feature = "metrics")]
            metrics: Some(Arc::new(crate::metrics::MetricsRegistry::new())),
            current_graph: RwLock::new(None),
            read_only: false,
            #[cfg(feature = "cognitive")]
            cognitive_engine: None,
            #[cfg(feature = "cognitive")]
            _cognitive_scheduler: None,
        })
    }

    #[must_use]
    pub fn session(&self) -> Session {
        let session_cfg = || crate::session::SessionConfig {
            transaction_manager: Arc::clone(&self.transaction_manager),
            query_cache: Arc::clone(&self.query_cache),
            catalog: Arc::clone(&self.catalog),
            adaptive_config: self.config.adaptive.clone(),
            factorized_execution: self.config.factorized_execution,
            graph_model: self.config.graph_model,
            query_timeout: self.config.query_timeout,
            commit_counter: Arc::clone(&self.commit_counter),
            gc_interval: self.config.gc_interval,
            read_only: self.read_only,
        };

        if let Some(ref ext_store) = self.external_store {
            return Session::with_external_store(Arc::clone(ext_store), session_cfg())
                .expect("arena allocation for external store session");
        }

        #[cfg(feature = "rdf")]
        let mut session = Session::with_rdf_store_and_adaptive(
            Arc::clone(&self.store),
            Arc::clone(&self.rdf_store),
            session_cfg(),
        );
        #[cfg(not(feature = "rdf"))]
        let mut session = Session::with_adaptive(
            Arc::clone(&self.store) as Arc<dyn GraphStoreMut>,
            session_cfg(),
        );

        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            session.set_wal(Arc::clone(wal), Arc::clone(&self.wal_graph_context));
        }

        #[cfg(feature = "cdc")]
        session.set_cdc_log(Arc::clone(&self.cdc_log));

        #[cfg(feature = "metrics")]
        {
            if let Some(ref m) = self.metrics {
                session.set_metrics(Arc::clone(m));
                m.session_created
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m.session_active
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        // Propagate persistent graph context to the new session
        if let Some(ref graph) = *self.current_graph.read() {
            session.use_graph(graph);
        }

        // Suppress unused_mut when cdc/wal are disabled
        let _ = &mut session;

        session
    }

    /// Returns the current graph name, if any.
    ///
    /// This is the persistent graph context used by one-shot `execute()` calls.
    /// It is updated whenever `execute()` encounters `USE GRAPH`, `SESSION SET GRAPH`,
    /// or `SESSION RESET`.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_graph.read().clone()
    }

    /// Sets the current graph context for subsequent one-shot `execute()` calls.
    ///
    /// This is equivalent to running `USE GRAPH <name>` but without creating a session.
    /// Pass `None` to reset to the default graph.
    pub fn set_current_graph(&self, name: Option<&str>) {
        *self.current_graph.write() = name.map(ToString::to_string);
    }

    /// Returns the adaptive execution configuration.
    #[must_use]
    pub fn adaptive_config(&self) -> &crate::config::AdaptiveConfig {
        &self.config.adaptive
    }

    /// Returns `true` if this database was opened in read-only mode.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the cognitive engine, if the `cognitive` feature is enabled and
    /// the engine was initialized.
    ///
    /// This provides unified access to energy, synapse, fabric, and other
    /// cognitive subsystems.
    #[cfg(feature = "cognitive")]
    #[must_use]
    pub fn cognitive_engine(&self) -> Option<&Arc<dyn obrain_cognitive::CognitiveEngine>> {
        self.cognitive_engine.as_ref()
    }

    /// Returns the graph data model of this database.
    #[must_use]
    pub fn graph_model(&self) -> crate::config::GraphModel {
        self.config.graph_model
    }

    /// Returns the configured memory limit in bytes, if any.
    #[must_use]
    pub fn memory_limit(&self) -> Option<usize> {
        self.config.memory_limit
    }

    /// Returns a point-in-time snapshot of all metrics.
    ///
    /// If the `metrics` feature is disabled or the registry is not
    /// initialized, returns a default (all-zero) snapshot.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics(&self) -> crate::metrics::MetricsSnapshot {
        let mut snapshot = self
            .metrics
            .as_ref()
            .map_or_else(crate::metrics::MetricsSnapshot::default, |m| m.snapshot());

        // Augment with cache stats from the query cache (not tracked in the registry)
        let cache_stats = self.query_cache.stats();
        snapshot.cache_hits = cache_stats.parsed_hits + cache_stats.optimized_hits;
        snapshot.cache_misses = cache_stats.parsed_misses + cache_stats.optimized_misses;
        snapshot.cache_size = cache_stats.parsed_size + cache_stats.optimized_size;
        snapshot.cache_invalidations = cache_stats.invalidations;

        snapshot
    }

    /// Returns all metrics in Prometheus text exposition format.
    ///
    /// The output is ready to serve from an HTTP `/metrics` endpoint.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn metrics_prometheus(&self) -> String {
        self.metrics
            .as_ref()
            .map_or_else(String::new, |m| m.to_prometheus())
    }

    /// Resets all metrics counters and histograms to zero.
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&self) {
        if let Some(ref m) = self.metrics {
            m.reset();
        }
        self.query_cache.reset_stats();
    }

    /// Returns the underlying (default) store.
    ///
    /// Returns the underlying graph store as `&Arc<dyn GraphStoreMut>`.
    ///
    /// T17 Step 24 (2026-04-23): retyped from the concrete
    /// `&Arc<LpgStore>` to the trait object. Callers that previously
    /// relied on LpgStore-inherent methods (memory_breakdown,
    /// edges_with_type, all_edges, nodes_with_label, create/drop
    /// property index, named-graph API) now reach the same surface
    /// through `GraphStoreMut` trait methods — substrate-compatible
    /// defaults return empty/zero values; LpgStore overrides
    /// delegate to inherent impls.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn GraphStoreMut> {
        &self.store
    }

    /// Returns the read-only mmap store if available (v2 native format).
    ///
    /// Returns the store for the currently active graph.
    ///
    /// If [`current_graph`](Self::current_graph) is `None` or `"default"`, returns
    /// the default store. Otherwise looks up the named graph in the root store.
    /// Falls back to the default store if the named graph does not exist.
    #[allow(dead_code)] // Reserved for future graph-aware CRUD methods
    fn active_store(&self) -> Arc<dyn GraphStoreMut> {
        let graph_name = self.current_graph.read().clone();
        match graph_name {
            None => Arc::clone(&self.store),
            Some(ref name) if name.eq_ignore_ascii_case("default") => Arc::clone(&self.store),
            Some(ref name) => self
                .store
                .named_graph(name)
                .unwrap_or_else(|| Arc::clone(&self.store)),
        }
    }

    // === Named Graph Management ===

    /// Creates a named graph. Returns `true` if created, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if arena allocation fails, or if the database is
    /// substrate-backed (substrate is a single-graph store — named graphs
    /// are an LpgStore-only feature; see T17b for future substrate support).
    pub fn create_graph(&self, name: &str) -> Result<bool> {
        // T17 W3c slice 4: named graphs are an LpgStore-inherent feature
        // absent from `SubstrateStore`. Creating against the dummy LpgStore
        // would succeed locally but would not be visible to any substrate
        // query path. Gate explicitly.
        if self.substrate_store.is_some() {
            return Err(obrain_common::utils::error::Error::Internal(
                "create_graph() is not supported in substrate mode — substrate \
                 is single-graph by design. Named-graph support is tracked as \
                 T17b."
                    .to_string(),
            ));
        }
        self.store
            .create_named_graph(name)
            .map_err(|e| obrain_common::utils::error::Error::Internal(e.to_string()))
    }

    /// Drops a named graph. Returns `true` if dropped, `false` if it did not exist.
    ///
    /// In substrate mode this always returns `false` — substrate is a
    /// single-graph store with no named graphs to drop.
    pub fn drop_graph(&self, name: &str) -> bool {
        if self.substrate_store.is_some() {
            // No named graphs exist on substrate, so there is nothing to drop.
            return false;
        }
        self.store.drop_named_graph(name)
    }

    /// Returns all named graph names.
    ///
    /// In substrate mode this always returns an empty `Vec` — substrate is
    /// a single-graph store (only the default graph exists).
    #[must_use]
    pub fn list_graphs(&self) -> Vec<String> {
        if self.substrate_store.is_some() {
            return Vec::new();
        }
        self.store.named_graph_names()
    }

    /// Returns the active data store as a trait object — the single
    /// authoritative backend for data operations (CRUD, queries, epoch,
    /// stats).
    ///
    /// Resolution priority:
    /// 1. `substrate_store` — typed substrate handle (preferred when present).
    /// 2. `external_store` — user-supplied `Arc<dyn GraphStoreMut>` from
    ///    `with_store()`.
    /// 3. `self.store` — the legacy `Arc<LpgStore>` field. In substrate
    ///    mode this holds a **dummy** created in `with_store()` and should
    ///    never be reached through this path; it is kept only as the
    ///    LpgStore-backed fallback for `with_config()` and tests.
    ///
    /// Use this helper internally in place of `self.store.X()` whenever
    /// `X` is a trait method (`current_epoch`, `node_count`, `edge_count`,
    /// `create_node`, `get_node`, ...). That routes the call to the real
    /// backend instead of the dummy, fixing the latent T17 bug where
    /// substrate-backed DBs silently returned dummy zeros/no-ops.
    ///
    /// See gotcha note `0b9fcabe-a780-4149-8709-ed32ee9ed82e`.
    #[must_use]
    pub(super) fn data_store(&self) -> Arc<dyn GraphStoreMut> {
        if let Some(ref sub) = self.substrate_store {
            return Arc::clone(sub) as Arc<dyn GraphStoreMut>;
        }
        if let Some(ref ext_store) = self.external_store {
            return Arc::clone(ext_store);
        }
        Arc::clone(&self.store) as Arc<dyn GraphStoreMut>
    }

    /// Returns the graph store as a trait object.
    ///
    /// This provides the [`GraphStoreMut`] interface for code that should work
    /// with any storage backend. Use this when you only need graph read/write
    /// operations and don't need admin methods like index management.
    ///
    /// [`GraphStoreMut`]: obrain_core::graph::GraphStoreMut
    #[must_use]
    pub fn graph_store(&self) -> Arc<dyn GraphStoreMut> {
        self.data_store()
    }

    /// Garbage collects old MVCC versions that are no longer visible.
    ///
    /// Determines the minimum epoch required by active transactions and prunes
    /// version chains older than that threshold. Also cleans up completed
    /// transaction metadata in the transaction manager.
    pub fn gc(&self) {
        let min_epoch = self.transaction_manager.min_active_epoch();
        self.store.gc_versions(min_epoch);
        self.transaction_manager.gc();
    }

    /// Returns the buffer manager for memory-aware operations.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns the query cache.
    #[must_use]
    pub fn query_cache(&self) -> &Arc<QueryCache> {
        &self.query_cache
    }

    /// Clears all cached query plans.
    ///
    /// This is called automatically after DDL operations, but can also be
    /// invoked manually after external schema changes (e.g., WAL replay,
    /// import) or when you want to force re-optimization of all queries.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Closes the database, flushing all pending writes.
    ///
    /// For persistent databases, this ensures everything is safely on disk.
    /// Called automatically when the database is dropped, but you can call
    /// it explicitly if you need to guarantee durability at a specific point.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions).
    pub fn close(&self) -> Result<()> {
        let mut is_open = self.is_open.write();
        if !*is_open {
            return Ok(());
        }

        // Read-only databases: just release the shared lock, no checkpointing
        if self.read_only {
            *is_open = false;
            return Ok(());
        }

        // T17 final cutover: `is_single_file` is always `false` — substrate
        // databases are directory-based, legacy single-file `.obrain` v1/v2
        // was retired (feature `obrain-file` removed in commit `b0f5bd11`).
        let is_single_file = false;

        // Commit WAL records (legacy directory format only).
        //
        // IMPORTANT: We must NOT write checkpoint metadata here because the
        // directory format has no base snapshot — data lives only in the WAL
        // files. Writing a checkpoint would cause recovery to skip all WAL
        // files before the checkpoint sequence, effectively losing all data.
        // (The single-file `.obrain` format handles this correctly via
        // `checkpoint_to_file()` which exports a real snapshot first.)
        #[cfg(feature = "wal")]
        if !is_single_file && let Some(ref wal) = self.wal {
            // Use the last assigned transaction ID, or create one for the commit
            let commit_tx = self
                .transaction_manager
                .last_assigned_transaction_id()
                .unwrap_or_else(|| self.transaction_manager.begin());

            // Log a TransactionCommit to mark all pending records as committed,
            // so recovery knows to replay them.
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: commit_tx,
            })?;

            // Flush to disk but do NOT write checkpoint.meta
            wal.sync()?;
        }

        // Substrate is the single backend (T17 cutover): persist its dict
        // (registries + high-water marks) and msync the mmap zones before
        // marking the database closed. Without this, reopen sees an empty
        // or stale dict and treats all previously allocated nodes/edges as
        // non-live (is_live_on_disk → false).
        if let Some(ref sub) = self.substrate_store
            && let Err(e) = sub.flush()
        {
            tracing::error!("substrate flush during close failed: {e}");
        }

        *is_open = false;
        Ok(())
    }

    /// Commits the WAL — flushes pending records and optionally checkpoints
    /// the snapshot to the `.obrain` file.
    ///
    /// This is the primary durability API: callers should invoke this after
    /// a batch of mutations that must survive process restarts.
    #[cfg(feature = "wal")]
    pub fn commit_wal(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.flush()?;
        }

        // T17 final cutover: `file_manager` (single-file `.obrain` v1/v2
        // format) was retired; no checkpoint_to_file step is needed in
        // substrate mode — the substrate's own flush below handles
        // durability for mmap zones and dict columns.

        // Substrate dict + mmap zones must be persisted alongside the WAL
        // commit for cross-restart durability (T17 cutover — unconditional).
        if let Some(ref sub) = self.substrate_store {
            sub.flush().map_err(|e| {
                obrain_common::utils::error::Error::Internal(format!(
                    "substrate flush during commit_wal failed: {e}"
                ))
            })?;
        }

        Ok(())
    }

    /// Returns the typed WAL if available.
    #[cfg(feature = "wal")]
    #[must_use]
    pub fn wal(&self) -> Option<&Arc<LpgWal>> {
        self.wal.as_ref()
    }

    /// Returns a WAL-aware [`GraphStoreMut`] handle for direct-mutation consumers.
    ///
    /// When the database has a WAL, this returns an
    /// [`Arc<WalGraphStore>`](wal_store::WalGraphStore) that logs every mutation
    /// (create/delete nodes & edges, set properties) before delegating to the
    /// in-memory [`LpgStore`]. Without a WAL, it returns the raw
    /// `Arc<LpgStore>` directly.
    ///
    /// This is the sanctioned entry point for subsystems that need to mutate
    /// the graph outside a query (e.g. hub-level stores: AssetRegistry,
    /// UserBrain, PersonaStore, etc.) while preserving durability. It replaces
    /// the now-removed `PersistentStore` wrapper that used to live in
    /// `obrain-hub`.
    #[must_use]
    pub fn graph_store_mut(&self) -> Arc<dyn GraphStoreMut> {
        // T17 cutover: substrate is the single authoritative store. Its
        // `GraphStoreMut` impl is already WAL-native (every mutation logs a
        // WalRecord synchronously before touching mmap), so we do NOT wrap it
        // with `WalGraphStore` — doing so would funnel writes through the
        // dummy `self.store` LpgStore that lives next to the substrate in
        // `with_store`, silently dropping them on disk. Return the
        // substrate's erased handle directly.
        if let Some(ref sub) = self.substrate_store {
            return Arc::clone(sub) as Arc<dyn GraphStoreMut>;
        }
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            return Arc::new(wal_store::WalGraphStore::new(
                Arc::clone(&self.store) as Arc<dyn GraphStoreMut>,
                Arc::clone(wal),
                Arc::clone(&self.wal_graph_context),
            ));
        }
        Arc::clone(&self.store) as Arc<dyn GraphStoreMut>
    }

    /// Returns the typed [`SubstrateStore`] handle when the database is
    /// backed by the substrate backend, or `None` otherwise.
    ///
    /// This is the progressive-cutover hook for T6: cognitive stores
    /// (energy / scar / utility / affinity / synapse) use this to route
    /// cognitive mutations through the substrate column APIs
    /// (`reinforce_edge_synapse_f32`, `decay_all_edge_synapse`,
    /// `boost_edge_synapse_f32`, ...) which aren't part of the
    /// `GraphStoreMut` trait. When `None`, cognitive stores fall back to
    /// the LPG-property path (legacy behaviour).
    ///
    /// A typical call sequence from the hub:
    ///
    /// ```ignore
    /// let synapse = match db.substrate_handle() {
    ///     Some(sub) => SynapseStore::with_substrate(cfg, graph_store, sub),
    ///     None => SynapseStore::new(cfg, graph_store),
    /// };
    /// ```
    ///
    /// [`SubstrateStore`]: obrain_substrate::SubstrateStore
    #[must_use]
    pub fn substrate_handle(&self) -> Option<Arc<obrain_substrate::SubstrateStore>> {
        self.substrate_store.as_ref().map(Arc::clone)
    }

    /// Logs a WAL record if WAL is enabled.
    #[cfg(feature = "wal")]
    pub(super) fn log_wal(&self, record: &WalRecord) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.log(record)?;
        }
        Ok(())
    }

    // T17 final cutover (2026-04-23): `checkpoint_to_file` and
    // `file_manager` accessors were retired with the `obrain-file`
    // feature (commit `b0f5bd11`). Substrate is mmap + WAL-native and
    // persists automatically through its own flush path — no separate
    // single-file snapshot export is needed. A future slice (T17b)
    // may reintroduce an explicit snapshot export via
    // `SubstrateStore::snapshot_to_path()`.
}

impl Drop for ObrainDB {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            tracing::error!("Error closing database: {}", e);
        }
    }
}

impl crate::admin::AdminService for ObrainDB {
    fn info(&self) -> crate::admin::DatabaseInfo {
        self.info()
    }

    fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        self.detailed_stats()
    }

    fn schema(&self) -> crate::admin::SchemaInfo {
        self.schema()
    }

    fn validate(&self) -> crate::admin::ValidationResult {
        self.validate()
    }

    fn wal_status(&self) -> crate::admin::WalStatus {
        self.wal_status()
    }

    fn wal_checkpoint(&self) -> Result<()> {
        self.wal_checkpoint()
    }
}

// =========================================================================
// Query Result Types
// =========================================================================

/// The result of running a query.
///
/// Contains rows and columns, like a table. Use [`iter()`](Self::iter) to
/// loop through rows, or [`scalar()`](Self::scalar) if you expect a single value.
///
/// # Examples
///
/// ```
/// use obrain_engine::ObrainDB;
///
/// let db = ObrainDB::new_in_memory();
/// db.create_node(&["Person"]);
///
/// let result = db.execute("MATCH (p:Person) RETURN count(p) AS total")?;
///
/// // Check what we got
/// println!("Columns: {:?}", result.columns);
/// println!("Rows: {}", result.row_count());
///
/// // Iterate through results
/// for row in result.iter() {
///     println!("{:?}", row);
/// }
/// # Ok::<(), obrain_common::utils::error::Error>(())
/// ```
#[derive(Debug)]
pub struct QueryResult {
    /// Column names from the RETURN clause.
    pub columns: Vec<String>,
    /// Column types - useful for distinguishing NodeId/EdgeId from plain integers.
    pub column_types: Vec<obrain_common::types::LogicalType>,
    /// The actual result rows.
    pub rows: Vec<Vec<obrain_common::types::Value>>,
    /// Query execution time in milliseconds (if timing was enabled).
    pub execution_time_ms: Option<f64>,
    /// Number of rows scanned during query execution (estimate).
    pub rows_scanned: Option<u64>,
    /// Status message for DDL and session commands (e.g., "Created node type 'Person'").
    pub status_message: Option<String>,
    /// GQLSTATUS code per ISO/IEC 39075:2024, sec 23.
    pub gql_status: obrain_common::utils::GqlStatus,
}

impl QueryResult {
    /// Creates a fully empty query result (no columns, no rows).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: obrain_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a query result with only a status message (for DDL commands).
    #[must_use]
    pub fn status(msg: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: Some(msg.into()),
            gql_status: obrain_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![obrain_common::types::LogicalType::Any; len],
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: obrain_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Creates a new empty query result with column types.
    #[must_use]
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<obrain_common::types::LogicalType>,
    ) -> Self {
        Self {
            columns,
            column_types,
            rows: Vec::new(),
            execution_time_ms: None,
            rows_scanned: None,
            status_message: None,
            gql_status: obrain_common::utils::GqlStatus::SUCCESS,
        }
    }

    /// Sets the execution metrics on this result.
    pub fn with_metrics(mut self, execution_time_ms: f64, rows_scanned: u64) -> Self {
        self.execution_time_ms = Some(execution_time_ms);
        self.rows_scanned = Some(rows_scanned);
        self
    }

    /// Returns the execution time in milliseconds, if available.
    #[must_use]
    pub fn execution_time_ms(&self) -> Option<f64> {
        self.execution_time_ms
    }

    /// Returns the number of rows scanned, if available.
    #[must_use]
    pub fn rows_scanned(&self) -> Option<u64> {
        self.rows_scanned
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Extracts a single value from the result.
    ///
    /// Use this when your query returns exactly one row with one column,
    /// like `RETURN count(n)` or `RETURN sum(p.amount)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the result has multiple rows or columns.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.rows.len() != 1 || self.columns.len() != 1 {
            return Err(obrain_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows[0][0])
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<obrain_common::types::Value>> {
        self.rows.iter()
    }
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let table = obrain_common::fmt::format_result_table(
            &self.columns,
            &self.rows,
            self.execution_time_ms,
            self.status_message.as_deref(),
        );
        f.write_str(&table)
    }
}

/// Converts a [`obrain_common::types::Value`] to a concrete Rust type.
///
/// Implemented for common types like `i64`, `f64`, `String`, and `bool`.
/// Used by [`QueryResult::scalar()`] to extract typed values.
pub trait FromValue: Sized {
    /// Attempts the conversion, returning an error on type mismatch.
    fn from_value(value: &obrain_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &obrain_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| obrain_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &obrain_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| obrain_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &obrain_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            obrain_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &obrain_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| obrain_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_in_memory_database() {
        let db = ObrainDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config = Config::in_memory().with_threads(4).with_query_logging();

        let db = ObrainDB::with_config(config).unwrap();
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = ObrainDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_persistent_database_recovery() {
        use obrain_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Create database and add some data.
        // NodeId allocation is backend-dependent: substrate reserves NodeId(0)
        // as a sentinel and allocates from 1 upward, whereas the legacy
        // LpgStore started at 0. Capture the IDs returned by `create_node` so
        // the assertions below stay portable across backends.
        let (alix_id, gus_id) = {
            let db = ObrainDB::open(&db_path).unwrap();

            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));

            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));

            let _edge = db.create_edge(alix, gus, "KNOWS");

            // Explicitly close to flush WAL
            db.close().unwrap();
            (alix, gus)
        };

        // Reopen and verify data was recovered
        {
            let db = ObrainDB::open(&db_path).unwrap();

            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            // Verify nodes exist (use captured IDs — backend-agnostic).
            let node0 = db.get_node(alix_id);
            assert!(node0.is_some());

            let node1 = db.get_node(gus_id);
            assert!(node1.is_some());
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db");

        let db = ObrainDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]);
        db.delete_node(node);

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_wal_recovery_multiple_sessions() {
        // Tests that WAL recovery works correctly across multiple open/close cycles
        use obrain_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("multi_session_db");

        // Session 1: Create initial data.
        // Capture the substrate-allocated IDs so the cross-session assertions
        // below remain portable (substrate starts at NodeId(1); legacy LpgStore
        // started at 0).
        let alix_id = {
            let db = ObrainDB::open(&db_path).unwrap();
            let alix = db.create_node(&["Person"]);
            db.set_node_property(alix, "name", Value::from("Alix"));
            db.close().unwrap();
            alix
        };

        // Session 2: Add more data
        let gus_id = {
            let db = ObrainDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 1); // Previous data recovered
            let gus = db.create_node(&["Person"]);
            db.set_node_property(gus, "name", Value::from("Gus"));
            db.close().unwrap();
            gus
        };

        // Session 3: Verify all data
        {
            let db = ObrainDB::open(&db_path).unwrap();
            assert_eq!(db.node_count(), 2);

            // Verify properties were recovered correctly (use captured IDs).
            let node0 = db.get_node(alix_id).unwrap();
            assert!(node0.labels.iter().any(|l| l.as_str() == "Person"));

            let node1 = db.get_node(gus_id).unwrap();
            assert!(node1.labels.iter().any(|l| l.as_str() == "Person"));
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_database_consistency_after_mutations() {
        // Tests that database remains consistent after a series of create/delete operations
        use obrain_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("consistency_db");

        // Capture substrate-allocated IDs so the cross-session assertions
        // below remain portable across backends (substrate reserves NodeId(0)
        // as a sentinel).
        let (a_id, b_id, c_id) = {
            let db = ObrainDB::open(&db_path).unwrap();

            // Create nodes
            let a = db.create_node(&["Node"]);
            let b = db.create_node(&["Node"]);
            let c = db.create_node(&["Node"]);

            // Create edges
            let e1 = db.create_edge(a, b, "LINKS");
            let _e2 = db.create_edge(b, c, "LINKS");

            // Delete middle node and its edge
            db.delete_edge(e1);
            db.delete_node(b);

            // Set properties on remaining nodes
            db.set_node_property(a, "value", Value::Int64(1));
            db.set_node_property(c, "value", Value::Int64(3));

            db.close().unwrap();
            (a, b, c)
        };

        // Reopen and verify consistency
        {
            let db = ObrainDB::open(&db_path).unwrap();

            // Should have 2 nodes (a and c), b was deleted
            // Note: node_count includes deleted nodes in some implementations
            // What matters is that the non-deleted nodes are accessible
            let node_a = db.get_node(a_id);
            assert!(node_a.is_some());

            let node_c = db.get_node(c_id);
            assert!(node_c.is_some());

            // Middle node should be deleted
            let node_b = db.get_node(b_id);
            assert!(node_b.is_none());
        }
    }

    #[cfg(feature = "wal")]
    #[test]
    fn test_close_is_idempotent() {
        // Calling close() multiple times should not cause errors
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("close_test_db");

        let db = ObrainDB::open(&db_path).unwrap();
        db.create_node(&["Test"]);

        // First close should succeed
        assert!(db.close().is_ok());

        // Second close should also succeed (idempotent)
        assert!(db.close().is_ok());
    }

    #[test]
    fn test_with_store_external_backend() {
        let external = Arc::new(obrain_substrate::SubstrateStore::open_tempfile().unwrap())
            as Arc<dyn obrain_core::graph::GraphStoreMut>;

        // Seed data on the external store directly
        let n1 = external.create_node(&["Person"]);
        external.set_node_property(n1, "name", obrain_common::types::Value::from("Alix"));

        let db = ObrainDB::with_store(
            Arc::clone(&external) as Arc<dyn GraphStoreMut>,
            Config::in_memory(),
        )
        .unwrap();

        let session = db.session();

        // Session should see data from the external store via execute
        #[cfg(feature = "gql")]
        {
            let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
            assert_eq!(result.rows.len(), 1);
        }
    }

    #[test]
    fn test_with_config_custom_memory_limit() {
        let config = Config::in_memory().with_memory_limit(64 * 1024 * 1024); // 64 MB

        let db = ObrainDB::with_config(config).unwrap();
        assert_eq!(db.config().memory_limit, Some(64 * 1024 * 1024));
        assert_eq!(db.node_count(), 0);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn test_database_metrics_registry() {
        let db = ObrainDB::new_in_memory();

        // Perform some operations
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        // Check that metrics snapshot returns data
        let snap = db.metrics();
        // Session created counter should reflect at least 0 (metrics is initialized)
        assert_eq!(snap.query_count, 0); // No queries executed yet
    }

    #[test]
    fn test_query_result_has_metrics() {
        // Verifies that query results include execution metrics
        let db = ObrainDB::new_in_memory();
        db.create_node(&["Person"]);
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            let result = db.execute("MATCH (n:Person) RETURN n").unwrap();

            // Metrics should be populated
            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert!(result.execution_time_ms.unwrap() >= 0.0);
            assert_eq!(result.rows_scanned.unwrap(), 2);
        }
    }

    #[test]
    fn test_empty_query_result_metrics() {
        // Verifies metrics are correct for queries returning no results
        let db = ObrainDB::new_in_memory();
        db.create_node(&["Person"]);

        #[cfg(feature = "gql")]
        {
            // Query that matches nothing
            let result = db.execute("MATCH (n:NonExistent) RETURN n").unwrap();

            assert!(result.execution_time_ms.is_some());
            assert!(result.rows_scanned.is_some());
            assert_eq!(result.rows_scanned.unwrap(), 0);
        }
    }

    #[cfg(feature = "cdc")]
    mod cdc_integration {
        use super::*;

        #[test]
        fn test_node_lifecycle_history() {
            let db = ObrainDB::new_in_memory();

            // Create
            let id = db.create_node(&["Person"]);
            // Update
            db.set_node_property(id, "name", "Alix".into());
            db.set_node_property(id, "name", "Gus".into());
            // Delete
            db.delete_node(id);

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 4); // create + 2 updates + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert!(history[1].before.is_none()); // first set_node_property has no prior value
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Update);
            assert!(history[2].before.is_some()); // second update has prior "Alix"
            assert_eq!(history[3].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_edge_lifecycle_history() {
            let db = ObrainDB::new_in_memory();

            let alix = db.create_node(&["Person"]);
            let gus = db.create_node(&["Person"]);
            let edge = db.create_edge(alix, gus, "KNOWS");
            db.set_edge_property(edge, "since", 2024i64.into());
            db.delete_edge(edge);

            let history = db.history(edge).unwrap();
            assert_eq!(history.len(), 3); // create + update + delete
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            assert_eq!(history[1].kind, crate::cdc::ChangeKind::Update);
            assert_eq!(history[2].kind, crate::cdc::ChangeKind::Delete);
        }

        #[test]
        fn test_create_node_with_props_cdc() {
            let db = ObrainDB::new_in_memory();

            let id = db.create_node_with_props(
                &["Person"],
                vec![
                    ("name", obrain_common::types::Value::from("Alix")),
                    ("age", obrain_common::types::Value::from(30i64)),
                ],
            );

            let history = db.history(id).unwrap();
            assert_eq!(history.len(), 1);
            assert_eq!(history[0].kind, crate::cdc::ChangeKind::Create);
            // Props should be captured
            let after = history[0].after.as_ref().unwrap();
            assert_eq!(after.len(), 2);
        }

        #[test]
        fn test_changes_between() {
            let db = ObrainDB::new_in_memory();

            let id1 = db.create_node(&["A"]);
            let _id2 = db.create_node(&["B"]);
            db.set_node_property(id1, "x", 1i64.into());

            // All events should be at the same epoch (in-memory, epoch doesn't advance without tx)
            let changes = db
                .changes_between(
                    obrain_common::types::EpochId(0),
                    obrain_common::types::EpochId(u64::MAX),
                )
                .unwrap();
            assert_eq!(changes.len(), 3); // 2 creates + 1 update
        }
    }

    #[test]
    fn test_with_store_basic() {
        let store = Arc::new(obrain_substrate::SubstrateStore::open_tempfile().unwrap())
            as Arc<dyn obrain_core::graph::GraphStoreMut>;
        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", "Alix".into());

        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = ObrainDB::with_store(graph_store, Config::in_memory()).unwrap();

        let result = db.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_session() {
        let store = Arc::new(obrain_substrate::SubstrateStore::open_tempfile().unwrap())
            as Arc<dyn obrain_core::graph::GraphStoreMut>;
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = ObrainDB::with_store(graph_store, Config::in_memory()).unwrap();

        let session = db.session();
        let result = session.execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_with_store_mutations() {
        let store = Arc::new(obrain_substrate::SubstrateStore::open_tempfile().unwrap())
            as Arc<dyn obrain_core::graph::GraphStoreMut>;
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreMut>;
        let db = ObrainDB::with_store(graph_store, Config::in_memory()).unwrap();

        let mut session = db.session();

        // Use an explicit transaction so INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
        assert_eq!(result.rows.len(), 1);

        session.commit().unwrap();
    }
}
