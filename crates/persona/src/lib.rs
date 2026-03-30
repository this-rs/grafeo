//! Persona — persistent identity, facts, reward, GNN inference, and neural persistence.

pub mod bm25;
pub mod db;
pub mod fact_gnn;
pub mod facts;
pub mod formulas;
pub mod persist_net;
pub mod reward;

pub use bm25::{MessageHit, MessageIndex};
pub use db::{PersonaDB, SelfMetrics, XiStats};
pub use facts::{PatternMatch, detect_facts, detect_facts_from_graph};
pub use persist_net::PersistNet;
pub use reward::{RewardDetector, RewardSignals};
