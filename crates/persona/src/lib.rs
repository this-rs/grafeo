//! Persona — persistent identity, facts, reward, GNN inference, and neural persistence.

pub mod db;
pub mod bm25;
pub mod reward;
pub mod facts;
pub mod fact_gnn;
pub mod persist_net;

pub use db::{PersonaDB, XiStats};
pub use bm25::{MessageIndex, MessageHit};
pub use reward::{RewardDetector, RewardSignals};
pub use facts::{PatternMatch, detect_facts_from_graph, detect_facts};
pub use persist_net::PersistNet;
