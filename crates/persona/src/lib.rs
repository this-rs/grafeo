//! Persona — persistent identity, facts, reward, GNN inference, and neural persistence.

pub mod db;
pub mod reward;
pub mod facts;
pub mod fact_gnn;
pub mod persist_net;

pub use db::{PersonaDB, XiStats};
pub use reward::RewardDetector;
pub use facts::{PatternMatch, detect_facts_from_graph, detect_facts};
pub use persist_net::PersistNet;
