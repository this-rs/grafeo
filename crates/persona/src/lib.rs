//! Persona — persistent identity, facts, reward, and GNN inference.

pub mod db;
pub mod reward;
pub mod facts;
pub mod fact_gnn;

pub use db::{PersonaDB, XiStats};
pub use reward::RewardDetector;
pub use facts::{PatternMatch, detect_facts_from_graph, detect_facts};
