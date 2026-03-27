//! Caching utilities for Obrain.
//!
//! This module provides cache implementations optimized for graph database workloads:
//!
//! - [`SecondChanceLru`] - LRU cache with lock-free access marking

mod second_chance;

pub use second_chance::SecondChanceLru;
