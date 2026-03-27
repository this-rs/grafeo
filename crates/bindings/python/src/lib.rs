//! Use Obrain from Python with native Rust performance.
//!
//! You get full access to the graph database through a Pythonic API - same
//! query speed, same durability, with the convenience of Python's ecosystem.
//!
//! ## Quick Start
//!
//! ```python
//! from obrain import ObrainDB
//!
//! # Create an in-memory database (or pass a path for persistence)
//! db = ObrainDB()
//!
//! # Create some people
//! db.execute("INSERT (:Person {name: 'Alix', role: 'Engineer'})")
//! db.execute("INSERT (:Person {name: 'Gus', role: 'Manager'})")
//! db.execute("""
//!     MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
//!     INSERT (a)-[:REPORTS_TO]->(b)
//! """)
//!
//! # Query the graph
//! result = db.execute("MATCH (p:Person)-[:REPORTS_TO]->(m) RETURN p.name, m.name")
//! for row in result:
//!     print(f"{row['p.name']} reports to {row['m.name']}")
//! ```
//!
//! ## Data Science Integration
//!
//! | Library | How to use | Best for |
//! | ------- | ---------- | -------- |
//! | pandas | `result.to_pandas()` or `db.nodes_df()` | Tabular operations |
//! | polars | `result.to_polars()` | Fast columnar analytics |
//! | NetworkX | `db.as_networkx().to_networkx()` | Graph visualization, analysis |
//! | solvOR | `db.as_solvor()` | Operations research algorithms |

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use pyo3::prelude::*;

mod bridges;
#[cfg(feature = "cognitive")]
mod cognitive;
mod database;
mod error;
mod graph;
mod quantization;
mod query;
mod types;

#[cfg(feature = "algos")]
use bridges::{PyAlgorithms, PyNetworkXAdapter, PySolvORAdapter};
#[cfg(feature = "cognitive")]
use cognitive::{PyCognitiveEngine, PyCognitiveSearch, PyGDS};
use database::{AsyncQueryResult, AsyncQueryResultIter, PyObrainDB};
use graph::{PyEdge, PyNode};
use query::PyQueryResult;
use types::PyValue;

/// Returns the active SIMD instruction set for vector operations.
///
/// Useful for debugging and verifying that SIMD acceleration is being used.
///
/// Returns one of: "avx2", "sse", "neon", or "scalar"
///
/// Example:
///     import obrain
///     print(f"SIMD support: {obrain.simd_support()}")  # e.g., "avx2"
#[pyfunction]
fn simd_support() -> &'static str {
    obrain_core::index::vector::simd_support()
}

/// Obrain Python module.
#[pymodule]
fn obrain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObrainDB>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<AsyncQueryResult>()?;
    m.add_class::<AsyncQueryResultIter>()?;
    m.add_class::<PyValue>()?;
    #[cfg(feature = "algos")]
    {
        m.add_class::<PyAlgorithms>()?;
        m.add_class::<PyNetworkXAdapter>()?;
        m.add_class::<PySolvORAdapter>()?;
    }

    #[cfg(feature = "cognitive")]
    {
        m.add_class::<PyCognitiveEngine>()?;
        m.add_class::<PyCognitiveSearch>()?;
        m.add_class::<PyGDS>()?;
    }

    // Register quantization types
    quantization::register(m)?;

    // Add module-level functions
    m.add_function(wrap_pyfunction!(simd_support, m)?)?;
    m.add_function(wrap_pyfunction!(types::vector, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
