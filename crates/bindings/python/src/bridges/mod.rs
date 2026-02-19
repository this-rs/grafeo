//! Connect Grafeo to Python's graph analysis ecosystem.
//!
//! | Adapter | Python library | When to use |
//! | ------- | -------------- | ----------- |
//! | [`PyNetworkXAdapter`] | [NetworkX](https://networkx.org/) | Visualization, graph algorithms |
//! | [`PySolvORAdapter`] | [solvOR](https://pypi.org/project/solvor/) | Operations Research problems |
//! | [`PyAlgorithms`] | (native) | Best performance, no dependencies |

#[cfg(feature = "algos")]
pub mod algorithms;
#[cfg(feature = "algos")]
pub mod networkx;
#[cfg(feature = "algos")]
pub mod solvor;

#[cfg(feature = "algos")]
pub use algorithms::PyAlgorithms;
#[cfg(feature = "algos")]
pub use networkx::PyNetworkXAdapter;
#[cfg(feature = "algos")]
pub use solvor::PySolvORAdapter;
