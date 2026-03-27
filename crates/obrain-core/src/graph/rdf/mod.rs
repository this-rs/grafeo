//! RDF (Resource Description Framework) Graph Model.
//!
//! This module implements an RDF triple store following the W3C RDF 1.1 specification.
//!
//! RDF represents data as triples (Subject, Predicate, Object) forming a directed graph
//! where edges are predicates connecting subject and object nodes.
//!
//! # Key Concepts
//!
//! - **IRI**: Internationalized Resource Identifier, uniquely identifies resources
//! - **Blank Node**: Anonymous node within a graph
//! - **Literal**: Data value (string, number, date, etc.) with optional datatype/language
//! - **Triple**: The fundamental unit (subject, predicate, object)
//! - **Named Graph**: A set of triples identified by an IRI
//!
//! # Example
//!
//! ```
//! use grafeo_core::graph::rdf::{RdfStore, Term, Triple};
//!
//! let store = RdfStore::new();
//!
//! // Add a triple: <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix"
//! store.insert(Triple::new(
//!     Term::iri("http://example.org/alix"),
//!     Term::iri("http://xmlns.com/foaf/0.1/name"),
//!     Term::literal("Alix"),
//! ));
//!
//! // Query triples with a specific subject
//! let subject = Term::iri("http://example.org/alix");
//! for triple in store.triples_with_subject(&subject) {
//!     println!("{:?}", triple);
//! }
//! ```

mod store;
mod term;
mod triple;

pub use store::{BulkLoadResult, NTriplesError, RdfStore, RdfStoreConfig};
pub use term::{BlankNode, Iri, Literal, Term};
pub use triple::{Triple, TriplePattern};
