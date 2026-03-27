//! SpectralEncoder — converts an engram's node ensemble into a compact spectral
//! signature for similarity search.
//!
//! The spectral signature is a fixed-dimensional float vector that captures the
//! "shape" of a node group. It is used with [`VectorIndex`] for nearest-neighbor
//! retrieval during recall and warm-up selection.
//!
//! # Extensibility
//! Users can provide a custom encoder by implementing [`SpectralEncoderTrait`]
//! and passing it to [`SpectralEncoder::with_encoder`].

use obrain_common::types::NodeId;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Trait — pluggable encoding strategy
// ---------------------------------------------------------------------------

/// Trait for converting a weighted node ensemble into a spectral signature.
///
/// Implementations must be `Send + Sync` so the encoder can be shared across
/// threads (e.g., inside an `Arc<SpectralEncoder>`).
pub trait SpectralEncoderTrait: Send + Sync + std::fmt::Debug {
    /// Encode a slice of (NodeId, weight) pairs into a fixed-length vector.
    ///
    /// The output vector should be L2-normalized.
    fn encode(&self, nodes: &[(NodeId, f64)]) -> Vec<f64>;

    /// The number of dimensions in the output vector.
    fn dimensions(&self) -> usize;
}

// ---------------------------------------------------------------------------
// DefaultSpectralEncoder — hash-based projection
// ---------------------------------------------------------------------------

/// Default spectral encoder that projects node IDs into a fixed-dimensional
/// space using a deterministic hashing scheme.
///
/// Each node is hashed to determine which dimensions it contributes to, and
/// its contribution is weighted by its ensemble weight. The final vector is
/// L2-normalized.
#[derive(Debug, Clone)]
pub struct DefaultSpectralEncoder {
    dims: usize,
}

impl DefaultSpectralEncoder {
    /// Creates a new default encoder with the given number of dimensions.
    #[allow(dead_code)]
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Default for DefaultSpectralEncoder {
    fn default() -> Self {
        Self { dims: 64 }
    }
}

impl SpectralEncoderTrait for DefaultSpectralEncoder {
    fn encode(&self, nodes: &[(NodeId, f64)]) -> Vec<f64> {
        let mut signature = vec![0.0f64; self.dims];

        if nodes.is_empty() {
            return signature;
        }

        for &(node_id, weight) in nodes {
            // Hash the NodeId to get a deterministic spread across dimensions.
            let mut hasher = DefaultHasher::new();
            node_id.0.hash(&mut hasher);
            let hash = hasher.finish();

            // Use different parts of the hash to select multiple dimensions,
            // giving each node a richer fingerprint.
            for sub in 0..4u64 {
                let mut sub_hasher = DefaultHasher::new();
                (hash, sub).hash(&mut sub_hasher);
                let sub_hash = sub_hasher.finish();

                let dim = (sub_hash as usize) % self.dims;

                // Alternate sign based on another bit to allow cancellation,
                // producing a more expressive projection.
                let sign = if (sub_hash >> 32) & 1 == 0 { 1.0 } else { -1.0 };

                signature[dim] += sign * weight;
            }
        }

        // L2-normalize the signature.
        l2_normalize(&mut signature);

        signature
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

// ---------------------------------------------------------------------------
// SpectralEncoder — wrapper with dynamic dispatch
// ---------------------------------------------------------------------------

/// A spectral encoder that delegates to a `SpectralEncoderTrait` implementation.
///
/// By default it uses `DefaultSpectralEncoder` with 64 dimensions. Users can
/// swap in a custom encoder via [`SpectralEncoder::with_encoder`].
#[derive(Debug)]
pub struct SpectralEncoder {
    inner: Box<dyn SpectralEncoderTrait>,
}

impl SpectralEncoder {
    /// Creates a new `SpectralEncoder` with the default hash-based strategy (64 dims).
    pub fn new() -> Self {
        Self {
            inner: Box::new(DefaultSpectralEncoder::default()),
        }
    }

    /// Creates a `SpectralEncoder` backed by a custom encoder.
    pub fn with_encoder(encoder: Box<dyn SpectralEncoderTrait>) -> Self {
        Self { inner: encoder }
    }

    /// Encode a weighted node ensemble into a spectral signature vector.
    pub fn encode(&self, nodes: &[(NodeId, f64)]) -> Vec<f64> {
        self.inner.encode(nodes)
    }

    /// Returns the dimensionality of the output vectors.
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}

impl Default for SpectralEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// L2-normalize a vector in place. If the norm is zero the vector is left unchanged.
fn l2_normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_encoder_produces_correct_dimensions() {
        let enc = SpectralEncoder::new();
        assert_eq!(enc.dimensions(), 64);

        let sig = enc.encode(&[(NodeId(1), 1.0), (NodeId(2), 0.5)]);
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn output_is_l2_normalized() {
        let enc = SpectralEncoder::new();
        let sig = enc.encode(&[(NodeId(42), 1.0), (NodeId(99), 0.7), (NodeId(3), 0.3)]);

        let norm: f64 = sig.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "expected unit norm, got {}",
            norm
        );
    }

    #[test]
    fn empty_input_returns_zero_vector() {
        let enc = SpectralEncoder::new();
        let sig = enc.encode(&[]);
        assert!(sig.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn deterministic_encoding() {
        let enc = SpectralEncoder::new();
        let nodes = vec![(NodeId(10), 1.0), (NodeId(20), 0.5)];

        let sig1 = enc.encode(&nodes);
        let sig2 = enc.encode(&nodes);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn different_nodes_produce_different_signatures() {
        let enc = SpectralEncoder::new();
        let sig_a = enc.encode(&[(NodeId(1), 1.0)]);
        let sig_b = enc.encode(&[(NodeId(999), 1.0)]);
        assert_ne!(sig_a, sig_b);
    }

    #[test]
    fn custom_encoder_works() {
        #[derive(Debug)]
        struct TinyEncoder;
        impl SpectralEncoderTrait for TinyEncoder {
            fn encode(&self, nodes: &[(NodeId, f64)]) -> Vec<f64> {
                vec![nodes.len() as f64]
            }
            fn dimensions(&self) -> usize {
                1
            }
        }

        let enc = SpectralEncoder::with_encoder(Box::new(TinyEncoder));
        assert_eq!(enc.dimensions(), 1);
        assert_eq!(enc.encode(&[(NodeId(1), 1.0), (NodeId(2), 0.5)]), vec![2.0]);
    }

    #[test]
    fn custom_dimensions() {
        let enc = SpectralEncoder::with_encoder(Box::new(DefaultSpectralEncoder::new(128)));
        assert_eq!(enc.dimensions(), 128);

        let sig = enc.encode(&[(NodeId(1), 1.0)]);
        assert_eq!(sig.len(), 128);
    }
}
