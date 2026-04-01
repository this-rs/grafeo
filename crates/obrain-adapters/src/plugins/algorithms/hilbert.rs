//! Hilbert curve encoding for graph embeddings.
//!
//! Maps 2D normalized coordinates to 1D Hilbert curve indices, preserving
//! spatial locality better than row-major or Z-order curves. Provides both
//! the classic `xy↔d` conversion and a **multi-resolution encoding** that
//! captures structure at multiple scales.
//!
//! ## Multi-resolution encoding
//!
//! `hilbert_encode_point(point, levels)` encodes a 2D point `[0,1]²` into
//! `levels` dimensions, where each dimension `k` is the Hilbert index at
//! order `k+1` normalized to `[0,1]`. Coarse levels capture global position,
//! fine levels capture local neighborhood.
//!
//! ## Usage
//!
//! ```no_run
//! use obrain_adapters::plugins::algorithms::hilbert::{
//!     hilbert_xy2d, hilbert_d2xy, hilbert_encode_point,
//! };
//!
//! // Basic conversion
//! let d = hilbert_xy2d(4, 3, 5);
//! let (x, y) = hilbert_d2xy(4, d);
//! assert_eq!((x, y), (3, 5));
//!
//! // Multi-resolution encoding (8 levels → 8 dimensions)
//! let encoding = hilbert_encode_point([0.3, 0.7], 8);
//! assert_eq!(encoding.len(), 8);
//! ```

// ============================================================================
// Core Hilbert curve primitives
// ============================================================================

/// Convert (x, y) coordinates to a Hilbert curve index.
///
/// `order` is the curve order: the grid is `2^order × 2^order`.
///
/// # Panics
///
/// Debug-asserts if x or y ≥ 2^order.
#[allow(clippy::many_single_char_names)]
pub fn hilbert_xy2d(order: u32, x: u32, y: u32) -> u32 {
    let n = 1u32 << order;
    debug_assert!(x < n && y < n, "({x},{y}) out of range for order={order}");

    let mut d = 0u32;
    let mut x = x;
    let mut y = y;

    let mut s = n >> 1;
    while s > 0 {
        let rx = u32::from((x & s) > 0);
        let ry = u32::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &mut x, &mut y, rx, ry);
        s >>= 1;
    }

    d
}

/// Convert a Hilbert curve index `d` to (x, y) coordinates.
///
/// `order` is the curve order: the grid is `2^order × 2^order`.
///
/// # Panics
///
/// Debug-asserts if `d >= 4^order`.
#[allow(clippy::many_single_char_names)]
pub fn hilbert_d2xy(order: u32, d: u32) -> (u32, u32) {
    let n = 1u32 << order;
    debug_assert!(
        d < n * n,
        "d={d} out of range for order={order} (max={})",
        n * n - 1
    );

    let mut x = 0u32;
    let mut y = 0u32;
    let mut d = d;
    let mut s = 1u32;

    while s < n {
        let rx = u32::from((d & 2) != 0);
        let ry = u32::from((d & 1) != 0) ^ rx;
        rot(s, &mut x, &mut y, rx, ry);
        x += s * rx;
        y += s * ry;
        d >>= 2;
        s <<= 1;
    }

    (x, y)
}

/// Rotate/flip a quadrant (internal helper).
#[inline]
fn rot(n: u32, x: &mut u32, y: &mut u32, rx: u32, ry: u32) {
    if ry == 0 {
        if rx == 1 {
            *x = n.wrapping_sub(1).wrapping_sub(*x);
            *y = n.wrapping_sub(1).wrapping_sub(*y);
        }
        std::mem::swap(x, y);
    }
}

// ============================================================================
// Multi-resolution encoding
// ============================================================================

/// Encode a 2D point into a multi-resolution Hilbert vector.
///
/// Takes a point in `[0,1]²` and produces a `Vec<f32>` of length `levels`,
/// where each element `k` is the Hilbert index at order `k+1` normalized
/// to `[0,1]`. This captures structure at multiple scales:
///
/// - Level 0 (order 1): coarsest, 4 cells → global quadrant
/// - Level 7 (order 8): finest, 65536 cells → precise neighborhood
///
/// # Arguments
///
/// * `point` - 2D coordinates in `[0,1]²` (clamped if outside)
/// * `levels` - Number of resolution levels (typically 8)
///
/// # Returns
///
/// `Vec<f32>` of length `levels`, each value in `[0, 1]`.
pub fn hilbert_encode_point(point: [f32; 2], levels: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(levels);

    for k in 0..levels {
        let order = (k + 1) as u32;
        let side = (1u32 << order) as f32;
        let max_coord = (1u32 << order) - 1;
        let max_index = (1u32 << (2 * order)) - 1; // 4^order - 1

        let qx = ((point[0].clamp(0.0, 1.0) * side) as u32).min(max_coord);
        let qy = ((point[1].clamp(0.0, 1.0) * side) as u32).min(max_coord);

        let d = hilbert_xy2d(order, qx, qy);
        let normalized = if max_index > 0 {
            d as f32 / max_index as f32
        } else {
            0.0
        };
        result.push(normalized);
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_order2() {
        let order = 2;
        let n = 1u32 << order;
        for x in 0..n {
            for y in 0..n {
                let d = hilbert_xy2d(order, x, y);
                let (rx, ry) = hilbert_d2xy(order, d);
                assert_eq!((rx, ry), (x, y), "Roundtrip failed for ({x},{y})");
            }
        }
    }

    #[test]
    fn test_roundtrip_order4() {
        let order = 4;
        let n = 1u32 << order;
        for x in 0..n {
            for y in 0..n {
                let d = hilbert_xy2d(order, x, y);
                let (rx, ry) = hilbert_d2xy(order, d);
                assert_eq!((rx, ry), (x, y));
            }
        }
    }

    #[test]
    fn test_roundtrip_all_orders() {
        for order in 1..=6 {
            let n = 1u32 << order;
            for x in 0..n {
                for y in 0..n {
                    let d = hilbert_xy2d(order, x, y);
                    let (rx, ry) = hilbert_d2xy(order, d);
                    assert_eq!(
                        (rx, ry),
                        (x, y),
                        "Roundtrip failed: order={order}, ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_locality() {
        // Adjacent points on the Hilbert curve should be neighbors in 2D
        let order = 4;
        let n = (1u32 << order) * (1u32 << order);
        for d in 0..n - 1 {
            let (x1, y1) = hilbert_d2xy(order, d);
            let (x2, y2) = hilbert_d2xy(order, d + 1);
            let dist =
                (x1 as i32 - x2 as i32).unsigned_abs() + (y1 as i32 - y2 as i32).unsigned_abs();
            assert_eq!(dist, 1, "Adjacent d={d},{} not neighbors in 2D", d + 1);
        }
    }

    #[test]
    fn test_encode_point_levels() {
        let enc = hilbert_encode_point([0.3, 0.7], 8);
        assert_eq!(enc.len(), 8);
        for &v in &enc {
            assert!((0.0..=1.0).contains(&v), "Encoded value {v} not in [0,1]");
        }
    }

    #[test]
    fn test_encode_point_determinism() {
        let point = [0.42, 0.58];
        let enc1 = hilbert_encode_point(point, 8);
        for _ in 0..10 {
            let enc2 = hilbert_encode_point(point, 8);
            assert_eq!(enc1, enc2, "Non-deterministic encoding");
        }
    }

    #[test]
    fn test_encode_point_edge_cases() {
        // (0,0) - bottom-left corner
        let enc = hilbert_encode_point([0.0, 0.0], 8);
        assert_eq!(enc.len(), 8);
        assert_eq!(enc[0], 0.0); // order 1: (0,0) → d=0

        // (1,1) - top-right area
        let enc = hilbert_encode_point([1.0, 1.0], 8);
        assert_eq!(enc.len(), 8);
        for &v in &enc {
            assert!((0.0..=1.0).contains(&v));
        }

        // (0.5, 0.5) - center
        let enc = hilbert_encode_point([0.5, 0.5], 8);
        assert_eq!(enc.len(), 8);
        for &v in &enc {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_encode_point_locality_preserved() {
        // Two nearby points should have similar encodings
        let a = hilbert_encode_point([0.30, 0.70], 8);
        let b = hilbert_encode_point([0.31, 0.71], 8);
        let c = hilbert_encode_point([0.90, 0.10], 8); // far away

        let dist_ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let dist_ac: f32 = a.iter().zip(c.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        assert!(
            dist_ab < dist_ac,
            "Nearby points should be closer: dist_ab={dist_ab} >= dist_ac={dist_ac}"
        );
    }

    #[test]
    fn test_encode_point_clamping() {
        // Values outside [0,1] should be clamped
        let enc_neg = hilbert_encode_point([-0.5, -0.5], 4);
        let enc_zero = hilbert_encode_point([0.0, 0.0], 4);
        assert_eq!(enc_neg, enc_zero, "Negative values should clamp to 0");

        let enc_over = hilbert_encode_point([1.5, 1.5], 4);
        let enc_one = hilbert_encode_point([1.0, 1.0], 4);
        assert_eq!(enc_over, enc_one, "Values > 1 should clamp to 1");
    }
}
