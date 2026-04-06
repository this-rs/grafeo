//! # Kernel math primitives — Dense linear algebra, PRNG, activations
//!
//! Self-contained math module for the irreducible kernel C = Phi_0 . A^1(H^inf).
//! Zero external dependencies. Internal computations in f64, with conversion to f32
//! for storage as `Value::Vector(Arc<[f32]>)`.
//!
//! Ported from ai-noyau/src/math.rs with additions for obrain integration.

use std::fmt;

// ============================================================================
// PRNG -- Deterministic pseudo-random number generator (xorshift64)
// ============================================================================

/// Deterministic PRNG based on xorshift64.
///
/// Used for reproducible weight initialization (Xavier) and Fisher-Yates
/// neighbor sampling in the per-neighborhood kernel.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new PRNG with the given seed. Seed 0 is mapped to 1
    /// (xorshift requires non-zero state).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform in [-1, 1].
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }

    /// Approximate normal via Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Uniform integer in [0, n).
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher-Yates shuffle on a mutable slice.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// Matrix -- Dense row-major linear algebra
// ============================================================================

/// Dense row-major matrix of f64 values.
///
/// Core data structure for the kernel. All internal computations use f64
/// for numerical precision; conversion to f32 happens only at storage time
/// via [`Matrix::to_f32_vec`].
#[derive(Clone)]
pub struct Matrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Flat row-major data buffer.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a zero-initialized matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a matrix filled with `val`.
    pub fn filled(rows: usize, cols: usize, val: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![val; rows * cols],
        }
    }

    /// Create an identity matrix of size n x n.
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Xavier-initialized random matrix.
    pub fn randn(rows: usize, cols: usize, rng: &mut Rng) -> Self {
        let scale = (2.0 / (rows + cols) as f64).sqrt();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.next_normal() * scale)
            .collect();
        Self { rows, cols, data }
    }

    /// Create a matrix from a flat Vec, consuming it.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        Self { rows, cols, data }
    }

    /// Create a single-row matrix from a slice.
    pub fn from_row(row: &[f64]) -> Self {
        Self {
            rows: 1,
            cols: row.len(),
            data: row.to_vec(),
        }
    }

    // -- Indexing --

    /// Flat index for (row, col).
    #[inline]
    pub fn idx(&self, r: usize, c: usize) -> usize {
        debug_assert!(r < self.rows && c < self.cols);
        r * self.cols + c
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[self.idx(r, c)]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        let i = self.idx(r, c);
        self.data[i] = val;
    }

    /// Get row i as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get row i as a mutable slice.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    // -- Linear algebra --

    /// Matrix multiplication: C = self * other.
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul dimension mismatch: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.data[i * self.cols + k];
                if a_ik == 0.0 {
                    continue;
                }
                let row_start = i * other.cols;
                let other_row_start = k * other.cols;
                for j in 0..other.cols {
                    result.data[row_start + j] += a_ik * other.data[other_row_start + j];
                }
            }
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Scale all elements by `s`.
    pub fn scale(&self, s: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "add: row mismatch");
        assert_eq!(self.cols, other.cols, "add: col mismatch");
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    /// Frobenius norm: sqrt(sum of squares).
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Frobenius distance: ||self - other||_F.
    pub fn diff_norm(&self, other: &Matrix) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Apply a function element-wise.
    pub fn map(&self, f: fn(f64) -> f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Cosine similarity between row i and row j.
    pub fn cosine_similarity(&self, i: usize, j: usize) -> f64 {
        let row_i = self.row(i);
        let row_j = self.row(j);
        let mut dot = 0.0;
        let mut norm_i = 0.0;
        let mut norm_j = 0.0;
        for c in 0..self.cols {
            let a = row_i[c];
            let b = row_j[c];
            dot += a * b;
            norm_i += a * a;
            norm_j += b * b;
        }
        let denom = norm_i.sqrt() * norm_j.sqrt();
        if denom < 1e-12 { 0.0 } else { dot / denom }
    }

    /// Cosine similarity between a row and an external vector.
    pub fn cosine_similarity_vec(&self, row_idx: usize, vec: &[f64]) -> f64 {
        assert_eq!(self.cols, vec.len());
        let row = self.row(row_idx);
        let mut dot = 0.0;
        let mut norm_r = 0.0;
        let mut norm_v = 0.0;
        for c in 0..self.cols {
            dot += row[c] * vec[c];
            norm_r += row[c] * row[c];
            norm_v += vec[c] * vec[c];
        }
        let denom = norm_r.sqrt() * norm_v.sqrt();
        if denom < 1e-12 { 0.0 } else { dot / denom }
    }

    /// Extract a sub-matrix: rows [r_start..r_end), cols [c_start..c_end).
    pub fn slice(&self, r_start: usize, r_end: usize, c_start: usize, c_end: usize) -> Matrix {
        let rows = r_end - r_start;
        let cols = c_end - c_start;
        let mut result = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.set(i, j, self.get(r_start + i, c_start + j));
            }
        }
        result
    }

    /// Horizontal concatenation: [self | other].
    pub fn hcat(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "hcat: row mismatch");
        let mut result = Matrix::zeros(self.rows, self.cols + other.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j));
            }
            for j in 0..other.cols {
                result.set(i, self.cols + j, other.get(i, j));
            }
        }
        result
    }

    /// Set a specific row from a slice.
    pub fn set_row(&mut self, row_idx: usize, values: &[f64]) {
        assert_eq!(values.len(), self.cols);
        let start = row_idx * self.cols;
        self.data[start..start + self.cols].copy_from_slice(values);
    }

    // -- Conversion --

    /// Convert to f32 vector for storage as `Value::Vector(Arc<[f32]>)`.
    ///
    /// Returns a flat Vec<f32> in row-major order. For a single-row matrix
    /// (one node's embedding), this is directly the embedding vector.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x as f32).collect()
    }

    /// Convert a single row to f32 vector.
    pub fn row_to_f32(&self, row_idx: usize) -> Vec<f32> {
        self.row(row_idx).iter().map(|&x| x as f32).collect()
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let show_rows = self.rows.min(6);
        let show_cols = self.cols.min(8);
        for i in 0..show_rows {
            write!(f, "  [")?;
            for j in 0..show_cols {
                write!(f, " {:>8.4}", self.get(i, j))?;
            }
            if self.cols > show_cols {
                write!(f, "  ...")?;
            }
            writeln!(f, " ]")?;
        }
        if self.rows > show_rows {
            writeln!(f, "  ...")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix({}x{}, norm={:.4})",
            self.rows,
            self.cols,
            self.norm()
        )
    }
}

// ============================================================================
// Activation functions
// ============================================================================

/// Softmax per row -- numerically stable (subtract max before exp).
pub fn softmax_rows(m: &Matrix) -> Matrix {
    let mut result = Matrix::zeros(m.rows, m.cols);
    for i in 0..m.rows {
        let row = m.row(i);
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..m.cols {
            let e = (row[j] - max_val).exp();
            result.set(i, j, e);
            sum += e;
        }
        if sum > 0.0 {
            let row_out = result.row_mut(i);
            for v in row_out.iter_mut() {
                *v /= sum;
            }
        }
    }
    result
}

/// GELU activation (approximate, via tanh).
pub fn gelu(x: f64) -> f64 {
    let k = (2.0_f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

/// RMS normalization per row (no mean subtraction, unlike LayerNorm).
///
/// For each row: x_out = x / RMS(x), where RMS(x) = sqrt(mean(x^2) + eps).
/// Used by the kernel instead of LayerNorm -- simpler and sufficient for
/// fixed-point convergence.
pub fn rms_norm(m: &Matrix) -> Matrix {
    let eps = 1e-8;
    let mut result = Matrix::zeros(m.rows, m.cols);
    for i in 0..m.rows {
        let row = m.row(i);
        let mean_sq = row.iter().map(|x| x * x).sum::<f64>() / m.cols as f64;
        let rms = (mean_sq + eps).sqrt();
        let row_out = result.row_mut(i);
        for j in 0..m.cols {
            row_out[j] = row[j] / rms;
        }
    }
    result
}

/// Shannon entropy of an attention distribution (per row, averaged).
///
/// Input is assumed to be pre-softmax logits. Returns average entropy
/// across rows: H = -sum(p * ln(p)).
pub fn shannon_entropy(m: &Matrix) -> f64 {
    let sm = softmax_rows(m);
    let mut total = 0.0;
    for i in 0..sm.rows {
        let row = sm.row(i);
        for &p in row {
            if p > 1e-12 {
                total -= p * p.ln();
            }
        }
    }
    total / sm.rows as f64
}

// ============================================================================
// Metrics helpers (used by training and benchmarks)
// ============================================================================

/// Compute diversity of a matrix: mean pairwise cosine distance.
///
/// diversity = 1 - mean(|cosine(row_i, row_j)|) for i < j.
/// High diversity = rows point in different directions (collapse broken).
pub fn diversity(m: &Matrix) -> f64 {
    if m.rows < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..m.rows {
        for j in (i + 1)..m.rows {
            sum += m.cosine_similarity(i, j).abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        1.0 - sum / count as f64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Rng tests --

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_seed_zero_handled() {
        let mut r = Rng::new(0);
        // Should not be stuck at 0
        assert_ne!(r.next_u64(), 0);
    }

    #[test]
    fn test_rng_next_f64_range() {
        let mut r = Rng::new(123);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((-1.0..=1.0).contains(&v), "next_f64 out of range: {}", v);
        }
    }

    #[test]
    fn test_rng_normal_distribution() {
        let mut r = Rng::new(42);
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| r.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        // Mean should be near 0, variance near 1
        assert!(mean.abs() < 0.05, "normal mean too far from 0: {}", mean);
        assert!(
            (var - 1.0).abs() < 0.1,
            "normal variance too far from 1: {}",
            var
        );
    }

    #[test]
    fn test_rng_shuffle() {
        let mut r = Rng::new(42);
        let mut v: Vec<usize> = (0..20).collect();
        let original = v.clone();
        r.shuffle(&mut v);
        // Should be a permutation (same elements)
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
        // Should be different order (extremely unlikely to stay same with 20 elements)
        assert_ne!(v, original);
    }

    // -- Matrix construction tests --

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_matrix_eye() {
        let m = Matrix::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(m.get(i, j), 1.0);
                } else {
                    assert_eq!(m.get(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_matrix_from_vec() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_matrix_from_vec_bad_size() {
        Matrix::from_vec(2, 3, vec![1.0, 2.0]);
    }

    #[test]
    fn test_matrix_randn_xavier() {
        let mut rng = Rng::new(42);
        let m = Matrix::randn(100, 80, &mut rng);
        // Xavier: variance should be ~2/(100+80) = 0.0111
        let var = m.data.iter().map(|x| x * x).sum::<f64>() / m.data.len() as f64;
        let expected_var = 2.0 / (100.0 + 80.0);
        assert!(
            (var - expected_var).abs() < 0.005,
            "Xavier variance off: got {:.4}, expected {:.4}",
            var,
            expected_var
        );
    }

    // -- Matrix operations tests --

    #[test]
    fn test_matmul_identity() {
        let mut rng = Rng::new(42);
        let a = Matrix::randn(5, 4, &mut rng);
        let eye = Matrix::eye(4);
        let result = a.matmul(&eye);
        assert!(a.diff_norm(&result) < 1e-12, "A * I should equal A");
    }

    #[test]
    fn test_matmul_known_values() {
        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_matmul_dimension_mismatch() {
        let a = Matrix::zeros(3, 4);
        let b = Matrix::zeros(5, 2);
        a.matmul(&b);
    }

    #[test]
    fn test_matmul_80x80_performance() {
        let mut rng = Rng::new(42);
        let a = Matrix::randn(80, 80, &mut rng);
        let b = Matrix::randn(80, 80, &mut rng);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = a.matmul(&b);
        }
        let elapsed = start.elapsed();
        let per_mul = elapsed / 100;
        // Acceptance: 80x80 matmul < 1ms
        assert!(
            per_mul.as_millis() < 1,
            "80x80 matmul too slow: {:?}",
            per_mul
        );
    }

    #[test]
    fn test_transpose_roundtrip() {
        let mut rng = Rng::new(42);
        let a = Matrix::randn(5, 8, &mut rng);
        let att = a.transpose().transpose();
        assert!(a.diff_norm(&att) < 1e-12, "A^T^T should equal A");
    }

    #[test]
    fn test_transpose_dimensions() {
        let a = Matrix::zeros(3, 7);
        let at = a.transpose();
        assert_eq!(at.rows, 7);
        assert_eq!(at.cols, 3);
    }

    #[test]
    fn test_scale() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let scaled = m.scale(2.0);
        assert_eq!(scaled.get(0, 0), 2.0);
        assert_eq!(scaled.get(0, 1), 4.0);
        assert_eq!(scaled.get(0, 2), 6.0);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::from_vec(1, 3, vec![4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 5.0);
        assert_eq!(c.get(0, 1), 7.0);
        assert_eq!(c.get(0, 2), 9.0);
    }

    #[test]
    fn test_norm_and_diff_norm() {
        let a = Matrix::from_vec(1, 3, vec![3.0, 4.0, 0.0]);
        assert!((a.norm() - 5.0).abs() < 1e-12);

        let b = Matrix::from_vec(1, 3, vec![3.0, 4.0, 0.0]);
        assert!(a.diff_norm(&b) < 1e-12);

        let c = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]);
        assert!((a.diff_norm(&c) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_map() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 4.0, 9.0]);
        let sq = m.map(f64::sqrt);
        assert!((sq.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((sq.get(0, 1) - 2.0).abs() < 1e-12);
        assert!((sq.get(0, 2) - 3.0).abs() < 1e-12);
    }

    // -- Cosine similarity tests --

    #[test]
    fn test_cosine_self_similarity() {
        let mut rng = Rng::new(42);
        let m = Matrix::randn(5, 80, &mut rng);
        for i in 0..5 {
            let sim = m.cosine_similarity(i, i);
            assert!(
                (sim - 1.0).abs() < 1e-10,
                "cosine(v, v) should be 1.0, got {}",
                sim
            );
        }
    }

    #[test]
    fn test_cosine_orthogonal() {
        // Two orthogonal vectors
        let m = Matrix::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let sim = m.cosine_similarity(0, 1);
        assert!(sim.abs() < 1e-12, "orthogonal vectors should have cosine 0");
    }

    #[test]
    fn test_cosine_opposite() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        let sim = m.cosine_similarity(0, 1);
        assert!(
            (sim + 1.0).abs() < 1e-10,
            "opposite vectors should have cosine -1, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_zero_vector() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let sim = m.cosine_similarity(0, 1);
        assert_eq!(sim, 0.0, "zero vector cosine should be 0");
    }

    #[test]
    fn test_cosine_similarity_vec() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 0.0, 0.0]);
        let v = vec![1.0, 0.0, 0.0];
        assert!((m.cosine_similarity_vec(0, &v) - 1.0).abs() < 1e-12);
        let w = vec![0.0, 1.0, 0.0];
        assert!(m.cosine_similarity_vec(0, &w).abs() < 1e-12);
    }

    // -- Slice and hcat tests --

    #[test]
    fn test_slice() {
        let m = Matrix::from_vec(3, 4, (0..12).map(|x| x as f64).collect());
        let s = m.slice(1, 3, 1, 3);
        assert_eq!(s.rows, 2);
        assert_eq!(s.cols, 2);
        assert_eq!(s.get(0, 0), 5.0); // m[1][1]
        assert_eq!(s.get(0, 1), 6.0); // m[1][2]
        assert_eq!(s.get(1, 0), 9.0); // m[2][1]
        assert_eq!(s.get(1, 1), 10.0); // m[2][2]
    }

    #[test]
    fn test_hcat() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 1, vec![5.0, 6.0]);
        let c = a.hcat(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 3);
        assert_eq!(c.get(0, 2), 5.0);
        assert_eq!(c.get(1, 2), 6.0);
    }

    #[test]
    fn test_set_row() {
        let mut m = Matrix::zeros(3, 2);
        m.set_row(1, &[7.0, 8.0]);
        assert_eq!(m.get(1, 0), 7.0);
        assert_eq!(m.get(1, 1), 8.0);
        assert_eq!(m.get(0, 0), 0.0); // untouched
    }

    // -- Conversion tests --

    #[test]
    fn test_to_f32_vec() {
        let m = Matrix::from_vec(1, 3, vec![1.5, 2.5, 3.5]);
        let v = m.to_f32_vec();
        assert_eq!(v, vec![1.5f32, 2.5f32, 3.5f32]);
    }

    #[test]
    fn test_row_to_f32() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.row_to_f32(0), vec![1.0f32, 2.0f32]);
        assert_eq!(m.row_to_f32(1), vec![3.0f32, 4.0f32]);
    }

    #[test]
    fn test_f64_to_f32_precision() {
        // Ensure no catastrophic precision loss for typical embedding values
        let m = Matrix::from_vec(1, 4, vec![0.123456789, -0.987654321, 1e-6, 1e6]);
        let v = m.to_f32_vec();
        for (f64_val, f32_val) in m.data.iter().zip(v.iter()) {
            let rel_err = ((*f64_val as f32) - *f32_val).abs();
            assert!(
                rel_err < 1e-30,
                "f32 conversion error too large for {}",
                f64_val
            );
        }
    }

    // -- Activation function tests --

    #[test]
    fn test_softmax_rows_sum_one() {
        let m = Matrix::from_vec(3, 4, (0..12).map(|x| x as f64 * 0.5).collect());
        let sm = softmax_rows(&m);
        for i in 0..3 {
            let sum: f64 = sm.row(i).iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "softmax row {} sum = {}, expected 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_softmax_rows_positive() {
        let m = Matrix::from_vec(1, 4, vec![-100.0, -50.0, 0.0, 50.0]);
        let sm = softmax_rows(&m);
        for j in 0..4 {
            assert!(sm.get(0, j) >= 0.0, "softmax should be non-negative");
        }
    }

    #[test]
    fn test_softmax_rows_numerically_stable() {
        // Large values shouldn't cause overflow
        let m = Matrix::from_vec(1, 3, vec![1000.0, 1001.0, 1002.0]);
        let sm = softmax_rows(&m);
        let sum: f64 = sm.row(0).iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax unstable with large values: sum={}",
            sum
        );
        assert!(
            sm.data.iter().all(|x| x.is_finite()),
            "softmax produced non-finite"
        );
    }

    #[test]
    fn test_softmax_rows_uniform() {
        // Equal inputs should give uniform output
        let m = Matrix::from_vec(1, 5, vec![3.0; 5]);
        let sm = softmax_rows(&m);
        for j in 0..5 {
            assert!(
                (sm.get(0, j) - 0.2).abs() < 1e-10,
                "equal inputs should give uniform softmax"
            );
        }
    }

    #[test]
    fn test_gelu() {
        // GELU(0) = 0
        assert!((gelu(0.0)).abs() < 1e-12);
        // GELU(x) > 0 for x > 0
        assert!(gelu(1.0) > 0.0);
        // GELU(x) < 0 for small negative x (but > -0.17)
        assert!(gelu(-0.5) < 0.0);
        // GELU approaches x for large x
        assert!((gelu(5.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_unit_rms() {
        let m = Matrix::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0]);
        let normed = rms_norm(&m);
        for i in 0..2 {
            let row = normed.row(i);
            let mean_sq = row.iter().map(|x| x * x).sum::<f64>() / row.len() as f64;
            let rms = mean_sq.sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-6,
                "RMS norm row {} rms = {:.6}, expected ~1.0",
                i,
                rms
            );
        }
    }

    #[test]
    fn test_rms_norm_preserves_direction() {
        let m = Matrix::from_vec(1, 3, vec![2.0, 4.0, 6.0]);
        let normed = rms_norm(&m);
        // Ratios should be preserved: 1:2:3
        let r01 = normed.get(0, 0) / normed.get(0, 1);
        let r12 = normed.get(0, 1) / normed.get(0, 2);
        assert!((r01 - 0.5).abs() < 1e-10);
        assert!((r12 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms_norm_no_mean_subtraction() {
        // Unlike LayerNorm, RMS norm does NOT center the data
        let m = Matrix::from_vec(1, 4, vec![10.0, 10.0, 10.0, 10.0]);
        let normed = rms_norm(&m);
        // All values should be ~1.0 (not 0.0 as LayerNorm would give)
        for j in 0..4 {
            assert!(
                (normed.get(0, j) - 1.0).abs() < 1e-6,
                "RMS norm should not center: got {}",
                normed.get(0, j)
            );
        }
    }

    #[test]
    fn test_rms_norm_zero_vector() {
        let m = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]);
        let normed = rms_norm(&m);
        // Should not produce NaN (eps handles this)
        assert!(normed.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        // Equal logits → uniform distribution → max entropy
        let m = Matrix::from_vec(1, 4, vec![1.0; 4]);
        let h = shannon_entropy(&m);
        let expected = (4.0_f64).ln(); // ln(4) ≈ 1.386
        assert!(
            (h - expected).abs() < 1e-6,
            "uniform entropy should be ln(n)={:.4}, got {:.4}",
            expected,
            h
        );
    }

    #[test]
    fn test_shannon_entropy_peaked() {
        // One very large logit → near-zero entropy
        let m = Matrix::from_vec(1, 4, vec![100.0, 0.0, 0.0, 0.0]);
        let h = shannon_entropy(&m);
        assert!(
            h < 0.01,
            "peaked distribution should have near-zero entropy: {}",
            h
        );
    }

    // -- Diversity tests --

    #[test]
    fn test_diversity_identical_rows() {
        let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        let d = diversity(&m);
        assert!(
            d.abs() < 1e-10,
            "identical rows should have zero diversity: {}",
            d
        );
    }

    #[test]
    fn test_diversity_orthogonal_rows() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let d = diversity(&m);
        assert!(
            (d - 1.0).abs() < 1e-10,
            "orthogonal rows should have diversity 1.0: {}",
            d
        );
    }

    #[test]
    fn test_diversity_single_row() {
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        assert_eq!(diversity(&m), 0.0, "single row has no diversity");
    }

    // -- Debug/Display tests --

    #[test]
    fn test_matrix_debug_format() {
        let m = Matrix::zeros(3, 4);
        let debug = format!("{:?}", m);
        assert!(debug.contains("Matrix(3x4"));
    }
}
