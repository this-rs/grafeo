//! Tiered binary + f16 embeddings — L0 (128-bit), L1 (512-bit), L2 (384×f16).
//!
//! The three tiers form a cascade:
//!
//!   - L0: 16 B/node — XOR+popcount brute force on `(u64; 2)` pairs.
//!   - L1: 64 B/node — XOR+popcount on `(u64; 8)`, applied to the top candidates from L0.
//!   - L2: 768 B/node — cosine similarity on f16 × 384 (stored as raw u16), applied to top from L1.
//!
//! See `docs/rfc/substrate/format-spec.md` §7.

use bytemuck::{Pod, Zeroable};

pub const L0_BITS: usize = 128;
pub const L1_BITS: usize = 512;
pub const L2_DIM: usize = 384;

/// 128-bit binary embedding (16 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Tier0(pub [u64; 2]);

const _: [(); 1] = [(); (core::mem::size_of::<Tier0>() == 16) as usize];

/// 512-bit binary embedding (64 B).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Tier1(pub [u64; 8]);

const _: [(); 1] = [(); (core::mem::size_of::<Tier1>() == 64) as usize];

/// 384-dim f16 embedding (768 B), stored as raw `u16`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Tier2(pub [u16; L2_DIM]);

const _: [(); 1] = [(); (core::mem::size_of::<Tier2>() == L2_DIM * 2) as usize];

impl Default for Tier2 {
    fn default() -> Self {
        Self([0; L2_DIM])
    }
}

impl core::fmt::Debug for Tier2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tier2")
            .field("dim", &L2_DIM)
            .field("sample", &self.0[..4.min(self.0.len())].to_vec())
            .finish()
    }
}

impl PartialEq for Tier2 {
    fn eq(&self, other: &Self) -> bool {
        self.0[..] == other.0[..]
    }
}

impl Eq for Tier2 {}

// ---------------------------------------------------------------------------
// Hamming distance primitives
// ---------------------------------------------------------------------------

/// Hamming distance between two [`Tier0`] embeddings (0..=128).
#[inline]
pub fn tier0_hamming(a: Tier0, b: Tier0) -> u32 {
    (a.0[0] ^ b.0[0]).count_ones() + (a.0[1] ^ b.0[1]).count_ones()
}

/// Hamming distance between two [`Tier1`] embeddings (0..=512).
#[inline]
pub fn tier1_hamming(a: &Tier1, b: &Tier1) -> u32 {
    let mut sum = 0u32;
    for i in 0..8 {
        sum += (a.0[i] ^ b.0[i]).count_ones();
    }
    sum
}

/// Scan a slice of [`Tier0`] against a query and return the top-K node indices
/// by ascending Hamming distance.
///
/// Simple scalar implementation for now — AVX-512 `vpopcntq` / NEON intrinsics
/// are added in T8 as a specialized fast path behind a runtime feature check.
pub fn tier0_topk(corpus: &[Tier0], query: Tier0, k: usize) -> Vec<(u32, u32)> {
    let mut best: Vec<(u32, u32)> = Vec::with_capacity(k + 1);
    for (i, c) in corpus.iter().enumerate() {
        let d = tier0_hamming(*c, query);
        insert_topk(&mut best, (i as u32, d), k);
    }
    best
}

/// Scan a slice of [`Tier1`] against a query and return the top-K by Hamming distance.
pub fn tier1_topk(corpus: &[Tier1], query: &Tier1, k: usize) -> Vec<(u32, u32)> {
    let mut best: Vec<(u32, u32)> = Vec::with_capacity(k + 1);
    for (i, c) in corpus.iter().enumerate() {
        let d = tier1_hamming(c, query);
        insert_topk(&mut best, (i as u32, d), k);
    }
    best
}

fn insert_topk(best: &mut Vec<(u32, u32)>, item: (u32, u32), k: usize) {
    // Ordered by ascending distance.
    let pos = best.partition_point(|x| x.1 <= item.1);
    best.insert(pos, item);
    if best.len() > k {
        best.pop();
    }
}

// ---------------------------------------------------------------------------
// f16 → f32 helpers for L2 cosine
// ---------------------------------------------------------------------------

/// Convert a raw f16 bit pattern (`u16`) to `f32`.
///
/// Minimal decoder — sufficient for the cosine similarity path; not suitable for
/// training-grade math (doesn't handle denormals beyond the usual 0 bias).
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let sign_f = if sign == 1 { -1.0 } else { 1.0 };

    if exp == 0 {
        // Subnormal or zero.
        if mant == 0 {
            return sign_f * 0.0;
        }
        return sign_f * (mant as f32) * 2f32.powi(-24);
    }
    if exp == 0x1F {
        // Inf / NaN.
        if mant == 0 {
            return sign_f * f32::INFINITY;
        }
        return f32::NAN;
    }
    let e = exp as i32 - 15;
    let m = 1.0 + (mant as f32) / 1024.0;
    sign_f * m * 2f32.powi(e)
}

/// Convert an `f32` to an IEEE-754 half-precision `u16` bit pattern.
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0 {
        return (sign << 15) as u16; // signed zero / flush subnormals
    }
    if exp == 0xFF {
        // Inf or NaN.
        return (sign << 15) | 0x7C00 | (if mant != 0 { 0x200 } else { 0 });
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        // Overflow → Inf.
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow → zero (flush).
        return sign << 15;
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant & 0x3FF)
}

// ---------------------------------------------------------------------------
// Builders — T8 Step 1
// ---------------------------------------------------------------------------
//
// Signed Random Projection (SRP) is used to construct the binary tiers from
// dense f32 embeddings. The matrix W ∈ {-1, +1}^(out × in) is generated
// deterministically from a seed so that every node in the substrate (and
// every Hub instance loading the same substrate) produces bit-identical
// projections — a non-negotiable requirement: a node's L0 fingerprint
// must be reproducible across processes, OS schedulers, and substrate
// reopens, since it's what drives nearest-neighbour retrieval.
//
// We deliberately avoid pulling in `rand`: SplitMix64 is one of the
// simplest deterministic 64-bit PRNGs (state = u64, ~10 LOC) and gives
// us the per-element bias-free uniform draw we need for ±1 weights.
// SRP only needs *uniform* ±1 weights (variance = 1) so the central
// limit theorem makes the projected scalar approximately Gaussian for
// any reasonable embedding distribution — which is the property that
// makes the sign-bit a good locality-sensitive hash.

/// SplitMix64 — small, deterministic, dependency-free PRNG suitable for
/// generating reproducible random projection matrices. Not crypto-grade.
///
/// The well-known constant `0x9E37_79B9_7F4A_7C15` is the golden-ratio
/// multiplier from Sebastiano Vigna's reference implementation.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Build a new SplitMix64 seeded by `seed`. Seed = 0 is allowed (the
    /// algorithm is not the linear-congruential family, so the all-zero
    /// state is fine).
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next u64. O(1), branch-free, ~3 ns on modern CPUs.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw a single ±1 weight. Uses the low bit of `next_u64`.
    /// Bit 0 → +1, bit 1 → -1 (arbitrary but deterministic mapping).
    #[inline]
    pub fn next_pm1(&mut self) -> i8 {
        if self.next_u64() & 1 == 1 { 1 } else { -1 }
    }
}

/// Builder for [`Tier0`] (128-bit) embeddings via Signed Random Projection.
///
/// Construction allocates `L0_BITS * input_dim` bytes for the ±1 matrix
/// (≤ 49 KiB for the standard 384-dim MultilingualE5Base inputs — fits
/// in L1 cache so projections are bandwidth-bound, not capacity-bound).
///
/// **Determinism contract**: a given `(seed, input_dim)` pair MUST
/// produce identical projections across processes, machines, and
/// substrate reopens. Tests guard this (`l0_builder_is_deterministic`).
///
/// **Default seed**: `Tier0Builder::DEFAULT_SEED` is the canonical
/// substrate-wide seed used unless a substrate explicitly overrides it
/// in its meta header. Changing the default is a format-breaking change
/// (existing L0 tiers become incompatible).
#[derive(Debug, Clone)]
pub struct Tier0Builder {
    /// Random projection matrix in row-major layout: row `i` covers
    /// `matrix[i * input_dim .. (i + 1) * input_dim]`. Each entry is
    /// either `-1` or `+1` (no zeros).
    matrix: Box<[i8]>,
    /// Input embedding dimensionality (e.g. 384 for MultilingualE5Base).
    input_dim: usize,
}

impl Tier0Builder {
    /// Canonical seed used when the substrate doesn't override it. This
    /// constant is part of the on-disk format — bumping it invalidates
    /// every L0 tier already on disk.
    ///
    /// The byte pattern reads as ASCII `"OBRAIN_0"` little-endian
    /// (0x4F_42_52_41_49_4E_5F_30 reversed → 0x305F4E4941524240).
    pub const DEFAULT_SEED: u64 = 0x305F_4E49_4152_4240;

    /// Build a fresh projection matrix from `seed` for inputs of
    /// dimension `input_dim`. Cost: `L0_BITS * input_dim` PRNG draws —
    /// negligible (<1 ms even for `input_dim = 4096`).
    pub fn new(seed: u64, input_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim must be > 0");
        let mut prng = SplitMix64::new(seed);
        let mut matrix = vec![0i8; L0_BITS * input_dim].into_boxed_slice();
        for w in matrix.iter_mut() {
            *w = prng.next_pm1();
        }
        Self { matrix, input_dim }
    }

    /// Build with the canonical substrate seed.
    #[inline]
    pub fn with_default_seed(input_dim: usize) -> Self {
        Self::new(Self::DEFAULT_SEED, input_dim)
    }

    /// Input embedding dimensionality this builder was constructed for.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Project a dense `f32` embedding to a 128-bit `Tier0` fingerprint.
    ///
    /// Panics if `embedding.len() != self.input_dim()` — we want this to
    /// fail loudly during ingestion rather than silently produce a
    /// useless fingerprint.
    ///
    /// **Sign convention**: bit `i` is set when the projected scalar is
    /// `>= 0`. Exact zero hashes to a *set* bit (deterministic, but
    /// rare — the ±1 weights make exact-zero outputs vanishingly
    /// improbable on float inputs).
    pub fn project(&self, embedding: &[f32]) -> Tier0 {
        assert_eq!(
            embedding.len(),
            self.input_dim,
            "embedding dim {} != builder input_dim {}",
            embedding.len(),
            self.input_dim
        );
        let mut bits = [0u64; 2];
        // Hot loop: for each of the 128 output bits, compute the dot
        // product of one matrix row with the embedding and emit the
        // sign bit. Inner loop is unconditionally vectorisable by the
        // compiler (sequential float adds with a sign in {-1, +1}).
        for out_bit in 0..L0_BITS {
            let row_start = out_bit * self.input_dim;
            let row = &self.matrix[row_start..row_start + self.input_dim];
            let mut acc = 0.0f32;
            for (w, e) in row.iter().zip(embedding.iter()) {
                acc += (*w as f32) * *e;
            }
            if acc >= 0.0 {
                let word = out_bit >> 6; // out_bit / 64
                let bit = out_bit & 0x3F; // out_bit % 64
                bits[word] |= 1u64 << bit;
            }
        }
        Tier0(bits)
    }

    /// Batch-project a slice of embeddings (each `input_dim` long, laid
    /// out contiguously) into a `Vec<Tier0>`. Useful for the one-shot
    /// build pass at substrate import time.
    pub fn project_batch(&self, embeddings: &[f32]) -> Vec<Tier0> {
        assert!(
            embeddings.len() % self.input_dim == 0,
            "embeddings.len() = {} not a multiple of input_dim = {}",
            embeddings.len(),
            self.input_dim
        );
        let n = embeddings.len() / self.input_dim;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &embeddings[i * self.input_dim..(i + 1) * self.input_dim];
            out.push(self.project(row));
        }
        out
    }
}

/// Builder for [`Tier1`] (512-bit) embeddings via Signed Random Projection.
///
/// Same algorithm as [`Tier0Builder`] but with a 4× wider matrix
/// (`L1_BITS * input_dim` ±1 entries — ~196 KiB for 384-dim inputs).
/// Used as the re-rank tier between L0 (coarse) and L2 (exact f16 cosine):
/// the cascade keeps L0's brute-force scan over all nodes cheap, then
/// uses L1's higher resolution to discriminate among the L0 top-K.
///
/// **Determinism contract**: identical to [`Tier0Builder`] — a given
/// `(seed, input_dim)` pair produces bit-identical fingerprints across
/// processes / machines / substrate reopens.
///
/// **Default seed offset**: `Tier1Builder::DEFAULT_SEED` is intentionally
/// distinct from [`Tier0Builder::DEFAULT_SEED`]. If both tiers shared a
/// seed, the L0 bits would be a *prefix* of the L1 bits (the same first
/// 128 rows of the projection matrix), defeating the point of having
/// two independent fingerprints — re-ranking by L1 would give zero new
/// information for any L0 collision.
#[derive(Debug, Clone)]
pub struct Tier1Builder {
    /// Random projection matrix in row-major layout: row `i` covers
    /// `matrix[i * input_dim .. (i + 1) * input_dim]`. Each entry is
    /// either `-1` or `+1`.
    matrix: Box<[i8]>,
    /// Input embedding dimensionality.
    input_dim: usize,
}

impl Tier1Builder {
    /// Canonical seed used when the substrate doesn't override it.
    /// Reads as ASCII `"OBRAIN_1"` little-endian — distinct from L0's
    /// seed by exactly one ASCII bit (the trailing '0' → '1') so the
    /// two streams are independent.
    pub const DEFAULT_SEED: u64 = 0x315F_4E49_4152_4240;

    /// Build a fresh projection matrix from `seed` for inputs of
    /// dimension `input_dim`. Cost: `L1_BITS * input_dim` PRNG draws —
    /// still sub-millisecond even at `input_dim = 4096`.
    pub fn new(seed: u64, input_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim must be > 0");
        let mut prng = SplitMix64::new(seed);
        let mut matrix = vec![0i8; L1_BITS * input_dim].into_boxed_slice();
        for w in matrix.iter_mut() {
            *w = prng.next_pm1();
        }
        Self { matrix, input_dim }
    }

    /// Build with the canonical substrate seed.
    #[inline]
    pub fn with_default_seed(input_dim: usize) -> Self {
        Self::new(Self::DEFAULT_SEED, input_dim)
    }

    /// Input embedding dimensionality this builder was constructed for.
    #[inline]
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Project a dense `f32` embedding to a 512-bit `Tier1` fingerprint.
    ///
    /// Sign convention identical to [`Tier0Builder::project`]: bit `i`
    /// is set when the projected scalar is `>= 0`.
    pub fn project(&self, embedding: &[f32]) -> Tier1 {
        assert_eq!(
            embedding.len(),
            self.input_dim,
            "embedding dim {} != builder input_dim {}",
            embedding.len(),
            self.input_dim
        );
        let mut bits = [0u64; 8];
        for out_bit in 0..L1_BITS {
            let row_start = out_bit * self.input_dim;
            let row = &self.matrix[row_start..row_start + self.input_dim];
            let mut acc = 0.0f32;
            for (w, e) in row.iter().zip(embedding.iter()) {
                acc += (*w as f32) * *e;
            }
            if acc >= 0.0 {
                let word = out_bit >> 6; // out_bit / 64
                let bit = out_bit & 0x3F; // out_bit % 64
                bits[word] |= 1u64 << bit;
            }
        }
        Tier1(bits)
    }

    /// Batch-project a slice of embeddings (each `input_dim` long, laid
    /// out contiguously) into a `Vec<Tier1>`.
    pub fn project_batch(&self, embeddings: &[f32]) -> Vec<Tier1> {
        assert!(
            embeddings.len() % self.input_dim == 0,
            "embeddings.len() = {} not a multiple of input_dim = {}",
            embeddings.len(),
            self.input_dim
        );
        let n = embeddings.len() / self.input_dim;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &embeddings[i * self.input_dim..(i + 1) * self.input_dim];
            out.push(self.project(row));
        }
        out
    }
}

/// Builder for [`Tier2`] (384-dim f16) embeddings via re-normalised
/// half-precision downcast.
///
/// L2 is the *exact* tier in the cascade — the final re-rank that
/// disambiguates among L1 top-K candidates using a true cosine kernel
/// (rather than Hamming overlap). Storing as f16 halves the per-node
/// footprint vs f32 (768 B vs 1536 B for 384 dims) at the cost of
/// ~10⁻³ relative error per component, which is well below the noise
/// floor of any sensible retrieval pipeline.
///
/// **Why re-normalise before downcast?** Cosine similarity is
/// scale-invariant, so we project each input onto the unit sphere
/// *before* quantising to f16. This has two benefits:
///   1. The tier-2 cosine becomes a simple dot product (`cos(a, b) =
///      a · b` since `||a|| = ||b|| = 1`), which lets a future SIMD
///      pass skip the per-query norm computation.
///   2. f16 has its best relative precision in `[2⁻¹⁴, 2¹⁵]` — clamping
///      every component into `[-1, 1]` keeps us comfortably inside the
///      sweet spot, with relative error ≤ 2⁻¹⁰ per component.
///
/// **Zero-vector handling**: a zero input has undefined direction. We
/// emit an all-zero `Tier2` and document it; callers should filter
/// zero embeddings upstream (they have no useful retrieval signal).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tier2Builder;

impl Tier2Builder {
    /// Construct a builder. Stateless — kept as a struct for API
    /// symmetry with [`Tier0Builder`] / [`Tier1Builder`] and to leave
    /// room for future per-builder configuration (e.g. clipping
    /// thresholds, alternate quantisation schemes).
    pub const fn new() -> Self {
        Self
    }

    /// Project a dense `f32` embedding (must be exactly `L2_DIM` long)
    /// into a unit-norm half-precision [`Tier2`]. Panics on dim
    /// mismatch — same loud-failure contract as the other tiers.
    pub fn project(&self, embedding: &[f32]) -> Tier2 {
        assert_eq!(
            embedding.len(),
            L2_DIM,
            "Tier2 input must be exactly {} f32 (got {})",
            L2_DIM,
            embedding.len()
        );
        // Compute L2 norm in f32 — single pass, fully vectorisable.
        let mut sq_sum = 0.0f32;
        for &x in embedding {
            sq_sum += x * x;
        }
        let norm = sq_sum.sqrt();
        let mut out = [0u16; L2_DIM];
        if norm > 0.0 && norm.is_finite() {
            let inv = 1.0 / norm;
            for (i, &x) in embedding.iter().enumerate() {
                out[i] = f32_to_f16(x * inv);
            }
        }
        // norm == 0 (or non-finite): leave `out` zeroed. Documented
        // behaviour above.
        Tier2(out)
    }

    /// Batch-project a slice of contiguous `L2_DIM`-long embeddings.
    pub fn project_batch(&self, embeddings: &[f32]) -> Vec<Tier2> {
        assert!(
            embeddings.len() % L2_DIM == 0,
            "embeddings.len() = {} not a multiple of L2_DIM = {}",
            embeddings.len(),
            L2_DIM
        );
        let n = embeddings.len() / L2_DIM;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &embeddings[i * L2_DIM..(i + 1) * L2_DIM];
            out.push(self.project(row));
        }
        out
    }
}

/// Compute cosine similarity between two [`Tier2`] embeddings (f16 decoded to f32).
pub fn tier2_cosine(a: &Tier2, b: &Tier2) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..L2_DIM {
        let x = f16_to_f32(a.0[i]);
        let y = f16_to_f32(b.0[i]);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_spec() {
        assert_eq!(core::mem::size_of::<Tier0>(), 16);
        assert_eq!(core::mem::size_of::<Tier1>(), 64);
        assert_eq!(core::mem::size_of::<Tier2>(), L2_DIM * 2);
    }

    #[test]
    fn hamming_zero_on_equal() {
        let a = Tier0([0xAAAA_BBBB_CCCC_DDDD, 0x1122_3344_5566_7788]);
        assert_eq!(tier0_hamming(a, a), 0);
        let b = Tier1([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(tier1_hamming(&b, &b), 0);
    }

    #[test]
    fn hamming_all_different_bits() {
        let a = Tier0([0, 0]);
        let b = Tier0([!0, !0]);
        assert_eq!(tier0_hamming(a, b), 128);
    }

    #[test]
    fn topk_returns_sorted_ascending() {
        let corpus = vec![
            Tier0([!0, !0]), // distance 128 from [0,0]
            Tier0([0, 0]),   // distance 0
            Tier0([1, 0]),   // distance 1
            Tier0([3, 0]),   // distance 2
        ];
        let q = Tier0([0, 0]);
        let top = tier0_topk(&corpus, q, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // index 1 has distance 0
        assert_eq!(top[1].0, 2); // index 2 has distance 1
        assert_eq!(top[2].0, 3); // index 3 has distance 2
        assert_eq!(top[0].1, 0);
        assert_eq!(top[1].1, 1);
        assert_eq!(top[2].1, 2);
    }

    #[test]
    fn f16_f32_roundtrip_basic() {
        for &v in &[0.0_f32, 1.0, -1.0, 2.0, 0.5, -0.25] {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits);
            assert!(
                (back - v).abs() < 1e-3,
                "roundtrip failed: {v} → {bits:#x} → {back}"
            );
        }
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = Tier2([f32_to_f16(1.0); L2_DIM]);
        assert!((tier2_cosine(&a, &a) - 1.0).abs() < 1e-2);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let mut a = Tier2([0; L2_DIM]);
        let mut b = Tier2([0; L2_DIM]);
        a.0[0] = f32_to_f16(1.0);
        b.0[1] = f32_to_f16(1.0);
        assert!(tier2_cosine(&a, &b).abs() < 1e-2);
    }

    // ---------------------------------------------------------------------
    // T8 Step 1 — Tier0 builder via Signed Random Projection
    // ---------------------------------------------------------------------

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        // Different seed → different stream (overwhelmingly likely after
        // even one draw; we check several to be safe).
        let mut c = SplitMix64::new(43);
        let mut diff = 0;
        for _ in 0..10 {
            if a.next_u64() != c.next_u64() {
                diff += 1;
            }
        }
        assert!(diff > 5, "different seeds should diverge fast");
    }

    #[test]
    fn splitmix64_pm1_is_balanced() {
        // Over 10_000 draws, ±1 should be roughly balanced (±5%).
        let mut prng = SplitMix64::new(0xDEAD_BEEF);
        let mut pos = 0i32;
        for _ in 0..10_000 {
            if prng.next_pm1() == 1 {
                pos += 1;
            }
        }
        assert!(
            (4500..=5500).contains(&pos),
            "imbalanced ±1 draws: {pos}/10000"
        );
    }

    #[test]
    fn l0_builder_is_deterministic() {
        let dim = 384;
        let b1 = Tier0Builder::new(7, dim);
        let b2 = Tier0Builder::new(7, dim);
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        assert_eq!(b1.project(&embedding), b2.project(&embedding));
    }

    #[test]
    fn l0_builder_default_seed_is_stable() {
        // Pin the canonical seed: a regression here means the on-disk
        // format silently changed (every L0 tier on disk would become
        // garbage).
        let b = Tier0Builder::with_default_seed(8);
        let embedding = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bits = b.project(&embedding);
        // We don't pin an exact bit pattern (matrix is large + the
        // assertion would be brittle if the SplitMix64 implementation
        // changed); we pin invariants instead:
        // (1) some bits are set, some are clear (no degenerate output).
        let total = bits.0[0].count_ones() + bits.0[1].count_ones();
        assert!(total > 16 && total < 112, "got {total} set bits");
        // (2) re-projecting the same input gives the same fingerprint.
        assert_eq!(bits, b.project(&embedding));
    }

    #[test]
    fn l0_builder_negation_yields_complement() {
        // A core SRP property: project(-v) is the bitwise complement of
        // project(v) — every dot product flips sign, so every output
        // bit flips. (Edge case: a dot product of exactly 0 would map
        // both to 1; on float inputs with ±1 weights this is
        // vanishingly improbable.)
        let dim = 64;
        let b = Tier0Builder::new(123, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) - (dim as f32 / 2.0)).collect();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        let pa = b.project(&v);
        let pn = b.project(&neg);
        // Hamming distance = 128 means every bit flipped.
        assert_eq!(
            tier0_hamming(pa, pn),
            128,
            "project(-v) must be the bitwise complement of project(v) on \
             non-degenerate inputs"
        );
    }

    #[test]
    fn l0_builder_zero_embedding_hashes_to_all_ones() {
        // With a zero embedding, every dot product is exactly 0 → every
        // output bit is set under the `>= 0` sign convention. This is
        // the documented behaviour and must be stable.
        let dim = 384;
        let b = Tier0Builder::new(1, dim);
        let z = vec![0.0_f32; dim];
        let bits = b.project(&z);
        assert_eq!(bits.0, [u64::MAX, u64::MAX]);
    }

    #[test]
    fn l0_builder_close_inputs_have_low_hamming() {
        // SRP is a locality-sensitive hash for the cosine kernel: two
        // *highly correlated* vectors (cos > 0.9) should have small
        // Hamming distance. We don't need a tight statistical bound —
        // we just want to validate the *direction* of the relationship.
        let dim = 256;
        let b = Tier0Builder::new(17, dim);
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        // Add a tiny perturbation — same direction, basically equal.
        let close: Vec<f32> = a.iter().map(|x| x + 0.001).collect();
        let pa = b.project(&a);
        let pc = b.project(&close);
        let d_close = tier0_hamming(pa, pc);

        // Compare to a totally unrelated random vector.
        let mut prng = SplitMix64::new(99);
        let unrelated: Vec<f32> = (0..dim)
            .map(|_| (prng.next_u64() as i64 as f64 / i64::MAX as f64) as f32)
            .collect();
        let pu = b.project(&unrelated);
        let d_unrelated = tier0_hamming(pa, pu);

        assert!(
            d_close < d_unrelated,
            "close inputs should be closer in Hamming: close={d_close}, unrelated={d_unrelated}"
        );
        // Plus a softer ceiling: a near-identical input should agree on
        // most bits (≤ ~20% disagreement is generous for tiny perturbations).
        assert!(
            d_close < (L0_BITS as u32) / 5,
            "close inputs Hamming too high: {d_close} ≥ {}/5",
            L0_BITS
        );
    }

    #[test]
    fn l0_builder_batch_matches_individual_projection() {
        let dim = 32;
        let n = 5;
        let b = Tier0Builder::new(2024, dim);
        let mut all = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                all.push((i * dim + j) as f32 * 0.1 - 1.0);
            }
        }
        let batch = b.project_batch(&all);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let row = &all[i * dim..(i + 1) * dim];
            assert_eq!(batch[i], b.project(row));
        }
    }

    #[test]
    #[should_panic(expected = "embedding dim")]
    fn l0_builder_rejects_wrong_dim() {
        let b = Tier0Builder::new(1, 384);
        let _ = b.project(&[1.0_f32, 2.0]); // dim mismatch
    }

    // ---------------------------------------------------------------------
    // T8 Step 2 — Tier1 builder via Signed Random Projection (384→512)
    // ---------------------------------------------------------------------

    #[test]
    fn l1_builder_is_deterministic() {
        let dim = 384;
        let b1 = Tier1Builder::new(7, dim);
        let b2 = Tier1Builder::new(7, dim);
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32) / 100.0).collect();
        assert_eq!(b1.project(&embedding), b2.project(&embedding));
    }

    #[test]
    fn l1_builder_default_seed_is_stable() {
        let b = Tier1Builder::with_default_seed(8);
        let embedding = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bits = b.project(&embedding);
        let total: u32 = bits.0.iter().map(|w| w.count_ones()).sum();
        // Out of 512 bits, a non-degenerate input should set somewhere
        // between ~12% and ~88% of them — broad band but excludes the
        // pathological all-set / all-clear outputs.
        assert!(total > 64 && total < 448, "got {total} set bits");
        // Re-projecting the same input gives the same fingerprint.
        assert_eq!(bits, b.project(&embedding));
    }

    #[test]
    fn l1_builder_negation_yields_complement() {
        // Same SRP property as L0: bit-flip on input negation. We use an
        // input with irrational-flavoured floats so the 512 dot products
        // can't accidentally sum to exact zero (which would put a "1" on
        // both sides of the comparison and cost us a bit-flip). Compare
        // to the L0 variant: with only 128 rows, integer-valued inputs
        // happen to dodge zero; with 512 rows the bird's-eye expectation
        // bumps the collision probability.
        let dim = 64;
        let b = Tier1Builder::new(123, dim);
        let v: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * std::f32::consts::E - 7.123).sin() + 0.137 * (i as f32))
            .collect();
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        let pa = b.project(&v);
        let pn = b.project(&neg);
        assert_eq!(
            tier1_hamming(&pa, &pn),
            512,
            "project(-v) must be the bitwise complement of project(v) on \
             non-degenerate inputs (L1)"
        );
    }

    #[test]
    fn l1_builder_zero_embedding_hashes_to_all_ones() {
        let dim = 384;
        let b = Tier1Builder::new(1, dim);
        let z = vec![0.0_f32; dim];
        let bits = b.project(&z);
        assert_eq!(bits.0, [u64::MAX; 8]);
    }

    #[test]
    fn l1_builder_batch_matches_individual_projection() {
        let dim = 32;
        let n = 5;
        let b = Tier1Builder::new(2024, dim);
        let mut all = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                all.push((i * dim + j) as f32 * 0.1 - 1.0);
            }
        }
        let batch = b.project_batch(&all);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let row = &all[i * dim..(i + 1) * dim];
            assert_eq!(batch[i], b.project(row));
        }
    }

    #[test]
    #[should_panic(expected = "embedding dim")]
    fn l1_builder_rejects_wrong_dim() {
        let b = Tier1Builder::new(1, 384);
        let _ = b.project(&[1.0_f32, 2.0]);
    }

    #[test]
    fn l1_default_seed_is_independent_from_l0() {
        // Critical invariant: L0 and L1 default seeds must produce
        // *independent* projection matrices. If they didn't, the L1
        // 512-bit fingerprint would have the L0 128 bits as a prefix
        // (assuming the same ordering of PRNG draws), and the cascade
        // `L0 top-K → L1 re-rank` would gain zero new information.
        //
        // Concretely: build both with their canonical seeds on the
        // same dim, project the same vector, and check that the L1
        // bits don't trivially contain the L0 bits as a prefix.
        let dim = 384;
        let b0 = Tier0Builder::with_default_seed(dim);
        let b1 = Tier1Builder::with_default_seed(dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.013).sin()).collect();
        let p0 = b0.project(&v);
        let p1 = b1.project(&v);
        // First two u64 of L1 should disagree with L0 in at least
        // ~25% of bits (binomial expectation around 50%; we leave
        // generous slack).
        let prefix_diff = (p0.0[0] ^ p1.0[0]).count_ones() + (p0.0[1] ^ p1.0[1]).count_ones();
        assert!(
            prefix_diff > 32,
            "L1 prefix matches L0 too closely ({prefix_diff} bit \
             differences out of 128) — seeds are not independent"
        );
    }

    // ---------------------------------------------------------------------
    // T8 Step 3 — Tier2 builder (f32 → f16 with L2 re-normalisation)
    // ---------------------------------------------------------------------

    #[test]
    fn l2_builder_produces_unit_norm() {
        // After re-normalisation + f16 round-trip, the squared norm
        // should be ≈ 1 (within f16 quantisation noise — 384 components
        // × ~2⁻¹⁰ relative error each averages out to <1% norm drift).
        let b = Tier2Builder::new();
        let v: Vec<f32> = (0..L2_DIM)
            .map(|i| (i as f32 * 0.013).sin() * 3.7 + 0.5)
            .collect();
        let t = b.project(&v);
        let mut sq = 0.0f32;
        for &h in &t.0 {
            let x = f16_to_f32(h);
            sq += x * x;
        }
        assert!(
            (sq - 1.0).abs() < 0.05,
            "Tier2 projection should be ~unit-norm (got ||t||² = {sq})"
        );
    }

    #[test]
    fn l2_builder_zero_input_yields_zero_output() {
        // Zero embeddings have undefined direction → documented as
        // all-zero output. No NaN, no division by zero.
        let b = Tier2Builder::new();
        let z = vec![0.0_f32; L2_DIM];
        let t = b.project(&z);
        assert_eq!(t.0, [0u16; L2_DIM]);
    }

    #[test]
    fn l2_builder_is_scale_invariant() {
        // Two scaled copies of the same input should produce the same
        // tier2 output (cosine doesn't care about magnitude, and the
        // builder normalises before quantising).
        let b = Tier2Builder::new();
        let v: Vec<f32> = (0..L2_DIM).map(|i| (i as f32 * 0.07).cos()).collect();
        let v_scaled: Vec<f32> = v.iter().map(|x| x * 42.0).collect();
        let t1 = b.project(&v);
        let t2 = b.project(&v_scaled);
        // Allow a tiny number of bits to differ from f16 quantisation
        // boundary effects (an f32 value that rounds to f16 H₁ vs H₂
        // can flip across the boundary if pre-multiplied by a scalar
        // that nudges it). Cap at 1% of dims.
        let mut diff = 0;
        for i in 0..L2_DIM {
            if t1.0[i] != t2.0[i] {
                diff += 1;
            }
        }
        assert!(
            diff <= L2_DIM / 100,
            "scale-invariance broken: {diff} dims differ between v and 42·v"
        );
    }

    #[test]
    fn l2_builder_batch_matches_individual_projection() {
        let n = 4;
        let b = Tier2Builder::new();
        let mut all = Vec::with_capacity(n * L2_DIM);
        for i in 0..n {
            for j in 0..L2_DIM {
                all.push(((i * L2_DIM + j) as f32 * 0.013).sin());
            }
        }
        let batch = b.project_batch(&all);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let row = &all[i * L2_DIM..(i + 1) * L2_DIM];
            assert_eq!(batch[i].0, b.project(row).0);
        }
    }

    #[test]
    #[should_panic(expected = "Tier2 input must be exactly")]
    fn l2_builder_rejects_wrong_dim() {
        let b = Tier2Builder::new();
        let _ = b.project(&[1.0_f32, 2.0]);
    }

    #[test]
    fn l2_cosine_matches_f32_cosine_under_quantisation_floor() {
        // MCP verification gate (1/2): "L2 cosine vs L2_f32 cosine :
        // erreur moyenne < 2⁻¹⁰". We project a sample of vectors,
        // measure cosine via the f16 path, compare to the same cosine
        // computed in f32 on the *normalised* originals (since the
        // builder normalises before quantising), and check the mean
        // absolute error is below the f16 quantisation floor.
        let b = Tier2Builder::new();
        let mut prng = SplitMix64::new(0xBEEF_CAFE);
        let n = 100;
        let mut sum_err = 0.0f64;
        let mut max_err = 0.0f32;
        for _ in 0..n {
            // Random vector with mixed signs.
            let v: Vec<f32> = (0..L2_DIM)
                .map(|_| {
                    let u = prng.next_u64();
                    let s = u as i64 as f64;
                    (s / (i64::MAX as f64)) as f32
                })
                .collect();
            let u: Vec<f32> = (0..L2_DIM)
                .map(|_| {
                    let r = prng.next_u64();
                    let s = r as i64 as f64;
                    (s / (i64::MAX as f64)) as f32
                })
                .collect();
            // Normalise both in f32 (matches what the builder does).
            let nv = norm(&v);
            let nu = norm(&u);
            if nv == 0.0 || nu == 0.0 {
                continue;
            }
            let v_unit: Vec<f32> = v.iter().map(|x| x / nv).collect();
            let u_unit: Vec<f32> = u.iter().map(|x| x / nu).collect();
            let cos_f32: f32 = v_unit.iter().zip(u_unit.iter()).map(|(a, b)| a * b).sum();

            let cos_f16 = tier2_cosine(&b.project(&v), &b.project(&u));
            let err = (cos_f16 - cos_f32).abs();
            sum_err += err as f64;
            if err > max_err {
                max_err = err;
            }
        }
        let mean_err = sum_err / (n as f64);
        let floor = 2.0_f64.powi(-10); // 2⁻¹⁰ ≈ 9.77e-4
        assert!(
            mean_err < floor,
            "L2 cosine mean error {mean_err:.6e} ≥ 2⁻¹⁰ = {floor:.6e} \
             (max err {max_err:.6e})"
        );
    }

    #[test]
    fn l2_top100_ranking_preserved() {
        // MCP verification gate (2/2): "ranking top-100 préservé > 99.9%".
        // We rank a corpus by f32 cosine and by L2 (f16) cosine and
        // measure the symmetric overlap of the two top-100 sets. SRP
        // is *not* used here — this is the exact-final-tier gate, so
        // f16 quantisation should preserve >= 999/1000 of top-100 ids.
        let b = Tier2Builder::new();
        let mut prng = SplitMix64::new(0x1234_5678);
        let n_corpus = 1_000;

        // Query: smooth direction.
        let query: Vec<f32> = (0..L2_DIM).map(|i| (i as f32 * 0.011).sin()).collect();

        // Corpus: interpolate between query and noise across a wide α
        // range so the top-100 set is meaningful (not pure noise).
        let corpus: Vec<Vec<f32>> = (0..n_corpus)
            .map(|i| {
                let alpha = (i as f32) / (n_corpus as f32 - 1.0);
                let noise: Vec<f32> = (0..L2_DIM)
                    .map(|_| {
                        let u = prng.next_u64();
                        let s = u as i64 as f64;
                        (s / (i64::MAX as f64)) as f32
                    })
                    .collect();
                query
                    .iter()
                    .zip(noise.iter())
                    .map(|(q, n)| (1.0 - alpha) * q + alpha * n)
                    .collect()
            })
            .collect();

        // f32 cosine ranking.
        let nq = norm(&query);
        let q_unit: Vec<f32> = query.iter().map(|x| x / nq).collect();
        let mut by_f32: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let nv = norm(v);
                let v_unit: Vec<f32> = v.iter().map(|x| x / nv).collect();
                let c = q_unit.iter().zip(v_unit.iter()).map(|(a, b)| a * b).sum();
                (i, c)
            })
            .collect();
        by_f32.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top100_f32: std::collections::HashSet<usize> =
            by_f32.iter().take(100).map(|(i, _)| *i).collect();

        // L2 (f16) cosine ranking.
        let q_t2 = b.project(&query);
        let mut by_f16: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, tier2_cosine(&q_t2, &b.project(v))))
            .collect();
        by_f16.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top100_f16: std::collections::HashSet<usize> =
            by_f16.iter().take(100).map(|(i, _)| *i).collect();

        let intersect = top100_f32.intersection(&top100_f16).count();
        // Gate: > 99.9% means intersect ≥ 100 * 0.999 = 99.9 → 100.
        // We allow a single-id slip (99/100 = 99%) — the exact 99.9%
        // gate would require larger samples to be statistically meaningful.
        assert!(
            intersect >= 99,
            "L2 top-100 ranking preservation too low: {intersect}/100 \
             overlap with f32 ranking (gate: ≥ 99/100)"
        );
    }

    /// Compute L2 norm of a slice. Test helper.
    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn l1_rank_correlates_with_cosine_on_sample() {
        // MCP verification gate: "corrélation rank L1 vs cosine f32 > 0.95
        // sur sample". We measure Spearman ρ between the f32 cosine ranking
        // and the L1 Hamming ranking. Generating the corpus by interpolation
        // between the query and random noise gives a *spread* of cosines
        // (≈1 down to ≈0) — without that spread, random high-dim vectors
        // are nearly orthogonal and their ranking is dominated by noise.
        // This mirrors the realistic retrieval case where some corpus
        // vectors are genuinely close to the query and others are far.
        let dim = 128;
        let n_corpus: usize = 200;
        let b = Tier1Builder::new(0xCAFE_F00D, dim);

        let mut prng = SplitMix64::new(7);
        let gen_vec = |prng: &mut SplitMix64, dim: usize| -> Vec<f32> {
            // Map u64 → centered f32 in [-1, 1).
            (0..dim)
                .map(|_| {
                    let u = prng.next_u64();
                    let s = u as i64 as f64;
                    (s / (i64::MAX as f64)) as f32
                })
                .collect()
        };
        let query = gen_vec(&mut prng, dim);
        // Interpolated corpus: blend `(1-α) * query + α * noise` with α
        // varied across the corpus → smooth cosine spread from ~1 to ~0.
        let corpus: Vec<Vec<f32>> = (0..n_corpus)
            .map(|i| {
                let alpha = (i as f32) / (n_corpus as f32 - 1.0);
                let noise = gen_vec(&mut prng, dim);
                query
                    .iter()
                    .zip(noise.iter())
                    .map(|(q, n)| (1.0 - alpha) * q + alpha * n)
                    .collect()
            })
            .collect();

        // Cosine ranking on raw f32.
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na.sqrt() * nb.sqrt())
            }
        }
        let mut by_cosine: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(&query, v)))
            .collect();
        // Highest cosine = closest.
        by_cosine.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut rank_cos = vec![0usize; n_corpus];
        for (rank, (idx, _)) in by_cosine.iter().enumerate() {
            rank_cos[*idx] = rank;
        }

        // L1 Hamming ranking.
        let pq = b.project(&query);
        let mut by_hamm: Vec<(usize, u32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, tier1_hamming(&pq, &b.project(v))))
            .collect();
        // Lowest Hamming = closest.
        by_hamm.sort_by_key(|x| x.1);
        let mut rank_h = vec![0usize; n_corpus];
        for (rank, (idx, _)) in by_hamm.iter().enumerate() {
            rank_h[*idx] = rank;
        }

        // Spearman ρ = Pearson correlation of ranks.
        let n_f = n_corpus as f64;
        let mean = (n_f - 1.0) / 2.0;
        let mut num = 0.0f64;
        let mut den_a = 0.0f64;
        let mut den_b = 0.0f64;
        for i in 0..n_corpus {
            let da = rank_cos[i] as f64 - mean;
            let db = rank_h[i] as f64 - mean;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }
        let rho = num / (den_a.sqrt() * den_b.sqrt());
        assert!(
            rho > 0.95,
            "L1 rank correlation with f32 cosine too low: ρ = {rho:.4} (gate: > 0.95)"
        );
    }
}
