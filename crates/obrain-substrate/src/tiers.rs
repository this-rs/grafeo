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
}
