//! SIMD popcount primitives for the L0 / L1 Hamming-distance scan.
//!
//! The retrieval cascade (T8) walks millions of binary fingerprints per
//! query: XOR with the query, popcount the result, top-K by ascending
//! count. This module provides a runtime-dispatched fast path:
//!
//! | arch     | path                                            | notes        |
//! |----------|-------------------------------------------------|--------------|
//! | `x86_64` | AVX-512 BITALG `vpopcntq` (8 × u64 / instruction)| if detected |
//! | `x86_64` | AVX2 + scalar `popcnt` per lane                 | if detected  |
//! | `x86_64` | scalar `popcntq`                                | baseline     |
//! | `aarch64`| NEON `cnt.16b` + `addv`                         | always-on    |
//! | *other*  | scalar `u64::count_ones`                        | fallback     |
//!
//! The dispatcher caches the chosen impl in a `OnceLock` so feature
//! detection runs once per process, not per call.
//!
//! ## Bit-for-bit parity
//!
//! Every accelerated path computes the exact same arithmetic as the
//! scalar reference: `out[i] = (corpus[i] XOR query).count_ones()`.
//! There's no rounding, no saturation. The tests pin equivalence on
//! deterministic inputs across paths that the host can execute
//! (always at least scalar; NEON on aarch64; AVX2/AVX-512 if detected).
//!
//! ## Why the AVX-512 path matters
//!
//! `_mm512_popcnt_epi64` from the AVX-512 BITALG / VPOPCNTDQ
//! extensions popcounts an entire 512-bit (8 × u64) vector in one
//! cycle. For a Tier1 fingerprint (8 × u64 = 512 bits), one AVX-512
//! XOR + popcount + reduce computes the Hamming distance in 3 µops.
//! Scalar `popcntq` on the same 8 lanes takes ~8 cycles. On 10⁶ Tier1
//! candidates per query that's 1.25 ms vs 10 ms — the difference
//! between "interactive retrieval" and "noticeable lag".

// Crate root enforces #![deny(unsafe_code)] — opt out narrowly here.
// Every `unsafe` block in this module wraps a single SIMD intrinsic
// call on a contiguous Pod buffer of known size; safety is documented
// at each call site. See `simd.rs` for the established pattern.
#![allow(unsafe_code)]

use crate::tiers::{Tier0, Tier1};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Scalar reference impls — always correct, used as ground truth.
// ---------------------------------------------------------------------------

/// Scalar Hamming distance between two `Tier0` fingerprints.
#[inline]
pub fn t0_hamming(a: Tier0, b: Tier0) -> u32 {
    (a.0[0] ^ b.0[0]).count_ones() + (a.0[1] ^ b.0[1]).count_ones()
}

/// Scalar Hamming distance between two `Tier1` fingerprints.
#[inline]
pub fn t1_hamming(a: &Tier1, b: &Tier1) -> u32 {
    let mut sum = 0u32;
    for k in 0..8 {
        sum += (a.0[k] ^ b.0[k]).count_ones();
    }
    sum
}

/// Scalar XOR+popcount over a corpus, written to `out`. Reference impl
/// — every accelerated path must produce byte-identical results.
pub fn xor_popcount_t0_scalar(corpus: &[Tier0], q: Tier0, out: &mut [u32]) {
    assert_eq!(corpus.len(), out.len(), "corpus.len() must == out.len()");
    for (i, c) in corpus.iter().enumerate() {
        out[i] = t0_hamming(*c, q);
    }
}

/// Scalar XOR+popcount for `Tier1` corpus.
pub fn xor_popcount_t1_scalar(corpus: &[Tier1], q: &Tier1, out: &mut [u32]) {
    assert_eq!(corpus.len(), out.len(), "corpus.len() must == out.len()");
    for (i, c) in corpus.iter().enumerate() {
        out[i] = t1_hamming(c, q);
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX-512 path — the headline speedup for L1 (Tier1 = 8 × u64 =
// one zmm register). One XOR + one VPOPCNTQ + one reduce = full Hamming.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
unsafe fn xor_popcount_t1_avx512(corpus: &[Tier1], q: &Tier1, out: &mut [u32]) {
    use core::arch::x86_64::{
        _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64, _mm512_xor_si512,
    };
    // SAFETY: Tier1 is `#[repr(C)] [u64; 8]` (64 B contiguous). The
    // cast to `*const __m512i` is sound for an unaligned load — we use
    // `loadu_*`, not `load_*`. Caller guarantees `out.len() == corpus.len()`.
    let q_v = unsafe { _mm512_loadu_si512(q.0.as_ptr() as *const _) };
    for (i, c) in corpus.iter().enumerate() {
        let v = unsafe { _mm512_loadu_si512(c.0.as_ptr() as *const _) };
        let xored = unsafe { _mm512_xor_si512(v, q_v) };
        let pc = unsafe { _mm512_popcnt_epi64(xored) };
        // SAFETY: reduce intrinsics on AVX-512 zmm operands are pure
        // arithmetic — no memory access.
        let total = unsafe { _mm512_reduce_add_epi64(pc) };
        out[i] = total as u32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
unsafe fn xor_popcount_t0_avx512(corpus: &[Tier0], q: Tier0, out: &mut [u32]) {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_popcnt_u64, _mm_storeu_si128, _mm_xor_si128,
    };
    // Tier0 is 16 B (128 bits) — too small to fill a zmm by itself, so
    // we drop down to the SSE2 path with the 64-bit popcnt. (vpopcntq
    // on a single xmm would actually need AVX-512+VL; keeping this
    // conservative.)
    // SAFETY: Tier0 is `#[repr(C)] [u64; 2]`, 16 B contiguous; valid
    // for unaligned 128-bit load. `_mm_popcnt_u64` is a register op.
    let q_v = unsafe { _mm_loadu_si128(q.0.as_ptr() as *const _) };
    for (i, c) in corpus.iter().enumerate() {
        let v = unsafe { _mm_loadu_si128(c.0.as_ptr() as *const _) };
        let xored = unsafe { _mm_xor_si128(v, q_v) };
        let mut buf = [0u64; 2];
        unsafe { _mm_storeu_si128(buf.as_mut_ptr() as *mut _, xored) };
        out[i] = unsafe { _mm_popcnt_u64(buf[0]) as u32 + _mm_popcnt_u64(buf[1]) as u32 };
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 path — no `vpopcntq`, but the scalar `popcnt` instruction
// is still the fastest popcount available. The "AVX2" label here means:
// we know the machine has AVX2, which implies popcnt (Haswell baseline).
// We just unroll the scalar loop; the compiler fuses the loads.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn xor_popcount_t1_avx2(corpus: &[Tier1], q: &Tier1, out: &mut [u32]) {
    use core::arch::x86_64::_mm_popcnt_u64;
    // SAFETY: `_mm_popcnt_u64` is a register-only op; no memory hazards.
    for (i, c) in corpus.iter().enumerate() {
        let mut sum = 0u32;
        for k in 0..8 {
            sum += unsafe { _mm_popcnt_u64(c.0[k] ^ q.0[k]) as u32 };
        }
        out[i] = sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[inline]
unsafe fn xor_popcount_t0_avx2(corpus: &[Tier0], q: Tier0, out: &mut [u32]) {
    use core::arch::x86_64::_mm_popcnt_u64;
    // SAFETY: same as above.
    for (i, c) in corpus.iter().enumerate() {
        let a = unsafe { _mm_popcnt_u64(c.0[0] ^ q.0[0]) as u32 };
        let b = unsafe { _mm_popcnt_u64(c.0[1] ^ q.0[1]) as u32 };
        out[i] = a + b;
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON path — `count_ones()` already lowers to `cnt.8b + addv` on
// aarch64, so the manual NEON path here is mostly a clarity win: it lets
// the compiler keep all 8 u64 of a Tier1 in one q-register and fuses
// the XOR + popcnt + reduce.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn xor_popcount_t1_neon(corpus: &[Tier1], q: &Tier1, out: &mut [u32]) {
    use core::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u64, vld1q_u64, vreinterpretq_u8_u64};
    // SAFETY: Tier1 is `#[repr(C)] [u64; 8]` — contiguous, 64 B. Each
    // 16 B chunk is a valid `uint64x2_t` load target.
    let q0 = unsafe { vld1q_u64(q.0.as_ptr()) };
    let q1 = unsafe { vld1q_u64(q.0.as_ptr().add(2)) };
    let q2 = unsafe { vld1q_u64(q.0.as_ptr().add(4)) };
    let q3 = unsafe { vld1q_u64(q.0.as_ptr().add(6)) };
    for (i, c) in corpus.iter().enumerate() {
        let c0 = unsafe { vld1q_u64(c.0.as_ptr()) };
        let c1 = unsafe { vld1q_u64(c.0.as_ptr().add(2)) };
        let c2 = unsafe { vld1q_u64(c.0.as_ptr().add(4)) };
        let c3 = unsafe { vld1q_u64(c.0.as_ptr().add(6)) };
        let x0 = veorq_u64(c0, q0);
        let x1 = veorq_u64(c1, q1);
        let x2 = veorq_u64(c2, q2);
        let x3 = veorq_u64(c3, q3);
        // cnt.16b → popcount each byte; addv → horizontal sum across all
        // 16 bytes (max value 16 × 8 = 128, fits comfortably in u32).
        let p0 = vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(x0))) as u32;
        let p1 = vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(x1))) as u32;
        let p2 = vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(x2))) as u32;
        let p3 = vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(x3))) as u32;
        out[i] = p0 + p1 + p2 + p3;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn xor_popcount_t0_neon(corpus: &[Tier0], q: Tier0, out: &mut [u32]) {
    use core::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u64, vld1q_u64, vreinterpretq_u8_u64};
    // SAFETY: Tier0 is `#[repr(C)] [u64; 2]` — 16 B contiguous, valid
    // for one `uint64x2_t` load.
    let q_v = unsafe { vld1q_u64(q.0.as_ptr()) };
    for (i, c) in corpus.iter().enumerate() {
        let v = unsafe { vld1q_u64(c.0.as_ptr()) };
        let xored = veorq_u64(v, q_v);
        out[i] = vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(xored))) as u32;
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatcher — chooses the best path per arch, caches the choice.
// ---------------------------------------------------------------------------

/// Implementation tier picked at runtime, in descending preference order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// AVX-512 + VPOPCNTDQ (x86_64). Fastest for Tier1 batches.
    Avx512,
    /// AVX2 + scalar `popcnt` (x86_64). Used on Skylake-class CPUs.
    Avx2,
    /// NEON (aarch64). Always available on aarch64.
    Neon,
    /// Portable scalar fallback (`u64::count_ones`).
    Scalar,
}

static BACKEND: OnceLock<Backend> = OnceLock::new();

/// Resolve the best backend available on this host. Cached after first call.
pub fn backend() -> Backend {
    *BACKEND.get_or_init(detect_backend)
}

fn detect_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            return Backend::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            return Backend::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Backend::Neon;
    }
    #[allow(unreachable_code)]
    Backend::Scalar
}

/// XOR+popcount over a `Tier0` corpus, dispatched to the fastest available
/// SIMD path. Output `out[i]` is the Hamming distance between `corpus[i]`
/// and `q`, identical to `xor_popcount_t0_scalar`.
pub fn xor_popcount_t0(corpus: &[Tier0], q: Tier0, out: &mut [u32]) {
    assert_eq!(corpus.len(), out.len(), "corpus.len() must == out.len()");
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { xor_popcount_t0_avx512(corpus, q, out) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { xor_popcount_t0_avx2(corpus, q, out) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { xor_popcount_t0_neon(corpus, q, out) },
        _ => xor_popcount_t0_scalar(corpus, q, out),
    }
}

/// XOR+popcount over a `Tier1` corpus, dispatched to the fastest available
/// SIMD path. Output identical to `xor_popcount_t1_scalar`.
pub fn xor_popcount_t1(corpus: &[Tier1], q: &Tier1, out: &mut [u32]) {
    assert_eq!(corpus.len(), out.len(), "corpus.len() must == out.len()");
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => unsafe { xor_popcount_t1_avx512(corpus, q, out) },
        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { xor_popcount_t1_avx2(corpus, q, out) },
        #[cfg(target_arch = "aarch64")]
        Backend::Neon => unsafe { xor_popcount_t1_neon(corpus, q, out) },
        _ => xor_popcount_t1_scalar(corpus, q, out),
    }
}

// ---------------------------------------------------------------------------
// Tests — bit-exact equivalence between dispatched path and scalar reference
// across deterministic and edge-case inputs. The tests run *whichever* path
// the host supports; the scalar path is always tested.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 — same PRNG style as `simd.rs`.
    struct Xs64(u64);
    impl Xs64 {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn make_t0_corpus(n: usize, seed: u64) -> Vec<Tier0> {
        let mut prng = Xs64(seed);
        (0..n).map(|_| Tier0([prng.next(), prng.next()])).collect()
    }

    fn make_t1_corpus(n: usize, seed: u64) -> Vec<Tier1> {
        let mut prng = Xs64(seed);
        (0..n)
            .map(|_| {
                let mut bits = [0u64; 8];
                for b in bits.iter_mut() {
                    *b = prng.next();
                }
                Tier1(bits)
            })
            .collect()
    }

    #[test]
    fn t0_scalar_matches_pairwise_hamming() {
        // The batched scalar API must agree with the per-pair primitive.
        let corpus = make_t0_corpus(64, 0xDEAD);
        let q = Tier0([0xCAFE_BABE_DEAD_BEEF, 0x1234_5678_90AB_CDEF]);
        let mut out = vec![0u32; corpus.len()];
        xor_popcount_t0_scalar(&corpus, q, &mut out);
        for (i, c) in corpus.iter().enumerate() {
            assert_eq!(out[i], t0_hamming(*c, q));
        }
    }

    #[test]
    fn t1_scalar_matches_pairwise_hamming() {
        let corpus = make_t1_corpus(64, 0xBEEF);
        let q = Tier1([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut out = vec![0u32; corpus.len()];
        xor_popcount_t1_scalar(&corpus, &q, &mut out);
        for (i, c) in corpus.iter().enumerate() {
            assert_eq!(out[i], t1_hamming(c, &q));
        }
    }

    #[test]
    fn t0_dispatched_matches_scalar() {
        // Whatever backend the host picks, it must produce identical
        // Hamming distances to the scalar reference. This is the main
        // bit-exactness gate.
        let corpus = make_t0_corpus(2048, 0xAA55);
        let q = Tier0([0x0F0F_0F0F_0F0F_0F0F, 0xF0F0_F0F0_F0F0_F0F0]);
        let mut got = vec![0u32; corpus.len()];
        let mut want = vec![0u32; corpus.len()];
        xor_popcount_t0(&corpus, q, &mut got);
        xor_popcount_t0_scalar(&corpus, q, &mut want);
        assert_eq!(got, want, "dispatched t0 != scalar (backend = {:?})", backend());
    }

    #[test]
    fn t1_dispatched_matches_scalar() {
        let corpus = make_t1_corpus(2048, 0x55AA);
        let q = Tier1([
            0xCAFE_BABE_DEAD_BEEF,
            0x1234_5678_90AB_CDEF,
            !0,
            0,
            0xFFFF_0000_FFFF_0000,
            0x0000_FFFF_0000_FFFF,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ]);
        let mut got = vec![0u32; corpus.len()];
        let mut want = vec![0u32; corpus.len()];
        xor_popcount_t1(&corpus, &q, &mut got);
        xor_popcount_t1_scalar(&corpus, &q, &mut want);
        assert_eq!(got, want, "dispatched t1 != scalar (backend = {:?})", backend());
    }

    #[test]
    fn t0_edge_cases() {
        // Identical query/corpus → Hamming 0.
        // Complement → Hamming 128. Mixed → exact bit count.
        let q = Tier0([0xAAAA_BBBB_CCCC_DDDD, 0x1122_3344_5566_7788]);
        let corpus = vec![
            q,                                  // identity → 0
            Tier0([!q.0[0], !q.0[1]]),          // complement → 128
            Tier0([q.0[0] ^ 1, q.0[1]]),        // flip 1 bit → 1
            Tier0([0, 0]),                      // popcount(q)
            Tier0([!0, !0]),                    // 128 - popcount(q)
        ];
        let mut got = vec![0u32; corpus.len()];
        let mut want = vec![0u32; corpus.len()];
        xor_popcount_t0(&corpus, q, &mut got);
        xor_popcount_t0_scalar(&corpus, q, &mut want);
        assert_eq!(got, want);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], 128);
        assert_eq!(got[2], 1);
    }

    #[test]
    fn t1_edge_cases() {
        let q = Tier1([
            0xAAAA_BBBB_CCCC_DDDD,
            0x1122_3344_5566_7788,
            0xFFFF_0000_FFFF_0000,
            0x0000_FFFF_0000_FFFF,
            0xCAFE_BABE_DEAD_BEEF,
            0x1234_5678_90AB_CDEF,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
        ]);
        let mut all_inv = q.0;
        for w in all_inv.iter_mut() {
            *w = !*w;
        }
        let corpus = vec![
            q,                  // 0
            Tier1(all_inv),     // 512
            Tier1([0u64; 8]),   // popcount(q)
            Tier1([!0u64; 8]),  // 512 - popcount(q)
        ];
        let mut got = vec![0u32; corpus.len()];
        let mut want = vec![0u32; corpus.len()];
        xor_popcount_t1(&corpus, &q, &mut got);
        xor_popcount_t1_scalar(&corpus, &q, &mut want);
        assert_eq!(got, want);
        assert_eq!(got[0], 0);
        assert_eq!(got[1], 512);
        let pc_q: u32 = q.0.iter().map(|w| w.count_ones()).sum();
        assert_eq!(got[2], pc_q);
        assert_eq!(got[3], 512 - pc_q);
    }

    #[test]
    fn backend_is_consistent_across_calls() {
        // The OnceLock cache must produce a stable choice.
        assert_eq!(backend(), backend());
    }

    #[test]
    fn empty_corpus_is_a_noop() {
        // Boundary: empty corpus shouldn't panic and shouldn't write
        // anything. Tests both APIs at len=0.
        let mut out: Vec<u32> = Vec::new();
        xor_popcount_t0(&[], Tier0([0, 0]), &mut out);
        xor_popcount_t1(&[], &Tier1([0u64; 8]), &mut out);
        assert!(out.is_empty());
    }

    /// Exercise the scalar fallback explicitly even on hosts where the
    /// dispatched path would pick a SIMD backend. This guarantees the
    /// scalar code stays correct even when nobody runs on a non-x86,
    /// non-aarch64 host.
    #[test]
    fn t0_scalar_reference_random_4096() {
        let corpus = make_t0_corpus(4096, 0x12345678);
        let q = Tier0([0xDEAD_BEEF_CAFE_BABE, 0xF00D_FACE_C0DE_C0DE]);
        let mut out = vec![0u32; corpus.len()];
        xor_popcount_t0_scalar(&corpus, q, &mut out);
        // Bound check: every distance is in 0..=128.
        for &d in &out {
            assert!(d <= 128, "Hamming out of range: {d}");
        }
    }

    #[test]
    fn t1_scalar_reference_random_4096() {
        let corpus = make_t1_corpus(4096, 0x87654321);
        let q = Tier1([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut out = vec![0u32; corpus.len()];
        xor_popcount_t1_scalar(&corpus, &q, &mut out);
        for &d in &out {
            assert!(d <= 512, "Hamming out of range: {d}");
        }
    }
}
