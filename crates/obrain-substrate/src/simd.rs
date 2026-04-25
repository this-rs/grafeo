//! SIMD primitives for the hot u16 column decay loops.
//!
//! Both [`apply_energy_decay_to_zone`](crate::writer::apply_energy_decay_to_zone)
//! and [`apply_synapse_decay_to_zone`](crate::writer::apply_synapse_decay_to_zone)
//! reduce to the same arithmetic kernel applied slot-by-slot:
//!
//! ```text
//!   out = (in × factor_q16) >> 16        (Q0.16 × Q1.15 / Q0.16)
//! ```
//!
//! This module lowers the 8-lane batch of that kernel to the widest SIMD
//! instruction set detected at compile time:
//!
//! | target_arch | intrinsic path                                        |
//! |-------------|-------------------------------------------------------|
//! | `x86_64`    | `_mm_mulhi_epu16` (SSE2, baseline on every x86_64)    |
//! | `aarch64`   | `vmull_u16` + `vshrn_n_u32` (NEON, baseline on AArch64)|
//! | *other*     | scalar fallback (`(a as u32 * b) >> 16`)              |
//!
//! The kernel is intentionally **side-effect free** on a fixed-size `[u16; 8]`
//! array: the caller gathers 8 strided u16 column values into a stack buffer,
//! calls [`decay_u16x8`], then scatters the results back, applying any
//! tombstone / flag mask at scatter time. That pattern lets the SIMD code
//! ignore the 32 B record stride entirely — it only sees contiguous u16 lanes.
//!
//! ## Bit-for-bit parity
//!
//! The `x86_64` SSE2 path (`_mm_mulhi_epu16`) and the `aarch64` NEON path
//! (`vmull_u16` + `vshrn_n_u32`) both compute the high 16 bits of an
//! unsigned 16 × 16 → 32 multiply. That is exactly
//! `((a as u32) * (b as u32)) >> 16` — no saturation, no rounding. The
//! result therefore matches the scalar reference byte-for-byte for every
//! input. A property test in this module's `tests` pins that invariant.
//!
//! ## Safety
//!
//! The two SIMD paths each wrap a single `unsafe` block around an
//! unaligned load + vectorized mul + unaligned store on a `[u16; 8]` whose
//! alignment is at least 2 B (u16). SSE2 and NEON both tolerate unaligned
//! 128-bit loads via their `_mm_loadu_si128` / `vld1q_u16` variants, and
//! the `[u16; 8]` buffer is guaranteed contiguous by the Rust ABI. No
//! lifetime or aliasing invariant is violated.

// Narrowly scoped opt-out — the crate root enforces `#![deny(unsafe_code)]`.
// SIMD intrinsics on stable Rust require `unsafe`, so we allow it in this
// module only. The `Safety` doc-comment above pins the load/store invariants.
#![allow(unsafe_code)]

/// Multiply each of the 8 input lanes by `factor_q16`, taking the high 16
/// bits of the unsigned 16 × 16 → 32 product. This is the `>>16` division
/// baked into the Q0.16 multiplier semantics used by Substrate's energy /
/// synapse decay.
///
/// The output slot for lane `i` is `((input[i] as u32) * factor_q16 as u32) >> 16`.
///
/// Dispatches at compile time to:
/// - `_mm_mulhi_epu16` (SSE2 — always available on x86_64)
/// - `vmull_u16` + `vshrn_n_u32` (NEON — always available on aarch64)
/// - scalar fallback elsewhere
#[inline(always)]
pub fn decay_u16x8(values: &mut [u16; 8], factor_q16: u16) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline ABI. The input is a
        // stack-allocated `[u16; 8]` (16 B) — a valid `__m128i` load target.
        unsafe { decay_u16x8_sse2(values, factor_q16) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on AArch64. The input is a contiguous
        // `[u16; 8]`, valid for an unaligned `vld1q_u16`.
        unsafe { decay_u16x8_neon(values, factor_q16) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        decay_u16x8_scalar(values, factor_q16);
    }
}

/// Reference scalar implementation — the SIMD paths must match this
/// byte-for-byte. Exposed for property tests and as the ultimate fallback.
#[inline(always)]
pub fn decay_u16x8_scalar(values: &mut [u16; 8], factor_q16: u16) {
    let f = factor_q16 as u32;
    for v in values.iter_mut() {
        *v = (((*v as u32) * f) >> 16) as u16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn decay_u16x8_sse2(values: &mut [u16; 8], factor_q16: u16) {
    use core::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_mulhi_epu16, _mm_set1_epi16, _mm_storeu_si128,
    };

    let v = unsafe { _mm_loadu_si128(values.as_ptr() as *const __m128i) };
    let f = _mm_set1_epi16(factor_q16 as i16);
    // _mm_mulhi_epu16: each u16 lane gets the high 16 bits of the u32
    // product — exactly `(a × b) >> 16`, no saturation.
    let out = unsafe { _mm_mulhi_epu16(v, f) };
    unsafe { _mm_storeu_si128(values.as_mut_ptr() as *mut __m128i, out) };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn decay_u16x8_neon(values: &mut [u16; 8], factor_q16: u16) {
    use core::arch::aarch64::{
        vcombine_u16, vdup_n_u16, vget_high_u16, vget_low_u16, vld1q_u16, vmull_u16, vshrn_n_u32,
        vst1q_u16,
    };

    // Load 8 u16s and broadcast the factor.
    let v = unsafe { vld1q_u16(values.as_ptr()) };
    let f = vdup_n_u16(factor_q16); // u16x4 splat, reused twice
    // Two widening multiplies: u16x4 × u16x4 → u32x4 each.
    let lo = vmull_u16(vget_low_u16(v), f);
    let hi = vmull_u16(vget_high_u16(v), f);
    // Narrow the high 16 bits of each u32 lane back to u16.
    let lo_high = vshrn_n_u32::<16>(lo);
    let hi_high = vshrn_n_u32::<16>(hi);
    let out = vcombine_u16(lo_high, hi_high);
    unsafe { vst1q_u16(values.as_mut_ptr(), out) };
}

// ---------------------------------------------------------------------------
// Tests — parity between SIMD and scalar, on randomized + edge-case inputs.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference function used by every parity test.
    fn scalar_expected(values: [u16; 8], factor: u16) -> [u16; 8] {
        let mut out = values;
        decay_u16x8_scalar(&mut out, factor);
        out
    }

    #[test]
    fn zero_factor_zeros_every_lane() {
        let input = [
            0x1234, 0xFFFF, 0x8000, 0x0001, 0xABCD, 0x4242, 0x0F0F, 0xDEAD,
        ];
        let mut got = input;
        decay_u16x8(&mut got, 0);
        assert_eq!(got, [0u16; 8]);
    }

    #[test]
    fn identity_factor_preserves_high_15_bits() {
        // factor=0xFFFF approximates ×1.0 (0.999985); result == input.wrapping_sub(1)
        // for non-zero lanes, exactly 0 for lane=0.
        let input = [0u16, 1, 2, 100, 1000, 10_000, 32_768, 65_535];
        let mut got = input;
        decay_u16x8(&mut got, 0xFFFF);
        let expected = scalar_expected(input, 0xFFFF);
        assert_eq!(got, expected);
    }

    #[test]
    fn half_factor_halves_every_lane() {
        let input = [0, 2, 100, 1000, 32_768, 40_000, 50_000, 65_535];
        let mut got = input;
        decay_u16x8(&mut got, 32_768); // ×0.5
        let expected = scalar_expected(input, 32_768);
        assert_eq!(got, expected);
    }

    #[test]
    fn saturation_edge_cases_match_scalar() {
        // Boundary inputs: 0, 1, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF, and two random.
        let boundary = [0u16, 1, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF, 0x1234, 0xDEAD];
        for factor in [0u16, 1, 0x7FFF, 0x8000, 0xCCCC, 0xFFFF] {
            let mut got = boundary;
            decay_u16x8(&mut got, factor);
            let expected = scalar_expected(boundary, factor);
            assert_eq!(
                got, expected,
                "SIMD ≠ scalar at factor=0x{factor:04X}, input={boundary:?}"
            );
        }
    }

    #[test]
    fn randomized_parity_4096_vectors() {
        // Deterministic xorshift64 — no proptest dep bloat for a pure parity test.
        let mut state: u64 = 0xA5A5_5A5A_DEAD_BEEF;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..4096 {
            let input: [u16; 8] = [
                next() as u16,
                (next() >> 16) as u16,
                (next() >> 32) as u16,
                (next() >> 48) as u16,
                next() as u16,
                (next() >> 16) as u16,
                (next() >> 32) as u16,
                (next() >> 48) as u16,
            ];
            let factor = (next() & 0xFFFF) as u16;
            let mut got = input;
            decay_u16x8(&mut got, factor);
            let expected = scalar_expected(input, factor);
            assert_eq!(
                got, expected,
                "parity fail at factor=0x{factor:04X}, input={input:?}"
            );
        }
    }
}
