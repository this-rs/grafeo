//! FSRS (Free Spaced Repetition Scheduler) implementation for engram decay.
//!
//! Implements the FSRS-5 algorithm to model memory retention and optimal review
//! scheduling. Each engram carries an [`FsrsState`] that tracks its stability,
//! difficulty, repetition count, and lapse count.
//!
//! # References
//! - FSRS-5: <https://github.com/open-spaced-repetition/fsrs-rs>
//! - The 19-parameter model captures initial stability, difficulty dynamics,
//!   and the power-law forgetting curve.

use super::types::FsrsState;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FSRS Configuration — the 19 model parameters
// ---------------------------------------------------------------------------

/// The 19 FSRS-5 model parameters.
///
/// These weights control every aspect of the scheduling algorithm: initial
/// stability for each rating, difficulty initialization and updates, stability
/// after successful recall, and stability after a lapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsrsConfig {
    /// The 19 FSRS-5 weights.
    ///
    /// - `w[0..4]`:  Initial stability for Again/Hard/Good/Easy on first review
    /// - `w[4..6]`:  Difficulty initialization parameters
    /// - `w[6]`:     Difficulty update rate
    /// - `w[7]`:     Stability increase exponent (grade factor)
    /// - `w[8..11]`: Stability after recall (success) parameters
    /// - `w[11..14]`: Stability after lapse (failure) parameters
    /// - `w[14..17]`: Short-term stability parameters
    /// - `w[17..19]`: Additional calibration parameters
    pub w: [f64; 19],
}

impl Default for FsrsConfig {
    /// Returns the standard FSRS-5 default weights.
    fn default() -> Self {
        Self {
            w: [
                0.4072, // w[0]  — initial stability for Again
                1.1829, // w[1]  — initial stability for Hard
                3.1262, // w[2]  — initial stability for Good
                15.4722, // w[3]  — initial stability for Easy
                7.2102, // w[4]  — difficulty init: base
                0.5316, // w[5]  — difficulty init: rating factor
                1.0651, // w[6]  — difficulty update rate
                0.0046, // w[7]  — grade factor for stability increase
                1.5400, // w[8]  — stability recall: exponent base
                0.1192, // w[9]  — stability recall: S^(-w9)
                1.0100, // w[10] — stability recall: retention factor
                1.9395, // w[11] — stability lapse: base
                0.1100, // w[12] — stability lapse: D factor
                0.2900, // w[13] — stability lapse: S factor
                2.2700, // w[14] — short-term stability param
                0.0460, // w[15] — short-term stability param
                0.2000, // w[16] — short-term stability param
                1.1000, // w[17] — calibration param
                0.0600, // w[18] — calibration param
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// ReviewRating
// ---------------------------------------------------------------------------

/// The four possible review ratings in FSRS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ReviewRating {
    /// Complete failure to recall — the memory has lapsed.
    Again = 1,
    /// Recalled with significant difficulty.
    Hard = 2,
    /// Recalled with moderate effort (the standard "pass").
    Good = 3,
    /// Recalled effortlessly.
    Easy = 4,
}

impl ReviewRating {
    /// Returns the numeric value of this rating (1-4).
    fn as_f64(self) -> f64 {
        self as u8 as f64
    }
}

// ---------------------------------------------------------------------------
// FsrsScheduler
// ---------------------------------------------------------------------------

/// Scheduler that applies FSRS-5 updates to engram memory states.
///
/// The scheduler is stateless — all state lives in [`FsrsState`] on each engram.
/// This makes it cheap to clone and share across threads.
#[derive(Debug, Clone)]
pub struct FsrsScheduler {
    config: FsrsConfig,
}

impl FsrsScheduler {
    /// Creates a new scheduler with the given configuration.
    pub fn new(config: FsrsConfig) -> Self {
        Self { config }
    }

    /// Creates a new scheduler with default FSRS-5 parameters.
    pub fn with_defaults() -> Self {
        Self::new(FsrsConfig::default())
    }

    /// Performs a review and returns the updated FSRS state.
    ///
    /// # Parameters
    /// - `state`: The current FSRS state of the engram.
    /// - `rating`: How well the engram was recalled.
    /// - `elapsed_days`: Days since the last review (or since creation).
    ///
    /// # Returns
    /// A new `FsrsState` with updated stability, difficulty, reps, and lapses.
    pub fn review(&self, state: &FsrsState, rating: ReviewRating, elapsed_days: f64) -> FsrsState {
        let w = &self.config.w;

        // --- First review: use initial stability from w[0..4] ---
        if state.reps == 0 {
            let initial_stability = match rating {
                ReviewRating::Again => w[0],
                ReviewRating::Hard => w[1],
                ReviewRating::Good => w[2],
                ReviewRating::Easy => w[3],
            };

            // Initial difficulty: D0 = w[4] - w[5] * (rating - 3)
            let initial_difficulty = (w[4] - w[5] * (rating.as_f64() - 3.0)).clamp(0.0, 1.0);

            return FsrsState {
                stability: initial_stability,
                difficulty: initial_difficulty,
                reps: 1,
                lapses: if rating == ReviewRating::Again {
                    state.lapses + 1
                } else {
                    state.lapses
                },
                last_review: state.last_review,
                next_review: state.next_review,
            };
        }

        let s = state.stability;
        let d = state.difficulty;
        let r = self.retention_probability(state, elapsed_days);

        // --- Update difficulty ---
        // D' = clamp(D + w[6] * (rating_factor - 3), 0, 1)
        let new_difficulty = (d + w[6] * (rating.as_f64() - 3.0)).clamp(0.0, 1.0);

        // --- Update stability ---
        let new_stability = if rating == ReviewRating::Again {
            // Lapse: stability after forgetting
            // S_lapse = w[11] * D^(-w[12]) * ((S + 1)^w[13] - 1)
            let s_lapse = w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0);
            s_lapse.max(0.01) // Floor to prevent zero stability
        } else {
            // Successful recall: stability increase
            // S' = S * (e^w[8] * (11 - D) * S^(-w[9]) * (e^(w[10]*(1-R)) - 1) * grade_factor)
            let grade_factor = match rating {
                ReviewRating::Hard => w[14],
                ReviewRating::Good => 1.0,
                ReviewRating::Easy => w[15],
                ReviewRating::Again => unreachable!(),
            };

            let recall_factor = w[8].exp()
                * (11.0 - d)
                * s.powf(-w[9])
                * ((w[10] * (1.0 - r)).exp() - 1.0)
                * grade_factor;

            // New stability is at least as large as the old one for successful recall
            (s * (1.0 + recall_factor)).max(s)
        };

        // --- Update reps and lapses ---
        let (new_reps, new_lapses) = if rating == ReviewRating::Again {
            (0, state.lapses + 1)
        } else {
            (state.reps + 1, state.lapses)
        };

        FsrsState {
            stability: new_stability,
            difficulty: new_difficulty,
            reps: new_reps,
            lapses: new_lapses,
            last_review: state.last_review,
            next_review: state.next_review,
        }
    }

    /// Calculates the probability of successful recall after `elapsed_days`.
    ///
    /// Uses the FSRS power-law forgetting curve:
    ///   R = (1 + elapsed_days / (9 * S))^(-1)
    ///
    /// where S is the stability in days.
    pub fn retention_probability(&self, state: &FsrsState, elapsed_days: f64) -> f64 {
        if state.stability <= 0.0 {
            return 0.0;
        }
        let factor = 1.0 + elapsed_days / (9.0 * state.stability);
        factor.powf(-1.0)
    }

    /// Calculates the number of days until retention drops to `desired_retention`.
    ///
    /// This is the inverse of the retention formula:
    ///   days = 9 * S * (R^(-1) - 1)
    ///
    /// # Panics
    /// Returns `f64::INFINITY` if `desired_retention` is 0 or negative.
    pub fn next_review_days(&self, state: &FsrsState, desired_retention: f64) -> f64 {
        if desired_retention <= 0.0 {
            return f64::INFINITY;
        }
        if desired_retention >= 1.0 {
            return 0.0;
        }
        9.0 * state.stability * (desired_retention.recip() - 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scheduler() -> FsrsScheduler {
        FsrsScheduler::with_defaults()
    }

    #[test]
    fn initial_review_sets_stability() {
        let sched = default_scheduler();
        let state = FsrsState::default();

        let after_good = sched.review(&state, ReviewRating::Good, 0.0);
        assert!(after_good.stability > 0.0);
        assert_eq!(after_good.reps, 1);
        assert_eq!(after_good.lapses, 0);

        let after_again = sched.review(&state, ReviewRating::Again, 0.0);
        assert!(after_again.stability > 0.0);
        assert_eq!(after_again.lapses, 1);
    }

    #[test]
    fn retention_decays_over_time() {
        let sched = default_scheduler();
        let state = FsrsState {
            stability: 5.0,
            difficulty: 0.3,
            reps: 3,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let r0 = sched.retention_probability(&state, 0.0);
        let r5 = sched.retention_probability(&state, 5.0);
        let r30 = sched.retention_probability(&state, 30.0);

        assert!((r0 - 1.0).abs() < f64::EPSILON);
        assert!(r5 < r0);
        assert!(r30 < r5);
    }

    #[test]
    fn next_review_days_inverts_retention() {
        let sched = default_scheduler();
        let state = FsrsState {
            stability: 10.0,
            difficulty: 0.3,
            reps: 5,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let days = sched.next_review_days(&state, 0.9);
        let retention = sched.retention_probability(&state, days);
        assert!((retention - 0.9).abs() < 1e-6);
    }

    #[test]
    fn good_review_increases_stability() {
        let sched = default_scheduler();
        let mut state = FsrsState {
            stability: 3.0,
            difficulty: 0.3,
            reps: 2,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let old_stability = state.stability;
        state = sched.review(&state, ReviewRating::Good, 3.0);
        assert!(state.stability >= old_stability);
    }

    #[test]
    fn again_resets_reps_and_increments_lapses() {
        let sched = default_scheduler();
        let state = FsrsState {
            stability: 10.0,
            difficulty: 0.3,
            reps: 5,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let after = sched.review(&state, ReviewRating::Again, 15.0);
        assert_eq!(after.reps, 0);
        assert_eq!(after.lapses, 1);
        assert!(after.stability < state.stability);
    }

    #[test]
    fn difficulty_clamps_to_unit_range() {
        let sched = default_scheduler();
        let hard_state = FsrsState {
            stability: 5.0,
            difficulty: 0.95,
            reps: 3,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let after = sched.review(&hard_state, ReviewRating::Again, 5.0);
        assert!(after.difficulty >= 0.0);
        assert!(after.difficulty <= 1.0);
    }

    #[test]
    fn edge_case_zero_stability() {
        let sched = default_scheduler();
        let state = FsrsState {
            stability: 0.0,
            difficulty: 0.3,
            reps: 1,
            lapses: 0,
            last_review: None,
            next_review: None,
        };

        let r = sched.retention_probability(&state, 1.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn next_review_edge_cases() {
        let sched = default_scheduler();
        let state = FsrsState::default();

        assert_eq!(sched.next_review_days(&state, 0.0), f64::INFINITY);
        assert_eq!(sched.next_review_days(&state, 1.0), 0.0);
    }
}
