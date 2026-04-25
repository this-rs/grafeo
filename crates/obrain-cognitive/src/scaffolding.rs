//! # Adaptive Scaffolding
//!
//! Computes a scaffolding level (0-4) from the cognitive kernel config.
//! Level 0 = full guidance (novice), Level 4 = minimal guidance (expert).
//! The level is derived from context_budget_tokens: higher budget means more context,
//! which implies less expertise, hence a lower scaffolding level.

use crate::kernel_params::CognitiveKernelConfig;

/// Scaffolding level computed from kernel parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScaffoldingLevel(pub u8);

impl ScaffoldingLevel {
    /// Computes scaffolding level from kernel config.
    ///
    /// The formula is:
    /// ```text
    /// level = clamp(0, 4, round((1 - budget_normalized) * 4))
    /// ```
    /// where `budget_normalized = (budget - min) / (max - min)` with min=100 and max=4000.
    ///
    /// A high budget (4000) yields level 0 (novice, full guidance).
    /// A low budget (100) yields level 4 (expert, minimal guidance).
    pub fn from_config(config: &CognitiveKernelConfig) -> Self {
        let min = 100.0_f64;
        let max = 4000.0_f64;
        let budget = config.context_budget_tokens as f64;
        let normalized = ((budget - min) / (max - min)).clamp(0.0, 1.0);
        let level = ((1.0 - normalized) * 4.0).round() as u8;
        Self(level.min(4))
    }

    /// Maximum number of notes to include at this scaffolding level.
    pub fn max_notes(&self) -> usize {
        [8, 6, 4, 3, 2][self.0 as usize]
    }

    /// Maximum number of decisions to include at this scaffolding level.
    pub fn max_decisions(&self) -> usize {
        [5, 4, 3, 2, 1][self.0 as usize]
    }

    /// Maximum number of communities to include at this scaffolding level.
    pub fn max_communities(&self) -> usize {
        [6, 4, 3, 2, 1][self.0 as usize]
    }

    /// Whether to include examples at this level (levels 0 and 1 only).
    pub fn include_examples(&self) -> bool {
        self.0 <= 1
    }

    /// Whether to include step-by-step guidance (level 0 only).
    pub fn include_step_by_step(&self) -> bool {
        self.0 == 0
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_budget(budget: u32) -> CognitiveKernelConfig {
        CognitiveKernelConfig {
            context_budget_tokens: budget,
            ..CognitiveKernelConfig::default()
        }
    }

    #[test]
    fn default_budget_750_gives_level_3() {
        let config = config_with_budget(750);
        let level = ScaffoldingLevel::from_config(&config);
        assert_eq!(level.0, 3);
    }

    #[test]
    fn max_budget_4000_gives_level_0_novice() {
        let config = config_with_budget(4000);
        let level = ScaffoldingLevel::from_config(&config);
        assert_eq!(level.0, 0);
    }

    #[test]
    fn min_budget_100_gives_level_4_expert() {
        let config = config_with_budget(100);
        let level = ScaffoldingLevel::from_config(&config);
        assert_eq!(level.0, 4);
    }

    #[test]
    fn max_notes_decreases_with_level() {
        let notes: Vec<usize> = (0..=4).map(|l| ScaffoldingLevel(l).max_notes()).collect();
        for i in 0..4 {
            assert!(
                notes[i] >= notes[i + 1],
                "max_notes should decrease: level {} ({}) >= level {} ({})",
                i,
                notes[i],
                i + 1,
                notes[i + 1]
            );
        }
    }

    #[test]
    fn max_decisions_decreases_with_level() {
        let decisions: Vec<usize> = (0..=4)
            .map(|l| ScaffoldingLevel(l).max_decisions())
            .collect();
        for i in 0..4 {
            assert!(decisions[i] >= decisions[i + 1]);
        }
    }

    #[test]
    fn max_communities_decreases_with_level() {
        let communities: Vec<usize> = (0..=4)
            .map(|l| ScaffoldingLevel(l).max_communities())
            .collect();
        for i in 0..4 {
            assert!(communities[i] >= communities[i + 1]);
        }
    }

    #[test]
    fn examples_only_at_low_levels() {
        assert!(ScaffoldingLevel(0).include_examples());
        assert!(ScaffoldingLevel(1).include_examples());
        assert!(!ScaffoldingLevel(2).include_examples());
        assert!(!ScaffoldingLevel(3).include_examples());
        assert!(!ScaffoldingLevel(4).include_examples());
    }

    #[test]
    fn step_by_step_only_at_level_0() {
        assert!(ScaffoldingLevel(0).include_step_by_step());
        assert!(!ScaffoldingLevel(1).include_step_by_step());
        assert!(!ScaffoldingLevel(4).include_step_by_step());
    }

    #[test]
    fn budget_below_min_clamps_to_level_4() {
        let config = config_with_budget(50);
        let level = ScaffoldingLevel::from_config(&config);
        assert_eq!(level.0, 4);
    }

    #[test]
    fn budget_above_max_clamps_to_level_0() {
        let config = config_with_budget(10000);
        let level = ScaffoldingLevel::from_config(&config);
        assert_eq!(level.0, 0);
    }
}
