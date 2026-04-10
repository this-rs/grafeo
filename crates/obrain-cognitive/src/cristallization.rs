//! # Automatic Crystallization
//!
//! Detects stable community patterns and materializes them into skills.
//! Cycle: Observe -> Detect stability -> Propose -> Crystallize -> Monitor -> Dissolve

use crate::kernel_params::CognitiveKernelConfig;
use obrain_common::types::NodeId;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Community snapshot
// ---------------------------------------------------------------------------

/// A community snapshot for stability tracking.
#[derive(Debug, Clone)]
pub struct CommunitySnapshot {
    /// Unique identifier for the community.
    pub community_id: u64,
    /// Members of the community (node IDs).
    pub members: Vec<NodeId>,
    /// Cohesion score of the community (0.0 to 1.0).
    pub cohesion: f64,
    /// Timestamp when the snapshot was taken (epoch millis).
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Stability tracker
// ---------------------------------------------------------------------------

/// Tracks community stability across sessions.
pub struct StabilityTracker {
    /// Per-session community snapshots (most recent last).
    snapshots: Vec<Vec<CommunitySnapshot>>,
    /// Maximum number of sessions to retain.
    max_history: usize,
}

impl StabilityTracker {
    /// Creates a new stability tracker with the given history depth.
    pub fn new(max_history: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_history,
        }
    }

    /// Records a session's community snapshots.
    pub fn record_session(&mut self, communities: Vec<CommunitySnapshot>) {
        self.snapshots.push(communities);
        if self.snapshots.len() > self.max_history {
            self.snapshots
                .drain(..self.snapshots.len() - self.max_history);
        }
    }

    /// Compute Jaccard similarity between two member sets.
    pub fn jaccard(a: &[NodeId], b: &[NodeId]) -> f64 {
        let set_a: HashSet<NodeId> = a.iter().copied().collect();
        let set_b: HashSet<NodeId> = b.iter().copied().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            return 0.0;
        }
        intersection as f64 / union as f64
    }

    /// Find communities that are stable across `k` consecutive sessions.
    ///
    /// A community is considered stable if it appears in `k` consecutive
    /// sessions (including the most recent) with Jaccard similarity >= `min_jaccard`
    /// between consecutive sessions.
    pub fn find_stable(&self, k: u32, min_jaccard: f64) -> Vec<CristallizationCandidate> {
        let k = k as usize;
        if self.snapshots.len() < k || k == 0 {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let start = self.snapshots.len() - k;
        let recent_sessions = &self.snapshots[start..];

        // For each community in the most recent session, trace back through history
        if let Some(latest) = recent_sessions.last() {
            for community in latest {
                let mut stable_count = 1u32;
                let mut current_members = &community.members;
                let mut total_cohesion = community.cohesion;

                // Walk backwards through sessions
                for i in (0..recent_sessions.len() - 1).rev() {
                    let session = &recent_sessions[i];
                    // Find the best-matching community in this session
                    let best_match = session.iter().max_by(|a, b| {
                        let ja = Self::jaccard(current_members, &a.members);
                        let jb = Self::jaccard(current_members, &b.members);
                        ja.partial_cmp(&jb).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    if let Some(matched) = best_match {
                        let jaccard = Self::jaccard(current_members, &matched.members);
                        if jaccard >= min_jaccard {
                            stable_count += 1;
                            total_cohesion += matched.cohesion;
                            current_members = &matched.members;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                if stable_count >= k as u32 {
                    let avg_cohesion = total_cohesion / stable_count as f64;
                    candidates.push(CristallizationCandidate {
                        community_id: community.community_id,
                        members: community.members.clone(),
                        stability_count: stable_count,
                        cohesion: avg_cohesion,
                        average_energy: avg_cohesion, // proxy: use cohesion as energy estimate
                        suggested_type: if community.members.len() > 5 {
                            CristallizationType::FeatureGraph
                        } else {
                            CristallizationType::Skill
                        },
                    });
                }
            }
        }

        candidates
    }

    /// Returns the number of recorded sessions.
    pub fn session_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ---------------------------------------------------------------------------
// Crystallization candidate
// ---------------------------------------------------------------------------

/// Candidate for crystallization.
#[derive(Debug, Clone)]
pub struct CristallizationCandidate {
    /// Community ID this candidate originates from.
    pub community_id: u64,
    /// Members of the community.
    pub members: Vec<NodeId>,
    /// Number of consecutive sessions this community was stable.
    pub stability_count: u32,
    /// Average cohesion across sessions.
    pub cohesion: f64,
    /// Average energy of the community members.
    pub average_energy: f64,
    /// Suggested crystallization type.
    pub suggested_type: CristallizationType,
}

/// Type of crystallization to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CristallizationType {
    /// Crystallize into a reusable skill.
    Skill,
    /// Crystallize into a feature graph.
    FeatureGraph,
}

// ---------------------------------------------------------------------------
// Crystallization / dissolution decisions
// ---------------------------------------------------------------------------

/// Checks if a candidate should be crystallized based on kernel params.
pub fn should_crystallize(
    candidate: &CristallizationCandidate,
    config: &CognitiveKernelConfig,
) -> bool {
    candidate.stability_count >= config.cristallization_sessions
        && candidate.average_energy >= config.cristallization_energy
}

/// Checks if a crystallized skill should be dissolved.
///
/// A skill is dissolved when its hit rate drops below the configured threshold.
pub fn should_dissolve(hit_rate: f64, config: &CognitiveKernelConfig) -> bool {
    hit_rate < config.dissolution_hit_rate
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeId {
        NodeId::from(id)
    }

    #[test]
    fn jaccard_identical_sets() {
        let a = vec![node(1), node(2), node(3)];
        let b = vec![node(1), node(2), node(3)];
        assert!((StabilityTracker::jaccard(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_disjoint_sets() {
        let a = vec![node(1), node(2)];
        let b = vec![node(3), node(4)];
        assert!((StabilityTracker::jaccard(&a, &b)).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = vec![node(1), node(2), node(3)];
        let b = vec![node(2), node(3), node(4)];
        // intersection = {2, 3} = 2, union = {1, 2, 3, 4} = 4
        assert!((StabilityTracker::jaccard(&a, &b) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn jaccard_empty_sets() {
        let a: Vec<NodeId> = vec![];
        let b: Vec<NodeId> = vec![];
        assert!((StabilityTracker::jaccard(&a, &b)).abs() < f64::EPSILON);
    }

    #[test]
    fn stability_detection_across_sessions() {
        let mut tracker = StabilityTracker::new(10);

        let members = vec![node(1), node(2), node(3)];

        // Record 5 sessions with the same community
        for i in 0..5 {
            tracker.record_session(vec![CommunitySnapshot {
                community_id: 42,
                members: members.clone(),
                cohesion: 0.9,
                timestamp: i * 1000,
            }]);
        }

        let stable = tracker.find_stable(5, 0.8);
        assert_eq!(stable.len(), 1);
        assert_eq!(stable[0].community_id, 42);
        assert_eq!(stable[0].stability_count, 5);
    }

    #[test]
    fn stability_not_enough_sessions() {
        let mut tracker = StabilityTracker::new(10);

        tracker.record_session(vec![CommunitySnapshot {
            community_id: 1,
            members: vec![node(1), node(2)],
            cohesion: 0.9,
            timestamp: 0,
        }]);

        let stable = tracker.find_stable(3, 0.8);
        assert!(stable.is_empty());
    }

    #[test]
    fn stability_broken_by_membership_change() {
        let mut tracker = StabilityTracker::new(10);

        // Session 1-2: community {1, 2, 3}
        for i in 0..2 {
            tracker.record_session(vec![CommunitySnapshot {
                community_id: 1,
                members: vec![node(1), node(2), node(3)],
                cohesion: 0.9,
                timestamp: i * 1000,
            }]);
        }

        // Session 3: completely different community
        tracker.record_session(vec![CommunitySnapshot {
            community_id: 1,
            members: vec![node(10), node(20), node(30)],
            cohesion: 0.9,
            timestamp: 2000,
        }]);

        let stable = tracker.find_stable(3, 0.8);
        assert!(stable.is_empty());
    }

    #[test]
    fn should_crystallize_threshold() {
        let config = CognitiveKernelConfig::default();
        // Default: cristallization_sessions = 5, cristallization_energy = 0.8

        let candidate = CristallizationCandidate {
            community_id: 1,
            members: vec![node(1), node(2)],
            stability_count: 5,
            cohesion: 0.9,
            average_energy: 0.85,
            suggested_type: CristallizationType::Skill,
        };
        assert!(should_crystallize(&candidate, &config));

        // Not enough sessions
        let candidate_few = CristallizationCandidate {
            stability_count: 3,
            ..candidate.clone()
        };
        assert!(!should_crystallize(&candidate_few, &config));

        // Low energy
        let candidate_low_energy = CristallizationCandidate {
            average_energy: 0.5,
            ..candidate
        };
        assert!(!should_crystallize(&candidate_low_energy, &config));
    }

    #[test]
    fn should_dissolve_threshold() {
        let config = CognitiveKernelConfig::default();
        // Default: dissolution_hit_rate = 0.05

        assert!(should_dissolve(0.01, &config)); // below threshold
        assert!(should_dissolve(0.04, &config)); // below threshold
        assert!(!should_dissolve(0.05, &config)); // at threshold
        assert!(!should_dissolve(0.1, &config)); // above threshold
    }

    #[test]
    fn history_trimmed_to_max() {
        let mut tracker = StabilityTracker::new(3);
        for i in 0..10 {
            tracker.record_session(vec![CommunitySnapshot {
                community_id: 1,
                members: vec![node(1)],
                cohesion: 0.9,
                timestamp: i * 1000,
            }]);
        }
        assert_eq!(tracker.session_count(), 3);
    }
}
